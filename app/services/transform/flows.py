from prefect import flow, task
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import aiofiles
import asyncio
import uuid
from datetime import timezone
from fastapi import UploadFile
from app.utils.logger import logger
from app.schemas.transform import DocumentMetadata, ValidationResult, StorageLocation
from app.services.transform.validators import FileValidator
from app.services.transform.storage import DocumentStorage
from app.config import settings
from app.services.quality.tasks import quality_validation_task
from app.services.marker.tasks import convert_pdf_to_markdown
from app.services.chunking.tasks import chunk_documents
from app.services.transform.tasks import construct_knowledge_graph
from app.services.storage.tasks import store_knowledge_graph
from app.services.transform.progress_tracker import ProgressTracker
from app.services.quality.models import QualityResults

from app.services.transform.status_models import TransformationStage, ErrorSummary
from app.services.usage_tracking import usage_tracking_service
from app.schemas.usage import DocumentUsageRequest, ProcessingStatus
from PyPDF2 import PdfReader, PdfWriter

progress_tracker = ProgressTracker()


@task(
    name="document-validation",
    retries=settings.CHUNKING_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
)
async def validate_document(file_path: str) -> ValidationResult:
    """Validate document before processing"""
    validator = FileValidator()

    # Create UploadFile from file path
    path = Path(file_path)
    async with aiofiles.open(path, "rb") as f:
        content = await f.read()

    upload_file = UploadFile(filename=path.name, file=None)
    upload_file._file = content

    return await validator.validate(upload_file)


@task(
    name="document-storage",
    retries=settings.STORAGE_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
)
async def store_document(
    file_path: str, transform_id: str, metadata: DocumentMetadata
) -> StorageLocation:
    """Store document in persistent storage"""
    storage = DocumentStorage(base_path=settings.UPLOAD_DIR)
    return await storage.save_document(file_path, transform_id, metadata)


async def update_stage_progress(
    transform_id: str,
    stage: TransformationStage,
    items_processed: int,
    items_total: int,
):
    """Update progress for transformation stage"""
    try:
        await progress_tracker.update_stage_progress(
            transform_id, stage, items_processed, items_total
        )
    except Exception as e:
        logger.error(f"Failed to update progress: {str(e)}")


def should_retry_flow_error(exc: Exception) -> bool:
    """Determine if flow error should be retried"""
    error_msg = str(exc).lower()

    # Don't retry authentication/configuration errors
    non_retryable_patterns = [
        "api key not valid",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "invalid_argument",
        "permission denied",
        "quota exceeded",
        "billing",
        "api_key_invalid",
        "bamlclienthttperror",
        "quality validation failed",
    ]

    for pattern in non_retryable_patterns:
        if pattern in error_msg:
            return False

    return True


@flow(
    name="document-transformation",
    description="Transform document to knowledge graph",
    version="1.0.0",
    retries=2,
    retry_delay_seconds=30,
)
async def document_transformation_flow(
    transform_id: str,
    ontology_id: str,
    file_paths: List[str],
    metadata: List[DocumentMetadata],
    user_id: str,
    chunking_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Main transformation flow

    Args:
        transform_id: Unique ID for this transformation
        ontology_id: ID of the ontology to use
        file_paths: List of paths to documents
        metadata: List of document metadata
        user_id: User's ID for database configuration
    """
    logger.info(
        f"Starting transformation flow with ID: {transform_id} for user: {user_id}"
    )

    # Track document processing usage
    document_usage_records = []
    processing_start_time = datetime.now(timezone.utc)

    try:

        logger.info(f"Starting transformation flow {transform_id}")

        # Create usage tracking records for each document
        for file_path, doc_metadata in zip(file_paths, metadata):
            try:
                path = Path(file_path)
                file_size = path.stat().st_size if path.exists() else 0

                # Estimate page count for non-PDF files
                page_count = 1  # Default
                if path.suffix.lower() == ".pdf":
                    try:
                        from PyPDF2 import PdfReader

                        reader = PdfReader(file_path)
                        page_count = len(reader.pages)
                    except Exception:
                        page_count = 1  # Fallback

                usage_request = DocumentUsageRequest(
                    transform_id=transform_id,
                    document_name=path.name,
                    document_type=path.suffix.lstrip(".").upper() or "UNKNOWN",
                    document_size_bytes=file_size,
                    page_count=page_count,
                )

                usage_record = await usage_tracking_service.track_document_processing(
                    user_id=user_id,
                    request=usage_request,
                    processing_started_at=processing_start_time,
                )
                document_usage_records.append(usage_record)
                logger.info(f"Started tracking usage for document: {path.name}")

            except Exception as e:
                logger.error(f"Failed to track usage for {file_path}: {str(e)}")
                # Continue processing even if tracking fails

        processed_paths = []
        doc_chunk_results = []
        pdf_files = []
        graphs = []

        # Start PARSE stage
        await progress_tracker.start_stage(transform_id, TransformationStage.PARSE)
        processed_paths = await parse_docs(transform_id, file_paths, metadata)
        await progress_tracker.complete_stage(transform_id, TransformationStage.PARSE)

        # Start CHUNK stage
        await progress_tracker.start_stage(transform_id, TransformationStage.CHUNK)
        pdf_folder = Path(settings.UPLOAD_DIR) / transform_id / "pdf"
        pdf_folder.mkdir(parents=True, exist_ok=True)
        text_paths: List[str] = []
        for processed_path in processed_paths:
            if Path(processed_path).suffix.lower() == ".pdf":
                pdf_splits = split_pdf(
                    input_pdf=processed_path, location=pdf_folder, pages=100
                )
                pdf_files.extend(pdf_splits)
            else:
                text_paths.append(processed_path)

        if text_paths:
            text_chunk_results = await chunk_documents(
                transform_id=transform_id,
                processed_paths=text_paths,
                chunking_config=chunking_config,
            )
            doc_chunk_results.extend(text_chunk_results)
            await update_stage_progress(
                transform_id,
                TransformationStage.CHUNK,
                len(text_chunk_results),
                len(text_paths),
            )
        else:
            logger.info("No textual documents to chunk for transform %s", transform_id)
        await progress_tracker.complete_stage(transform_id, TransformationStage.CHUNK)

        # Start TRANSFORM stage
        await progress_tracker.start_stage(transform_id, TransformationStage.TRANSFORM)

        # Initialize metrics
        total_nodes = 0
        total_relationships = 0
        ontology_path = Path(settings.ONTOLOGY_DIR).expanduser() / f"{ontology_id}.yaml"

        # Process each document
        for res in doc_chunk_results:
            result, graph = res
            if result and result.chunks:
                graph, metrics = await construct_knowledge_graph(
                    chunks=result.chunks,
                    ontology_path=ontology_path,
                    transform_id=transform_id,
                    progress_callback=lambda i, t: asyncio.create_task(
                        update_stage_progress(
                            transform_id, TransformationStage.TRANSFORM, i, t
                        )
                    ),
                    user_id=user_id,
                )
                if graph and metrics:
                    total_nodes += metrics.total_nodes
                    total_relationships += metrics.total_relationships
                graphs.append(graph)

        pdf_graph, metrics = await construct_knowledge_graph(
            pdf_paths=pdf_files,
            ontology_path=ontology_path,
            transform_id=transform_id,
            progress_callback=lambda i, t: asyncio.create_task(
                update_stage_progress(transform_id, TransformationStage.TRANSFORM, i, t)
            ),
            user_id=user_id,
        )
        if pdf_graph:
            graphs.append(pdf_graph)

        await progress_tracker.complete_stage(
            transform_id, TransformationStage.TRANSFORM
        )

        logger.debug("Transformation graphs snapshot: %s", graphs)

        # Start LOAD stage
        await progress_tracker.start_stage(transform_id, TransformationStage.LOAD)

        # Optional: Quality validation step (if available and enabled)
        quality_results = None
        logger.info(f"Quality validation check: {len(graphs)} graphs generated")
        if len(graphs) > 0:
            try:
                # For now, validate the first non-None graph
                # TODO: Handle multiple graphs or combine them
                graph_to_validate = next((g for g in graphs if g is not None), None)
                if graph_to_validate:
                    logger.info(
                        f"Starting quality validation for transform {transform_id} "
                        f"with graph containing {len(graph_to_validate.nodes)} nodes, "
                        f"{len(graph_to_validate.relationships)} relationships"
                    )

                    # Load ontology for quality rules (simplified for now)
                    from app.services.ontology_storage_service import (
                        OntologyStorageService,
                    )
                    import yaml

                    ontology_service = OntologyStorageService()
                    ontology_record = await ontology_service.get_ontology(
                        user_id, ontology_id
                    )

                    if ontology_record and ontology_record.get("yaml_content"):
                        try:
                            ontology_with_rules = yaml.safe_load(
                                ontology_record["yaml_content"]
                            )
                            logger.info(
                                "Successfully loaded ontology with %s entity types",
                                len(ontology_with_rules.get("entities", {})),
                            )
                        except yaml.YAMLError as yaml_error:
                            logger.error(
                                "Failed to parse ontology YAML for transform %s: %s",
                                transform_id,
                                yaml_error,
                            )
                            raise ValueError(
                                "Quality validation failed: ontology content invalid"
                            )
                    else:
                        logger.error(
                            "No ontology found or empty content for transform %s (ontology_id=%s, user=%s)",
                            transform_id,
                            ontology_id,
                            user_id,
                        )
                        raise ValueError(
                            "Quality validation failed: ontology not available"
                        )

                    if ontology_with_rules:
                        # Run quality validation
                        quality_results = await quality_validation_task(
                            knowledge_graph=graph_to_validate,
                            ontology_with_rules=ontology_with_rules,
                            transform_id=transform_id,
                            user_id=user_id,
                        )

                        logger.info(
                            "Quality validation completed",
                            extra={
                                "transform_id": transform_id,
                                "score": quality_results.overall_score,
                                "grade": quality_results.grade,
                                "violations": len(quality_results.violations),
                            },
                        )

                        if quality_results.overall_score < settings.QUALITY_FAIL_SCORE:
                            logger.error(
                                "Quality score %.1f below minimum %.1f",
                                quality_results.overall_score,
                                settings.QUALITY_FAIL_SCORE,
                            )
                            raise ValueError(
                                "Quality validation failed: score below minimum threshold"
                            )

                        if (
                            settings.QUALITY_FAIL_ON_VIOLATION
                            and quality_results.violations
                            and quality_results.requires_review
                        ):
                            logger.error(
                                "Quality validation found %s violations requiring review",
                                len(quality_results.violations),
                            )
                            raise ValueError(
                                "Quality validation failed: unresolved violations"
                            )

                        if quality_results.overall_score >= settings.QUALITY_MIN_SCORE:
                            logger.info(
                                f"Transform {transform_id} meets auto-approval threshold"
                            )
                        else:
                            logger.info(
                                f"Transform {transform_id} requires manual review; score %.1f below auto-approve %.1f",
                                quality_results.overall_score,
                                settings.QUALITY_MIN_SCORE,
                            )
                            await _persist_quality_violations(
                                transform_id,
                                user_id,
                                quality_results,
                            )
                            await _persist_quality_violations(
                                transform_id,
                                user_id,
                                quality_results,
                            )

                    else:
                        logger.warning(
                            f"Skipping quality validation for transform {transform_id}: no ontology available"
                        )
                else:
                    logger.error(
                        f"Skipping quality validation for transform {transform_id}: no valid graphs to validate"
                    )
                    raise ValueError("Quality validation failed: no graphs generated")
            except Exception as e:
                logger.error(
                    f"Quality validation failed for transform {transform_id}: {e}"
                )
                import traceback

                traceback.print_exc()
                raise
        else:
            logger.error(
                f"Skipping quality validation for transform {transform_id}: no graphs generated"
            )
            raise ValueError("Quality validation failed: no graphs generated")

        # Store knowledge graph
        nodes_stored = 0
        relationships_stored = 0
        storage_time = 0
        storage_retries = 0
        for graph in graphs:
            if graph is not None:  # Only store non-None graphs
                storage_result = await store_knowledge_graph(
                    graph, transform_id, user_id
                )
                nodes_stored = nodes_stored + storage_result.nodes_stored
                relationships_stored = (
                    relationships_stored + storage_result.relationships_stored
                )
                storage_time = storage_time + storage_result.metrics.storage_time_ms
                storage_retries = storage_retries + storage_result.metrics.retries
            else:
                logger.warning(f"Skipping None graph for transform {transform_id}")

        # Check if we have any stored graphs
        if nodes_stored == 0 and relationships_stored == 0:
            raise ValueError("No graphs were successfully processed and stored")

        # Update progress with storage metrics
        await update_stage_progress(
            transform_id,
            TransformationStage.LOAD,
            (nodes_stored + relationships_stored),
            (total_nodes + total_relationships),
        )

        await progress_tracker.complete_stage(transform_id, TransformationStage.LOAD)

        # Update usage tracking with final results
        for i, usage_record in enumerate(document_usage_records):
            try:
                # Calculate metrics for this document
                doc_chunks = len(doc_chunk_results) if i < len(doc_chunk_results) else 0
                doc_nodes = (
                    total_nodes // len(document_usage_records)
                    if document_usage_records
                    else 0
                )
                doc_relationships = (
                    total_relationships // len(document_usage_records)
                    if document_usage_records
                    else 0
                )

                await usage_tracking_service.update_document_processing(
                    document_usage_id=usage_record.id,
                    processing_completed_at=datetime.now(timezone.utc),
                    processing_status=ProcessingStatus.SUCCESS,
                    chunks_created=doc_chunks,
                    nodes_extracted=doc_nodes,
                    relationships_extracted=doc_relationships,
                    success_rate=1.0,
                )
                logger.info(
                    f"Updated usage tracking for document: {usage_record.document_name}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to update usage tracking for {usage_record.id}: {str(e)}"
                )

        # Return flow results
        return {
            "transform_id": transform_id,
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
            "documents_processed": len(document_usage_records),
        }

    except Exception as e:
        logger.error(f"Transform failed: {str(e)}")

        # Determine current stage based on error type and context
        error_message = str(e).lower()
        current_stage = TransformationStage.PARSE  # Default stage

        # Map error types to likely stages
        if "chunk" in error_message or "chunking" in error_message:
            current_stage = TransformationStage.CHUNK
        elif (
            "pdf" in error_message
            or "markdown" in error_message
            or "conversion" in error_message
        ):
            current_stage = TransformationStage.PARSE
        elif (
            "knowledge graph" in error_message
            or "extraction" in error_message
            or "baml" in error_message
            or "api key" in error_message
        ):
            current_stage = TransformationStage.TRANSFORM
        elif (
            "storage" in error_message
            or "store" in error_message
            or "database" in error_message
        ):
            current_stage = TransformationStage.LOAD

        # Determine if error is recoverable (should not retry)
        is_recoverable = True
        error_message_str = str(e)

        # Non-recoverable errors that should fail immediately
        non_recoverable_patterns = [
            "api key not valid",
            "invalid api key",
            "authentication failed",
            "unauthorized",
            "invalid_argument",
            "permission denied",
            "quota exceeded",
            "billing",
            "api_key_invalid",
        ]

        for pattern in non_recoverable_patterns:
            if pattern in error_message.lower():
                is_recoverable = False
                break

        # Record error
        error = ErrorSummary(
            stage=current_stage,
            error_type=type(e).__name__,
            error_message=error_message_str,
            error_timestamp=datetime.now(),
            stack_trace=None,  # TODO: Add stack trace
            affected_components=[],
            retry_count=0,  # We can't get retry count from context reliably
            is_recoverable=is_recoverable,
            recovery_instructions=(
                "Check API key configuration" if not is_recoverable else None
            ),
        )

        await progress_tracker.fail_stage(transform_id, current_stage, error)

        # Update usage tracking for failed processing
        for usage_record in document_usage_records:
            try:
                await usage_tracking_service.update_document_processing(
                    document_usage_id=usage_record.id,
                    processing_completed_at=datetime.now(timezone.utc),
                    processing_status=ProcessingStatus.FAILED,
                    success_rate=0.0,
                    error_message=str(e),
                )
            except Exception as update_error:
                logger.error(
                    f"Failed to update failed usage tracking for {usage_record.id}: {str(update_error)}"
                )

        raise


async def parse_docs(transform_id, file_paths, metadata) -> List[str]:
    processed_paths = []
    for file_path, doc_metadata in zip(file_paths, metadata):
        try:
            # Validate document
            validation_result = await validate_document(file_path)
            if not validation_result.is_valid:
                logger.error(
                    f"Validation failed for {file_path}: {validation_result.errors}"
                )
                continue

                # Store document
            storage_location = await store_document(
                file_path, transform_id, doc_metadata
            )
            logger.info(f"Document stored at {storage_location.original_path}")

            # Convert PDF to markdown if needed
            if settings.PDF_PROCESSOR == "marker":
                if Path(file_path).suffix.lower() == ".pdf":
                    conversion_result = await convert_pdf_to_markdown(
                        file_path=Path(file_path), transform_id=transform_id
                    )
                    if conversion_result:
                        processed_paths.append(conversion_result.markdown_path)
                        logger.info(
                            f"PDF converted to markdown: {conversion_result.markdown_path}"
                        )
                else:
                    # For non-PDF files, use the original path
                    processed_paths.append(file_path)
                    logger.info(f"Using original file: {file_path}")
            else:
                # For gemini files, use the original path
                processed_paths.append(file_path)
                logger.info(f"Using original file: {file_path}")

                # Update progress
            await update_stage_progress(
                transform_id,
                TransformationStage.PARSE,
                len(processed_paths),
                len(file_paths),
            )

        except Exception as e:
            logger.error(
                f"Processing failed for file {file_path}",
                extra={"transform_id": transform_id, "error": str(e)},
            )
            continue

    return processed_paths


async def _persist_quality_violations(
    transform_id: str,
    user_id: str,
    quality_results: QualityResults,
) -> None:
    """Persist quality violations for review. Currently logs details; extend to storage later."""

    if not quality_results or not quality_results.violations:
        return

    violations_summary = [
        {
            "rule": violation.rule_type,
            "severity": violation.severity,
            "entity": violation.entity_type,
            "message": violation.message,
        }
        for violation in quality_results.violations
    ]

    logger.warning(
        "Quality violations recorded for transform %s",
        transform_id,
        extra={
            "transform_id": transform_id,
            "user_id": user_id,
            "violations": violations_summary,
        },
    )


def split_pdf(input_pdf: str, location: Path, pages=100):
    reader = PdfReader(input_pdf)
    idx = 0
    writer = PdfWriter()
    output = []
    for i, page in enumerate(reader.pages):
        idx = idx + 1
        writer.add_page(page)
        if idx == pages or (i == len(reader.pages) - 1):
            uuid_key = str(uuid.uuid4())
            output_filename = location / f"page_{uuid_key}_{i+1}.pdf"
            with open(output_filename, "wb") as output_file:
                writer.write(output_file)
            writer = PdfWriter()
            idx = 0
            output.append(output_filename.as_posix())
    return output
