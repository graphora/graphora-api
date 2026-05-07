from dataclasses import dataclass

from prefect import flow, task
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import aiofiles
import asyncio
import uuid
from datetime import timezone
from fastapi import UploadFile
from graphora_server.utils.logger import logger
from graphora_server.schemas.transform import (
    DocumentMetadata,
    ValidationResult,
    StorageLocation,
)
from graphora_server.services.transform.validators import FileValidator
from graphora_server.services.transform.storage import DocumentStorage
from graphora_server.config import settings
from graphora_server.services.quality.tasks import quality_validation_task
from graphora_server.services.quality.exceptions import (
    QualityValidationError,
    QualityThresholdNotMetError,
    QualityViolationError,
)
from graphora_server.services.chunking.tasks import chunk_document
from graphora_server.services.chunking.config import ChunkingStrategy
from graphora_server.services.chunking.models import ChunkingResult, ChunkMetadata
from graphora_server.services.transform.tasks import (
    construct_knowledge_graph,
    ExtractionError,
)
from graphora_server.services.storage.tasks import store_knowledge_graph
from graphora_server.services.transform.progress_tracker import ProgressTracker
from graphora_server.services.quality.models import QualityResults

from graphora_server.services.transform.status_models import (
    TransformationStage,
    ErrorSummary,
    TransformFailureReason,
)
from graphora_server.services.usage_tracking import usage_tracking_service
from graphora_server.schemas.usage import DocumentUsageRequest, ProcessingStatus
from pypdf import PdfReader, PdfWriter

progress_tracker = ProgressTracker()


async def _should_pre_extract_pdfs(user_id: str) -> bool:
    """Return True when the active LLM provider needs pre-extracted text.

    Gemini ingests PDF bytes natively (multimodal). Ollama is text-only,
    so for Ollama users we extract PDF → text via DocumentParser before
    routing through ``construct_knowledge_graph(chunks=...)``.

    Resolution order matches ``get_llm_client_for_user``:
        1. ``LLM_PROVIDER=ollama`` env var → True (no DB lookup)
        2. User's stored provider == "ollama" → True
        3. Anything else → False (the existing Gemini-binary fast path)

    Errors fetching the user's stored config fail closed to False so a
    transient DB issue doesn't silently switch every user to the
    text-only path.
    """
    from graphora_server.config import get_settings

    settings_obj = get_settings()
    if (settings_obj.LLM_PROVIDER or "").lower() == "ollama":
        return True

    try:
        from graphora_server.services.ai_config_service import AIConfigService

        result = await AIConfigService().get_user_provider_secret(user_id)
        if not result:
            return False
        provider_name, _api_key, _model = result
        return provider_name == "ollama"
    except Exception as exc:
        logger.warning(
            "PDF-routing provider lookup failed for user %s: %s — defaulting "
            "to Gemini binary path",
            user_id,
            exc,
        )
        return False


async def _pdf_to_text_file(pdf_path: str, output_dir: Path) -> Optional[str]:
    """Extract a PDF to a sibling .txt file using DocumentParser.

    Returns the new path on success, None when no text could be
    extracted (e.g., scanned PDFs without OCR fallback). Used by the
    Ollama branch — the resulting text file goes through the regular
    chunking pipeline as if it were a .txt input.
    """
    from graphora_server.services.document_parser import DocumentParser

    parser = DocumentParser()
    text = await parser.parse_file(pdf_path)
    if not text or not text.strip():
        logger.warning(
            "DocumentParser returned no text for %s; cannot route through "
            "the Ollama text path",
            pdf_path,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (Path(pdf_path).stem + ".txt")
    async with aiofiles.open(out_path, "w", encoding="utf-8") as fh:
        await fh.write(text)
    return str(out_path)


@task(name="usage-track-document", retries=2, retry_delay_seconds=10)
async def track_document_usage_task(
    user_id: str, request: DocumentUsageRequest, processing_started_at: datetime
):
    """Record the start of document processing for usage tracking."""

    return await usage_tracking_service.track_document_processing(
        user_id=user_id, request=request, processing_started_at=processing_started_at
    )


@task(name="usage-update-document", retries=2, retry_delay_seconds=10)
async def update_document_usage_task(
    document_usage_id: str,
    completed_at: datetime,
    processing_status: ProcessingStatus,
    chunks_created: int = 0,
    nodes_extracted: int = 0,
    relationships_extracted: int = 0,
    success_rate: Optional[float] = None,
    error_message: Optional[str] = None,
):
    """Finalize document usage tracking metrics."""

    return await usage_tracking_service.update_document_processing(
        document_usage_id=document_usage_id,
        processing_completed_at=completed_at,
        processing_status=processing_status,
        chunks_created=chunks_created,
        nodes_extracted=nodes_extracted,
        relationships_extracted=relationships_extracted,
        success_rate=success_rate,
        error_message=error_message,
    )


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
    if isinstance(exc, QualityValidationError):
        return exc.retry_allowed

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
    retries=0,
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
                        from pypdf import PdfReader

                        reader = PdfReader(file_path)
                        page_count = len(reader.pages)
                    except Exception:
                        page_count = 1  # Fallback

                usage_request = DocumentUsageRequest(
                    transform_id=transform_id,
                    session_id=ontology_id,
                    document_name=path.name,
                    document_type=path.suffix.lstrip(".").upper() or "UNKNOWN",
                    document_size_bytes=file_size,
                    page_count=page_count,
                )

                usage_record = await track_document_usage_task(
                    user_id=user_id,
                    request=usage_request,
                    processing_started_at=processing_start_time,
                )
                document_usage_records.append(usage_record)
                logger.info(f"Started tracking usage for document: {path.name}")

            except Exception as e:
                logger.error(f"Failed to track usage for {file_path}: {str(e)}")
                # Continue processing even if tracking fails

        processed_paths: List[str] = []
        doc_chunk_results: List[Tuple[ChunkingResult, List[ChunkMetadata]]] = []
        pdf_files: List[str] = []
        # A1-prov: per-split ChunkMetadata for the PDF-binary path,
        # built alongside pdf_files so build_graph_from_pdfs can stamp
        # source_file / source_chunk_id / page_number on emitted nodes.
        pdf_metadatas: List[ChunkMetadata] = []
        graphs = []

        # Start PARSE stage
        await progress_tracker.start_stage(transform_id, TransformationStage.PARSE)
        total_documents = len(file_paths)

        if total_documents:
            for index, (file_path, doc_metadata) in enumerate(
                zip(file_paths, metadata), start=1
            ):
                try:
                    validation_result = await validate_document(file_path)

                    if not validation_result.is_valid:
                        logger.error(
                            "Validation failed for %s: %s",
                            file_path,
                            validation_result.errors,
                        )
                        continue

                    storage_location = await store_document(
                        file_path, transform_id, doc_metadata
                    )
                    stored_path = storage_location.original_path
                    processed_paths.append(stored_path)

                except Exception as parse_error:
                    logger.error(
                        "Failed to prepare document %s: %s",
                        file_path,
                        parse_error,
                        extra={"transform_id": transform_id},
                    )
                finally:
                    await update_stage_progress(
                        transform_id,
                        TransformationStage.PARSE,
                        index,
                        total_documents,
                    )

        await progress_tracker.complete_stage(transform_id, TransformationStage.PARSE)

        # Start CHUNK stage
        await progress_tracker.start_stage(transform_id, TransformationStage.CHUNK)
        pdf_folder = Path(settings.UPLOAD_DIR) / transform_id / "pdf"
        pdf_folder.mkdir(parents=True, exist_ok=True)

        # Provider gate: Ollama is text-only, so PDFs need to be
        # pre-extracted to text instead of going through the Gemini
        # binary path. Resolved once per flow — checking per-PDF would
        # cost a DB lookup we don't need.
        pdf_needs_text_extraction = await _should_pre_extract_pdfs(user_id)
        pdf_text_folder = Path(settings.UPLOAD_DIR) / transform_id / "pdf-text"

        # A1-prov: when we extract a PDF to a text sidecar (Ollama
        # path), the chunker sees `report.txt` and would record that
        # as document_name. Track the mapping so we can rewrite the
        # ChunkMetadata.source_file back to the original PDF after
        # chunking. The Evidence tab should cite the file the user
        # uploaded, not the intermediate sidecar.
        text_sidecar_to_original: Dict[str, str] = {}

        # Collect paths to chunk
        doc_paths_to_chunk: List[Tuple[str, Optional[ChunkingStrategy]]] = []
        for processed_path in processed_paths:
            suffix = Path(processed_path).suffix.lower()
            if suffix == ".pdf":
                if pdf_needs_text_extraction:
                    text_path = await _pdf_to_text_file(processed_path, pdf_text_folder)
                    if text_path:
                        doc_paths_to_chunk.append(
                            (text_path, ChunkingStrategy.STRUCTURAL)
                        )
                        text_sidecar_to_original[text_path] = Path(processed_path).name
                    else:
                        logger.warning(
                            "Skipping %s — PDF text extraction produced no "
                            "content and the active provider cannot ingest "
                            "PDFs natively",
                            processed_path,
                        )
                else:
                    pdf_splits = split_pdf(
                        input_pdf=processed_path, location=pdf_folder, pages=100
                    )
                    # A1-prov: build per-split ChunkMetadata so the
                    # binary-PDF extraction path stamps source_file
                    # (the original PDF filename) and source_chunk_id
                    # (the split filename) on every emitted node and
                    # edge.
                    #
                    # source_text capture is gated on the layout-aware
                    # PDF backend (pymupdf4llm) being available. The
                    # raw-text backends (pymupdf/pypdf/pdfplumber)
                    # produce garbled output on real-world PDFs with
                    # multi-column layouts, tables, and footnotes
                    # (10K filings, research papers) — the Evidence
                    # tab ended up surfacing jumbled letters with
                    # words reordered across columns. Commit beafa92
                    # disabled source_text entirely as a fix; this
                    # path re-enables it ONLY when pymupdf4llm is
                    # installed (the [pdf-llm] extra). Operators on
                    # the raw-text backends keep the post-beafa92
                    # behaviour (source_text=None) — better an empty
                    # Evidence tab than a misleading one.
                    #
                    # page_number is intentionally NOT set here.
                    # split_pdf's filename trailing integer is the
                    # last page in the chunk, not the page a fact
                    # came from — citing it as page_number would be
                    # wrong provenance for any fact from earlier
                    # pages. Per-page page_number requires the LLM
                    # emitting it during extraction, deferred to
                    # Gate 4.
                    from graphora_server.services.document_parser import (
                        DocumentParser,
                    )

                    layout_aware = DocumentParser.has_layout_aware_backend()
                    parser = DocumentParser() if layout_aware else None
                    original_name = Path(processed_path).name
                    for split_path in pdf_splits:
                        split_name = Path(split_path).name
                        split_text: Optional[str] = None
                        if parser is not None:
                            try:
                                split_text = await parser.parse_file(split_path)
                            except Exception as exc:  # pragma: no cover
                                logger.warning(
                                    "Layout-aware parse failed for %s: %s; "
                                    "leaving source_text=None",
                                    split_path,
                                    exc,
                                )
                                split_text = None
                        pdf_metadatas.append(
                            ChunkMetadata(
                                transform_id=transform_id,
                                chunk_id=split_name,
                                source_file=original_name,
                                source_text=split_text,
                            )
                        )
                    pdf_files.extend(pdf_splits)
            else:
                strategy_override = (
                    ChunkingStrategy.STRUCTURAL
                    if suffix in {".md", ".markdown", ".txt"}
                    else None
                )
                doc_paths_to_chunk.append((processed_path, strategy_override))

        chunk_failures: List[Tuple[str, str]] = []
        total_chunk_jobs = len(doc_paths_to_chunk)
        if total_chunk_jobs:
            for index, (source_path, strategy_override) in enumerate(
                doc_paths_to_chunk, start=1
            ):
                try:
                    chunk_result, chunk_metadata = await chunk_document(
                        file_path=source_path,
                        transform_id=transform_id,
                        config=chunking_config,
                        strategy_override=strategy_override,
                    )
                    if chunk_result and chunk_metadata:
                        # A1-prov: when this chunked path was a text
                        # sidecar generated from a PDF (Ollama route),
                        # rewrite source_file to the original PDF
                        # filename so Evidence cites the user's upload,
                        # not the intermediate `.txt`.
                        original_pdf = text_sidecar_to_original.get(source_path)
                        if original_pdf:
                            for cm in chunk_metadata:
                                cm.source_file = original_pdf
                            if chunk_result.chunk_metadata:
                                for cm in chunk_result.chunk_metadata:
                                    cm.source_file = original_pdf
                        doc_chunk_results.append((chunk_result, chunk_metadata))
                    else:
                        chunk_failures.append((source_path, "No chunks produced"))
                except Exception as exc:
                    chunk_failures.append((source_path, str(exc)))
                finally:
                    await update_stage_progress(
                        transform_id,
                        TransformationStage.CHUNK,
                        index,
                        total_chunk_jobs,
                    )

            if chunk_failures:
                logger.warning(
                    "Chunking completed with %s failure(s)",
                    len(chunk_failures),
                    extra={
                        "transform_id": transform_id,
                        "failures": chunk_failures,
                    },
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

        # Process chunked documents
        for chunk_result, chunk_metadata in doc_chunk_results:
            if chunk_result and getattr(chunk_result, "chunks", None):
                try:
                    graph_result, metrics = await construct_knowledge_graph(
                        chunks=chunk_result.chunks,
                        ontology_path=ontology_path,
                        transform_id=transform_id,
                        progress_callback=lambda i, t: asyncio.create_task(
                            update_stage_progress(
                                transform_id, TransformationStage.TRANSFORM, i, t
                            )
                        ),
                        user_id=user_id,
                        # A1-prov: forward per-chunk metadata so the
                        # extraction pipeline can stamp source-span
                        # properties (document_name, page_number,
                        # source_text, chunk_offset) on every node/edge.
                        chunk_metadatas=chunk_metadata,
                    )
                    if graph_result:
                        graphs.append(graph_result)
                        if metrics:
                            total_nodes += metrics.total_nodes
                            total_relationships += metrics.total_relationships
                except Exception as extraction_error:
                    logger.error(
                        "Knowledge graph construction failed: %s",
                        extraction_error,
                        extra={"transform_id": transform_id},
                    )

        # Process PDF files
        if pdf_files:
            try:
                pdf_graph_result, metrics = await construct_knowledge_graph(
                    pdf_paths=[Path(p) for p in pdf_files],
                    ontology_path=ontology_path,
                    transform_id=transform_id,
                    progress_callback=lambda i, t: asyncio.create_task(
                        update_stage_progress(
                            transform_id, TransformationStage.TRANSFORM, i, t
                        )
                    ),
                    user_id=user_id,
                    chunk_metadatas=pdf_metadatas,
                )
                if pdf_graph_result:
                    graphs.append(pdf_graph_result)
                    if metrics:
                        total_nodes += metrics.total_nodes
                        total_relationships += metrics.total_relationships
            except Exception as extraction_error:
                logger.error(
                    "PDF knowledge graph construction failed: %s",
                    extraction_error,
                    extra={"transform_id": transform_id},
                )

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
                    from graphora_server.services.ontology_storage_service import (
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
                            raise QualityThresholdNotMetError(
                                "Quality validation failed: score below minimum threshold",
                                score=float(quality_results.overall_score),
                                threshold=float(settings.QUALITY_FAIL_SCORE),
                                violations=[
                                    violation.model_dump()
                                    for violation in quality_results.violations
                                ],
                                quality_results=quality_results.model_dump(),
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
                            raise QualityViolationError(
                                "Quality validation failed: unresolved violations",
                                violations=[
                                    violation.model_dump()
                                    for violation in quality_results.violations
                                ],
                                quality_results=quality_results.model_dump(),
                            )

                        gating_status = getattr(
                            quality_results, "quality_gate_status", "pass"
                        )
                        gating_reasons = getattr(
                            quality_results, "quality_gate_reasons", []
                        )

                        if gating_status == "fail":
                            gating_config = (
                                quality_results.validation_config.get("gating", {})
                                if quality_results.validation_config
                                else {}
                            )
                            threshold = gating_config.get(
                                "hard_fail_score", settings.QUALITY_MIN_SCORE
                            )
                            await _persist_quality_violations(
                                transform_id,
                                user_id,
                                quality_results,
                            )
                            raise QualityThresholdNotMetError(
                                "Quality validation failed hard gate",
                                score=quality_results.overall_score,
                                threshold=threshold,
                                violations=[
                                    violation.model_dump()
                                    for violation in quality_results.violations
                                ],
                                quality_results=quality_results.model_dump(),
                            )

                        if quality_results.requires_review:
                            logger.info(
                                "Quality results flagged for review; evaluating gating configuration"
                            )

                        if gating_status == "warn":
                            logger.warning(
                                "Quality validation produced warnings: %s",
                                gating_reasons,
                            )
                            await _persist_quality_violations(
                                transform_id,
                                user_id,
                                quality_results,
                            )
                        else:
                            logger.info(
                                "Transform %s passed quality gate",
                                transform_id,
                                extra={
                                    "transform_id": transform_id,
                                    "quality_score": quality_results.overall_score,
                                },
                            )

                    else:
                        logger.warning(
                            f"Skipping quality validation for transform {transform_id}: no ontology available"
                        )
                else:
                    logger.error(
                        f"Skipping quality validation for transform {transform_id}: no valid graphs to validate"
                    )
                    raise QualityValidationError(
                        "Quality validation failed: no graphs generated",
                        details={
                            "reason": "no_graphs_generated",
                            "documents_processed": len(document_usage_records),
                        },
                    )
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
            raise QualityValidationError(
                "Quality validation failed: no graphs generated",
                details={
                    "reason": "no_graphs_generated",
                    "documents_processed": len(document_usage_records),
                },
            )

        # Store knowledge graph
        nodes_stored = 0
        relationships_stored = 0
        storage_time = 0
        storage_retries = 0

        # Filter out None graphs
        graphs_to_store = [g for g in graphs if g is not None]
        if not graphs_to_store:
            raise ValueError("No graphs were successfully processed and stored")

        for index, graph in enumerate(graphs_to_store, start=1):
            storage_result = await store_knowledge_graph(graph, transform_id, user_id)
            nodes_stored += storage_result.nodes_stored
            relationships_stored += storage_result.relationships_stored
            storage_time += storage_result.metrics.storage_time_ms
            storage_retries += storage_result.metrics.retries

            await update_stage_progress(
                transform_id,
                TransformationStage.LOAD,
                nodes_stored + relationships_stored,
                max(total_nodes + total_relationships, 1),
            )

        if nodes_stored == 0 and relationships_stored == 0:
            raise ValueError("No graphs were successfully processed and stored")

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

                await update_document_usage_task(
                    document_usage_id=usage_record.id,
                    completed_at=datetime.now(timezone.utc),
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

        error_message_str = str(e)

        classification = _classify_transform_failure(
            e,
            current_stage,
            documents_processed=len(document_usage_records),
        )

        failure_code = classification.code
        failure_details = classification.details
        is_recoverable = classification.is_recoverable
        recovery_instructions = classification.recovery_instructions

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
            recovery_instructions=recovery_instructions,
            failure_code=failure_code,
            details=failure_details,
            failure_reason=classification.reason,
        )

        await progress_tracker.fail_stage(transform_id, current_stage, error)

        # Update usage tracking for failed processing
        for usage_record in document_usage_records:
            try:
                await update_document_usage_task(
                    document_usage_id=usage_record.id,
                    completed_at=datetime.now(timezone.utc),
                    processing_status=ProcessingStatus.FAILED,
                    success_rate=0.0,
                    error_message=str(e),
                )
            except Exception as update_error:
                logger.error(
                    f"Failed to update failed usage tracking for {usage_record.id}: {str(update_error)}"
                )

        raise


@dataclass
class FailureClassification:
    reason: TransformFailureReason
    code: str
    details: Dict[str, Any]
    is_recoverable: bool
    recovery_instructions: Optional[str]


def _extract_original_exception(exc: Exception) -> Optional[Exception]:
    original = getattr(exc, "original", None)
    return original or None


def _is_llm_unavailable_error(exc: Optional[Exception]) -> bool:
    if exc is None:
        return False

    text = str(exc).lower()
    transient_markers = [
        "model is overloaded",
        "temporarily unavailable",
        "unavailable",
        "service unavailable",
        "try again later",
        "rate limit",
        "overloaded",
    ]

    if any(marker in text for marker in transient_markers):
        return True

    for attr in ("status", "status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value == 503:
            return True
        if isinstance(value, str) and value.strip().lower() in {
            "503",
            "service_unavailable",
            "unavailable",
        }:
            return True

    return False


def _classify_transform_failure(
    exc: Exception,
    stage: TransformationStage,
    *,
    documents_processed: Optional[int] = None,
) -> FailureClassification:
    """Derive a normalized failure classification for frontend consumption."""

    details: Dict[str, Any] = {}
    reason = TransformFailureReason.UNKNOWN_ERROR
    is_recoverable = False
    recovery_instructions: Optional[str] = None
    code: Optional[str] = None

    if isinstance(exc, QualityValidationError):
        details = dict(exc.details or {})
        reason_key = details.get("reason")
        if reason_key == "no_graphs_generated":
            reason = TransformFailureReason.NO_GRAPH_GENERATED
            recovery_instructions = "Ensure the source documents contain extractable entities or relationships."
        else:
            reason = TransformFailureReason.QUALITY_GATE_FAILED
            recovery_instructions = (
                "Review the quality violations and adjust the ontology or source data."
            )
        code = exc.code
        is_recoverable = exc.retry_allowed

    elif isinstance(exc, ExtractionError):
        underlying = _extract_original_exception(exc)
        if _is_llm_unavailable_error(underlying):
            reason = TransformFailureReason.LLM_UNAVAILABLE
            code = "llm_unavailable"
            is_recoverable = True
            recovery_instructions = "Retry shortly; the upstream language model reported temporary unavailability."
        else:
            reason = TransformFailureReason.TRANSFORM_EXECUTION_FAILED
            code = "extraction_failed"
            is_recoverable = True
            recovery_instructions = (
                "Retry the transform; if the issue persists, contact support."
            )
        if underlying is not None:
            details["underlying_exception"] = type(underlying).__name__
            details["underlying_message"] = str(underlying)

    elif (
        isinstance(exc, ValueError)
        and "no graphs were successfully processed" in str(exc).lower()
    ):
        reason = TransformFailureReason.NO_GRAPH_GENERATED
        code = "no_graph_generated"
        recovery_instructions = "Validate that the extraction produced entities and relationships before storage."

    if code is None:
        if stage == TransformationStage.PARSE:
            reason = TransformFailureReason.PARSE_FAILED
            code = "parse_failed"
            recovery_instructions = (
                "Verify the document format is supported and not corrupt."
            )
        elif stage == TransformationStage.CHUNK:
            reason = TransformFailureReason.CHUNKING_FAILED
            code = "chunking_failed"
            recovery_instructions = (
                "Review chunking configuration or document structure."
            )
        elif stage == TransformationStage.LOAD:
            reason = TransformFailureReason.STORAGE_FAILED
            code = "storage_failed"
            recovery_instructions = "Check graph storage connectivity and credentials."
        elif stage == TransformationStage.TRANSFORM:
            if reason == TransformFailureReason.UNKNOWN_ERROR:
                reason = TransformFailureReason.TRANSFORM_EXECUTION_FAILED
            code = code or "transform_failed"
        else:
            code = code or reason.value

    # Inspect for non-recoverable configuration patterns unless already classified
    original = _extract_original_exception(exc) or exc
    error_text = str(original).lower()
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

    if any(pattern in error_text for pattern in non_recoverable_patterns):
        is_recoverable = False
        if reason == TransformFailureReason.UNKNOWN_ERROR:
            reason = TransformFailureReason.TRANSFORM_EXECUTION_FAILED
        recovery_instructions = (
            "Verify API credentials, quotas, and billing status before retrying."
        )
        code = code or "configuration_error"

    details.setdefault("exception_type", type(exc).__name__)
    details.setdefault("message", str(exc))
    details.setdefault("stage", stage.value)
    if documents_processed is not None:
        details.setdefault("documents_processed", documents_processed)

    return FailureClassification(
        reason=reason,
        code=code or reason.value,
        details=details,
        is_recoverable=is_recoverable,
        recovery_instructions=recovery_instructions,
    )


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
            "rule_id": violation.rule_id,
            "severity": violation.severity,
            "entity": violation.entity_type,
            "entity_id": violation.entity_id,
            "message": violation.message,
            "suggestion": violation.suggestion,
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
