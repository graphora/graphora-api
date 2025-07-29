from prefect import flow, task
from prefect.context import get_run_context
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import aiofiles
import asyncio
import uuid
from datetime import timezone
from fastapi import UploadFile
from app.utils.logger import logger
from app.schemas.transform import (
    DocumentMetadata,
    ValidationResult,
    StorageLocation
)
from app.services.transform.validators import FileValidator
from app.services.transform.storage import DocumentStorage
from app.config import settings
from app.services.quality.tasks import quality_validation_task
from app.services.marker.tasks import convert_pdf_to_markdown
from app.services.chunking.tasks import chunk_document, check_chunk_quality
from app.services.transform.tasks import construct_knowledge_graph
from app.services.storage.tasks import store_knowledge_graph
from app.services.transform.progress_tracker import ProgressTracker

from app.services.transform.status_models import (
    TransformationStage,
    ErrorSummary
)
from app.services.chunking.models import (
    ChunkingResult,
    ChunkMetadata
)
from app.services.usage_tracking import usage_tracking_service
from app.schemas.usage import DocumentUsageRequest, ProcessingStatus
from PyPDF2 import PdfReader, PdfWriter

progress_tracker = ProgressTracker()

@task(
    name="document-validation",
    retries=settings.CHUNKING_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS
)
async def validate_document(
    file_path: str
) -> ValidationResult:
    """Validate document before processing"""
    validator = FileValidator()
    
    # Create UploadFile from file path
    path = Path(file_path)
    async with aiofiles.open(path, 'rb') as f:
        content = await f.read()
        
    upload_file = UploadFile(
        filename=path.name,
        file=None
    )
    upload_file._file = content
    
    return await validator.validate(upload_file)

@task(
    name="document-storage",
    retries=settings.STORAGE_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS
)
async def store_document(
    file_path: str,
    transform_id: str,
    metadata: DocumentMetadata
) -> StorageLocation:
    """Store document in persistent storage"""
    storage = DocumentStorage(base_path=settings.UPLOAD_DIR)
    return await storage.save_document(file_path, transform_id, metadata)

async def update_stage_progress(transform_id: str, stage: TransformationStage, 
                                items_processed: int, items_total: int):
    """Update progress for transformation stage"""
    try:
        await progress_tracker.update_stage_progress(
            transform_id,
            stage,
            items_processed,
            items_total
        )
    except Exception as e:
        logger.error(f"Failed to update progress: {str(e)}")

def should_retry_flow_error(exc: Exception) -> bool:
    """Determine if flow error should be retried"""
    error_msg = str(exc).lower()
    
    # Don't retry authentication/configuration errors
    non_retryable_patterns = [
        'api key not valid',
        'invalid api key',
        'authentication failed',
        'unauthorized',
        'invalid_argument',
        'permission denied',
        'quota exceeded',
        'billing',
        'api_key_invalid',
        'bamlclienthttperror'
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
    retry_delay_seconds=30
)
async def document_transformation_flow(
    transform_id: str,
    ontology_id: str,
    file_paths: List[str],
    metadata: List[DocumentMetadata],
    user_id: str,
    chunking_config: Optional[Any] = None
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
    logger.info(f"Starting transformation flow with ID: {transform_id} for user: {user_id}")
    
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
                if path.suffix.lower() == '.pdf':
                    try:
                        from PyPDF2 import PdfReader
                        reader = PdfReader(file_path)
                        page_count = len(reader.pages)
                    except:
                        page_count = 1  # Fallback
                
                usage_request = DocumentUsageRequest(
                    transform_id=transform_id,
                    document_name=path.name,
                    document_type=path.suffix.lstrip('.').upper() or 'UNKNOWN',
                    document_size_bytes=file_size,
                    page_count=page_count
                )
                
                usage_record = await usage_tracking_service.track_document_processing(
                    user_id=user_id,
                    request=usage_request,
                    processing_started_at=processing_start_time
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
        await progress_tracker.start_stage(
            transform_id,
            TransformationStage.PARSE
        )
        processed_paths = await parse_docs(transform_id, file_paths, metadata)
        await progress_tracker.complete_stage(
            transform_id,
            TransformationStage.PARSE
        )
        
        # Start CHUNK stage
        await progress_tracker.start_stage(
            transform_id,
            TransformationStage.CHUNK
        )
        pdf_folder = Path(settings.UPLOAD_DIR) / transform_id / 'pdf'
        pdf_folder.mkdir(parents=True, exist_ok=True)
        for processed_path in processed_paths:
            if Path(processed_path).suffix.lower() == '.pdf':
                pdf_splits = split_pdf(input_pdf=processed_path, location=pdf_folder, pages=100)
                pdf_files.extend(pdf_splits)
            else:
                doc_chunk_results = await chunk_documents(transform_id, processed_path, doc_chunk_results, chunking_config)
            await update_stage_progress(transform_id, TransformationStage.CHUNK, 
                                    len(doc_chunk_results), len(processed_paths))
        await progress_tracker.complete_stage(
            transform_id,
            TransformationStage.CHUNK
        )
        
        # Start TRANSFORM stage
        await progress_tracker.start_stage(
            transform_id,
            TransformationStage.TRANSFORM
        )
        
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
                        update_stage_progress(transform_id, TransformationStage.TRANSFORM, i, t)
                        ),
                    user_id=user_id
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
            user_id=user_id
        )
        if pdf_graph:
            graphs.append(pdf_graph)
        
        await progress_tracker.complete_stage(
            transform_id,
            TransformationStage.TRANSFORM
        )
        
        print(graphs)
        
        # Start LOAD stage
        await progress_tracker.start_stage(
            transform_id,
            TransformationStage.LOAD
        )
        
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
                    from app.services.ontology_storage_service import OntologyStorageService
                    import yaml
                    
                    ontology_service = OntologyStorageService()
                    ontology_record = await ontology_service.get_ontology(user_id, ontology_id)
                    
                    if ontology_record and ontology_record.get('yaml_content'):
                        # Parse the YAML content to get the actual ontology structure
                        try:
                            ontology_with_rules = yaml.safe_load(ontology_record['yaml_content'])
                            logger.info(f"Successfully loaded ontology with {len(ontology_with_rules.get('entities', {}))} entity types")
                        except yaml.YAMLError as e:
                            logger.error(f"Failed to parse ontology YAML: {e}")
                            ontology_with_rules = None
                    else:
                        logger.warning(f"No ontology found or no content for ontology_id {ontology_id}, user {user_id}")
                        ontology_with_rules = None
                    
                    if ontology_with_rules:
                        # Run quality validation
                        quality_results = await quality_validation_task(
                            knowledge_graph=graph_to_validate,
                            ontology_with_rules=ontology_with_rules,
                            transform_id=transform_id,
                            user_id=user_id
                        )
                        
                        logger.info(
                            f"Quality validation completed: Score={quality_results.overall_score:.1f}, "
                            f"Grade={quality_results.grade}, Violations={len(quality_results.violations)}"
                        )
                        
                        # Check for auto-approval
                        if quality_results.overall_score >= 90.0 and not quality_results.requires_review:
                            logger.info(f"Transform {transform_id} auto-approved for high quality")
                        else:
                            logger.info(f"Transform {transform_id} requires manual quality review")
                    
                    else:
                        logger.warning(
                            f"Skipping quality validation for transform {transform_id}: no ontology available"
                        )
                else:
                    logger.warning(
                        f"Skipping quality validation for transform {transform_id}: no valid graphs to validate"
                    )
            except Exception as e:
                logger.error(f"Quality validation failed for transform {transform_id}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with storage even if quality validation fails
                quality_results = None
        else:
            logger.warning(
                f"Skipping quality validation for transform {transform_id}: no graphs generated"
            )
        
        # Store knowledge graph
        nodes_stored = 0
        relationships_stored = 0
        storage_time = 0
        storage_retries = 0
        for graph in graphs:
            if graph is not None:  # Only store non-None graphs
                storage_result = await store_knowledge_graph(
                    graph,
                    transform_id,
                    user_id
                )
                nodes_stored = nodes_stored + storage_result.nodes_stored
                relationships_stored = relationships_stored + storage_result.relationships_stored
                storage_time = storage_time + storage_result.metrics.storage_time_ms
                storage_retries = storage_retries + storage_result.metrics.retries
            else:
                logger.warning(f"Skipping None graph for transform {transform_id}")
        
        # Check if we have any stored graphs
        if nodes_stored == 0 and relationships_stored == 0:
            raise ValueError("No graphs were successfully processed and stored")
        
        # Update progress with storage metrics
        await update_stage_progress(transform_id, TransformationStage.LOAD, 
                                    (nodes_stored + relationships_stored), 
                                    (total_nodes + total_relationships))
        
        await progress_tracker.complete_stage(
            transform_id,
            TransformationStage.LOAD
        )
        
        # Update usage tracking with final results
        for i, usage_record in enumerate(document_usage_records):
            try:
                # Calculate metrics for this document
                doc_chunks = len(doc_chunk_results) if i < len(doc_chunk_results) else 0
                doc_nodes = total_nodes // len(document_usage_records) if document_usage_records else 0
                doc_relationships = total_relationships // len(document_usage_records) if document_usage_records else 0
                
                await usage_tracking_service.update_document_processing(
                    document_usage_id=usage_record.id,
                    processing_completed_at=datetime.now(timezone.utc),
                    processing_status=ProcessingStatus.SUCCESS,
                    chunks_created=doc_chunks,
                    nodes_extracted=doc_nodes,
                    relationships_extracted=doc_relationships,
                    success_rate=1.0
                )
                logger.info(f"Updated usage tracking for document: {usage_record.document_name}")
                
            except Exception as e:
                logger.error(f"Failed to update usage tracking for {usage_record.id}: {str(e)}")
        
        # Return flow results
        return {
            'transform_id': transform_id,
            'total_nodes': total_nodes,
            'total_relationships': total_relationships,
            'documents_processed': len(document_usage_records)
        }
        
    except Exception as e:
        logger.error(f"Transform failed: {str(e)}")
        
        # Determine current stage based on error type and context
        error_message = str(e).lower()
        current_stage = TransformationStage.PARSE  # Default stage
        
        # Map error types to likely stages
        if 'chunk' in error_message or 'chunking' in error_message:
            current_stage = TransformationStage.CHUNK
        elif 'pdf' in error_message or 'markdown' in error_message or 'conversion' in error_message:
            current_stage = TransformationStage.PARSE
        elif 'knowledge graph' in error_message or 'extraction' in error_message or 'baml' in error_message or 'api key' in error_message:
            current_stage = TransformationStage.TRANSFORM
        elif 'storage' in error_message or 'store' in error_message or 'database' in error_message:
            current_stage = TransformationStage.LOAD
        
        # Determine if error is recoverable (should not retry)
        is_recoverable = True
        error_message_str = str(e)
        
        # Non-recoverable errors that should fail immediately
        non_recoverable_patterns = [
            'api key not valid',
            'invalid api key',
            'authentication failed',
            'unauthorized',
            'invalid_argument',
            'permission denied',
            'quota exceeded',
            'billing',
            'api_key_invalid'
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
            recovery_instructions="Check API key configuration" if not is_recoverable else None
        )
        
        await progress_tracker.fail_stage(
            transform_id,
            current_stage,
            error
        )
        
        # Update usage tracking for failed processing
        for usage_record in document_usage_records:
            try:
                await usage_tracking_service.update_document_processing(
                    document_usage_id=usage_record.id,
                    processing_completed_at=datetime.now(timezone.utc),
                    processing_status=ProcessingStatus.FAILED,
                    success_rate=0.0,
                    error_message=str(e)
                )
            except Exception as update_error:
                logger.error(f"Failed to update failed usage tracking for {usage_record.id}: {str(update_error)}")
        
        raise

async def parse_docs(transform_id, file_paths, metadata) -> List[str]:
    processed_paths = []
    for file_path, doc_metadata in zip(file_paths, metadata):
        try:
                # Validate document
            validation_result = await validate_document(file_path)
            if not validation_result.is_valid:
                logger.error(f"Validation failed for {file_path}: {validation_result.errors}")
                continue
                
                # Store document
            storage_location = await store_document(
                    file_path,
                    transform_id,
                    doc_metadata
                )
            logger.info(f"Document stored at {storage_location.original_path}")
                
                # Convert PDF to markdown if needed
            if settings.PDF_PROCESSOR == 'marker':
                if Path(file_path).suffix.lower() == '.pdf':
                    conversion_result = await convert_pdf_to_markdown(
                            file_path=Path(file_path),
                            transform_id=transform_id
                        )
                    if conversion_result:
                        processed_paths.append(conversion_result.markdown_path)
                        logger.info(f"PDF converted to markdown: {conversion_result.markdown_path}")
                else:
                        # For non-PDF files, use the original path
                    processed_paths.append(file_path)
                    logger.info(f"Using original file: {file_path}")
            else:
                    # For gemini files, use the original path
                processed_paths.append(file_path)
                logger.info(f"Using original file: {file_path}")
                
                # Update progress
            await update_stage_progress(transform_id, TransformationStage.PARSE, 
                                            len(processed_paths), len(file_paths))
                
        except Exception as e:
            logger.error(
                    f"Processing failed for file {file_path}",
                    extra={
                        "transform_id": transform_id,
                        "error": str(e)
                    }
                )
            continue

    return processed_paths


async def chunk_documents(transform_id: str, 
                          processed_path: str, 
                          doc_chunk_results: List[Tuple[ChunkingResult, List[ChunkMetadata]]],
                          chunking_config: Optional[Any] = None
                          ) -> List[Tuple[ChunkingResult, List[ChunkMetadata]]]:
    
    result, doc_chunks = await chunk_document(
            file_path=Path(processed_path),
            transform_id=transform_id,
            config=chunking_config
        )
    if doc_chunks:
        doc_chunk_results.append((result, doc_chunks))
        logger.info(f"Document chunked into {len(doc_chunks)} parts")
        
        # Verify chunk quality
        # quality_ok = await check_chunk_quality(doc_chunks)
        # if not quality_ok:
        #     logger.warning(
        #         f"Chunk quality check failed",
        #         extra={"transform_id": transform_id}
        #     )
            
    return doc_chunk_results

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
