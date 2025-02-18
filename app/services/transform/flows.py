from pathlib import Path
from prefect import flow, task
from prefect.context import get_run_context
from datetime import datetime
from typing import List
from app.schemas.transform import (
    DocumentMetadata,
    ValidationResult,
    StorageLocation
)
from app.services.transform.validators import FileValidator
from app.services.transform.storage import DocumentStorage
from app.config import settings
import aiofiles
from fastapi import UploadFile
from prefect.logging import get_run_logger
from app.services.marker.tasks import convert_pdf_to_markdown
from app.services.chunking.tasks import chunk_document, check_chunk_quality
from app.services.transform.progress_tracker import ProgressTracker
from app.services.transform.status_models import (
    TransformationStage,
    ErrorSummary
)

progress_tracker = ProgressTracker()

@task(
    name="document-validation",
    retries=3,
    retry_delay_seconds=60,
    tags=["transform", "validation"]
)
async def validate_document(file_path: Path) -> ValidationResult:
    """Validate document before processing"""
    validator = FileValidator()
    # Convert Path to UploadFile for validation
    async with aiofiles.open(file_path, 'rb') as f:
        content = await f.read()
    
    upload_file = UploadFile(
        filename=file_path.name,
        file=content
    )
    return await validator.validate(upload_file)

@task(
    name="document-storage",
    retries=2,
    retry_delay_seconds=30,
    tags=["transform", "storage"]
)
async def store_document(
    file_path: Path,
    transform_id: str,
    metadata: DocumentMetadata
) -> StorageLocation:
    """Store document and metadata"""
    storage = DocumentStorage(Path(settings.UPLOAD_DIR))
    async with aiofiles.open(file_path, 'rb') as f:
        content = await f.read()
    
    upload_file = UploadFile(
        filename=file_path.name,
        file=content
    )
    return await storage.save_document(upload_file, transform_id, metadata)

@flow(
    name="document-transformation",
    description="Transform document to knowledge graph",
    version="1.0.0",
    log_prints=True
)
async def document_transformation_flow(
    transform_id: str,
    file_paths: List[Path],
    metadata: List[DocumentMetadata]
) -> None:
    """
    Main document transformation flow
    
    Args:
        transform_id: Unique transform ID
        file_paths: List of paths to uploaded files
        metadata: List of document metadata
    """
    logger = get_run_logger()
    
    try:
        # Initialize progress tracking
        await progress_tracker.initialize_transform(transform_id)
        
        # Start PARSE stage
        await progress_tracker.start_stage(
            transform_id,
            TransformationStage.PARSE
        )
        
        logger.info(f"Starting transformation flow {transform_id}")
        
        processed_paths = []
        chunk_paths = []
        
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
                if file_path.suffix.lower() == '.pdf':
                    conversion_result = await convert_pdf_to_markdown(
                        file_path=file_path,
                        transform_id=transform_id
                    )
                    if conversion_result:
                        processed_paths.extend(conversion_result.markdown_paths)
                        logger.info(f"PDF converted to markdown: {conversion_result.markdown_paths}")
                else:
                    # For non-PDF files, use the original path
                    processed_paths.append(str(file_path))
                    logger.info(f"Using original file: {file_path}")
                
                # Chunk document
                doc_chunks = await chunk_document(
                    file_path=Path(processed_paths[-1]),
                    transform_id=transform_id
                )
                if doc_chunks:
                    chunk_paths.extend(doc_chunks)
                    logger.info(f"Document chunked into {len(doc_chunks)} parts")
                    
                    # Verify chunk quality
                    quality_ok = await check_chunk_quality(doc_chunks, transform_id)
                    if not quality_ok:
                        logger.warning(
                            f"Chunk quality check failed",
                            extra={"transform_id": transform_id}
                        )
                
                # Update progress
                await progress_tracker.update_stage_progress(
                    transform_id,
                    TransformationStage.PARSE,
                    len(processed_paths),
                    len(file_paths)
                )
            
            except Exception as e:
                logger.error(
                    f"Processing failed for file {file_path}",
                    extra={
                        "transform_id": transform_id,
                        "error": str(e)
                    }
                )
                continue
        
        await progress_tracker.complete_stage(
            transform_id,
            TransformationStage.PARSE
        )
        
        # Future tasks will be added here:
        # - Extract information
        # - Transform to graph
        # - Load to database
        
    except Exception as e:
        logger.error(f"Transform failed: {str(e)}")
        
        # Get current stage from context
        context = get_run_context()
        current_task = context.task_run.task_key if context.task_run else None
        stage_map = {
            'chunk_document': TransformationStage.CHUNK,
            'check_chunk_quality': TransformationStage.CHUNK,
            'convert_pdf_to_markdown': TransformationStage.PARSE,
            'store_document': TransformationStage.PARSE,
            'validate_document': TransformationStage.PARSE
        }
        current_stage = stage_map.get(
            current_task,
            TransformationStage.PARSE
        )
        
        # Record error
        error = ErrorSummary(
            stage=current_stage,
            error_type=type(e).__name__,
            error_message=str(e),
            error_timestamp=datetime.now(),
            stack_trace=None,  # TODO: Add stack trace
            affected_components=[current_task] if current_task else [],
            retry_count=context.task_run.run_count if context.task_run else 0,
            is_recoverable=True,  # TODO: Determine if recoverable
            recovery_instructions=None  # TODO: Add recovery instructions
        )
        
        await progress_tracker.fail_stage(
            transform_id,
            current_stage,
            error
        )
        
        raise
