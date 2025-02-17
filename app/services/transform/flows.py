from pathlib import Path
from prefect import flow, task
from prefect.context import get_run_context
from datetime import datetime
from typing import List

from app.schemas.transform import (
    DocumentMetadata,
    ValidationResult,
    StorageLocation,
    TransformStatus
)
from app.services.transform.validators import FileValidator
from app.services.transform.storage import DocumentStorage
from app.config import settings
import aiofiles
from fastapi import UploadFile
from prefect.logging import get_run_logger
from app.services.marker.tasks import convert_pdf_to_markdown

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
    version="1.0.0"
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
    flow_run = get_run_context().flow_run
    logger = get_run_logger()
    
    logger.info(f"Starting transformation flow {transform_id}")
    
    processed_paths = []
    
    for file_path, doc_metadata in zip(file_paths, metadata):
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
        
        # Future tasks will be added here:
        # - Parse document
        # - Chunk content
        # - Extract information
        # - Transform to graph
        # - Load to database
