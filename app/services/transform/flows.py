from prefect import flow, task
from prefect.context import get_run_context
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import aiofiles
import asyncio
import uuid
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
from prefect.logging import get_run_logger
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
    metadata: List[DocumentMetadata]
) -> Dict[str, Any]:
    """
    Main transformation flow
    
    Args:
        transform_id: Unique ID for this transformation
        ontology_id: ID of the ontology to use
        file_paths: List of paths to documents
        metadata: List of document metadata
    """
    logger.info(f"Starting transformation flow with ID: {transform_id}")
    try:
        
        logger.info(f"Starting transformation flow {transform_id}")
        
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
                doc_chunk_results = await chunk_documents(transform_id, processed_path, doc_chunk_results)
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
                        )
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
            )
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
        
        # Store knowledge graph
        nodes_stored = 0
        relationships_stored = 0
        storage_time = 0
        storage_retries = 0
        for graph in graphs:
            storage_result = await store_knowledge_graph(
                graph,
                transform_id
            )
            nodes_stored = nodes_stored + storage_result.nodes_stored
            relationships_stored = relationships_stored + storage_result.relationships_stored
            storage_time = storage_time + storage_result.metrics.storage_time_ms
            storage_retries = storage_retries + storage_result.metrics.retries
        
        # Update progress with storage metrics
        await update_stage_progress(transform_id, TransformationStage.LOAD, 
                                    (nodes_stored + relationships_stored), 
                                    (total_nodes + total_relationships))
        
        await progress_tracker.complete_stage(
            transform_id,
            TransformationStage.LOAD
        )
        
        # Return flow results
        return {
            'transform_id': transform_id,
            'total_nodes': total_nodes,
            'total_relationships': total_relationships
        }
        
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
            'validate_document': TransformationStage.PARSE,
            'construct_knowledge_graph': TransformationStage.TRANSFORM,
            'store_knowledge_graph': TransformationStage.LOAD
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
                          doc_chunk_results: List[Tuple[ChunkingResult, List[ChunkMetadata]]]
                          ) -> List[Tuple[ChunkingResult, List[ChunkMetadata]]]:
    
    result, doc_chunks = await chunk_document(
            file_path=Path(processed_path),
            transform_id=transform_id
        )
    if doc_chunks:
        doc_chunk_results.append((result, doc_chunks))
        logger.info(f"Document chunked into {len(doc_chunks)} parts")
        
        # Verify chunk quality
        quality_ok = await check_chunk_quality(doc_chunks)
        if not quality_ok:
            logger.warning(
                f"Chunk quality check failed",
                extra={"transform_id": transform_id}
            )
            
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
