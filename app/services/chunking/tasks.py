from pathlib import Path
from typing import Optional, List, Tuple
from prefect import task, get_run_logger
import json
import aiofiles

from app.services.chunking.chunker import HybridChunker, ChunkingError
from app.services.chunking.models import ChunkingResult
from app.config import settings

async def store_chunking_result(
    result: ChunkingResult,
    transform_id: str,
    base_path: Path
) -> List[Path]:
    """
    Store chunking results and metadata
    
    Args:
        result: Chunking result to store
        transform_id: Transform ID
        base_path: Base storage path
        
    Returns:
        List of paths to stored chunk files
    """
    # Create directory structure
    transform_dir = base_path / transform_id
    chunks_dir = transform_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_paths = []
    
    # Store individual chunks
    for i, (chunk, metadata) in enumerate(zip(result.chunks, result.chunk_metadata)):
        chunk_path = chunks_dir / f"chunk_{i+1}.txt"
        async with aiofiles.open(chunk_path, "w") as f:
            await f.write(chunk)
        chunk_paths.append(chunk_path)
    
    # Store metadata
    metadata_path = transform_dir / "chunking_metadata.json"
    async with aiofiles.open(metadata_path, "w") as f:
        await f.write(result.model_dump_json(indent=2))
    
    return chunk_paths

def log_chunking_metrics(metrics: dict, transform_id: str) -> None:
    """Log metrics in structured format for Prefect UI"""
    logger = get_run_logger()
    
    logger.info(
        "Chunking Metrics",
        extra={
            "transform_id": transform_id,
            "metrics": {
                "performance": {
                    "total_time_ms": metrics["total_processing_time_ms"],
                    "embedding_time_ms": metrics["embedding_time_ms"],
                    "refinement_time_ms": metrics["chunk_refinement_time_ms"]
                },
                "chunks": {
                    "total": metrics["total_chunks"],
                    "avg_size": metrics["avg_chunk_size"],
                    "max_size": metrics["max_chunk_size"],
                    "min_size": metrics["min_chunk_size"]
                },
                "quality": {
                    "semantic_splits": metrics["semantic_splits"],
                    "forced_splits": metrics["forced_splits"],
                    "merged_chunks": metrics["merged_chunks"]
                },
                "resources": {
                    "peak_memory_mb": metrics["peak_memory_mb"],
                    "embedding_latency_ms": metrics["embedding_model_latency_ms"]
                }
            }
        }
    )

@task(
    name="document-chunking",
    retries=settings.CHUNKING_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
    tags=["processing", "chunking"]
)
async def chunk_document(
    file_path: Path,
    transform_id: str
) -> Optional[Tuple[ChunkingResult, List[Path]]]:
    """
    Chunk document into semantically meaningful parts
    
    Args:
        file_path: Path to the document file
        transform_id: Transform ID for tracking
        
    Returns:
        List of paths to chunk files if successful, None if skipped
    """
    logger = get_run_logger()
    
    try:
        # Read document content
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
        
        logger.info(
            f"Starting document chunking",
            extra={
                "transform_id": transform_id,
                "file_path": str(file_path),
                "content_size": len(content)
            }
        )
        
        # Initialize chunker and process document
        chunker = HybridChunker()
        result = await chunker.process_document(content, transform_id)
        
        # Log metrics
        log_chunking_metrics(result.metrics, transform_id)
        
        # Store results
        chunk_paths = await store_chunking_result(
            result,
            transform_id,
            Path(settings.UPLOAD_DIR)
        )
        
        logger.info(
            f"Chunking completed successfully",
            extra={
                "transform_id": transform_id,
                "num_chunks": len(chunk_paths)
            }
        )
        
        return (result, chunk_paths)
        
    except Exception as e:
        logger.error(
            f"Chunking failed",
            extra={
                "transform_id": transform_id,
                "error": str(e)
            }
        )
        raise ChunkingError(f"Failed to chunk document: {str(e)}")

@task(
    name="chunk-quality-check",
    retries=2,
    tags=["monitoring"]
)
async def check_chunk_quality(
    chunk_paths: List[Path],
    transform_id: str
) -> bool:
    """
    Verify quality of generated chunks
    
    Args:
        chunk_paths: List of paths to chunk files
        transform_id: Transform ID
        
    Returns:
        True if quality checks pass, False otherwise
    """
    logger = get_run_logger()
    
    try:
        # Read metadata
        metadata_path = Path(settings.UPLOAD_DIR) / transform_id / "chunking_metadata.json"
        async with aiofiles.open(metadata_path, "r") as f:
            metadata = json.loads(await f.read())
        
        # Verify chunk count
        if len(chunk_paths) != metadata["total_chunks"]:
            logger.warning(
                f"Chunk count mismatch",
                extra={
                    "transform_id": transform_id,
                    "expected": metadata["total_chunks"],
                    "actual": len(chunk_paths)
                }
            )
            return False
        
        # Verify chunk sizes
        for chunk_path in chunk_paths:
            async with aiofiles.open(chunk_path, "r") as f:
                content = await f.read()
            
            if not settings.MIN_CHUNK_SIZE <= len(content) <= settings.MAX_CHUNK_SIZE:
                logger.warning(
                    f"Chunk size outside bounds",
                    extra={
                        "transform_id": transform_id,
                        "chunk_path": str(chunk_path),
                        "size": len(content)
                    }
                )
                return False
        
        return True
        
    except Exception as e:
        logger.error(
            f"Quality check failed",
            extra={
                "transform_id": transform_id,
                "error": str(e)
            }
        )
        return False
