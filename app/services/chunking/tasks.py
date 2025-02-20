from typing import List, Tuple
import aiofiles
from app.services.chunking.chunker import (
    DocumentChunker,
    ChunkingError
)
from app.services.chunking.models import (
    ChunkMetadata,
    ChunkingResult
)
from app.config import settings
from app.utils.logger import logger
from prefect import task

@task(
    name="document-chunking",
    retries=settings.CHUNKING_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS
)
async def chunk_document(
    file_path: str,
    transform_id: str
) -> Tuple[ChunkingResult, List[ChunkMetadata]]:
    """
    Chunk document into semantically meaningful chunks
    
    Args:
        file_path: Path to document file
        transform_id: Transform ID
        
    Returns:
        Tuple of (ChunkingResult, List[ChunkMetadata])
    """
    try:
        # Initialize chunker with CPU device
        chunker = DocumentChunker()
        
        # Read file content
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            
        logger.info("Starting document chunking")
        result = await chunker.process_document(content, transform_id)
        
        if not result:
            raise ChunkingError("Chunking failed")
            
        return result
        
    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        raise ChunkingError(f"Failed to chunk document: {str(e)}")

@task(
    name="chunk-quality-check",
    retries=settings.CHUNKING_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS
)
async def check_chunk_quality(
    chunks: List[ChunkMetadata]
) -> bool:
    """
    Check quality of document chunks
    
    Args:
        chunks: List of chunk metadata
        
    Returns:
        True if chunks pass quality check, False otherwise
    """
    try:
        if not chunks:
            return False
            
        # Check basic quality metrics
        min_coherence = 0.6
        min_relevance = 0.6
        min_size_score = 0.5
        
        for chunk in chunks:
            if (
                chunk.quality_metrics.coherence_score < min_coherence or
                chunk.quality_metrics.relevance_score < min_relevance or
                chunk.quality_metrics.size_score < min_size_score
            ):
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Quality check failed: {str(e)}")
        return False
