"""Service for generating text embeddings for vector search"""

import asyncio
from typing import List, Optional
from app.config import settings
from app.utils.logger import logger
from app.services.chunking.chunker import DocumentChunker

# Global chunker instance for embeddings
_chunker = None

async def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text.
    
    Args:
        text: The text to embed
        
    Returns:
        List[float]: The embedding vector
    """
    global _chunker
    
    # Initialize chunker if not already done
    if _chunker is None:
        try:
            _chunker = DocumentChunker()
            logger.info("Initialized embedding service")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise
    
    try:
        # Use the chunker's embedding model
        embedding = _chunker.embeddings.embed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return zero vector as fallback
        return [0.0] * 1536  # Default size for OpenAI embeddings

async def get_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batch.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    # Process in parallel
    tasks = [get_embedding(text) for text in texts]
    return await asyncio.gather(*tasks) 