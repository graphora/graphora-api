"""Factory for creating vector storage instances"""

from typing import Optional

from app.services.storage.vector_storage_interface import VectorStorageInterface
from app.services.storage.qdrant_storage import QdrantVectorStorage
from app.config import settings


def get_vector_storage(
    storage_type: Optional[str] = None,
    collection_name: str = "resolution_patterns",
    vector_size: Optional[int] = None
) -> VectorStorageInterface:
    """Factory function to get the appropriate vector storage implementation
    
    Args:
        storage_type: Type of vector storage to use (e.g., "qdrant")
        collection_name: Name of the collection to use
        vector_size: Size of the vectors to store
        
    Returns:
        An instance of a VectorStorageInterface implementation
        
    Raises:
        ValueError: If the specified storage type is not supported
    """
    # Use settings if not provided
    storage_type = storage_type or getattr(settings, "VECTOR_STORAGE_TYPE", "qdrant")
    vector_size = vector_size or getattr(settings, "QDRANT_VECTOR_SIZE", 384)
    
    # Create the appropriate storage implementation
    if storage_type.lower() == "qdrant":
        return QdrantVectorStorage(
            collection_name=collection_name,
            vector_size=vector_size,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            distance_metric=getattr(settings, "QDRANT_DISTANCE_METRIC", "cosine")
        )
    # Add more implementations as needed
    else:
        raise ValueError(f"Unsupported vector storage type: {storage_type}") 