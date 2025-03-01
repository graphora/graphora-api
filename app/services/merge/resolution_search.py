"""Service for finding similar resolution patterns for conflicts"""

from typing import List, Dict, Any, Optional, Tuple
import time
from app.schemas.conflicts import Conflict, ConflictType
from app.services.storage.vector_storage import QdrantResolutionStorage, ResolutionPattern
from app.services.embedding import get_embedding
from app.utils.logger import logger


class ResolutionEmbeddingGenerator:
    """Generates embeddings for conflicts to use in similarity search"""
    
    async def generate_embedding(self, conflict: Conflict) -> List[float]:
        """Generate embedding for a conflict"""
        conflict_text = self._conflict_to_text(conflict)
        try:
            embedding = await get_embedding(conflict_text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for conflict {conflict.id}: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 1536  # Default size for the embedding model
    
    def _conflict_to_text(self, conflict: Conflict) -> str:
        """Convert conflict to text representation for embedding"""
        text_parts = [
            f"Conflict type: {conflict.conflict_type.value}",
            f"Description: {conflict.description}"
        ]
        
        # Add entity information if available
        if conflict.entity_type:
            text_parts.append(f"Entity type: {conflict.entity_type}")
        
        # Add property information for property conflicts
        if conflict.conflict_type in [ConflictType.PROPERTY_VALUE, ConflictType.PROPERTY_MISSING, ConflictType.PROPERTY_TYPE]:
            if conflict.property_name:
                text_parts.append(f"Property name: {conflict.property_name}")
            elif conflict.context and "property_name" in conflict.context:
                text_parts.append(f"Property name: {conflict.context['property_name']}")
        
        # Add context fields
        for key, value in conflict.context.items():
            if key not in ["entity_type", "property_name"]:  # Skip already added fields
                text_parts.append(f"{key}: {value}")
        
        # Add values if available
        if conflict.staging_value is not None:
            text_parts.append(f"Staging value: {conflict.staging_value}")
        if conflict.production_value is not None:
            text_parts.append(f"Production value: {conflict.production_value}")
            
        return " ".join(text_parts)


class ResolutionPatternSearchService:
    """Service for finding similar resolution patterns"""
    
    def __init__(
        self,
        vector_storage: QdrantResolutionStorage,
        embedding_generator: Optional[ResolutionEmbeddingGenerator] = None,
        similarity_threshold: float = 0.7,
        collection_name: str = "resolution_patterns"
    ):
        self.vector_storage = vector_storage
        self.embedding_generator = embedding_generator or ResolutionEmbeddingGenerator()
        self.similarity_threshold = similarity_threshold
        self.collection_name = collection_name
    
    async def find_similar_resolutions(
        self,
        conflict: Conflict,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[ResolutionPattern, float]]:
        """
        Find similar past resolutions for a conflict
        
        Args:
            conflict: The conflict to find similar resolutions for
            limit: Maximum number of results to return
            filters: Optional additional filters to apply
            
        Returns:
            List of tuples containing (resolution_pattern, similarity_score)
        """
        start_time = time.time()
        
        try:
            logger.info(f"Finding similar resolutions for conflict {conflict.id} with limit {limit}")
            
            # Generate embedding for the conflict
            embedding = await self.embedding_generator.generate_embedding(conflict)
            logger.info(f"Generated embedding for conflict {conflict.id} with length {len(embedding)}")
            
            # Prepare filters
            query_filters = self._build_filters(conflict, filters)
            logger.info(f"Built filters for search: {query_filters}")
            
            # Query vector storage
            logger.info(f"Querying vector storage with threshold {self.similarity_threshold}")
            results = await self.vector_storage.search_similar_resolutions(
                conflict=conflict,
                top_k=limit,
                score_threshold=self.similarity_threshold,
                filter_by_conflict_type=True,
                additional_filters=filters
            )
            
            # Log performance
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.info(f"Resolution pattern search completed in {elapsed_time:.2f}ms, found {len(results)} results")
            
            if results:
                for i, (pattern, score) in enumerate(results):
                    logger.info(f"Result {i+1}: Pattern {pattern.id} with score {score}, strategy: {pattern.resolution_strategy}")
            else:
                logger.warning(f"No similar resolution patterns found for conflict {conflict.id}")
            
            return results
        except Exception as e:
            logger.error(f"Error finding similar resolutions: {str(e)}")
            return []
    
    async def batch_find_similar_resolutions(
        self,
        conflicts: List[Conflict],
        limit_per_conflict: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Tuple[ResolutionPattern, float]]]:
        """
        Find similar past resolutions for multiple conflicts
        
        Args:
            conflicts: List of conflicts to find similar resolutions for
            limit_per_conflict: Maximum number of results to return per conflict
            filters: Optional additional filters to apply
            
        Returns:
            Dictionary mapping conflict IDs to lists of (resolution_pattern, similarity_score) tuples
        """
        start_time = time.time()
        results = {}
        
        try:
            # Process each conflict
            for conflict in conflicts:
                # Find similar resolutions for this conflict
                similar_resolutions = await self.find_similar_resolutions(
                    conflict=conflict,
                    limit=limit_per_conflict,
                    filters=filters  # Pass the filters to find_similar_resolutions
                )
                
                # Store results
                results[conflict.id] = similar_resolutions
            
            # Log performance
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            total_results = sum(len(res) for res in results.values())
            logger.info(f"Batch resolution pattern search completed in {elapsed_time:.2f}ms, found {total_results} results for {len(conflicts)} conflicts")
            
            return results
        except Exception as e:
            logger.error(f"Error in batch finding similar resolutions: {str(e)}")
            return {}
    
    def _build_filters(
        self, 
        conflict: Conflict, 
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build filters for the vector search query"""
        filters = {
            "conflict_type": conflict.conflict_type.value
        }
        
        # Add entity type filter if available
        if conflict.entity_type:
            filters["entity_type"] = conflict.entity_type
        elif conflict.context and "entity_type" in conflict.context:
            filters["entity_type"] = conflict.context["entity_type"]
            
        # Add property name filter for property conflicts
        if conflict.conflict_type in [ConflictType.PROPERTY_VALUE, ConflictType.PROPERTY_MISSING, ConflictType.PROPERTY_TYPE]:
            if conflict.property_name:
                filters["property_name"] = conflict.property_name
            elif conflict.context and "property_name" in conflict.context:
                filters["property_name"] = conflict.context["property_name"]
        
        # Override with custom filters if provided
        if additional_filters:
            filters.update(additional_filters)
        
        return filters 