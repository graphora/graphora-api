"""Qdrant implementation of vector storage interface"""

from typing import List, Dict, Any, Optional, Union
import asyncio
import uuid
import traceback
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.services.storage.vector_storage_interface import VectorStorageInterface, ResolutionPattern
from app.config import settings
from app.utils.logger import logger


class QdrantVectorStorage(VectorStorageInterface):
    """Qdrant implementation of vector storage interface"""
    
    def __init__(
        self,
        collection_name: str = "resolution_patterns",
        vector_size: int = 384,  # Default size for sentence embeddings
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance_metric: str = "cosine"
    ):
        """Initialize Qdrant storage with configuration parameters"""
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        
        # Parse distance metric
        if distance_metric.lower() == "cosine":
            self.distance = models.Distance.COSINE
        elif distance_metric.lower() == "euclid":
            self.distance = models.Distance.EUCLID
        elif distance_metric.lower() == "dot":
            self.distance = models.Distance.DOT
        else:
            logger.warning(f"Unknown distance metric: {distance_metric}, using COSINE")
            self.distance = models.Distance.COSINE
        
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize Qdrant client and ensure collection exists"""
        # Create client with retry logic
        await self._init_client()
        
        # Ensure collection exists
        await self._ensure_collection()
    
    async def _init_client(self, max_retries: int = 3):
        """Initialize Qdrant client with retry logic"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
                # Test connection
                self.client.get_collections()
                logger.info(f"Connected to Qdrant at {self.url}")
                return
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to connect to Qdrant after {max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Qdrant connection attempt {retry_count} failed: {str(e)}. Retrying...")
                await asyncio.sleep(1)  # Wait before retry
    
    async def _ensure_collection(self):
        """Create collection if it doesn't exist or recreate if vector size doesn't match"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            # Check if collection exists
            if self.collection_name in collection_names:
                # Check if vector size matches
                try:
                    collection_info = self.client.get_collection(self.collection_name)
                    actual_vector_size = collection_info.config.params.vectors.size
                    
                    if actual_vector_size != self.vector_size:
                        logger.warning(f"Vector size mismatch: collection has {actual_vector_size}, expected {self.vector_size}. Recreating collection.")
                        # Delete and recreate collection
                        self.client.delete_collection(self.collection_name)
                        await self._create_new_collection()
                    else:
                        logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                except Exception as e:
                    logger.error(f"Error checking collection vector size: {str(e)}")
                    # Assume collection is fine
                    logger.info(f"Using existing Qdrant collection: {self.collection_name}")
            else:
                # Create new collection
                await self._create_new_collection()
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            traceback.print_exc()
            raise
    
    async def _create_new_collection(self):
        """Create a new collection with the current settings"""
        logger.info(f"Creating Qdrant collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=self.distance
            )
        )
        
        # Create payload index for faster filtering
        await self._create_payload_indexes()
        logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    async def _create_payload_indexes(self):
        """Create payload indexes for efficient filtering"""
        try:
            # Index for conflict_type
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="conflict_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for confidence
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="confidence",
                field_schema=models.PayloadSchemaType.FLOAT
            )
            
            logger.info(f"Created payload indexes for collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Error creating payload indexes: {str(e)}")
    
    def _format_point_id(self, id_str: str) -> str:
        """Format ID to be compatible with Qdrant (UUID without hyphens)"""
        try:
            # Try to parse as UUID and return without hyphens
            return str(uuid.UUID(id_str)).replace('-', '')
        except ValueError:
            # If not a valid UUID, create a new UUID based on the string
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str)).replace('-', '')
    
    async def store_pattern(self, pattern: ResolutionPattern) -> str:
        """Store a resolution pattern in Qdrant"""
        if not self.client:
            await self.initialize()
            
        try:
            # Format ID for Qdrant
            point_id = self._format_point_id(pattern.id)
            
            # Create point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=pattern.vector,
                        payload={
                            "conflict_type": pattern.conflict_type,
                            "resolution_strategy": pattern.resolution_strategy,
                            "context_features": pattern.context_features,
                            "metadata": pattern.metadata,
                            "confidence": pattern.confidence
                        }
                    )
                ]
            )
            
            logger.info(f"Stored resolution pattern: {pattern.id}")
            return pattern.id
        except Exception as e:
            logger.error(f"Error storing resolution pattern: {str(e)}")
            raise
    
    async def get_pattern(self, pattern_id: str) -> Optional[ResolutionPattern]:
        """Retrieve a pattern by ID"""
        if not self.client:
            await self.initialize()
            
        try:
            # Format ID for Qdrant
            point_id = self._format_point_id(pattern_id)
            
            # Get point
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return None
                
            point = points[0]
            
            # Convert to ResolutionPattern
            payload = point.payload
            return ResolutionPattern(
                id=pattern_id,
                conflict_type=payload.get("conflict_type"),
                resolution_strategy=payload.get("resolution_strategy"),
                context_features=payload.get("context_features", {}),
                vector=point.vector,
                metadata=payload.get("metadata", {}),
                confidence=payload.get("confidence", 0.0)
            )
        except Exception as e:
            logger.error(f"Error retrieving pattern {pattern_id}: {str(e)}")
            return None
    
    async def update_pattern(self, pattern: ResolutionPattern) -> bool:
        """Update an existing pattern"""
        if not self.client:
            await self.initialize()
            
        try:
            # Check if exists
            existing = await self.get_pattern(pattern.id)
            if not existing:
                return False
                
            # Update point (reuse store_pattern)
            await self.store_pattern(pattern)
            return True
        except Exception as e:
            logger.error(f"Error updating pattern {pattern.id}: {str(e)}")
            return False
    
    async def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern"""
        if not self.client:
            await self.initialize()
            
        try:
            # Format ID for Qdrant
            point_id = self._format_point_id(pattern_id)
            
            # Check if the point exists first
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=False,
                with_vectors=False
            )
            
            # If point doesn't exist, return False
            if not points:
                return False
                
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            logger.info(f"Deleted resolution pattern: {pattern_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting pattern {pattern_id}: {str(e)}")
            return False
    
    async def search_similar(
        self, 
        vector: List[float], 
        conflict_type: Optional[str] = None,
        limit: int = 10, 
        threshold: float = 0.7
    ) -> List[ResolutionPattern]:
        """Search for similar patterns"""
        if not self.client:
            await self.initialize()
            
        try:
            # Prepare filter
            filter_query = None
            if conflict_type:
                filter_query = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="conflict_type",
                            match=models.MatchValue(value=conflict_type)
                        )
                    ]
                )
                
            # Search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                query_filter=filter_query,
                with_payload=True,
                with_vectors=True,
                score_threshold=threshold
            )
            
            # Convert to ResolutionPattern objects
            patterns = []
            for result in search_results:
                payload = result.payload
                pattern = ResolutionPattern(
                    id=str(result.id).replace('_', '-'),  # Convert back from Qdrant ID format
                    conflict_type=payload.get("conflict_type"),
                    resolution_strategy=payload.get("resolution_strategy"),
                    context_features=payload.get("context_features", {}),
                    vector=result.vector,
                    metadata=payload.get("metadata", {}),
                    confidence=payload.get("confidence", 0.0)
                )
                patterns.append(pattern)
                
            return patterns
        except Exception as e:
            logger.error(f"Error searching similar patterns: {str(e)}")
            return []
    
    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 10
    ) -> List[ResolutionPattern]:
        """Search patterns using metadata filters"""
        if not self.client:
            await self.initialize()
            
        try:
            # Prepare filter conditions
            filter_conditions = []
            for key, value in filters.items():
                if key.startswith("metadata."):
                    # For nested metadata fields
                    field = key
                else:
                    # For top-level fields
                    field = key
                    
                filter_conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value)
                    )
                )
                
            # Create filter
            filter_query = models.Filter(
                must=filter_conditions
            ) if filter_conditions else None
            
            # Search
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                scroll_filter=filter_query,
                with_payload=True,
                with_vectors=True
            )[0]  # First element is points, second is next page offset
            
            # Convert to ResolutionPattern objects
            patterns = []
            for result in search_results:
                payload = result.payload
                pattern = ResolutionPattern(
                    id=str(result.id).replace('_', '-'),  # Convert back from Qdrant ID format
                    conflict_type=payload.get("conflict_type"),
                    resolution_strategy=payload.get("resolution_strategy"),
                    context_features=payload.get("context_features", {}),
                    vector=result.vector,
                    metadata=payload.get("metadata", {}),
                    confidence=payload.get("confidence", 0.0)
                )
                patterns.append(pattern)
                
            return patterns
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            return []
    
    async def close(self) -> None:
        """Close Qdrant client"""
        # Qdrant client doesn't have an explicit close method
        # but we can set it to None to free resources
        self.client = None 