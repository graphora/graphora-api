"""Service for storing and retrieving resolution patterns in Qdrant vector database"""

from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
import json
import uuid
from datetime import datetime
import traceback
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.schemas.conflicts import Conflict, ResolutionOption, ConflictType
from app.config import settings
from app.utils.logger import logger
from app.services.embedding import get_embedding, get_batch_embeddings


class ResolutionPattern(BaseModel):
    """Pattern extracted from a conflict resolution"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: str
    entity_types: List[str]
    property_names: Optional[List[str]] = None
    relationship_types: Optional[List[str]] = None
    resolution_strategy: str
    resolution_data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    original_conflict_id: str
    original_merge_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QdrantResolutionStorage:
    """Handles storage and retrieval of resolution patterns in Qdrant"""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance_metric: Optional[str] = None
    ):
        """Initialize Qdrant client and ensure collection exists"""
        # Use settings if not provided
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size or settings.QDRANT_VECTOR_SIZE
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        
        # Parse distance metric
        distance_metric_str = distance_metric or settings.QDRANT_DISTANCE_METRIC
        if distance_metric_str.lower() == "cosine":
            self.distance = models.Distance.COSINE
        elif distance_metric_str.lower() == "euclid":
            self.distance = models.Distance.EUCLID
        elif distance_metric_str.lower() == "dot":
            self.distance = models.Distance.DOT
        else:
            logger.warning(f"Unknown distance metric: {distance_metric_str}, using COSINE")
            self.distance = models.Distance.COSINE
        
        # Initialize client with retry logic
        self._init_client()
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _init_client(self, max_retries: int = 3):
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
                asyncio.sleep(1)  # Wait before retry
    
    def _ensure_collection(self):
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
                        self._create_new_collection()
                    else:
                        logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                except Exception as e:
                    logger.error(f"Error checking collection vector size: {str(e)}")
                    # Assume collection is fine
                    logger.info(f"Using existing Qdrant collection: {self.collection_name}")
            else:
                # Create new collection
                self._create_new_collection()
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            traceback.print_exc()
            raise
    
    def _create_new_collection(self):
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
        self._create_payload_indexes()
        logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def _create_payload_indexes(self):
        """Create payload indexes for efficient filtering"""
        try:
            # Index for conflict_type
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="conflict_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for entity_types
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="entity_types",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index for confidence
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="confidence",
                field_schema=models.PayloadSchemaType.FLOAT
            )
            
            # Index for created_at
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="created_at",
                field_schema=models.PayloadSchemaType.DATETIME
            )
            
            logger.info(f"Created payload indexes for collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Error creating payload indexes: {str(e)}")
    
    async def store_resolution(self, pattern: ResolutionPattern) -> str:
        """Store a resolution pattern in Qdrant"""
        try:
            # Generate embedding if not provided
            if not pattern.embedding:
                pattern.embedding = await self._generate_embedding(pattern)
            
            # Convert ID to a format Qdrant accepts (UUID without hyphens)
            point_id = self._format_point_id(pattern.id)
            
            # Create point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=pattern.embedding,
                        payload=pattern.model_dump(exclude={"embedding"})
                    )
                ]
            )
            
            logger.info(f"Stored resolution pattern: {pattern.id}")
            return pattern.id
        except Exception as e:
            logger.error(f"Error storing resolution pattern: {str(e)}")
            raise
    
    def _format_point_id(self, id_str: str) -> str:
        """Format ID to be compatible with Qdrant (UUID without hyphens)"""
        try:
            # Try to parse as UUID and return without hyphens
            return str(uuid.UUID(id_str)).replace('-', '')
        except ValueError:
            # If not a valid UUID, create a new UUID based on the string
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str)).replace('-', '')
    
    async def search_similar_resolutions(
        self,
        conflict: Conflict,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_by_conflict_type: bool = True
    ) -> List[Tuple[ResolutionPattern, float]]:
        """
        Search for similar resolution patterns
        
        Args:
            conflict: The conflict to find similar resolutions for
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_by_conflict_type: Whether to filter by conflict type
            
        Returns:
            List of (ResolutionPattern, score) tuples
        """
        try:
            # Use default values from settings if not provided
            top_k = top_k or settings.QDRANT_SEARCH_LIMIT
            score_threshold = score_threshold or settings.QDRANT_SCORE_THRESHOLD
            
            # Create query embedding
            query_embedding = await self._generate_embedding_for_conflict(conflict)
            
            # Create filter conditions
            filter_conditions = None
            if filter_by_conflict_type:
                filter_conditions = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="conflict_type",
                            match=models.MatchValue(value=conflict.conflict_type.value)
                        )
                    ]
                )
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Convert to ResolutionPattern objects with scores
            patterns_with_scores = []
            for point in search_result:
                payload = point.payload
                # Use the original ID from the payload instead of the point.id (which is formatted)
                # The original ID is stored in the payload during upsert
                pattern = ResolutionPattern(**payload)
                pattern.embedding = None  # Embedding not included in payload
                patterns_with_scores.append((pattern, point.score))
            
            logger.info(f"Found {len(patterns_with_scores)} similar resolution patterns")
            return patterns_with_scores
        except Exception as e:
            logger.error(f"Error searching similar resolutions: {str(e)}")
            return []
    
    async def batch_store_resolutions(self, patterns: List[ResolutionPattern]) -> List[str]:
        """Store multiple resolution patterns in batch"""
        if not patterns:
            return []
            
        try:
            # Generate embeddings for patterns that don't have them
            patterns_without_embeddings = [p for p in patterns if not p.embedding]
            if patterns_without_embeddings:
                await self._generate_batch_embeddings(patterns_without_embeddings)
            
            # Create points
            points = []
            for pattern in patterns:
                if not pattern.embedding:
                    logger.warning(f"Pattern {pattern.id} has no embedding, skipping")
                    continue
                
                # Convert ID to a format Qdrant accepts
                point_id = self._format_point_id(pattern.id)
                    
                points.append(models.PointStruct(
                    id=point_id,
                    vector=pattern.embedding,
                    payload=pattern.model_dump(exclude={"embedding"})
                ))
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                logger.info(f"Batch stored {len(points)} resolution patterns")
            
            return [pattern.id for pattern in patterns if pattern.embedding]
        except Exception as e:
            logger.error(f"Error batch storing resolution patterns: {str(e)}")
            raise
    
    async def get_resolution_by_id(self, resolution_id: str) -> Optional[ResolutionPattern]:
        """Get resolution pattern by ID"""
        try:
            # Format ID for Qdrant
            point_id = self._format_point_id(resolution_id)
            
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            
            if not points:
                logger.warning(f"Resolution pattern not found: {resolution_id}")
                return None
                
            point = points[0]
            payload = point.payload
            payload["id"] = resolution_id  # Use original ID in the returned object
            
            return ResolutionPattern(**payload)
        except Exception as e:
            logger.error(f"Error retrieving resolution {resolution_id}: {str(e)}")
            return None
    
    async def delete_resolution(self, resolution_id: str) -> bool:
        """Delete resolution pattern"""
        try:
            # Format ID for Qdrant
            point_id = self._format_point_id(resolution_id)
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            logger.info(f"Deleted resolution pattern: {resolution_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting resolution {resolution_id}: {str(e)}")
            return False
    
    async def _generate_embedding(self, pattern: ResolutionPattern) -> List[float]:
        """Generate embedding for a resolution pattern"""
        # Create text representation
        pattern_text = f"""
        Conflict type: {pattern.conflict_type}
        Entity types: {', '.join(pattern.entity_types)}
        Property names: {', '.join(pattern.property_names or [])}
        Relationship types: {', '.join(pattern.relationship_types or [])}
        Resolution strategy: {pattern.resolution_strategy}
        Resolution data: {json.dumps(pattern.resolution_data)}
        """
        
        # Use embedding service
        return await get_embedding(pattern_text)
    
    async def _generate_batch_embeddings(self, patterns: List[ResolutionPattern]):
        """Generate embeddings for multiple patterns in batch"""
        # Create text representations
        texts = []
        for pattern in patterns:
            pattern_text = f"""
            Conflict type: {pattern.conflict_type}
            Entity types: {', '.join(pattern.entity_types)}
            Property names: {', '.join(pattern.property_names or [])}
            Relationship types: {', '.join(pattern.relationship_types or [])}
            Resolution strategy: {pattern.resolution_strategy}
            Resolution data: {json.dumps(pattern.resolution_data)}
            """
            texts.append(pattern_text)
        
        # Get embeddings in batch
        embeddings = await get_batch_embeddings(texts)
        
        # Assign embeddings to patterns
        for i, pattern in enumerate(patterns):
            pattern.embedding = embeddings[i]
    
    async def _generate_embedding_for_conflict(self, conflict: Conflict) -> List[float]:
        """Generate embedding for a conflict"""
        # Create text representation
        entity_type = conflict.entity_type or conflict.context.get('entity_type', '') if conflict.context else ''
        property_name = conflict.property_name or conflict.context.get('property_name', '') if conflict.context else ''
        
        conflict_text = f"""
        Conflict type: {conflict.conflict_type.value}
        Entity type: {entity_type}
        Property name: {property_name}
        Staging IDs: {', '.join(conflict.staging_ids or [])}
        Production IDs: {', '.join(conflict.production_ids or [])}
        Description: {conflict.description}
        """
        
        # Add context data if available
        if conflict.context:
            context_str = ", ".join(f"{k}: {v}" for k, v in conflict.context.items() 
                                  if k not in ['entity_type', 'property_name'])
            conflict_text += f"Context: {context_str}\n"
        
        # Use embedding service
        return await get_embedding(conflict_text)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector storage"""
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Count points
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=None
            )
            
            # Count by conflict type
            conflict_type_counts = {}
            for conflict_type in [ct.value for ct in ConflictType]:
                count = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="conflict_type",
                                match=models.MatchValue(value=conflict_type)
                            )
                        ]
                    )
                )
                conflict_type_counts[conflict_type] = count.count
            
            return {
                "total_patterns": count_result.count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": str(collection_info.config.params.vectors.distance),
                "by_conflict_type": conflict_type_counts
            }
        except Exception as e:
            logger.error(f"Error getting vector storage stats: {str(e)}")
            return {
                "error": str(e),
                "total_patterns": 0
            } 