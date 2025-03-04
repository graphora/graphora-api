"""Service for extracting and managing resolution patterns from conflict resolutions"""

import uuid
import json
import redis
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ResolutionOption,
    ResolutionPattern
)
from app.config import settings
from app.utils.logger import logger
from app.services.chunking.chunker import DocumentChunker

class ResolutionPatternExtractor:
    """Service to extract resolution patterns from conflicts"""
    
    def __init__(self):
        """Initialize service with Redis connection and embedding model"""
        self.redis = redis.Redis.from_url(settings.REDIS_URL)
        try:
            self.chunker = DocumentChunker()
            logger.info("Initialized pattern extractor with embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    async def extract_pattern(
        self,
        conflict: Conflict,
        resolution: ResolutionOption,
        success: bool = True
    ) -> ResolutionPattern:
        """Extract a pattern from a resolved conflict"""
        # Create new pattern with basic metadata
        pattern = ResolutionPattern(
            id=str(uuid.uuid4()),
            conflict_type=conflict.conflict_type,
            resolution_action=resolution.resolution_type,
            confidence_score=resolution.confidence if success else resolution.confidence * 0.5,
            occurrence_count=1,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Extract type-specific pattern elements
        pattern = await self._extract_type_specific_features(conflict, resolution, pattern)
        
        # Generate embedding for the pattern
        pattern.embedding = await self._generate_embedding(pattern)
        
        # Store pattern in Redis
        await self._store_pattern(pattern)
        
        return pattern
    
    async def _extract_type_specific_features(
        self,
        conflict: Conflict,
        resolution: ResolutionOption,
        pattern: ResolutionPattern
    ) -> ResolutionPattern:
        """Extract features specific to the conflict type"""
        
        # Extract context features
        context_features = {}
        
        # Add entity type if available
        if conflict.entity_type:
            context_features["entity_type"] = conflict.entity_type
        elif "entity_type" in conflict.context:
            context_features["entity_type"] = conflict.context["entity_type"]
            
        # Add property name for property conflicts
        if conflict.conflict_type in [ConflictType.PROPERTY_VALUE, ConflictType.PROPERTY_MISSING]:
            if conflict.property_name:
                context_features["property_name"] = conflict.property_name
            elif "property_name" in conflict.context:
                context_features["property_name"] = conflict.context["property_name"]
                
        # Add relationship types for relationship conflicts
        if conflict.conflict_type in [ConflictType.RELATIONSHIP_TYPE, ConflictType.RELATIONSHIP_DIRECTION]:
            if "staging_type" in conflict.context:
                context_features["staging_rel_type"] = conflict.context["staging_type"]
            if "production_type" in conflict.context:
                context_features["production_rel_type"] = conflict.context["production_type"]
        
        pattern.context_features = context_features
        
        # Extract condition features based on resolution type
        condition_features = {}
        
        if resolution.resolution_type == "keep_staging":
            condition_features.update({
                "property_importance": await self._determine_property_importance(
                    context_features.get("property_name"),
                    context_features.get("entity_type")
                ),
                "is_newer": True  # Assumption: staging data is newer
            })
        elif resolution.resolution_type == "keep_production":
            condition_features.update({
                "property_importance": await self._determine_property_importance(
                    context_features.get("property_name"),
                    context_features.get("entity_type")
                ),
                "is_newer": False
            })
        elif resolution.resolution_type == "merge_values":
            condition_features["can_merge"] = True
            
        pattern.condition_features = condition_features
        
        # Set resolution parameters
        pattern.resolution_params = resolution.resolution_data
        
        return pattern
    
    async def _determine_property_importance(
        self,
        property_name: Optional[str],
        entity_type: Optional[str]
    ) -> float:
        """Determine importance score for a property"""
        if not property_name:
            return 0.5
            
        # Higher importance for key properties
        if property_name.lower() in ["id", "name", "key", "identifier"]:
            return 0.9
            
        # Medium importance for descriptive properties
        if property_name.lower() in ["description", "title", "label"]:
            return 0.7
            
        # Lower importance for metadata properties
        if property_name.lower() in ["created_at", "updated_at", "metadata"]:
            return 0.3
            
        # Default medium-low importance
        return 0.5
    
    async def _generate_embedding(self, pattern: ResolutionPattern) -> List[float]:
        """Generate vector embedding for pattern"""
        # Combine relevant features into text representation
        features_text = f"{pattern.conflict_type} {pattern.resolution_action} "
        features_text += " ".join(f"{k}:{v}" for k, v in pattern.context_features.items())
        features_text += " ".join(f"{k}:{v}" for k, v in pattern.condition_features.items())
        
        # Use chunker's embedding model to generate vector
        try:
            embedding = await self._get_text_embedding(features_text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return [0.0] * 384  # Return zero vector as fallback
    
    async def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using chunker's embedding model"""
        try:
            # Use the chunker's embedding model
            embedding = self.chunker.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def _store_pattern(self, pattern: ResolutionPattern):
        """Store pattern in Redis with indexes"""
        # Store main pattern
        self.redis.set(
            f"resolution_pattern:{pattern.id}",
            pattern.model_dump_json()
        )
        
        # Add to conflict type index
        self.redis.sadd(
            f"pattern_index:conflict_type:{pattern.conflict_type.value}",
            pattern.id
        )
        
        # Add to entity type index if present
        if "entity_type" in pattern.context_features:
            self.redis.sadd(
                f"pattern_index:entity_type:{pattern.context_features['entity_type']}",
                pattern.id
            )
        
        # Add to property name index if present
        if "property_name" in pattern.context_features:
            self.redis.sadd(
                f"pattern_index:property_name:{pattern.context_features['property_name']}",
                pattern.id
            )
        
        # Store embedding for similarity search
        if pattern.embedding:
            self.redis.hset(
                "pattern_embeddings",
                pattern.id,
                json.dumps(pattern.embedding)
            )
    
    async def find_similar_patterns(
        self,
        conflict: Conflict,
        min_confidence: float = 0.6,
        limit: int = 5
    ) -> List[Tuple[ResolutionPattern, float]]:
        """Find similar patterns for a given conflict"""
        # Get candidate patterns
        candidates = await self._get_candidate_patterns(conflict)
        
        if not candidates:
            return []
            
        # Generate query embedding
        query_embedding = await self._generate_conflict_embedding(conflict)
        
        # Calculate similarity scores
        pattern_scores = []
        for pattern in candidates:
            if pattern.embedding and pattern.confidence_score >= min_confidence:
                similarity = self._calculate_similarity(query_embedding, pattern.embedding)
                pattern_scores.append((pattern, similarity))
        
        # Sort by similarity score and return top matches
        return sorted(pattern_scores, key=lambda x: x[1], reverse=True)[:limit]
    
    async def _get_candidate_patterns(self, conflict: Conflict) -> List[ResolutionPattern]:
        """Get candidate patterns for similarity search"""
        candidates = set()
        
        # Get patterns for this conflict type
        type_matches = self.redis.smembers(
            f"pattern_index:conflict_type:{conflict.conflict_type.value}"
        )
        candidates.update(type_matches)
        
        # Filter by entity type if available
        if conflict.entity_type:
            entity_matches = self.redis.smembers(
                f"pattern_index:entity_type:{conflict.entity_type}"
            )
            if candidates:
                candidates &= entity_matches
            else:
                candidates.update(entity_matches)
        
        # Filter by property name for property conflicts
        if conflict.conflict_type in [ConflictType.PROPERTY_VALUE, ConflictType.PROPERTY_MISSING]:
            if conflict.property_name:
                prop_matches = self.redis.smembers(
                    f"pattern_index:property_name:{conflict.property_name}"
                )
                if candidates:
                    candidates &= prop_matches
                else:
                    candidates.update(prop_matches)
        
        # Get full pattern objects
        patterns = []
        for pattern_id in candidates:
            pattern_id_str = pattern_id.decode('utf-8') if isinstance(pattern_id, bytes) else pattern_id
            pattern_json = self.redis.get(f"resolution_pattern:{pattern_id_str}")
            if pattern_json:
                pattern = ResolutionPattern.model_validate_json(pattern_json)
                # Get embedding
                embedding_json = self.redis.hget("pattern_embeddings", pattern.id)
                if embedding_json:
                    pattern.embedding = json.loads(embedding_json)
                patterns.append(pattern)
        
        return patterns
    
    async def _generate_conflict_embedding(self, conflict: Conflict) -> List[float]:
        """Generate embedding for a conflict"""
        # Create text representation of conflict
        conflict_text = f"{conflict.conflict_type} "
        if conflict.entity_type:
            conflict_text += f"entity_type:{conflict.entity_type} "
        if conflict.property_name:
            conflict_text += f"property_name:{conflict.property_name} "
        if conflict.context:
            conflict_text += " ".join(f"{k}:{v}" for k, v in conflict.context.items())
            
        return await self._get_text_embedding(conflict_text)
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Convert to numpy arrays for efficient calculation
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 * norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    async def update_pattern_stats(
        self,
        pattern_id: str,
        success: bool = True,
        feedback: Optional[str] = None
    ):
        """Update pattern statistics based on usage"""
        pattern_json = self.redis.get(f"resolution_pattern:{pattern_id}")
        if not pattern_json:
            return
            
        pattern = ResolutionPattern.model_validate_json(pattern_json)
        
        # Update occurrence count
        pattern.occurrence_count += 1
        
        # Adjust confidence score based on success
        if success:
            # Increase confidence, but cap at 1.0
            pattern.confidence_score = min(
                1.0,
                pattern.confidence_score + (0.1 / pattern.occurrence_count)
            )
        else:
            # Decrease confidence more significantly on failure
            pattern.confidence_score *= 0.8
        
        pattern.updated_at = datetime.now()
        
        # Store updated pattern
        self.redis.set(
            f"resolution_pattern:{pattern.id}",
            pattern.model_dump_json()
        ) 