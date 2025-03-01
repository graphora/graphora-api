"""Service for learning from past resolutions and automatically applying learned patterns"""

from typing import List, Dict, Any, Optional, Tuple, Set
import time
import asyncio
from datetime import datetime, timedelta
import json
import uuid
from pydantic import BaseModel, Field

from app.schemas.conflicts import Conflict, ConflictType, ResolutionOption
from app.services.storage.vector_storage import QdrantResolutionStorage, ResolutionPattern
from app.services.merge.resolution_search import ResolutionPatternSearchService
from app.services.embedding import get_embedding
from app.utils.logger import logger
from app.config import settings


class ResolutionLearningConfig(BaseModel):
    """Configuration for the resolution learning service"""
    high_confidence_threshold: float = Field(
        default=settings.RESOLUTION_LEARNING_HIGH_CONFIDENCE,
        description="Threshold for automatic application of learned resolutions"
    )
    medium_confidence_threshold: float = Field(
        default=settings.RESOLUTION_LEARNING_MEDIUM_CONFIDENCE,
        description="Threshold for suggesting learned resolutions"
    )
    rate_limit_per_minute: int = Field(
        default=settings.RESOLUTION_LEARNING_RATE_LIMIT,
        description="Maximum number of automatic resolutions per minute"
    )
    blacklisted_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Patterns that should not be automatically applied"
    )
    weight_factors: Dict[str, float] = Field(
        default_factory=lambda: {
            "similarity_score": 0.6,
            "success_rate": 0.3,
            "recency": 0.1
        },
        description="Weight factors for confidence score calculation"
    )
    min_success_rate: float = Field(
        default=settings.RESOLUTION_LEARNING_MIN_SUCCESS_RATE,
        description="Minimum success rate required for automatic application"
    )
    min_resolution_count: int = Field(
        default=settings.RESOLUTION_LEARNING_MIN_RESOLUTION_COUNT,
        description="Minimum number of successful resolutions required for learning"
    )
    max_resolution_age_days: int = Field(
        default=settings.RESOLUTION_LEARNING_MAX_AGE_DAYS,
        description="Maximum age of resolutions to consider for learning (in days)"
    )


class ResolutionLearningService:
    """Service for learning from past resolutions and automatically applying learned patterns"""
    
    def __init__(
        self,
        vector_storage: Optional[QdrantResolutionStorage] = None,
        search_service: Optional[ResolutionPatternSearchService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the resolution learning service"""
        # Initialize storage and search service
        self.vector_storage = vector_storage or QdrantResolutionStorage()
        self.search_service = search_service or ResolutionPatternSearchService(self.vector_storage)
        
        # Load configuration
        config_dict = config or {}
        self.config = ResolutionLearningConfig(**config_dict)
        
        # Initialize rate limiting
        self.last_applied_resolutions: List[datetime] = []
        self.blacklist_cache: Set[str] = set()
        self._load_blacklist()
    
    def _load_blacklist(self) -> None:
        """Load blacklisted patterns into memory for faster checking"""
        for pattern in self.config.blacklisted_patterns:
            pattern_key = self._generate_pattern_key(pattern)
            self.blacklist_cache.add(pattern_key)
    
    def _generate_pattern_key(self, pattern: Dict[str, Any]) -> str:
        """Generate a unique key for a pattern to use in blacklist checking"""
        # Create a deterministic string representation of the pattern
        key_parts = []
        
        # Add conflict type if present
        if "conflict_type" in pattern:
            key_parts.append(f"type:{pattern['conflict_type']}")
        
        # Add entity types if present
        if "entity_types" in pattern:
            entity_types = sorted(pattern["entity_types"])
            key_parts.append(f"entities:{','.join(entity_types)}")
        
        # Add property names if present
        if "property_names" in pattern:
            property_names = sorted(pattern["property_names"])
            key_parts.append(f"props:{','.join(property_names)}")
        
        # Add resolution strategy if present
        if "resolution_strategy" in pattern:
            key_parts.append(f"strategy:{pattern['resolution_strategy']}")
        
        return "|".join(key_parts)
    
    def _is_blacklisted(self, pattern: ResolutionPattern) -> bool:
        """Check if a resolution pattern is blacklisted"""
        # Convert pattern to dict for key generation
        pattern_dict = {
            "conflict_type": pattern.conflict_type,
            "entity_types": pattern.entity_types,
            "property_names": pattern.property_names or [],
            "resolution_strategy": pattern.resolution_strategy
        }
        
        pattern_key = self._generate_pattern_key(pattern_dict)
        return pattern_key in self.blacklist_cache
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows for another automatic resolution"""
        # Remove old timestamps
        current_time = datetime.now()
        one_minute_ago = current_time - timedelta(minutes=1)
        
        # Filter out timestamps older than 1 minute
        self.last_applied_resolutions = [
            ts for ts in self.last_applied_resolutions 
            if ts > one_minute_ago
        ]
        
        # Check if we're under the rate limit
        return len(self.last_applied_resolutions) < self.config.rate_limit_per_minute
    
    def _update_rate_limit(self) -> None:
        """Update rate limit tracking after applying a resolution"""
        self.last_applied_resolutions.append(datetime.now())
    
    async def calculate_confidence_score(
        self,
        conflict: Conflict,
        resolution_pattern: ResolutionPattern,
        similarity_score: float
    ) -> float:
        """
        Calculate confidence score for applying a learned resolution
        
        The confidence score is a weighted combination of:
        - Similarity score between the current conflict and the pattern
        - Success rate of the pattern in past applications
        - Recency of the pattern (newer patterns get higher scores)
        
        Args:
            conflict: The conflict to resolve
            resolution_pattern: The candidate resolution pattern
            similarity_score: The similarity score between the conflict and pattern
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get success rate for this pattern
        success_rate = await self._get_pattern_success_rate(resolution_pattern.id)
        
        # Calculate recency score (1.0 for brand new, decreasing with age)
        pattern_age_days = (datetime.now() - resolution_pattern.created_at).days
        max_age = self.config.max_resolution_age_days
        recency_score = max(0, (max_age - pattern_age_days) / max_age)
        
        # Calculate weighted score
        weights = self.config.weight_factors
        confidence = (
            similarity_score * weights["similarity_score"] +
            success_rate * weights["success_rate"] +
            recency_score * weights["recency"]
        )
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    async def _get_pattern_success_rate(self, pattern_id: str) -> float:
        """Get the success rate for a resolution pattern"""
        # In a real implementation, this would query a database of resolution applications
        # For now, we'll use a placeholder implementation
        
        # TODO: Implement actual success rate tracking
        # This would involve:
        # 1. Tracking each time a pattern is applied
        # 2. Recording whether the application was successful
        # 3. Calculating the success rate as successful_applications / total_applications
        
        # For now, return a default success rate of 0.8
        return 0.8
    
    async def find_learned_resolutions(
        self,
        conflict: Conflict,
        limit: int = 5
    ) -> List[Tuple[ResolutionPattern, float, float]]:
        """
        Find learned resolutions for a conflict
        
        Args:
            conflict: The conflict to find resolutions for
            limit: Maximum number of resolutions to return
            
        Returns:
            List of tuples containing (resolution_pattern, similarity_score, confidence_score)
        """
        start_time = time.time()
        
        try:
            # Find similar resolution patterns
            logger.info(f"Searching for similar resolutions for conflict {conflict.id} with limit {limit}")
            similar_patterns = await self.search_service.find_similar_resolutions(
                conflict=conflict,
                limit=limit
            )
            
            logger.info(f"Found {len(similar_patterns)} similar patterns before filtering")
            
            # Calculate confidence scores
            results = []
            for pattern, similarity_score in similar_patterns:
                logger.info(f"Processing pattern {pattern.id} with similarity {similarity_score}")
                
                # Skip blacklisted patterns
                if self._is_blacklisted(pattern):
                    logger.info(f"Skipping blacklisted pattern {pattern.id}")
                    continue
                
                # Calculate confidence score
                confidence_score = await self.calculate_confidence_score(
                    conflict=conflict,
                    resolution_pattern=pattern,
                    similarity_score=similarity_score
                )
                
                logger.info(f"Pattern {pattern.id} confidence score: {confidence_score}")
                
                results.append((pattern, similarity_score, confidence_score))
            
            # Sort by confidence score (highest first)
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Log performance
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.info(f"Found {len(results)} learned resolutions in {elapsed_time:.2f}ms")
            
            return results
        except Exception as e:
            logger.error(f"Error finding learned resolutions: {str(e)}")
            return []
    
    async def apply_learned_resolution(
        self,
        conflict: Conflict,
        pattern: ResolutionPattern
    ) -> Optional[ResolutionOption]:
        """
        Apply a learned resolution pattern to a conflict
        
        Args:
            conflict: The conflict to resolve
            pattern: The resolution pattern to apply
            
        Returns:
            ResolutionOption if successful, None otherwise
        """
        try:
            # Extract resolution strategy and data
            strategy = pattern.resolution_strategy
            resolution_data = pattern.resolution_data
            
            # Create a new resolution option
            resolution_id = f"learned_{str(uuid.uuid4())[:8]}"
            
            # Create resolution option based on strategy
            if strategy == "prefer_staging":
                return ResolutionOption(
                    id=resolution_id,
                    resolution_type="prefer_staging",
                    description="Automatically applied: Keep staging value",
                    confidence=pattern.confidence,
                    resolution_data={
                        "learned": True,
                        "pattern_id": pattern.id,
                        "original_conflict_id": pattern.original_conflict_id
                    }
                )
            elif strategy == "prefer_production":
                return ResolutionOption(
                    id=resolution_id,
                    resolution_type="prefer_production",
                    description="Automatically applied: Keep production value",
                    confidence=pattern.confidence,
                    resolution_data={
                        "learned": True,
                        "pattern_id": pattern.id,
                        "original_conflict_id": pattern.original_conflict_id
                    }
                )
            elif strategy == "custom_value":
                # Extract custom value from resolution data
                if "value" not in resolution_data:
                    logger.error(f"Missing 'value' in resolution data for pattern {pattern.id}")
                    return None
                
                # Create a combined resolution_data dictionary with both the custom value and metadata
                combined_data = {
                    "learned": True,
                    "pattern_id": pattern.id,
                    "original_conflict_id": pattern.original_conflict_id,
                    "value": resolution_data["value"]
                }
                
                return ResolutionOption(
                    id=resolution_id,
                    resolution_type="custom_value",
                    description="Automatically applied: Use custom value",
                    confidence=pattern.confidence,
                    resolution_data=combined_data
                )
            else:
                logger.warning(f"Unsupported resolution strategy: {strategy}")
                return None
        except Exception as e:
            logger.error(f"Error applying learned resolution: {str(e)}")
            return None
    
    async def process_conflict(
        self,
        conflict: Conflict
    ) -> Tuple[Optional[ResolutionOption], List[ResolutionOption], bool]:
        """
        Process a conflict using learned resolutions
        
        Args:
            conflict: The conflict to process
            
        Returns:
            Tuple containing:
            - Automatically applied resolution (if any)
            - List of suggested resolutions
            - Boolean indicating whether automatic resolution was applied
        """
        # Check rate limit first
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded for automatic resolutions")
            return None, [], False
        
        # Find learned resolutions
        learned_resolutions = await self.find_learned_resolutions(conflict)
        
        if not learned_resolutions:
            return None, [], False
        
        # Check for high confidence resolution
        auto_applied = None
        auto_applied_flag = False
        suggestions = []
        
        for pattern, similarity, confidence in learned_resolutions:
            # High confidence: automatic application
            if confidence >= self.config.high_confidence_threshold:
                # Apply the resolution
                resolution = await self.apply_learned_resolution(conflict, pattern)
                if resolution:
                    auto_applied = resolution
                    auto_applied_flag = True
                    self._update_rate_limit()
                    
                    # Log the automatic application
                    logger.info(
                        f"Automatically applied resolution for conflict {conflict.id} "
                        f"using pattern {pattern.id} with confidence {confidence:.2f}"
                    )
                    # Return early once we've found a high-confidence match
                    return auto_applied, suggestions, auto_applied_flag
            
            # Medium confidence: suggest resolution
            elif confidence >= self.config.medium_confidence_threshold:
                resolution = await self.apply_learned_resolution(conflict, pattern)
                if resolution:
                    # Modify description to indicate it's a suggestion
                    resolution.description = f"Suggested: {resolution.description}"
                    suggestions.append(resolution)
        
        return auto_applied, suggestions, auto_applied_flag
    
    async def track_resolution_outcome(
        self,
        conflict_id: str,
        resolution_id: str,
        pattern_id: str,
        success: bool,
        feedback: Optional[str] = None
    ) -> None:
        """
        Track the outcome of an applied resolution for continuous learning
        
        Args:
            conflict_id: ID of the resolved conflict
            resolution_id: ID of the applied resolution
            pattern_id: ID of the resolution pattern
            success: Whether the resolution was successful
            feedback: Optional feedback about the resolution
        """
        # TODO: Implement tracking of resolution outcomes
        # This would involve:
        # 1. Storing the outcome in a database
        # 2. Updating success rate statistics
        # 3. Potentially adjusting confidence calculations based on feedback
        
        logger.info(
            f"Tracked resolution outcome for conflict {conflict_id}, "
            f"resolution {resolution_id}, pattern {pattern_id}: "
            f"{'Success' if success else 'Failure'}"
        )
        
        if feedback:
            logger.info(f"Feedback: {feedback}")
    
    async def add_to_blacklist(self, pattern: Dict[str, Any]) -> None:
        """
        Add a pattern to the blacklist
        
        Args:
            pattern: Pattern to blacklist
        """
        # Add to in-memory cache
        pattern_key = self._generate_pattern_key(pattern)
        self.blacklist_cache.add(pattern_key)
        
        # Add to config
        self.config.blacklisted_patterns.append(pattern)
        
        logger.info(f"Added pattern to blacklist: {pattern_key}")
    
    async def remove_from_blacklist(self, pattern: Dict[str, Any]) -> bool:
        """
        Remove a pattern from the blacklist
        
        Args:
            pattern: Pattern to remove from blacklist
            
        Returns:
            True if pattern was removed, False if not found
        """
        pattern_key = self._generate_pattern_key(pattern)
        
        # Remove from in-memory cache
        if pattern_key in self.blacklist_cache:
            self.blacklist_cache.remove(pattern_key)
        else:
            return False
        
        # Remove from config
        for i, blacklisted in enumerate(self.config.blacklisted_patterns):
            if self._generate_pattern_key(blacklisted) == pattern_key:
                self.config.blacklisted_patterns.pop(i)
                logger.info(f"Removed pattern from blacklist: {pattern_key}")
                return True
        
        return False
    
    async def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the service configuration
        
        Args:
            config: New configuration values
        """
        # Update config
        self.config = ResolutionLearningConfig(**{
            **self.config.model_dump(),
            **config
        })
        
        # Reload blacklist
        self.blacklist_cache.clear()
        self._load_blacklist()
        
        logger.info("Updated resolution learning configuration") 