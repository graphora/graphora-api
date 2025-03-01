"""Abstract interface for vector storage operations"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import uuid

from pydantic import BaseModel, Field


class ResolutionPattern(BaseModel):
    """Data model for resolution patterns"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: str
    resolution_strategy: str
    context_features: Dict[str, Any] = Field(default_factory=dict)
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class VectorStorageInterface(ABC):
    """Abstract interface for vector storage operations"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage (create collections, etc.)"""
        pass
    
    @abstractmethod
    async def store_pattern(self, pattern: ResolutionPattern) -> str:
        """Store a resolution pattern, return ID"""
        pass
    
    @abstractmethod
    async def get_pattern(self, pattern_id: str) -> Optional[ResolutionPattern]:
        """Retrieve a specific pattern by ID"""
        pass
    
    @abstractmethod
    async def update_pattern(self, pattern: ResolutionPattern) -> bool:
        """Update an existing pattern, return success status"""
        pass
    
    @abstractmethod
    async def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern, return success status"""
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        vector: List[float], 
        conflict_type: Optional[str] = None,
        limit: int = 10, 
        threshold: float = 0.7
    ) -> List[ResolutionPattern]:
        """Search for similar patterns"""
        pass
    
    @abstractmethod
    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 10
    ) -> List[ResolutionPattern]:
        """Search patterns using metadata filters"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up resources"""
        pass 