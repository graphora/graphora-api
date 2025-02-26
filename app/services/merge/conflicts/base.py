"""Base classes for conflict detection"""
from abc import ABC, abstractmethod
from typing import List, Any

from app.schemas.conflicts import Conflict
from app.schemas.graph import GraphResponse
from app.services.storage.interface import GraphStorageInterface

class ConflictDetector(ABC):
    """Base class for conflict detectors"""
    
    def __init__(self, storage: GraphStorageInterface):
        self.storage = storage
    
    @abstractmethod
    async def detect_conflicts(
        self,
        staging_graph: GraphResponse,
        **kwargs
    ) -> List[Conflict]:
        """Detect conflicts in the staging graph"""
        pass

class ConflictCreator(ABC):
    """Base class for conflict creators"""
    
    @abstractmethod
    def create_conflict(
        self,
        conflict_id: str,
        **kwargs
    ) -> Conflict:
        """Create a conflict with the given parameters"""
        pass

class ConflictAnalyzer(ABC):
    """Base class for conflict analyzers"""
    
    @abstractmethod
    async def analyze(self, **kwargs) -> Any:
        """Analyze conflict data and return analysis results"""
        pass
