"""Base classes for conflict detection"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from app.schemas.conflicts import Conflict
from app.schemas.graph import Node, Edge
from app.services.storage.interface import GraphStorageInterface

class BaseConflictDetector(ABC):
    """Base class for all conflict detectors"""
    
    def __init__(self, storage: GraphStorageInterface):
        self.storage = storage
        
    @abstractmethod
    async def detect_conflicts(self, **kwargs) -> List[Conflict]:
        """Detect conflicts in the given context"""
        pass
        
    def _are_property_values_equal(self, value1: Any, value2: Any) -> bool:
        """Compare property values with type-specific logic"""
        if value1 is None and value2 is None:
            return True
            
        if value1 is None or value2 is None:
            return False
            
        # String comparison (case-insensitive)
        if isinstance(value1, str) and isinstance(value2, str):
            return value1.lower() == value2.lower()
            
        # Numeric comparison with tolerance
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            if isinstance(value1, int) and isinstance(value2, int):
                return value1 == value2
            else:
                return abs(float(value1) - float(value2)) < 1e-6
                
        # List comparison (order-insensitive)
        if isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
                
            # For primitive values, compare as sets
            if all(isinstance(v, (str, int, float, bool, type(None))) 
                  for v in value1 + value2):
                return set(str(x) if x is not None else 'None' for x in value1) == \
                       set(str(x) if x is not None else 'None' for x in value2)
                       
            # For complex items, try to match each item
            matched_indices = set()
            for i, item1 in enumerate(value1):
                for j, item2 in enumerate(value2):
                    if j not in matched_indices and self._are_property_values_equal(item1, item2):
                        matched_indices.add(j)
                        break
                else:
                    return False
            return len(matched_indices) == len(value2)
            
        # Dict comparison
        if isinstance(value1, dict) and isinstance(value2, dict):
            if set(value1.keys()) != set(value2.keys()):
                return False
            return all(
                self._are_property_values_equal(value1[k], value2[k])
                for k in value1.keys()
            )
            
        # Default comparison
        return value1 == value2

class BatchConflictDetector(BaseConflictDetector):
    """Base class for detectors that process items in batches"""
    
    def __init__(
        self,
        storage: GraphStorageInterface,
        batch_size: int = 100,
        max_workers: int = 10
    ):
        super().__init__(storage)
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_batch(
        self,
        items: List[Any],
        processor: Any,
        *args
    ) -> List[Conflict]:
        """Process a batch of items in parallel
        
        Args:
            items: List of items to process
            processor: Async function to process each item
            args: Additional arguments to pass to processor
            
        Returns:
            List of conflicts from processing all items
        """
        all_conflicts = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch in parallel
            batch_conflicts = await asyncio.gather(
                *[
                    processor(item, *args)
                    for item in batch
                ]
            )
            
            # Flatten conflicts
            all_conflicts.extend([
                conflict
                for item_conflicts in batch_conflicts
                for conflict in item_conflicts
            ])
            
        return all_conflicts
