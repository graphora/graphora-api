"""Interface for conflict storage operations"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from app.schemas.conflicts import Conflict

class ConflictStorageInterface(ABC):
    """Interface for conflict storage operations"""
    
    @abstractmethod
    async def store_conflict(self, conflict: Conflict) -> bool:
        """
        Store a conflict in the database
        
        Args:
            conflict: The conflict to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_conflict(self, conflict_id: str) -> Optional[Conflict]:
        """
        Retrieve a conflict by ID
        
        Args:
            conflict_id: The ID of the conflict to retrieve
            
        Returns:
            Optional[Conflict]: The conflict if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_conflicts(
        self,
        merge_id: str,
        conflict_type: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
        entity_type: Optional[str] = None
    ) -> Tuple[List[Conflict], int]:
        """
        Retrieve conflicts with optional filtering
        
        Args:
            merge_id: The ID of the merge process
            conflict_type: Optional filter by conflict type
            severity: Optional filter by severity
            resolved: Optional filter by resolution status
            limit: Maximum number of conflicts to return
            offset: Pagination offset
            entity_type: Optional filter by entity type
            
        Returns:
            Tuple[List[Conflict], int]: List of conflicts and total count
        """
        pass
    
    @abstractmethod
    async def update_conflict(self, conflict: Conflict) -> bool:
        """
        Update an existing conflict
        
        Args:
            conflict: The conflict with updated fields
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_conflict(self, conflict_id: str) -> bool:
        """
        Delete a conflict
        
        Args:
            conflict_id: The ID of the conflict to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_conflict_summary(self, merge_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for conflicts in a merge process
        
        Args:
            merge_id: The ID of the merge process
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        pass
    
    @abstractmethod
    async def get_conflicts_by_entity(self, merge_id: str, entity_id: str) -> List[Conflict]:
        """
        Get all conflicts related to a specific entity
        
        Args:
            merge_id: The ID of the merge process
            entity_id: The ID of the entity
            
        Returns:
            List[Conflict]: List of conflicts related to the entity
        """
        pass 