"""Interface for graph storage operations"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.schemas.graph import Node, Edge

class GraphStorageInterface(ABC):
    """Interface for graph storage operations"""
    
    @abstractmethod
    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """
        Retrieve a node by ID
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            Optional[Node]: The node if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_relationship_by_id(self, rel_id: str) -> Optional[Edge]:
        """
        Retrieve a relationship by ID
        
        Args:
            rel_id: The ID of the relationship to retrieve
            
        Returns:
            Optional[Edge]: The relationship if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def update_node_property(self, node_id: str, prop_name: str, value: Any) -> bool:
        """
        Update a property of a node
        
        Args:
            node_id: The ID of the node to update
            prop_name: The name of the property to update
            value: The new value for the property
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def remove_node_property(self, node_id: str, prop_name: str) -> bool:
        """
        Remove a property from a node
        
        Args:
            node_id: The ID of the node
            prop_name: The name of the property to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> Optional[Node]:
        """
        Update multiple properties of a node
        
        Args:
            node_id: The ID of the node to update
            properties: Dictionary of property names and values
            
        Returns:
            Optional[Node]: The updated node if successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def update_relationship_type(self, rel_id: str, new_type: str) -> bool:
        """
        Update the type of a relationship
        
        Args:
            rel_id: The ID of the relationship
            new_type: The new type for the relationship
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        rel_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[Edge]:
        """
        Create a new relationship between nodes
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            rel_type: The type of the relationship
            properties: Optional properties for the relationship
            
        Returns:
            Optional[Edge]: The created relationship if successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete_relationship(self, rel_id: str) -> bool:
        """
        Delete a relationship
        
        Args:
            rel_id: The ID of the relationship to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node
        
        Args:
            node_id: The ID of the node to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_relationships_between(self, source_id: str, target_id: str) -> List[Edge]:
        """
        Get all relationships between two nodes
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            
        Returns:
            List[Edge]: List of relationships between the nodes
        """
        pass
    
    @abstractmethod
    async def get_incoming_relationships(self, node_id: str) -> List[Edge]:
        """
        Get all incoming relationships for a node
        
        Args:
            node_id: The ID of the node
            
        Returns:
            List[Edge]: List of incoming relationships
        """
        pass
    
    @abstractmethod
    async def get_outgoing_relationships(self, node_id: str) -> List[Edge]:
        """
        Get all outgoing relationships for a node
        
        Args:
            node_id: The ID of the node
            
        Returns:
            List[Edge]: List of outgoing relationships
        """
        pass 