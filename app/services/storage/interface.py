from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from app.services.storage.models import (
    StorageBatchResult,
    StorageCheckpoint,
    StorageStage,
    TransformationResult,
    Node,
    Edge
)

class GraphStorageInterface(ABC):
    """Abstract interface for graph storage"""
    
    @abstractmethod
    async def store_nodes(
        self,
        nodes: List[Dict],
        batch_index: int,
        transform_id: str,
        merge: bool = True
    ) -> StorageBatchResult:
        """Store nodes in batch"""
        pass
    
    @abstractmethod
    async def store_relationships(
        self,
        relationships: List[Dict],
        batch_index: int,
        transform_id: str,
        merge: bool = True
    ) -> StorageBatchResult:
        """Store relationships in batch"""
        pass
    
    @abstractmethod
    async def get_storage_status(
        self,
        transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status"""
        pass
    
    @abstractmethod
    async def update_checkpoint(
        self,
        transform_id: str,
        last_index: int,
        stage: StorageStage
    ) -> None:
        """Update storage checkpoint"""
        pass
    
    @abstractmethod
    async def get_transformation_data(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships for a transformation"""
        pass
    
    @abstractmethod
    async def get_production_graph_for_transform(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships from production that were affected by a transform
        
        Args:
            transform_id: ID of the transformation
            
        Returns:
            TransformationResult containing nodes and relationships from production
        """
        pass
    
    @abstractmethod
    async def get_nodes_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Node]:
        """Get all nodes with the specified property value"""
        pass
    
    @abstractmethod
    async def get_relationships_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Edge]:
        """Get all relationships between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Optional type to filter relationships
            
        Returns:
            List of edges between the nodes. Empty list if none exist.
        """
        pass
        
    @abstractmethod
    async def get_relationships_between_nodes(
        self,
        node_ids: List[str]
    ) -> List[Edge]:
        """Get all relationships between a set of nodes
        
        Args:
            node_ids: List of node IDs to find relationships between
            
        Returns:
            List of edges between any of the nodes
        """
        pass
        
    @abstractmethod
    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True
    ) -> List[Node]:
        """
        Find nodes with matching property value.
        
        Args:
            label: Node label to filter by
            property_name: Name of the property to match
            property_value: Value to match against
            exact_match: If True, requires exact value match. If False, allows partial matches
            
        Returns:
            List of matching nodes
        """
        pass
        
    @abstractmethod
    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True
    ) -> List[Node]:
        """
        Find nodes with similar properties using fuzzy matching.
        
        Args:
            label: Node label to filter by
            properties: Properties to compare for similarity
            similarity_threshold: Minimum similarity score (0-1) to include in results
            max_results: Maximum number of similar nodes to return
            include_relationships: Whether to include relationship patterns in similarity calculation
            
        Returns:
            List of similar nodes sorted by similarity score (highest first)
        """
        pass
        
    @abstractmethod
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Create a new node"""
        pass
        
    @abstractmethod
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Update an existing node"""
        pass
        
    @abstractmethod
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> Edge:
        """Create a relationship between nodes"""
        pass
        
    @abstractmethod
    async def update_relationship(
        self,
        rel_id: str,
        properties: Dict[str, Any]
    ) -> Edge:
        """Update an existing relationship"""
        pass
        
    @abstractmethod
    async def get_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str
    ) -> Optional[Edge]:
        """Get a specific relationship between two nodes by type
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            rel_type: Type of relationship to find
            
        Returns:
            Edge if found, None otherwise
        """
        pass
        
    @abstractmethod
    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node if found, None otherwise
        """
        pass
        
    @abstractmethod
    async def get_edges_between(
        self,
        source_id: str,
        target_id: str
    ) -> List[Edge]:
        """Get all edges between two nodes
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            List of edges between the nodes, empty list if none found
        """
        pass
