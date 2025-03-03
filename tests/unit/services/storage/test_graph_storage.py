"""Mock implementation of GraphStorageInterface for testing"""
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from app.services.storage.interface import GraphStorageInterface
from app.services.storage.models import (
    StorageBatchResult,
    StorageCheckpoint,
    StorageStage,
    TransformationResult,
    Node,
    Edge
)

class MockGraphStorage(GraphStorageInterface):
    """Mock implementation of GraphStorageInterface for testing"""
    
    def __init__(self):
        """Initialize mock storage with empty data structures"""
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.checkpoints: Dict[str, StorageCheckpoint] = {}
    
    def add_test_node(self, node: Node):
        """Add a test node to the mock storage"""
        self.nodes[node.id] = node
    
    def add_test_relationship(self, edge: Edge):
        """Add a test relationship to the mock storage"""
        self.edges[edge.id] = edge
    
    async def store_nodes(
        self,
        nodes: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store nodes in batch"""
        try:
            for node_data in nodes:
                node = Node(**node_data)
                self.nodes[node.id] = node
            
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=len(nodes),
                processing_time_ms=0.1,
                success=True
            )
        except Exception as e:
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=0,
                processing_time_ms=0.1,
                success=False,
                error=str(e)
            )
    
    async def store_relationships(
        self,
        relationships: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store relationships in batch"""
        try:
            for rel_data in relationships:
                edge = Edge(**rel_data)
                self.edges[edge.id] = edge
            
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=len(relationships),
                processing_time_ms=0.1,
                success=True
            )
        except Exception as e:
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=0,
                processing_time_ms=0.1,
                success=False,
                error=str(e)
            )
    
    async def get_storage_status(
        self,
        transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status"""
        return self.checkpoints.get(transform_id)
    
    async def update_checkpoint(
        self,
        transform_id: str,
        last_index: int,
        stage: StorageStage
    ) -> None:
        """Update storage checkpoint"""
        self.checkpoints[transform_id] = StorageCheckpoint(
            transform_id=transform_id,
            last_processed_index=last_index,
            stage=stage,
            timestamp=datetime.now()
        )
    
    async def get_transformation_data(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships for a transformation"""
        # Filter nodes and relationships by transform_id
        nodes = [
            node.model_dump()
            for node in self.nodes.values()
            if node.properties.get("transform_id") == transform_id
        ]
        
        relationships = [
            edge.model_dump()
            for edge in self.edges.values()
            if edge.properties.get("transform_id") == transform_id
        ]
        
        return TransformationResult(
            transform_id=transform_id,
            nodes=nodes,
            relationships=relationships,
            timestamp=datetime.now()
        )
    
    async def get_production_graph_for_transform(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships from production that were affected by a transform"""
        # For testing, return all nodes and relationships
        return TransformationResult(
            transform_id=transform_id,
            nodes=list(self.nodes.values()),
            relationships=list(self.edges.values()),
            timestamp=datetime.now()
        )
    
    async def get_nodes_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Node]:
        """Get all nodes with the specified property value"""
        return [
            node for node in self.nodes.values()
            if node.properties.get(property_name) == property_value
        ]
    
    async def get_relationships_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Edge]:
        """Get all relationships between two nodes"""
        edges = [
            edge for edge in self.edges.values()
            if edge.source == source_id and edge.target == target_id
        ]
        
        if relationship_type:
            edges = [edge for edge in edges if edge.type == relationship_type]
            
        return edges
    
    async def get_relationships_between_nodes(
        self,
        node_ids: List[str]
    ) -> List[Edge]:
        """Get all relationships between a set of nodes"""
        return [
            edge for edge in self.edges.values()
            if edge.source in node_ids and edge.target in node_ids
        ]
    
    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True
    ) -> List[Node]:
        """Find nodes with matching property value"""
        nodes = []
        for node in self.nodes.values():
            if node.label != label:
                continue
                
            value = node.properties.get(property_name)
            if exact_match:
                if value == property_value:
                    nodes.append(node)
            else:
                # Simple partial matching for strings
                if isinstance(value, str) and isinstance(property_value, str):
                    if property_value.lower() in value.lower():
                        nodes.append(node)
                        
        return nodes
    
    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True
    ) -> List[Node]:
        """Find nodes with similar properties"""
        # Simple implementation - match if any property matches
        nodes = []
        for node in self.nodes.values():
            if node.label != label:
                continue
                
            matches = 0
            total = len(properties)
            
            for prop_name, prop_value in properties.items():
                if node.properties.get(prop_name) == prop_value:
                    matches += 1
            
            similarity = matches / total if total > 0 else 0
            if similarity >= similarity_threshold:
                nodes.append(node)
                
            if len(nodes) >= max_results:
                break
                
        return nodes
    
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Create a new node"""
        node = Node(
            id=f"n{len(self.nodes)}",
            label=label,
            type=label,
            properties=properties
        )
        self.nodes[node.id] = node
        return node
    
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Update an existing node"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
            
        node = self.nodes[node_id]
        node.properties.update(properties)
        return node
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> Edge:
        """Create a relationship between nodes"""
        if source_id not in self.nodes:
            raise ValueError(f"Source node {source_id} not found")
        if target_id not in self.nodes:
            raise ValueError(f"Target node {target_id} not found")
            
        edge = Edge(
            id=f"e{len(self.edges)}",
            source=source_id,
            target=target_id,
            type=rel_type,
            properties=properties or {}
        )
        self.edges[edge.id] = edge
        return edge
    
    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID"""
        return self.nodes.get(node_id)
    
    async def get_edges_between(
        self,
        source_id: str,
        target_id: str
    ) -> List[Edge]:
        """Get all edges between two nodes"""
        return [
            edge for edge in self.edges.values()
            if edge.source == source_id and edge.target == target_id
        ]
    
    async def get_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str
    ) -> Optional[Edge]:
        """Get a specific relationship between two nodes by type"""
        edges = [
            edge for edge in self.edges.values()
            if edge.source == source_id and edge.target == target_id and edge.type == rel_type
        ]
        return edges[0] if edges else None
    
    async def update_relationship(
        self,
        rel_id: str,
        properties: Dict[str, Any]
    ) -> Edge:
        """Update an existing relationship"""
        if rel_id not in self.edges:
            raise ValueError(f"Relationship {rel_id} not found")
            
        edge = self.edges[rel_id]
        edge.properties.update(properties)
        return edge 