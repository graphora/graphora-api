"""Models for graph data structures"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import pytz

class Node(BaseModel):
    """Model representing a node in a graph"""
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Primary label for the node")
    type: str = Field(..., description="Type of the node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties of the node")
    labels: Optional[List[str]] = Field(default=None, description="Additional labels for the node")
    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="When the node was created")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="When the node was last updated")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True

class Edge(BaseModel):
    """Model representing an edge/relationship in a graph"""
    id: str = Field(..., description="Unique identifier for the relationship")
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    type: str = Field(..., description="Type of the relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties of the relationship")
    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="When the relationship was created")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="When the relationship was last updated")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True

class GraphResponse(BaseModel):
    """Model representing a graph response"""
    nodes: List[Node] = Field(default_factory=list, description="Nodes in the graph")
    edges: List[Edge] = Field(default_factory=list, description="Relationships in the graph")
    total_nodes: Optional[int] = Field(default=0, description="Total number of nodes with this label")
    total_edges: Optional[int] = Field(default=0, description="Total number of relationships for these nodes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the graph")
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_relationship_by_id(self, rel_id: str) -> Optional[Edge]:
        """Get a relationship by its ID"""
        for rel in self.edges:
            if rel.id == rel_id:
                return rel
        return None
