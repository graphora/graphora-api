from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Node(BaseModel):
    """Node in the knowledge graph"""
    id: str = Field(..., description="Unique identifier of the node")
    label: str = Field(..., description="Label/type of the node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    type: str = Field(..., description="Type of the node")

class Edge(BaseModel):
    """Edge/relationship in the knowledge graph"""
    id: str = Field(..., description="Unique identifier of the relationship")
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    type: str = Field(..., description="Type of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")

class GraphResponse(BaseModel):
    """Response model for graph data"""
    nodes: List[Node] = Field(default_factory=list, description="List of nodes in the graph")
    edges: List[Edge] = Field(default_factory=list, description="List of edges/relationships in the graph")
    total_nodes: int = Field(..., description="Total number of nodes with this label")
    total_edges: int = Field(..., description="Total number of relationships for these nodes")
