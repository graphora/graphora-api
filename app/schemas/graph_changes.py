from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from uuid import UUID

class NodeCreation(BaseModel):
    """Model for creating new nodes"""
    id: str = Field(..., description="Unique identifier for the node")
    type: str = Field(..., description="Type/class of the node")
    label: str = Field(..., description="Label for the node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")

class NodeUpdate(BaseModel):
    """Model for updating existing nodes"""
    id: str = Field(..., description="ID of node to update")
    properties: Dict[str, Any] = Field(..., description="Updated properties")

class EdgeCreation(BaseModel):
    """Model for creating new edges"""
    id: str = Field(..., description="Unique identifier for the edge")
    source: str = Field(..., description="ID of source node")
    target: str = Field(..., description="ID of target node")
    type: str = Field(..., description="Type of relationship")
    label: str = Field(..., description="Label for the relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge properties")

class EdgeUpdate(BaseModel):
    """Model for updating existing edges"""
    id: str = Field(..., description="ID of edge to update")
    properties: Dict[str, Any] = Field(..., description="Updated properties")

class NodeChanges(BaseModel):
    """Collection of node modifications"""
    created: List[NodeCreation] = Field(default_factory=list)
    updated: List[NodeUpdate] = Field(default_factory=list)
    deleted: List[str] = Field(default_factory=list)

class EdgeChanges(BaseModel):
    """Collection of edge modifications"""
    created: List[EdgeCreation] = Field(default_factory=list)
    updated: List[EdgeUpdate] = Field(default_factory=list)
    deleted: List[str] = Field(default_factory=list)

class SaveGraphRequest(BaseModel):
    """Request model for saving graph changes"""
    nodes: Optional[NodeChanges] = Field(None, description="Node modifications")
    edges: Optional[EdgeChanges] = Field(None, description="Edge modifications")

class Message(BaseModel):
    """Message for warnings or info"""
    type: Literal['warning', 'info']
    message: str

class SaveGraphResponse(BaseModel):
    """Response model for graph save operation"""
    data: Dict[str, List[Dict[str, Any]]] = Field(..., description="Updated graph data")
    messages: Optional[List[Message]] = Field(None, description="Warning or info messages")
