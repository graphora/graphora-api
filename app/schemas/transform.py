from pydantic import BaseModel, Field
from typing import Optional, Literal, Any, Dict, List

class UploadResponse(BaseModel):
    id: str
    status: Literal['success', 'error']
    message: Optional[str] = None

class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass

class Metadata(BaseModel):
  section: str = Field(
    ..., description="Section Id associated with the text chunk"
  )
  subsections: List[str] = Field(
    ..., description="Sub Section Id associated with the text chunk"
  )
  properties: Dict[str, Any] = Field(
    {}, description="Attributes for the metadata"
  )


class Node(BaseModel):
  id: str = Field(..., description="Unique identifier for the node")
  type_: Optional[str] = Field('Node', alias="type")
  properties: Dict[str, Any] = Field(
    {}, description="Additional attributes for the node"
  )


class Edge(BaseModel):
  from_: str = Field(..., alias="from", description="Origin node ID")
  to: str = Field(..., description="Destination node ID")
  relationship: str = Field(..., description="Name of relationship between the nodes, in SNAKE_CASE")
  properties: Dict[str, Any] = Field(
    {}, description="Additional attributes for the edge"
  )


class KnowledgeGraph(BaseModel):
  """Generate a knowledge graph with entities and relationships.
  """

  metadata: Metadata = Field(..., description="Metadata for the knowledge graph")
  nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
  edges: List[Edge] = Field(..., description="List of edges in the knowledge graph")
