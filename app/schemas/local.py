from typing import Any, Dict, List
from pydantic import BaseModel, Field

class LocalMetadata(BaseModel):
  section: str = Field(
    ..., description="Section Id associated with the text chunk"
  )
  subsections: List[str] = Field(
    ..., description="Sub Section Id associated with the text chunk"
  )

  class Config:
    populate_by_name = True


class LocalNode(BaseModel):
  id: str = Field(..., description="Unique identifier for the node")
  properties: Dict[str, Any] = Field(
    {}, description="Additional attributes for the node"
  )
  type_: str = Field("", alias="type", description="Type of the node, in CamelCase")
  metadata: Dict[str, Any] = Field(
    {}, description="Metadata for the node"
  )

  class Config:
    populate_by_name = True


class LocalEdge(BaseModel):
  from_: str = Field(..., alias="from", description="Origin node ID")
  to: str = Field(..., description="Destination node ID")
  relationship: str = Field(..., description="Name of relationship between the nodes, in SNAKE_CASE")
  properties: Dict[str, Any] = Field(
    {}, description="Additional attributes for the edge"
  )
  metadata: Dict[str, Any] = Field(
    {}, description="Metadata for the node"
  )

  class Config:
    populate_by_name = True


class LocalGraph(BaseModel):
  """Generate a knowledge graph with entities and relationships.
  """

  metadata: LocalMetadata = Field(..., description="Metadata for the knowledge graph")
  nodes: List[LocalNode] = Field(..., description="List of nodes in the knowledge graph")
  edges: List[LocalEdge] = Field(..., description="List of edges in the knowledge graph")

  class Config:
    populate_by_name = True