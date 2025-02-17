from pydantic import BaseModel, Field
from typing import Optional, Literal, Any, Dict, List
from enum import Enum
from datetime import datetime

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


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk"""
    source: str
    content: str
    chunk_size: int
    chunk_overlap: int
    properties: Dict[str, Any] = Field(default_factory=dict)


class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MD = "md"

class ProcessingPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class TransformStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentMetadata(BaseModel):
    source: str
    document_type: DocumentType
    tags: Optional[List[str]] = []
    priority: Optional[ProcessingPriority] = ProcessingPriority.NORMAL

class DocumentInfo(BaseModel):
    filename: str
    size: int
    document_type: DocumentType
    metadata: DocumentMetadata

class TransformInitResponse(BaseModel):
    id: str  # Prefect flow_id
    upload_timestamp: datetime
    status: TransformStatus
    document_info: DocumentInfo

class ValidationResult(BaseModel):
    is_valid: bool
    errors: Optional[List[str]] = None

class StorageLocation(BaseModel):
    transform_id: str
    original_path: str
    processed_path: Optional[str] = None
    metadata_path: str
