from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime


class ExtractedEntity(BaseModel):
    """Entity extracted from text following ontology definition"""
    id: str
    type: str 
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = {}

class ExtractedRelationship(BaseModel):
    """Relationship between entities following ontology definition"""
    source_id: str
    target_id: str
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = {}

class ChunkExtraction(BaseModel):
    """Complete extraction results from a text chunk"""
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    chunk_id: str

class EntityExtractionResponse(BaseModel):
    entities: List[ExtractedEntity] = Field(description="List of extracted entities")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)

class RelationshipExtractionResponse(BaseModel):
    relationships: List[ExtractedRelationship] = Field(description="List of extracted relationships")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
