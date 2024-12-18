from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Entity(BaseModel):
    id: str = Field(description="Unique identifier for the entity")
    type: str = Field(description="Type of the entity")
    value: str = Field(description="Value or name of the entity")
    confidence: float = Field(description="Confidence score of the extraction")

class Relationship(BaseModel):
    source_id: str = Field(description="ID of the source entity")
    target_id: str = Field(description="ID of the target entity")
    type: str = Field(description="Type of relationship")
    confidence: float = Field(description="Confidence score of the relationship")

class DocumentOutput(BaseModel):
    id: str = Field(description="Document unique identifier")
    content: str = Field(description="Document content")
    entities: List[Entity] = Field(description="List of extracted entities")
    relationships: List[Relationship] = Field(description="List of extracted relationships")
    created_at: datetime = Field(default_factory=datetime.now, description="Document creation timestamp")

class MetadataInput(BaseModel):
    source: Optional[str] = Field(default=None, description="Source of the document")
    tags: List[str] = Field(default_factory=list, description="Document tags")

class DocumentInput(BaseModel):
    content: str = Field(description="Document content to process")
    metadata: Optional[MetadataInput] = Field(default=None, description="Additional document metadata")

class DocumentResponse(BaseModel):
    id: str
    content: str
    entities: List[dict]
    relationships: List[dict]
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "Sample document content",
                "entities": [{"id": "1", "type": "PERSON", "value": "John Doe", "confidence": 0.95}],
                "relationships": [{"source_id": "1", "target_id": "2", "type": "WORKS_FOR", "confidence": 0.8}]
            }
        }