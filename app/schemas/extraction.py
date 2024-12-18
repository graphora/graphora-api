from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ExtractedEntity(BaseModel):
    id: str = Field(description="Unique identifier for the entity")
    type: str = Field(description="Type of entity (e.g., PERSON, ORGANIZATION, LOCATION)")
    value: str = Field(description="The actual text value of the entity")
    confidence: float = Field(description="Confidence score of the extraction", ge=0.0, le=1.0)

class ExtractedRelationship(BaseModel):
    source_id: str = Field(description="ID of the source entity")
    target_id: str = Field(description="ID of the target entity")
    type: str = Field(description="Type of relationship between entities")
    confidence: float = Field(description="Confidence score of the relationship", ge=0.0, le=1.0)

class EntityExtractionResponse(BaseModel):
    entities: List[ExtractedEntity] = Field(description="List of extracted entities")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)

class RelationshipExtractionResponse(BaseModel):
    relationships: List[ExtractedRelationship] = Field(description="List of extracted relationships")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
