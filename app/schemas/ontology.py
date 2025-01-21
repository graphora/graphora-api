from pydantic import BaseModel, Field
from typing import Optional

class OntologyRequest(BaseModel):
    """Request model for ontology validation"""
    text: str = Field(..., description="Ontology definition in YAML format")

class OntologyResponse(BaseModel):
    """Response model for ontology validation"""
    id: str = Field(..., description="Unique ID for the validated ontology")
