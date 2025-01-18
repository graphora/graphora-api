from pydantic import BaseModel, Field
from typing import Optional

class OntologyRequest(BaseModel):
    text: str = Field(
        ..., 
        description="YAML string containing ontology definition"
    )

class OntologyResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    uuid: Optional[str] = None
