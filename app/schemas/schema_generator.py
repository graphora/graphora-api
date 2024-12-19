from pydantic import BaseModel
from typing import Optional
from app.services.ontology_generator_service import Neo4jOntology

class SchemaGeneratorInput(BaseModel):
    text: str
    base_schema_name: Optional[str] = "GeneratedSchema"
    description: Optional[str] = None

class SchemaGeneratorResponse(BaseModel):
    ontology: Neo4jOntology
    session_id: str
