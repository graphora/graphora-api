from pydantic import BaseModel
from typing import Optional, Dict, Any

class SchemaGeneratorInput(BaseModel):
    text: str
    base_schema_name: Optional[str] = "GeneratedSchema"
    description: Optional[str] = None

class SchemaGeneratorResponse(BaseModel):
    models: Dict[str, Any]
    message: Optional[str] = None
