from pydantic import BaseModel

class SchemaGeneratorInput(BaseModel):
    text: str

class SchemaGeneratorResponse(BaseModel):
    session_id: str
