from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.schemas.conflicts import ConflictType, ConflictSeverity

class ResolutionHistoryEntry(BaseModel):
    """Record of a conflict resolution for learning"""
    id: str = Field(..., description="Unique identifier for this resolution history entry")
    conflict_id: str = Field(..., description="ID of the original conflict")
    merge_id: str = Field(..., description="ID of the merge process")
    conflict_type: ConflictType = Field(..., description="Type of conflict resolved")
    severity: ConflictSeverity = Field(..., description="Severity of conflict")
    context: Dict[str, Any] = Field(..., description="Context of the original conflict")
    resolution_id: str = Field(..., description="ID of the chosen resolution")
    resolution_type: str = Field(..., description="Type of resolution applied")
    resolution_data: Dict[str, Any] = Field(default_factory=dict, description="Data of the applied resolution")
    entity_types: List[str] = Field(..., description="Types of entities involved")
    property_names: List[str] = Field(default_factory=list, description="Names of properties involved, if applicable")
    relationship_types: List[str] = Field(default_factory=list, description="Types of relationships involved, if applicable")
    applied_by: str = Field(..., description="User or system that applied the resolution")
    applied_at: datetime = Field(default_factory=datetime.now, description="When the resolution was applied")
    success: bool = Field(True, description="Whether the resolution was successful")
    feedback: Optional[str] = Field(None, description="Optional user feedback on resolution quality")
    tags: List[str] = Field(default_factory=list, description="Additional tags for classification")
    vector_embedding: Optional[List[float]] = Field(None, description="Vector embedding for similarity search") 