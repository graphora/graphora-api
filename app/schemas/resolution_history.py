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
    effectiveness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Effectiveness rating (0.0-1.0)")

class ResolutionFilter(BaseModel):
    """Filter parameters for resolution queries"""
    conflict_type: Optional[ConflictType] = Field(None, description="Filter by conflict type")
    resolution_type: Optional[str] = Field(None, description="Filter by resolution strategy used")
    start_date: Optional[datetime] = Field(None, description="Filter by start date (inclusive)")
    end_date: Optional[datetime] = Field(None, description="Filter by end date (inclusive)")
    user: Optional[str] = Field(None, description="Filter by user who applied resolution")
    effectiveness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Filter by effectiveness rating")
    entity_type: Optional[str] = Field(None, description="Filter by entity type")
    property_name: Optional[str] = Field(None, description="Filter by property name")
    relationship_type: Optional[str] = Field(None, description="Filter by relationship type")
    success: Optional[bool] = Field(None, description="Filter by success status")

class PaginationParams(BaseModel):
    """Pagination parameters"""
    limit: int = Field(10, ge=1, le=100, description="Maximum number of items to return")
    offset: int = Field(0, ge=0, description="Number of items to skip")
    sort_by: str = Field("applied_at", description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order (asc or desc)")

class ResolutionResponse(BaseModel):
    """Response model for resolution queries"""
    items: List[ResolutionHistoryEntry] = Field(..., description="List of resolution history entries")
    total: int = Field(..., description="Total number of items matching the query")
    limit: int = Field(..., description="Maximum number of items returned")
    offset: int = Field(..., description="Number of items skipped")

class ResolutionStats(BaseModel):
    """Statistics about resolutions"""
    total_resolutions: int = Field(..., description="Total number of resolutions")
    by_conflict_type: Dict[str, int] = Field(..., description="Resolutions by conflict type")
    by_resolution_type: Dict[str, int] = Field(..., description="Resolutions by resolution type")
    by_entity_type: Dict[str, int] = Field(..., description="Resolutions by entity type")
    by_user: Dict[str, int] = Field(..., description="Resolutions by user")
    success_rate: float = Field(..., description="Overall success rate")
    average_effectiveness: float = Field(0.0, description="Average effectiveness rating")
    time_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution by time period") 