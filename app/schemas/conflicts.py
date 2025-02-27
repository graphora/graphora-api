"""Models for graph merge conflict detection and resolution"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import pytz

class ConflictType(str, Enum):
    """Types of conflicts that can occur during merge"""
    PROPERTY = "property"
    RELATIONSHIP = "relationship"
    RELATIONSHIP_PROPERTY = "relationship_property"
    RELATIONSHIP_MISSING = "relationship_missing"
    RELATIONSHIP_TYPE = "relationship_type"
    RELATIONSHIP_DIRECTION = "relationship_direction"
    ENTITY_MATCH = "entity_match"
    DUPLICATE_ENTITY = "duplicate_entity"
    SCHEMA = "schema"
    OTHER = "other"

class ConflictSeverity(str, Enum):
    """Severity levels for conflicts"""
    CRITICAL = "critical"  # Must be resolved manually
    MAJOR = "major"        # Can be auto-resolved but needs review
    MINOR = "minor"        # Can be auto-resolved with confidence

class ResolutionStrategy(str, Enum):
    """Available strategies for conflict resolution"""
    KEEP_STAGING = "keep_staging"  # Use staging value/entity
    KEEP_PRODUCTION = "keep_production"  # Use production value/entity
    MERGE_VALUES = "merge_values"  # Combine values (e.g., concat arrays)
    CREATE_NEW = "create_new"  # Create new entity with merged properties
    CUSTOM = "custom"  # Custom resolution logic
    IGNORE = "ignore"  # Ignore conflict
    KEEP_BOTH = "keep_both"  # Keep both values
    KEEP_STAGING_REL = "keep_staging_rel"  # Keep staging relationship
    KEEP_PRODUCTION_REL = "keep_production_rel"  # Keep production relationship
    KEEP_BOTH_RELS = "keep_both_rels"  # Keep both relationships
    REVERSE_RELATIONSHIP = "reverse_relationship"  # Reverse relationship direction
    MERGE_REL_PROPS = "merge_rel_props"  # Merge relationship properties
    MATCH_ENTITY = "match_entity"  # Match with a specific production entity

class ResolutionOption(BaseModel):
    """Resolution option for a conflict"""
    id: str = Field(..., description="Unique identifier for this resolution option")
    description: str = Field(..., description="Human-readable description of the resolution")
    resolution_type: str = Field(..., description="Type of resolution action to take")
    resolution_data: Dict[str, Any] = Field(default_factory=dict, description="Data needed to apply the resolution")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this resolution option")
    reasoning: Optional[str] = Field(default=None, description="Reasoning behind this resolution option")
    requires_review: bool = Field(default=True, description="Whether this resolution needs human review")
    auto_resolvable: bool = Field(default=False, description="Whether this can be auto-resolved")

class ConflictResolutionRequest(BaseModel):
    """Request model for conflict resolution"""
    resolution_id: str = Field(..., description="ID of the chosen resolution option")
    resolution_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional data for resolution")
    comments: Optional[str] = Field(default=None, description="Optional comments about the resolution")
    auto_resolve_similar: bool = Field(default=False, description="Whether to auto-resolve similar conflicts")

class Conflict(BaseModel):
    """Represents a conflict between staging and production graphs"""
    id: str = Field(..., description="Unique identifier for this conflict")
    merge_id: str = Field(..., description="ID of the merge operation")
    conflict_type: ConflictType = Field(..., description="Type of conflict")
    severity: ConflictSeverity = Field(..., description="Severity of the conflict")
    entity_id: Optional[str] = Field(None, description="ID of the entity involved")
    entity_type: Optional[str] = Field(None, description="Type of the entity involved")
    property_name: Optional[str] = Field(None, description="Name of the property in conflict")
    staging_value: Optional[Any] = Field(None, description="Value of the property in staging")
    production_value: Optional[Any] = Field(None, description="Value of the property in production")
    description: str = Field(..., description="Human-readable description of the conflict")
    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="When the conflict was detected")
    resolved: bool = Field(False, description="Whether the conflict has been resolved")
    resolution_options: List[ResolutionOption] = Field(default_factory=list, description="Available resolution options")
    resolution: Optional[ResolutionOption] = Field(None, description="Chosen resolution for the conflict")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis of the conflict")

class ConflictGroup(BaseModel):
    """Group of similar conflicts that can be resolved together"""
    id: str = Field(..., description="Unique identifier for this group")
    entity_type: str = Field(..., description="Type of entity involved")
    property_name: str = Field(..., description="Name of the property in conflict")
    value_type: str = Field(..., description="Type of the property value")
    conflict_ids: List[str] = Field(..., description="IDs of conflicts in this group")
    total_conflicts: int = Field(..., description="Total number of conflicts in group")
    pattern: str = Field(..., description="Common pattern identified in conflicts")
    batch_resolvable: bool = Field(..., description="Whether conflicts can be batch resolved")
    recommended_strategy: Optional[str] = Field(None, description="Recommended resolution strategy")
    confidence: float = Field(..., description="Confidence in batch resolution (0-1)")
    risks: List[str] = Field(default_factory=list, description="Potential risks of batch resolution")

class ConflictBatch(BaseModel):
    """Batch of conflicts for efficient processing"""
    batch_id: str = Field(..., description="Unique identifier for this batch")
    merge_id: str = Field(..., description="ID of the merge operation")
    conflicts: List[Conflict] = Field(default_factory=list, description="Conflicts in this batch")
    conflict_groups: List[ConflictGroup] = Field(default_factory=list, description="Groups of similar conflicts")
    total_conflicts: int = Field(default=0, description="Total number of conflicts in batch")
    processed_conflicts: int = Field(default=0, description="Number of conflicts processed")
    created_at: datetime = Field(default_factory=datetime.now, description="When this batch was created")
    completed_at: Optional[datetime] = Field(default=None, description="When batch processing completed")

class ConflictListResponse(BaseModel):
    """Response model for conflict list API"""
    merge_id: str
    conflicts: List[Conflict]
    total_count: int
    summary: Dict[str, Any]
    limit: int
    offset: int

class ConflictFilter(BaseModel):
    """Filter criteria for conflict queries"""
    conflict_types: Optional[List[ConflictType]] = None
    severities: Optional[List[ConflictSeverity]] = None
    resolved: Optional[bool] = None
    entity_ids: Optional[List[str]] = None
    detected_after: Optional[datetime] = None
    detected_before: Optional[datetime] = None
