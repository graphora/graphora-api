"""Models for graph merge conflict detection and resolution"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import pytz

class ConflictStatus(str, Enum):
    """Status of a conflict in the merge process"""
    PENDING = "pending"      # Conflict detected but not yet resolved
    RESOLVED = "resolved"    # Conflict has been resolved
    AUTO_RESOLVED = "auto_resolved"  # Conflict was automatically resolved
    IGNORED = "ignored"      # Conflict was intentionally ignored
    FAILED = "failed"        # Resolution attempt failed

class ConflictType(str, Enum):
    """Types of conflicts that can occur during merge"""
    PROPERTY = "property"
    PROPERTY_VALUE = "property_value"  # Property has different values
    PROPERTY_MISSING = "property_missing"  # Property exists in one graph but not the other
    PROPERTY_TYPE = "property_type"  # Property has different types
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
    KEEP_BOTH_RELS = "keep_both_relationships"  # Keep both relationships
    REVERSE_RELATIONSHIP = "reverse_relationship"  # Reverse relationship direction
    MERGE_REL_PROPS = "merge_rel_props"  # Merge relationship properties
    MATCH_ENTITY = "match_entity"  # Match with a specific production entity

class StrategyType(str, Enum):
    """Types of resolution strategies"""
    PREFER_STAGING = "prefer_staging"
    PREFER_PRODUCTION = "prefer_production"
    MERGE_VALUES = "merge_values"
    KEEP_BOTH = "keep_both"
    LLM_ASSISTED = "llm_assisted"
    CUSTOM = "custom"
    RULE_BASED = "rule_based"

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

class StrategyResult(BaseModel):
    """Result of applying a resolution strategy"""
    strategy_name: str = Field(..., description="Name of the strategy used")
    strategy_type: StrategyType = Field(..., description="Type of strategy used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this strategy")
    explanation: str = Field(..., description="Explanation of why this strategy was chosen")
    selected_option_id: Optional[str] = Field(None, description="ID of the selected resolution option")
    applied_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="When the strategy was applied")

class ConflictResolutionRequest(BaseModel):
    """Request model for conflict resolution"""
    resolution_id: Optional[str] = None
    resolution_type: Optional[str] = None
    additional_data: Dict[str, Any] = Field(default_factory=dict)
    resolved_by: str = Field(default="user")
    comments: Optional[str] = Field(default=None, description="Optional comments about the resolution")
    auto_resolve_similar: bool = Field(default=False, description="Whether to auto-resolve similar conflicts")

class ConflictResolutionResponse(BaseModel):
    """Response for conflict resolution"""
    merge_id: str
    conflict_id: str
    resolution_id: Optional[str]
    success: bool
    resolved: bool
    error: Optional[str] = None

class ConflictResolutionResult(BaseModel):
    """Result of applying a resolution to a conflict"""
    conflict_id: str
    success: bool
    resolved: bool
    error: Optional[str] = None

class BulkResolutionRequest(BaseModel):
    """Request model for bulk conflict resolution"""
    conflict_ids: List[str]
    resolution_type: str
    additional_data: Optional[Dict[str, Any]] = None
    resolved_by: str
    comments: Optional[str] = None

class BulkResolutionResult(BaseModel):
    """Result of applying a resolution to a single conflict in bulk operation"""
    conflict_id: str
    resolved: bool
    error: Optional[str] = None

class BulkResolutionResponse(BaseModel):
    """Response model for bulk conflict resolution"""
    merge_id: str
    total: int
    resolved: int
    results: List[BulkResolutionResult]

class Conflict(BaseModel):
    """Conflict model representing a detected conflict between staging and production graphs"""
    id: str
    merge_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    property_name: Optional[str] = None
    staging_ids: Optional[List[str]] = None
    production_ids: Optional[List[str]] = None
    staging_value: Optional[Any] = None
    production_value: Optional[Any] = None
    source_data: Optional[Any] = None
    target_data: Optional[Any] = None
    description: str
    context: Optional[Dict[str, Any]] = None
    resolution_options: List[ResolutionOption] = []
    resolution: Optional[ResolutionOption] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolved_by: Optional[str] = None
    requires_review: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc))
    
    def __init__(self, **data):
        super().__init__(**data)
        # For backward compatibility
        if self.entity_id is None and self.staging_ids:
            self.entity_id = self.staging_ids[0] if self.staging_ids else None
        if self.entity_type is None and self.context and "entity_type" in self.context:
            self.entity_type = self.context.get("entity_type")

class PendingConflictsResponse(BaseModel):
    """Response model for pending conflicts"""
    merge_id: str
    conflicts: List[Conflict]
    total: int
    limit: int
    offset: int

class ConflictGroup(BaseModel):
    """Group of related conflicts"""
    id: str
    merge_id: str
    conflicts: List[Conflict]
    pattern: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc))

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
    auto_resolution_stats: Dict[str, Any] = Field(default_factory=dict, description="Statistics about auto-resolution")

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
    auto_resolved: Optional[bool] = None
    strategy_types: Optional[List[StrategyType]] = None

class StrategyConfig(BaseModel):
    """Configuration for a resolution strategy"""
    enabled: bool = Field(default=True, description="Whether this strategy is enabled")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")
    priority: int = Field(default=0, description="Priority of this strategy (higher = tried first)")
    entity_types: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Entity-specific settings")
    property_rules: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Property-specific rules")
    conflict_types: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Conflict type settings")

class StrategySelectionConfig(BaseModel):
    """Configuration for the strategy selection engine"""
    default_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Default minimum confidence threshold")
    always_use_llm: bool = Field(default=False, description="Whether to always use LLM for strategy selection")
    strategies: Dict[str, StrategyConfig] = Field(default_factory=dict, description="Strategy-specific configurations")
    custom_strategies: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Custom strategy configurations")
    auto_resolution_enabled: bool = Field(default=True, description="Whether auto-resolution is enabled")

class ResolutionRequest(BaseModel):
    """Request to apply a resolution"""
    resolution_id: str = Field(..., description="ID of the resolution option to apply")

class ResolutionResult(BaseModel):
    """Result of applying a resolution"""
    applied: bool = Field(..., description="Whether the resolution was successfully applied")
    conflict_id: str = Field(..., description="ID of the conflict that was resolved")
    resolution_id: Optional[str] = Field(None, description="ID of the resolution that was applied")
    verification: Optional[Dict[str, Any]] = Field(None, description="Verification results")
    error: Optional[str] = Field(None, description="Error message if application failed")
    changes: Optional[Dict[str, Any]] = Field(None, description="Changes that were applied")

class BatchResolutionItem(BaseModel):
    """Single resolution in a batch"""
    conflict_id: str = Field(..., description="ID of the conflict to resolve")
    resolution_id: str = Field(..., description="ID of the resolution to apply")

class BatchResolutionRequest(BaseModel):
    """Request to apply multiple resolutions"""
    resolutions: List[BatchResolutionItem] = Field(..., description="List of resolutions to apply")

class BatchResolutionResult(BaseModel):
    """Result of applying multiple resolutions"""
    total: int = Field(..., description="Total number of resolutions requested")
    success_count: int = Field(..., description="Number of successful applications")
    failure_count: int = Field(..., description="Number of failed applications")
    results: List[ResolutionResult] = Field(..., description="Individual resolution results")

class GroupBatchResolutionRequest(BaseModel):
    """Request for batch conflict resolution"""
    group_key: str = Field(..., description="Key identifying the conflict group")
    resolution_option: ResolutionOption = Field(..., description="Resolution option to apply to all conflicts in the group")
    exceptions: List[str] = Field(default_factory=list, description="List of conflict IDs to exclude from batch resolution")

class ResolutionPattern(BaseModel):
    """Data model for resolution patterns extracted from conflict resolutions"""
    id: str = Field(..., description="Unique identifier for this pattern")
    conflict_type: ConflictType = Field(..., description="Type of conflict this pattern applies to")
    context_features: Dict[str, Any] = Field(default_factory=dict, description="Context features that define when this pattern applies")
    condition_features: Dict[str, Any] = Field(default_factory=dict, description="Condition features that determine when this pattern should be applied")
    resolution_action: str = Field(..., description="Type of resolution action to take")
    resolution_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters needed to apply the resolution")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this pattern")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for semantic similarity search")
    occurrence_count: int = Field(default=1, description="Number of times this pattern has been observed")
    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="When this pattern was first created")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="When this pattern was last updated")
