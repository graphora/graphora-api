"""Models for merge operations"""
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

from app.services.storage.models import Node, Edge

class MergeStage(str, Enum):
    """Stages in the merge process"""
    EXTRACT = "extract"
    ANALYZE = "analyze"
    CONFLICT_DETECTION = "conflict_detection"
    RESOLUTION = "resolution"
    MERGE = "merge"
    APPLY_CHANGES = "apply_changes"

class MergeStatus(str, Enum):
    """Status of a merge operation"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"

class StageStatus(str, Enum):
    """Status of an individual stage"""
    PENDING = "pending"
    RUNNING = "running"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class ResourceMetrics(BaseModel):
    """Resource usage metrics"""
    cpu_percent: float = Field(default=0.0)
    memory_mb: float = Field(default=0.0)
    elapsed_time_ms: float = Field(default=0.0)
    nodes_per_second: Optional[float] = None

class MergeStageProgress(BaseModel):
    """Progress tracking for a merge stage"""
    stage: MergeStage
    status: StageStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    percentage_complete: float = Field(default=0.0, ge=0.0, le=100.0)
    items_total: Optional[int] = None
    items_processed: Optional[int] = None
    error_details: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    pause_reason: Optional[str] = None

    @field_validator('start_time', 'end_time', mode='before')
    @classmethod
    def validate_datetime(cls, v):
        """Convert string timestamps to datetime objects"""
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return None
        return v

    def update_progress(
        self,
        items_processed: int,
        items_total: int,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update progress metrics"""
        self.items_processed = items_processed
        self.items_total = items_total
        self.percentage_complete = (items_processed / items_total * 100) if items_total > 0 else 0
        
        if metrics:
            self.metrics.update(metrics)

class MergeProgress(BaseModel):
    """Overall merge progress tracking"""
    merge_id: str
    overall_status: MergeStatus
    current_stage: Optional[MergeStage] = None
    stages_progress: Dict[MergeStage, MergeStageProgress]
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    resource_metrics: ResourceMetrics = Field(default_factory=ResourceMetrics)

    @field_validator('start_time', 'end_time', mode='before')
    @classmethod
    def validate_datetime(cls, v):
        """Convert string timestamps to datetime objects"""
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                return None
        return v

    def start_stage(self, stage: MergeStage) -> None:
        """Start a new stage"""
        if stage not in self.stages_progress:
            self.stages_progress[stage] = MergeStageProgress(
                stage=stage,
                status=StageStatus.PENDING
            )
        
        stage_progress = self.stages_progress[stage]
        stage_progress.status = StageStatus.RUNNING
        stage_progress.start_time = datetime.now()
        self.current_stage = stage
        self.overall_status = MergeStatus.RUNNING

    def complete_stage(self, stage: MergeStage) -> None:
        """Complete a stage"""
        if stage in self.stages_progress:
            stage_progress = self.stages_progress[stage]
            stage_progress.status = StageStatus.COMPLETED
            stage_progress.end_time = datetime.now()
            stage_progress.percentage_complete = 100.0

    def fail_stage(self, stage: MergeStage, error: str) -> None:
        """Mark a stage as failed"""
        if stage in self.stages_progress:
            stage_progress = self.stages_progress[stage]
            stage_progress.status = StageStatus.FAILED
            stage_progress.end_time = datetime.now()
            stage_progress.error_details = {"error": error}

    def update_resource_metrics(self, metrics: ResourceMetrics) -> None:
        """Update resource usage metrics"""
        self.resource_metrics = metrics

class MergeInitResponse(BaseModel):
    """Response for merge initialization"""
    merge_id: str
    status: MergeStatus
    start_time: datetime

class GraphResponse(BaseModel):
    """Response model for graph extraction"""
    nodes: List[Node]
    edges: List[Edge]
    total_nodes: int
    total_edges: int
    extraction_time_ms: Optional[float] = None

class EntityMatch(BaseModel):
    """Model for entity matching results"""
    staging_id: str
    production_matches: List[str]
    match_confidence: float
    match_strategy: str
    metadata: Dict[str, Any] = {}

class EntityMappingResult(BaseModel):
    """Result of entity mapping process"""
    matches: Dict[str, EntityMatch]
    total_entities: int
    matched_entities: int
    mapping_time_ms: float
    metadata: Dict[str, Any] = {}

class MatchStrategy(str, Enum):
    """Available strategies for entity matching"""
    EXACT_ID = "exact_id"
    EXACT_NAME = "exact_name"
    PROPERTY_SIMILARITY = "property_similarity"
    CUSTOM = "custom"

class ValidationSeverity(str, Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class ValidationIssueType(str, Enum):
    """Types of validation issues"""
    UNKNOWN_ENTITY_TYPE = "unknown_entity_type"
    MISSING_REQUIRED_PROPERTIES = "missing_required_properties"
    INVALID_PROPERTY_TYPE = "invalid_property_type"
    ORPHANED_NODE = "orphaned_node"
    MISSING_REQUIRED_RELATIONSHIP = "missing_required_relationship"
    INVALID_RELATIONSHIP_TYPE = "invalid_relationship_type"
    INVALID_RELATIONSHIP_REFERENCE = "invalid_relationship_reference"
    INVALID_RELATIONSHIP_SOURCE = "invalid_relationship_source"
    INVALID_RELATIONSHIP_TARGET = "invalid_relationship_target"
    UNRESOLVED_CONFLICTS = "unresolved_conflicts"
    VALIDATION_ERROR = "validation_error"

class ValidationIssue(BaseModel):
    """Model for validation issues"""
    type: ValidationIssueType
    message: str
    affected_ids: List[str]
    severity: ValidationSeverity
    metadata: Dict[str, Any] = {}

class ValidationResult(BaseModel):
    """Result of graph validation"""
    valid: bool
    issues: List[ValidationIssue]
    critical_count: int
    warning_count: int
    info_count: int
    total_nodes: int
    total_edges: int
    validation_time_ms: float
    metadata: Dict[str, Any] = {}

class RollbackType(str, Enum):
    """Types of rollback operations"""
    COMPLETE = "complete"
    PARTIAL = "partial"

class RollbackOptions(BaseModel):
    """Options for rollback operations"""
    rollback_type: RollbackType = Field(..., description="Type of rollback to perform")
    entity_ids: Optional[List[str]] = Field(None, description="Entity IDs to rollback (for partial rollback)")
    auto_rollback_on_validation_failure: bool = Field(False, description="Whether to automatically rollback on validation failure")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the rollback")

    @field_validator('entity_ids')
    @classmethod
    def validate_entity_ids(cls, v, values):
        """Validate that entity_ids is provided for partial rollback"""
        if values.data.get('rollback_type') == RollbackType.PARTIAL and (not v or len(v) == 0):
            raise ValueError("entity_ids must be provided for partial rollback")
        return v

class RollbackResponse(BaseModel):
    """Response for rollback operations"""
    rollback_id: str = Field(..., description="ID of the rollback operation")
    merge_id: str = Field(..., description="ID of the merge that was rolled back")
    status: str = Field(..., description="Status of the rollback operation")
    timestamp: datetime = Field(..., description="Timestamp of the rollback operation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Details about the rollback operation")

class SnapshotData(BaseModel):
    """Data for a snapshot of the graph state before merge"""
    snapshot_id: str = Field(..., description="ID of the snapshot")
    merge_id: str = Field(..., description="ID of the merge operation")
    nodes: List[Node] = Field(default_factory=list, description="Nodes in the snapshot")
    relationships: List[Edge] = Field(default_factory=list, description="Relationships in the snapshot")
    timestamp: datetime = Field(..., description="Timestamp when the snapshot was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the snapshot")
