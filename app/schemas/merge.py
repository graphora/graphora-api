"""Response models for merge progress tracking API"""
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.services.merge.models import MergeStage as ModelMergeStage

class MergeStage(str, Enum):
    """Stages of the merge process"""
    VALIDATION = "validation"
    EXECUTION = "execution"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    CONFLICT_DETECTION = "conflict_detection"
    RESOLUTION = "resolution"
    MERGE = "merge"
    APPLY_CHANGES = "apply_changes"
    VERIFICATION = "verification"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"

class MergeStageProgressResponse(BaseModel):
    """Response model for merge stage progress"""
    stage: str
    status: str
    percentage_complete: float = Field(ge=0.0, le=100.0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

# Add StageProgressResponse as an alias for MergeStageProgressResponse
class StageProgressResponse(BaseModel):
    """Response model for stage progress"""
    status: str
    percentage_complete: float = Field(ge=0.0, le=100.0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None

class MergeProgressResponse(BaseModel):
    """Response model for merge progress"""
    merge_id: str
    overall_status: str
    current_stage: Optional[ModelMergeStage] = None
    progress_percentage: float = Field(ge=0.0, le=100.0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    estimated_time_remaining_seconds: Optional[float] = None
    elapsed_time_seconds: float = 0.0
    stages_progress: Dict[str, StageProgressResponse] = Field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if merge is still active"""
        return self.overall_status not in ["completed", "failed", "cancelled", "rolled_back"]

class NodeStatistics(BaseModel):
    """Statistics for nodes in merge operation"""
    total: int
    processed: int
    created: int
    updated: int
    unchanged: int
    failed: int

class RelationshipStatistics(BaseModel):
    """Statistics for relationships in merge operation"""
    total: int
    processed: int
    created: int
    updated: int
    unchanged: int
    failed: int

class MergeStatisticsResponse(BaseModel):
    """Response model for detailed merge statistics"""
    merge_id: str
    transform_id: str
    nodes: NodeStatistics
    relationships: RelationshipStatistics
    conflicts_resolved: int
    memory_usage_mb: float
    processing_time_ms: float
    performed_by: str
    errors: List[str] = Field(default_factory=list)

class MergeSummaryResponse(BaseModel):
    """Summary response for merge history"""
    merge_id: str
    transform_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    nodes_affected: int
    relationships_affected: int
    performed_by: str

class VerificationCheckResponse(BaseModel):
    """Response model for a verification check"""
    check_type: str
    success: bool
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    affected_entities: List[str] = Field(default_factory=list)

class VerificationResultResponse(BaseModel):
    """Response model for merge verification results"""
    merge_id: str
    transform_id: str
    success: bool
    checks: List[VerificationCheckResponse] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None
    verification_time_ms: float = 0.0 