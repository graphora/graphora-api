"""Response models for merge progress tracking API"""
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class MergeStage(str, Enum):
    """Stages of the merge process"""
    VALIDATION = "validation"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"

class MergeProgressResponse(BaseModel):
    """Response model for merge progress"""
    merge_id: str
    transform_id: str
    status: str
    current_stage: MergeStage
    progress_percentage: float = Field(ge=0.0, le=100.0)
    started_at: datetime
    estimated_completion_time: Optional[datetime] = None
    elapsed_time_seconds: float
    is_active: bool

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