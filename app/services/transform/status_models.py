from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class TransformationStage(str, Enum):
    """Stages of the transformation process"""
    UPLOAD = "upload"
    PARSE = "parse"
    CHUNK = "chunk"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"

class StageStatus(str, Enum):
    """Status of a transformation stage"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TransformStatus(str, Enum):
    """Overall status of the transformation"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ResourceMetrics(BaseModel):
    """Resource usage metrics"""
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)
    peak_memory_mb: float = Field(default=0.0, ge=0.0)
    disk_usage_mb: float = Field(default=0.0, ge=0.0)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    llm_tokens_used: int = Field(default=0, ge=0)
    api_calls_made: int = Field(default=0, ge=0)
    
    @property
    def memory_usage_gb(self) -> float:
        """Memory usage in GB"""
        return self.memory_usage_mb / 1024
    
    @property
    def peak_memory_gb(self) -> float:
        """Peak memory usage in GB"""
        return self.peak_memory_mb / 1024

class ErrorSummary(BaseModel):
    """Error details for a transformation stage"""
    stage: TransformationStage
    error_type: str
    error_message: str
    error_timestamp: datetime
    stack_trace: Optional[str] = None
    affected_components: List[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    is_recoverable: bool = True
    recovery_instructions: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            **self.model_dump(),
            'error_timestamp': self.error_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorSummary':
        """Create from dictionary"""
        if isinstance(data['error_timestamp'], str):
            data['error_timestamp'] = datetime.fromisoformat(
                data['error_timestamp']
            )
        return cls(**data)

class StageProgress(BaseModel):
    """Progress tracking for a transformation stage"""
    stage: TransformationStage
    status: StageStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    percentage_complete: float = Field(default=0.0, ge=0.0, le=100.0)
    items_total: Optional[int] = None
    items_processed: Optional[int] = None
    error_details: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds"""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000
    
    @property
    def is_complete(self) -> bool:
        """Check if stage is complete"""
        return self.status in [StageStatus.COMPLETED, StageStatus.FAILED]
    
    def update_progress(
        self,
        items_processed: int,
        items_total: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Update progress with new metrics"""
        self.items_processed = items_processed
        self.items_total = items_total
        self.percentage_complete = (
            (items_processed / items_total) * 100
            if items_total > 0 else 0.0
        )
        if metrics:
            self.metrics.update(metrics)
    
    def start(self):
        """Start the stage"""
        self.start_time = datetime.now()
        self.status = StageStatus.IN_PROGRESS
    
    def complete(self):
        """Complete the stage"""
        self.end_time = datetime.now()
        self.status = StageStatus.COMPLETED
        self.percentage_complete = 100.0
    
    def fail(self, error: ErrorSummary):
        """Mark stage as failed"""
        self.end_time = datetime.now()
        self.status = StageStatus.FAILED
        self.error_details = error.to_dict()

class DetailedTransformStatus(BaseModel):
    """Detailed transformation status"""
    transform_id: str
    overall_status: TransformStatus
    current_stage: TransformationStage
    stages_progress: Dict[TransformationStage, StageProgress]
    start_time: datetime
    estimated_completion_time: Optional[datetime] = None
    error_summary: Optional[ErrorSummary] = None
    resource_metrics: ResourceMetrics
    
    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds"""
        return (
            datetime.now() - self.start_time
        ).total_seconds() * 1000
    
    @property
    def percentage_complete(self) -> float:
        """Overall completion percentage"""
        if not self.stages_progress:
            return 0.0
            
        total_percentage = sum(
            stage.percentage_complete
            for stage in self.stages_progress.values()
        )
        return total_percentage / len(self.stages_progress)
    
    def update_stage_progress(
        self,
        stage: TransformationStage,
        items_processed: int,
        items_total: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Update progress for a stage"""
        if stage in self.stages_progress:
            self.stages_progress[stage].update_progress(
                items_processed,
                items_total,
                metrics
            )
    
    def start_stage(self, stage: TransformationStage):
        """Start a new stage"""
        if stage in self.stages_progress:
            self.stages_progress[stage].start()
        self.current_stage = stage
    
    def complete_stage(self, stage: TransformationStage):
        """Complete a stage"""
        if stage in self.stages_progress:
            self.stages_progress[stage].complete()
    
    def fail_stage(
        self,
        stage: TransformationStage,
        error: ErrorSummary
    ):
        """Mark a stage as failed"""
        if stage in self.stages_progress:
            self.stages_progress[stage].fail(error)
        self.error_summary = error
        self.overall_status = TransformStatus.FAILED
    
    def update_resource_metrics(
        self,
        metrics: Dict[str, float]
    ):
        """Update resource metrics"""
        self.resource_metrics = ResourceMetrics(**metrics)
    
    def estimate_completion_time(
        self,
        historical_timings: Dict[TransformationStage, List[float]]
    ):
        """Estimate completion time based on historical data"""
        if self.overall_status != TransformStatus.RUNNING:
            return
            
        try:
            # Calculate remaining time for each stage
            remaining_ms = 0
            
            for stage, progress in self.stages_progress.items():
                if not progress.is_complete:
                    # Get historical timings for this stage
                    stage_timings = historical_timings.get(stage, [])
                    if not stage_timings:
                        continue
                    
                    # Calculate average time for this stage
                    avg_time = sum(stage_timings) / len(stage_timings)
                    
                    # Calculate remaining time based on progress
                    if progress.percentage_complete > 0:
                        remaining = (
                            avg_time * 
                            (100 - progress.percentage_complete) / 100
                        )
                    else:
                        remaining = avg_time
                    
                    remaining_ms += remaining
            
            if remaining_ms > 0:
                self.estimated_completion_time = datetime.now().fromtimestamp(
                    datetime.now().timestamp() + 
                    (remaining_ms / 1000)
                )
                
        except Exception:
            # Don't fail if estimation fails
            pass
