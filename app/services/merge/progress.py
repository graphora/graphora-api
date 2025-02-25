"""Progress tracking for merge operations"""
import psutil
import time
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from redis import Redis

from app.services.merge.models import (
    MergeStage,
    MergeStatus,
    StageStatus,
    MergeProgress,
    MergeStageProgress,
    ResourceMetrics
)
from app.config import settings

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Tracks progress of merge operations"""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        """Initialize progress tracker"""
        self.redis = redis_client or Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        self.process = psutil.Process()
    
    def _get_redis_key(self, merge_id: str, suffix: str) -> str:
        """Get Redis key for merge data"""
        return f"merge:{merge_id}:{suffix}"
    
    def _get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics"""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_mb=memory_info.rss / 1024 / 1024,  # Convert to MB
                elapsed_time_ms=time.time() * 1000
            )
        except Exception as e:
            logger.error(f"Failed to get resource metrics: {str(e)}")
            return ResourceMetrics()
    
    async def initialize_merge(self, merge_id: str) -> MergeProgress:
        """Initialize merge status tracking"""
        try:
            # Create initial stage progress
            stages_progress = {
                stage: MergeStageProgress(
                    stage=stage,
                    status=StageStatus.PENDING,
                    percentage_complete=0.0
                )
                for stage in MergeStage
            }
            
            # Create initial status
            status = MergeProgress(
                merge_id=merge_id,
                overall_status=MergeStatus.PENDING,
                current_stage=None,
                stages_progress=stages_progress,
                start_time=datetime.now(),
                resource_metrics=self._get_resource_metrics()
            )
            
            # Store in Redis
            self.redis.set(
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to initialize merge: {str(e)}")
            raise
    
    async def start_merge_stage(
        self,
        merge_id: str,
        stage: MergeStage
    ) -> Optional[MergeProgress]:
        """Start a new merge stage"""
        try:
            # Get current status
            status_data = self.redis.get(
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return await self.initialize_merge(merge_id)
            
            # Parse and update status
            status = MergeProgress.model_validate_json(status_data)
            status.start_stage(stage)
            
            # Store updated status
            self.redis.set(
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to start merge stage: {str(e)}")
            return None
    
    async def update_merge_progress(
        self,
        merge_id: str,
        stage: MergeStage,
        items_processed: int,
        items_total: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Update progress for a merge stage"""
        try:
            # Get current status
            status_data = self.redis.get(
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return
            
            # Parse status
            status = MergeProgress.model_validate_json(status_data)
            
            # Update stage progress
            if stage in status.stages_progress:
                stage_progress = status.stages_progress[stage]
                stage_progress.update_progress(
                    items_processed,
                    items_total,
                    metrics
                )
            
            # Update resource metrics
            resource_metrics = self._get_resource_metrics()
            if items_processed > 0 and resource_metrics.elapsed_time_ms > 0:
                resource_metrics.nodes_per_second = (
                    items_processed / (resource_metrics.elapsed_time_ms / 1000)
                )
            status.update_resource_metrics(resource_metrics)
            
            # Store updated status
            self.redis.set(
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to update merge progress: {str(e)}")
    
    async def complete_merge_stage(
        self,
        merge_id: str,
        stage: MergeStage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete a merge stage"""
        try:
            # Get current status
            status_data = self.redis.get(
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            status = MergeProgress.model_validate_json(status_data)
            status.complete_stage(stage)
            
            # Update stage metrics
            if metadata and stage in status.stages_progress:
                status.stages_progress[stage].metrics.update(metadata)
            
            # Update overall status if this was the final stage
            if stage == MergeStage.MERGE:
                status.overall_status = MergeStatus.COMPLETED
                status.end_time = datetime.now()
            
            # Store updated status
            self.redis.set(
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to complete merge stage: {str(e)}")
    
    async def fail_merge(
        self,
        merge_id: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark merge as failed"""
        try:
            # Get current status
            status_data = self.redis.get(
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            status = MergeProgress.model_validate_json(status_data)
            
            # Update current stage if exists
            if status.current_stage:
                status.fail_stage(status.current_stage, error)
            
            # Update overall status
            status.overall_status = MergeStatus.FAILED
            status.end_time = datetime.now()
            status.error = error
            
            if metadata:
                status.stages_progress[status.current_stage].metrics.update(metadata)
            
            # Store updated status
            self.redis.set(
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to mark merge as failed: {str(e)}")
    
    async def get_progress(self, merge_id: str) -> Optional[MergeProgress]:
        """Get current progress of a merge operation"""
        try:
            status_data = self.redis.get(
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return None
                
            return MergeProgress.model_validate_json(status_data)
            
        except Exception as e:
            logger.error(f"Failed to get merge progress: {str(e)}")
            return None
