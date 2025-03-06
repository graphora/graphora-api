"""Progress tracking for merge operations"""
import psutil
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging
import asyncio
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError
import traceback

from app.services.merge.models import (
    MergeStage,
    MergeStatus,
    StageStatus,
    MergeProgress,
    MergeStageProgress,
    ResourceMetrics
)
from app.schemas.conflicts import ConflictBatch, ConflictType, ConflictSeverity
from app.config import settings
from app.utils.redis import DateTimeEncoder

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Tracks progress of merge operations"""
    
    def __init__(self, redis_client: Optional[Redis] = None, max_retries: int = 3):
        """Initialize progress tracker"""
        self.redis = redis_client or Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            socket_timeout=5,  # 5 second timeout
            socket_connect_timeout=5,
            retry_on_timeout=True,
            decode_responses=True  # Automatically decode responses to strings
        )
        self.max_retries = max_retries
        self.process = psutil.Process()
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.redis.ping()  # Test connection
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.aclose()
            self.redis = None
            
    async def _redis_operation(self, operation, *args, **kwargs) -> Any:
        """Execute Redis operation with retries"""
        if not self.redis:
            raise RuntimeError("Redis client is not initialized")
            
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Redis operation failed after {self.max_retries} attempts: {str(e)}")
                    raise
                await asyncio.sleep(1)  # Wait before retrying
                
    def _get_redis_key(self, merge_id: str, suffix: str) -> str:
        """Get Redis key for merge data"""
        # Remove any existing merge: prefix to avoid double prefixing
        merge_id = merge_id.replace("merge:", "")
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
            await self._redis_operation(
                self.redis.set,
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
            status_data = await self._redis_operation(
                self.redis.get,
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return await self.initialize_merge(merge_id)
            
            # Parse and update status
            status = MergeProgress.model_validate_json(status_data)
            status.start_stage(stage)
            
            # Store updated status
            await self._redis_operation(
                self.redis.set,
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
            status_data = await self._redis_operation(
                self.redis.get,
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
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to update merge progress: {str(e)}")
    
    async def update_conflicts(
        self,
        merge_id: str,
        conflict_batch: ConflictBatch
    ) -> None:
        """Update conflict information in the progress tracker
        
        Args:
            merge_id: ID of the merge operation
            conflict_batch: Batch of conflicts detected during merge
        """
        try:
            logger.info(f"Updating conflicts for merge {merge_id}")
            logger.info(f"Conflict batch ID: {conflict_batch.batch_id}")
            logger.info(f"Total conflicts: {conflict_batch.total_conflicts}")

            # Get current status
            status_data = await self._redis_operation(
                lambda: self.redis.get(self._get_redis_key(merge_id, "status"))
            )
            if not status_data:
                logger.warning(f"No status data found for merge {merge_id}")
                return
            
            # Parse status
            status = MergeProgress.model_validate_json(status_data)
            
            # Update conflict detection stage with conflict metrics
            if MergeStage.CONFLICT_DETECTION in status.stages_progress:
                stage_progress = status.stages_progress[MergeStage.CONFLICT_DETECTION]
                
                # Add conflict metrics
                conflict_metrics = {
                    "total_conflicts": conflict_batch.total_conflicts,
                    "conflict_types": {
                        conflict_type.value: sum(1 for c in conflict_batch.conflicts if c.conflict_type == conflict_type)
                        for conflict_type in ConflictType
                    },
                    "conflict_severities": {
                        severity.value: sum(1 for c in conflict_batch.conflicts if c.severity == severity)
                        for severity in ConflictSeverity
                    },
                    "conflict_groups": len(conflict_batch.conflict_groups)
                }
                
                # Update stage metrics
                if "metrics" not in stage_progress.metrics:
                    stage_progress.metrics["metrics"] = {}
                stage_progress.metrics["metrics"].update(conflict_metrics)
                logger.info(f"Updated conflict metrics: {conflict_metrics}")
            
            # Store updated status
            await self._redis_operation(
                lambda: self.redis.set(
                    self._get_redis_key(merge_id, "status"),
                    status.model_dump_json()
                )
            )
            
            # Store conflict batch ID and data for reference
            batch_key = self._get_redis_key(merge_id, f"conflicts:batch:{conflict_batch.batch_id}")
            logger.info(f"Storing conflict batch at key: {batch_key}")
            await self._redis_operation(
                lambda: self.redis.set(
                    batch_key,
                    conflict_batch.model_dump_json(),
                    ex=settings.CONFLICT_BATCH_TTL
                )
            )
            
            # Store reference to batch ID
            await self._redis_operation(
                lambda: self.redis.set(
                    self._get_redis_key(merge_id, "conflict_batch_id"),
                    conflict_batch.batch_id
                )
            )
            
            logger.info("Successfully updated conflicts in progress tracker")
            
        except Exception as e:
            logger.error(f"Failed to update conflicts in progress tracker: {str(e)}")
            raise
    
    async def complete_merge_stage(
        self,
        merge_id: str,
        stage: MergeStage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete a merge stage"""
        try:
            # Get current status
            status_data = await self._redis_operation(
                self.redis.get,
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            status = MergeProgress.model_validate_json(status_data)
            
            # Update stage status
            if stage in status.stages_progress:
                stage_progress = status.stages_progress[stage]
                stage_progress.status = StageStatus.COMPLETED
                stage_progress.end_time = datetime.now(timezone.utc)
                
                # Calculate percentage complete if items_total is available
                if stage_progress.items_total is not None and stage_progress.items_total > 0:
                    stage_progress.percentage_complete = 100
                    stage_progress.items_processed = stage_progress.items_total
            
            # Update stage metrics
            if metadata and stage in status.stages_progress:
                if not stage_progress.metrics:
                    stage_progress.metrics = {}
                stage_progress.metrics.update(metadata)
            
            # Update overall status if this was the final stage
            if stage == MergeStage.VERIFICATION or stage == MergeStage.MERGE:
                # Check if all stages are completed or if this is the final stage
                all_completed = all(
                    s.status == StageStatus.COMPLETED 
                    for s in status.stages_progress.values() 
                    if s.status != StageStatus.PENDING
                )
                
                if all_completed:
                    status.overall_status = MergeStatus.COMPLETED
                    status.end_time = datetime.now(timezone.utc)
                    logger.info(f"Merge {merge_id} completed successfully")
            
            # Store updated status
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            traceback.print_exc()
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
            status_data = await self._redis_operation(
                self.redis.get,
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
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to mark merge as failed: {str(e)}")
    
    async def fail_merge_stage(
        self,
        merge_id: str,
        stage: MergeStage,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a merge stage as failed"""
        try:
            # Get current status
            status_data = await self._redis_operation(
                self.redis.get,
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            status = MergeProgress.model_validate_json(status_data)
            status.fail_stage(stage, error)
            
            # Update stage metrics
            if metadata and stage in status.stages_progress:
                status.stages_progress[stage].metrics.update(metadata)
            
            # Store updated status
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to mark merge stage as failed: {str(e)}")
    
    async def get_progress(self, merge_id: str) -> Optional[MergeProgress]:
        """Get current merge progress"""
        try:
            status_data = await self._redis_operation(
                self.redis.get,
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return None
                
            return MergeProgress.model_validate_json(status_data)
            
        except Exception as e:
            logger.error(f"Failed to get merge progress: {str(e)}")
            return None
    
    async def cancel_merge(
        self,
        merge_id: str,
        reason: str = "Cancelled by user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cancel a merge operation
        
        Args:
            merge_id: ID of the merge operation
            reason: Reason for cancellation
            metadata: Optional metadata about the cancellation
        """
        try:
            # Get current status
            progress = await self.get_progress(merge_id)
            if not progress:
                return
                
            # Update status to cancelled
            progress.overall_status = MergeStatus.CANCELLED
            
            # If there's a current stage, mark it as failed
            if progress.current_stage:
                stage = progress.current_stage
                if stage in progress.stages_progress:
                    progress.stages_progress[stage].status = StageStatus.FAILED
                    progress.stages_progress[stage].end_time = datetime.now(timezone.utc)
                    progress.stages_progress[stage].error_details = {
                        "reason": reason,
                        "cancelled_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Add metadata if provided
                    if metadata:
                        progress.stages_progress[stage].metrics.update(metadata)
            
            # Set end time
            progress.end_time = datetime.now(timezone.utc)
            
            # Store updated status
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                progress.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to cancel merge: {str(e)}")
    
    async def get_all_merge_ids(self) -> List[str]:
        """Get all merge IDs from Redis
        
        Returns:
            List of merge IDs that have status information in Redis
        """
        try:
            # Get all keys matching the pattern merge:*:status
            keys = await self._redis_operation(
                self.redis.keys,
                "merge:*:status"
            )
            
            # Extract merge IDs from keys
            merge_ids = []
            for key in keys:
                # Format is merge:{merge_id}:status
                parts = key.split(":")
                if len(parts) == 3 and parts[0] == "merge" and parts[2] == "status":
                    merge_ids.append(parts[1])
            
            return merge_ids
            
        except Exception as e:
            logger.error(f"Failed to get all merge IDs: {str(e)}")
            return []
    
    async def pause_merge_stage(
        self,
        merge_id: str,
        stage: MergeStage,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Pause a merge stage for human review or other reasons.
        
        Args:
            merge_id: The ID of the merge operation
            stage: The stage to pause
            reason: The reason for pausing the stage
            details: Optional additional details about the pause
        """
        try:
            status_key = f"merge:{merge_id}:status"
            status_json = await self._redis_operation(
                self.redis.get,
                status_key
            )
            
            if not status_json:
                raise ValueError(f"No status found for merge {merge_id}")
            
            status = MergeProgress.model_validate_json(status_json)
            
            # Check if the stage exists in stages_progress, if not, add it
            if stage not in status.stages_progress:
                status.stages_progress[stage] = MergeStageProgress(
                    stage=stage,
                    status=StageStatus.PENDING
                )
            
            # Update the stage status to PAUSED
            stage_progress = status.stages_progress[stage]
            stage_progress.status = StageStatus.PAUSED
            stage_progress.pause_reason = reason
            
            # Add details if provided
            if details:
                if not stage_progress.metrics:
                    stage_progress.metrics = {}
                stage_progress.metrics["pause_details"] = details
                stage_progress.metrics["paused_at"] = datetime.now(timezone.utc).isoformat()
            
            # Save the updated status
            await self._redis_operation(
                self.redis.set,
                status_key,
                status.model_dump_json()
            )
            
            logger.info(f"Paused merge {merge_id} at stage {stage.value} due to: {reason}")
        except Exception as e:
            logger.error(f"Failed to pause merge {merge_id} at stage {stage.value}: {str(e)}")
            raise
    
    async def update_merge_stage(
        self,
        merge_id: str,
        stage: MergeStage
    ) -> Optional[MergeProgress]:
        """Update the current stage of a merge operation
        
        Args:
            merge_id: ID of the merge operation
            stage: New stage to set
            
        Returns:
            Updated MergeProgress if successful, None otherwise
        """
        try:
            # Get current status
            status_data = await self._redis_operation(
                self.redis.get,
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return None
            
            # Parse and update status
            status = MergeProgress.model_validate_json(status_data)
            status.current_stage = stage
            
            # If stage is not already initialized, initialize it
            if stage not in status.stages_progress:
                status.stages_progress[stage] = MergeStageProgress(
                    stage=stage,
                    status=StageStatus.PENDING,
                    percentage_complete=0.0
                )
            
            # Store updated status
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to update merge stage: {str(e)}")
            return None
    
    async def update_stage_status(
        self,
        merge_id: str,
        stage: MergeStage,
        status_value: str
    ) -> None:
        """Update the status of a merge stage
        
        Args:
            merge_id: ID of the merge operation
            stage: Stage to update
            status_value: New status value
        """
        try:
            # Get current status
            status_data = await self._redis_operation(
                self.redis.get,
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            status = MergeProgress.model_validate_json(status_data)
            
            # Update stage status
            if stage in status.stages_progress:
                stage_progress = status.stages_progress[stage]
                stage_progress.status = StageStatus(status_value)
                
                # Set start/end times based on status
                if status_value == "in_progress" and not stage_progress.start_time:
                    stage_progress.start_time = datetime.now()
                elif status_value in ["completed", "failed"] and not stage_progress.end_time:
                    stage_progress.end_time = datetime.now()
            
            # Update overall status if needed
            if status_value == "failed":
                status.overall_status = MergeStatus.FAILED
            elif status_value == "completed" and stage == MergeStage.VERIFICATION:
                status.overall_status = MergeStatus.COMPLETED
                status.end_time = datetime.now()
            
            # Store updated status
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to update stage status: {str(e)}")
    
    async def initialize_merge_progress(self, merge_id: str) -> MergeProgress:
        """Initialize merge progress tracking for verification
        
        This is a simplified version of initialize_merge that can be used
        when only verification is needed, without going through the full merge process.
        
        Args:
            merge_id: ID of the merge operation
            
        Returns:
            MergeProgress object
        """
        try:
            # Check if merge progress already exists
            existing_progress = await self.get_progress(merge_id)
            if existing_progress:
                return existing_progress
            
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
                overall_status=MergeStatus.RUNNING,
                current_stage=MergeStage.VERIFICATION,
                stages_progress=stages_progress,
                start_time=datetime.now(),
                resource_metrics=self._get_resource_metrics()
            )
            
            # Store in Redis
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                status.model_dump_json()
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to initialize merge progress: {str(e)}")
            raise
    
    async def get_merge_progress(self, merge_id: str) -> Optional[MergeProgress]:
        """Get the current progress of a merge operation
        
        Args:
            merge_id: ID of the merge operation
            
        Returns:
            MergeProgress if found, None otherwise
        """
        try:
            # Get current status
            status_data = await self._redis_operation(
                self.redis.get,
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return None
            
            # Parse status
            status = MergeProgress.model_validate_json(status_data)
            return status
            
        except Exception as e:
            logger.error(f"Failed to get merge progress: {str(e)}")
            return None
    
    async def update_merge_status(
        self,
        merge_id: str,
        status: MergeStatus
    ) -> None:
        """Update the overall status of a merge operation
        
        Args:
            merge_id: ID of the merge operation
            status: New merge status
        """
        try:
            # Get current status
            status_data = await self._redis_operation(
                self.redis.get,
                self._get_redis_key(merge_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            progress = MergeProgress.model_validate_json(status_data)
            progress.overall_status = status
            
            # If status is terminal, set end time
            if status in [MergeStatus.COMPLETED, MergeStatus.FAILED, MergeStatus.ROLLED_BACK]:
                progress.end_time = datetime.now(timezone.utc)
            
            # Store updated status
            await self._redis_operation(
                self.redis.set,
                self._get_redis_key(merge_id, "status"),
                progress.model_dump_json()
            )
            
            logger.info(f"Updated merge {merge_id} status to {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to update merge status: {str(e)}")
