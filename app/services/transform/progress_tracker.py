from typing import Dict, List, Optional
from datetime import datetime
import json
import psutil
import redis
from prefect import get_client
import platform
import time

from app.services.transform.status_models import (
    TransformationStage,
    StageStatus,
    TransformStatus,
    StageProgress,
    DetailedTransformStatus,
    ErrorSummary,
    ResourceMetrics
)
from app.config import settings

class ProgressTracker:
    """Track transformation progress and resource usage"""
    
    def __init__(
        self,
        redis_url: str = settings.REDIS_URL,
        timing_window: int = settings.TIMING_WINDOW_HOURS
    ):
        """Initialize tracker with Redis connection"""
        self.redis = redis.from_url(redis_url)
        self.timing_window = timing_window
        
        # Initialize stages in order
        self.stages = [
            TransformationStage.UPLOAD,
            TransformationStage.PARSE,
            TransformationStage.CHUNK,
            TransformationStage.TRANSFORM,
            TransformationStage.LOAD
        ]
    
    def _get_redis_key(self, transform_id: str, suffix: str) -> str:
        """Get Redis key for a component"""
        return f"transform:{transform_id}:{suffix}"
    
    async def _get_prefect_status(
        self,
        transform_id: str
    ) -> Optional[str]:
        """Get flow run status from Prefect"""
        try:
            async with get_client() as client:
                flow_run = await client.read_flow_run(transform_id)
                return flow_run.state.type.value
        except Exception:
            return None
    
    def _get_resource_metrics(self) -> Dict[str, float]:
        """Get current resource usage metrics in a cross-platform compatible way"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Base metrics available on all platforms
        metrics = {
            'cpu_usage_percent': process.cpu_percent(),
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'disk_usage_mb': 0.0,  # TODO: Implement if needed
            'processing_time_ms': 0.0,  # Set by caller
            'llm_tokens_used': 0,  # Set by caller
            'api_calls_made': 0  # Set by caller
        }
        
        # Platform-specific peak memory tracking
        system = platform.system()
        if system == 'Windows':
            # Windows-specific peak memory attribute
            if hasattr(memory_info, 'peak_wset'):
                metrics['peak_memory_mb'] = memory_info.peak_wset / 1024 / 1024
            else:
                metrics['peak_memory_mb'] = metrics['memory_usage_mb']
        elif system == 'Linux':
            # On Linux, read from /proc/[pid]/status for peak memory
            try:
                with open(f'/proc/{process.pid}/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmHWM:'):  # "High Water Mark"
                            peak_kb = float(line.split()[1])
                            metrics['peak_memory_mb'] = peak_kb / 1024
                            break
                    else:
                        # If VmHWM not found, use current memory as fallback
                        metrics['peak_memory_mb'] = metrics['memory_usage_mb']
            except (IOError, ValueError):
                metrics['peak_memory_mb'] = metrics['memory_usage_mb']
        elif system == 'Darwin':  # macOS
            # macOS doesn't provide peak memory usage through standard interfaces
            # Use current memory as an approximation, or maintain your own high watermark
            try:
                # Try to use ru_maxrss from resource module as alternative
                import resource
                rusage = resource.getrusage(resource.RUSAGE_SELF)
                # On macOS, ru_maxrss is in bytes
                metrics['peak_memory_mb'] = rusage.ru_maxrss / 1024 / 1024
            except (ImportError, AttributeError):
                metrics['peak_memory_mb'] = metrics['memory_usage_mb']
        else:
            # Other platforms - use current memory as fallback
            metrics['peak_memory_mb'] = metrics['memory_usage_mb']
        
        # Add process uptime
        metrics['process_uptime_seconds'] = time.time() - process.create_time()
        
        return metrics
    
    async def initialize_transform(
        self,
        transform_id: str
    ) -> DetailedTransformStatus:
        """Initialize transformation status"""
        # Create initial stage progress
        stages_progress = {
            stage: StageProgress(
                stage=stage,
                status=StageStatus.PENDING,
                percentage_complete=0.0
            )
            for stage in self.stages
        }
        
        # Create initial status
        status = DetailedTransformStatus(
            transform_id=transform_id,
            overall_status=TransformStatus.INITIALIZING,
            current_stage=TransformationStage.UPLOAD,
            stages_progress=stages_progress,
            start_time=datetime.now(),
            resource_metrics=ResourceMetrics()
        )
        
        # Store in Redis
        self.redis.set(
            self._get_redis_key(transform_id, "status"),
            status.model_dump_json()
        )
        
        return status
    
    async def update_stage_progress(
        self,
        transform_id: str,
        stage: TransformationStage,
        items_processed: int,
        items_total: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Update progress for a stage"""
        try:
            # Get current status
            status_data = self.redis.get(
                self._get_redis_key(transform_id, "status")
            )
            if not status_data:
                return
            
            # Parse status
            status = DetailedTransformStatus.model_validate_json(
                status_data
            )
            
            # Update stage progress
            status.update_stage_progress(
                stage,
                items_processed,
                items_total,
                metrics
            )
            
            # Update resource metrics
            current_metrics = self._get_resource_metrics()
            if metrics:
                current_metrics.update(metrics)
            status.update_resource_metrics(current_metrics)
            
            # Store updated status
            self.redis.set(
                self._get_redis_key(transform_id, "status"),
                status.model_dump_json()
            )
            
            # Store timing if stage completed
            if items_processed == items_total:
                self._store_stage_timing(
                    transform_id,
                    stage,
                    status.stages_progress[stage].duration_ms or 0
                )
            
        except Exception as e:
            print(f"Failed to update progress: {str(e)}")
    
    def _store_stage_timing(
        self,
        transform_id: str,
        stage: TransformationStage,
        duration_ms: float
    ):
        """Store stage timing for estimation"""
        try:
            timing_key = f"timing:{stage.value}"
            
            # Get existing timings
            timings_data = self.redis.get(timing_key)
            timings = (
                json.loads(timings_data)
                if timings_data else []
            )
            
            # Add new timing
            timings.append({
                'transform_id': transform_id,
                'duration_ms': duration_ms,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only recent timings
            cutoff = datetime.now().timestamp() - (
                self.timing_window * 3600
            )
            timings = [
                t for t in timings
                if datetime.fromisoformat(
                    t['timestamp']
                ).timestamp() > cutoff
            ]
            
            # Store updated timings
            self.redis.set(
                timing_key,
                json.dumps(timings)
            )
            
        except Exception as e:
            print(f"Failed to store timing: {str(e)}")
    
    def _get_historical_timings(
        self
    ) -> Dict[TransformationStage, List[float]]:
        """Get historical timing data for estimation"""
        timings = {}
        
        try:
            for stage in self.stages:
                timing_key = f"timing:{stage.value}"
                timings_data = self.redis.get(timing_key)
                
                if timings_data:
                    stage_timings = json.loads(timings_data)
                    timings[stage] = [
                        t['duration_ms']
                        for t in stage_timings
                    ]
                    
        except Exception as e:
            print(f"Failed to get timings: {str(e)}")
        
        return timings
    
    async def get_detailed_status(
        self,
        transform_id: str
    ) -> Optional[DetailedTransformStatus]:
        """Get comprehensive transformation status"""
        try:
            # Get status from Redis
            status_data = self.redis.get(
                self._get_redis_key(transform_id, "status")
            )
            if not status_data:
                return None
            
            # Parse status
            status = DetailedTransformStatus.model_validate_json(
                status_data
            )
            
            # Update Prefect status
            prefect_status = await self._get_prefect_status(transform_id)
            if prefect_status:
                status.overall_status = TransformStatus(prefect_status)
            
            # Update resource metrics
            current_metrics = self._get_resource_metrics()
            status.update_resource_metrics(current_metrics)
            
            # Update completion estimate
            status.estimate_completion_time(
                self._get_historical_timings()
            )
            
            return status
            
        except Exception as e:
            print(f"Failed to get status: {str(e)}")
            return None
    
    async def start_stage(
        self,
        transform_id: str,
        stage: TransformationStage
    ):
        """Start a new stage"""
        try:
            # Get current status
            status_data = self.redis.get(
                self._get_redis_key(transform_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            status = DetailedTransformStatus.model_validate_json(
                status_data
            )
            status.start_stage(stage)
            
            prefect_status = await self._get_prefect_status(transform_id)
            if prefect_status:
                status.overall_status = TransformStatus(prefect_status)
            
            # Store updated status
            self.redis.set(
                self._get_redis_key(transform_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            print(f"Failed to start stage: {str(e)}")
    
    async def complete_stage(
        self,
        transform_id: str,
        stage: TransformationStage
    ):
        """Complete a stage"""
        try:
            # Get current status
            status_data = self.redis.get(
                self._get_redis_key(transform_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            status = DetailedTransformStatus.model_validate_json(
                status_data
            )
            status.complete_stage(stage)
            if stage == TransformationStage.LOAD:
                status.overall_status = TransformStatus.COMPLETED
            # Store updated status
            self.redis.set(
                self._get_redis_key(transform_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            print(f"Failed to complete stage: {str(e)}")
    
    async def fail_stage(
        self,
        transform_id: str,
        stage: TransformationStage,
        error: ErrorSummary
    ):
        """Mark a stage as failed"""
        try:
            # Get current status
            status_data = self.redis.get(
                self._get_redis_key(transform_id, "status")
            )
            if not status_data:
                return
            
            # Parse and update status
            status = DetailedTransformStatus.model_validate_json(
                status_data
            )
            status.fail_stage(stage, error)
            status.overall_status = TransformStatus.FAILED
            
            # Store updated status
            self.redis.set(
                self._get_redis_key(transform_id, "status"),
                status.model_dump_json()
            )
            
        except Exception as e:
            print(f"Failed to fail stage: {str(e)}")
    
    def cleanup_transform(self, transform_id: str):
        """Clean up transformation data"""
        try:
            # Remove status
            self.redis.delete(
                self._get_redis_key(transform_id, "status")
            )
            
        except Exception as e:
            print(f"Failed to cleanup: {str(e)}")
