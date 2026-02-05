from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import platform
import time
import shutil
from pathlib import Path

import psutil
import redis
from prefect import get_client

from app.services.transform.status_models import (
    TransformationStage,
    StageStatus,
    TransformStatus,
    StageProgress,
    DetailedTransformStatus,
    ErrorSummary,
    ResourceMetrics,
)
from app.config import settings
from app.utils.logger import logger


class InMemoryStore:
    """Simple in-memory key-value store as Redis fallback"""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[bytes]:
        value = self._data.get(key)
        return value.encode() if isinstance(value, str) else value

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        self._data[key] = value if isinstance(value, bytes) else str(value)
        return True

    def delete(self, key: str) -> int:
        if key in self._data:
            del self._data[key]
            return 1
        return 0


class ProgressTracker:
    """Track transformation progress and resource usage"""

    def __init__(
        self,
        redis_url: str = settings.REDIS_URL,
        timing_window: int = settings.TIMING_WINDOW_HOURS,
    ):
        """Initialize tracker with Redis connection, falling back to in-memory if unavailable"""
        self.timing_window = timing_window
        self._using_memory = False

        # Try to connect to Redis, fall back to in-memory if unavailable
        try:
            self.redis = redis.from_url(redis_url)
            self.redis.ping()  # Test connection
            logger.info("Progress tracker using Redis")
        except (redis.exceptions.ConnectionError, redis.exceptions.RedisError) as e:
            logger.warning(
                f"Redis unavailable ({e}), using in-memory progress tracking"
            )
            self.redis = InMemoryStore()
            self._using_memory = True

        # Initialize stages in order
        self.stages = [
            TransformationStage.UPLOAD,
            TransformationStage.PARSE,
            TransformationStage.CHUNK,
            TransformationStage.TRANSFORM,
            TransformationStage.LOAD,
            TransformationStage.FAILED,
        ]

    def _get_redis_key(self, transform_id: str, suffix: str) -> str:
        """Get Redis key for a component"""
        return f"transform:{transform_id}:{suffix}"

    async def _get_prefect_status(self, transform_id: str) -> Optional[str]:
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
            "cpu_usage_percent": process.cpu_percent(),
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "disk_usage_mb": 0.0,  # TODO: Implement if needed
            "processing_time_ms": 0.0,  # Set by caller
            "llm_tokens_used": 0,  # Set by caller
            "api_calls_made": 0,  # Set by caller
        }

        # Platform-specific peak memory tracking
        system = platform.system()
        if system == "Windows":
            # Windows-specific peak memory attribute
            if hasattr(memory_info, "peak_wset"):
                metrics["peak_memory_mb"] = memory_info.peak_wset / 1024 / 1024
            else:
                metrics["peak_memory_mb"] = metrics["memory_usage_mb"]
        elif system == "Linux":
            # On Linux, read from /proc/[pid]/status for peak memory
            try:
                with open(f"/proc/{process.pid}/status", "r") as f:
                    for line in f:
                        if line.startswith("VmHWM:"):  # "High Water Mark"
                            peak_kb = float(line.split()[1])
                            metrics["peak_memory_mb"] = peak_kb / 1024
                            break
                    else:
                        # If VmHWM not found, use current memory as fallback
                        metrics["peak_memory_mb"] = metrics["memory_usage_mb"]
            except (IOError, ValueError):
                metrics["peak_memory_mb"] = metrics["memory_usage_mb"]
        elif system == "Darwin":  # macOS
            # macOS doesn't provide peak memory usage through standard interfaces
            # Use current memory as an approximation, or maintain your own high watermark
            try:
                # Try to use ru_maxrss from resource module as alternative
                import resource

                rusage = resource.getrusage(resource.RUSAGE_SELF)
                # On macOS, ru_maxrss is in bytes
                metrics["peak_memory_mb"] = rusage.ru_maxrss / 1024 / 1024
            except (ImportError, AttributeError):
                metrics["peak_memory_mb"] = metrics["memory_usage_mb"]
        else:
            # Other platforms - use current memory as fallback
            metrics["peak_memory_mb"] = metrics["memory_usage_mb"]

        # Add process uptime
        metrics["process_uptime_seconds"] = time.time() - process.create_time()

        return metrics

    async def initialize_transform(self, transform_id: str) -> DetailedTransformStatus:
        """Initialize transformation status"""
        # Create initial stage progress
        stages_progress = {
            stage: StageProgress(
                stage=stage, status=StageStatus.PENDING, percentage_complete=0.0
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
            resource_metrics=ResourceMetrics(),
        )

        # Store in Redis
        self.redis.set(
            self._get_redis_key(transform_id, "status"), status.model_dump_json()
        )

        return status

    async def update_stage_progress(
        self,
        transform_id: str,
        stage: TransformationStage,
        items_processed: int,
        items_total: int,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Update progress for a stage"""
        try:
            # Get current status
            status_data = self.redis.get(self._get_redis_key(transform_id, "status"))
            if not status_data:
                return

            # Parse status
            status = DetailedTransformStatus.model_validate_json(status_data)

            # Update stage progress
            status.update_stage_progress(stage, items_processed, items_total, metrics)

            # Update resource metrics
            current_metrics = self._get_resource_metrics()
            if metrics:
                current_metrics.update(metrics)
            status.update_resource_metrics(current_metrics)

            # Store updated status
            self.redis.set(
                self._get_redis_key(transform_id, "status"), status.model_dump_json()
            )

            # Store timing if stage completed
            if items_processed == items_total:
                self._store_stage_timing(
                    transform_id, stage, status.stages_progress[stage].duration_ms or 0
                )

        except Exception as e:
            logger.debug(f"Failed to update progress: {str(e)}")

    def _store_stage_timing(
        self, transform_id: str, stage: TransformationStage, duration_ms: float
    ):
        """Store stage timing for estimation"""
        try:
            timing_key = f"timing:{stage.value}"

            # Get existing timings
            timings_data = self.redis.get(timing_key)
            timings = json.loads(timings_data) if timings_data else []

            # Add new timing
            timings.append(
                {
                    "transform_id": transform_id,
                    "duration_ms": duration_ms,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Keep only recent timings
            cutoff = datetime.now().timestamp() - (self.timing_window * 3600)
            timings = [
                t
                for t in timings
                if datetime.fromisoformat(t["timestamp"]).timestamp() > cutoff
            ]

            # Store updated timings
            self.redis.set(timing_key, json.dumps(timings))

        except Exception as e:
            logger.debug(f"Failed to store timing: {str(e)}")

    def _get_historical_timings(self) -> Dict[TransformationStage, List[float]]:
        """Get historical timing data for estimation"""
        timings = {}

        try:
            for stage in self.stages:
                timing_key = f"timing:{stage.value}"
                timings_data = self.redis.get(timing_key)

                if timings_data:
                    stage_timings = json.loads(timings_data)
                    timings[stage] = [t["duration_ms"] for t in stage_timings]

        except Exception as e:
            logger.debug(f"Failed to get timings: {str(e)}")

        return timings

    async def get_detailed_status(
        self, transform_id: str
    ) -> Optional[DetailedTransformStatus]:
        """Get comprehensive transformation status"""
        try:
            # Get status from Redis
            status_data = self.redis.get(self._get_redis_key(transform_id, "status"))
            if not status_data:
                return None

            # Parse status
            status = DetailedTransformStatus.model_validate_json(status_data)

            # Only update from Prefect if our internal status is not terminal
            # This prevents Prefect's "RUNNING" from overwriting our "COMPLETED"
            if status.overall_status not in (
                TransformStatus.COMPLETED,
                TransformStatus.FAILED,
            ):
                prefect_status = await self._get_prefect_status(transform_id)
                if prefect_status:
                    # If Prefect says failed, always update (failure takes precedence)
                    if prefect_status.lower() == "failed":
                        status.overall_status = TransformStatus.FAILED
                    else:
                        status.overall_status = TransformStatus(prefect_status)

            # Update resource metrics
            current_metrics = self._get_resource_metrics()
            status.update_resource_metrics(current_metrics)

            # Update completion estimate
            status.estimate_completion_time(self._get_historical_timings())

            return status

        except Exception as e:
            logger.debug(f"Failed to get status: {str(e)}")
            return None

    async def start_stage(self, transform_id: str, stage: TransformationStage):
        """Start a new stage"""
        try:
            # Get current status
            status_data = self.redis.get(self._get_redis_key(transform_id, "status"))
            if not status_data:
                return

            # Parse and update status
            status = DetailedTransformStatus.model_validate_json(status_data)
            status.start_stage(stage)

            prefect_status = await self._get_prefect_status(transform_id)
            if prefect_status:
                status.overall_status = TransformStatus(prefect_status)

            # Store updated status
            self.redis.set(
                self._get_redis_key(transform_id, "status"), status.model_dump_json()
            )

        except Exception as e:
            logger.debug(f"Failed to start stage: {str(e)}")

    async def complete_stage(self, transform_id: str, stage: TransformationStage):
        """Complete a stage"""
        try:
            # Get current status
            status_data = self.redis.get(self._get_redis_key(transform_id, "status"))
            if not status_data:
                return

            # Parse and update status
            status = DetailedTransformStatus.model_validate_json(status_data)
            status.complete_stage(stage)
            if stage == TransformationStage.LOAD:
                status.overall_status = TransformStatus.COMPLETED
            # Store updated status
            self.redis.set(
                self._get_redis_key(transform_id, "status"), status.model_dump_json()
            )

        except Exception as e:
            logger.debug(f"Failed to complete stage: {str(e)}")

    async def fail_stage(
        self, transform_id: str, stage: TransformationStage, error: ErrorSummary
    ):
        """Mark a stage as failed"""
        try:
            # Get current status
            status_data = self.redis.get(self._get_redis_key(transform_id, "status"))
            if not status_data:
                return

            # Parse and update status
            status = DetailedTransformStatus.model_validate_json(status_data)
            status.fail_stage(stage, error)
            status.overall_status = TransformStatus.FAILED

            # Store updated status
            self.redis.set(
                self._get_redis_key(transform_id, "status"), status.model_dump_json()
            )

        except Exception as e:
            logger.debug(f"Failed to fail stage: {str(e)}")

    def cleanup_transform(self, transform_id: str):
        """Clean up transformation data including Redis status and filesystem files"""
        try:
            # Remove Redis status
            self.redis.delete(self._get_redis_key(transform_id, "status"))
            logger.info(f"Cleaned up Redis status for transform {transform_id}")

        except Exception as e:
            logger.warning(f"Failed to cleanup Redis status: {str(e)}")

        # Clean up filesystem files
        try:
            transform_dir = Path(settings.UPLOAD_DIR) / transform_id
            if transform_dir.exists():
                shutil.rmtree(transform_dir)
                logger.info(f"Cleaned up transform directory: {transform_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup transform directory: {str(e)}")

    def cleanup_old_transforms(self, max_age_hours: int = 24):
        """
        Clean up old transform directories that exceed the maximum age.

        Args:
            max_age_hours: Maximum age in hours before a transform directory is cleaned up
        """
        try:
            upload_dir = Path(settings.UPLOAD_DIR)
            if not upload_dir.exists():
                return

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            cleaned_count = 0
            for item in upload_dir.iterdir():
                if item.is_dir() and item.name.startswith("transform_"):
                    try:
                        # Check directory age based on modification time
                        dir_mtime = item.stat().st_mtime
                        age_seconds = current_time - dir_mtime

                        if age_seconds > max_age_seconds:
                            shutil.rmtree(item)
                            cleaned_count += 1
                            logger.info(
                                f"Cleaned up old transform directory: {item.name} "
                                f"(age: {age_seconds / 3600:.1f} hours)"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup directory {item.name}: {str(e)}"
                        )

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old transform directories")

        except Exception as e:
            logger.warning(f"Failed to cleanup old transforms: {str(e)}")
