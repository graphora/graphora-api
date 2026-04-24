"""Unit tests for Progress Tracker.

Phase 5: Transform Service Tests - Progress Tracker
Tests for transformation progress tracking with mocked Redis.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from graphora_server.services.transform.status_models import (
    TransformationStage,
    StageStatus,
    TransformStatus,
    StageProgress,
    DetailedTransformStatus,
    ErrorSummary,
    ResourceMetrics,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = MagicMock()
    mock.get = MagicMock(return_value=None)
    mock.set = MagicMock()
    mock.delete = MagicMock()
    return mock


@pytest.fixture
def mock_settings():
    """Mock settings."""
    with patch("graphora_server.services.transform.progress_tracker.settings") as mock:
        mock.REDIS_URL = "redis://localhost:6379"
        mock.TIMING_WINDOW_HOURS = 24
        yield mock


@pytest.fixture
def progress_tracker(mock_redis, mock_settings):
    """Create ProgressTracker with mocked dependencies."""
    with patch("graphora_server.services.transform.progress_tracker.redis") as mock_redis_module:
        mock_redis_module.from_url.return_value = mock_redis

        from graphora_server.services.transform.progress_tracker import ProgressTracker

        tracker = ProgressTracker()
        tracker.redis = mock_redis
        return tracker


@pytest.fixture
def sample_status():
    """Create sample DetailedTransformStatus."""
    stages_progress = {
        TransformationStage.UPLOAD: StageProgress(
            stage=TransformationStage.UPLOAD,
            status=StageStatus.COMPLETED,
            percentage_complete=100.0,
        ),
        TransformationStage.PARSE: StageProgress(
            stage=TransformationStage.PARSE,
            status=StageStatus.IN_PROGRESS,
            percentage_complete=50.0,
        ),
    }

    return DetailedTransformStatus(
        transform_id="transform-123",
        overall_status=TransformStatus.RUNNING,
        current_stage=TransformationStage.PARSE,
        stages_progress=stages_progress,
        start_time=datetime.now(),
        resource_metrics=ResourceMetrics(),
    )


# ============================================================
# Redis Key Tests
# ============================================================


class TestRedisKeyGeneration:
    """Test Redis key generation."""

    def test_should_generate_correct_key_format(self, progress_tracker):
        """Should generate keys with correct prefix and suffix."""
        key = progress_tracker._get_redis_key("transform-abc", "status")
        assert key == "transform:transform-abc:status"

    def test_should_include_transform_id_in_key(self, progress_tracker):
        """Should include transform ID in key."""
        key = progress_tracker._get_redis_key("my-transform-123", "progress")
        assert "my-transform-123" in key

    def test_should_include_suffix_in_key(self, progress_tracker):
        """Should include suffix in key."""
        key = progress_tracker._get_redis_key("transform-1", "metrics")
        assert key.endswith(":metrics")


# ============================================================
# Initialize Transform Tests
# ============================================================


class TestInitializeTransform:
    """Test transform initialization."""

    @pytest.mark.asyncio
    async def test_should_create_initial_status(self, progress_tracker, mock_redis):
        """Should create initial status with all stages."""
        result = await progress_tracker.initialize_transform("transform-new")

        assert result.transform_id == "transform-new"
        assert result.overall_status == TransformStatus.INITIALIZING
        assert result.current_stage == TransformationStage.UPLOAD

    @pytest.mark.asyncio
    async def test_should_initialize_all_stages_as_pending(
        self, progress_tracker, mock_redis
    ):
        """Should initialize all stages with PENDING status."""
        result = await progress_tracker.initialize_transform("transform-new")

        for stage in progress_tracker.stages:
            assert stage in result.stages_progress
            assert result.stages_progress[stage].status == StageStatus.PENDING
            assert result.stages_progress[stage].percentage_complete == 0.0

    @pytest.mark.asyncio
    async def test_should_store_status_in_redis(self, progress_tracker, mock_redis):
        """Should store initial status in Redis."""
        await progress_tracker.initialize_transform("transform-new")

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert "transform:transform-new:status" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_should_set_start_time(self, progress_tracker, mock_redis):
        """Should set start time on initialization."""
        before = datetime.now()
        result = await progress_tracker.initialize_transform("transform-new")
        after = datetime.now()

        assert before <= result.start_time <= after

    @pytest.mark.asyncio
    async def test_should_initialize_empty_resource_metrics(
        self, progress_tracker, mock_redis
    ):
        """Should initialize with empty resource metrics."""
        result = await progress_tracker.initialize_transform("transform-new")

        assert result.resource_metrics is not None
        assert result.resource_metrics.cpu_usage_percent == 0.0


# ============================================================
# Get Detailed Status Tests
# ============================================================


class TestGetDetailedStatus:
    """Test getting detailed status."""

    @pytest.mark.asyncio
    async def test_should_return_none_when_not_found(
        self, progress_tracker, mock_redis
    ):
        """Should return None when transform not found."""
        mock_redis.get.return_value = None

        result = await progress_tracker.get_detailed_status("unknown-transform")

        assert result is None

    @pytest.mark.asyncio
    async def test_should_parse_status_from_redis(
        self, progress_tracker, mock_redis, sample_status
    ):
        """Should parse status from Redis."""
        mock_redis.get.return_value = sample_status.model_dump_json()

        with patch.object(
            progress_tracker, "_get_prefect_status", new_callable=AsyncMock
        ) as mock_prefect:
            mock_prefect.return_value = None

            with patch.object(
                progress_tracker, "_get_resource_metrics"
            ) as mock_metrics:
                mock_metrics.return_value = {
                    "cpu_usage_percent": 25.0,
                    "memory_usage_mb": 512.0,
                    "peak_memory_mb": 512.0,
                    "disk_usage_mb": 0.0,
                    "processing_time_ms": 0.0,
                    "llm_tokens_used": 0,
                    "api_calls_made": 0,
                }

                result = await progress_tracker.get_detailed_status("transform-123")

        assert result is not None
        assert result.transform_id == "transform-123"

    @pytest.mark.asyncio
    async def test_should_handle_redis_errors_gracefully(
        self, progress_tracker, mock_redis
    ):
        """Should return None on Redis errors."""
        mock_redis.get.side_effect = Exception("Redis connection error")

        result = await progress_tracker.get_detailed_status("transform-123")

        assert result is None


# ============================================================
# Start Stage Tests
# ============================================================


class TestStartStage:
    """Test starting a stage."""

    @pytest.mark.asyncio
    async def test_should_do_nothing_when_status_not_found(
        self, progress_tracker, mock_redis
    ):
        """Should do nothing when status not found."""
        mock_redis.get.return_value = None

        await progress_tracker.start_stage("unknown", TransformationStage.PARSE)

        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_update_stage_to_in_progress(
        self, progress_tracker, mock_redis, sample_status
    ):
        """Should update stage status to IN_PROGRESS."""
        mock_redis.get.return_value = sample_status.model_dump_json()

        with patch.object(
            progress_tracker, "_get_prefect_status", new_callable=AsyncMock
        ) as mock_prefect:
            mock_prefect.return_value = None

            await progress_tracker.start_stage(
                "transform-123", TransformationStage.CHUNK
            )

        # Verify set was called to save updated status
        mock_redis.set.assert_called_once()


# ============================================================
# Complete Stage Tests
# ============================================================


class TestCompleteStage:
    """Test completing a stage."""

    @pytest.mark.asyncio
    async def test_should_do_nothing_when_status_not_found(
        self, progress_tracker, mock_redis
    ):
        """Should do nothing when status not found."""
        mock_redis.get.return_value = None

        await progress_tracker.complete_stage("unknown", TransformationStage.PARSE)

        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_update_status_in_redis(
        self, progress_tracker, mock_redis, sample_status
    ):
        """Should update status in Redis."""
        mock_redis.get.return_value = sample_status.model_dump_json()

        await progress_tracker.complete_stage(
            "transform-123", TransformationStage.PARSE
        )

        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_set_overall_completed_when_load_completes(
        self, progress_tracker, mock_redis, sample_status
    ):
        """Should set overall status to COMPLETED when LOAD completes."""
        mock_redis.get.return_value = sample_status.model_dump_json()

        await progress_tracker.complete_stage("transform-123", TransformationStage.LOAD)

        # Verify the saved status has COMPLETED overall
        call_args = mock_redis.set.call_args
        saved_json = call_args[0][1]
        saved_status = json.loads(saved_json)
        assert saved_status["overall_status"] == TransformStatus.COMPLETED.value


# ============================================================
# Fail Stage Tests
# ============================================================


class TestFailStage:
    """Test failing a stage."""

    @pytest.mark.asyncio
    async def test_should_do_nothing_when_status_not_found(
        self, progress_tracker, mock_redis
    ):
        """Should do nothing when status not found."""
        mock_redis.get.return_value = None

        error = ErrorSummary(
            stage=TransformationStage.PARSE,
            error_type="TestError",
            error_message="Test error",
            error_timestamp=datetime.now(timezone.utc),
        )
        await progress_tracker.fail_stage("unknown", TransformationStage.PARSE, error)

        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_set_overall_status_to_failed(
        self, progress_tracker, mock_redis, sample_status
    ):
        """Should set overall status to FAILED."""
        mock_redis.get.return_value = sample_status.model_dump_json()

        error = ErrorSummary(
            stage=TransformationStage.PARSE,
            error_type="ParseError",
            error_message="Parse failed",
            error_timestamp=datetime.now(timezone.utc),
        )
        await progress_tracker.fail_stage(
            "transform-123", TransformationStage.PARSE, error
        )

        call_args = mock_redis.set.call_args
        saved_json = call_args[0][1]
        saved_status = json.loads(saved_json)
        assert saved_status["overall_status"] == TransformStatus.FAILED.value


# ============================================================
# Cleanup Transform Tests
# ============================================================


class TestCleanupTransform:
    """Test cleanup functionality."""

    def test_should_delete_redis_key(self, progress_tracker, mock_redis):
        """Should delete status from Redis."""
        progress_tracker.cleanup_transform("transform-123")

        mock_redis.delete.assert_called_once()
        call_args = mock_redis.delete.call_args[0]
        assert "transform:transform-123:status" in call_args[0]

    def test_should_handle_redis_errors_gracefully(self, progress_tracker, mock_redis):
        """Should handle Redis errors gracefully."""
        mock_redis.delete.side_effect = Exception("Redis error")

        # Should not raise
        progress_tracker.cleanup_transform("transform-123")


# ============================================================
# Update Stage Progress Tests
# ============================================================


class TestUpdateStageProgress:
    """Test updating stage progress."""

    @pytest.mark.asyncio
    async def test_should_do_nothing_when_status_not_found(
        self, progress_tracker, mock_redis
    ):
        """Should do nothing when status not found."""
        mock_redis.get.return_value = None

        await progress_tracker.update_stage_progress(
            "unknown",
            TransformationStage.PARSE,
            items_processed=5,
            items_total=10,
        )

        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_update_progress_in_redis(
        self, progress_tracker, mock_redis, sample_status
    ):
        """Should update progress in Redis."""
        mock_redis.get.return_value = sample_status.model_dump_json()

        with patch.object(progress_tracker, "_get_resource_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "cpu_usage_percent": 25.0,
                "memory_usage_mb": 512.0,
                "peak_memory_mb": 512.0,
                "disk_usage_mb": 0.0,
                "processing_time_ms": 0.0,
                "llm_tokens_used": 0,
                "api_calls_made": 0,
            }

            await progress_tracker.update_stage_progress(
                "transform-123",
                TransformationStage.PARSE,
                items_processed=5,
                items_total=10,
            )

        mock_redis.set.assert_called()

    @pytest.mark.asyncio
    async def test_should_store_timing_when_stage_completes(
        self, progress_tracker, mock_redis, sample_status
    ):
        """Should store timing when stage completes (items_processed == items_total)."""
        mock_redis.get.return_value = sample_status.model_dump_json()

        with patch.object(progress_tracker, "_get_resource_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "cpu_usage_percent": 25.0,
                "memory_usage_mb": 512.0,
                "peak_memory_mb": 512.0,
                "disk_usage_mb": 0.0,
                "processing_time_ms": 0.0,
                "llm_tokens_used": 0,
                "api_calls_made": 0,
            }

            with patch.object(
                progress_tracker, "_store_stage_timing"
            ) as mock_store_timing:
                await progress_tracker.update_stage_progress(
                    "transform-123",
                    TransformationStage.PARSE,
                    items_processed=10,  # Complete
                    items_total=10,
                )

                mock_store_timing.assert_called_once()


# ============================================================
# Resource Metrics Tests
# ============================================================


class TestResourceMetrics:
    """Test resource metrics collection."""

    def test_should_return_dict_with_required_keys(self, progress_tracker):
        """Should return metrics dict with required keys."""
        with patch("graphora_server.services.transform.progress_tracker.psutil") as mock_psutil:
            mock_process = MagicMock()
            mock_process.cpu_percent.return_value = 25.0
            mock_process.memory_info.return_value = MagicMock(rss=1024 * 1024 * 512)
            mock_process.create_time.return_value = 0
            mock_psutil.Process.return_value = mock_process

            metrics = progress_tracker._get_resource_metrics()

            assert "cpu_usage_percent" in metrics
            assert "memory_usage_mb" in metrics
            assert "peak_memory_mb" in metrics
            assert "disk_usage_mb" in metrics


# ============================================================
# Stage Timing Tests
# ============================================================


class TestStageTiming:
    """Test stage timing storage and retrieval."""

    def test_should_store_timing_in_redis(self, progress_tracker, mock_redis):
        """Should store timing data in Redis."""
        mock_redis.get.return_value = None  # No existing timings

        progress_tracker._store_stage_timing(
            "transform-123", TransformationStage.PARSE, 1500.0
        )

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args[0]
        assert "timing:parse" in call_args[0]

    def test_should_append_to_existing_timings(self, progress_tracker, mock_redis):
        """Should append to existing timing data."""
        existing_timings = [
            {
                "transform_id": "old-transform",
                "duration_ms": 1000.0,
                "timestamp": datetime.now().isoformat(),
            }
        ]
        mock_redis.get.return_value = json.dumps(existing_timings)

        progress_tracker._store_stage_timing(
            "transform-123", TransformationStage.PARSE, 1500.0
        )

        call_args = mock_redis.set.call_args[0]
        saved_timings = json.loads(call_args[1])
        assert len(saved_timings) == 2

    def test_should_get_historical_timings(self, progress_tracker, mock_redis):
        """Should retrieve historical timing data."""
        timings = [
            {
                "transform_id": "t1",
                "duration_ms": 1000.0,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "transform_id": "t2",
                "duration_ms": 1200.0,
                "timestamp": datetime.now().isoformat(),
            },
        ]
        mock_redis.get.return_value = json.dumps(timings)

        result = progress_tracker._get_historical_timings()

        # Should have data for stages queried
        assert isinstance(result, dict)


# ============================================================
# Transformation Stages Tests
# ============================================================


class TestTransformationStages:
    """Test transformation stage configuration."""

    def test_should_have_all_stages_defined(self, progress_tracker):
        """Should have all transformation stages defined."""
        expected_stages = [
            TransformationStage.UPLOAD,
            TransformationStage.PARSE,
            TransformationStage.CHUNK,
            TransformationStage.TRANSFORM,
            TransformationStage.LOAD,
            TransformationStage.FAILED,
        ]

        assert progress_tracker.stages == expected_stages

    def test_should_have_correct_stage_order(self, progress_tracker):
        """Stages should be in correct order."""
        # UPLOAD should be first (index 0)
        assert progress_tracker.stages[0] == TransformationStage.UPLOAD

        # LOAD should be second to last
        assert progress_tracker.stages[-2] == TransformationStage.LOAD

        # FAILED should be last
        assert progress_tracker.stages[-1] == TransformationStage.FAILED
