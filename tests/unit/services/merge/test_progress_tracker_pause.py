"""Tests for progress tracker pause functionality"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
from datetime import datetime, timezone
from app.services.merge.progress import ProgressTracker
from app.services.merge.models import MergeStage, StageStatus, MergeProgress, MergeStageProgress

@pytest.mark.asyncio
async def test_pause_merge_stage():
    """Test pausing a merge stage for human review"""
    # Mock Redis client
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=json.dumps({
        "merge_id": "test-merge-id",
        "overall_status": "running",
        "current_stage": "resolution",
        "stages_progress": {
            "resolution": {
                "stage": "resolution",
                "status": "running",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "percentage_complete": 50.0,
                "metrics": {}
            }
        },
        "start_time": datetime.now(timezone.utc).isoformat(),
        "resource_metrics": {}
    }))
    
    # Create progress tracker with mock Redis
    tracker = ProgressTracker(redis_client=mock_redis)
    
    # Call pause_merge_stage
    await tracker.pause_merge_stage(
        merge_id="test-merge-id",
        stage=MergeStage.RESOLUTION,
        reason="waiting_for_human_review",
        details={"critical_conflicts": ["conflict1", "conflict2"]}
    )
    
    # Verify Redis was called with updated progress
    mock_redis.set.assert_called_once()
    
    # Get the JSON that was passed to Redis
    call_args = mock_redis.set.call_args[0]
    assert call_args[0] == "merge:test-merge-id:progress"
    
    # Parse the JSON to verify the pause metadata was added
    progress_data = json.loads(call_args[1])
    
    # Verify stage status is still RUNNING (with pause metadata)
    assert progress_data["stages_progress"]["resolution"]["status"] == "running"
    
    # Verify pause metadata
    metrics = progress_data["stages_progress"]["resolution"]["metrics"]
    assert metrics["paused"] is True
    assert metrics["pause_reason"] == "waiting_for_human_review"
    assert "paused_at" in metrics
    assert metrics["pause_details"]["critical_conflicts"] == ["conflict1", "conflict2"]

@pytest.mark.asyncio
async def test_pause_merge_stage_not_found():
    """Test pausing a merge stage that doesn't exist"""
    # Mock Redis client
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    
    # Create progress tracker with mock Redis
    tracker = ProgressTracker(redis_client=mock_redis)
    
    # Call pause_merge_stage
    await tracker.pause_merge_stage(
        merge_id="test-merge-id",
        stage=MergeStage.RESOLUTION,
        reason="waiting_for_human_review"
    )
    
    # Verify Redis get was called but set was not
    mock_redis.get.assert_called_once()
    mock_redis.set.assert_not_called()

@pytest.mark.asyncio
async def test_pause_merge_stage_invalid_stage():
    """Test pausing a merge stage that doesn't exist in the progress"""
    # Mock Redis client
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=json.dumps({
        "merge_id": "test-merge-id",
        "overall_status": "running",
        "current_stage": "conflict_detection",
        "stages_progress": {
            "conflict_detection": {
                "stage": "conflict_detection",
                "status": "running",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "percentage_complete": 50.0,
                "metrics": {}
            }
            # Resolution stage not present
        },
        "start_time": datetime.now(timezone.utc).isoformat(),
        "resource_metrics": {}
    }))
    
    # Create progress tracker with mock Redis
    tracker = ProgressTracker(redis_client=mock_redis)
    
    # Call pause_merge_stage
    await tracker.pause_merge_stage(
        merge_id="test-merge-id",
        stage=MergeStage.RESOLUTION,  # This stage doesn't exist in the progress
        reason="waiting_for_human_review"
    )
    
    # Verify Redis get was called but set was not
    mock_redis.get.assert_called_once()
    mock_redis.set.assert_not_called() 