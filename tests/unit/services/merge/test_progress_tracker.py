import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime, timezone

from app.services.merge.progress import ProgressTracker
from app.services.merge.models import MergeStage, MergeStatus, StageStatus

@pytest.fixture
def mock_redis_client():
    """Mock Redis client fixture"""
    redis_client = AsyncMock()
    redis_client.set = AsyncMock()
    redis_client.get = AsyncMock()
    redis_client.keys = AsyncMock()
    return redis_client

@pytest.fixture
def progress_tracker(mock_redis_client):
    """Progress tracker fixture with mock Redis client"""
    tracker = ProgressTracker(redis_client=mock_redis_client)
    return tracker

@pytest.mark.asyncio
async def test_initialize_merge(progress_tracker, mock_redis_client):
    """Test initializing merge progress"""
    merge_id = "test_merge_id"
    
    # Initialize merge
    status = await progress_tracker.initialize_merge(merge_id)
    
    # Verify Redis set was called
    mock_redis_client.set.assert_called_once()
    
    # Verify status
    assert status.merge_id == merge_id
    assert status.overall_status == MergeStatus.PENDING
    assert status.current_stage is None
    assert len(status.stages_progress) == len(MergeStage)

@pytest.mark.asyncio
async def test_start_merge_stage(progress_tracker, mock_redis_client):
    """Test starting a merge stage"""
    merge_id = "test_merge_id"
    stage = MergeStage.CONFLICT_DETECTION
    
    # Mock get to return None (no existing status)
    mock_redis_client.get.return_value = None
    
    # Start merge stage
    status = await progress_tracker.start_merge_stage(merge_id, stage)
    
    # Verify Redis operations
    assert mock_redis_client.get.call_count == 1
    assert mock_redis_client.set.call_count == 1
    
    # Verify status
    assert status.merge_id == merge_id
    assert status.overall_status == MergeStatus.PENDING  # Initial status is PENDING
    assert status.current_stage is None  # Current stage is None initially
    assert len(status.stages_progress) == len(MergeStage)

@pytest.mark.asyncio
async def test_get_all_merge_ids(progress_tracker, mock_redis_client):
    """Test getting all merge IDs"""
    # Mock keys to return some merge keys
    mock_redis_client.keys.return_value = [
        "merge:merge_id_1:status",
        "merge:merge_id_2:status",
        "merge:merge_id_3:status",
        "some:other:key"  # Should be ignored
    ]
    
    # Get all merge IDs
    merge_ids = await progress_tracker.get_all_merge_ids()
    
    # Verify Redis keys was called
    mock_redis_client.keys.assert_called_once_with("merge:*:status")
    
    # Verify merge IDs
    assert len(merge_ids) == 3
    assert "merge_id_1" in merge_ids
    assert "merge_id_2" in merge_ids
    assert "merge_id_3" in merge_ids
    assert "other" not in merge_ids

@pytest.mark.asyncio
async def test_cancel_merge(progress_tracker, mock_redis_client):
    """Test cancelling a merge"""
    merge_id = "test_merge_id"
    
    # Mock get_progress to return a progress object
    progress = MagicMock()
    progress.overall_status = MergeStatus.RUNNING
    progress.current_stage = MergeStage.MERGE
    progress.stages_progress = {MergeStage.MERGE: MagicMock()}
    progress.model_dump_json.return_value = "{}"
    
    progress_tracker.get_progress = AsyncMock(return_value=progress)
    
    # Cancel merge
    await progress_tracker.cancel_merge(merge_id)
    
    # Verify progress was updated
    assert progress.overall_status == MergeStatus.CANCELLED
    assert progress.end_time is not None
    
    # Verify Redis set was called
    mock_redis_client.set.assert_called_once() 