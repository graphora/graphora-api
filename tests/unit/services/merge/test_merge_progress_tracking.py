"""Unit tests for merge progress tracking functionality"""
import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from app.services.merge.service import MergeService
from app.schemas.merge import (
    MergeProgressResponse,
    MergeStatisticsResponse,
    MergeSummaryResponse,
    MergeStage
)

@pytest.mark.asyncio
async def test_get_merge_progress():
    """Test getting merge progress"""
    # Arrange
    mock_progress_tracker = AsyncMock()
    
    # Create mock progress data
    from app.services.merge.models import MergeProgress, MergeStageProgress, StageStatus, MergeStage
    
    start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    mock_progress = MergeProgress(
        merge_id="merge_123",
        overall_status="running",
        start_time=start_time,
        end_time=None,
        current_stage=MergeStage.EXTRACT,
        stages_progress={
            MergeStage.EXTRACT: MergeStageProgress(
                stage=MergeStage.EXTRACT,
                status=StageStatus.RUNNING,
                percentage_complete=50.0,
                start_time=start_time,
                end_time=None,
                metrics={}
            )
        }
    )
    
    # Setup mock progress tracker to return the mock progress
    mock_progress_tracker.get_progress.return_value = mock_progress
    
    # Create mock service
    merge_service = MergeService(
        storage=MagicMock(),
        production_storage=MagicMock(),
        progress_tracker=mock_progress_tracker
    )
    
    # Mock Redis client
    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps({
        "status": "running",
        "current_stage": "extract",
        "overall_progress": 50.0,
        "start_time": start_time.isoformat(),
        "stages": {
            "extract": {
                "status": "running",
                "percentage_complete": 50.0,
                "start_time": start_time.isoformat(),
                "end_time": None
            }
        }
    })
    
    with patch('redis.asyncio.Redis.from_url') as mock_redis_factory:
        mock_redis_factory.return_value.__aenter__.return_value = mock_redis
        
        # Act
        result = await merge_service.get_merge_progress("merge_123")
        
        # Assert
        assert result is not None
        assert result.merge_id == "merge_123"
        assert result.overall_status == "running"
        assert result.current_stage == MergeStage.EXTRACT
        assert result.progress_percentage > 0.0
        assert result.start_time == start_time
        assert result.end_time is None
        assert result.elapsed_time_seconds > 0.0
        assert MergeStage.EXTRACT in result.stages_progress

@pytest.mark.asyncio
async def test_get_merge_statistics():
    """Test getting merge statistics"""
    # Arrange
    mock_redis = AsyncMock()
    
    # Mock status data
    merge_status = {
        "transform_id": "transform_123",
        "status": "completed",
        "started_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Mock statistics data
    merge_stats = {
        "nodes": {
            "total": 100,
            "processed": 75,
            "created": 50,
            "updated": 25,
            "unchanged": 0,
            "failed": 0
        },
        "relationships": {
            "total": 50,
            "processed": 40,
            "created": 30,
            "updated": 10,
            "unchanged": 0,
            "failed": 0
        },
        "conflicts_resolved": 10,
        "memory_usage_mb": 256.5,
        "processing_time_ms": 1500.0,
        "performed_by": "test_user"
    }
    
    # Setup mock Redis get method to return different values based on key
    async def mock_get(key):
        if key == "merge:merge_123:status":
            return json.dumps(merge_status)
        elif key == "merge:merge_123:statistics":
            return json.dumps(merge_stats)
        return None
    
    mock_redis.get.side_effect = mock_get
    
    # Create mock service
    merge_service = MergeService(
        storage=MagicMock(),
        production_storage=MagicMock(),
        progress_tracker=MagicMock()
    )
    
    # Mock Redis client
    with patch('redis.asyncio.Redis.from_url') as mock_redis_factory:
        mock_redis_factory.return_value.__aenter__.return_value = mock_redis
        
        # Act
        result = await merge_service.get_merge_statistics("merge_123")
        
        # Assert
        assert result is not None
        assert result.merge_id == "merge_123"
        assert result.transform_id == "transform_123"
        assert result.nodes.total == 100
        assert result.nodes.processed == 75
        assert result.nodes.created == 50
        assert result.nodes.updated == 25
        assert result.relationships.total == 50
        assert result.conflicts_resolved == 10
        assert result.performed_by == "test_user"

@pytest.mark.asyncio
async def test_get_merge_history_with_filtering():
    """Test getting merge history with filtering"""
    # Arrange
    mock_redis = AsyncMock()
    merge_ids = [b"merge_1", b"merge_2", b"merge_3"]
    mock_redis.smembers.return_value = merge_ids
    
    # Create mock status data with more explicit dates for testing
    now = datetime.now(timezone.utc)
    
    merge_1_status = {
        "transform_id": "transform_1",
        "status": "completed",
        "started_at": (now - timedelta(days=1)).isoformat(),
        "completed_at": now.isoformat()
    }
    
    merge_2_status = {
        "transform_id": "transform_2",
        "status": "failed",
        "started_at": (now - timedelta(days=2)).isoformat(),
        "completed_at": (now - timedelta(days=2, hours=1)).isoformat()
    }
    
    merge_3_status = {
        "transform_id": "transform_1", # Same transform as merge_1
        "status": "completed",
        "started_at": (now - timedelta(days=3)).isoformat(),
        "completed_at": (now - timedelta(days=3, hours=1)).isoformat()
    }
    
    # Stats for each merge
    merge_1_stats = {
        "nodes": {"created": 10, "updated": 5},
        "relationships": {"created": 20, "updated": 10},
        "performed_by": "user_1"
    }
    
    merge_2_stats = {
        "nodes": {"created": 5, "updated": 0},
        "relationships": {"created": 0, "updated": 0},
        "performed_by": "user_2"
    }
    
    merge_3_stats = {
        "nodes": {"created": 15, "updated": 0},
        "relationships": {"created": 25, "updated": 0},
        "performed_by": "user_1"
    }
    
    # Setup Redis mock for status and stats
    async def mock_get(key):
        if key == "merge:merge_1:status":
            return json.dumps(merge_1_status)
        elif key == "merge:merge_2:status":
            return json.dumps(merge_2_status)
        elif key == "merge:merge_3:status":
            return json.dumps(merge_3_status)
        elif key == "merge:merge_1:statistics":
            return json.dumps(merge_1_stats)
        elif key == "merge:merge_2:statistics":
            return json.dumps(merge_2_stats)
        elif key == "merge:merge_3:statistics":
            return json.dumps(merge_3_stats)
        return None
        
    mock_redis.get.side_effect = mock_get
    
    # Create mock service
    merge_service = MergeService(
        storage=MagicMock(),
        production_storage=MagicMock(),
        progress_tracker=MagicMock()
    )
    
    # Mock Redis client
    with patch('redis.asyncio.Redis.from_url') as mock_redis_factory:
        mock_redis_factory.return_value.__aenter__.return_value = mock_redis
        
        # Act - Test filtering by transform_id
        result_transform_filter = await merge_service.get_merge_history(transform_id="transform_1")
        
        # Act - Test filtering by status
        result_status_filter = await merge_service.get_merge_history(status="failed")
        
        # Act - Test filtering by date - use a date that includes both merge_1 and merge_2
        two_days_ago = now - timedelta(days=2, hours=2)  # This should include merge_1 and merge_2
        result_date_filter = await merge_service.get_merge_history(start_date=two_days_ago)
        
        # Assert - Transform filter should return merge_1 and merge_3
        assert len(result_transform_filter) == 2
        assert "merge_1" in [r.merge_id for r in result_transform_filter]
        assert "merge_3" in [r.merge_id for r in result_transform_filter]
        
        # Assert - Status filter should return only merge_2
        assert len(result_status_filter) == 1
        assert result_status_filter[0].merge_id == "merge_2"
        assert result_status_filter[0].status == "failed"
        
        # Assert - Date filter should return merge_1 and merge_2 (newer than 2 days ago)
        merge_ids_in_result = [r.merge_id for r in result_date_filter]
        assert "merge_1" in merge_ids_in_result, f"merge_1 should be in date filter results, got {merge_ids_in_result}"
        assert "merge_2" in merge_ids_in_result, f"merge_2 should be in date filter results, got {merge_ids_in_result}"
        assert len(result_date_filter) == 2, f"Expected 2 results, got {len(result_date_filter)}: {merge_ids_in_result}"

@pytest.mark.asyncio
async def test_merge_progress_not_found():
    """Test handling of non-existent merge ID for progress"""
    # Arrange
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    
    # Create mock service
    merge_service = MergeService(
        storage=MagicMock(),
        production_storage=MagicMock(),
        progress_tracker=MagicMock()
    )
    
    # Mock Redis client
    with patch('redis.asyncio.Redis.from_url') as mock_redis_factory:
        mock_redis_factory.return_value.__aenter__.return_value = mock_redis
        
        # Act & Assert
        with pytest.raises(ValueError, match="Merge nonexistent_id not found"):
            await merge_service.get_merge_progress("nonexistent_id")

@pytest.mark.asyncio
async def test_merge_statistics_not_found():
    """Test handling of non-existent merge ID for statistics"""
    # Arrange
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    
    # Create mock service
    merge_service = MergeService(
        storage=MagicMock(),
        production_storage=MagicMock(),
        progress_tracker=MagicMock()
    )
    
    # Mock Redis client
    with patch('redis.asyncio.Redis.from_url') as mock_redis_factory:
        mock_redis_factory.return_value.__aenter__.return_value = mock_redis
        
        # Act
        result = await merge_service.get_merge_statistics("nonexistent_id")
        
        # Assert
        assert result is None 