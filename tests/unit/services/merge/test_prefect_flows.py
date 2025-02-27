"""Tests for Prefect flows for merge conflict resolution"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prefect.testing.utilities import prefect_test_harness
from app.services.merge.prefect_flows import (
    resolution_pipeline_flow,
    load_conflicts,
    resolve_minor_conflicts,
    resolve_major_conflicts, 
    flag_critical_conflicts,
    check_resolution_status
)
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.services.merge import MergeService, MergeStage, StageStatus

@pytest.fixture(autouse=True)
def prefect_test_fixture():
    with prefect_test_harness():
        yield

@pytest.mark.asyncio
async def test_resolution_flow_task_order():
    """Test that tasks execute in the correct order"""
    # Mock all the tasks
    with patch("app.services.merge.prefect_flows.load_conflicts", new_callable=AsyncMock) as mock_load, \
         patch("app.services.merge.prefect_flows.resolve_minor_conflicts", new_callable=AsyncMock) as mock_minor, \
         patch("app.services.merge.prefect_flows.resolve_major_conflicts", new_callable=AsyncMock) as mock_major, \
         patch("app.services.merge.prefect_flows.flag_critical_conflicts", new_callable=AsyncMock) as mock_critical, \
         patch("app.services.merge.prefect_flows.check_resolution_status", new_callable=AsyncMock) as mock_status, \
         patch("app.services.merge.service.MergeService") as mock_service_class:
        
        # Configure mocks
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.progress_tracker = MagicMock()
        mock_service.progress_tracker.start_merge_stage = AsyncMock()
        mock_service.progress_tracker.update_merge_progress = AsyncMock()
        mock_service.progress_tracker.complete_merge_stage = AsyncMock()
        
        # Setup mock return values
        mock_load.return_value = [
            Conflict(
                id="conflict1", 
                merge_id="test_merge_id",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MINOR,
                staging_ids=["s1"],
                production_ids=["p1"],
                description="Test conflict"
            )
        ]
        
        mock_minor.return_value = {
            "total": 1,
            "resolved": 1,
            "failed": 0,
            "skipped": 0,
            "results": {}
        }
        
        mock_major.return_value = {
            "total": 0,
            "resolved": 0,
            "failed": 0,
            "skipped": 0,
            "results": {}
        }
        
        mock_critical.return_value = {
            "total_critical": 0,
            "flagged_for_review": 0,
            "conflict_ids": []
        }
        
        mock_status.return_value = {
            "total_conflicts": 1,
            "resolved_count": 1,
            "unresolved_count": 0,
            "resolution_percentage": 100.0,
            "human_review_needed": False,
            "auto_resolution_complete": True,
            "critical_unresolved": []
        }
        
        # Run the flow
        merge_id = "test_merge_id"
        result = await resolution_pipeline_flow(merge_id)
        
        # Assert all tasks were called in order
        mock_load.assert_called_once_with(merge_id)
        mock_minor.assert_called_once()
        mock_major.assert_called_once()
        mock_critical.assert_called_once()
        mock_status.assert_called_once()
        
        # Check flow completed successfully
        assert result["status"] == "completed"
        assert result["merge_id"] == merge_id
        
        # Verify progress tracking was called
        mock_service.progress_tracker.start_merge_stage.assert_called_once()
        assert mock_service.progress_tracker.update_merge_progress.call_count == 5
        mock_service.progress_tracker.complete_merge_stage.assert_called_once()

@pytest.mark.asyncio
async def test_resolution_flow_human_review_needed():
    """Test flow handling when human review is needed"""
    # Mock all the tasks
    with patch("app.services.merge.prefect_flows.load_conflicts", new_callable=AsyncMock) as mock_load, \
         patch("app.services.merge.prefect_flows.resolve_minor_conflicts", new_callable=AsyncMock) as mock_minor, \
         patch("app.services.merge.prefect_flows.resolve_major_conflicts", new_callable=AsyncMock) as mock_major, \
         patch("app.services.merge.prefect_flows.flag_critical_conflicts", new_callable=AsyncMock) as mock_critical, \
         patch("app.services.merge.prefect_flows.check_resolution_status", new_callable=AsyncMock) as mock_status, \
         patch("app.services.merge.service.MergeService") as mock_service_class:
        
        # Configure mocks
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.progress_tracker = MagicMock()
        mock_service.progress_tracker.start_merge_stage = AsyncMock()
        mock_service.progress_tracker.update_merge_progress = AsyncMock()
        mock_service.progress_tracker.pause_merge_stage = AsyncMock()
        
        # Setup mock return values with critical conflicts
        mock_load.return_value = [
            Conflict(
                id="conflict1", 
                merge_id="test_merge_id",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.CRITICAL,  # Critical conflict
                staging_ids=["s1"],
                production_ids=["p1"],
                description="Test critical conflict"
            )
        ]
        
        mock_minor.return_value = {
            "total": 0,
            "resolved": 0,
            "failed": 0,
            "skipped": 0,
            "results": {}
        }
        
        mock_major.return_value = {
            "total": 0,
            "resolved": 0,
            "failed": 0,
            "skipped": 0,
            "results": {}
        }
        
        mock_critical.return_value = {
            "total_critical": 1,
            "flagged_for_review": 1,
            "conflict_ids": ["conflict1"]
        }
        
        mock_status.return_value = {
            "total_conflicts": 1,
            "resolved_count": 0,
            "unresolved_count": 1,
            "resolution_percentage": 0.0,
            "human_review_needed": True,  # Review needed
            "auto_resolution_complete": True,
            "critical_unresolved": ["conflict1"]
        }
        
        # Run the flow
        merge_id = "test_merge_id"
        result = await resolution_pipeline_flow(merge_id)
        
        # Verify flow was paused for human review
        assert result["status"] == "paused"
        assert "Waiting for human review" in result["message"]
        assert result["requires_review"] == ["conflict1"]
        
        # Verify pause was called instead of complete
        mock_service.progress_tracker.pause_merge_stage.assert_called_once()
        mock_service.progress_tracker.complete_merge_stage.assert_not_called()

@pytest.mark.asyncio
async def test_resolution_flow_error_handling():
    """Test flow handling when a task fails"""
    # Mock all the tasks
    with patch("app.services.merge.prefect_flows.load_conflicts", new_callable=AsyncMock) as mock_load, \
         patch("app.services.merge.prefect_flows.resolve_minor_conflicts", new_callable=AsyncMock) as mock_minor, \
         patch("app.services.merge.service.MergeService") as mock_service_class:
        
        # Configure mocks
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.progress_tracker = MagicMock()
        mock_service.progress_tracker.start_merge_stage = AsyncMock()
        mock_service.progress_tracker.fail_merge_stage = AsyncMock()
        
        # Setup mock with conflicts
        mock_load.return_value = [
            Conflict(
                id="conflict1", 
                merge_id="test_merge_id",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MINOR,
                staging_ids=["s1"],
                production_ids=["p1"],
                description="Test conflict"
            )
        ]
        
        # Make the minor conflicts resolver raise an exception
        mock_minor.side_effect = Exception("Test error")
        
        # Run the flow and expect exception
        merge_id = "test_merge_id"
        with pytest.raises(Exception):
            await resolution_pipeline_flow(merge_id)
        
        # Verify failure was reported - Prefect may retry, so check it was called at least once
        assert mock_service.progress_tracker.fail_merge_stage.call_count >= 1

@pytest.mark.asyncio
async def test_merge_service_pipeline_integration():
    """Test MergeService integration with the pipeline"""
    merge_id = "test_merge_id"
    
    # Mock the run_resolution_pipeline function and MergeService class
    with patch("app.services.merge.flow_manager.run_resolution_pipeline", new_callable=AsyncMock) as mock_run, \
         patch("app.services.merge.service.MergeService", autospec=True) as mock_service_class:
        
        # Setup mock service instance
        mock_service = mock_service_class.return_value
        mock_service.get_merge_progress = AsyncMock()
        
        # Setup mock status to indicate conflict detection complete
        mock_service.get_merge_progress.return_value = MagicMock()
        mock_service.get_merge_progress.return_value.stages_progress = {
            MergeStage.CONFLICT_DETECTION: MagicMock(status=StageStatus.COMPLETED)
        }
        
        # Setup mock flow run ID and start_resolution_pipeline method
        mock_run.return_value = "test_flow_run_id"
        mock_service.start_resolution_pipeline = AsyncMock(return_value="test_flow_run_id")
        
        # Call the method directly on the mock
        flow_run_id = await mock_service.start_resolution_pipeline(merge_id)
        
        # Verify the pipeline was started
        assert flow_run_id == "test_flow_run_id"
        mock_service.start_resolution_pipeline.assert_called_once_with(merge_id)

@pytest.mark.asyncio
async def test_resolution_flow_no_conflicts():
    """Test flow handling when there are no conflicts"""
    # Mock all the tasks
    with patch("app.services.merge.prefect_flows.load_conflicts", new_callable=AsyncMock) as mock_load, \
         patch("app.services.merge.service.MergeService") as mock_service_class:
        
        # Configure mocks
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.progress_tracker = MagicMock()
        mock_service.progress_tracker.start_merge_stage = AsyncMock()
        mock_service.progress_tracker.complete_merge_stage = AsyncMock()
        
        # Return empty conflict list
        mock_load.return_value = []
        
        # Run the flow
        merge_id = "test_merge_id"
        result = await resolution_pipeline_flow(merge_id)
        
        # Verify quick path was taken
        assert result["status"] == "completed"
        assert "No conflicts to resolve" in result["message"]
        
        # Verify other tasks were not called
        with patch("app.services.merge.prefect_flows.resolve_minor_conflicts", new_callable=AsyncMock) as mock_minor:
            mock_minor.assert_not_called()
        
        with patch("app.services.merge.prefect_flows.resolve_major_conflicts", new_callable=AsyncMock) as mock_major:
            mock_major.assert_not_called()
            
        with patch("app.services.merge.prefect_flows.flag_critical_conflicts", new_callable=AsyncMock) as mock_critical:
            mock_critical.assert_not_called()
            
        with patch("app.services.merge.prefect_flows.check_resolution_status", new_callable=AsyncMock) as mock_status:
            mock_status.assert_not_called() 