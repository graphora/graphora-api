import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.services.merge.service import MergeService
from app.services.merge.models import MergeStatus, MergeStage

@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker fixture"""
    tracker = AsyncMock()
    tracker.get_all_merge_ids = AsyncMock()
    tracker.get_progress = AsyncMock()
    return tracker

@pytest.fixture
def mock_storage():
    """Mock storage fixture"""
    storage = AsyncMock()
    return storage

@pytest.fixture
def merge_service(mock_progress_tracker, mock_storage):
    """Merge service fixture with mock progress tracker"""
    service = MergeService(
        storage=mock_storage,
        production_storage=mock_storage,
        progress_tracker=mock_progress_tracker
    )
    return service

@pytest.mark.asyncio
async def test_get_all_merges(merge_service, mock_progress_tracker):
    """Test getting all merges"""
    # Mock get_all_merge_ids to return some merge IDs
    mock_progress_tracker.get_all_merge_ids.return_value = ["merge_id_1", "merge_id_2"]
    
    # Mock get_progress to return progress objects
    progress1 = MagicMock()
    progress1.model_dump.return_value = {
        "merge_id": "merge_id_1",
        "overall_status": MergeStatus.COMPLETED.value,
        "current_stage": None
    }
    
    progress2 = MagicMock()
    progress2.model_dump.return_value = {
        "merge_id": "merge_id_2",
        "overall_status": MergeStatus.RUNNING.value,
        "current_stage": MergeStage.MERGE.value
    }
    
    async def mock_get_progress(merge_id):
        if merge_id == "merge_id_1":
            return progress1
        elif merge_id == "merge_id_2":
            return progress2
        return None
    
    mock_progress_tracker.get_progress.side_effect = mock_get_progress
    
    # Get all merges
    merges = await merge_service.get_all_merges()
    
    # Verify methods were called correctly
    mock_progress_tracker.get_all_merge_ids.assert_called_once()
    assert mock_progress_tracker.get_progress.call_count == 2
    
    # Verify merges
    assert len(merges) == 2
    assert merges[0]["merge_id"] == "merge_id_1"
    assert merges[0]["overall_status"] == MergeStatus.COMPLETED.value
    assert merges[1]["merge_id"] == "merge_id_2"
    assert merges[1]["overall_status"] == MergeStatus.RUNNING.value
    assert merges[1]["current_stage"] == MergeStage.MERGE.value

@pytest.mark.asyncio
async def test_cancel_merge(merge_service, mock_progress_tracker):
    """Test cancelling a merge"""
    merge_id = "test_merge_id"
    
    # Mock get_progress to return a running merge
    progress = MagicMock()
    progress.overall_status = MergeStatus.RUNNING
    mock_progress_tracker.get_progress.return_value = progress
    
    # Mock MergeExecutionService
    with patch("app.services.merge.execution_service.MergeExecutionService") as mock_execution_service_class:
        # Set up the mock execution service
        mock_execution_service = AsyncMock()
        mock_execution_service.cancel_merge.return_value = True
        mock_execution_service_class.return_value = mock_execution_service
        
        # Cancel merge
        result = await merge_service.cancel_merge(merge_id)
        
        # Verify methods were called correctly
        mock_progress_tracker.get_progress.assert_called_once_with(merge_id)
        mock_execution_service.cancel_merge.assert_called_once_with(merge_id)
        
        # Verify result
        assert result is True

@pytest.mark.asyncio
async def test_cancel_merge_already_completed(merge_service, mock_progress_tracker):
    """Test cancelling a merge that's already completed"""
    merge_id = "test_merge_id"
    
    # Mock get_progress to return a completed merge
    progress = MagicMock()
    progress.overall_status = MergeStatus.COMPLETED
    mock_progress_tracker.get_progress.return_value = progress
    
    # Cancel merge
    result = await merge_service.cancel_merge(merge_id)
    
    # Verify methods were called correctly
    mock_progress_tracker.get_progress.assert_called_once_with(merge_id)
    
    # Verify result
    assert result is False 