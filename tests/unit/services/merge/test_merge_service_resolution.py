import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.merge.service import MergeService
from app.services.merge.resolution_applicator import ResolutionApplicator
from app.schemas.conflicts import (
    Conflict, ConflictType, ConflictSeverity, ResolutionOption,
    ConflictStatus, BatchResolutionItem
)
from app.schemas.graph import Node, Edge
from app.storage.conflicts import ConflictStorageInterface
from app.storage.graph import GraphStorageInterface
from app.services.resolution_history_service import ResolutionHistoryService
from app.schemas.resolution_history import ResolutionHistoryEntry

@pytest.fixture
def mock_conflict_storage():
    """Create a mock conflict storage for testing"""
    mock_storage = AsyncMock(spec=ConflictStorageInterface)
    return mock_storage

@pytest.fixture
def mock_graph_storages():
    """Create mock graph storages for testing"""
    mock_staging = AsyncMock(spec=GraphStorageInterface)
    mock_production = AsyncMock(spec=GraphStorageInterface)
    return mock_staging, mock_production

@pytest.fixture
def mock_resolution_applicator():
    """Create a mock resolution applicator for testing"""
    mock_applicator = AsyncMock(spec=ResolutionApplicator)
    return mock_applicator

@pytest.fixture
def sample_conflict():
    """Create a sample conflict for testing"""
    return Conflict(
        id="conflict-123",
        merge_id="merge-456",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        staging_ids=["node-s1"],
        production_ids=["node-p1"],
        description="Property 'name' has different values",
        resolved=False,
        context={
            "property_name": "name",
            "staging_value": "Alice Smith",
            "production_value": "Alice Jones",
            "entity_type": "Person"
        },
        resolution_options=[
            ResolutionOption(
                id="resolution-789",
                description="Keep staging value: Alice Smith",
                resolution_type="keep_staging",
                resolution_data={"property_name": "name"},
                confidence=0.8
            ),
            ResolutionOption(
                id="resolution-790",
                description="Keep production value: Alice Jones",
                resolution_type="keep_production",
                resolution_data={"property_name": "name"},
                confidence=0.5
            )
        ]
    )

@pytest.fixture
def mock_resolution_history_service():
    service = MagicMock(spec=ResolutionHistoryService)
    
    # Mock store_resolution method
    service.store_resolution = AsyncMock()
    service.store_resolution.return_value = ResolutionHistoryEntry(
        id="history1",
        conflict_id="conflict1",
        merge_id="merge1",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        context={"property_name": "age"},
        resolution_id="opt1",
        resolution_type="keep_staging",
        entity_types=["Person"],
        property_names=["age"],
        applied_by="test_user"
    )
    
    # Mock find_similar_resolutions method
    service.find_similar_resolutions = AsyncMock()
    service.find_similar_resolutions.return_value = [
        {
            "entry": {
                "id": "history1",
                "conflict_id": "conflict1",
                "merge_id": "merge1",
                "conflict_type": "property_value",
                "resolution_type": "keep_staging",
                "resolution_data": {"property_name": "age"},
                "applied_at": "2023-01-01T00:00:00",
                "success": True,
                "context": {"property_name": "age"}
            },
            "similarity_score": 0.9
        }
    ]
    
    return service

@pytest.fixture
def merge_service_with_mocks(mock_conflict_storage, mock_graph_storages, mock_resolution_applicator, mock_resolution_history_service):
    """Create a MergeService with mocked dependencies"""
    mock_staging, mock_production = mock_graph_storages
    
    # Create the service with mocked dependencies
    service = MergeService(
        storage=mock_conflict_storage,
        production_storage=mock_production,
        progress_tracker=AsyncMock()
    )
    
    # Set the resolution history service directly
    service.resolution_history = mock_resolution_history_service
    
    # Add mock for _update_conflict method
    service._update_conflict = AsyncMock()
    
    # Set the resolution applicator directly
    service.resolution_applicator = mock_resolution_applicator
    
    # Mock the get_conflict method
    service.get_conflict = AsyncMock()
    
    # Mock the ResolutionApplicator creation
    with patch("app.services.merge.service.ResolutionApplicator", return_value=mock_resolution_applicator):
        yield service

class TestMergeServiceResolution:
    @pytest.mark.asyncio
    async def test_apply_conflict_resolution_success(
        self, merge_service_with_mocks, mock_conflict_storage, 
        mock_resolution_applicator, sample_conflict
    ):
        """Test successful application of a conflict resolution"""
        # Arrange
        conflict_id = "conflict-123"
        resolution_id = "resolution-789"
        merge_id = "merge-456"
        
        # Mock the get_conflict method directly on the service
        merge_service_with_mocks.get_conflict = AsyncMock(return_value=sample_conflict)
        
        # Mock the resolution applicator to return a successful result
        mock_resolution_applicator.apply_resolution.return_value = {
            "applied": True,
            "conflict_id": conflict_id,
            "resolution_id": resolution_id,
            "verification": {"verified": True},
            "changes": {
                "property": "name",
                "old_value": "Alice Jones",
                "new_value": "Alice Smith",
                "action": "updated_production"
            }
        }
        
        # Act
        result = await merge_service_with_mocks.apply_conflict_resolution(
            merge_id=merge_id,
            conflict_id=conflict_id,
            resolution_id=resolution_id
        )
        
        # Assert
        assert result["applied"] == True
        assert result["conflict_id"] == conflict_id
        assert result["resolution_id"] == resolution_id
        assert result["verification"]["verified"] == True
        
        # Verify the conflict was retrieved
        merge_service_with_mocks.get_conflict.assert_called_once_with(merge_id, conflict_id)
        
        # Check that the conflict was updated with the new status
        merge_service_with_mocks._update_conflict.assert_called_once()
        
        # Verify the resolution applicator was called with the correct parameters
        mock_resolution_applicator.apply_resolution.assert_called_once()
        call_args = mock_resolution_applicator.apply_resolution.call_args[0]
        assert call_args[0].id == conflict_id  # First arg is the conflict
        assert call_args[1].id == resolution_id  # Second arg is the resolution option
    
    @pytest.mark.asyncio
    async def test_apply_conflict_resolution_conflict_not_found(
        self, merge_service_with_mocks, mock_conflict_storage
    ):
        """Test applying resolution to a non-existent conflict"""
        # Arrange
        conflict_id = "nonexistent-conflict"
        resolution_id = "resolution-789"
        merge_id = "merge-456"
        
        # Mock the get_conflict method to return None for the conflict
        merge_service_with_mocks.get_conflict = AsyncMock(return_value=None)
        
        # Act & Assert
        with pytest.raises(ValueError, match=f"Conflict {conflict_id} not found"):
            await merge_service_with_mocks.apply_conflict_resolution(
                merge_id=merge_id,
                conflict_id=conflict_id,
                resolution_id=resolution_id
            )
        
        # Verify the conflict was retrieved
        merge_service_with_mocks.get_conflict.assert_called_once_with(merge_id, conflict_id)
    
    @pytest.mark.asyncio
    async def test_apply_conflict_resolution_resolution_not_found(
        self, merge_service_with_mocks, mock_conflict_storage, sample_conflict
    ):
        """Test applying a non-existent resolution to a conflict"""
        # Arrange
        conflict_id = "conflict-123"
        resolution_id = "nonexistent-resolution"
        merge_id = "merge-456"
        
        # Mock the get_conflict method to return our sample conflict
        merge_service_with_mocks.get_conflict = AsyncMock(return_value=sample_conflict)
        
        # Act & Assert
        with pytest.raises(ValueError, match=f"Resolution option not found for conflict {conflict_id}"):
            await merge_service_with_mocks.apply_conflict_resolution(
                merge_id=merge_id,
                conflict_id=conflict_id,
                resolution_id=resolution_id
            )
        
        # Verify the conflict was retrieved
        merge_service_with_mocks.get_conflict.assert_called_once_with(merge_id, conflict_id)
    
    @pytest.mark.asyncio
    async def test_apply_conflict_resolution_already_resolved(
        self, merge_service_with_mocks, mock_conflict_storage, sample_conflict
    ):
        """Test applying resolution to an already resolved conflict"""
        # Arrange
        conflict_id = "conflict-123"
        resolution_id = "resolution-789"
        merge_id = "merge-456"
        
        # Modify the sample conflict to be already resolved
        resolved_conflict = sample_conflict.model_copy()
        resolved_conflict.resolved = True
        
        # Mock the get_conflict method to return our resolved conflict
        merge_service_with_mocks.get_conflict = AsyncMock(return_value=resolved_conflict)
        
        # Act & Assert
        with pytest.raises(ValueError, match=f"Conflict {conflict_id} is already resolved"):
            await merge_service_with_mocks.apply_conflict_resolution(
                merge_id=merge_id,
                conflict_id=conflict_id,
                resolution_id=resolution_id
            )
        
        # Verify the conflict was retrieved
        merge_service_with_mocks.get_conflict.assert_called_once_with(merge_id, conflict_id)
    
    @pytest.mark.asyncio
    async def test_apply_conflict_resolution_application_failure(
        self, merge_service_with_mocks, mock_conflict_storage, 
        mock_resolution_applicator, sample_conflict
    ):
        """Test handling of resolution application failure"""
        # Arrange
        conflict_id = "conflict-123"
        resolution_id = "resolution-789"
        merge_id = "merge-456"
        
        # Mock the get_conflict method to return our sample conflict
        merge_service_with_mocks.get_conflict = AsyncMock(return_value=sample_conflict)
        
        # Mock the resolution applicator to return a failure result
        mock_resolution_applicator.apply_resolution.return_value = {
            "applied": False,
            "conflict_id": conflict_id,
            "resolution_id": resolution_id,
            "error": "Failed to update production node"
        }
        
        # Act
        result = await merge_service_with_mocks.apply_conflict_resolution(
            merge_id=merge_id,
            conflict_id=conflict_id,
            resolution_id=resolution_id
        )
        
        # Assert
        assert result["applied"] == False
        assert result["conflict_id"] == conflict_id
        assert result["resolution_id"] == resolution_id
        assert "error" in result
        
        # Verify the conflict was retrieved
        merge_service_with_mocks.get_conflict.assert_called_once_with(merge_id, conflict_id)
        
        # Verify the conflict was NOT updated
        merge_service_with_mocks._update_conflict.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_apply_batch_resolutions_success(
        self, merge_service_with_mocks, mock_conflict_storage, 
        mock_resolution_applicator, sample_conflict
    ):
        """Test successful application of batch resolutions"""
        # Arrange
        merge_id = "merge-456"
        resolutions = [
            {"conflict_id": "conflict-123", "resolution_id": "resolution-789"},
            {"conflict_id": "conflict-456", "resolution_id": "resolution-012"}
        ]
        
        # Create a second sample conflict
        second_conflict = sample_conflict.model_copy()
        second_conflict.id = "conflict-456"
        second_conflict.resolution_options[0].id = "resolution-012"
        
        # Mock the apply_conflict_resolution method
        merge_service_with_mocks.apply_conflict_resolution = AsyncMock()
        merge_service_with_mocks.apply_conflict_resolution.side_effect = [
            {
                "applied": True,
                "conflict_id": "conflict-123",
                "resolution_id": "resolution-789",
                "verification": {"verified": True},
                "changes": {"action": "updated_production"}
            },
            {
                "applied": True,
                "conflict_id": "conflict-456",
                "resolution_id": "resolution-012",
                "verification": {"verified": True},
                "changes": {"action": "added_to_production"}
            }
        ]
        
        # Act
        result = await merge_service_with_mocks.apply_batch_resolutions(
            merge_id=merge_id,
            resolutions=resolutions
        )
        
        # Assert
        assert result["total"] == 2
        assert result["success_count"] == 2
        assert result["failure_count"] == 0
        assert len(result["results"]) == 2
        
        # Verify apply_conflict_resolution was called for each resolution
        assert merge_service_with_mocks.apply_conflict_resolution.call_count == 2
        merge_service_with_mocks.apply_conflict_resolution.assert_any_call(
            merge_id=merge_id,
            conflict_id="conflict-123", 
            resolution_id="resolution-789"
        )
        merge_service_with_mocks.apply_conflict_resolution.assert_any_call(
            merge_id=merge_id,
            conflict_id="conflict-456", 
            resolution_id="resolution-012"
        )
    
    @pytest.mark.asyncio
    async def test_apply_batch_resolutions_partial_success(
        self, merge_service_with_mocks, mock_conflict_storage, 
        mock_resolution_applicator, sample_conflict
    ):
        """Test batch resolutions with partial success"""
        # Arrange
        merge_id = "merge-456"
        resolutions = [
            {"conflict_id": "conflict-123", "resolution_id": "resolution-789"},
            {"conflict_id": "conflict-456", "resolution_id": "nonexistent-resolution"}
        ]
        
        # Mock the apply_conflict_resolution method
        merge_service_with_mocks.apply_conflict_resolution = AsyncMock()
        merge_service_with_mocks.apply_conflict_resolution.side_effect = [
            {
                "applied": True,
                "conflict_id": "conflict-123",
                "resolution_id": "resolution-789",
                "verification": {"verified": True},
                "changes": {"action": "updated_production"}
            },
            ValueError("Resolution option not found")
        ]
        
        # Act
        result = await merge_service_with_mocks.apply_batch_resolutions(
            merge_id=merge_id,
            resolutions=resolutions
        )
        
        # Assert
        assert result["total"] == 2
        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert len(result["results"]) == 2
        assert result["results"][0]["applied"] == True
        assert result["results"][1]["applied"] == False
        assert "error" in result["results"][1]
        
        # Verify apply_conflict_resolution was called for each resolution
        assert merge_service_with_mocks.apply_conflict_resolution.call_count == 2
    
    @pytest.mark.asyncio
    async def test_apply_conflict_resolution_stores_history(self, merge_service_with_mocks):
        # Arrange
        service = merge_service_with_mocks
        
        # Mock get_conflict to return a conflict
        sample_conflict = Conflict(
            id="conflict1",
            merge_id="merge1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'age' has different values",
            context={
                "property_name": "age",
                "staging_value": 30,
                "production_value": 32,
                "entity_type": "Person"
            },
            resolution_options=[
                ResolutionOption(
                    id="opt1",
                    description="Keep staging value: 30",
                    resolution_type="keep_staging",
                    resolution_data={"property_name": "age"},
                    confidence=0.5
                )
            ]
        )
        service.get_conflict.return_value = sample_conflict
        
        # Get the real method from MergeService
        real_method = MergeService.apply_conflict_resolution
        
        # Act
        result = await real_method(
            service,
            merge_id="merge1",
            conflict_id="conflict1",
            resolution_id="opt1",
            resolved_by="test_user"
        )
        
        # Assert
        service.get_conflict.assert_called_once_with("merge1", "conflict1")
        service._update_conflict.assert_called_once()
        service.resolution_history.store_resolution.assert_called_once_with(
            conflict=sample_conflict,
            resolution_id="opt1",
            applied_by="test_user",
            merge_id="merge1",
            success=True
        )
    
    
    @pytest.mark.asyncio
    async def test_apply_conflict_resolution_not_found(self, merge_service_with_mocks):
        # Arrange
        service = merge_service_with_mocks
        service.get_conflict.return_value = None
        
        # Get the real method from MergeService
        real_method = MergeService.apply_conflict_resolution
        
        # Act/Assert
        with pytest.raises(ValueError, match="Conflict .* not found"):
            await real_method(
                service,
                merge_id="merge1",
                conflict_id="nonexistent",
                resolution_id="opt1",
                resolved_by="test_user"
            )
    
    