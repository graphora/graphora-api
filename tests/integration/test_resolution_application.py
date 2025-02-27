import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app
from app.schemas.conflicts import (
    Conflict, ConflictType, ConflictSeverity, ResolutionOption,
    ResolutionRequest, BatchResolutionItem, BatchResolutionRequest
)
from app.services.merge.service import MergeService
from app.services.merge.resolution_applicator import ResolutionApplicator
from app.config import settings
from app.dependencies import get_merge_service

client = TestClient(app)

@pytest.fixture
def mock_merge_service():
    """Create a mock merge service for testing"""
    # Create a mock service with the required methods
    mock_service = AsyncMock(spec=MergeService)
    
    # Add the get_resolution_option method to the mock
    mock_service.get_resolution_option = AsyncMock()
    
    # Create a function that returns our mock service
    async def override_get_merge_service():
        return mock_service
    
    # Store the original dependency
    original = app.dependency_overrides.copy()
    
    # Override the dependency
    app.dependency_overrides[get_merge_service] = override_get_merge_service
    
    yield mock_service
    
    # Restore original dependencies
    app.dependency_overrides = original

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
        context={
            "property_name": "name",
            "staging_value": "Alice Smith",
            "production_value": "Alice Jones",
            "entity_type": "Person"
        }
    )

@pytest.fixture
def sample_resolution_option():
    """Create a sample resolution option for testing"""
    return ResolutionOption(
        id="resolution-789",
        description="Keep staging value: Alice Smith",
        resolution_type="keep_staging",
        resolution_data={"property_name": "name"},
        confidence=0.8
    )

class TestResolutionApplication:
    def test_resolve_conflict_success(self, mock_merge_service, sample_conflict, sample_resolution_option):
        """Test successful resolution of a conflict"""
        # Arrange
        merge_id = "merge-456"
        conflict_id = "conflict-123"
        resolution_request = ResolutionRequest(resolution_id="resolution-789")
        
        # Mock the service methods
        mock_merge_service.get_conflict.return_value = sample_conflict
        mock_merge_service.get_resolution_option.return_value = sample_resolution_option
        mock_merge_service.apply_conflict_resolution.return_value = {
            "applied": True,
            "conflict_id": conflict_id,
            "resolution_id": "resolution-789",
            "verification": {"verified": True},
            "changes": {
                "property": "name",
                "old_value": "Alice Jones",
                "new_value": "Alice Smith",
                "action": "updated_production"
            }
        }
        
        # Act
        response = client.post(
            f"{settings.API_V1_STR}/merge/conflicts/{merge_id}/{conflict_id}/resolve",
            json={"resolution_id": "resolution-789"}
        )
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["applied"] == True
        assert result["conflict_id"] == conflict_id
        assert result["resolution_id"] == "resolution-789"
        assert result["verification"]["verified"] == True
        assert result["changes"]["action"] == "updated_production"
        
        # Verify service method was called correctly
        mock_merge_service.apply_conflict_resolution.assert_called_once_with(
            conflict_id=conflict_id,
            resolution_id="resolution-789"
        )
    
    def test_resolve_conflict_not_found(self, mock_merge_service):
        """Test resolution of a non-existent conflict"""
        # Arrange
        merge_id = "merge-456"
        conflict_id = "nonexistent-conflict"
        resolution_request = ResolutionRequest(resolution_id="resolution-789")
        
        # Mock the service to raise ValueError for non-existent conflict
        mock_merge_service.apply_conflict_resolution.side_effect = ValueError("Conflict not found")
        
        # Act
        response = client.post(
            f"{settings.API_V1_STR}/merge/conflicts/{merge_id}/{conflict_id}/resolve",
            json={"resolution_id": "resolution-789"}
        )
        
        # Assert
        assert response.status_code == 404
        result = response.json()
        assert "detail" in result
        assert "not found" in result["detail"].lower()
    
    def test_resolve_conflict_server_error(self, mock_merge_service, sample_conflict, sample_resolution_option):
        """Test server error during conflict resolution"""
        # Arrange
        merge_id = "merge-456"
        conflict_id = "conflict-123"
        resolution_request = ResolutionRequest(resolution_id="resolution-789")
        
        # Mock the service methods
        mock_merge_service.get_conflict.return_value = sample_conflict
        mock_merge_service.get_resolution_option.return_value = sample_resolution_option
        mock_merge_service.apply_conflict_resolution.side_effect = Exception("Database error")
        
        # Act
        response = client.post(
            f"{settings.API_V1_STR}/merge/conflicts/{merge_id}/{conflict_id}/resolve",
            json={"resolution_id": "resolution-789"}
        )
        
        # Assert
        assert response.status_code == 500
        result = response.json()
        assert "detail" in result
        assert "database error" in result["detail"].lower()
    
    def test_batch_resolve_conflicts_success(self, mock_merge_service):
        """Test successful batch resolution of conflicts"""
        # Arrange
        merge_id = "merge-456"
        batch_request = BatchResolutionRequest(
            resolutions=[
                BatchResolutionItem(conflict_id="conflict-123", resolution_id="resolution-789"),
                BatchResolutionItem(conflict_id="conflict-456", resolution_id="resolution-012")
            ]
        )
        
        # Mock the service method
        mock_merge_service.apply_batch_resolutions.return_value = {
            "total": 2,
            "success_count": 2,
            "failure_count": 0,
            "results": [
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
        }
        
        # Act
        response = client.post(
            f"{settings.API_V1_STR}/merge/conflicts/{merge_id}/batch-resolve",
            json=batch_request.model_dump()
        )
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["total"] == 2
        assert result["success_count"] == 2
        assert result["failure_count"] == 0
        assert len(result["results"]) == 2
        
        # Verify service method was called correctly
        mock_merge_service.apply_batch_resolutions.assert_called_once_with(
            merge_id=merge_id,
            resolutions=batch_request.resolutions
        )
    
    def test_batch_resolve_conflicts_partial_success(self, mock_merge_service):
        """Test batch resolution with some failures"""
        # Arrange
        merge_id = "merge-456"
        batch_request = BatchResolutionRequest(
            resolutions=[
                BatchResolutionItem(conflict_id="conflict-123", resolution_id="resolution-789"),
                BatchResolutionItem(conflict_id="conflict-456", resolution_id="resolution-012")
            ]
        )
        
        # Mock the service method
        mock_merge_service.apply_batch_resolutions.return_value = {
            "total": 2,
            "success_count": 1,
            "failure_count": 1,
            "results": [
                {
                    "applied": True,
                    "conflict_id": "conflict-123",
                    "resolution_id": "resolution-789",
                    "verification": {"verified": True},
                    "changes": {"action": "updated_production"}
                },
                {
                    "applied": False,
                    "conflict_id": "conflict-456",
                    "resolution_id": "resolution-012",
                    "error": "Conflict not found"
                }
            ]
        }
        
        # Act
        response = client.post(
            f"{settings.API_V1_STR}/merge/conflicts/{merge_id}/batch-resolve",
            json=batch_request.model_dump()
        )
        
        # Assert
        assert response.status_code == 200  # Still 200 for partial success
        result = response.json()
        assert result["total"] == 2
        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert len(result["results"]) == 2
        assert result["results"][0]["applied"] == True
        assert result["results"][1]["applied"] == False
        assert "error" in result["results"][1]
    
    def test_batch_resolve_conflicts_server_error(self, mock_merge_service):
        """Test server error during batch resolution"""
        # Arrange
        merge_id = "merge-456"
        batch_request = BatchResolutionRequest(
            resolutions=[
                BatchResolutionItem(conflict_id="conflict-123", resolution_id="resolution-789"),
                BatchResolutionItem(conflict_id="conflict-456", resolution_id="resolution-012")
            ]
        )
        
        # Mock the service method to raise an exception
        mock_merge_service.apply_batch_resolutions.side_effect = Exception("Database connection error")
        
        # Act
        response = client.post(
            f"{settings.API_V1_STR}/merge/conflicts/{merge_id}/batch-resolve",
            json=batch_request.model_dump()
        )
        
        # Assert
        assert response.status_code == 500
        result = response.json()
        assert "detail" in result
        assert "database connection error" in result["detail"].lower() 