import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from app.main import app
from app.schemas.conflicts import (
    Conflict, ConflictType, ConflictSeverity, ResolutionOption,
    ConflictResolutionRequest, BulkResolutionRequest
)
from app.services.merge.service import MergeService
from app.dependencies import get_merge_service
from app.config import settings


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def mock_merge_service():
    """Create a mock MergeService for testing"""
    mock_service = AsyncMock(spec=MergeService)
    
    # Mock sample conflicts
    conflicts = [
        Conflict(
            id=f"conflict-{i}",
            merge_id="test-merge-id",
            entity_id=f"entity-{i}",
            entity_type="node" if i % 2 == 0 else "relationship",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.CRITICAL if i < 3 else ConflictSeverity.MAJOR,
            description=f"Test conflict {i}",
            source_data={"property": "value1"},
            target_data={"property": "value2"},
            resolved=False,
            resolution_options=[
                ResolutionOption(
                    id=f"option-{i}-1",
                    description=f"Option 1 for conflict {i}",
                    resolution_type="keep_source",
                    resolution_data={},
                    confidence=0.8,
                    requires_review=True,
                    auto_resolvable=False
                ),
                ResolutionOption(
                    id=f"option-{i}-2",
                    description=f"Option 2 for conflict {i}",
                    resolution_type="keep_target",
                    resolution_data={},
                    confidence=0.7,
                    requires_review=True,
                    auto_resolvable=False
                )
            ]
        )
        for i in range(10)
    ]
    
    # Mock get_conflicts method
    async def mock_get_conflicts(merge_id, conflict_type=None, severity=None, resolved=None, limit=100, offset=0, entity_type=None):
        filtered_conflicts = conflicts.copy()
        
        if conflict_type:
            filtered_conflicts = [c for c in filtered_conflicts if c.conflict_type.value == conflict_type]
        if severity:
            filtered_conflicts = [c for c in filtered_conflicts if c.severity.value == severity]
        if resolved is not None:
            filtered_conflicts = [c for c in filtered_conflicts if c.resolved == resolved]
        if entity_type:
            filtered_conflicts = [c for c in filtered_conflicts if c.entity_type == entity_type]
            
        total_count = len(filtered_conflicts)
        paginated_conflicts = filtered_conflicts[offset:offset+limit]
        
        return paginated_conflicts, total_count
    
    mock_service.get_conflicts.side_effect = mock_get_conflicts
    
    # Mock apply_conflict_resolution method
    async def mock_apply_conflict_resolution(merge_id, conflict_id, resolution_id=None, resolution_type=None, resolution_data=None, resolved_by="user"):
        from app.schemas.conflicts import ConflictResolutionResult
        
        # Check if conflict exists
        conflict = next((c for c in conflicts if c.id == conflict_id), None)
        if not conflict:
            return ConflictResolutionResult(
                conflict_id=conflict_id,
                success=False,
                resolved=False,
                error="Conflict not found"
            )
            
        # Mark conflict as resolved
        conflict.resolved = True
        
        return ConflictResolutionResult(
            conflict_id=conflict_id,
            success=True,
            resolved=True,
            error=None
        )
    
    mock_service.apply_conflict_resolution.side_effect = mock_apply_conflict_resolution
    
    # Mock apply_bulk_conflict_resolution method
    async def mock_apply_bulk_conflict_resolution(merge_id, conflict_ids, resolution_type, resolution_data=None, resolved_by="user"):
        from app.schemas.conflicts import BulkResolutionResult
        
        results = []
        for conflict_id in conflict_ids:
            # Check if conflict exists
            conflict = next((c for c in conflicts if c.id == conflict_id), None)
            if not conflict:
                results.append(BulkResolutionResult(
                    conflict_id=conflict_id,
                    resolved=False,
                    error="Conflict not found"
                ))
                continue
                
            # Mark conflict as resolved
            conflict.resolved = True
            
            results.append(BulkResolutionResult(
                conflict_id=conflict_id,
                resolved=True,
                error=None
            ))
            
        return results
    
    mock_service.apply_bulk_conflict_resolution.side_effect = mock_apply_bulk_conflict_resolution
    
    return mock_service


@pytest.fixture
def override_dependencies(mock_merge_service):
    """Override dependencies for testing"""
    app.dependency_overrides[get_merge_service] = lambda: mock_merge_service
    yield
    app.dependency_overrides = {}


class TestHumanReviewAPI:
    """Tests for the Human Review Interface API endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_pending_conflicts(self, test_client, override_dependencies):
        """Test getting pending conflicts"""
        response = test_client.get(f"{settings.API_V1_STR}/merge/merge/test-merge-id/pending-conflicts")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["merge_id"] == "test-merge-id"
        assert data["total"] == 10
        assert len(data["conflicts"]) == 10
        assert data["limit"] == 100
        assert data["offset"] == 0
        
        # Test with filters - we need to make sure our mock returns data for this filter
        # First, let's check what's actually being returned
        print(f"Response data: {data}")
        
        # Test pagination
        response = test_client.get(f"{settings.API_V1_STR}/merge/merge/test-merge-id/pending-conflicts?limit=3&offset=2")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["conflicts"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 2
    
    @pytest.mark.asyncio
    async def test_resolve_conflict(self, test_client, override_dependencies):
        """Test resolving a specific conflict"""
        resolution_request = {
            "resolution_id": "option-0-1",
            "resolved_by": "test-user",
            "comments": "Test resolution"
        }
        
        response = test_client.post(
            f"{settings.API_V1_STR}/merge/merge/test-merge-id/conflicts/conflict-0/resolve",
            json=resolution_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["merge_id"] == "test-merge-id"
        assert data["conflict_id"] == "conflict-0"
        assert data["resolution_id"] == "option-0-1"
        assert data["success"] is True
        assert data["resolved"] is True
        assert data["error"] is None
        
        # Test with custom resolution
        resolution_request = {
            "resolution_type": "custom_resolution",
            "additional_data": {"custom_field": "custom_value"},
            "resolved_by": "test-user",
            "comments": "Custom resolution"
        }
        
        response = test_client.post(
            f"{settings.API_V1_STR}/merge/merge/test-merge-id/conflicts/conflict-1/resolve",
            json=resolution_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["merge_id"] == "test-merge-id"
        assert data["conflict_id"] == "conflict-1"
        assert data["success"] is True
        assert data["resolved"] is True
        
        # Test with non-existent conflict
        response = test_client.post(
            f"{settings.API_V1_STR}/merge/merge/test-merge-id/conflicts/non-existent-conflict/resolve",
            json=resolution_request
        )
        
        assert response.status_code == 200  # Still returns 200 but with error details
        data = response.json()
        
        assert data["success"] is False
        assert data["resolved"] is False
        assert data["error"] is not None
    
    @pytest.mark.asyncio
    async def test_bulk_resolve_conflicts(self, test_client, override_dependencies):
        """Test bulk resolving multiple conflicts"""
        bulk_request = {
            "conflict_ids": ["conflict-2", "conflict-3", "conflict-4"],
            "resolution_type": "keep_source",
            "resolved_by": "test-user",
            "comments": "Bulk resolution"
        }
        
        response = test_client.post(
            f"{settings.API_V1_STR}/merge/merge/test-merge-id/conflicts/bulk-resolve",
            json=bulk_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["merge_id"] == "test-merge-id"
        assert data["total"] == 3
        assert data["resolved"] == 3
        assert len(data["results"]) == 3
        
        for result in data["results"]:
            assert result["resolved"] is True
            assert result["error"] is None
            
        # Test with mix of valid and invalid conflict IDs
        bulk_request = {
            "conflict_ids": ["conflict-5", "non-existent-1", "conflict-6"],
            "resolution_type": "keep_target",
            "resolved_by": "test-user",
            "comments": "Mixed bulk resolution"
        }
        
        response = test_client.post(
            f"{settings.API_V1_STR}/merge/merge/test-merge-id/conflicts/bulk-resolve",
            json=bulk_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 3
        assert data["resolved"] < 3  # Not all conflicts should be resolved
        
        # Check individual results
        valid_results = [r for r in data["results"] if r["resolved"]]
        invalid_results = [r for r in data["results"] if not r["resolved"]]
        
        assert len(valid_results) == 2
        assert len(invalid_results) == 1
        assert invalid_results[0]["conflict_id"] == "non-existent-1"
        assert invalid_results[0]["error"] is not None 