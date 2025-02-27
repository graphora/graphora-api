import pytest
import os
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.services.merge.service import MergeService
from app.services.resolution_history_service import ResolutionHistoryService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.resolution_history import ResolutionHistoryEntry
from app.config import settings

# Skip these tests if E2E_TESTS env var is not set
pytestmark = pytest.mark.skipif(
    "E2E_TESTS" not in os.environ,
    reason="End-to-end tests are skipped by default"
)

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_merge_service():
    with patch("app.api.merge.get_merge_service") as mock_get_service:
        service = MagicMock(spec=MergeService)
        
        # Set up the resolution_history attribute with a real service
        service.resolution_history = ResolutionHistoryService()
        
        # Use test Redis database
        test_redis_url = settings.REDIS_URL.replace("db=0", "db=15")
        service.resolution_history.redis = service.resolution_history.redis.from_url(test_redis_url)
        
        # Clean up test database before tests
        service.resolution_history.redis.flushdb()
        
        # Set up mock methods
        service.get_conflict.return_value = Conflict(
            id="conflict_e2e",
            merge_id="merge_e2e",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'name' has different values",
            context={
                "property_name": "name",
                "staging_value": "John Smith",
                "production_value": "John Doe",
                "entity_type": "Person"
            },
            resolution_options=[
                ResolutionOption(
                    id="opt_e2e_1",
                    description="Keep staging value: John Smith",
                    resolution_type="keep_staging",
                    resolution_data={"property_name": "name"},
                    confidence=0.8
                ),
                ResolutionOption(
                    id="opt_e2e_2",
                    description="Keep production value: John Doe",
                    resolution_type="keep_production",
                    resolution_data={"property_name": "name"},
                    confidence=0.5
                )
            ]
        )
        
        # Mock apply_conflict_resolution to call the real resolution_history.store_resolution
        async def mock_apply_resolution(merge_id, conflict_id, resolution_id, **kwargs):
            conflict = await service.get_conflict(merge_id, conflict_id)
            resolution = next((r for r in conflict.resolution_options if r.id == resolution_id), None)
            
            # Store in resolution history
            await service.resolution_history.store_resolution(
                conflict=conflict,
                resolution_id=resolution_id,
                applied_by=kwargs.get("resolved_by", "e2e_test_user"),
                merge_id=merge_id,
                success=True
            )
            
            return {
                "applied": True,
                "conflict_id": conflict_id,
                "resolution_id": resolution_id,
                "verification": {"verified": True},
                "changes": {
                    "property": "name",
                    "old_value": "John Doe",
                    "new_value": "John Smith",
                    "action": "updated_production"
                }
            }
        
        service.apply_conflict_resolution = mock_apply_resolution
        
        # Pass through to the real resolution_history service for these methods
        service.get_resolution_suggestions.side_effect = lambda merge_id, conflict_id: service.resolution_history.find_similar_resolutions(service.get_conflict(merge_id, conflict_id))
        
        mock_get_service.return_value = service
        yield service
        
        # Clean up after tests
        service.resolution_history.redis.flushdb()

class TestResolutionHistoryE2E:
    @pytest.mark.asyncio
    async def test_complete_resolution_workflow(self, test_client, mock_merge_service):
        """Test the complete workflow from resolution to feedback"""
        # Step 1: Apply a resolution to a conflict
        resolution_response = test_client.post(
            "/api/v1/merge/merge_e2e/conflicts/conflict_e2e/resolve",
            json={
                "resolution_id": "opt_e2e_1",
                "resolved_by": "e2e_test_user"
            }
        )
        
        assert resolution_response.status_code == 200
        resolution_data = resolution_response.json()
        assert resolution_data["applied"] is True
        
        # Step 2: Get resolution history for the merge
        history_response = test_client.get("/api/v1/resolution/history?merge_id=merge_e2e")
        
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert len(history_data) == 1
        assert history_data[0]["conflict_id"] == "conflict_e2e"
        assert history_data[0]["resolution_type"] == "keep_staging"
        
        # Store the resolution ID for later use
        resolution_id = history_data[0]["id"]
        
        # Step 3: Get resolution suggestions for a similar conflict
        suggestions_response = test_client.get(
            f"/api/v1/conflicts/merge_e2e/conflict_e2e/suggestions"
        )
        
        assert suggestions_response.status_code == 200
        suggestions_data = suggestions_response.json()
        assert len(suggestions_data) > 0
        assert suggestions_data[0]["resolution_type"] == "keep_staging"
        assert suggestions_data[0]["similarity_score"] > 0.5
        
        # Step 4: Provide feedback on the resolution
        feedback_response = test_client.post(
            f"/api/v1/resolution/{resolution_id}/feedback?success=true",
            json={"feedback": "This resolution worked perfectly"}
        )
        
        assert feedback_response.status_code == 200
        feedback_data = feedback_response.json()
        assert feedback_data["updated"] is True
        
        # Step 5: Get resolution statistics
        stats_response = test_client.get("/api/v1/resolution/stats")
        
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        assert stats_data["total_resolutions"] == 1
        assert stats_data["success_count"] == 1
        assert stats_data["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_multiple_resolutions_and_stats(self, test_client, mock_merge_service):
        """Test storing multiple resolutions and checking statistics"""
        # Apply multiple resolutions
        for i in range(5):
            # Alternate between resolution options
            resolution_id = "opt_e2e_1" if i % 2 == 0 else "opt_e2e_2"
            
            resolution_response = test_client.post(
                f"/api/v1/merge/merge_e2e_{i}/conflicts/conflict_e2e/resolve",
                json={
                    "resolution_id": resolution_id,
                    "resolved_by": f"e2e_test_user_{i}"
                }
            )
            
            assert resolution_response.status_code == 200
        
        # Get resolution history
        history_response = test_client.get("/api/v1/resolution/history")
        
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert len(history_data) == 5
        
        # Get resolution statistics
        stats_response = test_client.get("/api/v1/resolution/stats")
        
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        assert stats_data["total_resolutions"] == 5
        assert stats_data["by_conflict_type"]["property_value"] == 5
        
        # Check that we have both types of resolutions
        keep_staging_count = sum(1 for entry in history_data if entry["resolution_type"] == "keep_staging")
        keep_production_count = sum(1 for entry in history_data if entry["resolution_type"] == "keep_production")
        
        assert keep_staging_count == 3  # For i=0,2,4
        assert keep_production_count == 2  # For i=1,3 