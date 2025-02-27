"""Integration tests for auto-resolution API"""
import pytest
from fastapi.testclient import TestClient
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

from app.main import app
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.services.merge.progress import ProgressTracker
from app.services.merge.service import MergeService
from app.dependencies import get_merge_service
import os

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)

@pytest.fixture
def test_merge_id():
    """Generate a test merge ID"""
    return str(uuid.uuid4())

@pytest.fixture
def mock_conflicts(test_merge_id):
    """Create mock conflicts for testing"""
    # Create 3 minor conflicts that should be auto-resolved
    minor_conflicts = []
    for i in range(3):
        minor_conflicts.append(
            Conflict(
                id=f"conflict-minor-{i}",
                merge_id=test_merge_id,
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MINOR,
                entity_id=f"entity-{i}",
                entity_type="Person",
                property_name="name",
                staging_value=f"John Doe {i}",
                production_value=f"john doe {i}",
                description=f"Case difference in name property for Person {i}",
                context={"entity_type": "Person"},
                resolution_options=[
                    ResolutionOption(
                        id=f"option-minor-{i}-1",
                        description="Keep staging value",
                        resolution_type="keep_staging",
                        confidence=0.95,
                        auto_resolvable=True,
                        requires_review=False,
                        resolution_data={}
                    ),
                    ResolutionOption(
                        id=f"option-minor-{i}-2",
                        description="Keep production value",
                        resolution_type="keep_production",
                        confidence=0.85,
                        auto_resolvable=True,
                        requires_review=False,
                        resolution_data={}
                    )
                ]
            )
        )
    
    # Create 2 major conflicts that should not be auto-resolved
    major_conflicts = []
    for i in range(2):
        major_conflicts.append(
            Conflict(
                id=f"conflict-major-{i}",
                merge_id=test_merge_id,
                conflict_type=ConflictType.ENTITY_MATCH,
                severity=ConflictSeverity.MAJOR,
                entity_id=f"entity-major-{i}",
                entity_type="Organization",
                description=f"Multiple potential matches for Organization {i}",
                context={"entity_type": "Organization"},
                resolution_options=[
                    ResolutionOption(
                        id=f"option-major-{i}",
                        description="Match with specific entity",
                        resolution_type="match_entity",
                        confidence=0.6,
                        auto_resolvable=False,
                        requires_review=True,
                        resolution_data={"target_entity_id": f"prod-entity-{i}"}
                    )
                ]
            )
        )
    
    return minor_conflicts + major_conflicts

class TestAutoResolutionIntegration:
    """Integration tests for auto-resolution functionality"""
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TESTS"),
        reason="Integration tests are skipped by default"
    )
    async def test_auto_resolution_api(self, test_client, test_merge_id, mock_conflicts):
        """Test the auto-resolution API endpoint"""
        # Create a mock response for auto_resolve_conflicts
        mock_response = {
            "total": 5,
            "auto_resolved": 3,
            "manual_required": 2,
            "by_type": {
                "PROPERTY_VALUE": 3
            }
        }
        
        # Use patch to mock the auto_resolve_conflicts method
        with patch('app.services.merge.service.MergeService.auto_resolve_conflicts', 
                  new_callable=AsyncMock, return_value=mock_response):
            
            # Call the auto-resolution API
            response = test_client.post(f"/api/v1/merge/{test_merge_id}/auto-resolve")
            
            # Print the response data for debugging
            print(f"Response data: {response.json()}")
            
            # Assert the response
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 5
            assert data["auto_resolved"] == 3
            assert data["manual_required"] == 2
            assert "by_type" in data 