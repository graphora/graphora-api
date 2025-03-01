import pytest
import os
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.conflicts import ConflictType, ConflictSeverity, Conflict, ResolutionOption
from app.schemas.resolution_history import ResolutionHistoryEntry
from app.services.resolution_history_service import ResolutionHistoryService

# Skip these tests if E2E_TESTS env var is not set
pytestmark = pytest.mark.skipif(
    "E2E_TESTS" not in os.environ,
    reason="End-to-end tests are skipped by default"
)

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def resolution_service():
    """Create a resolution history service for test data setup"""
    service = ResolutionHistoryService()
    
    # Clean up test database before tests
    service.redis.flushdb()
    
    yield service
    
    # Clean up after tests
    service.redis.flushdb()

@pytest.fixture
async def test_data(resolution_service):
    """Create test data for the resolution history API tests"""
    # Create test entries
    entries = []
    
    # Create entries for merge1
    for i in range(5):
        entry = await resolution_service.store_resolution(
            conflict=create_test_conflict(f"conflict{i}", "merge1", ConflictType.PROPERTY_VALUE),
            resolution_id="opt1",
            applied_by=f"user{i % 2 + 1}",  # Alternate between user1 and user2
            merge_id="merge1",
            success=i % 3 == 0,  # Every third entry is successful
            feedback=f"Feedback for resolution {i}" if i % 2 == 0 else None,
            effectiveness=0.5 + (i / 10.0)  # Effectiveness from 0.5 to 0.9
        )
        entries.append(entry)
    
    # Create entries for merge2
    for i in range(3):
        entry = await resolution_service.store_resolution(
            conflict=create_test_conflict(f"conflict{i+5}", "merge2", ConflictType.RELATIONSHIP_TYPE),
            resolution_id="opt2",
            applied_by="user1",
            merge_id="merge2",
            success=True,
            effectiveness=0.7
        )
        entries.append(entry)
    
    return entries

def create_test_conflict(conflict_id, merge_id, conflict_type):
    """Helper function to create a test conflict"""
    if conflict_type == ConflictType.PROPERTY_VALUE:
        return Conflict(
            id=conflict_id,
            merge_id=merge_id,
            conflict_type=conflict_type,
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
                ),
                ResolutionOption(
                    id="opt2",
                    description="Keep production value: 32",
                    resolution_type="keep_production",
                    resolution_data={"property_name": "age"},
                    confidence=0.5
                )
            ]
        )
    else:
        return Conflict(
            id=conflict_id,
            merge_id=merge_id,
            conflict_type=conflict_type,
            severity=ConflictSeverity.MINOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Relationship type is different",
            context={
                "staging_type": "WORKS_FOR",
                "production_type": "EMPLOYED_BY",
                "entity_type": "Person"
            },
            resolution_options=[
                ResolutionOption(
                    id="opt1",
                    description="Keep staging type: WORKS_FOR",
                    resolution_type="keep_staging",
                    resolution_data={"relationship_type": "WORKS_FOR"},
                    confidence=0.5
                ),
                ResolutionOption(
                    id="opt2",
                    description="Keep production type: EMPLOYED_BY",
                    resolution_type="keep_production",
                    resolution_data={"relationship_type": "EMPLOYED_BY"},
                    confidence=0.5
                )
            ]
        )

class TestResolutionHistoryE2E:
    """End-to-end tests for the resolution history API"""
    
    @pytest.mark.asyncio
    async def test_get_resolutions_by_merge_id(self, test_client, test_data):
        """Test retrieving resolutions by merge ID"""
        # Act
        response = test_client.get("/api/v1/resolutions/merge1")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5
        
        # Check that all items have the correct merge ID
        for item in data["items"]:
            assert item["merge_id"] == "merge1"
    
    @pytest.mark.asyncio
    async def test_filter_resolutions(self, test_client, test_data):
        """Test filtering resolutions"""
        # Act - Filter by conflict type
        response = test_client.get("/api/v1/resolutions?conflict_type=PROPERTY_VALUE")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        
        # Act - Filter by user
        response = test_client.get("/api/v1/resolutions?user=user1")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 3  # At least 3 entries for user1
        
        # Act - Filter by effectiveness
        response = test_client.get("/api/v1/resolutions?effectiveness=0.7")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 2  # At least 2 entries with effectiveness >= 0.7
        
        # Act - Filter with multiple criteria
        response = test_client.get(
            "/api/v1/resolutions?conflict_type=RELATIONSHIP_TYPE&user=user1&effectiveness=0.7"
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3  # All 3 entries for merge2
    
    @pytest.mark.asyncio
    async def test_get_resolution_stats(self, test_client, test_data):
        """Test retrieving resolution statistics"""
        # Act
        response = test_client.get("/api/v1/resolutions/stats")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_resolutions"] == 8  # 5 for merge1 + 3 for merge2
        assert "by_conflict_type" in data
        assert "property_value" in data["by_conflict_type"]
        assert "relationship_type" in data["by_conflict_type"]
        assert data["by_conflict_type"]["property_value"] == 5
        assert data["by_conflict_type"]["relationship_type"] == 3
        
        # Check resolution types
        assert "by_resolution_type" in data
        assert "keep_staging" in data["by_resolution_type"]
        assert "keep_production" in data["by_resolution_type"]
        
        # Check entity types
        assert "by_entity_type" in data
        assert "Person" in data["by_entity_type"]
        
        # Check users
        assert "by_user" in data
        assert "user1" in data["by_user"]
        assert "user2" in data["by_user"]
        
        # Check time distribution
        assert "time_distribution" in data
        assert len(data["time_distribution"]) > 0
    
    @pytest.mark.asyncio
    async def test_pagination_and_sorting(self, test_client, test_data):
        """Test pagination and sorting of resolutions"""
        # Act - First page
        response = test_client.get("/api/v1/resolutions?limit=3&offset=0&sort_by=applied_at&sort_order=desc")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 8
        assert len(data["items"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 0
        
        # Remember the first page items
        first_page_ids = [item["id"] for item in data["items"]]
        
        # Act - Second page
        response = test_client.get("/api/v1/resolutions?limit=3&offset=3&sort_by=applied_at&sort_order=desc")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 8
        assert len(data["items"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 3
        
        # Check that second page items are different from first page
        second_page_ids = [item["id"] for item in data["items"]]
        assert not any(id in first_page_ids for id in second_page_ids)
        
        # Act - Sort by effectiveness
        response = test_client.get("/api/v1/resolutions?sort_by=effectiveness&sort_order=desc")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check that items are sorted by effectiveness
        effectiveness_values = [item["effectiveness"] for item in data["items"]]
        assert all(effectiveness_values[i] >= effectiveness_values[i+1] for i in range(len(effectiveness_values)-1)) 