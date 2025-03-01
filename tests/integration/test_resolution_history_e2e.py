import pytest
import os
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.conflicts import ConflictType, ConflictSeverity, Conflict, ResolutionOption
from app.schemas.resolution_history import ResolutionHistoryEntry
from app.services.resolution_history_service import ResolutionHistoryService
from app.dependencies import get_merge_service

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
async def resolution_service():
    """Create a resolution history service for test data setup"""
    service = ResolutionHistoryService()
    
    # Clean up test database before tests
    await service.redis.flushdb()
    
    yield service
    
    # Clean up after tests
    await service.redis.flushdb()

@pytest.fixture
async def test_data(resolution_service):
    """Create test data for the resolution history API tests"""
    # Create test entries
    entries = []
    
    # Create entries for merge1
    for i in range(5):
        conflict = create_test_conflict(f"conflict{i}", "merge1", 
                                       ConflictType.property_conflict if i % 2 == 0 else ConflictType.relationship_conflict)
        
        entry = await resolution_service.store_resolution(
            conflict=conflict,
            resolution_id=f"resolution{i}",
            applied_by="user1" if i % 3 == 0 else "user2",
            merge_id="merge1",
            success=i % 2 == 0,
            feedback="Good resolution" if i % 2 == 0 else "Could be better",
            effectiveness=0.8 if i % 2 == 0 else 0.5
        )
        entries.append(entry)
    
    # Create entries for merge2
    for i in range(3):
        conflict = create_test_conflict(f"conflict{i+5}", "merge2", 
                                       ConflictType.entity_conflict if i % 2 == 0 else ConflictType.property_conflict)
        
        entry = await resolution_service.store_resolution(
            conflict=conflict,
            resolution_id=f"resolution{i+5}",
            applied_by="user3",
            merge_id="merge2",
            success=True,
            feedback=None,
            effectiveness=0.9 if i == 0 else None
        )
        entries.append(entry)
    
    return entries

def create_test_conflict(conflict_id, merge_id, conflict_type):
    """Helper function to create a test conflict"""
    context = {}
    resolution_options = []
    
    if conflict_type == ConflictType.property_conflict:
        context = {
            "entity_id": "entity123",
            "entity_type": "Person",
            "property_name": "name",
            "source_value": "John Smith",
            "target_value": "John A. Smith"
        }
        resolution_options = [
            ResolutionOption(
                id="keep_source",
                type="KEEP_SOURCE",
                description="Keep source value",
                data={"value": "John Smith"}
            ),
            ResolutionOption(
                id="keep_target",
                type="KEEP_TARGET",
                description="Keep target value",
                data={"value": "John A. Smith"}
            )
        ]
    elif conflict_type == ConflictType.relationship_conflict:
        context = {
            "entity_id": "entity123",
            "entity_type": "Person",
            "relationship_type": "WORKS_FOR",
            "target_entity_id": "org123",
            "target_entity_type": "Organization"
        }
        resolution_options = [
            ResolutionOption(
                id="keep_relationship",
                type="KEEP_RELATIONSHIP",
                description="Keep relationship",
                data={}
            ),
            ResolutionOption(
                id="remove_relationship",
                type="REMOVE_RELATIONSHIP",
                description="Remove relationship",
                data={}
            )
        ]
    else:  # entity_conflict
        context = {
            "entity_id": "entity123",
            "entity_type": "Person",
            "duplicate_id": "entity456"
        }
        resolution_options = [
            ResolutionOption(
                id="merge_entities",
                type="MERGE_ENTITIES",
                description="Merge entities",
                data={}
            ),
            ResolutionOption(
                id="keep_separate",
                type="KEEP_SEPARATE",
                description="Keep as separate entities",
                data={}
            )
        ]
    
    return Conflict(
        id=conflict_id,
        merge_id=merge_id,
        type=conflict_type,
        severity=ConflictSeverity.medium,
        context=context,
        resolution_options=resolution_options,
        resolved=False
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
        response = test_client.get("/api/v1/resolutions?conflict_type=property_conflict")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
        
        # Act - Filter by user
        response = test_client.get("/api/v1/resolutions?user=user1")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
        
        # Act - Filter by effectiveness
        response = test_client.get("/api/v1/resolutions?effectiveness=0.8")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
        
        # Act - Filter with multiple criteria
        response = test_client.get("/api/v1/resolutions?conflict_type=property_conflict&user=user1")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
    
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
        assert "property_conflict" in data["by_conflict_type"]
        assert "relationship_conflict" in data["by_conflict_type"]
        assert "entity_conflict" in data["by_conflict_type"]
        assert data["by_conflict_type"]["property_conflict"] == 5
        assert data["by_conflict_type"]["relationship_conflict"] == 3
        
        # Check resolution types
        assert "by_resolution_type" in data
        assert "keep_source" in data["by_resolution_type"]
        assert "keep_target" in data["by_resolution_type"]
        
        # Check entity types
        assert "by_entity_type" in data
        assert "Person" in data["by_entity_type"]
        
        # Check users
        assert "by_user" in data
        assert "user1" in data["by_user"]
        assert "user2" in data["by_user"]
        assert "user3" in data["by_user"]
        
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
    
    @pytest.mark.asyncio
    async def test_submit_resolution_feedback(self, test_client, test_data):
        """Test submitting feedback for a resolution."""
        # Get a resolution ID to update
        response = test_client.get("/api/v1/resolutions?limit=1")
        assert response.status_code == 200
        data = response.json()
        resolution_id = data["items"][0]["id"]
        
        # Submit feedback
        feedback_data = {
            "success": True,
            "feedback": "This resolution was excellent",
            "effectiveness": 0.95
        }
        
        response = test_client.post(
            f"/api/v1/resolutions/{resolution_id}/feedback",
            json=feedback_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] == True
        
        # Verify the feedback was updated
        response = test_client.get(f"/api/v1/resolutions?limit=10")
        assert response.status_code == 200
        data = response.json()
        
        # Find the updated resolution
        updated_resolution = None
        for item in data["items"]:
            if item["id"] == resolution_id:
                updated_resolution = item
                break
        
        assert updated_resolution is not None
        assert updated_resolution["success"] == True
        assert updated_resolution["feedback"] == "This resolution was excellent"
        assert updated_resolution["effectiveness"] == 0.95
    
    @pytest.mark.asyncio
    async def test_submit_resolution_feedback_minimal(self, test_client, test_data):
        """Test submitting minimal feedback (only success field) for a resolution."""
        # Get a resolution ID to update
        response = test_client.get("/api/v1/resolutions?limit=1")
        assert response.status_code == 200
        data = response.json()
        resolution_id = data["items"][0]["id"]
        
        # Submit minimal feedback
        feedback_data = {
            "success": False
        }
        
        response = test_client.post(
            f"/api/v1/resolutions/{resolution_id}/feedback",
            json=feedback_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] == True
        
        # Verify the feedback was updated
        response = test_client.get(f"/api/v1/resolutions?limit=10")
        assert response.status_code == 200
        data = response.json()
        
        # Find the updated resolution
        updated_resolution = None
        for item in data["items"]:
            if item["id"] == resolution_id:
                updated_resolution = item
                break
        
        assert updated_resolution is not None
        assert updated_resolution["success"] == False 