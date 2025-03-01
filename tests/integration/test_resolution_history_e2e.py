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
from unittest.mock import MagicMock

# Skip these tests if E2E_TESTS env var is not set
pytestmark = pytest.mark.skipif(
    "E2E_TESTS" not in os.environ,
    reason="End-to-end tests are skipped by default"
)

# Global variable to store test entries
test_entries = []

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
    service.redis.flushdb()
    
    yield service
    
    # Clean up after tests
    service.redis.flushdb()

@pytest.fixture
def mock_merge_service():
    """Create a mock merge service with the MockResolutionHistoryService"""
    # Create a mock service
    mock_service = MagicMock()
    
    # Set the resolution_history attribute to our mock implementation
    mock_service.resolution_history = MockResolutionHistoryService()
    
    # Override the dependency
    original = app.dependency_overrides.copy()
    app.dependency_overrides[get_merge_service] = lambda: mock_service
    
    yield mock_service
    
    # Restore original dependencies
    app.dependency_overrides = original

@pytest.fixture
async def test_data():
    """Create test data for the resolution history API tests"""
    # Clear existing test entries
    global test_entries
    test_entries = []
    
    # Create entries for merge1
    for i in range(5):
        conflict = create_test_conflict(f"conflict{i}", "merge1", 
                                       ConflictType.PROPERTY_VALUE if i % 2 == 0 else ConflictType.RELATIONSHIP_TYPE)
        
        # Use the appropriate resolution ID based on the conflict type
        if conflict.conflict_type == ConflictType.PROPERTY_VALUE:
            resolution_id = "keep_source" if i % 2 == 0 else "keep_target"
        elif conflict.conflict_type == ConflictType.RELATIONSHIP_TYPE:
            resolution_id = "keep_relationship" if i % 2 == 0 else "remove_relationship"
        else:
            resolution_id = "merge_entities"
        
        entry = ResolutionHistoryEntry(
            id=f"entry{i}",
            conflict_id=conflict.id,
            merge_id=conflict.merge_id,
            conflict_type=conflict.conflict_type,
            severity=conflict.severity,
            context=conflict.context,
            resolution_id=resolution_id,
            resolution_type=resolution_id.upper(),
            entity_types=[conflict.context.get("entity_type")] if conflict.context.get("entity_type") else [],
            property_names=[conflict.context.get("property_name")] if conflict.context.get("property_name") else [],
            relationship_types=[conflict.context.get("relationship_type")] if conflict.context.get("relationship_type") else [],
            applied_by="user1" if i % 3 == 0 else "user2",
            applied_at=datetime.now() - timedelta(days=i),
            success=i % 2 == 0,
            feedback="Good resolution" if i % 2 == 0 else "Could be better",
            effectiveness=0.8 if i % 2 == 0 else 0.5
        )
        test_entries.append(entry)
    
    # Create entries for merge2
    for i in range(3):
        conflict = create_test_conflict(f"conflict{i+5}", "merge2", 
                                       ConflictType.DUPLICATE_ENTITY if i % 2 == 0 else ConflictType.PROPERTY_VALUE)
        
        # Use the appropriate resolution ID based on the conflict type
        if conflict.conflict_type == ConflictType.DUPLICATE_ENTITY:
            resolution_id = "merge_entities" if i % 2 == 0 else "keep_separate"
        else:
            resolution_id = "keep_source" if i % 2 == 0 else "keep_target"
        
        entry = ResolutionHistoryEntry(
            id=f"entry{i+5}",
            conflict_id=conflict.id,
            merge_id=conflict.merge_id,
            conflict_type=conflict.conflict_type,
            severity=conflict.severity,
            context=conflict.context,
            resolution_id=resolution_id,
            resolution_type=resolution_id.upper(),
            entity_types=[conflict.context.get("entity_type")] if conflict.context.get("entity_type") else [],
            property_names=[conflict.context.get("property_name")] if conflict.context.get("property_name") else [],
            relationship_types=[conflict.context.get("relationship_type")] if conflict.context.get("relationship_type") else [],
            applied_by="user3",
            applied_at=datetime.now() - timedelta(days=i+5),
            success=True,
            feedback=None,
            effectiveness=0.9
        )
        test_entries.append(entry)
    
    return test_entries

def create_test_conflict(conflict_id, merge_id, conflict_type):
    """Helper function to create a test conflict"""
    context = {}
    resolution_options = []
    
    if conflict_type == ConflictType.PROPERTY_VALUE:
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
                description="Keep source value",
                resolution_type="KEEP_SOURCE",
                resolution_data={"value": "John Smith"},
                confidence=0.8
            ),
            ResolutionOption(
                id="keep_target",
                description="Keep target value",
                resolution_type="KEEP_TARGET",
                resolution_data={"value": "John A. Smith"},
                confidence=0.8
            )
        ]
    elif conflict_type == ConflictType.RELATIONSHIP_TYPE:
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
                description="Keep relationship",
                resolution_type="KEEP_RELATIONSHIP",
                resolution_data={},
                confidence=0.8
            ),
            ResolutionOption(
                id="remove_relationship",
                description="Remove relationship",
                resolution_type="REMOVE_RELATIONSHIP",
                resolution_data={},
                confidence=0.8
            )
        ]
    else:  # DUPLICATE_ENTITY
        context = {
            "entity_id": "entity123",
            "entity_type": "Person",
            "duplicate_id": "entity456"
        }
        resolution_options = [
            ResolutionOption(
                id="merge_entities",
                description="Merge entities",
                resolution_type="MERGE_ENTITIES",
                resolution_data={},
                confidence=0.8
            ),
            ResolutionOption(
                id="keep_separate",
                description="Keep as separate entities",
                resolution_type="KEEP_SEPARATE",
                resolution_data={},
                confidence=0.8
            )
        ]
    
    return Conflict(
        id=conflict_id,
        merge_id=merge_id,
        conflict_type=conflict_type,
        severity=ConflictSeverity.MAJOR,
        description="Test conflict",
        context=context,
        resolution_options=resolution_options,
        resolved=False
    )

class TestResolutionHistoryE2E:
    """End-to-end tests for the resolution history API"""
    
    @pytest.mark.asyncio
    async def test_get_resolutions_by_merge_id(self, test_client, test_data, mock_merge_service):
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
    async def test_filter_resolutions(self, test_client, test_data, mock_merge_service):
        """Test filtering resolutions"""
        # Act - Filter by conflict type
        response = test_client.get("/api/v1/resolutions?conflict_type=property_value")
        
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
        response = test_client.get("/api/v1/resolutions?conflict_type=property_value&user=user1")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
    
    @pytest.mark.asyncio
    async def test_get_resolution_stats(self, test_client, test_data, mock_merge_service):
        """Test retrieving resolution statistics"""
        # Act
        response = test_client.get("/api/v1/resolutions/stats")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check that the stats include the expected fields
        assert "total_resolutions" in data
        assert "by_conflict_type" in data
        assert "by_resolution_type" in data
        assert "by_entity_type" in data
        assert "by_user" in data
        assert "success_rate" in data
        assert "average_effectiveness" in data
        
        # Check specific values
        assert data["total_resolutions"] == 8
        assert "property_value" in data["by_conflict_type"]
        assert "relationship_type" in data["by_conflict_type"]
        assert "duplicate_entity" in data["by_conflict_type"]
        
    @pytest.mark.asyncio
    async def test_pagination_and_sorting(self, test_client, test_data, mock_merge_service):
        """Test pagination and sorting of resolutions"""
        # Act - First page
        response = test_client.get("/api/v1/resolutions?limit=3&offset=0&sort_by=applied_at&sort_order=desc")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
        assert data["total"] == 8
        assert data["limit"] == 3
        assert data["offset"] == 0
        
        # Get the first item's ID for later comparison
        first_page_ids = [item["id"] for item in data["items"]]
        
        # Act - Second page
        response = test_client.get("/api/v1/resolutions?limit=3&offset=3&sort_by=applied_at&sort_order=desc")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
        assert data["total"] == 8
        assert data["limit"] == 3
        assert data["offset"] == 3
        
        # Get the second page IDs
        second_page_ids = [item["id"] for item in data["items"]]
        
        # Ensure no overlap between pages
        assert not set(first_page_ids).intersection(set(second_page_ids))
        
        # Test sorting by effectiveness
        response = test_client.get("/api/v1/resolutions?sort_by=effectiveness&sort_order=desc")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check that items with effectiveness values are sorted correctly
        effectiveness_values = [item.get("effectiveness") for item in data["items"] if item.get("effectiveness") is not None]
        assert effectiveness_values == sorted(effectiveness_values, reverse=True)
    
    @pytest.mark.asyncio
    async def test_submit_resolution_feedback(self, test_client, test_data, mock_merge_service):
        """Test submitting feedback for a resolution."""
        # Get a resolution ID to update
        response = test_client.get("/api/v1/resolutions?limit=1")
        assert response.status_code == 200
        resolution_id = response.json()["items"][0]["id"]
        
        # Prepare feedback data
        feedback_data = {
            "success": True,
            "feedback": "This resolution worked perfectly",
            "effectiveness": 0.95
        }
        
        # Submit feedback
        response = test_client.post(f"/api/v1/resolutions/{resolution_id}/feedback", json=feedback_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] == True
        
        # Verify the update by getting the resolution again
        response = test_client.get(f"/api/v1/resolutions?limit=100")
        assert response.status_code == 200
        
        # Find our updated resolution
        updated_resolution = None
        for item in response.json()["items"]:
            if item["id"] == resolution_id:
                updated_resolution = item
                break
        
        assert updated_resolution is not None
        assert updated_resolution["success"] == feedback_data["success"]
        assert updated_resolution["feedback"] == feedback_data["feedback"]
        assert updated_resolution["effectiveness"] == feedback_data["effectiveness"]
    
    @pytest.mark.asyncio
    async def test_submit_resolution_feedback_minimal(self, test_client, test_data, mock_merge_service):
        """Test submitting minimal feedback (only success field) for a resolution."""
        # Get a resolution ID to update
        response = test_client.get("/api/v1/resolutions?limit=1")
        assert response.status_code == 200
        resolution_id = response.json()["items"][0]["id"]
        
        # Prepare minimal feedback data
        feedback_data = {
            "success": False
        }
        
        # Submit feedback
        response = test_client.post(f"/api/v1/resolutions/{resolution_id}/feedback", json=feedback_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] == True
        
        # Verify the update by getting the resolution again
        response = test_client.get(f"/api/v1/resolutions?limit=100")
        assert response.status_code == 200
        
        # Find our updated resolution
        updated_resolution = None
        for item in response.json()["items"]:
            if item["id"] == resolution_id:
                updated_resolution = item
                break
        
        assert updated_resolution is not None
        assert updated_resolution["success"] == feedback_data["success"]

class MockResolutionHistoryService:
    """Mock implementation of the ResolutionHistoryService for testing"""
    
    async def get_resolution_history(self, merge_id=None, limit=100, offset=0):
        """Get resolution history entries for a merge ID"""
        if merge_id == "merge1":
            return [entry for entry in test_entries if entry.merge_id == "merge1"][:limit]
        elif merge_id == "merge2":
            return [entry for entry in test_entries if entry.merge_id == "merge2"][:limit]
        else:
            return []
    
    async def get_resolution_count(self, merge_id=None):
        """Get count of resolutions for a merge ID"""
        if merge_id == "merge1":
            return len([entry for entry in test_entries if entry.merge_id == "merge1"])
        elif merge_id == "merge2":
            return len([entry for entry in test_entries if entry.merge_id == "merge2"])
        else:
            return 0
    
    async def filter_resolutions(self, filter_params, pagination_params):
        """Filter resolutions by various criteria"""
        filtered = []
        
        for entry in test_entries:
            # Apply filters
            if hasattr(filter_params, 'conflict_type') and filter_params.conflict_type and entry.conflict_type != filter_params.conflict_type:
                continue
                
            if hasattr(filter_params, 'resolution_type') and filter_params.resolution_type and entry.resolution_type != filter_params.resolution_type:
                continue
                
            if hasattr(filter_params, 'entity_type') and filter_params.entity_type and filter_params.entity_type not in entry.entity_types:
                continue
                
            if hasattr(filter_params, 'property_name') and filter_params.property_name and (not hasattr(entry, 'property_names') or filter_params.property_name not in entry.property_names):
                continue
                
            if hasattr(filter_params, 'relationship_type') and filter_params.relationship_type and (not hasattr(entry, 'relationship_types') or filter_params.relationship_type not in entry.relationship_types):
                continue
                
            if hasattr(filter_params, 'user') and filter_params.user and entry.applied_by != filter_params.user:
                continue
                
            if hasattr(filter_params, 'effectiveness') and filter_params.effectiveness is not None and (entry.effectiveness is None or entry.effectiveness < filter_params.effectiveness):
                continue
                
            filtered.append(entry)
        
        # Apply pagination
        total = len(filtered)
        
        # Apply sorting
        if pagination_params.sort_by == "applied_at":
            filtered.sort(key=lambda e: e.applied_at, reverse=(pagination_params.sort_order.lower() == "desc"))
        elif pagination_params.sort_by == "effectiveness" and all(e.effectiveness is not None for e in filtered):
            filtered.sort(key=lambda e: e.effectiveness or 0.0, reverse=(pagination_params.sort_order.lower() == "desc"))
        
        # Apply pagination
        paginated = filtered[pagination_params.offset:pagination_params.offset + pagination_params.limit]
        
        return paginated, total
    
    async def get_resolution_stats(self, start_date=None, end_date=None):
        """Get statistics about stored resolutions"""
        return {
            "total_resolutions": len(test_entries),
            "by_conflict_type": {
                "property_value": 3,
                "relationship_type": 2,
                "duplicate_entity": 1
            },
            "by_resolution_type": {
                "keep_source": 2,
                "keep_target": 2,
                "keep_relationship": 1,
                "remove_relationship": 1
            },
            "by_entity_type": {
                "Person": 6,
                "Organization": 2
            },
            "by_user": {
                "user1": 3,
                "user2": 2,
                "user3": 3
            },
            "success_count": 4,
            "success_rate": 0.67,
            "average_effectiveness": 0.75,
            "time_distribution": {
                "last_day": 2,
                "last_week": 4,
                "last_month": 8
            }
        }
    
    async def submit_resolution_feedback(self, resolution_id, success=None, effectiveness=None, feedback=None, tags=None):
        """Submit feedback for a resolution"""
        # Find the resolution
        for entry in test_entries:
            if entry.id == resolution_id:
                if success is not None:
                    entry.success = success
                if effectiveness is not None:
                    entry.effectiveness = effectiveness
                if feedback:
                    entry.feedback = feedback
                if tags:
                    entry.tags = tags
                return entry
        
        raise ValueError(f"Resolution {resolution_id} not found")
    
    async def update_resolution_success(self, resolution_id, success=None, effectiveness=None, feedback=None, tags=None):
        """Update a resolution with feedback"""
        # This is just an alias for submit_resolution_feedback in our mock
        return await self.submit_resolution_feedback(
            resolution_id=resolution_id,
            success=success,
            effectiveness=effectiveness,
            feedback=feedback,
            tags=tags
        ) 