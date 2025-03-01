import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from app.main import app
from app.schemas.conflicts import ConflictType, ConflictSeverity
from app.schemas.resolution_history import ResolutionHistoryEntry, ResolutionFilter, PaginationParams, ResolutionStats
from app.dependencies import get_merge_service

# Mock data
mock_entries = [
    ResolutionHistoryEntry(
        id="history1",
        conflict_id="conflict1",
        merge_id="merge1",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        context={"entity_id": "entity1", "property": "name"},
        resolution_id="resolution1",
        resolution_type="KEEP_STAGING",
        entity_types=["Person"],
        property_names=["name"],
        applied_by="user1",
        applied_at=datetime.now(),
        success=True,
        feedback=None,
        effectiveness=0.8
    ),
    ResolutionHistoryEntry(
        id="history2",
        conflict_id="conflict2",
        merge_id="merge1",
        conflict_type=ConflictType.RELATIONSHIP_TYPE,
        severity=ConflictSeverity.MINOR,
        context={"source_id": "entity1", "target_id": "entity2"},
        resolution_id="resolution2",
        resolution_type="KEEP_PRODUCTION",
        entity_types=["Person", "Organization"],
        relationship_types=["WORKS_FOR"],
        applied_by="user1",
        applied_at=datetime.now(),
        success=True,
        feedback=None,
        effectiveness=0.7
    )
]

mock_stats = {
    "total_resolutions": 100,
    "by_conflict_type": {
        "property_value": 60,
        "relationship_type": 40
    },
    "by_resolution_type": {
        "KEEP_STAGING": 45,
        "KEEP_PRODUCTION": 35,
        "MERGE": 20
    },
    "by_entity_type": {
        "Person": 50,
        "Organization": 30,
        "Product": 20
    },
    "by_user": {
        "user1": 60,
        "user2": 40
    },
    "success_rate": 0.85,
    "average_effectiveness": 0.75,
    "time_distribution": {
        "2023-01": 30,
        "2023-02": 40,
        "2023-03": 30
    }
}

mock_suggestions = [
    {
        "resolution_type": "KEEP_STAGING",
        "confidence": 0.85,
        "reason": "Similar conflicts were resolved this way"
    },
    {
        "resolution_type": "MERGE_VALUES",
        "confidence": 0.15,
        "reason": "Alternative resolution strategy"
    }
]

class MockResolutionHistoryService:
    """Mock implementation of ResolutionHistoryService"""
    
    async def get_resolution_history(
        self, 
        merge_id=None, 
        conflict_type=None, 
        entity_type=None, 
        limit=100, 
        offset=0,
        sort_by="applied_at",
        sort_order="desc"
    ):
        if merge_id and merge_id != "merge1":
            return []
            
        return mock_entries
    
    async def get_resolution_count(self, merge_id=None):
        if merge_id and merge_id != "merge1":
            return 0
        return len(mock_entries)
    
    async def filter_resolutions(
        self,
        filter_params,
        pagination_params
    ) -> Tuple[List[ResolutionHistoryEntry], int]:
        filtered_entries = []
        
        for entry in mock_entries:
            # Apply filters
            if filter_params.conflict_type and entry.conflict_type != filter_params.conflict_type:
                continue
                
            if filter_params.resolution_type and entry.resolution_type != filter_params.resolution_type:
                continue
                
            if filter_params.entity_type and filter_params.entity_type not in entry.entity_types:
                continue
                
            if filter_params.user and entry.applied_by != filter_params.user:
                continue
                
            if filter_params.effectiveness is not None and (entry.effectiveness is None or entry.effectiveness < filter_params.effectiveness):
                continue
                
            filtered_entries.append(entry)
            
        # Apply pagination
        total = len(filtered_entries)
        paginated = filtered_entries[pagination_params.offset:pagination_params.offset + pagination_params.limit]
        
        return paginated, total
    
    async def get_resolution_stats(self, start_date=None, end_date=None):
        return mock_stats
    
    async def update_resolution_success(self, resolution_id, success, feedback=None, effectiveness=None):
        if resolution_id == "nonexistent":
            return False
        return True

class MockMergeService:
    """Mock implementation of MergeService"""
    
    def __init__(self):
        self.resolution_history = MockResolutionHistoryService()
    
    async def get_resolution_suggestions(self, merge_id, conflict_id):
        """Mock implementation of get_resolution_suggestions method"""
        return mock_suggestions

# Create an async generator that yields our mock service
async def mock_get_merge_service():
    mock_service = MockMergeService()
    yield mock_service

@pytest.fixture
def test_client():
    """Create a test client with mocked dependencies"""
    # Patch the get_merge_service dependency
    app.dependency_overrides[get_merge_service] = mock_get_merge_service
    
    # Create and return the test client
    with TestClient(app) as client:
        yield client
    
    # Clean up the override after the test
    app.dependency_overrides.clear()

class TestResolutionHistoryAPI:
    """Tests for the resolution history API endpoints"""
    
    def test_get_resolutions_by_merge_id(self, test_client):
        # Act
        response = test_client.get("/api/v1/resolutions/merge1")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        assert data["items"][0]["id"] == "history1"
        assert data["items"][1]["id"] == "history2"
    
    def test_get_resolutions_by_merge_id_not_found(self, test_client):
        # Act
        response = test_client.get("/api/v1/resolutions/nonexistent")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["items"]) == 0
    
    def test_filter_resolutions(self, test_client):
        # Act
        response = test_client.get(
            "/api/v1/resolutions?conflict_type=property_value&limit=10&offset=0"
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["conflict_type"] == "property_value"
    
    def test_filter_resolutions_with_multiple_filters(self, test_client):
        # Act
        response = test_client.get(
            "/api/v1/resolutions?conflict_type=relationship_type&user=user1&effectiveness=0.7"
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["conflict_type"] == "relationship_type"
        assert data["items"][0]["applied_by"] == "user1"
    
    def test_get_resolution_stats(self, test_client):
        # Act
        response = test_client.get("/api/v1/resolutions/stats")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_resolutions"] == 100
        assert data["success_rate"] == 0.85
        assert data["by_conflict_type"]["property_value"] == 60
        assert data["by_resolution_type"]["KEEP_STAGING"] == 45
        assert data["by_entity_type"]["Person"] == 50
        assert data["by_user"]["user1"] == 60
        assert data["average_effectiveness"] == 0.75
        assert data["time_distribution"]["2023-01"] == 30
    
    def test_get_resolution_history(self, test_client):
        # Act
        response = test_client.get("/api/v1/merge/resolution/history?merge_id=merge1")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "history1"
        assert data[1]["id"] == "history2"
    
    def test_get_resolution_history_invalid_conflict_type(self, test_client):
        # Act
        response = test_client.get("/api/v1/merge/resolution/history?conflict_type=INVALID")
        
        # Assert
        assert response.status_code == 400
        assert "Invalid conflict type" in response.json()["detail"]
    
    def test_get_resolution_suggestions(self, test_client):
        # Act
        response = test_client.get("/api/v1/merge/conflicts/merge1/conflict1/suggestions")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["resolution_type"] == "KEEP_STAGING"
        assert data[0]["confidence"] == 0.85
    
    def test_update_resolution_feedback(self, test_client):
        # Act
        response = test_client.post(
            "/api/v1/merge/resolution/history1/feedback?success=false",
            json={"feedback": "Resolution didn't work as expected"}
        )
        
        # Assert
        assert response.status_code == 200
        assert response.json()["updated"] is True
    
    def test_update_resolution_feedback_not_found(self, test_client):
        # Act
        response = test_client.post(
            "/api/v1/merge/resolution/nonexistent/feedback?success=false"
        )
        
        # Assert
        assert response.status_code == 404
        # The exact error message might be "Resolution nonexistent not found"
        assert "nonexistent not found" in response.json()["detail"] 