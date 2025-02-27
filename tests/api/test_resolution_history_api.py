import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from app.main import app
from app.schemas.conflicts import ConflictType, ConflictSeverity
from app.schemas.resolution_history import ResolutionHistoryEntry
from app.dependencies import get_merge_service

# Mock data
mock_entries = [
    ResolutionHistoryEntry(
        id="history1",
        conflict_id="conflict1",
        merge_id="merge1",
        conflict_type=ConflictType.PROPERTY,
        severity=ConflictSeverity.MAJOR,
        context={"entity_id": "entity1", "property": "name"},
        resolution_id="resolution1",
        resolution_type="KEEP_STAGING",
        entity_types=["Person"],
        property_names=["name"],
        applied_by="user1",
        applied_at=datetime.now(),
        success=True,
        feedback=None
    ),
    ResolutionHistoryEntry(
        id="history2",
        conflict_id="conflict2",
        merge_id="merge1",
        conflict_type=ConflictType.RELATIONSHIP,
        severity=ConflictSeverity.MINOR,
        context={"source_id": "entity1", "target_id": "entity2"},
        resolution_id="resolution2",
        resolution_type="KEEP_PRODUCTION",
        entity_types=["Person", "Organization"],
        relationship_types=["WORKS_FOR"],
        applied_by="user1",
        applied_at=datetime.now(),
        success=True,
        feedback=None
    )
]

mock_stats = {
    "total_resolutions": 100,
    "success_rate": 0.85,
    "by_conflict_type": {
        "PROPERTY": 60,
        "RELATIONSHIP": 40
    },
    "by_resolution_type": {
        "KEEP_SOURCE": 45,
        "KEEP_TARGET": 35,
        "MERGE": 20
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
    
    async def get_resolution_history(self, merge_id=None, conflict_type=None, entity_type=None, limit=100, offset=0):
        if conflict_type and conflict_type not in [ConflictType.PROPERTY, ConflictType.RELATIONSHIP, 
                                                  ConflictType.PROPERTY_VALUE, ConflictType.RELATIONSHIP_TYPE]:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"Invalid conflict type: {conflict_type}")
        
        if merge_id and merge_id != "merge1":
            return []
            
        return mock_entries
    
    async def get_resolution_stats(self):
        return mock_stats
    
    async def get_resolution_suggestions(self, conflict_type, context):
        return mock_suggestions
    
    async def update_resolution_success(self, resolution_id, success, feedback=None):
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
    
    def test_get_resolution_history(self, test_client):
        # Act
        response = test_client.get("/api/v1/merge/resolution/history?merge_id=merge1")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "history1"
        assert data[1]["id"] == "history2"
    
    def test_get_resolution_stats(self, test_client):
        # Act
        response = test_client.get("/api/v1/merge/resolution/stats")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_resolutions"] == 100
        assert data["success_rate"] == 0.85
    
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