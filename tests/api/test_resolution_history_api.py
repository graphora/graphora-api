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
from app.services.resolution_history_service import ResolutionHistoryService
import json

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
def test_client(mock_resolution_service):
    """Create a test client with mocked dependencies"""
    # Patch the get_merge_service dependency
    async def mock_get_merge_service():
        mock_service = MagicMock()
        mock_service.resolution_history = mock_resolution_service
        yield mock_service
    
    app.dependency_overrides[get_merge_service] = mock_get_merge_service
    
    # Create and return the test client
    with TestClient(app) as client:
        yield client
    
    # Clean up the override after the test
    app.dependency_overrides.clear()

@pytest.fixture
def mock_resolution_service():
    with patch("app.dependencies.get_merge_service") as mock_get_service:
        mock_service = MagicMock()
        mock_resolution_service = AsyncMock(spec=ResolutionHistoryService)
        mock_service.resolution_history = mock_resolution_service
        mock_get_service.return_value = mock_service
        yield mock_resolution_service

class TestResolutionHistoryAPI:
    """Tests for the resolution history API endpoints"""
    
    def test_get_resolutions_by_merge_id(self, test_client, mock_resolution_service):
        # Setup mock data
        mock_entries = [
            ResolutionHistoryEntry(
                id="res1",
                conflict_id="conflict1",
                merge_id="merge123",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MAJOR,
                context={"entity_type": "Person"},
                resolution_id="opt1",
                resolution_type="keep_source",
                entity_types=["Person"],
                property_names=["name"],
                applied_by="user1",
                applied_at=datetime.now(),
                success=True,
                effectiveness=0.8
            ),
            ResolutionHistoryEntry(
                id="res2",
                conflict_id="conflict2",
                merge_id="merge123",
                conflict_type=ConflictType.RELATIONSHIP_TYPE,
                severity=ConflictSeverity.MAJOR,
                context={"entity_type": "Organization"},
                resolution_id="opt2",
                resolution_type="keep_target",
                entity_types=["Organization"],
                relationship_types=["EMPLOYS"],
                applied_by="user2",
                applied_at=datetime.now(),
                success=False,
                effectiveness=0.7
            )
        ]
        
        # Configure mock
        mock_resolution_service.get_resolution_history.return_value = mock_entries
        mock_resolution_service.get_resolution_count.return_value = len(mock_entries)
        
        # Make request
        response = test_client.get("/api/v1/resolutions/merge123")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        assert data["items"][0]["id"] == "res1"
        assert data["items"][1]["id"] == "res2"
        
        # Verify service was called correctly
        mock_resolution_service.get_resolution_history.assert_called_once_with(
            merge_id="merge123",
            limit=10,
            offset=0
        )
    
    def test_filter_resolutions(self, test_client, mock_resolution_service):
        # Setup mock data
        mock_entries = [
            ResolutionHistoryEntry(
                id="res1",
                conflict_id="conflict1",
                merge_id="merge123",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MAJOR,
                context={"entity_type": "Person"},
                resolution_id="opt1",
                resolution_type="keep_source",
                entity_types=["Person"],
                property_names=["name"],
                applied_by="user1",
                applied_at=datetime.now(),
                success=True,
                effectiveness=0.8
            )
        ]
        
        # Configure mock
        mock_resolution_service.filter_resolutions.return_value = (mock_entries, 1)
        
        # Make request with filters
        response = test_client.get(
            "/api/v1/resolutions?conflict_type=property_value&user=user1&effectiveness=0.8"
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == "res1"
        assert data["items"][0]["conflict_type"] == "property_value"
        assert data["items"][0]["applied_by"] == "user1"
        assert data["items"][0]["effectiveness"] == 0.8
    
    def test_filter_resolutions_with_multiple_filters(self, test_client, mock_resolution_service):
        # Setup mock data
        mock_entries = [
            ResolutionHistoryEntry(
                id="res1",
                conflict_id="conflict1",
                merge_id="merge123",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MAJOR,
                context={"entity_type": "Person"},
                resolution_id="opt1",
                resolution_type="keep_source",
                entity_types=["Person"],
                property_names=["name"],
                applied_by="user1",
                applied_at=datetime.now(),
                success=True,
                effectiveness=0.8
            )
        ]
        
        # Configure mock
        mock_resolution_service.filter_resolutions.return_value = (mock_entries, 1)
        
        # Make request with multiple filters
        response = test_client.get(
            "/api/v1/resolutions?conflict_type=property_value&entity_type=Person&resolution_type=keep_source"
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
    
    def test_get_resolution_stats(self, test_client, mock_resolution_service):
        # Setup mock stats
        mock_stats = {
            "total_resolutions": 10,
            "by_conflict_type": {
                "property_value": 5,
                "relationship_type": 3,
                "entity_match": 2
            },
            "by_resolution_type": {
                "keep_source": 4,
                "keep_target": 3,
                "merge_values": 3
            },
            "by_entity_type": {
                "Person": 5,
                "Organization": 3,
                "Product": 2
            },
            "by_user": {
                "user1": 6,
                "user2": 4
            },
            "success_rate": 0.8,
            "average_effectiveness": 0.75,
            "time_distribution": {
                "last_day": 2,
                "last_week": 5,
                "last_month": 3
            }
        }
        
        # Configure mock
        mock_resolution_service.get_resolution_stats.return_value = mock_stats
        
        # Make request
        response = test_client.get("/api/v1/resolutions/stats")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total_resolutions"] == 10
        assert data["by_conflict_type"]["property_value"] == 5
        assert data["success_rate"] == 0.8
        assert data["average_effectiveness"] == 0.75
    
    def test_pagination_and_sorting(self, test_client, mock_resolution_service):
        # Setup mock data
        mock_entries = [
            ResolutionHistoryEntry(
                id=f"res{i}",
                conflict_id=f"conflict{i}",
                merge_id="merge123",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MAJOR,
                context={"entity_type": "Person"},
                resolution_id=f"opt{i}",
                resolution_type="keep_source",
                entity_types=["Person"],
                property_names=["name"],
                applied_by="user1",
                applied_at=datetime.now(),
                success=True,
                effectiveness=0.5 + (i/10)
            ) for i in range(5)
        ]
        
        # Configure mock
        mock_resolution_service.filter_resolutions.return_value = (mock_entries[2:4], 5)
        
        # Make request with pagination and sorting
        response = test_client.get(
            "/api/v1/resolutions?limit=2&offset=2&sort_by=effectiveness&sort_order=desc"
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 2
        assert len(data["items"]) == 2
    
    def test_submit_resolution_feedback(self, test_client, mock_resolution_service):
        # Setup mock response
        mock_resolution_service.update_resolution_success.return_value = True
        
        # Prepare feedback data
        feedback_data = {
            "success": True,
            "feedback": "This resolution worked perfectly",
            "effectiveness": 0.95
        }
        
        # Make request
        response = test_client.post(
            "/api/v1/resolutions/res123/feedback",
            json=feedback_data
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] == True
        
        # Verify service was called correctly
        mock_resolution_service.update_resolution_success.assert_called_once_with(
            resolution_id="res123",
            success=True,
            feedback="This resolution worked perfectly",
            effectiveness=0.95
        )
    
    def test_submit_resolution_feedback_not_found(self, test_client, mock_resolution_service):
        # Setup mock response for non-existent resolution
        mock_resolution_service.update_resolution_success.return_value = False
        
        # Prepare feedback data
        feedback_data = {
            "success": True,
            "feedback": "This resolution worked perfectly",
            "effectiveness": 0.95
        }
        
        # Make request
        response = test_client.post(
            "/api/v1/resolutions/nonexistent/feedback",
            json=feedback_data
        )
        
        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    def test_submit_resolution_feedback_minimal(self, test_client, mock_resolution_service):
        # Setup mock response
        mock_resolution_service.update_resolution_success.return_value = True
        
        # Prepare minimal feedback data (only success is required)
        feedback_data = {
            "success": False
        }
        
        # Make request
        response = test_client.post(
            "/api/v1/resolutions/res123/feedback",
            json=feedback_data
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] == True
        
        # Verify service was called correctly with minimal data
        mock_resolution_service.update_resolution_success.assert_called_once_with(
            resolution_id="res123",
            success=False,
            feedback=None,
            effectiveness=None
        ) 