import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
from typing import List, Optional, AsyncGenerator
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager
from app.schemas.merge import (
    MergeProgressResponse, 
    MergeStatisticsResponse, 
    MergeSummaryResponse,
    MergeStage
)
from app.dependencies import get_merge_service
from app.main import app
from app.services.merge.models import MergeStatus

class MockMergeService:
    """Mock implementation of MergeService for testing"""
    
    async def get_merge_progress(self, merge_id: str) -> Optional[MergeProgressResponse]:
        """Mock implementation of get_merge_progress"""
        if merge_id == "nonexistent_id":
            return None
        return MergeProgressResponse(
            merge_id=merge_id,
            transform_id="test_transform_id",
            status="running",
            current_stage=MergeStage.EXECUTION,
            progress_percentage=45.0,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            estimated_completion_time=datetime.now(timezone.utc) + timedelta(minutes=5),
            elapsed_time_seconds=300.0,
            is_active=True
        )
    
    async def get_merge_statistics(self, merge_id: str) -> Optional[MergeStatisticsResponse]:
        """Mock implementation of get_merge_statistics"""
        if merge_id == "nonexistent_id":
            return None
        return MergeStatisticsResponse(
            merge_id=merge_id,
            transform_id="test_transform_id",
            nodes={
                "total": 1000,
                "processed": 450,
                "created": 300,
                "updated": 150,
                "unchanged": 0,
                "failed": 0
            },
            relationships={
                "total": 2000,
                "processed": 900,
                "created": 600,
                "updated": 300,
                "unchanged": 0,
                "failed": 0
            },
            conflicts_resolved=10,
            memory_usage_mb=128.5,
            processing_time_ms=2500.0,
            performed_by="test_user",
            errors=[]
        )
    
    async def get_merge_history(
        self,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transform_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[MergeSummaryResponse]:
        """Mock implementation of get_merge_history"""
        base_items = [
            MergeSummaryResponse(
                merge_id=f"merge_{i}",
                transform_id=f"transform_{(i % 3) + 1}",
                status="completed" if i % 2 == 0 else "failed",
                started_at=datetime.now() - timedelta(days=i),
                completed_at=datetime.now() - timedelta(days=i, hours=1),
                duration_seconds=3600,
                nodes_affected=100,
                relationships_affected=50,
                performed_by="test_user"
            )
            for i in range(1, 21)
        ]
        
        # Apply filters
        filtered_items = base_items
        
        if status:
            filtered_items = [item for item in filtered_items if item.status == status]
        
        if transform_id:
            filtered_items = [item for item in filtered_items if item.transform_id == transform_id]
        
        if start_date:
            filtered_items = [item for item in filtered_items if item.started_at >= start_date]
        
        if end_date:
            filtered_items = [item for item in filtered_items if item.started_at <= end_date]
        
        # Apply pagination
        paginated_items = filtered_items[offset:offset + limit]
        
        return paginated_items

# Create a direct dependency override function
async def get_mock_merge_service():
    """Mock implementation of get_merge_service that returns our mock service directly"""
    return MockMergeService()

@pytest.fixture
def test_client():
    """Create a test client with the mock service"""
    # Clear any existing overrides
    app.dependency_overrides.clear()
    
    # Override the get_merge_service dependency
    app.dependency_overrides[get_merge_service] = get_mock_merge_service
    
    # Create and return the test client
    with TestClient(app) as client:
        yield client
    
    # Clear the override after the test
    app.dependency_overrides.clear()

def test_merge_progress_api_endpoint(test_client):
    """Test the merge progress API endpoint"""
    # Act
    response = test_client.get("/api/v1/merge/progress/test_merge_id")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["merge_id"] == "test_merge_id"
    assert data["status"] == "running"
    assert data["progress_percentage"] == 45.0

def test_merge_statistics_api_endpoint(test_client):
    """Test the merge statistics API endpoint"""
    # Act
    response = test_client.get("/api/v1/merge/statistics/test_merge_id")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["merge_id"] == "test_merge_id"
    assert data["transform_id"] == "test_transform_id"
    assert data["nodes"]["total"] == 1000
    assert data["relationships"]["total"] == 2000

def test_merge_history_api_endpoint(test_client):
    """Test the merge history API endpoint"""
    # Act
    response = test_client.get("/api/v1/merge/history?limit=10&offset=0")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10  # Should respect the limit parameter

def test_merge_progress_not_found(test_client):
    """Test handling of non-existent merge ID for progress endpoint"""
    # Act
    response = test_client.get("/api/v1/merge/progress/nonexistent_id")
    
    # Assert
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()

def test_merge_statistics_not_found(test_client):
    """Test handling of non-existent merge ID for statistics endpoint"""
    # Act
    response = test_client.get("/api/v1/merge/statistics/nonexistent_id")
    
    # Assert
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()

def test_merge_history_filtering(test_client):
    """Test filtering in the merge history API endpoint"""
    # Act
    response = test_client.get(
        "/api/v1/merge/history?status=completed&transform_id=transform_1&limit=10&offset=0"
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    
    # Check that all returned items match the filter criteria
    for item in data:
        assert item["status"] == "completed"
        assert item["transform_id"] == "transform_1" 