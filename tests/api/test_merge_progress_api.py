"""Integration tests for merge progress tracking API endpoints"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import List, Optional, AsyncGenerator
from fastapi import HTTPException
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager
from app.main import app
from app.schemas.merge import (
    MergeProgressResponse,
    MergeStatisticsResponse,
    MergeSummaryResponse,
    MergeStage,
    NodeStatistics,
    RelationshipStatistics
)
from app.services.merge.service import MergeService
from app.api.merge import get_merge_progress, get_merge_statistics, get_merge_history
from app.dependencies import get_merge_service
import redis
import json
import uuid

@pytest.fixture
def mock_merge_service():
    """Create a mock merge service"""
    mock = AsyncMock()
    # Pre-configure common mock responses
    start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    mock.get_merge_progress.return_value = MergeProgressResponse(
        merge_id="test_merge_id",
        overall_status="running",
        current_stage=MergeStage.EXTRACT,
        progress_percentage=45.0,
        start_time=start_time,
        end_time=None,
        estimated_completion_time=datetime.now(timezone.utc) + timedelta(minutes=10),
        elapsed_time_seconds=300,
        stages_progress={}
    )
    return mock

@asynccontextmanager
async def mock_get_merge_service(mock_service):
    """Mock the get_merge_service dependency to return our mock service"""
    yield mock_service

@pytest.fixture
def test_client(mock_merge_service):
    """Create a test client with the mock merge service"""
    # Clear any existing overrides
    app.dependency_overrides = {}
    
    # Override the get_merge_service dependency to return the mock service directly
    async def override_get_merge_service():
        return mock_merge_service
    
    app.dependency_overrides[get_merge_service] = override_get_merge_service
    
    # Create and return the test client
    client = TestClient(app)
    yield client
    
    # Clean up
    app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_get_merge_progress_endpoint(mock_merge_service):
    """Test the get_merge_progress endpoint function"""
    # Arrange
    merge_id = "test_merge_id"
    start_time = datetime.now(timezone.utc)
    mock_progress = MergeProgressResponse(
        merge_id=merge_id,
        overall_status="running",
        current_stage=MergeStage.EXTRACT,
        progress_percentage=45.0,
        start_time=start_time,
        end_time=None,
        estimated_completion_time=datetime.now(timezone.utc) + timedelta(minutes=10),
        elapsed_time_seconds=300,
        stages_progress={}
    )
    mock_merge_service.get_merge_progress.return_value = mock_progress
    
    # Act
    result = await get_merge_progress(merge_id, merge_service=mock_merge_service)
    
    # Assert
    mock_merge_service.get_merge_progress.assert_called_once_with(merge_id)
    assert result == mock_progress

@pytest.mark.asyncio
async def test_get_merge_progress_not_found(mock_merge_service):
    """Test the get_merge_progress endpoint when merge is not found"""
    # Arrange
    merge_id = "nonexistent_id"
    mock_merge_service.get_merge_progress.side_effect = ValueError(f"Merge {merge_id} not found")
    
    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await get_merge_progress(merge_id, merge_service=mock_merge_service)
    
    # Verify
    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.detail.lower()
    mock_merge_service.get_merge_progress.assert_called_once_with(merge_id)

@pytest.mark.asyncio
async def test_get_merge_statistics_endpoint(mock_merge_service):
    """Test the get_merge_statistics endpoint function"""
    # Arrange
    merge_id = "test_merge_id"
    mock_statistics = MergeStatisticsResponse(
        merge_id=merge_id,
        transform_id="test_transform_id",
        nodes=NodeStatistics(
            total=100,
            processed=75,
            created=50,
            updated=25,
            unchanged=0,
            failed=0
        ),
        relationships=RelationshipStatistics(
            total=50,
            processed=40,
            created=30,
            updated=10,
            unchanged=0,
            failed=0
        ),
        conflicts_resolved=10,
        memory_usage_mb=256.5,
        processing_time_ms=1500.0,
        performed_by="test_user"
    )
    mock_merge_service.get_merge_statistics.return_value = mock_statistics
    
    # Act
    result = await get_merge_statistics(merge_id, merge_service=mock_merge_service)
    
    # Assert
    mock_merge_service.get_merge_statistics.assert_called_once_with(merge_id)
    assert result == mock_statistics

@pytest.mark.asyncio
async def test_get_merge_statistics_not_found(mock_merge_service):
    """Test the get_merge_statistics endpoint when merge is not found"""
    # Arrange
    merge_id = "nonexistent_id"
    mock_merge_service.get_merge_statistics.side_effect = ValueError(f"Merge {merge_id} not found")
    
    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await get_merge_statistics(merge_id, merge_service=mock_merge_service)
    
    # Verify
    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.detail.lower()
    mock_merge_service.get_merge_statistics.assert_called_once_with(merge_id)

@pytest.mark.asyncio
async def test_get_merge_history_endpoint(mock_merge_service):
    """Test the get_merge_history endpoint function"""
    # Arrange
    mock_history = [
        MergeSummaryResponse(
            merge_id="merge_1",
            transform_id="transform_1",
            status="completed",
            started_at=datetime.now(timezone.utc) - timedelta(days=1),
            completed_at=datetime.now(timezone.utc) - timedelta(days=1, hours=1),
            duration_seconds=3600,
            nodes_affected=100,
            relationships_affected=50,
            performed_by="user1"
        ),
        MergeSummaryResponse(
            merge_id="merge_2",
            transform_id="transform_2",
            status="failed",
            started_at=datetime.now(timezone.utc) - timedelta(days=2),
            completed_at=datetime.now(timezone.utc) - timedelta(days=2, hours=1),
            duration_seconds=3600,
            nodes_affected=200,
            relationships_affected=100,
            performed_by="user2"
        )
    ]
    mock_merge_service.get_merge_history.return_value = mock_history
    
    # Act
    result = await get_merge_history(
        status=None,
        start_date=None,
        end_date=None,
        transform_id=None,
        limit=10,
        offset=0,
        merge_service=mock_merge_service
    )
    
    # Assert
    mock_merge_service.get_merge_history.assert_called_once_with(
        status=None,
        start_date=None,
        end_date=None,
        transform_id=None,
        limit=10,
        offset=0
    )
    assert result == mock_history

@pytest.mark.asyncio
async def test_get_merge_history_with_filtering(mock_merge_service):
    """Test the get_merge_history endpoint with filtering parameters"""
    # Arrange
    status = "completed"
    start_date = datetime.now(timezone.utc) - timedelta(days=7)
    end_date = datetime.now(timezone.utc)
    transform_id = "transform_1"
    limit = 5
    offset = 0
    
    mock_history = [
        MergeSummaryResponse(
            merge_id="merge_1",
            transform_id=transform_id,
            status=status,
            started_at=datetime.now(timezone.utc) - timedelta(days=1),
            completed_at=datetime.now(timezone.utc) - timedelta(days=1, hours=1),
            duration_seconds=3600,
            nodes_affected=100,
            relationships_affected=50,
            performed_by="user1"
        )
    ]
    mock_merge_service.get_merge_history.return_value = mock_history
    
    # Act
    result = await get_merge_history(
        status=status,
        start_date=start_date,
        end_date=end_date,
        transform_id=transform_id,
        limit=limit,
        offset=offset,
        merge_service=mock_merge_service
    )
    
    # Assert
    mock_merge_service.get_merge_history.assert_called_once_with(
        status=status,
        start_date=start_date,
        end_date=end_date,
        transform_id=transform_id,
        limit=limit,
        offset=offset
    )
    assert result == mock_history

def test_merge_progress_api_endpoint(test_client, mock_merge_service):
    """Test the merge progress API endpoint"""
    # Arrange
    mock_progress = MergeProgressResponse(
        merge_id="test_merge_id",
        overall_status="running",
        current_stage=MergeStage.EXTRACT,
        progress_percentage=45.0,
        start_time=datetime.now(timezone.utc),
        end_time=None,
        estimated_completion_time=datetime.now(timezone.utc) + timedelta(minutes=10),
        elapsed_time_seconds=300,
        stages_progress={}
    )
    mock_merge_service.get_merge_progress.return_value = mock_progress

    # Act
    response = test_client.get("/api/v1/merge/progress/test_merge_id")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["merge_id"] == "test_merge_id"
    assert data["overall_status"] == "running"
    assert data["current_stage"] == "extract"
    assert data["progress_percentage"] == 45.0

def test_merge_statistics_api_endpoint(test_client, mock_merge_service):
    """Test the merge statistics API endpoint"""
    # Arrange
    merge_id = "test_merge_id"
    mock_statistics = MergeStatisticsResponse(
        merge_id=merge_id,
        transform_id="test_transform_id",
        nodes=NodeStatistics(
            total=100,
            processed=75,
            created=50,
            updated=25,
            unchanged=0,
            failed=0
        ),
        relationships=RelationshipStatistics(
            total=50,
            processed=40,
            created=30,
            updated=10,
            unchanged=0,
            failed=0
        ),
        conflicts_resolved=10,
        memory_usage_mb=256.5,
        processing_time_ms=1500.0,
        performed_by="test_user"
    )
    mock_merge_service.get_merge_statistics.return_value = mock_statistics
    
    # Act
    response = test_client.get(f"/api/v1/merge/statistics/{merge_id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["merge_id"] == merge_id
    assert data["transform_id"] == "test_transform_id"
    assert data["nodes"]["total"] == 100
    assert data["nodes"]["created"] == 50
    assert data["relationships"]["total"] == 50
    assert data["conflicts_resolved"] == 10

def test_merge_history_api_endpoint(test_client, mock_merge_service):
    """Test the merge history API endpoint"""
    # Arrange
    mock_history = [
        MergeSummaryResponse(
            merge_id="merge_1",
            transform_id="transform_1",
            status="completed",
            started_at=datetime.now(timezone.utc) - timedelta(days=1),
            completed_at=datetime.now(timezone.utc) - timedelta(days=1, hours=1),
            duration_seconds=3600,
            nodes_affected=100,
            relationships_affected=50,
            performed_by="user1"
        ),
        MergeSummaryResponse(
            merge_id="merge_2",
            transform_id="transform_2",
            status="failed",
            started_at=datetime.now(timezone.utc) - timedelta(days=2),
            completed_at=datetime.now(timezone.utc) - timedelta(days=2, hours=1),
            duration_seconds=3600,
            nodes_affected=200,
            relationships_affected=100,
            performed_by="user2"
        )
    ]
    mock_merge_service.get_merge_history.return_value = mock_history
    
    # Act
    response = test_client.get("/api/v1/merge/history")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["merge_id"] == "merge_1"
    assert data[0]["transform_id"] == "transform_1"
    assert data[0]["status"] == "completed"
    assert data[1]["merge_id"] == "merge_2"
    assert data[1]["transform_id"] == "transform_2"
    assert data[1]["status"] == "failed"

def test_merge_history_filtering(test_client, mock_merge_service):
    """Test the merge history API endpoint with filtering"""
    # Arrange
    status = "completed"
    transform_id = "transform_1"
    
    mock_history = [
        MergeSummaryResponse(
            merge_id="merge_1",
            transform_id=transform_id,
            status=status,
            started_at=datetime.now(timezone.utc) - timedelta(days=1),
            completed_at=datetime.now(timezone.utc) - timedelta(days=1, hours=1),
            duration_seconds=3600,
            nodes_affected=100,
            relationships_affected=50,
            performed_by="user1"
        )
    ]
    mock_merge_service.get_merge_history.return_value = mock_history
    
    # Act
    response = test_client.get(f"/api/v1/merge/history?status={status}&transform_id={transform_id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["merge_id"] == "merge_1"
    assert data[0]["transform_id"] == transform_id
    assert data[0]["status"] == status

def test_merge_progress_not_found(test_client, mock_merge_service):
    """Test the merge progress API endpoint when merge is not found"""
    # Arrange
    merge_id = "nonexistent_id"
    mock_merge_service.get_merge_progress.side_effect = ValueError(f"Merge {merge_id} not found")
    
    # Act
    response = test_client.get(f"/api/v1/merge/progress/{merge_id}")
    
    # Assert
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_merge_statistics_not_found(test_client, mock_merge_service):
    """Test the merge statistics API endpoint when merge is not found"""
    # Arrange
    merge_id = "nonexistent_id"
    mock_merge_service.get_merge_statistics.side_effect = ValueError(f"Merge {merge_id} not found")
    
    # Act
    response = test_client.get(f"/api/v1/merge/statistics/{merge_id}")
    
    # Assert
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

@pytest.fixture
async def redis_client():
    """Get Redis client for testing"""
    client = redis.Redis.from_url(settings.REDIS_URL)
    try:
        yield client
    finally:
        await client.aclose()

@pytest.fixture
async def test_merge_id(redis_client):
    """Create a test merge ID and clean up after test"""
    merge_id = f"test_merge_{uuid.uuid4().hex}"
    transform_id = f"test_transform_{uuid.uuid4().hex}"
    
    # Setup test data
    status_data = {
        "transform_id": transform_id,
        "status": "running",
        "current_stage": "validation",
        "validation_progress": 0.5,
        "execution_progress": 0.5,
        "started_at": datetime.now(timezone.utc).isoformat()
    }
    await redis_client.set(f"merge:{merge_id}:status", json.dumps(status_data))
    
    # Add merge metadata
    merge_metadata = {
        "snapshot_id": f"snapshot_{merge_id}",
        "transform_id": transform_id,
        "status": "running"
    }
    await redis_client.set(f"merge:{merge_id}:metadata", json.dumps(merge_metadata))
    
    yield merge_id, transform_id
    
    # Clean up
    await redis_client.delete(f"merge:{merge_id}:status")
    await redis_client.delete(f"merge:{merge_id}:metadata")
    await redis_client.delete(f"merge:{merge_id}:statistics")

@pytest.fixture
def merge_service(mock_redis_client, mock_progress_tracker):
    """Create a merge service with mocked dependencies"""
    service = MergeService(
        storage=AsyncMock(),
        production_storage=AsyncMock(),
        progress_tracker=mock_progress_tracker
    )
    # Add mock redis client
    service._redis_client = mock_redis_client
    
    # Mock _get_merge_metadata to return valid data
    async def mock_get_metadata(merge_id):
        return {
            "snapshot_id": "snapshot-123",
            "transform_id": "transform-123",
            "status": "completed"
        }
    service._get_merge_metadata = mock_get_metadata
    
    return service 