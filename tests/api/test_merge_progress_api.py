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
    mock.get_merge_progress.return_value = MergeProgressResponse(
        merge_id="test_merge_id",
        transform_id="test_transform_id",
        status="running",
        current_stage=MergeStage.EXECUTION,
        progress_percentage=45.0,
        started_at=datetime.now(timezone.utc),
        estimated_completion_time=datetime.now(timezone.utc) + timedelta(minutes=10),
        elapsed_time_seconds=300,
        is_active=True
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
    mock_progress = MergeProgressResponse(
        merge_id=merge_id,
        transform_id="test_transform_id",
        status="running",
        current_stage=MergeStage.EXECUTION,
        progress_percentage=45.0,
        started_at=datetime.now(timezone.utc),
        estimated_completion_time=datetime.now(timezone.utc) + timedelta(minutes=10),
        elapsed_time_seconds=300,
        is_active=True
    )
    mock_merge_service.get_merge_progress.return_value = mock_progress

    # Act
    result = await get_merge_progress(merge_id, mock_merge_service)

    # Assert
    assert result == mock_progress
    mock_merge_service.get_merge_progress.assert_called_once_with(merge_id)

@pytest.mark.asyncio
async def test_get_merge_progress_not_found(mock_merge_service):
    """Test the get_merge_progress endpoint function with non-existent merge ID"""
    # Arrange
    merge_id = "nonexistent_id"
    mock_merge_service.get_merge_progress.return_value = None

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await get_merge_progress(merge_id, mock_merge_service)
    
    assert exc_info.value.status_code == 404
    assert f"Merge {merge_id} not found" in str(exc_info.value.detail)

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
            processed=100,
            created=60,
            updated=30,
            unchanged=10,
            failed=0
        ),
        relationships=RelationshipStatistics(
            total=50,
            processed=50,
            created=30,
            updated=15,
            unchanged=5,
            failed=0
        ),
        conflicts_resolved=5,
        memory_usage_mb=128.5,
        processing_time_ms=2500.0,
        performed_by="test_user",
        errors=[]
    )
    mock_merge_service.get_merge_statistics.return_value = mock_statistics

    # Act
    result = await get_merge_statistics(merge_id, mock_merge_service)

    # Assert
    assert result == mock_statistics
    mock_merge_service.get_merge_statistics.assert_called_once_with(merge_id)

@pytest.mark.asyncio
async def test_get_merge_statistics_not_found(mock_merge_service):
    """Test the get_merge_statistics endpoint function with non-existent merge ID"""
    # Arrange
    merge_id = "nonexistent_id"
    mock_merge_service.get_merge_statistics.return_value = None

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await get_merge_statistics(merge_id, mock_merge_service)
    
    assert exc_info.value.status_code == 404
    assert f"Merge {merge_id} not found" in str(exc_info.value.detail)

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
            performed_by="user_1"
        ),
        MergeSummaryResponse(
            merge_id="merge_2",
            transform_id="transform_2",
            status="failed",
            started_at=datetime.now(timezone.utc) - timedelta(days=2),
            completed_at=datetime.now(timezone.utc) - timedelta(days=2, hours=1),
            duration_seconds=3600,
            nodes_affected=75,
            relationships_affected=30,
            performed_by="user_2"
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
    assert result == mock_history
    mock_merge_service.get_merge_history.assert_called_once_with(
        status=None,
        start_date=None,
        end_date=None,
        transform_id=None,
        limit=10,
        offset=0
    )

@pytest.mark.asyncio
async def test_get_merge_history_with_filtering(mock_merge_service):
    """Test the get_merge_history endpoint function with filtering"""
    # Arrange
    status = "completed"
    transform_id = "transform_1"
    limit = 5
    offset = 2
    
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
            performed_by="user_1"
        )
    ]
    mock_merge_service.get_merge_history.return_value = mock_history

    # Act
    result = await get_merge_history(
        status=status,
        start_date=None,
        end_date=None,
        transform_id=transform_id,
        limit=limit,
        offset=offset,
        merge_service=mock_merge_service
    )

    # Assert
    assert result == mock_history
    mock_merge_service.get_merge_history.assert_called_once_with(
        status=status,
        start_date=None,
        end_date=None,
        transform_id=transform_id,
        limit=limit,
        offset=offset
    )

def test_merge_progress_api_endpoint(test_client, mock_merge_service):
    """Test the merge progress API endpoint"""
    # Arrange
    mock_progress = MergeProgressResponse(
        merge_id="test_merge_id",
        transform_id="test_transform_id",
        status="running",
        current_stage=MergeStage.EXECUTION,
        progress_percentage=45.0,
        started_at=datetime.now(timezone.utc),
        estimated_completion_time=datetime.now(timezone.utc) + timedelta(minutes=10),
        elapsed_time_seconds=300,
        is_active=True
    )
    mock_merge_service.get_merge_progress.return_value = mock_progress

    # Act
    response = test_client.get("/api/v1/merge/progress/test_merge_id")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["merge_id"] == "test_merge_id"
    assert data["status"] == "running"
    assert data["current_stage"] == "execution"
    assert data["progress_percentage"] == 45.0

def test_merge_statistics_api_endpoint(test_client, mock_merge_service):
    """Test the merge statistics API endpoint"""
    # Arrange
    mock_statistics = MergeStatisticsResponse(
        merge_id="test_merge_id",
        transform_id="test_transform_id",
        nodes=NodeStatistics(
            total=100,
            processed=100,
            created=60,
            updated=30,
            unchanged=10,
            failed=0
        ),
        relationships=RelationshipStatistics(
            total=50,
            processed=50,
            created=30,
            updated=15,
            unchanged=5,
            failed=0
        ),
        conflicts_resolved=5,
        memory_usage_mb=128.5,
        processing_time_ms=2500.0,
        performed_by="test_user",
        errors=[]
    )
    mock_merge_service.get_merge_statistics.return_value = mock_statistics

    # Act
    response = test_client.get("/api/v1/merge/statistics/test_merge_id")

    # Assert
    assert response.status_code == 200
    assert response.json()["merge_id"] == "test_merge_id"
    assert response.json()["transform_id"] == "test_transform_id"
    assert response.json()["nodes"]["total"] == 100
    assert response.json()["nodes"]["created"] == 60
    assert response.json()["relationships"]["total"] == 50
    assert response.json()["conflicts_resolved"] == 5

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
            performed_by="user_1"
        ),
        MergeSummaryResponse(
            merge_id="merge_2",
            transform_id="transform_2",
            status="failed",
            started_at=datetime.now(timezone.utc) - timedelta(days=2),
            completed_at=datetime.now(timezone.utc) - timedelta(days=2, hours=1),
            duration_seconds=3600,
            nodes_affected=75,
            relationships_affected=30,
            performed_by="user_2"
        )
    ]
    mock_merge_service.get_merge_history.return_value = mock_history

    # Act
    response = test_client.get("/api/v1/merge/history?limit=10&offset=0")

    # Assert
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["merge_id"] == "merge_1"
    assert response.json()[1]["merge_id"] == "merge_2"

def test_merge_history_filtering(test_client, mock_merge_service):
    """Test filtering in the merge history API endpoint"""
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
            performed_by="user_1"
        )
    ]
    mock_merge_service.get_merge_history.return_value = mock_history

    # Act
    response = test_client.get(
        "/api/v1/merge/history?status=completed&transform_id=transform_1&limit=10&offset=0"
    )

    # Assert
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["merge_id"] == "merge_1"
    assert response.json()[0]["transform_id"] == "transform_1"
    assert response.json()[0]["status"] == "completed"
    
    # Verify that the correct parameters were passed to get_merge_history
    mock_merge_service.get_merge_history.assert_called_once()
    call_kwargs = mock_merge_service.get_merge_history.call_args.kwargs
    assert call_kwargs["status"] == "completed"
    assert call_kwargs["transform_id"] == "transform_1"
    assert call_kwargs["limit"] == 10
    assert call_kwargs["offset"] == 0

def test_merge_progress_not_found(test_client, mock_merge_service):
    """Test handling of non-existent merge ID for progress endpoint"""
    # Arrange
    mock_merge_service.get_merge_progress.return_value = None

    # Act
    response = test_client.get("/api/v1/merge/progress/nonexistent_id")

    # Assert
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_merge_statistics_not_found(test_client, mock_merge_service):
    """Test handling of non-existent merge ID for statistics endpoint"""
    # Arrange
    mock_merge_service.get_merge_statistics.return_value = None

    # Act
    response = test_client.get("/api/v1/merge/statistics/nonexistent_id")

    # Assert
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_merge_history_filtering(test_client, mock_merge_service):
    """Test filtering in the merge history API endpoint"""
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
            performed_by="user_1"
        )
    ]
    mock_merge_service.get_merge_history.return_value = mock_history

    # Act
    response = test_client.get(
        "/api/v1/merge/history?status=completed&transform_id=transform_1&limit=10&offset=0"
    )

    # Assert
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["merge_id"] == "merge_1"
    assert response.json()[0]["transform_id"] == "transform_1"
    assert response.json()[0]["status"] == "completed"

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