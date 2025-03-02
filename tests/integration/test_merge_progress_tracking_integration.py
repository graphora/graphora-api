"""Integration tests for merge progress tracking functionality"""
import pytest
import json
import uuid
import time
from datetime import datetime, timezone, timedelta
import redis.asyncio as redis
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.merge.service import MergeService
from app.services.merge.progress import ProgressTracker
from app.services.merge.models import MergeStage, MergeStatus, StageStatus
from app.schemas.merge import (
    MergeProgressResponse,
    MergeStatisticsResponse,
    MergeSummaryResponse
)
from app.dependencies import get_merge_service

@pytest.fixture
async def redis_client():
    """Get Redis client for testing"""
    client = redis.Redis.from_url(settings.REDIS_URL)
    yield client
    await client.aclose()

@pytest.fixture
async def test_merge_id():
    """Create a test merge ID and clean up after test"""
    merge_id = f"test_merge_{uuid.uuid4().hex}"
    transform_id = f"test_transform_{uuid.uuid4().hex}"
    
    # Setup test data
    async with redis.Redis.from_url(settings.REDIS_URL) as conn:
        # Add merge ID to the set of all merges
        await conn.sadd("merges:all", merge_id)
        
        # Create status data
        status_data = {
            "transform_id": transform_id,
            "status": "running",
            "current_stage": "resolution",
            "validation_progress": 1.0,
            "execution_progress": 0.5,
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        await conn.set(f"merge:{merge_id}:status", json.dumps(status_data))
        
        # Add merge metadata
        merge_metadata = {
            "snapshot_id": f"snapshot_{merge_id}",
            "transform_id": transform_id,
            "status": "running"
        }
        await conn.set(f"merge:{merge_id}:metadata", json.dumps(merge_metadata))
        
        # Create statistics data
        stats_data = {
            "nodes": {
                "total": 100,
                "processed": 75,
                "created": 50,
                "updated": 25,
                "unchanged": 0,
                "failed": 0
            },
            "relationships": {
                "total": 50,
                "processed": 40,
                "created": 30,
                "updated": 10,
                "unchanged": 0,
                "failed": 0
            },
            "conflicts_resolved": 10,
            "memory_usage_mb": 256.5,
            "processing_time_ms": 1500.0,
            "performed_by": "test_user"
        }
        await conn.set(f"merge:{merge_id}:statistics", json.dumps(stats_data))
    
    yield merge_id, transform_id
    
    # Clean up
    async with redis.Redis.from_url(settings.REDIS_URL) as conn:
        await conn.srem("merges:all", merge_id)
        await conn.delete(f"merge:{merge_id}:status")
        await conn.delete(f"merge:{merge_id}:statistics")
        await conn.delete(f"merge:{merge_id}:metadata")

@pytest.fixture
def test_client():
    """Create a test client for the API with proper dependency overrides"""
    # Clear any existing overrides
    app.dependency_overrides = {}
    
    # Create a wrapper for the async context manager
    async def get_merge_service_override():
        async with get_merge_service() as service:
            return service
    
    # Override the dependency
    app.dependency_overrides[get_merge_service] = get_merge_service_override
    
    # Create and return the test client
    client = TestClient(app)
    yield client
    
    # Clean up
    app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_get_merge_progress_integration(test_merge_id):
    """Test getting merge progress from Redis"""
    merge_id, transform_id = test_merge_id
    
    # Create service
    async with redis.Redis.from_url(settings.REDIS_URL) as conn:
        # Get progress directly
        progress = await MergeService(
            storage=None,
            production_storage=None,
            progress_tracker=None
        ).get_merge_progress(merge_id)
        
        # Verify
        assert progress is not None
        assert progress.merge_id == merge_id
        assert progress.transform_id == transform_id
        assert progress.current_stage.value == "execution"
        assert progress.progress_percentage == 0.0
        assert progress.is_active == True

@pytest.mark.asyncio
async def test_get_merge_statistics_integration(test_merge_id):
    """Test getting merge statistics from Redis"""
    merge_id, transform_id = test_merge_id
    
    # Create service
    async with redis.Redis.from_url(settings.REDIS_URL) as conn:
        # Get statistics directly
        statistics = await MergeService(
            storage=None,
            production_storage=None,
            progress_tracker=None
        ).get_merge_statistics(merge_id)
        
        # Verify
        assert statistics is not None
        assert statistics.merge_id == merge_id
        assert statistics.transform_id == transform_id
        assert statistics.nodes.total == 100
        assert statistics.nodes.processed == 75
        assert statistics.nodes.created == 50
        assert statistics.relationships.total == 50
        assert statistics.conflicts_resolved == 10
        assert statistics.performed_by == "test_user"

@pytest.mark.asyncio
async def test_get_merge_history_integration(test_merge_id):
    """Test getting merge history from Redis"""
    merge_id, transform_id = test_merge_id
    
    # Create service
    async with redis.Redis.from_url(settings.REDIS_URL) as conn:
        # Get history
        history = await MergeService(
            storage=None,
            production_storage=None,
            progress_tracker=None
        ).get_merge_history(transform_id=transform_id)
        
        # Verify
        assert len(history) >= 1
        
        # Find our test merge
        test_merge = next((m for m in history if m.merge_id == merge_id), None)
        assert test_merge is not None
        assert test_merge.transform_id == transform_id
        assert test_merge.status == "running"

@pytest.mark.integration
def test_merge_progress_api_integration(test_client, test_merge_id):
    """Test the merge progress API endpoint with real data"""
    merge_id, _ = test_merge_id
    
    # Act
    response = test_client.get(f"/api/v1/merge/progress/{merge_id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["merge_id"] == merge_id
    assert data["status"] == "running"
    assert data["current_stage"] == "execution"
    assert data["progress_percentage"] == 0.0

@pytest.mark.integration
def test_merge_statistics_api_integration(test_client, test_merge_id):
    """Test the merge statistics API endpoint with real data"""
    merge_id, _ = test_merge_id
    
    # Act
    response = test_client.get(f"/api/v1/merge/statistics/{merge_id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["merge_id"] == merge_id
    assert data["nodes"]["total"] == 100
    assert data["nodes"]["created"] == 50
    assert data["relationships"]["total"] == 50
    assert data["conflicts_resolved"] == 10

@pytest.mark.integration
def test_merge_history_api_integration(test_client, test_merge_id):
    """Test the merge history API endpoint with real data"""
    _, transform_id = test_merge_id
    
    # Act
    response = test_client.get(f"/api/v1/merge/history?transform_id={transform_id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    
    # Find our test merge
    test_merge = next((m for m in data if m["transform_id"] == transform_id), None)
    assert test_merge is not None
    assert test_merge["status"] == "running" 