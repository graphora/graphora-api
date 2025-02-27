"""Functional tests for LLM-assisted conflict analysis"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
from app.schemas.conflicts import ResolutionOption, ResolutionStrategy
from app.main import app
from fastapi.testclient import TestClient
from app.dependencies import get_staging_storage, get_production_storage, get_merge_service
from tests.unit.services.storage.test_graph_storage import MockGraphStorage

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_event_loop():
    """Set up event loop for tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def mock_storage():
    """Mock storage dependencies"""
    mock_storage = MockGraphStorage()
    
    # Create a mock MergeService
    mock_service = MagicMock()
    mock_service.analyze_conflicts_with_llm = AsyncMock(
        return_value={"total_conflicts": 2, "analyzed": 2}
    )
    
    # Override dependencies
    app.dependency_overrides[get_staging_storage] = lambda: mock_storage
    app.dependency_overrides[get_production_storage] = lambda: mock_storage
    app.dependency_overrides[get_merge_service] = lambda: mock_service
    
    yield mock_storage
    
    # Clean up
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_analyze_conflicts_api(mock_storage):
    """Test the analyze conflicts API endpoint"""
    # Mock the LLM analyzer
    mock_analyzer = AsyncMock()
    mock_analyzer.analyze_conflict.return_value = [
        ResolutionOption(
            id="opt1",
            description="Keep staging value",
            resolution_type=ResolutionStrategy.KEEP_STAGING,
            resolution_data={},
            confidence=0.8,
            reasoning="Staging data is more recent",
            auto_resolvable=True
        )
    ]
    
    with patch('app.services.merge.service.LLMConflictAnalyzer', return_value=mock_analyzer):
        response = client.post(
            "/api/v1/merge/test-merge-123/conflicts/analyze",
            json=["conflict-1", "conflict-2"]  # Send conflict_ids as array directly
        )
    
    assert response.status_code == 200
    result = response.json()
    assert "total_conflicts" in result
    assert "analyzed" in result

@pytest.mark.asyncio
async def test_analyze_conflicts_with_no_ids(mock_storage):
    """Test analyzing conflicts without specifying IDs"""
    # Mock the LLM analyzer
    mock_analyzer = AsyncMock()
    mock_analyzer.analyze_conflict.return_value = [
        ResolutionOption(
            id="opt1",
            description="Keep staging value",
            resolution_type=ResolutionStrategy.KEEP_STAGING,
            resolution_data={},
            confidence=0.8,
            reasoning="Staging data is more recent",
            auto_resolvable=True
        )
    ]
    
    with patch('app.services.merge.service.LLMConflictAnalyzer', return_value=mock_analyzer):
        response = client.post(
            "/api/v1/merge/test-merge-123/conflicts/analyze"
        )
    
    assert response.status_code == 200
    result = response.json()
    assert "total_conflicts" in result
    assert "analyzed" in result
