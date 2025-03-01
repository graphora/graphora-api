"""Tests for the resolution pattern search API endpoints"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient
import json

from app.main import app
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity
from app.services.storage.vector_storage import ResolutionPattern
from app.dependencies import get_merge_service
from app.services.merge.service import MergeService
from app.api.merge import find_similar_resolutions, batch_find_similar_resolutions
from app.api.merge import SimilarResolutionRequest, BatchSimilarResolutionRequest


# Mock the dependency
@pytest.fixture
def mock_merge_service():
    """Mock merge service for testing"""
    mock = AsyncMock(spec=MergeService)
    
    # Override the dependency in the app
    app.dependency_overrides[get_merge_service] = lambda: mock
    
    yield mock
    
    # Clean up after the test
    app.dependency_overrides.clear()


@pytest.fixture
def test_client(mock_merge_service):
    """Test client for the API"""
    return TestClient(app)


@pytest.fixture
def sample_conflict():
    """Sample conflict for testing"""
    return Conflict(
        id="test-conflict-1",
        merge_id="test-merge-1",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        entity_id="entity-1",
        entity_type="Person",
        property_name="name",
        staging_ids=["entity-1"],
        production_ids=["entity-2"],
        staging_value="John Smith",
        production_value="John A. Smith",
        description="Property 'name' has different values",
        context={
            "property_name": "name",
            "entity_type": "Person"
        }
    )


@pytest.fixture
def sample_conflicts():
    """Multiple sample conflicts for testing batch operations"""
    return [
        Conflict(
            id="test-conflict-1",
            merge_id="test-merge-1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            entity_id="entity-1",
            entity_type="Person",
            property_name="name",
            staging_ids=["entity-1"],
            production_ids=["entity-2"],
            staging_value="John Smith",
            production_value="John A. Smith",
            description="Property 'name' has different values",
            context={
                "property_name": "name",
                "entity_type": "Person"
            }
        ),
        Conflict(
            id="test-conflict-2",
            merge_id="test-merge-1",
            conflict_type=ConflictType.PROPERTY_MISSING,
            severity=ConflictSeverity.MINOR,
            entity_id="entity-3",
            entity_type="Organization",
            property_name="address",
            staging_ids=["entity-3"],
            production_ids=["entity-4"],
            staging_value="123 Main St",
            production_value=None,
            description="Property 'address' is missing in production",
            context={
                "property_name": "address",
                "entity_type": "Organization"
            }
        )
    ]


@pytest.fixture
def sample_resolution_pattern():
    """Sample resolution pattern for testing"""
    return ResolutionPattern(
        id="test-pattern-1",
        conflict_type="property_value",
        entity_types=["Person"],
        property_names=["name"],
        resolution_strategy="keep_staging",
        resolution_data={},
        confidence=0.9,
        original_conflict_id="conflict-1",
        original_merge_id="merge-1",
        created_at=datetime.now()
    )


class TestResolutionPatternSearchAPI:
    """Tests for the resolution pattern search API endpoints"""
    
    def test_find_similar_resolutions(self, test_client, mock_merge_service, sample_conflict, sample_resolution_pattern):
        """Test finding similar resolutions for a conflict"""
        # Setup mocks
        mock_merge_service.get_conflict.return_value = sample_conflict
        
        # Create a mock for ResolutionPatternSearchService
        mock_search_service = AsyncMock()
        mock_search_service.find_similar_resolutions.return_value = [
            (sample_resolution_pattern, 0.95),
            (sample_resolution_pattern, 0.85)
        ]
        
        # Create a mock for QdrantResolutionStorage
        mock_storage = MagicMock()
        
        # Patch the classes
        with patch("app.api.merge.ResolutionPatternSearchService", return_value=mock_search_service), \
             patch("app.api.merge.QdrantResolutionStorage", return_value=mock_storage):
            
            # Execute
            response = test_client.post(
                "/api/v1/merge/resolutions/similar",
                json={
                    "conflict_id": "test-conflict-1",
                    "limit": 5,
                    "min_similarity": 0.7
                }
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["conflict_id"] == "test-conflict-1"
            assert data["total_found"] == 2
            assert len(data["similar_resolutions"]) == 2
            assert data["similar_resolutions"][0]["similarity_score"] == 0.95
            assert data["similar_resolutions"][1]["similarity_score"] == 0.85
    
    def test_find_similar_resolutions_conflict_not_found(self, test_client, mock_merge_service):
        """Test finding similar resolutions for a non-existent conflict"""
        # Setup mocks
        mock_merge_service.get_conflict.return_value = None
        
        # Execute
        response = test_client.post(
            "/api/v1/merge/resolutions/similar",
            json={
                "conflict_id": "non-existent-conflict",
                "limit": 5,
                "min_similarity": 0.7
            }
        )
        
        # Assert - we expect a 500 error since the exception is caught and re-raised
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_batch_find_similar_resolutions(self, test_client, mock_merge_service, sample_conflicts, sample_resolution_pattern):
        """Test batch finding similar resolutions for multiple conflicts"""
        # Setup mocks
        mock_merge_service.get_conflict.side_effect = lambda conflict_id: next((c for c in sample_conflicts if c.id == conflict_id), None)
        
        # Create a mock for ResolutionPatternSearchService
        mock_search_service = AsyncMock()
        mock_search_service.batch_find_similar_resolutions.return_value = {
            "test-conflict-1": [
                (sample_resolution_pattern, 0.95),
                (sample_resolution_pattern, 0.85)
            ],
            "test-conflict-2": [
                (sample_resolution_pattern, 0.75)
            ]
        }
        
        # Create a mock for QdrantResolutionStorage
        mock_storage = MagicMock()
        
        # Patch the classes
        with patch("app.api.merge.ResolutionPatternSearchService", return_value=mock_search_service), \
             patch("app.api.merge.QdrantResolutionStorage", return_value=mock_storage):
            
            # Execute
            response = test_client.post(
                "/api/v1/merge/resolutions/batch-similar",
                json={
                    "conflict_ids": ["test-conflict-1", "test-conflict-2"],
                    "limit_per_conflict": 5,
                    "min_similarity": 0.7
                }
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "test-conflict-1" in data
            assert "test-conflict-2" in data
            assert data["test-conflict-1"]["total_found"] == 2
            assert data["test-conflict-2"]["total_found"] == 1
    
    def test_batch_find_similar_resolutions_no_conflicts_found(self, test_client, mock_merge_service):
        """Test batch finding similar resolutions when no conflicts are found"""
        # Setup mocks
        mock_merge_service.get_conflict.return_value = None
        
        # Execute
        response = test_client.post(
            "/api/v1/merge/resolutions/batch-similar",
            json={
                "conflict_ids": ["non-existent-conflict-1", "non-existent-conflict-2"],
                "limit_per_conflict": 5,
                "min_similarity": 0.7
            }
        )
        
        # Assert - we expect a 500 error since the exception is caught and re-raised
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "no valid conflicts found" in data["detail"].lower() 