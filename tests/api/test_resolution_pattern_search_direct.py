"""Tests for the resolution pattern search API functions directly"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity
from app.services.storage.vector_storage import ResolutionPattern
from app.api.merge import find_similar_resolutions, batch_find_similar_resolutions
from app.api.merge import SimilarResolutionRequest, BatchSimilarResolutionRequest, SimilarResolutionResponse


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


class TestResolutionPatternSearchDirect:
    """Tests for the resolution pattern search API functions directly"""
    
    @pytest.mark.asyncio
    @patch("app.api.merge.MergeService")
    @patch("app.api.merge.QdrantResolutionStorage")
    @patch("app.api.merge.ResolutionPatternSearchService")
    async def test_find_similar_resolutions(self, mock_search_service_class, mock_storage_class, mock_merge_service_class, sample_conflict, sample_resolution_pattern):
        """Test finding similar resolutions for a conflict"""
        # Setup mocks
        mock_merge_service = mock_merge_service_class.return_value
        mock_merge_service.get_conflict = AsyncMock(return_value=sample_conflict)
        
        mock_storage = mock_storage_class.return_value
        
        mock_search_service = mock_search_service_class.return_value
        mock_search_service.find_similar_resolutions = AsyncMock(return_value=[
            (sample_resolution_pattern, 0.95),
            (sample_resolution_pattern, 0.85)
        ])
        
        # Create request
        request = SimilarResolutionRequest(
            conflict_id="test-conflict-1",
            limit=5,
            min_similarity=0.7
        )
        
        # Execute
        result = await find_similar_resolutions(request, mock_merge_service)
        
        # Assert
        assert result.conflict_id == "test-conflict-1"
        assert result.total_found == 2
        assert len(result.similar_resolutions) == 2
        assert result.similar_resolutions[0]["similarity_score"] == 0.95
        assert result.similar_resolutions[1]["similarity_score"] == 0.85
        
        # Verify service calls
        mock_merge_service.get_conflict.assert_called_once_with("test-conflict-1")
        mock_search_service.find_similar_resolutions.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.api.merge.MergeService")
    async def test_find_similar_resolutions_conflict_not_found(self, mock_merge_service_class):
        """Test finding similar resolutions for a non-existent conflict"""
        # Setup mocks
        mock_merge_service = mock_merge_service_class.return_value
        mock_merge_service.get_conflict = AsyncMock(return_value=None)
        
        # Create request
        request = SimilarResolutionRequest(
            conflict_id="non-existent-conflict",
            limit=5,
            min_similarity=0.7
        )
        
        # Execute and assert
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await find_similar_resolutions(request, mock_merge_service)
        
        # Verify exception details
        assert excinfo.value.status_code == 500
        assert "not found" in excinfo.value.detail.lower()
        
        # Verify service calls
        mock_merge_service.get_conflict.assert_called_once_with("non-existent-conflict")
    
    @pytest.mark.asyncio
    @patch("app.api.merge.MergeService")
    @patch("app.api.merge.QdrantResolutionStorage")
    @patch("app.api.merge.ResolutionPatternSearchService")
    async def test_batch_find_similar_resolutions(self, mock_search_service_class, mock_storage_class, mock_merge_service_class, sample_conflicts, sample_resolution_pattern):
        """Test batch finding similar resolutions for multiple conflicts"""
        # Setup mocks
        mock_merge_service = mock_merge_service_class.return_value
        mock_merge_service.get_conflict = AsyncMock(side_effect=lambda conflict_id: next((c for c in sample_conflicts if c.id == conflict_id), None))
        
        mock_storage = mock_storage_class.return_value
        
        mock_search_service = mock_search_service_class.return_value
        mock_search_service.batch_find_similar_resolutions = AsyncMock(return_value={
            "test-conflict-1": [
                (sample_resolution_pattern, 0.95),
                (sample_resolution_pattern, 0.85)
            ],
            "test-conflict-2": [
                (sample_resolution_pattern, 0.75)
            ]
        })
        
        # Create request
        request = BatchSimilarResolutionRequest(
            conflict_ids=["test-conflict-1", "test-conflict-2"],
            limit_per_conflict=5,
            min_similarity=0.7
        )
        
        # Execute
        result = await batch_find_similar_resolutions(request, mock_merge_service)
        
        # Assert
        assert "test-conflict-1" in result
        assert "test-conflict-2" in result
        assert result["test-conflict-1"].total_found == 2
        assert result["test-conflict-2"].total_found == 1
        
        # Verify service calls
        assert mock_merge_service.get_conflict.call_count == 2
        mock_search_service.batch_find_similar_resolutions.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.api.merge.MergeService")
    async def test_batch_find_similar_resolutions_no_conflicts_found(self, mock_merge_service_class):
        """Test batch finding similar resolutions when no conflicts are found"""
        # Setup mocks
        mock_merge_service = mock_merge_service_class.return_value
        mock_merge_service.get_conflict = AsyncMock(return_value=None)
        
        # Create request
        request = BatchSimilarResolutionRequest(
            conflict_ids=["non-existent-conflict-1", "non-existent-conflict-2"],
            limit_per_conflict=5,
            min_similarity=0.7
        )
        
        # Execute and assert
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await batch_find_similar_resolutions(request, mock_merge_service)
        
        # Verify exception details
        assert excinfo.value.status_code == 500
        assert "no valid conflicts found" in excinfo.value.detail.lower()
        
        # Verify service calls
        assert mock_merge_service.get_conflict.call_count == 2 