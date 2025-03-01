"""Tests for the resolution pattern search functionality"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity
from app.services.storage.vector_storage import ResolutionPattern
from app.services.merge.resolution_search import ResolutionPatternSearchService, ResolutionEmbeddingGenerator


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


@pytest.fixture
def mock_vector_storage():
    """Mock vector storage for testing"""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator for testing"""
    mock = AsyncMock()
    mock.generate_embedding.return_value = [0.1] * 1536
    return mock


class TestResolutionPatternSearch:
    """Tests for the resolution pattern search functionality"""
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions(self, mock_vector_storage, mock_embedding_generator, sample_conflict, sample_resolution_pattern):
        """Test finding similar resolutions for a conflict"""
        # Setup
        mock_vector_storage.search_similar_resolutions.return_value = [
            (sample_resolution_pattern, 0.95),
            (sample_resolution_pattern, 0.85)
        ]
        
        service = ResolutionPatternSearchService(
            vector_storage=mock_vector_storage,
            embedding_generator=mock_embedding_generator
        )
        
        # Execute
        results = await service.find_similar_resolutions(
            conflict=sample_conflict,
            limit=5
        )
        
        # Assert
        assert len(results) == 2
        assert results[0][1] == 0.95
        assert results[1][1] == 0.85
        
        # Verify service calls
        mock_embedding_generator.generate_embedding.assert_called_once_with(sample_conflict)
        mock_vector_storage.search_similar_resolutions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions_with_filters(self, mock_vector_storage, mock_embedding_generator, sample_conflict):
        """Test finding similar resolutions with custom filters"""
        # Setup
        service = ResolutionPatternSearchService(
            vector_storage=mock_vector_storage,
            embedding_generator=mock_embedding_generator
        )
        
        custom_filters = {"custom_field": "custom_value"}
        
        # Execute
        await service.find_similar_resolutions(
            conflict=sample_conflict,
            limit=3,
            filters=custom_filters
        )
        
        # Verify service calls
        mock_vector_storage.search_similar_resolutions.assert_called_once()
        
        # Check that the service was called with the correct parameters
        call_kwargs = mock_vector_storage.search_similar_resolutions.call_args[1]
        assert call_kwargs["conflict"] == sample_conflict
        assert call_kwargs["top_k"] == 3
        
        # The filters might be merged with conflict-specific filters in the implementation
        # So we just verify that our custom filter was included somewhere
        filters_used = service._build_filters(sample_conflict, custom_filters)
        assert "custom_field" in filters_used
        assert filters_used["custom_field"] == "custom_value"
    
    @pytest.mark.asyncio
    async def test_batch_find_similar_resolutions(self, mock_vector_storage, mock_embedding_generator, sample_conflicts, sample_resolution_pattern):
        """Test finding similar resolutions for multiple conflicts"""
        # Setup
        mock_vector_storage.search_similar_resolutions.return_value = [
            (sample_resolution_pattern, 0.95)
        ]
        
        service = ResolutionPatternSearchService(
            vector_storage=mock_vector_storage,
            embedding_generator=mock_embedding_generator
        )
        
        # Execute
        results = await service.batch_find_similar_resolutions(
            conflicts=sample_conflicts,
            limit_per_conflict=3
        )
        
        # Assert
        assert len(results) == 2
        assert "test-conflict-1" in results
        assert "test-conflict-2" in results
        assert len(results["test-conflict-1"]) == 1
        assert len(results["test-conflict-2"]) == 1
        
        # Verify service calls
        assert mock_embedding_generator.generate_embedding.call_count == 2
        assert mock_vector_storage.search_similar_resolutions.call_count == 2
    
    def test_build_filters(self, sample_conflict):
        """Test building filters for vector search"""
        # Setup
        service = ResolutionPatternSearchService(
            vector_storage=MagicMock(),
            embedding_generator=MagicMock()
        )
        
        # Execute
        filters = service._build_filters(sample_conflict)
        
        # Assert
        assert filters["conflict_type"] == "property_value"
        assert filters["entity_type"] == "Person"
        assert filters["property_name"] == "name"
    
    def test_build_filters_with_additional_filters(self, sample_conflict):
        """Test building filters with additional custom filters"""
        # Setup
        service = ResolutionPatternSearchService(
            vector_storage=MagicMock(),
            embedding_generator=MagicMock()
        )
        
        additional_filters = {
            "custom_field": "custom_value",
            "entity_type": "CustomType"  # This should override the conflict's entity_type
        }
        
        # Execute
        filters = service._build_filters(sample_conflict, additional_filters)
        
        # Assert
        assert filters["conflict_type"] == "property_value"
        assert filters["entity_type"] == "CustomType"  # Should be overridden
        assert filters["property_name"] == "name"
        assert filters["custom_field"] == "custom_value" 