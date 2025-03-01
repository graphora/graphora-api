"""Tests for the ResolutionPatternSearchService"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime

from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity
from app.services.merge.resolution_search import ResolutionPatternSearchService, ResolutionEmbeddingGenerator
from app.services.storage.vector_storage import ResolutionPattern, QdrantResolutionStorage


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
    
    # Mock search_similar_resolutions
    sample_pattern = ResolutionPattern(
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
    
    mock.search_similar_resolutions.return_value = [
        (sample_pattern, 0.95),
        (sample_pattern, 0.85)
    ]
    
    return mock


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator for testing"""
    mock = AsyncMock()
    mock.generate_embedding.return_value = [0.1] * 1536
    return mock


class TestResolutionEmbeddingGenerator:
    """Tests for the ResolutionEmbeddingGenerator class"""
    
    @pytest.mark.asyncio
    @patch('app.services.merge.resolution_search.get_embedding')
    async def test_generate_embedding(self, mock_get_embedding, sample_conflict):
        """Test generating embeddings for a conflict"""
        # Setup
        mock_get_embedding.return_value = [0.1] * 1536
        generator = ResolutionEmbeddingGenerator()
        
        # Execute
        result = await generator.generate_embedding(sample_conflict)
        
        # Assert
        assert len(result) == 1536
        assert result[0] == 0.1
        mock_get_embedding.assert_called_once()
        
        # Check that the text representation was passed to the embedding function
        call_args = mock_get_embedding.call_args[0][0]
        assert "Conflict type: property_value" in call_args
        assert "Description: Property 'name' has different values" in call_args
        assert "Entity type: Person" in call_args
        assert "Property name: name" in call_args
    
    @pytest.mark.asyncio
    @patch('app.services.merge.resolution_search.get_embedding')
    async def test_generate_embedding_error_handling(self, mock_get_embedding, sample_conflict):
        """Test error handling when generating embeddings"""
        # Setup
        mock_get_embedding.side_effect = Exception("Test error")
        generator = ResolutionEmbeddingGenerator()
        
        # Execute
        result = await generator.generate_embedding(sample_conflict)
        
        # Assert
        assert len(result) == 1536
        assert all(val == 0.0 for val in result)  # Should return zero vector on error
    
    def test_conflict_to_text(self, sample_conflict):
        """Test converting a conflict to text representation"""
        # Setup
        generator = ResolutionEmbeddingGenerator()
        
        # Execute
        result = generator._conflict_to_text(sample_conflict)
        
        # Assert
        assert "Conflict type: property_value" in result
        assert "Description: Property 'name' has different values" in result
        assert "Entity type: Person" in result
        assert "Property name: name" in result
        assert "Staging value: John Smith" in result
        assert "Production value: John A. Smith" in result


class TestResolutionPatternSearchService:
    """Tests for the ResolutionPatternSearchService class"""
    
    def test_init(self, mock_vector_storage, mock_embedding_generator):
        """Test initialization of the service"""
        # Execute
        service = ResolutionPatternSearchService(
            vector_storage=mock_vector_storage,
            embedding_generator=mock_embedding_generator,
            similarity_threshold=0.8,
            collection_name="test_collection"
        )
        
        # Assert
        assert service.vector_storage == mock_vector_storage
        assert service.embedding_generator == mock_embedding_generator
        assert service.similarity_threshold == 0.8
        assert service.collection_name == "test_collection"
    
    def test_init_default_embedding_generator(self, mock_vector_storage):
        """Test initialization with default embedding generator"""
        # Execute
        service = ResolutionPatternSearchService(
            vector_storage=mock_vector_storage
        )
        
        # Assert
        assert service.vector_storage == mock_vector_storage
        assert isinstance(service.embedding_generator, ResolutionEmbeddingGenerator)
        assert service.similarity_threshold == 0.7
        assert service.collection_name == "resolution_patterns"
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions(self, mock_vector_storage, mock_embedding_generator, sample_conflict):
        """Test finding similar resolutions"""
        # Setup
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
        assert results[0][1] == 0.95  # First result score
        assert results[1][1] == 0.85  # Second result score
        
        # Check that the embedding generator was called
        mock_embedding_generator.generate_embedding.assert_called_once_with(sample_conflict)
        
        # Check that the vector storage was called with correct parameters
        mock_vector_storage.search_similar_resolutions.assert_called_once()
        call_kwargs = mock_vector_storage.search_similar_resolutions.call_args[1]
        assert call_kwargs["conflict"] == sample_conflict
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["score_threshold"] == 0.7
        assert call_kwargs["filter_by_conflict_type"] is True
    
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
        results = await service.find_similar_resolutions(
            conflict=sample_conflict,
            limit=3,
            filters=custom_filters
        )
        
        # Assert
        assert len(results) == 2
        
        # Check that the vector storage was called with correct parameters
        mock_vector_storage.search_similar_resolutions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions_error_handling(self, mock_vector_storage, mock_embedding_generator, sample_conflict):
        """Test error handling when finding similar resolutions"""
        # Setup
        mock_vector_storage.search_similar_resolutions.side_effect = Exception("Test error")
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
        assert len(results) == 0  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_batch_find_similar_resolutions(self, mock_vector_storage, mock_embedding_generator, sample_conflicts):
        """Test finding similar resolutions for multiple conflicts"""
        # Setup
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
        assert len(results["test-conflict-1"]) == 2
        assert len(results["test-conflict-2"]) == 2
        
        # Check that the find_similar_resolutions method was called for each conflict
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