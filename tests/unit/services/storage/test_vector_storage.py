"""Unit tests for the Qdrant vector storage service"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from qdrant_client.http import models

from app.services.storage.vector_storage import QdrantResolutionStorage, ResolutionPattern
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client"""
    with patch('app.services.storage.vector_storage.QdrantClient') as mock:
        client = MagicMock()
        mock.return_value = client
        
        # Mock collections response
        collections_response = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "resolution_patterns"
        collections_response.collections = [mock_collection]
        client.get_collections.return_value = collections_response
        
        # Mock query_points response
        query_response = MagicMock()
        point = MagicMock(
            id="test-id-1",
            score=0.95,
            payload={
                "conflict_type": "property_value",
                "entity_types": ["Person"],
                "property_names": ["name"],
                "resolution_strategy": "keep_staging",
                "resolution_data": {},
                "confidence": 0.9,
                "original_conflict_id": "conflict-1",
                "original_merge_id": "merge-1",
                "created_at": datetime.now().isoformat()
            }
        )
        query_response.points = [point]
        client.query_points.return_value = query_response
        
        # Mock retrieve response
        client.retrieve.return_value = [point]
        
        yield client


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
    with patch('app.services.storage.vector_storage.get_embedding', new_callable=AsyncMock) as mock:
        # Return a fixed embedding vector
        mock.return_value = [0.1] * 1536
        yield mock


@pytest.fixture
def mock_batch_embedding_service():
    """Mock batch embedding service"""
    with patch('app.services.storage.vector_storage.get_batch_embeddings', new_callable=AsyncMock) as mock:
        # Return fixed embedding vectors
        mock.return_value = [[0.1] * 1536, [0.2] * 1536]
        yield mock


@pytest.fixture
def resolution_pattern():
    """Sample resolution pattern"""
    return ResolutionPattern(
        id="test-pattern-1",
        conflict_type="property_value",
        entity_types=["Person"],
        property_names=["name"],
        resolution_strategy="keep_staging",
        resolution_data={},
        confidence=0.9,
        original_conflict_id="conflict-1",
        original_merge_id="merge-1"
    )


@pytest.fixture
def sample_conflict():
    """Sample conflict"""
    return Conflict(
        id="conflict-1",
        merge_id="merge-1",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        staging_ids=["s1"],
        production_ids=["p1"],
        description="Property 'name' has different values",
        context={
            "property_name": "name",
            "entity_type": "Person"
        }
    )


class TestQdrantResolutionStorage:
    
    def test_init_creates_collection_if_not_exists(self, mock_qdrant_client):
        """Test collection creation on initialization"""
        # Arrange
        mock_qdrant_client.get_collections.return_value.collections = []
        
        # Act
        storage = QdrantResolutionStorage()
        
        # Assert
        mock_qdrant_client.create_collection.assert_called_once()
        
    def test_init_uses_existing_collection(self, mock_qdrant_client):
        """Test using existing collection"""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.name = "resolution_patterns"
        mock_qdrant_client.get_collections.return_value.collections = [mock_collection]
        
        # Mock the collection info to return the correct vector size
        collection_info = MagicMock()
        collection_info.config.params.vectors.size = 1536  # Match the expected vector size
        mock_qdrant_client.get_collection.return_value = collection_info
        
        # Act
        storage = QdrantResolutionStorage()
        
        # Assert
        mock_qdrant_client.create_collection.assert_not_called()
        
    def test_init_with_custom_parameters(self, mock_qdrant_client):
        """Test initialization with custom parameters"""
        # Act
        storage = QdrantResolutionStorage(
            collection_name="custom_collection",
            vector_size=768,
            url="http://custom-url:6333",
            api_key="custom-api-key",
            distance_metric="euclid"
        )
        
        # Assert
        assert storage.collection_name == "custom_collection"
        assert storage.vector_size == 768
        assert storage.url == "http://custom-url:6333"
        assert storage.api_key == "custom-api-key"
        assert storage.distance == models.Distance.EUCLID
        
    def test_distance_metric_parsing(self, mock_qdrant_client):
        """Test parsing of distance metric"""
        # Act & Assert
        storage = QdrantResolutionStorage(distance_metric="cosine")
        assert storage.distance == models.Distance.COSINE
        
        storage = QdrantResolutionStorage(distance_metric="euclid")
        assert storage.distance == models.Distance.EUCLID
        
        storage = QdrantResolutionStorage(distance_metric="dot")
        assert storage.distance == models.Distance.DOT
        
        storage = QdrantResolutionStorage(distance_metric="invalid")
        assert storage.distance == models.Distance.COSINE  # Default to cosine
        
    @pytest.mark.asyncio
    async def test_store_resolution(self, mock_qdrant_client, mock_embedding_service, resolution_pattern):
        """Test storing a resolution pattern"""
        # Arrange
        storage = QdrantResolutionStorage()
        formatted_id = storage._format_point_id(resolution_pattern.id)
        
        # Act
        result = await storage.store_resolution(resolution_pattern)
        
        # Assert
        mock_qdrant_client.upsert.assert_called_once_with(
            collection_name="resolution_patterns",
            points=[
                models.PointStruct(
                    id=formatted_id,
                    vector=resolution_pattern.embedding,
                    payload=resolution_pattern.model_dump(exclude={"embedding"})
                )
            ]
        )
        assert result == resolution_pattern.id
        mock_embedding_service.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_store_resolution_with_existing_embedding(self, mock_qdrant_client, mock_embedding_service, resolution_pattern):
        """Test storing a resolution pattern with existing embedding"""
        # Arrange
        storage = QdrantResolutionStorage()
        resolution_pattern.embedding = [0.5] * 1536
        formatted_id = storage._format_point_id(resolution_pattern.id)
        
        # Act
        result = await storage.store_resolution(resolution_pattern)
        
        # Assert
        mock_qdrant_client.upsert.assert_called_once_with(
            collection_name="resolution_patterns",
            points=[
                models.PointStruct(
                    id=formatted_id,
                    vector=resolution_pattern.embedding,
                    payload=resolution_pattern.model_dump(exclude={"embedding"})
                )
            ]
        )
        assert result == resolution_pattern.id
        mock_embedding_service.assert_not_called()
        
    @pytest.mark.asyncio
    async def test_search_similar_resolutions(self, mock_qdrant_client, mock_embedding_service, sample_conflict):
        """Test searching for similar resolutions"""
        # Arrange
        storage = QdrantResolutionStorage()
        
        # Act
        results = await storage.search_similar_resolutions(sample_conflict)
        
        # Assert
        mock_qdrant_client.query_points.assert_called_once()
        assert len(results) == 1
        pattern, score = results[0]
        assert pattern.conflict_type == "property_value"
        assert score == 0.95
        mock_embedding_service.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_search_similar_resolutions_with_custom_params(self, mock_qdrant_client, mock_embedding_service, sample_conflict):
        """Test searching with custom parameters"""
        # Arrange
        storage = QdrantResolutionStorage()
        
        # Act
        results = await storage.search_similar_resolutions(
            sample_conflict,
            top_k=20,
            score_threshold=0.8,
            filter_by_conflict_type=False
        )
        
        # Assert
        mock_qdrant_client.query_points.assert_called_once()
        search_args = mock_qdrant_client.query_points.call_args[1]
        assert search_args["limit"] == 20
        assert search_args["score_threshold"] == 0.8
        assert search_args["query_filter"] is None  # No filter when filter_by_conflict_type=False
        
    @pytest.mark.asyncio
    async def test_batch_store_resolutions(self, mock_qdrant_client, mock_batch_embedding_service, resolution_pattern):
        """Test batch storing of resolution patterns"""
        # Arrange
        storage = QdrantResolutionStorage()
        patterns = [resolution_pattern, resolution_pattern.model_copy(update={"id": "test-pattern-2"})]
        
        # Generate embeddings for the patterns
        for pattern in patterns:
            pattern.embedding = [0.5] * 1536
            
        # Format IDs
        formatted_ids = [storage._format_point_id(p.id) for p in patterns]
        
        # Act
        result = await storage.batch_store_resolutions(patterns)
        
        # Assert
        # Check that upsert was called with the correct points
        call_args = mock_qdrant_client.upsert.call_args[1]
        assert call_args["collection_name"] == "resolution_patterns"
        assert len(call_args["points"]) == 2
        
        # Check that the points have the correct formatted IDs
        point_ids = [p.id for p in call_args["points"]]
        assert set(point_ids) == set(formatted_ids)
        
        assert len(result) == 2
        assert set(result) == set(p.id for p in patterns)  # Original IDs should be returned
        mock_batch_embedding_service.assert_not_called()  # We provided embeddings
        
    @pytest.mark.asyncio
    async def test_batch_store_empty_list(self, mock_qdrant_client, mock_batch_embedding_service):
        """Test batch storing with empty list"""
        # Arrange
        storage = QdrantResolutionStorage()
        
        # Act
        result = await storage.batch_store_resolutions([])
        
        # Assert
        mock_qdrant_client.upsert.assert_not_called()
        assert result == []
        mock_batch_embedding_service.assert_not_called()
        
    @pytest.mark.asyncio
    async def test_get_resolution_by_id(self, mock_qdrant_client, resolution_pattern):
        """Test retrieving a resolution by ID"""
        # Arrange
        storage = QdrantResolutionStorage()
        
        # Format the ID as it would be in the actual implementation
        formatted_id = storage._format_point_id(resolution_pattern.id)
        
        # Act
        result = await storage.get_resolution_by_id(resolution_pattern.id)
        
        # Assert
        mock_qdrant_client.retrieve.assert_called_once_with(
            collection_name="resolution_patterns",
            ids=[formatted_id]
        )
        assert result is not None
        assert result.conflict_type == "property_value"
        
    @pytest.mark.asyncio
    async def test_get_resolution_by_id_not_found(self, mock_qdrant_client):
        """Test retrieving a non-existent resolution"""
        # Arrange
        storage = QdrantResolutionStorage()
        mock_qdrant_client.retrieve.return_value = []
        
        # Act
        result = await storage.get_resolution_by_id("non-existent")
        
        # Assert
        assert result is None
        
    @pytest.mark.asyncio
    async def test_delete_resolution(self, mock_qdrant_client):
        """Test deleting a resolution"""
        # Arrange
        storage = QdrantResolutionStorage()
        test_id = "test-id"
        formatted_id = storage._format_point_id(test_id)
        
        # Act
        result = await storage.delete_resolution(test_id)
        
        # Assert
        mock_qdrant_client.delete.assert_called_once_with(
            collection_name="resolution_patterns",
            points_selector=models.PointIdsList(points=[formatted_id])
        )
        assert result is True
        
    @pytest.mark.asyncio
    async def test_delete_resolution_error(self, mock_qdrant_client):
        """Test error handling when deleting a resolution"""
        # Arrange
        storage = QdrantResolutionStorage()
        mock_qdrant_client.delete.side_effect = Exception("Test error")
        
        # Act
        result = await storage.delete_resolution("test-id")
        
        # Assert
        assert result is False
        
    @pytest.mark.asyncio
    async def test_get_stats(self, mock_qdrant_client):
        """Test getting storage statistics"""
        # Arrange
        storage = QdrantResolutionStorage()
        
        # Mock collection info
        collection_info = MagicMock()
        collection_info.config.params.vectors.size = 1536
        collection_info.config.params.vectors.distance = models.Distance.COSINE
        mock_qdrant_client.get_collection.return_value = collection_info
        
        # Mock count
        count_result = MagicMock()
        count_result.count = 42
        mock_qdrant_client.count.return_value = count_result
        
        # Act
        result = await storage.get_stats()
        
        # Assert
        assert result["total_patterns"] == 42
        assert result["vector_size"] == 1536
        assert "Cosine" in result["distance"]
        assert "by_conflict_type" in result 