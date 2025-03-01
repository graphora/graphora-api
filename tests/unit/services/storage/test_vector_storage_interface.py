"""Unit tests for the vector storage interface and implementations"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from typing import List, Dict, Any, Optional

from app.services.storage.vector_storage_interface import VectorStorageInterface, ResolutionPattern
from app.services.storage.qdrant_storage import QdrantVectorStorage
from app.services.storage.vector_storage_factory import get_vector_storage


# Mock implementation for testing
class MockVectorStorage(VectorStorageInterface):
    """Mock implementation for testing"""
    
    def __init__(self):
        self.patterns = {}  # id -> pattern
        self.initialized = False
        
    async def initialize(self) -> None:
        self.initialized = True
        
    async def store_pattern(self, pattern: ResolutionPattern) -> str:
        self.patterns[pattern.id] = pattern
        return pattern.id
        
    async def get_pattern(self, pattern_id: str) -> Optional[ResolutionPattern]:
        return self.patterns.get(pattern_id)
        
    async def update_pattern(self, pattern: ResolutionPattern) -> bool:
        if pattern.id not in self.patterns:
            return False
        self.patterns[pattern.id] = pattern
        return True
        
    async def delete_pattern(self, pattern_id: str) -> bool:
        if pattern_id not in self.patterns:
            return False
        del self.patterns[pattern_id]
        return True
        
    async def search_similar(
        self, 
        vector: List[float], 
        conflict_type: Optional[str] = None,
        limit: int = 10, 
        threshold: float = 0.7
    ) -> List[ResolutionPattern]:
        # Simple mock implementation using cosine similarity
        results = []
        for pattern in self.patterns.values():
            if conflict_type and pattern.conflict_type != conflict_type:
                continue
                
            # Calculate cosine similarity if vector exists
            if pattern.vector and len(pattern.vector) > 0:
                similarity = self._cosine_similarity(vector, pattern.vector)
                if similarity >= threshold:
                    results.append(pattern)
                    
        # Sort by similarity and return top 'limit'
        if len(results) > limit:
            results = results[:limit]
        return results
        
    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 10
    ) -> List[ResolutionPattern]:
        results = []
        for pattern in self.patterns.values():
            match = True
            for key, value in filters.items():
                if key.startswith("metadata."):
                    parts = key.split(".", 1)
                    if len(parts) > 1:
                        if not pattern.metadata.get(parts[1]) == value:
                            match = False
                            break
                elif key == "conflict_type":
                    if pattern.conflict_type != value:
                        match = False
                        break
                elif key == "resolution_strategy":
                    if pattern.resolution_strategy != value:
                        match = False
                        break
                        
            if match:
                results.append(pattern)
                if len(results) >= limit:
                    break
                    
        return results
        
    async def close(self) -> None:
        pass
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
            
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))


@pytest.fixture
def mock_vector_storage():
    return MockVectorStorage()


@pytest.fixture
def sample_pattern():
    return ResolutionPattern(
        id="test-pattern-1",
        conflict_type="property_value",
        resolution_strategy="keep_staging",
        context_features={
            "property_name": "name",
            "data_type": "string",
            "entity_type": "Person"
        },
        vector=[0.1, 0.2, 0.3, 0.4],
        metadata={
            "created_by": "user1",
            "confidence": 0.85
        },
        confidence=0.85
    )


@pytest.fixture
def mock_qdrant_client():
    with patch("app.services.storage.qdrant_storage.QdrantClient") as mock:
        client = MagicMock()
        mock.return_value = client
        
        # Setup mock responses
        collections_response = MagicMock()
        collections_response.collections = []
        client.get_collections.return_value = collections_response
        
        client.retrieve.return_value = []  # Default empty response
        
        yield client


class TestResolutionPattern:
    def test_resolution_pattern_creation(self):
        """Test creating a resolution pattern"""
        pattern = ResolutionPattern(
            conflict_type="property_value",
            resolution_strategy="keep_staging",
            context_features={"property_name": "name"},
            confidence=0.9
        )
        
        assert pattern.id is not None
        assert pattern.conflict_type == "property_value"
        assert pattern.resolution_strategy == "keep_staging"
        assert pattern.context_features == {"property_name": "name"}
        assert pattern.confidence == 0.9
        assert pattern.metadata == {}
        assert pattern.vector is None
        
    def test_resolution_pattern_with_all_fields(self, sample_pattern):
        """Test creating a resolution pattern with all fields"""
        assert sample_pattern.id == "test-pattern-1"
        assert sample_pattern.conflict_type == "property_value"
        assert sample_pattern.resolution_strategy == "keep_staging"
        assert sample_pattern.context_features["property_name"] == "name"
        assert sample_pattern.vector == [0.1, 0.2, 0.3, 0.4]
        assert sample_pattern.metadata["created_by"] == "user1"
        assert sample_pattern.confidence == 0.85


class TestMockVectorStorage:
    @pytest.mark.asyncio
    async def test_store_and_get_pattern(self, mock_vector_storage, sample_pattern):
        """Test storing and retrieving a pattern"""
        # Store pattern
        pattern_id = await mock_vector_storage.store_pattern(sample_pattern)
        assert pattern_id == sample_pattern.id
        
        # Get pattern
        retrieved = await mock_vector_storage.get_pattern(pattern_id)
        assert retrieved is not None
        assert retrieved.id == sample_pattern.id
        assert retrieved.conflict_type == sample_pattern.conflict_type
        assert retrieved.resolution_strategy == sample_pattern.resolution_strategy
        
    @pytest.mark.asyncio
    async def test_update_pattern(self, mock_vector_storage, sample_pattern):
        """Test updating a pattern"""
        # Store pattern
        await mock_vector_storage.store_pattern(sample_pattern)
        
        # Update pattern
        sample_pattern.confidence = 0.95
        success = await mock_vector_storage.update_pattern(sample_pattern)
        assert success is True
        
        # Verify update
        updated = await mock_vector_storage.get_pattern(sample_pattern.id)
        assert updated.confidence == 0.95
        
    @pytest.mark.asyncio
    async def test_update_nonexistent_pattern(self, mock_vector_storage, sample_pattern):
        """Test updating a pattern that doesn't exist"""
        # Try to update non-existent pattern
        success = await mock_vector_storage.update_pattern(sample_pattern)
        assert success is False
        
    @pytest.mark.asyncio
    async def test_delete_pattern(self, mock_vector_storage, sample_pattern):
        """Test deleting a pattern"""
        # Store pattern
        await mock_vector_storage.store_pattern(sample_pattern)
        
        # Delete pattern
        success = await mock_vector_storage.delete_pattern(sample_pattern.id)
        assert success is True
        
        # Verify deletion
        deleted = await mock_vector_storage.get_pattern(sample_pattern.id)
        assert deleted is None
        
    @pytest.mark.asyncio
    async def test_delete_nonexistent_pattern(self, mock_vector_storage):
        """Test deleting a pattern that doesn't exist"""
        success = await mock_vector_storage.delete_pattern("nonexistent-id")
        assert success is False
        
    @pytest.mark.asyncio
    async def test_search_similar(self, mock_vector_storage):
        """Test searching for similar patterns"""
        # Create patterns with specific vectors
        patterns = [
            ResolutionPattern(
                id="pattern1",
                conflict_type="property_value",
                vector=[1.0, 0.0, 0.0, 0.0],
                resolution_strategy="strategy1"
            ),
            ResolutionPattern(
                id="pattern2",
                conflict_type="property_value",
                vector=[0.0, 1.0, 0.0, 0.0],
                resolution_strategy="strategy2"
            ),
            ResolutionPattern(
                id="pattern3",
                conflict_type="relationship_type",
                vector=[0.0, 0.0, 1.0, 0.0],
                resolution_strategy="strategy3"
            )
        ]
        
        # Store patterns
        for pattern in patterns:
            await mock_vector_storage.store_pattern(pattern)
            
        # Search similar to pattern1
        results = await mock_vector_storage.search_similar(
            vector=[0.9, 0.1, 0.0, 0.0],
            threshold=0.8
        )
        
        assert len(results) == 1
        assert results[0].id == "pattern1"
        
        # Search with conflict type filter
        results = await mock_vector_storage.search_similar(
            vector=[0.1, 0.0, 0.9, 0.0],
            conflict_type="relationship_type",
            threshold=0.7
        )
        
        assert len(results) == 1
        assert results[0].id == "pattern3"
        
    @pytest.mark.asyncio
    async def test_search_by_metadata(self, mock_vector_storage, sample_pattern):
        """Test searching by metadata"""
        # Store pattern
        await mock_vector_storage.store_pattern(sample_pattern)
        
        # Search by metadata
        results = await mock_vector_storage.search_by_metadata({
            "metadata.created_by": "user1"
        })
        
        assert len(results) == 1
        assert results[0].id == sample_pattern.id
        
        # Search by conflict type
        results = await mock_vector_storage.search_by_metadata({
            "conflict_type": "property_value"
        })
        
        assert len(results) == 1
        assert results[0].id == sample_pattern.id
        
        # Search with non-matching filter
        results = await mock_vector_storage.search_by_metadata({
            "metadata.created_by": "user2"
        })
        
        assert len(results) == 0


class TestQdrantVectorStorage:
    @pytest.mark.asyncio
    async def test_initialization(self, mock_qdrant_client):
        """Test initialization of Qdrant storage"""
        # Test that collection is created if it doesn't exist
        storage = QdrantVectorStorage(collection_name="test_collection")
        await storage.initialize()
        
        # Verify collection creation was called
        mock_qdrant_client.create_collection.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_store_pattern(self, mock_qdrant_client, sample_pattern):
        """Test storing a pattern in Qdrant"""
        storage = QdrantVectorStorage(collection_name="test_collection")
        storage.client = mock_qdrant_client
        
        # Store pattern
        pattern_id = await storage.store_pattern(sample_pattern)
        
        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()
        
        # Check pattern ID was returned
        assert pattern_id == sample_pattern.id
        
    @pytest.mark.asyncio
    async def test_get_pattern(self, mock_qdrant_client, sample_pattern):
        """Test retrieving a pattern from Qdrant"""
        storage = QdrantVectorStorage(collection_name="test_collection")
        storage.client = mock_qdrant_client
        
        # Mock retrieve response
        point = MagicMock()
        point.id = storage._format_point_id(sample_pattern.id)
        point.vector = sample_pattern.vector
        point.payload = {
            "conflict_type": sample_pattern.conflict_type,
            "resolution_strategy": sample_pattern.resolution_strategy,
            "context_features": sample_pattern.context_features,
            "metadata": sample_pattern.metadata,
            "confidence": sample_pattern.confidence
        }
        mock_qdrant_client.retrieve.return_value = [point]
        
        # Get pattern
        pattern = await storage.get_pattern(sample_pattern.id)
        
        # Verify retrieve was called
        mock_qdrant_client.retrieve.assert_called_once()
        
        # Check pattern was returned correctly
        assert pattern is not None
        assert pattern.id == sample_pattern.id
        assert pattern.conflict_type == sample_pattern.conflict_type
        assert pattern.resolution_strategy == sample_pattern.resolution_strategy
        
    @pytest.mark.asyncio
    async def test_search_similar(self, mock_qdrant_client):
        """Test searching for similar patterns in Qdrant"""
        storage = QdrantVectorStorage(collection_name="test_collection")
        storage.client = mock_qdrant_client
        
        # Mock search response
        result1 = MagicMock()
        result1.id = "result1"
        result1.vector = [0.1, 0.2, 0.3, 0.4]
        result1.payload = {
            "conflict_type": "property_value",
            "resolution_strategy": "keep_staging",
            "context_features": {},
            "metadata": {},
            "confidence": 0.9
        }
        
        mock_qdrant_client.search.return_value = [result1]
        
        # Search
        results = await storage.search_similar(
            vector=[0.1, 0.2, 0.3, 0.4],
            conflict_type="property_value"
        )
        
        # Verify search was called
        mock_qdrant_client.search.assert_called_once()
        
        # Check results
        assert len(results) == 1
        assert results[0].conflict_type == "property_value"
        assert results[0].resolution_strategy == "keep_staging"


class TestVectorStorageFactory:
    def test_get_vector_storage_qdrant(self):
        """Test getting a Qdrant storage instance"""
        with patch("app.services.storage.vector_storage_factory.settings") as mock_settings:
            mock_settings.QDRANT_URL = "http://localhost:6333"
            mock_settings.QDRANT_API_KEY = "test_api_key"
            
            storage = get_vector_storage(storage_type="qdrant")
            
            assert isinstance(storage, QdrantVectorStorage)
            assert storage.url == "http://localhost:6333"
            assert storage.api_key == "test_api_key"
    
    def test_get_vector_storage_unsupported(self):
        """Test getting an unsupported storage type"""
        with pytest.raises(ValueError):
            get_vector_storage(storage_type="unsupported")

    def test_get_vector_storage_with_custom_params(self):
        """Test getting a storage instance with custom parameters"""
        with patch("app.services.storage.vector_storage_factory.settings") as mock_settings:
            mock_settings.QDRANT_URL = "http://localhost:6333"
            mock_settings.QDRANT_API_KEY = "test_api_key"
            
            storage = get_vector_storage(
                storage_type="qdrant",
                collection_name="custom_collection",
                vector_size=768
            )
            
            assert isinstance(storage, QdrantVectorStorage)
            assert storage.collection_name == "custom_collection"
            assert storage.vector_size == 768 