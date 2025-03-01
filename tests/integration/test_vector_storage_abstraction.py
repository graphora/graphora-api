"""Integration tests for the vector storage abstraction layer"""

import pytest
import os
import uuid
import numpy as np
from typing import List

from app.services.storage.vector_storage_interface import ResolutionPattern
from app.services.storage.qdrant_storage import QdrantVectorStorage
from app.services.storage.vector_storage_factory import get_vector_storage

# Skip tests if not running integration tests
pytestmark = pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Integration tests are skipped by default. Set INTEGRATION_TESTS=1 to run."
)


def generate_random_vector(dim: int = 4) -> List[float]:
    """Generate a random unit vector for testing"""
    vec = np.random.random(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


@pytest.fixture
async def qdrant_storage():
    """Create a Qdrant storage instance for testing"""
    # Use a test-specific collection to avoid conflicts
    collection_name = f"test_abstraction_{uuid.uuid4().hex[:8]}"
    
    # Create storage instance
    storage = QdrantVectorStorage(
        collection_name=collection_name,
        vector_size=4  # Small size for tests
    )
    
    # Initialize
    await storage.initialize()
    
    yield storage
    
    # Cleanup - try to delete all test patterns
    try:
        # We would need to implement a delete_collection method for proper cleanup
        # For now, we'll just close the connection
        await storage.close()
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


@pytest.fixture
def test_patterns():
    """Create test patterns with different properties"""
    return [
        ResolutionPattern(
            id=f"test-pattern-1-{uuid.uuid4().hex[:8]}",
            conflict_type="property_value",
            resolution_strategy="keep_staging",
            context_features={"property_name": "name", "entity_type": "Person"},
            vector=generate_random_vector(),
            metadata={"source": "integration_test", "priority": "high"},
            confidence=0.9
        ),
        ResolutionPattern(
            id=f"test-pattern-2-{uuid.uuid4().hex[:8]}",
            conflict_type="property_value",
            resolution_strategy="keep_production",
            context_features={"property_name": "age", "entity_type": "Person"},
            vector=generate_random_vector(),
            metadata={"source": "integration_test", "priority": "medium"},
            confidence=0.8
        ),
        ResolutionPattern(
            id=f"test-pattern-3-{uuid.uuid4().hex[:8]}",
            conflict_type="relationship_type",
            resolution_strategy="keep_staging",
            context_features={"relationship": "KNOWS", "entity_type": "Person"},
            vector=generate_random_vector(),
            metadata={"source": "integration_test", "priority": "low"},
            confidence=0.7
        )
    ]


class TestVectorStorageAbstraction:
    @pytest.mark.asyncio
    async def test_factory_creates_correct_implementation(self):
        """Test that the factory creates the correct implementation"""
        # Get storage from factory with default type (qdrant)
        storage = get_vector_storage(
            collection_name=f"test_factory_{uuid.uuid4().hex[:8]}",
            vector_size=4
        )
        
        # Verify type
        assert isinstance(storage, QdrantVectorStorage)
        
        # Initialize and close to clean up
        await storage.initialize()
        await storage.close()
    
    @pytest.mark.asyncio
    async def test_crud_operations(self, qdrant_storage, test_patterns):
        """Test basic CRUD operations through the interface"""
        # Store patterns
        pattern_ids = []
        for pattern in test_patterns:
            pattern_id = await qdrant_storage.store_pattern(pattern)
            pattern_ids.append(pattern_id)
            
        # Retrieve patterns
        for i, pattern_id in enumerate(pattern_ids):
            retrieved = await qdrant_storage.get_pattern(pattern_id)
            assert retrieved is not None
            assert retrieved.id == test_patterns[i].id
            assert retrieved.conflict_type == test_patterns[i].conflict_type
            assert retrieved.resolution_strategy == test_patterns[i].resolution_strategy
            
        # Update a pattern
        test_patterns[0].confidence = 0.95
        success = await qdrant_storage.update_pattern(test_patterns[0])
        assert success is True
        
        # Verify update
        updated = await qdrant_storage.get_pattern(test_patterns[0].id)
        assert updated.confidence == 0.95
        
        # Delete patterns
        for pattern_id in pattern_ids:
            success = await qdrant_storage.delete_pattern(pattern_id)
            assert success is True
            
        # Verify deletion
        for pattern_id in pattern_ids:
            deleted = await qdrant_storage.get_pattern(pattern_id)
            assert deleted is None
    
    @pytest.mark.asyncio
    async def test_search_operations(self, qdrant_storage, test_patterns):
        """Test search operations through the interface"""
        # Store patterns
        for pattern in test_patterns:
            await qdrant_storage.store_pattern(pattern)
            
        try:
            # Search by vector similarity
            # We'll use the vector from the first pattern to find similar patterns
            results = await qdrant_storage.search_similar(
                vector=test_patterns[0].vector,
                threshold=0.5  # Lower threshold to ensure we get results
            )
            
            # Should find at least one result (the pattern itself)
            assert len(results) > 0
            
            # Search with conflict type filter
            results = await qdrant_storage.search_similar(
                vector=test_patterns[2].vector,
                conflict_type="relationship_type",
                threshold=0.5
            )
            
            # Should find the relationship_type pattern
            assert len(results) > 0
            assert any(r.conflict_type == "relationship_type" for r in results)
            
            # Search by metadata
            results = await qdrant_storage.search_by_metadata({
                "metadata.priority": "high"
            })
            
            # Should find the high priority pattern
            assert len(results) > 0
            assert any(r.metadata.get("priority") == "high" for r in results)
        finally:
            # Clean up
            for pattern in test_patterns:
                await qdrant_storage.delete_pattern(pattern.id)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, qdrant_storage):
        """Test error handling in the interface"""
        # Try to get a non-existent pattern
        non_existent = await qdrant_storage.get_pattern("non-existent-id")
        assert non_existent is None
        
        # Try to update a non-existent pattern
        non_existent_pattern = ResolutionPattern(
            id="non-existent-id",
            conflict_type="property_value",
            resolution_strategy="keep_staging"
        )
        success = await qdrant_storage.update_pattern(non_existent_pattern)
        assert success is False
        
        # Try to delete a non-existent pattern
        success = await qdrant_storage.delete_pattern("non-existent-id")
        assert success is False 