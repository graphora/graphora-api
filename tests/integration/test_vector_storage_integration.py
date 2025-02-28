"""Integration tests for the Qdrant vector storage service"""

import pytest
import os
import uuid
from datetime import datetime

from app.services.storage.vector_storage import QdrantResolutionStorage, ResolutionPattern
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity

# Skip tests if not running integration tests
pytestmark = pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Integration tests are skipped by default. Set INTEGRATION_TESTS=1 to run."
)

@pytest.fixture
def resolution_storage():
    """Actual Qdrant storage instance for integration testing"""
    # Use test collection to avoid conflicts with production data
    # Set vector_size to 768 to match the actual embedding size used in tests
    storage = QdrantResolutionStorage(
        collection_name="test_resolution_patterns",
        vector_size=768  # Match the actual embedding size
    )
    yield storage
    
    # Cleanup - can optionally delete test collection after tests
    # This would require implementing a delete_collection method

@pytest.fixture
def test_pattern():
    """Test resolution pattern"""
    return ResolutionPattern(
        id=f"test-{uuid.uuid4()}",
        conflict_type="property_value",
        entity_types=["Person"],
        property_names=["name"],
        resolution_strategy="keep_staging",
        resolution_data={"reason": "Staging data is more recent"},
        confidence=0.9,
        original_conflict_id="test-conflict-1",
        original_merge_id="test-merge-1"
    )

@pytest.fixture
def test_conflict():
    """Test conflict for similarity search"""
    return Conflict(
        id="test-conflict-1",
        merge_id="test-merge-1",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        staging_ids=["s1"],
        production_ids=["p1"],
        description="Property 'name' has different values",
        context={
            "property_name": "name",
            "entity_type": "Person",
            "staging_value": "John Smith",
            "production_value": "John A. Smith"
        }
    )

class TestQdrantIntegration:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_resolution(self, resolution_storage, test_pattern):
        """Test end-to-end storage and retrieval"""
        # Store the pattern
        pattern_id = await resolution_storage.store_resolution(test_pattern)
        assert pattern_id is not None
        
        # Retrieve the pattern
        retrieved = await resolution_storage.get_resolution_by_id(pattern_id)
        assert retrieved is not None
        assert retrieved.conflict_type == test_pattern.conflict_type
        assert retrieved.entity_types == test_pattern.entity_types
        assert retrieved.resolution_strategy == test_pattern.resolution_strategy
        
        # Clean up
        await resolution_storage.delete_resolution(pattern_id)
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, resolution_storage, test_pattern, test_conflict):
        """Test similarity search functionality"""
        # Store the pattern
        pattern_id = await resolution_storage.store_resolution(test_pattern)
        
        # Search for similar patterns
        results = await resolution_storage.search_similar_resolutions(test_conflict)
        
        # Should find at least one result
        assert len(results) > 0
        assert any(pattern.id == pattern_id for pattern, _ in results)
        
        # Clean up
        await resolution_storage.delete_resolution(pattern_id)
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, resolution_storage):
        """Test batch operations"""
        # Create multiple patterns
        patterns = [
            ResolutionPattern(
                id=f"test-batch-{i}",
                conflict_type="property_value",
                entity_types=["Person"],
                property_names=["name"],
                resolution_strategy="keep_staging",
                confidence=0.9,
                original_conflict_id=f"conflict-{i}",
                original_merge_id="batch-test"
            )
            for i in range(5)
        ]
        
        # Batch store
        pattern_ids = await resolution_storage.batch_store_resolutions(patterns)
        assert len(pattern_ids) == 5
        
        # Retrieve one of the patterns
        retrieved = await resolution_storage.get_resolution_by_id(pattern_ids[2])
        assert retrieved is not None
        assert retrieved.original_conflict_id == "conflict-2"
        
        # Clean up
        for pattern_id in pattern_ids:
            await resolution_storage.delete_resolution(pattern_id)
    
    @pytest.mark.asyncio
    async def test_different_conflict_types(self, resolution_storage):
        """Test storing and retrieving patterns with different conflict types"""
        # Create patterns with different conflict types
        patterns = [
            ResolutionPattern(
                id=f"test-type-{conflict_type.value}",
                conflict_type=conflict_type.value,
                entity_types=["Person"],
                property_names=["name"] if "property" in conflict_type.value else None,
                relationship_types=["KNOWS"] if "relationship" in conflict_type.value else None,
                resolution_strategy="keep_staging",
                confidence=0.9,
                original_conflict_id=f"conflict-{conflict_type.value}",
                original_merge_id="type-test"
            )
            for conflict_type in [
                ConflictType.PROPERTY_VALUE,
                ConflictType.RELATIONSHIP_TYPE,
                ConflictType.ENTITY_MATCH
            ]
        ]
        
        # Batch store
        pattern_ids = await resolution_storage.batch_store_resolutions(patterns)
        assert len(pattern_ids) == 3
        
        # Get stats
        stats = await resolution_storage.get_stats()
        
        # Should have at least one pattern for each conflict type
        assert stats["by_conflict_type"]["property_value"] >= 1
        assert stats["by_conflict_type"]["relationship_type"] >= 1
        assert stats["by_conflict_type"]["entity_match"] >= 1
        
        # Clean up
        for pattern_id in pattern_ids:
            await resolution_storage.delete_resolution(pattern_id)
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, resolution_storage, test_conflict):
        """Test search with different filtering options"""
        # Create patterns with same conflict type but different entity types
        patterns = [
            ResolutionPattern(
                id=f"test-filter-{i}",
                conflict_type="property_value",
                entity_types=[entity_type],
                property_names=["name"],
                resolution_strategy="keep_staging",
                confidence=0.9,
                original_conflict_id=f"conflict-filter-{i}",
                original_merge_id="filter-test"
            )
            for i, entity_type in enumerate(["Person", "Organization", "Location"])
        ]
        
        # Batch store
        pattern_ids = await resolution_storage.batch_store_resolutions(patterns)
        assert len(pattern_ids) == 3
        
        # Search with conflict type filter (default)
        results1 = await resolution_storage.search_similar_resolutions(
            test_conflict,
            filter_by_conflict_type=True
        )
        
        # Search without conflict type filter
        results2 = await resolution_storage.search_similar_resolutions(
            test_conflict,
            filter_by_conflict_type=False
        )
        
        # Should find more results without the filter
        assert len(results2) >= len(results1)
        
        # Clean up
        for pattern_id in pattern_ids:
            await resolution_storage.delete_resolution(pattern_id) 