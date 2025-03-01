"""Integration tests for the resolution pattern search service"""

import pytest
import time
from typing import List, Dict, Any, Tuple
import uuid
from datetime import datetime

from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity
from app.services.merge.resolution_search import ResolutionPatternSearchService, ResolutionEmbeddingGenerator
from app.services.storage.vector_storage import QdrantResolutionStorage, ResolutionPattern


@pytest.fixture
async def vector_storage():
    """Create a real Qdrant vector storage instance for testing"""
    # Use a unique collection name for each test run to avoid conflicts
    collection_name = f"test_resolutions_{uuid.uuid4().hex[:8]}"
    
    # Create storage instance with test collection
    storage = QdrantResolutionStorage(
        collection_name=collection_name,
        vector_size=768,  # Use 768 to match the embedding dimension of the model
        url="http://localhost:6333",  # Use local Qdrant instance
    )
    
    yield storage
    
    # Cleanup: delete the test collection
    try:
        storage.client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print(f"Error cleaning up test collection: {str(e)}")


@pytest.fixture
async def populated_vector_storage(vector_storage):
    """Create and populate a vector storage with sample resolution patterns"""
    # Sample patterns
    patterns = [
        ResolutionPattern(
            id=f"test-pattern-1-{uuid.uuid4().hex[:8]}",
            conflict_type="property_value",
            entity_types=["Person"],
            property_names=["name"],
            resolution_strategy="keep_staging",
            resolution_data={},
            confidence=0.9,
            original_conflict_id="conflict-1",
            original_merge_id="merge-1",
            metadata={"test": "value1"}
        ),
        ResolutionPattern(
            id=f"test-pattern-2-{uuid.uuid4().hex[:8]}",
            conflict_type="property_value",
            entity_types=["Organization"],
            property_names=["address"],
            resolution_strategy="keep_production",
            resolution_data={},
            confidence=0.85,
            original_conflict_id="conflict-2",
            original_merge_id="merge-1",
            metadata={"test": "value2"}
        ),
        ResolutionPattern(
            id=f"test-pattern-3-{uuid.uuid4().hex[:8]}",
            conflict_type="property_missing",
            entity_types=["Person"],
            property_names=["email"],
            resolution_strategy="keep_staging",
            resolution_data={},
            confidence=0.95,
            original_conflict_id="conflict-3",
            original_merge_id="merge-2",
            metadata={"test": "value3"}
        ),
    ]
    
    # Store patterns
    for pattern in patterns:
        await vector_storage.store_resolution(pattern)
    
    # Wait for indexing to complete
    time.sleep(1)
    
    return vector_storage


@pytest.fixture
def sample_conflicts():
    """Sample conflicts for testing"""
    return [
        # Similar to pattern 1
        Conflict(
            id=f"test-conflict-1-{uuid.uuid4().hex[:8]}",
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
        # Similar to pattern 2
        Conflict(
            id=f"test-conflict-2-{uuid.uuid4().hex[:8]}",
            merge_id="test-merge-1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            entity_id="entity-3",
            entity_type="Organization",
            property_name="address",
            staging_ids=["entity-3"],
            production_ids=["entity-4"],
            staging_value="123 Main St",
            production_value="123 Main Street",
            description="Property 'address' has different values",
            context={
                "property_name": "address",
                "entity_type": "Organization"
            }
        ),
        # Similar to pattern 3
        Conflict(
            id=f"test-conflict-3-{uuid.uuid4().hex[:8]}",
            merge_id="test-merge-1",
            conflict_type=ConflictType.PROPERTY_MISSING,
            severity=ConflictSeverity.MINOR,
            entity_id="entity-5",
            entity_type="Person",
            property_name="email",
            staging_ids=["entity-5"],
            production_ids=["entity-6"],
            staging_value="john@example.com",
            production_value=None,
            description="Property 'email' is missing in production",
            context={
                "property_name": "email",
                "entity_type": "Person"
            }
        ),
        # No similar pattern
        Conflict(
            id=f"test-conflict-4-{uuid.uuid4().hex[:8]}",
            merge_id="test-merge-1",
            conflict_type=ConflictType.RELATIONSHIP_MISSING,
            severity=ConflictSeverity.MAJOR,
            entity_id="entity-7",
            entity_type="Person",
            staging_ids=["entity-7"],
            production_ids=["entity-8"],
            description="Relationship is missing in production",
            context={
                "relationship_type": "WORKS_FOR",
                "entity_type": "Person"
            }
        )
    ]


@pytest.mark.integration
class TestResolutionPatternSearchIntegration:
    """Integration tests for the ResolutionPatternSearchService"""
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions(self, populated_vector_storage, sample_conflicts):
        """Test finding similar resolutions with real vector storage"""
        # Setup
        service = ResolutionPatternSearchService(
            vector_storage=populated_vector_storage,
            similarity_threshold=0.7
        )
        
        # Execute - find similar resolutions for first conflict (Person name)
        results = await service.find_similar_resolutions(
            conflict=sample_conflicts[0],
            limit=5
        )
        
        # Assert
        assert len(results) > 0, "Should find at least one similar resolution"
        
        # The first result should be pattern 1 (Person name)
        first_result = results[0][0]
        assert first_result.conflict_type == "property_value"
        assert "Person" in first_result.entity_types
        assert "name" in first_result.property_names
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions_with_filters(self, populated_vector_storage, sample_conflicts):
        """Test finding similar resolutions with filters"""
        # Setup
        service = ResolutionPatternSearchService(
            vector_storage=populated_vector_storage,
            similarity_threshold=0.7
        )
        
        # Execute - find similar resolutions with entity_type filter
        results = await service.find_similar_resolutions(
            conflict=sample_conflicts[0],
            limit=5,
            filters={"entity_type": "Person"}
        )
        
        # Assert
        assert len(results) > 0, "Should find at least one similar resolution"
        
        # All results should have Person entity type
        for pattern, score in results:
            assert "Person" in pattern.entity_types
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions_no_results(self, populated_vector_storage, sample_conflicts):
        """Test finding similar resolutions with no matches"""
        # Setup
        service = ResolutionPatternSearchService(
            vector_storage=populated_vector_storage,
            similarity_threshold=0.99  # Very high threshold
        )
        
        # Execute - find similar resolutions with high threshold
        results = await service.find_similar_resolutions(
            conflict=sample_conflicts[0],
            limit=5
        )
        
        # Assert
        assert len(results) == 0, "Should not find any results with very high threshold"
    
    @pytest.mark.asyncio
    async def test_batch_find_similar_resolutions(self, populated_vector_storage, sample_conflicts):
        """Test finding similar resolutions for multiple conflicts"""
        # Setup
        service = ResolutionPatternSearchService(
            vector_storage=populated_vector_storage,
            similarity_threshold=0.7
        )
        
        # Execute - find similar resolutions for all conflicts
        results = await service.batch_find_similar_resolutions(
            conflicts=sample_conflicts,
            limit_per_conflict=3
        )
        
        # Assert
        assert len(results) == 4, "Should return results for all 4 conflicts"
        
        # Check results for each conflict
        assert sample_conflicts[0].id in results
        assert sample_conflicts[1].id in results
        assert sample_conflicts[2].id in results
        assert sample_conflicts[3].id in results
        
        # First three conflicts should have results
        assert len(results[sample_conflicts[0].id]) > 0
        assert len(results[sample_conflicts[1].id]) > 0
        assert len(results[sample_conflicts[2].id]) > 0
        
        # Fourth conflict (relationship) might not have results since we didn't add similar patterns
    
    @pytest.mark.asyncio
    async def test_performance(self, populated_vector_storage, sample_conflicts):
        """Test performance of resolution pattern search"""
        # Setup
        service = ResolutionPatternSearchService(
            vector_storage=populated_vector_storage,
            similarity_threshold=0.7
        )
        
        # Execute with timing
        start_time = time.time()
        results = await service.find_similar_resolutions(
            conflict=sample_conflicts[0],
            limit=5
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Assert
        assert elapsed_time < 500, f"Query took {elapsed_time:.2f}ms, should be under 500ms" 