"""Unit tests for Entity Store.

Tests for persistent entity storage and cross-document resolution.
"""

import pytest
import numpy as np

from app.services.entity_resolution.entity_store import (
    EntityStore,
    CrossDocumentResolver,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def entity_store():
    """Create EntityStore instance."""
    return EntityStore(
        user_id="test-user",
        namespace="test-namespace",
        embedding_dim=384,
    )


@pytest.fixture
def sample_entity():
    """Create sample entity data."""
    return {
        "canonical_id": "entity-123",
        "entity_type": "TestEntity",
        "properties": {
            "name": "Test Entity",
            "code": "TEST001",
        },
        "embedding": list(np.random.randn(384)),
    }


@pytest.fixture
def cross_document_resolver(entity_store):
    """Create CrossDocumentResolver instance."""
    return CrossDocumentResolver(
        entity_store=entity_store,
        similarity_threshold=0.85,
    )


# ============================================================
# EntityStore Initialization Tests
# ============================================================


class TestEntityStoreInit:
    """Test EntityStore initialization."""

    def test_should_create_with_required_params(self):
        """Should create store with required parameters."""
        store = EntityStore(user_id="user-1", namespace="default")
        assert store.user_id == "user-1"
        assert store.namespace == "default"

    def test_should_accept_custom_embedding_dim(self):
        """Should accept custom embedding dimension."""
        store = EntityStore(
            user_id="user-1",
            namespace="default",
            embedding_dim=768,
        )
        assert store.embedding_dim == 768

    def test_should_initialize_empty_indices(self):
        """Should initialize with empty indices."""
        store = EntityStore(user_id="user-1", namespace="default")
        assert len(store._entities) == 0
        assert len(store._embeddings) == 0
        assert len(store._type_index) == 0


# ============================================================
# Entity Storage Tests
# ============================================================


class TestStoreEntity:
    """Test entity storage functionality."""

    @pytest.mark.asyncio
    async def test_should_store_entity(self, entity_store, sample_entity):
        """Should store entity and return storage key."""
        storage_key = await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
        )

        assert storage_key is not None
        assert sample_entity["canonical_id"] in storage_key

    @pytest.mark.asyncio
    async def test_should_store_entity_with_embedding(
        self, entity_store, sample_entity
    ):
        """Should store entity with embedding."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
            embedding=sample_entity["embedding"],
        )

        embedding = await entity_store.get_entity_embedding(
            sample_entity["canonical_id"]
        )
        assert embedding is not None
        assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_should_update_type_index(self, entity_store, sample_entity):
        """Should update type index when storing entity."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
        )

        assert sample_entity["entity_type"] in entity_store._type_index
        assert (
            sample_entity["canonical_id"]
            in entity_store._type_index[sample_entity["entity_type"]]
        )

    @pytest.mark.asyncio
    async def test_should_set_timestamps(self, entity_store, sample_entity):
        """Should set created_at and updated_at timestamps."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
        )

        entity = await entity_store.get_entity(sample_entity["canonical_id"])
        assert "created_at" in entity
        assert "updated_at" in entity


# ============================================================
# Entity Retrieval Tests
# ============================================================


class TestGetEntity:
    """Test entity retrieval functionality."""

    @pytest.mark.asyncio
    async def test_should_get_stored_entity(self, entity_store, sample_entity):
        """Should retrieve stored entity by canonical ID."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
        )

        entity = await entity_store.get_entity(sample_entity["canonical_id"])

        assert entity is not None
        assert entity["canonical_id"] == sample_entity["canonical_id"]
        assert entity["properties"]["name"] == "Test Entity"

    @pytest.mark.asyncio
    async def test_should_return_none_for_unknown_entity(self, entity_store):
        """Should return None for unknown entity."""
        entity = await entity_store.get_entity("unknown-id")
        assert entity is None


class TestGetEntityEmbedding:
    """Test embedding retrieval functionality."""

    @pytest.mark.asyncio
    async def test_should_get_stored_embedding(self, entity_store, sample_entity):
        """Should retrieve stored embedding."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
            embedding=sample_entity["embedding"],
        )

        embedding = await entity_store.get_entity_embedding(
            sample_entity["canonical_id"]
        )

        assert embedding is not None
        np.testing.assert_array_almost_equal(embedding, sample_entity["embedding"])

    @pytest.mark.asyncio
    async def test_should_return_none_for_entity_without_embedding(
        self, entity_store, sample_entity
    ):
        """Should return None when entity has no embedding."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
            # No embedding
        )

        embedding = await entity_store.get_entity_embedding(
            sample_entity["canonical_id"]
        )
        assert embedding is None


# ============================================================
# Similarity Search Tests
# ============================================================


class TestFindSimilarEntities:
    """Test embedding-based similarity search."""

    @pytest.mark.asyncio
    async def test_should_find_similar_entities(self, entity_store):
        """Should find entities with similar embeddings."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Store entities with embeddings
        base_embedding = np.random.randn(384)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        # Similar embedding (very close to base)
        similar_embedding = base_embedding.copy()
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

        # Different embedding
        different_embedding = np.random.randn(384)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)

        await entity_store.store_entity(
            canonical_id="entity-1",
            entity_type="TestEntity",
            properties={"name": "Similar"},
            embedding=list(similar_embedding),
        )
        await entity_store.store_entity(
            canonical_id="entity-2",
            entity_type="TestEntity",
            properties={"name": "Different"},
            embedding=list(different_embedding),
        )

        results = await entity_store.find_similar_entities(
            query_embedding=list(base_embedding),
            threshold=0.5,
        )

        # Should find at least one result (the similar one has similarity 1.0)
        assert len(results) >= 1
        # Results should be tuples of (canonical_id, similarity, entity)
        assert len(results[0]) == 3

    @pytest.mark.asyncio
    async def test_should_filter_by_entity_type(self, entity_store):
        """Should filter results by entity type."""
        embedding = list(np.random.randn(384))

        await entity_store.store_entity(
            canonical_id="entity-a",
            entity_type="TypeA",
            properties={},
            embedding=embedding,
        )
        await entity_store.store_entity(
            canonical_id="entity-b",
            entity_type="TypeB",
            properties={},
            embedding=embedding,
        )

        results = await entity_store.find_similar_entities(
            query_embedding=embedding,
            entity_type="TypeA",
            threshold=0.0,
        )

        # Should only find TypeA entities
        for canonical_id, _, entity in results:
            assert entity["entity_type"] == "TypeA"

    @pytest.mark.asyncio
    async def test_should_respect_top_k_limit(self, entity_store):
        """Should limit results to top K."""
        embedding = list(np.random.randn(384))

        for i in range(10):
            await entity_store.store_entity(
                canonical_id=f"entity-{i}",
                entity_type="TestEntity",
                properties={},
                embedding=embedding,
            )

        results = await entity_store.find_similar_entities(
            query_embedding=embedding,
            threshold=0.0,
            top_k=3,
        )

        assert len(results) <= 3


# ============================================================
# Property-Based Search Tests
# ============================================================


class TestFindByProperties:
    """Test property-based entity search."""

    @pytest.mark.asyncio
    async def test_should_find_by_matching_properties(self, entity_store):
        """Should find entities with matching properties."""
        await entity_store.store_entity(
            canonical_id="entity-1",
            entity_type="TestEntity",
            properties={"name": "John Doe", "email": "john@example.com"},
        )
        await entity_store.store_entity(
            canonical_id="entity-2",
            entity_type="TestEntity",
            properties={"name": "Jane Doe", "email": "jane@example.com"},
        )

        results = await entity_store.find_by_properties(
            entity_type="TestEntity",
            properties={"name": "John Doe"},
            match_threshold=0.5,
        )

        assert len(results) >= 1
        assert results[0][0] == "entity-1"

    @pytest.mark.asyncio
    async def test_should_return_empty_for_no_match(self, entity_store):
        """Should return empty list when no matches found."""
        await entity_store.store_entity(
            canonical_id="entity-1",
            entity_type="TestEntity",
            properties={"name": "John"},
        )

        results = await entity_store.find_by_properties(
            entity_type="TestEntity",
            properties={"name": "NonExistent"},
            match_threshold=0.9,
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_should_handle_case_insensitive_match(self, entity_store):
        """Should match properties case-insensitively."""
        await entity_store.store_entity(
            canonical_id="entity-1",
            entity_type="TestEntity",
            properties={"name": "John Doe"},
        )

        results = await entity_store.find_by_properties(
            entity_type="TestEntity",
            properties={"name": "john doe"},  # lowercase
            match_threshold=0.5,
        )

        assert len(results) >= 1


# ============================================================
# Entity Update Tests
# ============================================================


class TestUpdateEntity:
    """Test entity update functionality."""

    @pytest.mark.asyncio
    async def test_should_update_properties(self, entity_store, sample_entity):
        """Should update entity properties."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
        )

        result = await entity_store.update_entity(
            canonical_id=sample_entity["canonical_id"],
            properties={"new_field": "new_value"},
        )

        assert result is True

        entity = await entity_store.get_entity(sample_entity["canonical_id"])
        assert entity["properties"]["new_field"] == "new_value"
        # Original properties should be preserved
        assert entity["properties"]["name"] == "Test Entity"

    @pytest.mark.asyncio
    async def test_should_return_false_for_unknown_entity(self, entity_store):
        """Should return False when entity not found."""
        result = await entity_store.update_entity(
            canonical_id="unknown",
            properties={"field": "value"},
        )
        assert result is False


# ============================================================
# Entity Deletion Tests
# ============================================================


class TestDeleteEntity:
    """Test entity deletion functionality."""

    @pytest.mark.asyncio
    async def test_should_delete_entity(self, entity_store, sample_entity):
        """Should delete entity from store."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
        )

        result = await entity_store.delete_entity(sample_entity["canonical_id"])

        assert result is True
        entity = await entity_store.get_entity(sample_entity["canonical_id"])
        assert entity is None

    @pytest.mark.asyncio
    async def test_should_remove_from_type_index(self, entity_store, sample_entity):
        """Should remove entity from type index."""
        await entity_store.store_entity(
            canonical_id=sample_entity["canonical_id"],
            entity_type=sample_entity["entity_type"],
            properties=sample_entity["properties"],
        )

        await entity_store.delete_entity(sample_entity["canonical_id"])

        assert sample_entity["canonical_id"] not in entity_store._type_index.get(
            sample_entity["entity_type"], []
        )

    @pytest.mark.asyncio
    async def test_should_return_false_for_unknown_entity(self, entity_store):
        """Should return False when entity not found."""
        result = await entity_store.delete_entity("unknown")
        assert result is False


# ============================================================
# Statistics Tests
# ============================================================


class TestGetStats:
    """Test statistics functionality."""

    @pytest.mark.asyncio
    async def test_should_return_store_statistics(self, entity_store):
        """Should return store statistics."""
        await entity_store.store_entity(
            canonical_id="entity-1",
            entity_type="TypeA",
            properties={},
        )
        await entity_store.store_entity(
            canonical_id="entity-2",
            entity_type="TypeB",
            properties={},
            embedding=list(np.random.randn(384)),
        )

        stats = await entity_store.get_stats()

        assert stats["user_id"] == "test-user"
        assert stats["namespace"] == "test-namespace"
        assert stats["total_entities"] == 2
        assert stats["entities_with_embeddings"] == 1
        assert "TypeA" in stats["entity_types"]
        assert "TypeB" in stats["entity_types"]


# ============================================================
# Clear Tests
# ============================================================


class TestClear:
    """Test clear functionality."""

    @pytest.mark.asyncio
    async def test_should_clear_all_entities(self, entity_store):
        """Should clear all entities from store."""
        await entity_store.store_entity(
            canonical_id="entity-1",
            entity_type="TestEntity",
            properties={},
        )
        await entity_store.store_entity(
            canonical_id="entity-2",
            entity_type="TestEntity",
            properties={},
        )

        await entity_store.clear()

        stats = await entity_store.get_stats()
        assert stats["total_entities"] == 0


# ============================================================
# CrossDocumentResolver Tests
# ============================================================


class TestCrossDocumentResolverResolveEntity:
    """Test entity resolution functionality."""

    @pytest.mark.asyncio
    async def test_should_find_existing_entity_by_properties(
        self, cross_document_resolver, entity_store
    ):
        """Should find existing entity with matching properties."""
        # Store an entity
        await entity_store.store_entity(
            canonical_id="existing-123",
            entity_type="TestEntity",
            properties={"name": "Test Entity", "code": "TEST001"},
        )

        # Try to resolve similar entity
        matched_id, is_new, confidence = await cross_document_resolver.resolve_entity(
            entity_type="TestEntity",
            properties={"name": "Test Entity", "code": "TEST001"},
        )

        assert matched_id == "existing-123"
        assert is_new is False
        assert confidence >= 0.9

    @pytest.mark.asyncio
    async def test_should_return_new_for_unmatched_entity(
        self, cross_document_resolver
    ):
        """Should indicate new entity when no match found."""
        matched_id, is_new, confidence = await cross_document_resolver.resolve_entity(
            entity_type="TestEntity",
            properties={"name": "Brand New Entity"},
        )

        assert matched_id is None
        assert is_new is True
        assert confidence == 1.0


class TestCrossDocumentResolverResolveAndStore:
    """Test resolve and store functionality."""

    @pytest.mark.asyncio
    async def test_should_store_new_entity(self, cross_document_resolver, entity_store):
        """Should store entity when no match found."""
        final_id, was_merged = await cross_document_resolver.resolve_and_store(
            canonical_id="new-entity-123",
            entity_type="TestEntity",
            properties={"name": "New Entity"},
        )

        assert final_id == "new-entity-123"
        assert was_merged is False

        # Verify entity was stored
        entity = await entity_store.get_entity("new-entity-123")
        assert entity is not None

    @pytest.mark.asyncio
    async def test_should_merge_with_existing_entity(
        self, cross_document_resolver, entity_store
    ):
        """Should merge with existing entity when match found."""
        # Store existing entity
        await entity_store.store_entity(
            canonical_id="existing-456",
            entity_type="TestEntity",
            properties={"name": "Existing Entity"},
        )

        # Resolve with matching properties
        final_id, was_merged = await cross_document_resolver.resolve_and_store(
            canonical_id="new-attempt-789",
            entity_type="TestEntity",
            properties={"name": "Existing Entity"},
        )

        assert final_id == "existing-456"
        assert was_merged is True
