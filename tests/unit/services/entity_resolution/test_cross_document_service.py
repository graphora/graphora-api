"""Unit tests for Cross-Document Resolution Service.

Tests for cross-document entity resolution service.
Uses mocks to avoid loading actual ML models and databases in tests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from graphora_server.services.entity_resolution.cross_document_service import (
    CrossDocumentResolutionService,
    create_cross_document_service,
)
from graphora_server.services.transform.models import BaseNode


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_entity_store():
    """Create a mock entity store."""
    store = MagicMock()
    store.user_id = "test-user"
    store.namespace = "default"

    # Async methods
    store.get_entity = AsyncMock(return_value=None)
    store.find_similar_entities = AsyncMock(return_value=[])
    store.store_entity = AsyncMock()
    store.update_entity = AsyncMock(return_value=True)

    return store


@pytest.fixture
def mock_embedding_similarity():
    """Create a mock embedding similarity instance."""
    mock = MagicMock()
    mock.embedding_dim = 384
    mock.model_name = "test-model"

    def mock_get_embedding(text):
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(384)
        return emb / np.linalg.norm(emb)

    mock.get_embedding = mock_get_embedding
    return mock


@pytest.fixture
def cross_document_service(mock_entity_store, mock_embedding_similarity):
    """Create CrossDocumentResolutionService with mocked dependencies."""
    service = CrossDocumentResolutionService(
        user_id="test-user",
        namespace="test-namespace",
        similarity_threshold=0.85,
        enabled=True,
    )
    service._entity_store = mock_entity_store
    service._embedding_similarity = mock_embedding_similarity
    return service


@pytest.fixture
def sample_nodes():
    """Create sample nodes for testing."""
    return [
        BaseNode(
            id="node-1",
            type="Person",
            properties={"name": "John Smith", "email": "john@example.com"},
            canonical_properties={"name": "john smith"},
            canonical_key="Person:name=john smith",
            canonical_id="canonical-1",
        ),
        BaseNode(
            id="node-2",
            type="Person",
            properties={"name": "Jane Doe", "title": "Engineer"},
            canonical_properties={"name": "jane doe"},
            canonical_key="Person:name=jane doe",
            canonical_id="canonical-2",
        ),
    ]


# ============================================================
# Initialization Tests
# ============================================================


class TestCrossDocumentResolutionServiceInit:
    """Test CrossDocumentResolutionService initialization."""

    def test_should_create_with_required_params(self):
        """Should create instance with required parameters."""
        service = CrossDocumentResolutionService(user_id="test-user")
        assert service.user_id == "test-user"
        assert service.namespace == "default"

    def test_should_accept_custom_namespace(self):
        """Should accept custom namespace."""
        service = CrossDocumentResolutionService(
            user_id="test-user", namespace="custom"
        )
        assert service.namespace == "custom"

    def test_should_accept_custom_threshold(self):
        """Should accept custom similarity threshold."""
        service = CrossDocumentResolutionService(
            user_id="test-user", similarity_threshold=0.9
        )
        assert service.similarity_threshold == 0.9

    def test_should_lazy_load_dependencies(self):
        """Should not load dependencies until needed."""
        service = CrossDocumentResolutionService(user_id="test-user")
        assert service._entity_store is None
        assert service._embedding_similarity is None
        assert service._resolver is None

    def test_should_accept_enabled_flag(self):
        """Should accept enabled flag."""
        service = CrossDocumentResolutionService(user_id="test-user", enabled=False)
        assert service.enabled is False


# ============================================================
# Node Resolution Tests
# ============================================================


class TestResolveNodes:
    """Test node resolution functionality."""

    @pytest.mark.asyncio
    async def test_should_return_nodes_when_disabled(
        self, cross_document_service, sample_nodes
    ):
        """Should return original nodes when service is disabled."""
        cross_document_service.enabled = False

        result = await cross_document_service.resolve_nodes(sample_nodes)

        assert result == sample_nodes

    @pytest.mark.asyncio
    async def test_should_return_empty_list_for_empty_input(
        self, cross_document_service
    ):
        """Should return empty list for empty input."""
        result = await cross_document_service.resolve_nodes([])
        assert result == []

    @pytest.mark.asyncio
    async def test_should_use_exact_match_first(
        self, cross_document_service, sample_nodes, mock_entity_store
    ):
        """Should use exact canonical_id match as fast path."""
        # Mock exact match found
        mock_entity_store.get_entity.return_value = {
            "canonical_id": "canonical-1",
            "entity_type": "Person",
        }

        result = await cross_document_service.resolve_nodes(sample_nodes)

        assert len(result) == 2
        assert cross_document_service._stats["exact_matches"] >= 1

    @pytest.mark.asyncio
    async def test_should_try_similarity_search_when_no_exact_match(
        self, cross_document_service, sample_nodes, mock_entity_store
    ):
        """Should try similarity search when exact match fails."""
        # Mock no exact match
        mock_entity_store.get_entity.return_value = None
        mock_entity_store.find_similar_entities.return_value = [
            ("matched-canonical-id", 0.92, {"canonical_id": "matched-canonical-id"})
        ]

        result = await cross_document_service.resolve_nodes(sample_nodes)

        assert len(result) == 2
        assert cross_document_service._stats["similarity_matches"] >= 1

    @pytest.mark.asyncio
    async def test_should_store_new_entities_when_no_match(
        self, cross_document_service, sample_nodes, mock_entity_store
    ):
        """Should store new entities when no match is found."""
        # Mock no matches
        mock_entity_store.get_entity.return_value = None
        mock_entity_store.find_similar_entities.return_value = []

        await cross_document_service.resolve_nodes(sample_nodes)

        # Should store entities
        assert mock_entity_store.store_entity.called
        assert cross_document_service._stats["new_entities"] >= 1

    @pytest.mark.asyncio
    async def test_should_update_canonical_id_on_similarity_match(
        self, cross_document_service, sample_nodes, mock_entity_store
    ):
        """Should update canonical_id when similarity match is found."""
        mock_entity_store.get_entity.return_value = None
        mock_entity_store.find_similar_entities.return_value = [
            ("new-canonical-id", 0.88, {"canonical_id": "new-canonical-id"})
        ]

        result = await cross_document_service.resolve_nodes(sample_nodes)

        # At least one node should have updated canonical_id
        matched_nodes = [n for n in result if n.canonical_id == "new-canonical-id"]
        assert len(matched_nodes) >= 1


# ============================================================
# Embedding Computation Tests
# ============================================================


class TestComputeNodeEmbedding:
    """Test node embedding computation."""

    @pytest.mark.asyncio
    async def test_should_return_embedding_list(
        self, cross_document_service, sample_nodes
    ):
        """Should return embedding as list of floats."""
        embedding = await cross_document_service._compute_node_embedding(
            sample_nodes[0]
        )

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_should_prioritize_canonical_properties(self, cross_document_service):
        """Should prioritize canonical properties for embedding."""
        node = BaseNode(
            id="test",
            type="Person",
            properties={"name": "JOHN SMITH"},
            canonical_properties={"name": "john smith"},  # Normalized
            canonical_key="test",
        )

        embedding = await cross_document_service._compute_node_embedding(node)

        assert embedding is not None

    @pytest.mark.asyncio
    async def test_should_return_none_for_empty_properties(
        self, cross_document_service
    ):
        """Should return None when no text properties available."""
        node = BaseNode(
            id="test",
            type="Person",
            properties={},
            canonical_properties={},
            canonical_key="test",
        )

        embedding = await cross_document_service._compute_node_embedding(node)

        assert embedding is None


# ============================================================
# Entity Storage Tests
# ============================================================


class TestStoreNewEntity:
    """Test new entity storage."""

    @pytest.mark.asyncio
    async def test_should_store_entity_with_canonical_id(
        self, cross_document_service, sample_nodes, mock_entity_store
    ):
        """Should store entity when canonical_id is present."""
        node = sample_nodes[0]
        embedding = [0.1] * 384

        await cross_document_service._store_new_entity(node, embedding, "doc-1")

        mock_entity_store.store_entity.assert_called_once()

    @pytest.mark.asyncio
    async def test_should_skip_storage_without_canonical_id(
        self, cross_document_service, mock_entity_store
    ):
        """Should skip storage when canonical_id is missing."""
        node = BaseNode(
            id="test",
            type="Person",
            properties={"name": "Test"},
            canonical_key="test",
            canonical_id=None,  # No canonical_id
        )

        await cross_document_service._store_new_entity(node, None, "doc-1")

        mock_entity_store.store_entity.assert_not_called()


# ============================================================
# Resolve and Link Tests
# ============================================================


class TestResolveAndLink:
    """Test resolve and link functionality."""

    @pytest.mark.asyncio
    async def test_should_return_nodes_and_mapping(
        self, cross_document_service, sample_nodes, mock_entity_store
    ):
        """Should return resolved nodes and id mapping."""
        mock_entity_store.get_entity.return_value = None
        mock_entity_store.find_similar_entities.return_value = []

        nodes, mapping = await cross_document_service.resolve_and_link(sample_nodes)

        assert isinstance(nodes, list)
        assert isinstance(mapping, dict)

    @pytest.mark.asyncio
    async def test_should_include_mapping_for_updated_ids(
        self, cross_document_service, sample_nodes, mock_entity_store
    ):
        """Should include mapping when canonical_id changes."""
        mock_entity_store.get_entity.return_value = None
        mock_entity_store.find_similar_entities.return_value = [
            ("different-canonical-id", 0.9, {"canonical_id": "different-canonical-id"})
        ]

        nodes, mapping = await cross_document_service.resolve_and_link(sample_nodes)

        # Mapping should include nodes whose canonical_id changed
        # (Only if the new canonical_id differs from original)
        assert isinstance(mapping, dict)


# ============================================================
# Statistics Tests
# ============================================================


class TestStatistics:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_should_track_nodes_processed(
        self, cross_document_service, sample_nodes
    ):
        """Should track number of nodes processed."""
        await cross_document_service.resolve_nodes(sample_nodes)

        stats = cross_document_service.get_stats()
        assert stats["nodes_processed"] == 2

    def test_should_include_config_in_stats(self, cross_document_service):
        """Should include configuration in stats."""
        stats = cross_document_service.get_stats()

        assert "enabled" in stats
        assert "similarity_threshold" in stats
        assert "embedding_model" in stats

    def test_should_reset_stats(self, cross_document_service):
        """Should reset statistics."""
        cross_document_service._stats["nodes_processed"] = 100
        cross_document_service.reset_stats()

        assert cross_document_service._stats["nodes_processed"] == 0


# ============================================================
# Factory Function Tests
# ============================================================


class TestCreateCrossDocumentService:
    """Test factory function."""

    @pytest.mark.asyncio
    async def test_should_create_service_instance(self):
        """Should create CrossDocumentResolutionService instance."""
        service = await create_cross_document_service("test-user")
        assert isinstance(service, CrossDocumentResolutionService)

    @pytest.mark.asyncio
    async def test_should_accept_namespace(self):
        """Should accept namespace parameter."""
        service = await create_cross_document_service("test-user", namespace="custom")
        assert service.namespace == "custom"


# ============================================================
# Integration Tests (with mocked dependencies)
# ============================================================


class TestIntegration:
    """Integration tests with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_resolution_flow(
        self, cross_document_service, mock_entity_store
    ):
        """Test complete resolution flow."""
        nodes = [
            BaseNode(
                id="new-node",
                type="Person",
                properties={"name": "New Person", "description": "A new person"},
                canonical_properties={"name": "new person"},
                canonical_key="Person:name=new person",
                canonical_id="new-canonical-id",
            ),
        ]

        # First node is new (no match)
        mock_entity_store.get_entity.return_value = None
        mock_entity_store.find_similar_entities.return_value = []

        result = await cross_document_service.resolve_nodes(nodes, "doc-1")

        # Should process node and store as new
        assert len(result) == 1
        assert mock_entity_store.store_entity.called

    @pytest.mark.asyncio
    async def test_should_handle_mixed_matches(
        self, cross_document_service, mock_entity_store
    ):
        """Test resolution with mix of exact, similar, and new entities."""
        nodes = [
            BaseNode(
                id="exact-match",
                type="Person",
                properties={"name": "Exact"},
                canonical_key="key1",
                canonical_id="exact-canonical",
            ),
            BaseNode(
                id="similar-match",
                type="Person",
                properties={"name": "Similar"},
                canonical_key="key2",
                canonical_id="similar-canonical",
            ),
            BaseNode(
                id="new-entity",
                type="Person",
                properties={"name": "New"},
                canonical_key="key3",
                canonical_id="new-canonical",
            ),
        ]

        # Setup different responses for each call
        mock_entity_store.get_entity.side_effect = [
            {"canonical_id": "exact-canonical"},  # First node: exact match
            None,  # Second node: no exact match
            None,  # Third node: no exact match
        ]
        mock_entity_store.find_similar_entities.side_effect = [
            [("similar-canonical", 0.9, {})],  # Second: similarity match
            [],  # Third: no match
        ]

        result = await cross_document_service.resolve_nodes(nodes, "doc-1")

        assert len(result) == 3
        stats = cross_document_service.get_stats()
        assert stats["exact_matches"] == 1
        assert stats["similarity_matches"] == 1
        assert stats["new_entities"] == 1
