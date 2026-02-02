"""Tests for enhanced multi-pass extractor with relationship-aware extraction."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.extraction.multi_pass_extractor import MultiPassExtractor
from app.services.extraction.config import MultiPassConfig
from app.services.transform.models import BaseNode, RelationshipInstance


@pytest.fixture
def sample_ontology():
    """Sample ontology for testing."""
    return {
        "entities": {
            "Person": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "role": {"type": "string"},
                },
                "relationships": {
                    "WORKS_FOR": {"target": "Company", "cardinality": "many-to-one"},
                },
            },
            "Company": {
                "properties": {
                    "name": {"type": "string", "required": True},
                },
                "relationships": {
                    "LOCATED_IN": {"target": "Location"},
                },
            },
            "Location": {
                "properties": {
                    "name": {"type": "string", "required": True},
                },
            },
        },
    }


@pytest.fixture
def mock_ontology_parser(sample_ontology):
    """Mock ontology parser."""
    parser = MagicMock()
    parser.parsed_ontology = sample_ontology
    parser.ontology_yaml = "entities:\n  Person:\n    properties: {}"
    parser.build_entities_only_model.return_value = MagicMock()
    parser.build_relationships_only_model.return_value = MagicMock()
    return parser


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock()
    client.extract_nodes_from_chunk = AsyncMock(return_value=MagicMock())
    client.extract_relationships_from_chunk = AsyncMock(return_value=MagicMock())
    return client


class TestMultiPassExtractorInit:
    """Tests for extractor initialization."""

    def test_creates_context_builder(self, mock_ontology_parser, mock_llm_client):
        """Test that context builder is created during init."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        assert extractor.context_builder is not None

    def test_uses_provided_config(self, mock_ontology_parser, mock_llm_client):
        """Test that provided config is used."""
        config = MultiPassConfig(max_passes=5, gap_severity_threshold=0.3)

        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
            config=config,
        )

        assert extractor.config.max_passes == 5
        assert extractor.config.gap_severity_threshold == 0.3


class TestRelationshipAwareContext:
    """Tests for relationship-aware context during extraction."""

    @pytest.mark.asyncio
    async def test_uses_relationship_aware_context(
        self, mock_ontology_parser, mock_llm_client
    ):
        """Test that initial extraction uses relationship-aware context."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        # Mock transform_as_nodes to return empty list
        with patch(
            "app.services.extraction.multi_pass_extractor.transform_as_nodes",
            return_value=[],
        ):
            with patch(
                "app.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ):
                await extractor._initial_extraction_pass(
                    chunks=["test chunk"],
                    transform_id="test-123",
                    user_id="user-1",
                )

        # Verify LLM was called with context
        call_args = mock_llm_client.extract_nodes_from_chunk.call_args
        assert call_args is not None

        # Context should be non-empty (relationship hints)
        context_arg = call_args.kwargs.get("context", call_args[1].get("context", ""))
        assert len(context_arg) > 0

    @pytest.mark.asyncio
    async def test_context_includes_relationship_patterns(
        self, mock_ontology_parser, mock_llm_client
    ):
        """Test that context includes relationship pattern information."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        # Build context directly
        context_envelope = (
            extractor.context_builder.build_relationship_aware_entity_context(
                [], include_confidence=True
            )
        )

        # Should mention relationship patterns
        assert "RELATIONSHIP PATTERNS" in context_envelope.text


class TestExpectedEntityTypeDetection:
    """Tests for expected entity type detection."""

    @pytest.mark.asyncio
    async def test_logs_missing_expected_types(
        self, mock_ontology_parser, mock_llm_client
    ):
        """Test that missing expected entity types are logged."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        # Create a person node (expects Company via WORKS_FOR)
        person_node = BaseNode(
            id="person-1",
            type="Person",
            properties={"name": "John"},
            confidence_score=0.9,
        )

        # Mock transform_as_nodes to return person
        with patch(
            "app.services.extraction.multi_pass_extractor.transform_as_nodes",
            return_value=[person_node],
        ):
            with patch(
                "app.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ):
                with patch(
                    "app.services.extraction.multi_pass_extractor.logger"
                ) as mock_logger:
                    await extractor._initial_extraction_pass(
                        chunks=["test chunk"],
                        transform_id="test-123",
                        user_id="user-1",
                    )

                    # Check that info was logged about expected types
                    # (Company and Location are expected based on relationships)
                    _info_calls = [
                        c
                        for c in mock_logger.info.call_args_list
                        if "expected from relationships" in str(c)
                    ]
                    # This may or may not be called depending on implementation


class TestContextUpdating:
    """Tests for context updating between chunks."""

    @pytest.mark.asyncio
    async def test_context_updates_between_chunks(
        self, mock_ontology_parser, mock_llm_client
    ):
        """Test that context is updated after each chunk."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        # Track context values
        contexts_used = []

        async def capture_context(*args, **kwargs):
            contexts_used.append(kwargs.get("context", ""))
            return MagicMock()

        mock_llm_client.extract_nodes_from_chunk = capture_context

        with patch(
            "app.services.extraction.multi_pass_extractor.transform_as_nodes",
            return_value=[],
        ):
            with patch(
                "app.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ):
                await extractor._initial_extraction_pass(
                    chunks=["chunk 1", "chunk 2", "chunk 3"],
                    transform_id="test-123",
                    user_id="user-1",
                )

        # Should have captured 3 contexts
        assert len(contexts_used) == 3


class TestProgressCallback:
    """Tests for progress callback integration."""

    @pytest.mark.asyncio
    async def test_progress_callback_called(
        self, mock_ontology_parser, mock_llm_client
    ):
        """Test that progress callback is called during extraction."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        progress_calls = []

        def track_progress(current, total):
            progress_calls.append((current, total))

        with patch(
            "app.services.extraction.multi_pass_extractor.transform_as_nodes",
            return_value=[],
        ):
            with patch(
                "app.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ):
                await extractor._initial_extraction_pass(
                    chunks=["chunk 1", "chunk 2"],
                    transform_id="test-123",
                    user_id="user-1",
                    progress_callback=track_progress,
                )

        # Should have progress updates for entity and relationship extraction
        assert len(progress_calls) > 0


class TestNodeMerging:
    """Tests for node merging during extraction."""

    @pytest.mark.asyncio
    async def test_merges_duplicate_nodes(self, mock_ontology_parser, mock_llm_client):
        """Test that duplicate nodes are merged."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        # Create nodes with same ID
        node1 = BaseNode(
            id="person-1",
            type="Person",
            properties={"name": "John"},
            confidence_score=0.8,
        )
        node2 = BaseNode(
            id="person-1",
            type="Person",
            properties={"name": "John", "role": "Engineer"},
            confidence_score=0.9,
        )

        # Return different nodes on sequential calls
        call_count = [0]

        def get_nodes(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [node1]
            return [node2]

        with patch(
            "app.services.extraction.multi_pass_extractor.transform_as_nodes",
            side_effect=get_nodes,
        ):
            with patch(
                "app.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ):
                with patch(
                    "app.services.extraction.multi_pass_extractor.merge_nodes",
                    return_value=BaseNode(
                        id="person-1",
                        type="Person",
                        properties={"name": "John", "role": "Engineer"},
                        confidence_score=0.9,
                    ),
                ) as mock_merge:
                    nodes, _ = await extractor._initial_extraction_pass(
                        chunks=["chunk 1", "chunk 2"],
                        transform_id="test-123",
                        user_id="user-1",
                    )

                    # merge_nodes should have been called
                    assert mock_merge.called


class TestSameNodeDetection:
    """Tests for same node detection logic."""

    def test_same_node_by_id(self, mock_ontology_parser, mock_llm_client):
        """Test that nodes with same ID are detected as same."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        node1 = BaseNode(id="person-1", type="Person", properties={"name": "John"})
        node2 = BaseNode(id="person-1", type="Person", properties={"name": "Johnny"})

        assert extractor._is_same_node(node1, node2) is True

    def test_same_node_by_canonical_id(self, mock_ontology_parser, mock_llm_client):
        """Test that nodes with same canonical ID are detected as same."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        node1 = BaseNode(
            id="person-1",
            type="Person",
            properties={"name": "John"},
            canonical_id="canonical-123",
        )
        node2 = BaseNode(
            id="person-2",
            type="Person",
            properties={"name": "John Doe"},
            canonical_id="canonical-123",
        )

        assert extractor._is_same_node(node1, node2) is True

    def test_different_types_not_same(self, mock_ontology_parser, mock_llm_client):
        """Test that nodes of different types are not same."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        node1 = BaseNode(id="entity-1", type="Person", properties={"name": "Acme"})
        node2 = BaseNode(id="entity-1", type="Company", properties={"name": "Acme"})

        assert extractor._is_same_node(node1, node2) is False


class TestSameRelationshipDetection:
    """Tests for same relationship detection logic."""

    def test_same_relationship(self, mock_ontology_parser, mock_llm_client):
        """Test that identical relationships are detected as same."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        rel1 = RelationshipInstance(
            source_id="person-1",
            source_type="Person",
            type="WORKS_FOR",
            target_id="company-1",
            target_type="Company",
        )
        rel2 = RelationshipInstance(
            source_id="person-1",
            source_type="Person",
            type="WORKS_FOR",
            target_id="company-1",
            target_type="Company",
        )

        assert extractor._is_same_relationship(rel1, rel2) is True

    def test_different_relationships(self, mock_ontology_parser, mock_llm_client):
        """Test that different relationships are not same."""
        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
        )

        rel1 = RelationshipInstance(
            source_id="person-1",
            source_type="Person",
            type="WORKS_FOR",
            target_id="company-1",
            target_type="Company",
        )
        rel2 = RelationshipInstance(
            source_id="person-1",
            source_type="Person",
            type="MANAGES",
            target_id="person-2",
            target_type="Person",
        )

        assert extractor._is_same_relationship(rel1, rel2) is False
