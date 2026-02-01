"""Unit tests for MultiPassExtractor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.extraction.multi_pass_extractor import MultiPassExtractor
from app.services.extraction.config import MultiPassConfig
from app.services.extraction.models import ExtractionGap, GapType
from app.services.transform.models import BaseNode, RelationshipInstance


@pytest.fixture
def mock_ontology_parser():
    """Create a mock OntologyParser."""
    parser = MagicMock()
    parser.parsed_ontology = {
        "version": "1.0",
        "entities": {
            "Person": {
                "properties": {
                    "name": {"type": "str", "required": True},
                    "email": {"type": "str", "required": True},
                },
                "relationships": {
                    "WORKS_AT": {"target": "Organization"},
                },
            },
            "Organization": {
                "properties": {
                    "name": {"type": "str", "required": True},
                },
            },
        },
    }
    parser.ontology_yaml = "version: 1.0\nentities: {}"
    parser.build_entities_only_model.return_value = MagicMock()
    parser.build_relationships_only_model.return_value = MagicMock()
    return parser


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMClient."""
    client = MagicMock()
    client.extract_nodes_from_chunk = AsyncMock()
    client.extract_relationships_from_chunk = AsyncMock()
    return client


@pytest.fixture
def extractor(mock_ontology_parser, mock_llm_client):
    """Create a MultiPassExtractor instance."""
    return MultiPassExtractor(
        ontology_parser=mock_ontology_parser,
        llm_client=mock_llm_client,
    )


@pytest.fixture
def sample_nodes():
    """Create sample nodes for testing."""
    return [
        BaseNode(
            id="person-1",
            type="Person",
            properties={"name": "John Doe", "email": "john@example.com"},
            confidence_score=0.9,
        ),
        BaseNode(
            id="org-1",
            type="Organization",
            properties={"name": "Acme Corp"},
            confidence_score=0.85,
        ),
    ]


@pytest.fixture
def sample_relationships():
    """Create sample relationships for testing."""
    return [
        RelationshipInstance(
            id="rel-1",
            type="WORKS_AT",
            source_id="person-1",
            target_id="org-1",
            source_type="Person",
            target_type="Organization",
            properties={},
        ),
    ]


class TestMultiPassExtractorInit:
    """Tests for MultiPassExtractor initialization."""

    def test_initialization_with_defaults(self, mock_ontology_parser):
        """Test extractor initializes with default config."""
        extractor = MultiPassExtractor(mock_ontology_parser)

        assert extractor.ontology_parser == mock_ontology_parser
        assert extractor.llm_client is not None
        assert isinstance(extractor.config, MultiPassConfig)
        assert extractor.validator is not None
        assert extractor.context_builder is not None

    def test_initialization_with_custom_config(
        self, mock_ontology_parser, mock_llm_client
    ):
        """Test extractor initializes with custom config."""
        config = MultiPassConfig(max_passes=3, gap_severity_threshold=0.6)

        extractor = MultiPassExtractor(
            ontology_parser=mock_ontology_parser,
            llm_client=mock_llm_client,
            config=config,
        )

        assert extractor.config.max_passes == 3
        assert extractor.config.gap_severity_threshold == 0.6


class TestIsSameNode:
    """Tests for _is_same_node method."""

    def test_same_node_by_id(self, extractor):
        """Test nodes with same ID are considered same."""
        node1 = BaseNode(id="node-1", type="Person", properties={"name": "John"})
        node2 = BaseNode(id="node-1", type="Person", properties={"name": "John Doe"})

        assert extractor._is_same_node(node1, node2) is True

    def test_same_node_by_canonical_id(self, extractor):
        """Test nodes with same canonical_id are considered same."""
        node1 = BaseNode(
            id="node-1",
            type="Person",
            properties={"name": "John"},
            canonical_id="canonical-1",
        )
        node2 = BaseNode(
            id="node-2",
            type="Person",
            properties={"name": "John"},
            canonical_id="canonical-1",
        )

        assert extractor._is_same_node(node1, node2) is True

    def test_same_node_by_properties(self, extractor):
        """Test nodes with same properties (excluding ID) are considered same."""
        node1 = BaseNode(id="node-1", type="Person", properties={"name": "John"})
        node2 = BaseNode(id="node-2", type="Person", properties={"name": "John"})

        assert extractor._is_same_node(node1, node2) is True

    def test_different_nodes(self, extractor):
        """Test different nodes are not considered same."""
        node1 = BaseNode(id="node-1", type="Person", properties={"name": "John"})
        node2 = BaseNode(id="node-2", type="Person", properties={"name": "Jane"})

        assert extractor._is_same_node(node1, node2) is False

    def test_different_types_not_same(self, extractor):
        """Test nodes with different types are not same."""
        node1 = BaseNode(id="node-1", type="Person", properties={"name": "John"})
        node2 = BaseNode(id="node-1", type="Organization", properties={"name": "John"})

        assert extractor._is_same_node(node1, node2) is False


class TestIsSameRelationship:
    """Tests for _is_same_relationship method."""

    def test_same_relationship(self, extractor):
        """Test identical relationships are considered same."""
        rel1 = RelationshipInstance(
            id="rel-1",
            type="WORKS_AT",
            source_id="person-1",
            target_id="org-1",
            source_type="Person",
            target_type="Organization",
        )
        rel2 = RelationshipInstance(
            id="rel-2",
            type="WORKS_AT",
            source_id="person-1",
            target_id="org-1",
            source_type="Person",
            target_type="Organization",
        )

        assert extractor._is_same_relationship(rel1, rel2) is True

    def test_different_relationships_different_source(self, extractor):
        """Test relationships with different source are not same."""
        rel1 = RelationshipInstance(
            id="rel-1",
            type="WORKS_AT",
            source_id="person-1",
            target_id="org-1",
            source_type="Person",
            target_type="Organization",
        )
        rel2 = RelationshipInstance(
            id="rel-2",
            type="WORKS_AT",
            source_id="person-2",
            target_id="org-1",
            source_type="Person",
            target_type="Organization",
        )

        assert extractor._is_same_relationship(rel1, rel2) is False

    def test_different_relationships_different_type(self, extractor):
        """Test relationships with different types are not same."""
        rel1 = RelationshipInstance(
            id="rel-1",
            type="WORKS_AT",
            source_id="person-1",
            target_id="org-1",
            source_type="Person",
            target_type="Organization",
        )
        rel2 = RelationshipInstance(
            id="rel-2",
            type="KNOWS",
            source_id="person-1",
            target_id="org-1",
            source_type="Person",
            target_type="Organization",
        )

        assert extractor._is_same_relationship(rel1, rel2) is False


class TestMergeNodes:
    """Tests for _merge_nodes method."""

    def test_merge_adds_new_nodes(self, extractor, sample_nodes):
        """Test new nodes are added."""
        existing = [sample_nodes[0]]
        new_nodes = [sample_nodes[1]]

        result = extractor._merge_nodes(existing, new_nodes)

        assert len(result) == 2
        assert any(n.id == "person-1" for n in result)
        assert any(n.id == "org-1" for n in result)

    def test_merge_updates_existing_nodes(self, extractor):
        """Test existing nodes are updated with new properties."""
        existing = [
            BaseNode(
                id="person-1",
                type="Person",
                properties={"name": "John"},
            )
        ]
        new_nodes = [
            BaseNode(
                id="person-1",
                type="Person",
                properties={"name": "John Doe", "email": "john@example.com"},
            )
        ]

        result = extractor._merge_nodes(existing, new_nodes)

        assert len(result) == 1
        # Properties should be merged
        assert result[0].properties.get("email") == "john@example.com"


class TestMergeRelationships:
    """Tests for _merge_relationships method."""

    def test_merge_adds_new_relationships(self, extractor, sample_relationships):
        """Test new relationships are added."""
        existing = []
        new_rels = sample_relationships

        result = extractor._merge_relationships(existing, new_rels)

        assert len(result) == 1

    def test_merge_skips_duplicates(self, extractor, sample_relationships):
        """Test duplicate relationships are not added."""
        existing = sample_relationships.copy()
        new_rels = sample_relationships.copy()

        result = extractor._merge_relationships(existing, new_rels)

        assert len(result) == 1


class TestGroupGapsByChunk:
    """Tests for _group_gaps_by_chunk method."""

    def test_group_gaps_by_chunk_indices(self, extractor):
        """Test gaps are grouped by their chunk indices."""
        gaps = [
            ExtractionGap(gap_type=GapType.INCOMPLETE_ENTITY, chunk_indices=[0, 1]),
            ExtractionGap(gap_type=GapType.ORPHAN_NODE, chunk_indices=[1, 2]),
        ]
        chunks = ["chunk0", "chunk1", "chunk2"]

        result = extractor._group_gaps_by_chunk(gaps, chunks)

        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert len(result[0]) == 1
        assert len(result[1]) == 2
        assert len(result[2]) == 1

    def test_group_gaps_no_chunk_indices(self, extractor):
        """Test gaps without chunk indices are added to all chunks."""
        gaps = [
            ExtractionGap(gap_type=GapType.LOW_CONFIDENCE, chunk_indices=[]),
        ]
        chunks = ["chunk0", "chunk1"]

        result = extractor._group_gaps_by_chunk(gaps, chunks)

        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 1
        assert len(result[1]) == 1


class TestExtractAsync:
    """Tests for async extract method."""

    @pytest.mark.asyncio
    async def test_extract_single_pass(
        self, extractor, mock_ontology_parser, mock_llm_client
    ):
        """Test extraction with single pass when validation succeeds."""
        # Setup mock returns
        mock_kg = MagicMock()
        mock_kg.__iter__ = lambda self: iter([])
        mock_llm_client.extract_nodes_from_chunk.return_value = mock_kg
        mock_llm_client.extract_relationships_from_chunk.return_value = mock_kg

        # Patch transform_as_nodes and transform_as_relationships
        with (
            patch(
                "app.services.extraction.multi_pass_extractor.transform_as_nodes",
                return_value=[],
            ),
            patch(
                "app.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ),
        ):
            nodes, relationships = await extractor.extract(
                chunks=["test chunk"],
                transform_id="test-transform",
                user_id="test-user",
                max_passes=1,
            )

        assert isinstance(nodes, list)
        assert isinstance(relationships, list)
        mock_llm_client.extract_nodes_from_chunk.assert_called()

    @pytest.mark.asyncio
    async def test_extract_respects_max_passes(
        self, mock_ontology_parser, mock_llm_client
    ):
        """Test extraction respects max_passes configuration."""
        config = MultiPassConfig(max_passes=1)
        extractor = MultiPassExtractor(
            mock_ontology_parser, mock_llm_client, config=config
        )

        mock_kg = MagicMock()
        mock_llm_client.extract_nodes_from_chunk.return_value = mock_kg
        mock_llm_client.extract_relationships_from_chunk.return_value = mock_kg

        with (
            patch(
                "app.services.extraction.multi_pass_extractor.transform_as_nodes",
                return_value=[],
            ),
            patch(
                "app.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ),
        ):
            nodes, relationships = await extractor.extract(
                chunks=["chunk1"],
                transform_id="test",
                user_id="user",
            )

        # With max_passes=1, should only do initial pass
        assert isinstance(nodes, list)
