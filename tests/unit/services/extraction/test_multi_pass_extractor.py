"""Unit tests for MultiPassExtractor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graphora_server.services.extraction.multi_pass_extractor import MultiPassExtractor
from graphora_server.services.extraction.config import MultiPassConfig
from graphora_server.services.extraction.models import ExtractionGap, GapType
from graphora_server.services.transform.models import BaseNode, RelationshipInstance


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
                "graphora_server.services.extraction.multi_pass_extractor.transform_as_nodes",
                return_value=[],
            ),
            patch(
                "graphora_server.services.extraction.multi_pass_extractor.transform_as_relationships",
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
                "graphora_server.services.extraction.multi_pass_extractor.transform_as_nodes",
                return_value=[],
            ),
            patch(
                "graphora_server.services.extraction.multi_pass_extractor.transform_as_relationships",
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


# ============================================================
# B5-obs slice 3: model routing — refinement pass swaps the
# extractor model when REFINEMENT_MODEL is configured. These
# tests pin the resolution helper and the wiring through
# _refinement_pass → _extract_for_gaps → LLMClient calls.
# ============================================================


class TestRefinementModelRouting:
    def test_resolve_refinement_model_falls_through_when_unset(self, monkeypatch):
        """Pre-slice-3 behavior: REFINEMENT_MODEL unset (None or
        empty) → refinement pass uses the same extractor_model as
        pass 1. Single-model deployments keep working unchanged."""
        from graphora_server.services.extraction.multi_pass_extractor import (
            _resolve_refinement_model,
        )
        from graphora_server.config import settings

        monkeypatch.setattr(settings, "REFINEMENT_MODEL", None)
        assert _resolve_refinement_model("gemini-1.5-flash") == "gemini-1.5-flash"

        # Empty string is also "unset" — env var unset and env var
        # set to "" both reach pydantic-settings as the same value
        # in some configurations. Pin both forms.
        monkeypatch.setattr(settings, "REFINEMENT_MODEL", "")
        assert _resolve_refinement_model("gemini-1.5-flash") == "gemini-1.5-flash"

    def test_resolve_refinement_model_uses_setting_when_set(self, monkeypatch):
        """When REFINEMENT_MODEL is configured, the refinement
        pass uses it instead of the user's primary model."""
        from graphora_server.services.extraction.multi_pass_extractor import (
            _resolve_refinement_model,
        )
        from graphora_server.config import settings

        monkeypatch.setattr(settings, "REFINEMENT_MODEL", "gemini-2.5-pro")
        assert _resolve_refinement_model("gemini-1.5-flash") == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_refinement_pass_routes_model_override(
        self, mock_ontology_parser, mock_llm_client, monkeypatch
    ):
        """End-to-end pin: REFINEMENT_MODEL configured → the
        refinement-pass LLM calls receive ``model_override=<setting>``.
        The initial pass calls do NOT carry the override — they
        run on the user's primary model.

        Repro: a validation gap kicks off a refinement pass; we
        verify the second batch of LLM calls (refinement) carries
        model_override while the first batch (initial) doesn't."""
        from graphora_server.services.extraction.config import MultiPassConfig
        from graphora_server.services.extraction.models import (
            ValidationResult,
            ExtractionGap,
            GapType,
        )
        from graphora_server.config import settings

        monkeypatch.setattr(settings, "REFINEMENT_MODEL", "gemini-2.5-pro")

        # Make the validator surface a high-severity gap so the
        # refinement pass actually runs. Without this, the loop
        # short-circuits after pass 1.
        gap = ExtractionGap(
            gap_type=GapType.LOW_CONFIDENCE,
            severity=0.9,  # above default gap_severity_threshold (0.5)
            description="forced gap",
            chunk_indices=[0],
        )
        validation_with_gap = ValidationResult(
            is_valid=False,
            overall_confidence=0.5,
            gaps=[gap],
        )
        validation_clean = ValidationResult(
            is_valid=True,
            overall_confidence=0.95,
            gaps=[],
        )
        # First validate() (after pass 1) finds the gap; second
        # (after refinement) is clean so the loop exits.
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [
            validation_with_gap,
            validation_clean,
        ]

        mock_llm_client.extract_nodes_from_chunk.return_value = MagicMock()
        mock_llm_client.extract_relationships_from_chunk.return_value = MagicMock()

        config = MultiPassConfig(max_passes=2)
        extractor = MultiPassExtractor(
            mock_ontology_parser, mock_llm_client, config=config
        )
        extractor.validator = mock_validator

        # Force the refinement-result path to "is_improvement=True"
        # by patching the merge helpers — we don't care about the
        # final node/edge state here, only the LLM call shape.
        with (
            patch(
                "graphora_server.services.extraction.multi_pass_extractor.transform_as_nodes",
                return_value=[],
            ),
            patch(
                "graphora_server.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ),
        ):
            await extractor.extract(
                chunks=["chunk1"],
                transform_id="tx-1",
                user_id="u-1",
                max_passes=2,
                extractor_model="gemini-1.5-flash",
            )

        # Pin the model_override flow across the call sequence.
        # Each call to extract_nodes_from_chunk surfaces its kwargs;
        # the initial pass calls must have model_override=None
        # (single-model behavior on the primary path), and any
        # refinement-pass call must carry model_override=<setting>.
        nodes_calls = mock_llm_client.extract_nodes_from_chunk.call_args_list
        # The first pass-1 call is at index 0; its model_override
        # is None (or absent — kwargs default).
        first_override = nodes_calls[0].kwargs.get("model_override")
        assert first_override is None, (
            f"Initial pass should run on the primary model "
            f"(model_override=None), got {first_override!r}"
        )
        # If a refinement call fired, it carries the routing
        # model. We allow >=1 refinement call — depending on
        # how many gaps surfaced. The presence of the override on
        # at least one call is the contract.
        override_calls = [
            c for c in nodes_calls if c.kwargs.get("model_override") is not None
        ]
        assert override_calls, (
            "Refinement pass did not pass model_override through. "
            "Either the validator didn't surface the forced gap or "
            "the routing wiring is broken."
        )
        assert all(
            c.kwargs.get("model_override") == "gemini-2.5-pro" for c in override_calls
        ), (
            f"Refinement-pass calls used the wrong model: "
            f"{[c.kwargs.get('model_override') for c in override_calls]!r}"
        )

    @pytest.mark.asyncio
    async def test_refinement_pass_no_override_when_setting_unset(
        self, mock_ontology_parser, mock_llm_client, monkeypatch
    ):
        """When REFINEMENT_MODEL is unset, both passes run on the
        user's primary model and NO LLM call carries
        model_override (single-model behavior preserved). The
        cache-key collision concern from slice 3's design relies
        on the override being None when there's no separate
        refinement model, so this pin doubles as a cache-key
        invariant."""
        from graphora_server.services.extraction.config import MultiPassConfig
        from graphora_server.services.extraction.models import (
            ValidationResult,
            ExtractionGap,
            GapType,
        )
        from graphora_server.config import settings

        monkeypatch.setattr(settings, "REFINEMENT_MODEL", None)

        gap = ExtractionGap(
            gap_type=GapType.LOW_CONFIDENCE,
            severity=0.9,  # above default gap_severity_threshold (0.5)
            description="forced gap",
            chunk_indices=[0],
        )
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [
            ValidationResult(is_valid=False, overall_confidence=0.5, gaps=[gap]),
            ValidationResult(is_valid=True, overall_confidence=0.95, gaps=[]),
        ]

        mock_llm_client.extract_nodes_from_chunk.return_value = MagicMock()
        mock_llm_client.extract_relationships_from_chunk.return_value = MagicMock()

        extractor = MultiPassExtractor(
            mock_ontology_parser,
            mock_llm_client,
            config=MultiPassConfig(max_passes=2),
        )
        extractor.validator = mock_validator

        with (
            patch(
                "graphora_server.services.extraction.multi_pass_extractor.transform_as_nodes",
                return_value=[],
            ),
            patch(
                "graphora_server.services.extraction.multi_pass_extractor.transform_as_relationships",
                return_value=[],
            ),
        ):
            await extractor.extract(
                chunks=["chunk1"],
                transform_id="tx-1",
                user_id="u-1",
                max_passes=2,
                extractor_model="gemini-1.5-flash",
            )

        # No call should carry model_override when the setting
        # is unset. This is the "pre-slice-3 path" — pin so a
        # regression that always passes the override surfaces here.
        nodes_calls = mock_llm_client.extract_nodes_from_chunk.call_args_list
        all_overrides = [c.kwargs.get("model_override") for c in nodes_calls]
        assert all(o is None for o in all_overrides), (
            f"Some LLM calls received a non-None model_override even "
            f"though REFINEMENT_MODEL is unset: {all_overrides!r}"
        )
