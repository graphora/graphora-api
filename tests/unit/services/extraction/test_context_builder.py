"""Unit tests for EnhancedContextBuilder."""

import pytest
from graphora_server.services.extraction.context_builder import EnhancedContextBuilder
from graphora_server.services.extraction.config import ContextConfig
from graphora_server.services.extraction.models import ExtractionGap, GapType, ValidationResult
from graphora_server.services.transform.models import BaseNode, RelationshipInstance


@pytest.fixture
def sample_ontology():
    """Create a sample ontology for testing."""
    return {
        "version": "1.0",
        "entities": {
            "Person": {
                "properties": {
                    "name": {
                        "type": "str",
                        "required": True,
                        "description": "Full name",
                    },
                    "email": {
                        "type": "str",
                        "required": True,
                        "description": "Email address",
                    },
                },
                "relationships": {
                    "WORKS_AT": {"target": "Organization"},
                    "KNOWS": {"target": "Person"},
                },
            },
            "Organization": {
                "properties": {
                    "name": {
                        "type": "str",
                        "required": True,
                        "description": "Org name",
                    },
                },
            },
        },
    }


@pytest.fixture
def context_builder(sample_ontology):
    """Create an EnhancedContextBuilder instance."""
    return EnhancedContextBuilder(sample_ontology)


@pytest.fixture
def person_node():
    """Create a Person node."""
    return BaseNode(
        id="person-1",
        type="Person",
        properties={"name": "John Doe", "email": "john@example.com"},
        confidence_score=0.9,
    )


@pytest.fixture
def person_node_low_confidence():
    """Create a low confidence Person node."""
    return BaseNode(
        id="person-2",
        type="Person",
        properties={"name": "Jane Doe"},
        confidence_score=0.5,
    )


@pytest.fixture
def organization_node():
    """Create an Organization node."""
    return BaseNode(
        id="org-1",
        type="Organization",
        properties={"name": "Acme Corp"},
        confidence_score=0.85,
    )


@pytest.fixture
def relationship():
    """Create a WORKS_AT relationship."""
    return RelationshipInstance(
        id="rel-1",
        type="WORKS_AT",
        source_id="person-1",
        target_id="org-1",
        source_type="Person",
        target_type="Organization",
        properties={"since": "2020"},
        confidence_score=0.8,
    )


class TestEnhancedContextBuilder:
    """Tests for EnhancedContextBuilder initialization."""

    def test_initialization_default_config(self, sample_ontology):
        """Test builder initializes with default config."""
        builder = EnhancedContextBuilder(sample_ontology)

        assert builder.ontology == sample_ontology
        assert isinstance(builder.config, ContextConfig)

    def test_initialization_custom_config(self, sample_ontology):
        """Test builder initializes with custom config."""
        config = ContextConfig(max_context_chars=4096)
        builder = EnhancedContextBuilder(sample_ontology, config=config)

        assert builder.config.max_context_chars == 4096


class TestBuildNodeContext:
    """Tests for build_node_context method."""

    def test_build_node_context_empty(self, context_builder):
        """Test context for empty node list."""
        envelope = context_builder.build_node_context([])

        assert envelope.text == "No entities extracted yet."
        assert envelope.truncated is False
        assert envelope.entity_count == 0

    def test_build_node_context_single_node(self, context_builder, person_node):
        """Test context for single node."""
        envelope = context_builder.build_node_context([person_node])

        assert "Person" in envelope.text
        assert "person-1" in envelope.text
        assert "John Doe" in envelope.text
        assert envelope.entity_count == 1

    def test_build_node_context_includes_confidence(self, context_builder, person_node):
        """Test context includes confidence scores by default."""
        envelope = context_builder.build_node_context(
            [person_node], include_confidence=True
        )

        assert "confidence" in envelope.text
        assert "0.90" in envelope.text

    def test_build_node_context_excludes_confidence(self, context_builder, person_node):
        """Test context excludes confidence when disabled."""
        envelope = context_builder.build_node_context(
            [person_node], include_confidence=False
        )

        assert "confidence" not in envelope.text

    def test_build_node_context_with_validation(self, context_builder, person_node):
        """Test context includes validation summary."""
        validation_result = ValidationResult(
            overall_confidence=0.85,
            property_completeness=0.9,
            recommendations=["Check entity completeness"],
        )

        envelope = context_builder.build_node_context(
            [person_node],
            include_validation=True,
            validation_result=validation_result,
        )

        assert "VALIDATION SUMMARY" in envelope.text
        assert "0.85" in envelope.text

    def test_build_node_context_prioritizes_low_confidence(self, sample_ontology):
        """Test low confidence nodes appear first when prioritized."""
        config = ContextConfig(prioritize_low_confidence=True)
        builder = EnhancedContextBuilder(sample_ontology, config=config)

        high_conf = BaseNode(
            id="high",
            type="Person",
            properties={"name": "High Conf"},
            confidence_score=0.9,
        )
        low_conf = BaseNode(
            id="low",
            type="Person",
            properties={"name": "Low Conf"},
            confidence_score=0.3,
        )

        envelope = builder.build_node_context([high_conf, low_conf])

        # Low confidence should appear first
        low_pos = envelope.text.find("Low Conf")
        high_pos = envelope.text.find("High Conf")
        assert low_pos < high_pos


class TestBuildRelationshipContext:
    """Tests for build_relationship_context method."""

    def test_build_relationship_context_empty(self, context_builder):
        """Test context for empty lists."""
        envelope = context_builder.build_relationship_context([], [])

        assert "No entities or relationships extracted yet." in envelope.text

    def test_build_relationship_context_with_relationships(
        self, context_builder, person_node, organization_node, relationship
    ):
        """Test context includes relationship information."""
        envelope = context_builder.build_relationship_context(
            [person_node, organization_node], [relationship]
        )

        assert "WORKS_AT" in envelope.text
        assert "person-1" in envelope.text
        assert "org-1" in envelope.text
        assert envelope.relationship_count == 1

    def test_build_relationship_context_highlights_orphans(
        self, context_builder, person_node, organization_node
    ):
        """Test orphan nodes are highlighted when included."""
        envelope = context_builder.build_relationship_context(
            [person_node, organization_node],
            [],  # No relationships
            include_orphans=True,
        )

        assert "WITHOUT RELATIONSHIPS" in envelope.text
        assert "person-1" in envelope.text
        assert "org-1" in envelope.text

    def test_build_relationship_context_no_orphans(
        self, context_builder, person_node, organization_node
    ):
        """Test orphans not shown when disabled."""
        envelope = context_builder.build_relationship_context(
            [person_node, organization_node],
            [],
            include_orphans=False,
        )

        assert "WITHOUT RELATIONSHIPS" not in envelope.text


class TestBuildRefinementContext:
    """Tests for build_refinement_context method."""

    def test_build_refinement_context_incomplete_entity(
        self, context_builder, person_node
    ):
        """Test refinement context for incomplete entity gap."""
        gap = ExtractionGap(
            gap_type=GapType.INCOMPLETE_ENTITY,
            entity_type="Person",
            entity_id="person-1",
            description="Missing email property",
            context={"missing_required": ["email"]},
        )

        context = context_builder.build_refinement_context(
            gaps=[gap],
            chunk_text="Some text about John Doe",
            existing_nodes=[person_node],
        )

        assert "EXTRACTION GAPS TO ADDRESS" in context
        assert "incomplete_entity" in context
        assert "EXTRACTION INSTRUCTIONS" in context

    def test_build_refinement_context_orphan_node(self, context_builder, person_node):
        """Test refinement context for orphan node gap."""
        gap = ExtractionGap(
            gap_type=GapType.ORPHAN_NODE,
            entity_type="Person",
            entity_id="person-1",
            description="Node has no relationships",
        )

        context = context_builder.build_refinement_context(
            gaps=[gap],
            chunk_text="John works at Acme Corp",
            existing_nodes=[person_node],
        )

        assert "orphan_node" in context
        assert "RELEVANT EXISTING ENTITIES" in context

    def test_build_refinement_context_with_focus_entities(
        self, context_builder, person_node, organization_node
    ):
        """Test refinement context focuses on specified entities."""
        gap = ExtractionGap(
            gap_type=GapType.LOW_CONFIDENCE,
            entity_id="person-1",
        )

        context = context_builder.build_refinement_context(
            gaps=[gap],
            chunk_text="Some text",
            existing_nodes=[person_node, organization_node],
            focus_entities=["person-1"],
        )

        assert "person-1" in context
        # org-1 should not be in relevant entities section


class TestBuildGapSpecificContext:
    """Tests for build_gap_specific_context method."""

    def test_gap_context_incomplete_entity(self, context_builder, person_node):
        """Test gap-specific context for incomplete entity."""
        gap = ExtractionGap(
            gap_type=GapType.INCOMPLETE_ENTITY,
            entity_type="Person",
            entity_id="person-1",
            description="Entity incomplete",
            severity=0.7,
            context={"missing_required": ["phone"]},
        )

        context = context_builder.build_gap_specific_context(
            gap=gap,
            nodes=[person_node],
            relationships=[],
        )

        assert "ADDRESSING:" in context
        assert "incomplete_entity" in context
        assert "0.70" in context
        assert "Current Entity State" in context

    def test_gap_context_orphan_node(self, context_builder, person_node):
        """Test gap-specific context for orphan node."""
        gap = ExtractionGap(
            gap_type=GapType.ORPHAN_NODE,
            entity_type="Person",
            entity_id="person-1",
            description="Node has no relationships",
        )

        context = context_builder.build_gap_specific_context(
            gap=gap,
            nodes=[person_node],
            relationships=[],
        )

        assert "Orphan Node" in context
        assert "Possible Relationships" in context
        assert "WORKS_AT" in context

    def test_gap_context_missing_relationship(
        self, context_builder, person_node, organization_node
    ):
        """Test gap-specific context for missing relationship."""
        gap = ExtractionGap(
            gap_type=GapType.MISSING_RELATIONSHIP,
            entity_type="Person",
            entity_id="person-1",
            description="Missing WORKS_AT relationship",
            context={
                "relationship_type": "WORKS_AT",
                "target_type": "Organization",
            },
        )

        context = context_builder.build_gap_specific_context(
            gap=gap,
            nodes=[person_node, organization_node],
            relationships=[],
        )

        assert "Expected Relationship: WORKS_AT" in context
        assert "Target Entity Type: Organization" in context
        assert "Potential Targets" in context


class TestContextTruncation:
    """Tests for context truncation."""

    def test_truncation_head_tail(self, sample_ontology):
        """Test head_tail truncation strategy."""
        config = ContextConfig(max_context_chars=100, truncation_strategy="head_tail")
        builder = EnhancedContextBuilder(sample_ontology, config=config)

        # Create many nodes to exceed limit
        nodes = [
            BaseNode(
                id=f"node-{i}",
                type="Person",
                properties={"name": f"Person {i}"},
            )
            for i in range(20)
        ]

        envelope = builder.build_node_context(nodes, include_confidence=False)

        assert envelope.truncated is True
        assert len(envelope.text) <= 100
        assert "...[truncated]..." in envelope.text

    def test_truncation_head_only(self, sample_ontology):
        """Test head truncation strategy."""
        config = ContextConfig(max_context_chars=100, truncation_strategy="head")
        builder = EnhancedContextBuilder(sample_ontology, config=config)

        nodes = [
            BaseNode(
                id=f"node-{i}",
                type="Person",
                properties={"name": f"Person {i}"},
            )
            for i in range(20)
        ]

        envelope = builder.build_node_context(nodes, include_confidence=False)

        assert envelope.truncated is True
        assert len(envelope.text) <= 100

    def test_no_truncation_within_limit(self, context_builder, person_node):
        """Test no truncation when within limit."""
        envelope = context_builder.build_node_context([person_node])

        assert envelope.truncated is False
