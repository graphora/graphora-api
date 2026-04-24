"""Tests for enhanced context builder with relationship-aware entity extraction."""

import pytest

from graphora_server.services.extraction.context_builder import EnhancedContextBuilder
from graphora_server.services.extraction.config import ContextConfig
from graphora_server.services.transform.models import BaseNode


@pytest.fixture
def sample_ontology():
    """Sample ontology with entities and relationships."""
    return {
        "entities": {
            "Person": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string"},
                    "role": {"type": "string"},
                },
                "relationships": {
                    "WORKS_FOR": {"target": "Company", "cardinality": "many-to-one"},
                    "MANAGES": {"target": "Person", "cardinality": "one-to-many"},
                },
            },
            "Company": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "industry": {"type": "string"},
                },
                "relationships": {
                    "LOCATED_IN": {"target": "Location", "cardinality": "many-to-one"},
                    "OWNS": {"target": "Product", "cardinality": "one-to-many"},
                },
            },
            "Location": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "country": {"type": "string"},
                },
            },
            "Product": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "category": {"type": "string"},
                },
            },
        },
        "relationships": {},
    }


@pytest.fixture
def sample_nodes():
    """Sample nodes for testing."""
    return [
        BaseNode(
            id="person-1",
            type="Person",
            properties={"name": "John Doe", "role": "Engineer"},
            confidence_score=0.9,
        ),
        BaseNode(
            id="company-1",
            type="Company",
            properties={"name": "Acme Corp", "industry": "Technology"},
            confidence_score=0.85,
        ),
    ]


@pytest.fixture
def context_builder(sample_ontology):
    """Create context builder with sample ontology."""
    return EnhancedContextBuilder(sample_ontology)


class TestRelationshipSchemaHints:
    """Tests for relationship schema hint generation."""

    def test_builds_relationship_schema_from_ontology(self, context_builder):
        """Test that relationship schema is correctly built from ontology."""
        schema = context_builder._compute_relationship_schema()

        # Person should have outgoing relationships
        assert "Person" in schema
        person_patterns = schema["Person"]
        person_rel_types = {p["type"] for p in person_patterns}
        assert "WORKS_FOR" in person_rel_types
        assert "MANAGES" in person_rel_types

        # Company should have both outgoing and incoming relationships
        assert "Company" in schema
        company_patterns = schema["Company"]
        company_rel_types = {p["type"] for p in company_patterns}
        assert "LOCATED_IN" in company_rel_types
        assert "WORKS_FOR" in company_rel_types  # Incoming from Person

    def test_relationship_schema_includes_cardinality(self, context_builder):
        """Test that cardinality is included in schema."""
        schema = context_builder._compute_relationship_schema()

        person_works_for = next(
            (p for p in schema["Person"] if p["type"] == "WORKS_FOR"), None
        )
        assert person_works_for is not None
        assert person_works_for["cardinality"] == "many-to-one"

    def test_build_relationship_schema_hints(self, context_builder):
        """Test building human-readable relationship hints."""
        entity_types = {"Person", "Company"}
        hints = context_builder._build_relationship_schema_hints(entity_types)

        # Should have hints for relationships
        assert len(hints) > 0

        # Should contain arrow patterns
        hint_text = "\n".join(hints)
        assert "-[:" in hint_text
        assert "WORKS_FOR" in hint_text

    def test_schema_caching(self, context_builder):
        """Test that relationship schema is cached."""
        # First call computes schema
        schema1 = context_builder._compute_relationship_schema()
        context_builder._relationship_schema_cache = schema1

        # Second call should use cache
        entity_types = {"Person"}
        hints1 = context_builder._build_relationship_schema_hints(entity_types)
        hints2 = context_builder._build_relationship_schema_hints(entity_types)

        assert hints1 == hints2


class TestRelationshipAwareEntityContext:
    """Tests for relationship-aware entity context building."""

    def test_includes_relationship_patterns_header(self, context_builder, sample_nodes):
        """Test that context includes relationship patterns section."""
        envelope = context_builder.build_relationship_aware_entity_context(
            sample_nodes, include_confidence=True
        )

        assert "RELATIONSHIP PATTERNS" in envelope.text
        assert "for context during entity extraction" in envelope.text

    def test_includes_guidance_notes(self, context_builder, sample_nodes):
        """Test that context includes guidance for extraction."""
        envelope = context_builder.build_relationship_aware_entity_context(
            sample_nodes, include_confidence=True
        )

        assert "consider what relationships" in envelope.text.lower()
        assert "missing entities may be implied" in envelope.text.lower()

    def test_includes_existing_entities(self, context_builder, sample_nodes):
        """Test that context includes previously identified entities."""
        envelope = context_builder.build_relationship_aware_entity_context(
            sample_nodes, include_confidence=True
        )

        assert "PREVIOUSLY IDENTIFIED ENTITIES" in envelope.text
        assert "John Doe" in envelope.text
        assert "Acme Corp" in envelope.text

    def test_empty_nodes_returns_placeholder(self, context_builder):
        """Test handling of empty nodes list."""
        envelope = context_builder.build_relationship_aware_entity_context(
            [], include_confidence=True
        )

        assert "No entities extracted yet" in envelope.text

    def test_entity_count_tracking(self, context_builder, sample_nodes):
        """Test that entity count is tracked in envelope."""
        envelope = context_builder.build_relationship_aware_entity_context(
            sample_nodes, include_confidence=True
        )

        assert envelope.entity_count == len(sample_nodes)


class TestExpectedEntityTypes:
    """Tests for identifying expected entity types from relationships."""

    def test_identifies_expected_types(self, context_builder, sample_nodes):
        """Test identification of expected entity types."""
        expected = context_builder.get_expected_entity_types_from_relationships(
            sample_nodes
        )

        # Person WORKS_FOR Company - Company exists
        # Person MANAGES Person - Person exists
        # Company LOCATED_IN Location - Location missing
        # Company OWNS Product - Product missing
        assert "Location" in expected
        assert "Product" in expected

    def test_no_false_positives_for_existing_types(self, context_builder, sample_nodes):
        """Test that existing types are not flagged as expected."""
        expected = context_builder.get_expected_entity_types_from_relationships(
            sample_nodes
        )

        # Person and Company already exist
        assert "Person" not in expected
        assert "Company" not in expected

    def test_empty_nodes_returns_empty_set(self, context_builder):
        """Test that empty nodes returns empty expected types."""
        expected = context_builder.get_expected_entity_types_from_relationships([])
        assert len(expected) == 0


class TestBuildNodeContextWithHints:
    """Tests for build_node_context with relationship hints parameter."""

    def test_includes_hints_when_requested(self, context_builder, sample_nodes):
        """Test that relationship hints are included when requested."""
        envelope = context_builder.build_node_context(
            sample_nodes,
            include_confidence=True,
            include_relationship_hints=True,
        )

        # Should contain relationship patterns
        text = envelope.text
        assert "-[:" in text or "RELATIONSHIP" in text

    def test_excludes_hints_when_not_requested(self, context_builder, sample_nodes):
        """Test that relationship hints are excluded by default."""
        envelope = context_builder.build_node_context(
            sample_nodes,
            include_confidence=True,
            include_relationship_hints=False,
        )

        # Should not have explicit relationship section header
        assert "RELATIONSHIP PATTERNS" not in envelope.text


class TestContextEnvelopeMetadata:
    """Tests for context envelope metadata."""

    def test_truncation_flag(self, sample_ontology, sample_nodes):
        """Test that truncation is properly flagged."""
        # Create builder with very small max chars
        config = ContextConfig(max_context_chars=100)
        builder = EnhancedContextBuilder(sample_ontology, config)

        envelope = builder.build_relationship_aware_entity_context(
            sample_nodes, include_confidence=True
        )

        # Should be truncated due to small limit
        assert envelope.truncated is True
        assert envelope.raw_length > len(envelope.text)

    def test_raw_length_tracking(self, context_builder, sample_nodes):
        """Test that raw length is tracked."""
        envelope = context_builder.build_relationship_aware_entity_context(
            sample_nodes, include_confidence=True
        )

        assert envelope.raw_length > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ontology_without_relationships(self):
        """Test handling of ontology without relationship definitions."""
        ontology = {
            "entities": {
                "SimpleEntity": {
                    "properties": {"name": {"type": "string"}},
                    # No relationships defined
                },
            },
        }
        builder = EnhancedContextBuilder(ontology)

        nodes = [
            BaseNode(
                id="simple-1",
                type="SimpleEntity",
                properties={"name": "Test"},
            )
        ]

        # Should not raise
        envelope = builder.build_relationship_aware_entity_context(nodes)
        assert envelope.text is not None

    def test_ontology_with_empty_relationships(self):
        """Test handling of entity with empty relationships dict."""
        ontology = {
            "entities": {
                "EmptyRelEntity": {
                    "properties": {"name": {"type": "string"}},
                    "relationships": {},
                },
            },
        }
        builder = EnhancedContextBuilder(ontology)

        schema = builder._compute_relationship_schema()
        assert "EmptyRelEntity" in schema
        assert len(schema["EmptyRelEntity"]) == 0

    def test_handles_missing_target_type(self):
        """Test handling of relationship with missing target type."""
        ontology = {
            "entities": {
                "BadEntity": {
                    "properties": {"name": {"type": "string"}},
                    "relationships": {
                        "BAD_REL": {},  # Missing target
                    },
                },
            },
        }
        builder = EnhancedContextBuilder(ontology)

        schema = builder._compute_relationship_schema()
        # Should handle gracefully with Unknown target
        bad_patterns = [
            p for p in schema.get("BadEntity", []) if p["type"] == "BAD_REL"
        ]
        assert len(bad_patterns) == 1
        assert bad_patterns[0]["target"] == "Unknown"

    def test_standalone_relationship_definitions(self):
        """Test handling of relationships defined at top level."""
        ontology = {
            "entities": {
                "EntityA": {"properties": {"name": {"type": "string"}}},
                "EntityB": {"properties": {"name": {"type": "string"}}},
            },
            "relationships": {
                "STANDALONE_REL": {
                    "source": "EntityA",
                    "target": "EntityB",
                    "cardinality": "one-to-many",
                },
            },
        }
        builder = EnhancedContextBuilder(ontology)

        schema = builder._compute_relationship_schema()

        # EntityA should have outgoing STANDALONE_REL
        a_patterns = schema.get("EntityA", [])
        standalone_out = [p for p in a_patterns if p["type"] == "STANDALONE_REL"]
        assert len(standalone_out) == 1
        assert standalone_out[0]["direction"] == "outgoing"

        # EntityB should have incoming STANDALONE_REL
        b_patterns = schema.get("EntityB", [])
        standalone_in = [p for p in b_patterns if p["type"] == "STANDALONE_REL"]
        assert len(standalone_in) == 1
        assert standalone_in[0]["direction"] == "incoming"
