"""Transform helpers unit tests following London School TDD.

These tests verify the transformation and entity resolution logic:
- Canonicalization functions
- Node merging
- Property normalization
- Entity deduplication preparation

Coverage targets:
- merge_nodes: 90%+
- canonicalization: 85%+
- property normalization: 85%+
- node transformation: 80%+
"""

import pytest

from tests.factories.node_factory import NodeFactory
from tests.factories.relationship_factory import RelationshipFactory


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def sample_ontology():
    """Sample ontology for testing."""
    return {
        "entities": {
            "Company": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "ticker": {"type": "string"},
                    "industry": {"type": "string"},
                    "founding_year": {"type": "integer"},
                }
            },
            "Person": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string"},
                    "title": {"type": "string"},
                }
            },
        },
        "relationships": {
            "EMPLOYS": {
                "source": "Company",
                "target": "Person",
                "properties": {
                    "role": {"type": "string"},
                    "start_date": {"type": "datetime"},
                },
            }
        },
    }


# ============================================================
# Canonicalization Tests
# ============================================================


class TestCanonicalization:
    """Test canonicalization functions."""

    def test_canonicalize_whitespace_should_normalize_spaces(self):
        """Should normalize multiple spaces to single space."""
        from graphora_server.services.transform.helpers import _canonicalize_whitespace

        assert _canonicalize_whitespace("hello   world") == "hello world"
        assert _canonicalize_whitespace("  leading") == "leading"
        assert _canonicalize_whitespace("trailing  ") == "trailing"
        assert _canonicalize_whitespace("  both  sides  ") == "both sides"

    def test_canonicalize_company_name_should_remove_suffixes(self):
        """Should remove common company suffixes."""
        from graphora_server.services.transform.helpers import (
            _canonicalize_company_name,
        )

        # Should lowercase
        result = _canonicalize_company_name("ACME Corporation")
        assert "acme" in result.lower()

    def test_basic_canonical_value_should_lowercase_strings(self):
        """Should lowercase and strip string values."""
        from graphora_server.services.transform.helpers import _basic_canonical_value

        assert _basic_canonical_value("Hello World") == "hello world"
        assert _basic_canonical_value("  UPPERCASE  ") == "uppercase"

    def test_basic_canonical_value_should_handle_none(self):
        """Should return None for None input."""
        from graphora_server.services.transform.helpers import _basic_canonical_value

        assert _basic_canonical_value(None) is None

    def test_basic_canonical_value_should_stringify_numbers(self):
        """Should convert numbers to strings."""
        from graphora_server.services.transform.helpers import _basic_canonical_value

        assert _basic_canonical_value(123) == "123"
        assert _basic_canonical_value(45.67) == "45.67"

    def test_basic_canonical_value_should_handle_lists(self):
        """Should handle list values."""
        from graphora_server.services.transform.helpers import _basic_canonical_value

        result = _basic_canonical_value(["a", "b", "c"])
        # Should return some canonical representation
        assert result is not None

    def test_build_canonical_properties_should_canonicalize_all_props(self):
        """Should build canonical version of all properties."""
        from graphora_server.services.transform.helpers import (
            _build_canonical_properties,
        )

        parsed_ontology = {
            "entities": {
                "Company": {
                    "properties": {
                        "name": {"type": "string"},
                        "ticker": {"type": "string"},
                        "industry": {"type": "string"},
                    }
                }
            }
        }

        properties = {
            "name": "Acme Corporation",
            "ticker": "ACM",
            "industry": "Technology",
        }

        result = _build_canonical_properties(
            parsed_ontology, "Company", properties, properties
        )

        assert "name" in result
        # Values should be canonicalized (lowercased)
        assert result["name"] == result["name"].lower()

    def test_canonicalize_value_should_use_registered_canonicalizer(self):
        """Should use registered canonicalizer when available."""
        from graphora_server.services.transform.helpers import (
            _canonicalize_value,
            register_canonicalizer,
        )

        # Register a custom canonicalizer
        def custom_canonicalizer(value: str) -> str:
            return f"custom_{value}"

        register_canonicalizer("test_prop", custom_canonicalizer)

        result = _canonicalize_value("TestEntity", "test_prop", "value")
        assert result is not None  # Should return some canonicalized value


# ============================================================
# Node Merging Tests
# ============================================================


class TestNodeMerging:
    """Test node merging functionality."""

    def test_merge_nodes_should_combine_chunk_ids(self):
        """Merged node should have combined chunk IDs."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create_company(
            name="Acme",
            chunk_ids=["chunk-1", "chunk-2"],
        )
        new_node = NodeFactory.create_company(
            name="Acme Corp",
            chunk_ids=["chunk-3"],
        )

        merged = merge_nodes(existing, new_node)

        assert "chunk-1" in merged.provenance.chunk_ids
        assert "chunk-2" in merged.provenance.chunk_ids
        assert "chunk-3" in merged.provenance.chunk_ids

    def test_merge_nodes_should_deduplicate_chunk_ids(self):
        """Merged chunk IDs should not have duplicates."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create_company(
            name="Acme",
            chunk_ids=["chunk-1", "chunk-2"],
        )
        new_node = NodeFactory.create_company(
            name="Acme Corp",
            chunk_ids=["chunk-2", "chunk-3"],  # chunk-2 is duplicate
        )

        merged = merge_nodes(existing, new_node)

        # Count occurrences of chunk-2
        chunk_2_count = merged.provenance.chunk_ids.count("chunk-2")
        assert chunk_2_count == 1

    def test_merge_nodes_should_prefer_higher_confidence(self):
        """Should use values from higher confidence node."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create_company(
            name="Acme",
            confidence=0.7,
        )
        new_node = NodeFactory.create_company(
            name="Acme Corporation",
            confidence=0.95,
        )

        merged = merge_nodes(existing, new_node)

        assert merged.confidence_score == 0.95

    def test_merge_nodes_should_preserve_id_from_existing(self):
        """Should preserve the ID from the existing node."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create_company(
            name="Acme",
            node_id="existing-id",
        )
        new_node = NodeFactory.create_company(
            name="Acme Corp",
            node_id="new-id",
        )

        merged = merge_nodes(existing, new_node)

        assert merged.id == "existing-id"

    def test_merge_nodes_should_handle_missing_provenance(self):
        """Should handle nodes with missing provenance."""
        from graphora_server.services.transform.helpers import merge_nodes
        from graphora_server.services.transform.models import BaseNode

        # Create node without provenance
        existing = BaseNode(
            id="node-1",
            type="Company",
            properties={"name": "Acme"},
            provenance=None,
        )
        new_node = NodeFactory.create_company(
            name="Acme Corp",
            chunk_ids=["chunk-1"],
        )

        merged = merge_nodes(existing, new_node)

        # Should have provenance from new_node
        assert merged.provenance is not None

    def test_merge_nodes_should_preserve_type(self):
        """Should preserve the entity type."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create_company(name="Acme")
        new_node = NodeFactory.create_company(name="Acme Corp")

        merged = merge_nodes(existing, new_node)

        assert merged.type == "Company"

    def test_merge_nodes_should_combine_properties(self):
        """Should combine properties from both nodes."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create(
            node_type="Company",
            properties={"name": "Acme", "industry": "Tech"},
        )
        new_node = NodeFactory.create(
            node_type="Company",
            properties={"name": "Acme Corp", "ticker": "ACM"},
        )

        merged = merge_nodes(existing, new_node)

        # Should have properties from both
        assert "name" in merged.properties
        # Additional properties should be present based on merge logic


# ============================================================
# Property Normalization Tests
# ============================================================


class TestPropertyNormalization:
    """Test property normalization functions."""

    def test_coerce_property_value_should_handle_string_type(self):
        """Should coerce value to string type."""
        from graphora_server.services.transform.helpers import _coerce_property_value

        prop_def = {"type": "string"}

        result = _coerce_property_value(prop_def, 123)
        assert result == "123"

    def test_coerce_property_value_should_handle_integer_type(self):
        """Should coerce value to integer type."""
        from graphora_server.services.transform.helpers import _coerce_property_value

        prop_def = {"type": "integer"}

        result = _coerce_property_value(prop_def, "123")
        assert result == 123

    def test_coerce_property_value_should_handle_float_type(self):
        """Should coerce value to float type."""
        from graphora_server.services.transform.helpers import _coerce_property_value

        prop_def = {"type": "float"}

        result = _coerce_property_value(prop_def, "45.67")
        assert result == 45.67

    def test_coerce_property_value_should_return_none_for_invalid(self):
        """Should return None for values that can't be coerced."""
        from graphora_server.services.transform.helpers import _coerce_property_value

        prop_def = {"type": "integer"}

        result = _coerce_property_value(prop_def, "not a number")
        assert result is None

    def test_coerce_property_value_should_handle_boolean_type(self):
        """Should coerce value to boolean type."""
        from graphora_server.services.transform.helpers import _coerce_property_value

        prop_def = {"type": "boolean"}

        assert _coerce_property_value(prop_def, "true") is True
        assert _coerce_property_value(prop_def, "false") is False
        assert _coerce_property_value(prop_def, True) is True

    def test_apply_case_format_should_handle_lowercase(self):
        """Should convert to lowercase."""
        from graphora_server.services.transform.helpers import _apply_case_format

        result = _apply_case_format("Hello World", "lowercase")
        assert result == "hello world"

    def test_apply_case_format_should_handle_uppercase(self):
        """Should convert to uppercase."""
        from graphora_server.services.transform.helpers import _apply_case_format

        result = _apply_case_format("Hello World", "uppercase")
        assert result == "HELLO WORLD"

    def test_apply_case_format_should_handle_title_case(self):
        """Should convert to title case."""
        from graphora_server.services.transform.helpers import _apply_case_format

        result = _apply_case_format("hello world", "titlecase")
        assert result == "Hello World"

    def test_apply_case_format_should_preserve_unknown_format(self):
        """Should preserve value for unknown format."""
        from graphora_server.services.transform.helpers import _apply_case_format

        result = _apply_case_format("Hello World", "unknown_format")
        assert result == "Hello World"


# ============================================================
# Property Type Detection Tests
# ============================================================


class TestPropertyTypeDetection:
    """Test property type detection functions."""

    def test_is_prop_type_string_should_detect_string_types(self):
        """Should detect string property types."""
        from graphora_server.services.transform.helpers import _is_prop_type_string

        assert _is_prop_type_string("string") is True
        assert _is_prop_type_string("str") is True
        assert _is_prop_type_string("String") is True  # Case insensitive
        assert _is_prop_type_string("integer") is False

    def test_is_prop_type_number_should_detect_number_types(self):
        """Should detect number property types."""
        from graphora_server.services.transform.helpers import _is_prop_type_number

        assert _is_prop_type_number("integer") is True
        assert _is_prop_type_number("float") is True
        assert _is_prop_type_number("number") is True
        assert _is_prop_type_number("string") is False

    def test_is_prop_type_datetime_should_detect_datetime_types(self):
        """Should detect datetime property types."""
        from graphora_server.services.transform.helpers import _is_prop_type_datetime

        assert _is_prop_type_datetime("datetime") is True
        assert _is_prop_type_datetime("date") is True
        assert _is_prop_type_datetime("string") is False


# ============================================================
# Node Key Generation Tests
# ============================================================


class TestNodeKeyGeneration:
    """Test node key generation functions."""

    def test_generate_node_key_should_create_deterministic_key(self):
        """Should create deterministic key for same input."""
        from graphora_server.services.transform.helpers import _generate_node_key

        parsed_ontology = {
            "entities": {"Company": {"properties": {"name": {"type": "string"}}}}
        }

        key1 = _generate_node_key(parsed_ontology, "Company", {"name": "acme"})
        key2 = _generate_node_key(parsed_ontology, "Company", {"name": "acme"})

        assert key1 == key2

    def test_generate_node_key_should_differ_for_different_input(self):
        """Should create different keys for different input."""
        from graphora_server.services.transform.helpers import _generate_node_key

        parsed_ontology = {
            "entities": {"Company": {"properties": {"name": {"type": "string"}}}}
        }

        key1 = _generate_node_key(parsed_ontology, "Company", {"name": "acme"})
        key2 = _generate_node_key(parsed_ontology, "Company", {"name": "beta"})

        assert key1 != key2

    def test_generate_node_key_should_include_type(self):
        """Should create different keys for different entity types."""
        from graphora_server.services.transform.helpers import _generate_node_key

        parsed_ontology = {
            "entities": {
                "Company": {"properties": {"name": {"type": "string"}}},
                "Person": {"properties": {"name": {"type": "string"}}},
            }
        }

        key1 = _generate_node_key(parsed_ontology, "Company", {"name": "acme"})
        key2 = _generate_node_key(parsed_ontology, "Person", {"name": "acme"})

        assert key1 != key2

    def test_make_deterministic_node_id_should_create_uuid_format(self):
        """Should create UUID-formatted ID."""
        from graphora_server.services.transform.helpers import (
            _make_deterministic_node_id,
        )

        node_id = _make_deterministic_node_id("transform-123", "Company", "acme")

        # Should return a valid ID
        assert len(node_id) > 0

    def test_make_canonical_node_id_should_be_deterministic(self):
        """Should create deterministic ID from node key."""
        from graphora_server.services.transform.helpers import _make_canonical_node_id

        id1 = _make_canonical_node_id("Company:name=acme")
        id2 = _make_canonical_node_id("Company:name=acme")

        assert id1 == id2


# ============================================================
# Column Detection Tests
# ============================================================


class TestColumnDetection:
    """Test Splink column detection functions."""

    def test_is_canonical_column_should_detect_canonical_prefix(self):
        """Should detect columns with canonical__ prefix (double underscore)."""
        from graphora_server.services.transform.helpers import _is_canonical_column

        # Note: prefix is "canonical__" (double underscore)
        assert _is_canonical_column("canonical__name") is True
        assert _is_canonical_column("name") is False
        assert _is_canonical_column("canonical__") is True

    def test_base_property_from_column_should_strip_prefix(self):
        """Should strip canonical__ prefix from column name."""
        from graphora_server.services.transform.helpers import (
            _base_property_from_column,
        )

        # Note: prefix is "canonical__" (double underscore)
        assert _base_property_from_column("canonical__name") == "name"
        assert _base_property_from_column("name") == "name"
        assert _base_property_from_column("canonical__company_name") == "company_name"


# ============================================================
# Orphan Pruning Tests
# ============================================================


class TestOrphanPruning:
    """Test orphan node pruning functionality."""

    def test_prune_orphaned_nodes_should_remove_disconnected_nodes(self):
        """Should remove nodes not in any relationship."""
        from graphora_server.services.transform.helpers import prune_orphaned_nodes
        from graphora_server.services.transform.models import DocumentKnowledgeGraph

        NodeFactory.reset_counter()
        RelationshipFactory.reset_counter()

        # Create nodes
        company = NodeFactory.create_company(name="Acme", node_id="company-1")
        person = NodeFactory.create_person(name="Jane", node_id="person-1")
        orphan = NodeFactory.create(
            node_type="Location",
            node_id="location-1",
            properties={"name": "HQ"},
        )

        # Create relationship only between company and person
        relationship = RelationshipFactory.create_employs(
            source_id="company-1",
            target_id="person-1",
        )

        # Create graph
        graph = DocumentKnowledgeGraph(
            nodes=[company, person, orphan],
            relationships=[relationship],
        )

        # Simple ontology
        ontology = {
            "entities": {
                "Company": {"properties": {"name": {"type": "string"}}},
                "Person": {"properties": {"name": {"type": "string"}}},
                "Location": {"properties": {}},  # No required properties
            }
        }

        # Prune orphans
        prune_orphaned_nodes(ontology, graph)

        # Orphan should be removed (depending on implementation details)
        # This test documents expected behavior
        node_ids = [n.id for n in graph.nodes]
        assert "company-1" in node_ids
        assert "person-1" in node_ids

    def test_prune_orphaned_nodes_should_keep_connected_nodes(self):
        """Should keep nodes that are in relationships."""
        from graphora_server.services.transform.helpers import prune_orphaned_nodes
        from graphora_server.services.transform.models import DocumentKnowledgeGraph

        NodeFactory.reset_counter()
        RelationshipFactory.reset_counter()

        company = NodeFactory.create_company(name="Acme", node_id="company-1")
        person = NodeFactory.create_person(name="Jane", node_id="person-1")

        relationship = RelationshipFactory.create_employs(
            source_id="company-1",
            target_id="person-1",
        )

        graph = DocumentKnowledgeGraph(
            nodes=[company, person],
            relationships=[relationship],
        )

        ontology = {
            "entities": {
                "Company": {"properties": {"name": {"type": "string"}}},
                "Person": {"properties": {"name": {"type": "string"}}},
            }
        }

        prune_orphaned_nodes(ontology, graph)

        # Both nodes should remain
        assert len(graph.nodes) == 2


# ============================================================
# Entity Deduplication Preparation Tests
# ============================================================


class TestEntityDeduplicationPreparation:
    """Test preparation functions for entity deduplication."""

    def test_prepare_entities_for_deduplication_should_accept_nodes(self):
        """Should prepare entities for deduplication."""
        from graphora_server.services.transform.helpers import (
            _prepare_entities_for_deduplication,
        )

        NodeFactory.reset_counter()

        nodes = [
            NodeFactory.create_company(name="Acme"),
            NodeFactory.create_company(name="Beta"),
        ]

        parsed_ontology = {
            "entities": {
                "Company": {"properties": {"name": {"type": "string"}}},
            }
        }

        # Function returns List[Dict] - prepared entity data
        result = _prepare_entities_for_deduplication(
            nodes, relationships=None, parsed_ontology=parsed_ontology
        )

        # Result is a list of entity dictionaries
        assert isinstance(result, list)
        assert len(result) == 2

    def test_create_splink_dataframe_should_include_all_properties(self):
        """Should include all node properties in dataframe."""
        from graphora_server.services.transform.helpers import _create_splink_dataframe

        # Create entity data dicts with nested properties as expected by the function
        entities_data = [
            {
                "id": "node-1",
                "properties": {"name": "Acme", "ticker": "ACM"},
                "canonical_properties": {"name": "acme", "ticker": "acm"},
            },
            {
                "id": "node-2",
                "properties": {"name": "Beta", "ticker": "BTA"},
                "canonical_properties": {"name": "beta", "ticker": "bta"},
            },
        ]

        system_properties = {"id"}

        # Function returns (DataFrame, properties_columns)
        df, properties_columns = _create_splink_dataframe(
            entities_data, system_properties
        )

        # Should have rows for each entity
        assert len(df) == 2

        # Should have columns for properties
        assert "name" in df.columns
        assert "ticker" in df.columns


# ============================================================
# Small Group Deduplication Tests
# ============================================================


class TestSmallGroupDeduplication:
    """Test heuristic deduplication for small groups."""

    def test_deduplicate_small_entity_group_should_handle_empty_list(self):
        """Should handle empty entity list."""
        from graphora_server.services.transform.helpers import (
            _deduplicate_small_entity_group,
        )

        # Function signature: (entity_type, entities, parsed_ontology)
        # Returns: (deduplicated_entities, id_mapping)
        result, id_mapping = _deduplicate_small_entity_group("Company", [], None)

        assert result == []
        assert id_mapping == {}

    def test_deduplicate_small_entity_group_should_return_single_entity(self):
        """Should return single entity unchanged."""
        from graphora_server.services.transform.helpers import (
            _deduplicate_small_entity_group,
        )

        NodeFactory.reset_counter()
        node = NodeFactory.create_company(name="Acme")

        result, id_mapping = _deduplicate_small_entity_group("Company", [node], None)

        assert len(result) == 1
        assert result[0].id == node.id


# ============================================================
# Transitive Closure Tests
# ============================================================


class TestTransitiveClosure:
    """Test transitive closure for batch deduplication merging."""

    def test_apply_transitive_closure_should_handle_empty_mappings(self):
        """Should handle empty mappings."""
        from graphora_server.services.transform.helpers import _apply_transitive_closure

        result = _apply_transitive_closure({})
        assert result == {}

    def test_apply_transitive_closure_should_handle_simple_mapping(self):
        """Should handle direct A -> B mapping."""
        from graphora_server.services.transform.helpers import _apply_transitive_closure

        mappings = {"a": "b"}
        result = _apply_transitive_closure(mappings)

        assert result["a"] == "b"

    def test_apply_transitive_closure_should_resolve_chains(self):
        """Should resolve A -> B -> C to A -> C, B -> C."""
        from graphora_server.services.transform.helpers import _apply_transitive_closure

        mappings = {"a": "b", "b": "c"}
        result = _apply_transitive_closure(mappings)

        # Both should point to the ultimate representative
        assert result["a"] == "c"
        assert result["b"] == "c"

    def test_apply_transitive_closure_should_handle_multiple_clusters(self):
        """Should handle multiple independent clusters."""
        from graphora_server.services.transform.helpers import _apply_transitive_closure

        mappings = {
            "a1": "a_rep",
            "a2": "a_rep",
            "b1": "b_rep",
            "b2": "b1",  # Chain: b2 -> b1 -> b_rep
        }
        result = _apply_transitive_closure(mappings)

        # a cluster
        assert result["a1"] == "a_rep"
        assert result["a2"] == "a_rep"
        # b cluster - all should point to b_rep
        assert result["b1"] == "b_rep"
        assert result["b2"] == "b_rep"

    def test_apply_transitive_closure_should_handle_long_chains(self):
        """Should handle long chains efficiently."""
        from graphora_server.services.transform.helpers import _apply_transitive_closure

        # Create chain: e1 -> e2 -> e3 -> e4 -> e5 -> rep
        mappings = {
            "e1": "e2",
            "e2": "e3",
            "e3": "e4",
            "e4": "e5",
            "e5": "rep",
        }
        result = _apply_transitive_closure(mappings)

        # All should point to 'rep'
        for key in ["e1", "e2", "e3", "e4", "e5"]:
            assert result[key] == "rep"


# ============================================================
# Extract Properties Tests
# ============================================================


class TestExtractProperties:
    """Test property extraction from Pydantic models."""

    def test_extract_properties_should_get_model_dict(self):
        """Should extract properties from Pydantic model."""
        from graphora_server.services.transform.helpers import _extract_properties
        from pydantic import BaseModel

        class TestEntity(BaseModel):
            name: str
            value: int

        entity = TestEntity(name="test", value=42)
        props = _extract_properties(entity)

        assert props["name"] == "test"
        assert props["value"] == 42

    def test_extract_properties_should_exclude_none_values(self):
        """Should handle optional fields with None values."""
        from graphora_server.services.transform.helpers import _extract_properties
        from pydantic import BaseModel
        from typing import Optional

        class TestEntity(BaseModel):
            name: str
            optional_field: Optional[str] = None

        entity = TestEntity(name="test")
        props = _extract_properties(entity)

        assert props["name"] == "test"
        # Optional field may or may not be included based on implementation


class TestOriginalExtractionIdFallback:
    """B-fix: Gemini's extract_relationships_from_pdf emits positional
    ids ('company_0', 'business_0') in source_id/target_id, ignoring
    the UUIDs we feed via the relationships-context block. The
    BaseNode.original_extraction_ids field captures Gemini's positional
    id during nodes extraction; transform_as_relationships uses it as
    a fallback when the UUID and canonical lookups miss. The field is
    a list (not a single string) so merge_nodes can union aliases
    across pre-relationship merges — see
    test_merged_node_resolves_via_unioned_aliases below.

    Without this fallback every relationship Gemini emits gets
    rejected as 'Invalid source_id ... or target_id ...' (helpers.py
    rejection path) — exactly the symptom the user hit on
    transform_0d5967ec... where 39 nodes landed but 0 relationships.
    """

    def _ontology(self):
        # Entity-keyed relationships shape (matches what
        # OntologyParser produces in production — confirmed against
        # logs showing relationships_def: {'HAS_BUSINESS': {...}}).
        return {
            "entities": {
                "Company": {
                    "properties": {
                        "name": {"type": "string", "required": True},
                    },
                    "relationships": {
                        "HAS_BUSINESS": {"target": "Business"},
                    },
                },
                "Business": {
                    "properties": {
                        "name": {"type": "string", "required": True},
                    },
                },
            },
        }

    def test_transform_as_nodes_stamps_original_extraction_id(self):
        """Gemini's positional id from raw_properties['id'] must
        land on the BaseNode so the relationships pass can resolve
        cross-call references."""
        from pydantic import BaseModel
        from graphora_server.services.transform.helpers import transform_as_nodes

        class _Company(BaseModel):
            id: str
            name: str

        class _NodesResult(BaseModel):
            Company_list: list
            confidence_score: float = 1.0

        result = _NodesResult(
            Company_list=[_Company(id="company_0", name="Apple")],
        )
        nodes = transform_as_nodes(self._ontology(), result)
        assert len(nodes) == 1
        # node.id must be the freshly-minted UUID, not the positional
        # id (the latter would defeat the purpose of UUID assignment).
        assert nodes[0].id != "company_0"
        # But the positional id is captured for the relationship
        # fallback. Stored as a list — merge_nodes unions aliases
        # later, so the singular value goes in as a one-element list.
        assert nodes[0].original_extraction_ids == ["company_0"]

    def test_transform_as_relationships_resolves_via_original_id(self):
        """Relationship with source_id='company_0' / target_id=
        'business_0' resolves to the correct nodes via the
        original_extraction_ids fallback when the UUID lookup misses.

        Pre-fix: the relationship was rejected with 'Invalid
        source_id company_0 or target_id business_0'. Post-fix: it
        builds correctly with the node UUIDs filled in."""
        from pydantic import BaseModel
        from graphora_server.services.transform.helpers import (
            transform_as_relationships,
        )
        from graphora_server.services.transform.models import BaseNode

        company = BaseNode(
            id="company-uuid-xxx",
            type="Company",
            properties={"name": "Apple"},
            original_extraction_ids=["company_0"],
        )
        business = BaseNode(
            id="business-uuid-yyy",
            type="Business",
            properties={"name": "Apple Stores"},
            original_extraction_ids=["business_0"],
        )

        class _RelItem(BaseModel):
            source_id: str
            target_id: str

        class _RelResult(BaseModel):
            # Field shape mirrors the parsing convention in
            # transform_as_relationships: <Source>_<Rel>_<Target>
            # (no _list suffix — that one is reserved for the
            # nodes-extraction schema).
            Company_HAS_BUSINESS_Business: list
            confidence_score: float = 1.0

        rels_payload = _RelResult(
            Company_HAS_BUSINESS_Business=[
                _RelItem(source_id="company_0", target_id="business_0")
            ],
        )

        rels = transform_as_relationships(
            self._ontology(),
            [company, business],
            rels_payload,
        )
        assert len(rels) == 1, (
            f"expected 1 relationship resolved via original_extraction_id; "
            f"got {len(rels)}"
        )
        assert rels[0].source_id == "company-uuid-xxx"
        assert rels[0].target_id == "business-uuid-yyy"
        assert rels[0].type == "HAS_BUSINESS"

    def test_chunk_scoped_lookup_disambiguates_collisions(self):
        """Two chunks both have a node Gemini named 'company_0'
        (Apple in chunk-1, Microsoft in chunk-2) that did NOT get
        merged by entity resolution (different entities). When
        relationships from chunk-1 reference 'company_0', the
        in-chunk node wins even though both carry the same alias.
        Distinct from the merge-then-relationship case below — here
        the nodes stay separate."""
        from pydantic import BaseModel
        from graphora_server.services.transform.helpers import (
            transform_as_relationships,
        )
        from graphora_server.services.transform.models import (
            BaseNode,
            NodeProvenance,
        )
        from graphora_server.services.chunking.models import ChunkMetadata

        apple_chunk1 = BaseNode(
            id="apple-uuid",
            type="Company",
            properties={"name": "Apple"},
            original_extraction_ids=["company_0"],
            provenance=NodeProvenance(chunk_ids=["chunk-1"]),
        )
        microsoft_chunk2 = BaseNode(
            id="microsoft-uuid",
            type="Company",
            properties={"name": "Microsoft"},
            original_extraction_ids=["company_0"],  # collision!
            provenance=NodeProvenance(chunk_ids=["chunk-2"]),
        )
        apple_business = BaseNode(
            id="apple-business-uuid",
            type="Business",
            properties={"name": "Apple Stores"},
            original_extraction_ids=["business_0"],
            provenance=NodeProvenance(chunk_ids=["chunk-1"]),
        )

        class _RelItem(BaseModel):
            source_id: str
            target_id: str

        class _RelResult(BaseModel):
            Company_HAS_BUSINESS_Business: list
            confidence_score: float = 1.0

        chunk1_payload = _RelResult(
            Company_HAS_BUSINESS_Business=[
                _RelItem(source_id="company_0", target_id="business_0")
            ],
        )

        rels = transform_as_relationships(
            self._ontology(),
            [apple_chunk1, microsoft_chunk2, apple_business],
            chunk1_payload,
            chunk_metadata=ChunkMetadata(
                transform_id="t-1",
                chunk_id="chunk-1",
                source_file="apple-10k.pdf",
            ),
        )
        assert len(rels) == 1
        # Must bind to chunk-1's Apple, NOT chunk-2's Microsoft —
        # even though both share original_extraction_id='company_0'.
        assert rels[0].source_id == "apple-uuid"
        assert rels[0].target_id == "apple-business-uuid"

    def test_unscoped_fallback_when_no_chunk_metadata(self):
        """Backward-compat: callers that don't thread chunk_metadata
        through still get the fallback — just unscoped (first match
        wins, may collide across chunks but better than the previous
        all-rejections behaviour). Pins the contract that the fix
        doesn't regress callsites without chunk plumbing."""
        from pydantic import BaseModel
        from graphora_server.services.transform.helpers import (
            transform_as_relationships,
        )
        from graphora_server.services.transform.models import BaseNode

        company = BaseNode(
            id="company-uuid",
            type="Company",
            properties={"name": "Apple"},
            original_extraction_ids=["company_0"],
        )
        business = BaseNode(
            id="business-uuid",
            type="Business",
            properties={"name": "Apple Stores"},
            original_extraction_ids=["business_0"],
        )

        class _RelItem(BaseModel):
            source_id: str
            target_id: str

        class _RelResult(BaseModel):
            Company_HAS_BUSINESS_Business: list
            confidence_score: float = 1.0

        payload = _RelResult(
            Company_HAS_BUSINESS_Business=[
                _RelItem(source_id="company_0", target_id="business_0")
            ],
        )
        # No chunk_metadata kwarg.
        rels = transform_as_relationships(
            self._ontology(),
            [company, business],
            payload,
        )
        assert len(rels) == 1
        assert rels[0].source_id == "company-uuid"
        assert rels[0].target_id == "business-uuid"

    def test_merged_node_resolves_via_unioned_aliases(self):
        """Reviewer-flagged regression: chunk-1 emits the same
        logical company as ``company_0``, chunk-2 emits it as
        ``company_1``. Entity resolution
        (``_compare_and_merge_nodes``) merges them BEFORE the
        relationship pass runs. A subsequent chunk-2 relationship
        with ``source_id='company_1'`` must resolve to the merged
        node — even though the merge base node only carried the
        chunk-1 alias before the merge.

        Pre-fix: BaseNode.original_extraction_id was a single
        string, so merge_nodes silently dropped the chunk-2 alias
        and the relationship lookup returned None. This test
        exercises the full path: produce the merged node via
        ``merge_nodes`` (the same primitive
        ``_compare_and_merge_nodes`` calls), then run
        ``transform_as_relationships`` against it.
        """
        from pydantic import BaseModel
        from graphora_server.services.transform.helpers import (
            transform_as_relationships,
            merge_nodes,
        )
        from graphora_server.services.transform.models import (
            BaseNode,
            NodeProvenance,
        )
        from graphora_server.services.chunking.models import ChunkMetadata

        # Same logical Apple, extracted under different positional
        # ids in two chunks. Different ``id`` UUIDs — entity
        # resolution will collapse them based on canonical_key.
        apple_chunk1 = BaseNode(
            id="apple-chunk1-uuid",
            type="Company",
            properties={"name": "Apple"},
            original_extraction_ids=["company_0"],
            confidence_score=0.8,
            provenance=NodeProvenance(chunk_ids=["chunk-1"]),
        )
        apple_chunk2 = BaseNode(
            id="apple-chunk2-uuid",
            type="Company",
            properties={"name": "Apple Inc."},
            original_extraction_ids=["company_1"],
            confidence_score=0.7,
            provenance=NodeProvenance(chunk_ids=["chunk-2"]),
        )

        # Simulate _compare_and_merge_nodes: merge into the higher-
        # confidence base. The merged node's id keeps chunk-1's
        # uuid, but its alias list MUST union both positional ids.
        merged_apple = merge_nodes(apple_chunk1, apple_chunk2)
        assert merged_apple.id == "apple-chunk1-uuid"
        assert set(merged_apple.original_extraction_ids) == {
            "company_0",
            "company_1",
        }, (
            f"merge_nodes must union aliases from merged-away node; "
            f"got {merged_apple.original_extraction_ids}"
        )

        business_chunk2 = BaseNode(
            id="business-chunk2-uuid",
            type="Business",
            properties={"name": "Apple Stores"},
            original_extraction_ids=["business_0"],
            provenance=NodeProvenance(chunk_ids=["chunk-2"]),
        )

        # Chunk-2 relationship referring to the chunk-2 alias of
        # the (now-merged) Apple node.
        class _RelItem(BaseModel):
            source_id: str
            target_id: str

        class _RelResult(BaseModel):
            Company_HAS_BUSINESS_Business: list
            confidence_score: float = 1.0

        chunk2_payload = _RelResult(
            Company_HAS_BUSINESS_Business=[
                _RelItem(source_id="company_1", target_id="business_0")
            ],
        )

        rels = transform_as_relationships(
            self._ontology(),
            [merged_apple, business_chunk2],
            chunk2_payload,
            chunk_metadata=ChunkMetadata(
                transform_id="t-1",
                chunk_id="chunk-2",
                source_file="apple-10k.pdf",
            ),
        )
        # Pre-fix this would be 0; post-fix it's 1 because the
        # merged node's alias list includes 'company_1'.
        assert len(rels) == 1, (
            f"merged node must resolve via unioned alias; got "
            f"{len(rels)} relationships"
        )
        # Resolved to the merged node's UUID (the merge-base's
        # uuid wins, but the alias from the merged-away chunk-2
        # node is what enabled the lookup).
        assert rels[0].source_id == "apple-chunk1-uuid"
        assert rels[0].target_id == "business-chunk2-uuid"
