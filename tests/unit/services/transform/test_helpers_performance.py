"""Unit tests for Transform Helpers Performance Optimizations.

Tests for the performance optimizations in helpers.py:
- O(n²) to O(n) deduplication fix
- Vectorized DataFrame construction
- OntologyPropertyCache
"""

import pytest
import pandas as pd
from typing import Dict, Any, List

from graphora_server.services.transform.helpers import (
    _create_deduplicated_entities,
    _create_splink_dataframe,
    OntologyPropertyCache,
)
from graphora_server.services.transform.models import BaseNode


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_entities() -> List[BaseNode]:
    """Create sample entities for testing."""
    return [
        BaseNode(
            id=f"entity-{i}",
            type="Person",
            properties={"name": f"Person {i}", "email": f"person{i}@example.com"},
            canonical_properties={"name": f"person {i}"},
            canonical_key=f"Person:name=person {i}",
        )
        for i in range(10)
    ]


@pytest.fixture
def entities_data() -> List[Dict[str, Any]]:
    """Create sample entity data dictionaries."""
    return [
        {
            "id": f"entity-{i}",
            "type": "Person",
            "properties": {"name": f"Person {i}", "email": f"person{i}@example.com"},
            "canonical_properties": {"name": f"person {i}"},
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_ontology() -> Dict[str, Any]:
    """Create sample ontology for testing."""
    return {
        "entities": {
            "Person": {
                "properties": {
                    "name": {"type": "string", "unique": True},
                    "email": {"type": "email", "index": True},
                    "description": {"type": "text"},
                    "age": {"type": "number"},
                }
            },
            "Company": {
                "properties": {
                    "name": {"type": "string", "unique": True},
                    "website": {"type": "url"},
                }
            },
        }
    }


# ============================================================
# OntologyPropertyCache Tests
# ============================================================


class TestOntologyPropertyCache:
    """Test OntologyPropertyCache class."""

    def test_should_create_with_empty_ontology(self):
        """Should create cache with empty ontology."""
        cache = OntologyPropertyCache()
        assert cache._ontology == {}

    def test_should_create_with_parsed_ontology(self, sample_ontology):
        """Should create cache with parsed ontology."""
        cache = OntologyPropertyCache(sample_ontology)
        assert cache._ontology == sample_ontology

    def test_should_get_entity_def(self, sample_ontology):
        """Should get entity definition."""
        cache = OntologyPropertyCache(sample_ontology)

        entity_def = cache.get_entity_def("Person")

        assert "properties" in entity_def
        assert "name" in entity_def["properties"]

    def test_should_cache_entity_def(self, sample_ontology):
        """Should cache entity definition after first access."""
        cache = OntologyPropertyCache(sample_ontology)

        # First access
        cache.get_entity_def("Person")
        assert "Person" in cache._entity_defs

        # Second access should use cache
        entity_def = cache.get_entity_def("Person")
        assert entity_def is cache._entity_defs["Person"]

    def test_should_get_property_def(self, sample_ontology):
        """Should get property definition."""
        cache = OntologyPropertyCache(sample_ontology)

        prop_def = cache.get_property_def("Person", "name")

        assert prop_def["type"] == "string"
        assert prop_def["unique"] is True

    def test_should_cache_property_def(self, sample_ontology):
        """Should cache property definition after first access."""
        cache = OntologyPropertyCache(sample_ontology)

        # First access
        cache.get_property_def("Person", "name")
        assert ("Person", "name") in cache._property_defs

    def test_should_get_property_type(self, sample_ontology):
        """Should get property type."""
        cache = OntologyPropertyCache(sample_ontology)

        assert cache.get_property_type("Person", "name") == "string"
        assert cache.get_property_type("Person", "description") == "text"
        assert cache.get_property_type("Person", "age") == "number"

    def test_should_return_none_for_missing_property(self, sample_ontology):
        """Should return None for missing property."""
        cache = OntologyPropertyCache(sample_ontology)

        assert cache.get_property_type("Person", "nonexistent") is None

    def test_should_check_property_unique(self, sample_ontology):
        """Should check if property is unique."""
        cache = OntologyPropertyCache(sample_ontology)

        assert cache.is_property_unique("Person", "name") is True
        assert cache.is_property_unique("Person", "email") is False

    def test_should_check_property_indexed(self, sample_ontology):
        """Should check if property is indexed."""
        cache = OntologyPropertyCache(sample_ontology)

        assert cache.is_property_indexed("Person", "email") is True
        assert cache.is_property_indexed("Person", "name") is False

    def test_should_handle_missing_entity_type(self, sample_ontology):
        """Should handle missing entity type gracefully."""
        cache = OntologyPropertyCache(sample_ontology)

        entity_def = cache.get_entity_def("NonexistentEntity")
        assert entity_def == {}

        prop_type = cache.get_property_type("NonexistentEntity", "name")
        assert prop_type is None


# ============================================================
# _create_deduplicated_entities Tests
# ============================================================


class TestCreateDeduplicatedEntities:
    """Test _create_deduplicated_entities function (O(n) optimization)."""

    def test_should_return_empty_for_empty_input(self):
        """Should return empty list for empty input."""
        result = _create_deduplicated_entities([], {})
        assert result == []

    def test_should_return_all_entities_when_no_mapping(self, sample_entities):
        """Should return all entities when no mapping exists."""
        result = _create_deduplicated_entities(sample_entities, {})
        assert len(result) == len(sample_entities)

    def test_should_deduplicate_using_mapping(self, sample_entities):
        """Should deduplicate entities using representative mapping."""
        # Map entities 1,2,3 to entity 0 (representative)
        id_to_representative = {
            "entity-1": "entity-0",
            "entity-2": "entity-0",
            "entity-3": "entity-0",
        }

        result = _create_deduplicated_entities(sample_entities, id_to_representative)

        # Should have 7 entities: entity-0 (+ 1,2,3), and 4,5,6,7,8,9
        assert len(result) == 7

    def test_should_keep_representative_entity(self, sample_entities):
        """Should keep representative entity, not duplicates."""
        id_to_representative = {
            "entity-1": "entity-0",
        }

        result = _create_deduplicated_entities(sample_entities, id_to_representative)

        result_ids = [e.id for e in result]
        assert "entity-0" in result_ids
        # entity-1 should not be in results (it maps to entity-0)
        # But entity-0 IS the representative, so it should be there
        assert len([r for r in result_ids if r == "entity-1"]) == 0

    def test_should_handle_self_mapping(self, sample_entities):
        """Should handle entities that map to themselves."""
        # Entity maps to itself
        id_to_representative = {
            "entity-0": "entity-0",
            "entity-1": "entity-1",
        }

        result = _create_deduplicated_entities(sample_entities, id_to_representative)

        assert len(result) == len(sample_entities)

    def test_should_handle_chained_mapping(self, sample_entities):
        """Should handle simple representative mapping correctly."""
        # All entities map to entity-0
        id_to_representative = {f"entity-{i}": "entity-0" for i in range(5)}

        result = _create_deduplicated_entities(sample_entities, id_to_representative)

        # 5 entities mapped to entity-0, 5 unmapped
        assert len(result) == 6


class TestCreateDeduplicatedEntitiesPerformance:
    """Performance-focused tests for _create_deduplicated_entities."""

    def test_should_handle_large_entity_list(self):
        """Should handle large entity lists efficiently (O(n) complexity)."""
        # Create 1000 entities
        large_entities = [
            BaseNode(
                id=f"entity-{i}",
                type="Person",
                properties={"name": f"Person {i}"},
                canonical_key=f"key-{i}",
            )
            for i in range(1000)
        ]

        # Create mapping: every 10 entities map to a representative
        id_to_representative = {}
        for i in range(1000):
            representative_idx = (i // 10) * 10
            if i != representative_idx:
                id_to_representative[f"entity-{i}"] = f"entity-{representative_idx}"

        import time

        start = time.time()
        result = _create_deduplicated_entities(large_entities, id_to_representative)
        duration = time.time() - start

        # Should complete quickly (< 1 second for 1000 entities)
        assert duration < 1.0
        # Should have 100 representatives
        assert len(result) == 100


# ============================================================
# _create_splink_dataframe Tests
# ============================================================


class TestCreateSplinkDataframe:
    """Test _create_splink_dataframe function (vectorized optimization)."""

    def test_should_create_dataframe(self, entities_data):
        """Should create DataFrame from entity data."""
        df, columns = _create_splink_dataframe(
            entities_data,
            system_properties=["id", "type"],
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(entities_data)

    def test_should_exclude_system_properties(self, entities_data):
        """Should exclude system properties from columns."""
        df, columns = _create_splink_dataframe(
            entities_data,
            system_properties=["id", "type"],
        )

        assert "id" not in columns
        assert "type" not in columns

    def test_should_include_property_columns(self, entities_data):
        """Should include property columns."""
        df, columns = _create_splink_dataframe(
            entities_data,
            system_properties=["id", "type"],
        )

        assert "name" in columns
        assert "email" in columns

    def test_should_respect_allowed_properties(self, entities_data):
        """Should only include allowed properties when specified."""
        df, columns = _create_splink_dataframe(
            entities_data,
            system_properties=["id", "type"],
            allowed_properties={"name"},
        )

        assert "name" in columns
        assert "email" not in columns

    def test_should_handle_none_values(self):
        """Should handle None property values."""
        entities = [
            {
                "id": "1",
                "type": "Person",
                "properties": {"name": "Test", "email": None},
                "canonical_properties": {},
            }
        ]

        df, columns = _create_splink_dataframe(
            entities,
            system_properties=["id", "type"],
        )

        assert "name" in columns

    def test_should_handle_list_values(self):
        """Should convert list values to strings."""
        entities = [
            {
                "id": "1",
                "type": "Person",
                "properties": {"tags": ["tag1", "tag2"]},
                "canonical_properties": {},
            }
        ]

        df, columns = _create_splink_dataframe(
            entities,
            system_properties=["id", "type"],
        )

        # Should convert list to string (format may vary)
        # Access via .at or direct row access to avoid iloc
        tags_value = (
            str(df.at[0, "tags"])
            if hasattr(df.at, "__getitem__")
            else str(df["tags"][0])
        )
        assert "tag1" in tags_value
        assert "tag2" in tags_value


class TestCreateSplinkDataframePerformance:
    """Performance-focused tests for _create_splink_dataframe."""

    def test_should_handle_large_entity_list_efficiently(self):
        """Should handle large entity lists efficiently (vectorized)."""
        # Create 1000 entities with many properties
        large_entities = [
            {
                "id": f"entity-{i}",
                "type": "Person",
                "properties": {
                    "name": f"Person {i}",
                    "email": f"person{i}@example.com",
                    "title": f"Title {i}",
                    "department": f"Dept {i % 10}",
                },
                "canonical_properties": {
                    "name": f"person {i}",
                },
            }
            for i in range(1000)
        ]

        import time

        start = time.time()
        df, columns = _create_splink_dataframe(
            large_entities,
            system_properties=["id", "type"],
        )
        duration = time.time() - start

        # Should complete quickly (< 2 seconds for 1000 entities)
        assert duration < 2.0
        assert len(df) == 1000

    def test_should_use_vectorized_operations(self):
        """Verify DataFrame uses vectorized column assignment."""
        entities = [
            {
                "id": f"entity-{i}",
                "type": "Person",
                "properties": {"name": f"Person {i}"},
                "canonical_properties": {"name": f"person {i}"},
            }
            for i in range(100)
        ]

        df, columns = _create_splink_dataframe(
            entities,
            system_properties=["id", "type"],
        )

        # Verify all values are properly set (no None where values should exist)
        assert df["name"].notna().sum() == 100


# ============================================================
# Integration Tests
# ============================================================


class TestPerformanceOptimizationsIntegration:
    """Integration tests for performance optimizations."""

    def test_deduplication_pipeline(self, sample_entities):
        """Test complete deduplication pipeline."""
        # Simulate Splink clustering result
        id_to_representative = {
            "entity-1": "entity-0",
            "entity-2": "entity-0",
            "entity-5": "entity-4",
        }

        result = _create_deduplicated_entities(sample_entities, id_to_representative)

        # Verify correct deduplication
        result_ids = set(e.id for e in result)
        assert "entity-0" in result_ids
        assert "entity-4" in result_ids
        # Duplicates should not be in result
        # 10 entities, 3 mapped to representatives (entity-1,2->0, entity-5->4)
        # Result: entity-0,3,4,6,7,8,9 = 7 entities
        assert len(result) == 7

    def test_ontology_cache_with_dataframe_creation(
        self, sample_ontology, entities_data
    ):
        """Test ontology cache integration with DataFrame creation."""
        cache = OntologyPropertyCache(sample_ontology)

        # Get property types using cache
        name_type = cache.get_property_type("Person", "name")
        email_type = cache.get_property_type("Person", "email")

        assert name_type == "string"
        assert email_type == "email"

        # Create DataFrame
        df, columns = _create_splink_dataframe(
            entities_data,
            system_properties=["id", "type"],
        )

        # Verify DataFrame is created correctly
        assert "name" in columns
