"""Unit tests for Comparison Rule Generator.

Tests for ontology-driven comparison rule generation.
Ensures domain-agnostic behavior - rules derived from ontology, not hardcoded.
"""

import pytest

from graphora_server.services.entity_resolution.comparison_rules import (
    ComparisonRuleGenerator,
    generate_rules_from_ontology,
    generate_config_from_ontology,
)
from graphora_server.services.entity_resolution.models import (
    ComparisonMethod,
    DataType,
    ComparisonRule,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def rule_generator():
    """Create ComparisonRuleGenerator instance."""
    return ComparisonRuleGenerator()


@pytest.fixture
def sample_ontology():
    """Create sample ontology for testing.

    This is a generic ontology - not domain-specific.
    The entity type names are placeholders that could be anything.
    """
    return {
        "version": "1.0",
        "entities": {
            "EntityA": {
                "properties": {
                    "name": {
                        "type": "string",
                        "is_identifier": False,
                        "matching_weight": 2.0,
                    },
                    "unique_code": {
                        "type": "identifier",
                        "is_identifier": True,
                        "matching_weight": 3.0,
                    },
                    "description": {
                        "type": "text",
                        "matching_weight": 1.0,
                    },
                    "count": {
                        "type": "integer",
                        "matching_weight": 0.5,
                    },
                    "created_date": {
                        "type": "date",
                        "matching_weight": 1.0,
                    },
                },
            },
            "EntityB": {
                "properties": {
                    "email": {
                        "type": "email",
                        "is_identifier": True,
                    },
                    "phone": {
                        "type": "phone",
                    },
                    "tags": {
                        "type": "list",
                    },
                },
            },
        },
    }


@pytest.fixture
def minimal_ontology():
    """Create minimal ontology with basic properties."""
    return {
        "version": "1.0",
        "entities": {
            "SimpleEntity": {
                "properties": {
                    "name": {"type": "str"},
                },
            },
        },
    }


# ============================================================
# ComparisonRuleGenerator Initialization Tests
# ============================================================


class TestComparisonRuleGeneratorInit:
    """Test ComparisonRuleGenerator initialization."""

    def test_should_create_with_defaults(self):
        """Should create generator with default settings."""
        generator = ComparisonRuleGenerator()
        assert generator.default_weight == 1.0
        assert generator.enable_embedding is True

    def test_should_accept_custom_config(self):
        """Should accept custom configuration."""
        config = {
            "default_weight": 1.5,
            "enable_embedding": False,
        }
        generator = ComparisonRuleGenerator(config)
        assert generator.default_weight == 1.5
        assert generator.enable_embedding is False

    def test_should_accept_custom_type_mappings(self):
        """Should accept custom type string mappings."""
        config = {
            "custom_type_mappings": {
                "custom_type": DataType.TEXT,
            },
        }
        generator = ComparisonRuleGenerator(config)
        assert "custom_type" in generator.type_mapping
        assert generator.type_mapping["custom_type"] == DataType.TEXT


# ============================================================
# Rule Generation Tests
# ============================================================


class TestGenerateRulesForEntity:
    """Test comparison rule generation from ontology."""

    def test_should_generate_rules_for_all_properties(
        self, rule_generator, sample_ontology
    ):
        """Should generate rules for all properties in entity."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = rule_generator.generate_rules_for_entity("EntityA", entity_def)

        # Should have rules for all properties
        property_names = {rule.property_name for rule in rules}
        expected = {"name", "unique_code", "description", "count", "created_date"}
        assert property_names == expected

    def test_should_use_jaro_winkler_for_string_type(
        self, rule_generator, sample_ontology
    ):
        """Should use Jaro-Winkler for string properties."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = rule_generator.generate_rules_for_entity("EntityA", entity_def)

        name_rules = [r for r in rules if r.property_name == "name"]
        assert len(name_rules) > 0
        assert any(
            r.comparison_method == ComparisonMethod.JARO_WINKLER for r in name_rules
        )

    def test_should_use_embedding_for_text_type(self, rule_generator, sample_ontology):
        """Should use embedding similarity for text properties."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = rule_generator.generate_rules_for_entity("EntityA", entity_def)

        desc_rules = [r for r in rules if r.property_name == "description"]
        assert len(desc_rules) > 0
        assert any(
            r.comparison_method == ComparisonMethod.EMBEDDING for r in desc_rules
        )

    def test_should_use_exact_for_identifier_type(
        self, rule_generator, sample_ontology
    ):
        """Should use exact match for identifier properties."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = rule_generator.generate_rules_for_entity("EntityA", entity_def)

        code_rules = [r for r in rules if r.property_name == "unique_code"]
        assert len(code_rules) > 0
        assert any(r.comparison_method == ComparisonMethod.EXACT for r in code_rules)

    def test_should_use_numeric_tolerance_for_integer(
        self, rule_generator, sample_ontology
    ):
        """Should use numeric tolerance for integer properties."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = rule_generator.generate_rules_for_entity("EntityA", entity_def)

        count_rules = [r for r in rules if r.property_name == "count"]
        assert len(count_rules) > 0
        assert any(
            r.comparison_method == ComparisonMethod.NUMERIC_TOLERANCE
            for r in count_rules
        )

    def test_should_use_date_tolerance_for_date(self, rule_generator, sample_ontology):
        """Should use date tolerance for date properties."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = rule_generator.generate_rules_for_entity("EntityA", entity_def)

        date_rules = [r for r in rules if r.property_name == "created_date"]
        assert len(date_rules) > 0
        assert any(
            r.comparison_method == ComparisonMethod.DATE_TOLERANCE for r in date_rules
        )

    def test_should_respect_matching_weight(self, rule_generator, sample_ontology):
        """Should use matching weight from ontology."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = rule_generator.generate_rules_for_entity("EntityA", entity_def)

        name_rules = [r for r in rules if r.property_name == "name"]
        assert all(r.weight == 2.0 for r in name_rules)

        code_rules = [r for r in rules if r.property_name == "unique_code"]
        assert all(r.weight == 3.0 for r in code_rules)

    def test_should_sort_rules_by_weight(self, rule_generator, sample_ontology):
        """Should sort rules by weight descending."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = rule_generator.generate_rules_for_entity("EntityA", entity_def)

        weights = [r.weight for r in rules]
        assert weights == sorted(weights, reverse=True)

    def test_should_handle_email_type(self, rule_generator, sample_ontology):
        """Should use exact normalized for email properties."""
        entity_def = sample_ontology["entities"]["EntityB"]
        rules = rule_generator.generate_rules_for_entity("EntityB", entity_def)

        email_rules = [r for r in rules if r.property_name == "email"]
        assert len(email_rules) > 0

    def test_should_handle_list_type(self, rule_generator, sample_ontology):
        """Should use jaccard for list properties."""
        entity_def = sample_ontology["entities"]["EntityB"]
        rules = rule_generator.generate_rules_for_entity("EntityB", entity_def)

        tags_rules = [r for r in rules if r.property_name == "tags"]
        assert len(tags_rules) > 0
        assert any(r.comparison_method == ComparisonMethod.JACCARD for r in tags_rules)

    def test_should_handle_unknown_type_as_string(self, rule_generator):
        """Should treat unknown types as string."""
        entity_def = {
            "properties": {
                "custom_field": {"type": "unknown_custom_type"},
            },
        }
        rules = rule_generator.generate_rules_for_entity("TestEntity", entity_def)

        assert len(rules) > 0
        # Should default to string comparison
        assert any(r.comparison_method == ComparisonMethod.JARO_WINKLER for r in rules)


# ============================================================
# Prior Generation Tests
# ============================================================


class TestPriorGeneration:
    """Test prior probability generation."""

    def test_should_use_high_confidence_prior_for_identifier(self, rule_generator):
        """Should use high confidence prior for identifier properties."""
        entity_def = {
            "properties": {
                "id": {"type": "string", "is_identifier": True},
            },
        }
        rules = rule_generator.generate_rules_for_entity("Test", entity_def)

        assert len(rules) > 0
        # Identifier should have high match probability
        assert rules[0].prior.m[0] >= 0.9

    def test_should_use_four_level_prior_for_string_similarity(self, rule_generator):
        """Should use 4-level prior for string similarity methods."""
        entity_def = {
            "properties": {
                "name": {"type": "string"},
            },
        }
        rules = rule_generator.generate_rules_for_entity("Test", entity_def)

        jw_rules = [
            r for r in rules if r.comparison_method == ComparisonMethod.JARO_WINKLER
        ]
        assert len(jw_rules) > 0
        assert len(jw_rules[0].prior.m) == 4

    def test_should_use_two_level_prior_for_exact_match(self, rule_generator):
        """Should use 2-level prior for exact match methods."""
        entity_def = {
            "properties": {
                "code": {"type": "identifier"},
            },
        }
        rules = rule_generator.generate_rules_for_entity("Test", entity_def)

        exact_rules = [
            r for r in rules if r.comparison_method == ComparisonMethod.EXACT
        ]
        assert len(exact_rules) > 0
        assert len(exact_rules[0].prior.m) == 2


# ============================================================
# EntityResolutionConfig Generation Tests
# ============================================================


class TestGenerateConfigForEntity:
    """Test complete config generation."""

    def test_should_generate_complete_config(self, rule_generator, sample_ontology):
        """Should generate EntityResolutionConfig with all components."""
        entity_def = sample_ontology["entities"]["EntityA"]

        config = rule_generator.generate_config_for_entity("EntityA", entity_def)

        assert config.entity_type == "EntityA"
        assert len(config.comparison_rules) > 0
        assert config.match_threshold == 0.7
        assert config.use_embedding_similarity is True

    def test_should_accept_custom_resolution_settings(
        self, rule_generator, sample_ontology
    ):
        """Should accept custom resolution settings."""
        entity_def = sample_ontology["entities"]["EntityA"]
        settings = {
            "match_threshold": 0.85,
            "review_threshold": 0.6,
            "embedding_model": "custom-model",
        }

        config = rule_generator.generate_config_for_entity(
            "EntityA", entity_def, settings
        )

        assert config.match_threshold == 0.85
        assert config.review_threshold == 0.6
        assert config.embedding_model == "custom-model"


# ============================================================
# Convenience Function Tests
# ============================================================


class TestGenerateRulesFromOntology:
    """Test generate_rules_from_ontology convenience function."""

    def test_should_generate_rules_from_parsed_ontology(self, sample_ontology):
        """Should generate rules from full ontology dict."""
        rules = generate_rules_from_ontology(sample_ontology, "EntityA")
        assert len(rules) > 0
        assert all(isinstance(r, ComparisonRule) for r in rules)

    def test_should_raise_for_unknown_entity_type(self, sample_ontology):
        """Should raise ValueError for unknown entity type."""
        with pytest.raises(ValueError, match="not found"):
            generate_rules_from_ontology(sample_ontology, "NonExistentEntity")


class TestGenerateConfigFromOntology:
    """Test generate_config_from_ontology convenience function."""

    def test_should_generate_config_from_parsed_ontology(self, sample_ontology):
        """Should generate config from full ontology dict."""
        config = generate_config_from_ontology(sample_ontology, "EntityA")
        assert config.entity_type == "EntityA"
        assert len(config.comparison_rules) > 0

    def test_should_raise_for_unknown_entity_type(self, sample_ontology):
        """Should raise ValueError for unknown entity type."""
        with pytest.raises(ValueError, match="not found"):
            generate_config_from_ontology(sample_ontology, "NonExistentEntity")


# ============================================================
# Domain Agnosticism Tests
# ============================================================


class TestDomainAgnosticism:
    """Test that rule generation is domain-agnostic."""

    def test_should_work_with_any_entity_type_name(self, rule_generator):
        """Should work with any entity type name."""
        # Use completely arbitrary names
        entity_def = {
            "properties": {
                "xyz_field": {"type": "string"},
            },
        }
        rules = rule_generator.generate_rules_for_entity(
            "ArbitraryTypeName123", entity_def
        )
        assert len(rules) > 0

    def test_should_work_with_any_property_name(self, rule_generator):
        """Should work with any property name."""
        entity_def = {
            "properties": {
                "completely_arbitrary_property_name": {"type": "string"},
                "_internal_field": {"type": "integer"},
                "CamelCaseField": {"type": "date"},
            },
        }
        rules = rule_generator.generate_rules_for_entity("Entity", entity_def)
        property_names = {r.property_name for r in rules}
        assert "completely_arbitrary_property_name" in property_names
        assert "_internal_field" in property_names
        assert "CamelCaseField" in property_names

    def test_should_derive_all_rules_from_ontology_metadata(self, rule_generator):
        """Should derive rules entirely from ontology - no hardcoded field names."""
        # Medical domain
        medical_entity = {
            "properties": {
                "diagnosis_code": {"type": "identifier", "is_identifier": True},
                "patient_notes": {"type": "text"},
                "dosage_mg": {"type": "float"},
            },
        }

        # Financial domain
        financial_entity = {
            "properties": {
                "account_number": {"type": "identifier", "is_identifier": True},
                "transaction_memo": {"type": "text"},
                "amount_usd": {"type": "float"},
            },
        }

        medical_rules = rule_generator.generate_rules_for_entity(
            "MedicalEntity", medical_entity
        )
        financial_rules = rule_generator.generate_rules_for_entity(
            "FinancialEntity", financial_entity
        )

        # Both should work equally well - no domain-specific logic
        assert len(medical_rules) == len(financial_rules)

        # Same types should produce same comparison methods
        medical_id_methods = {
            r.comparison_method
            for r in medical_rules
            if r.property_name == "diagnosis_code"
        }
        financial_id_methods = {
            r.comparison_method
            for r in financial_rules
            if r.property_name == "account_number"
        }
        assert medical_id_methods == financial_id_methods
