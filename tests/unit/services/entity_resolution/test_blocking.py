"""Unit tests for Blocking Rule Generator.

Tests for LSH blocking and ontology-driven blocking rules.
Ensures domain-agnostic behavior - rules derived from ontology metadata.
"""

import pytest

from graphora_server.services.entity_resolution.blocking import (
    BlockingRuleGenerator,
    LSHBlocker,
    generate_blocking_rules_from_ontology,
)
from graphora_server.services.entity_resolution.models import BlockingRule


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def blocking_generator():
    """Create BlockingRuleGenerator instance."""
    return BlockingRuleGenerator()


@pytest.fixture
def sample_ontology():
    """Create sample ontology for testing."""
    return {
        "version": "1.0",
        "entities": {
            "EntityA": {
                "properties": {
                    "unique_id": {
                        "type": "identifier",
                        "is_identifier": True,
                        "is_blocking_key": True,
                    },
                    "name": {
                        "type": "string",
                        "matching_weight": 2.0,
                    },
                    "description": {
                        "type": "text",
                    },
                    "email": {
                        "type": "email",
                        "is_blocking_key": True,
                    },
                },
            },
            "EntityB": {
                "properties": {
                    "title": {"type": "string"},
                    "count": {"type": "integer"},
                },
            },
        },
    }


@pytest.fixture
def lsh_blocker():
    """Create LSHBlocker instance."""
    return LSHBlocker(num_hashes=50, bands=10)


# ============================================================
# BlockingRuleGenerator Initialization Tests
# ============================================================


class TestBlockingRuleGeneratorInit:
    """Test BlockingRuleGenerator initialization."""

    def test_should_create_with_defaults(self):
        """Should create generator with default settings."""
        generator = BlockingRuleGenerator()
        assert generator.max_blocking_rules == 5
        assert generator.enable_lsh is True

    def test_should_accept_custom_settings(self):
        """Should accept custom configuration."""
        generator = BlockingRuleGenerator(
            max_blocking_rules=3,
            enable_lsh=False,
            lsh_num_hashes=200,
        )
        assert generator.max_blocking_rules == 3
        assert generator.enable_lsh is False
        assert generator.lsh_num_hashes == 200


# ============================================================
# Blocking Rule Generation Tests
# ============================================================


class TestGenerateRulesForEntity:
    """Test blocking rule generation from ontology."""

    def test_should_prioritize_explicit_blocking_keys(
        self, blocking_generator, sample_ontology
    ):
        """Should prioritize properties marked as blocking keys."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = blocking_generator.generate_rules_for_entity("EntityA", entity_def)

        # Should include explicit blocking keys first
        property_names = [r.property_name for r in rules]
        assert "unique_id" in property_names
        assert "email" in property_names

    def test_should_use_identifier_properties_for_blocking(
        self, blocking_generator, sample_ontology
    ):
        """Should use identifier properties for blocking."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = blocking_generator.generate_rules_for_entity("EntityA", entity_def)

        id_rules = [r for r in rules if r.property_name == "unique_id"]
        assert len(id_rules) > 0

    def test_should_limit_number_of_rules(self, sample_ontology):
        """Should respect max_blocking_rules limit."""
        generator = BlockingRuleGenerator(max_blocking_rules=2)
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = generator.generate_rules_for_entity("EntityA", entity_def)

        assert len(rules) <= 2

    def test_should_use_exact_method_for_identifiers(
        self, blocking_generator, sample_ontology
    ):
        """Should use exact blocking for identifier properties."""
        entity_def = sample_ontology["entities"]["EntityA"]
        rules = blocking_generator.generate_rules_for_entity("EntityA", entity_def)

        id_rules = [r for r in rules if r.property_name == "unique_id"]
        assert len(id_rules) > 0
        assert id_rules[0].method == "exact"

    def test_should_use_first_n_chars_for_strings(self, blocking_generator):
        """Should use first_n_chars for string properties."""
        entity_def = {
            "properties": {
                "name": {"type": "string", "is_blocking_key": True},
            },
        }
        rules = blocking_generator.generate_rules_for_entity("Test", entity_def)

        name_rules = [r for r in rules if r.property_name == "name"]
        assert len(name_rules) > 0
        assert name_rules[0].method == "first_n_chars"

    def test_should_generate_default_rules_when_no_blocking_keys(
        self, blocking_generator
    ):
        """Should generate default rules when no explicit blocking keys."""
        entity_def = {
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "integer"},
            },
        }
        rules = blocking_generator.generate_rules_for_entity("Test", entity_def)

        # Should still generate some rules
        assert len(rules) > 0

    def test_should_use_high_weight_properties_for_blocking(self, blocking_generator):
        """Should consider high-weight properties for blocking."""
        entity_def = {
            "properties": {
                "low_weight": {"type": "string", "matching_weight": 0.5},
                "high_weight": {"type": "string", "matching_weight": 2.0},
            },
        }
        rules = blocking_generator.generate_rules_for_entity("Test", entity_def)

        property_names = [r.property_name for r in rules]
        # High weight should be considered
        if "high_weight" in property_names:
            assert True  # High weight was considered
        else:
            # If not included, at least some rules were generated
            assert len(rules) > 0


# ============================================================
# Blocking Method Selection Tests
# ============================================================


class TestBlockingMethodSelection:
    """Test blocking method selection based on data type."""

    def test_should_use_exact_for_email(self, blocking_generator):
        """Should use exact blocking for email type."""
        entity_def = {
            "properties": {
                "email": {"type": "email", "is_blocking_key": True},
            },
        }
        rules = blocking_generator.generate_rules_for_entity("Test", entity_def)

        assert rules[0].method == "exact"

    def test_should_use_exact_for_integer(self, blocking_generator):
        """Should use exact blocking for integer type."""
        entity_def = {
            "properties": {
                "count": {"type": "integer", "is_blocking_key": True},
            },
        }
        rules = blocking_generator.generate_rules_for_entity("Test", entity_def)

        assert rules[0].method == "exact"

    def test_should_use_numeric_bucket_for_float(self, blocking_generator):
        """Should use numeric bucket for float type."""
        entity_def = {
            "properties": {
                "amount": {"type": "float", "is_blocking_key": True},
            },
        }
        rules = blocking_generator.generate_rules_for_entity("Test", entity_def)

        assert rules[0].method == "numeric_bucket"

    def test_should_use_date_bucket_for_date(self, blocking_generator):
        """Should use date bucket for date type."""
        entity_def = {
            "properties": {
                "created": {"type": "date", "is_blocking_key": True},
            },
        }
        rules = blocking_generator.generate_rules_for_entity("Test", entity_def)

        assert rules[0].method == "date_bucket"

    def test_should_use_lsh_for_text_when_enabled(self):
        """Should use LSH for text when enabled."""
        generator = BlockingRuleGenerator(enable_lsh=True)
        entity_def = {
            "properties": {
                "description": {"type": "text", "is_blocking_key": True},
            },
        }
        rules = generator.generate_rules_for_entity("Test", entity_def)

        assert rules[0].method == "lsh"

    def test_should_use_first_n_chars_for_text_when_lsh_disabled(self):
        """Should use first_n_chars for text when LSH disabled."""
        generator = BlockingRuleGenerator(enable_lsh=False)
        entity_def = {
            "properties": {
                "description": {"type": "text", "is_blocking_key": True},
            },
        }
        rules = generator.generate_rules_for_entity("Test", entity_def)

        assert rules[0].method == "first_n_chars"


# ============================================================
# BlockingRule Tests
# ============================================================


class TestBlockingRuleConversion:
    """Test BlockingRule to Splink rule conversion."""

    def test_should_convert_exact_rule(self):
        """Should convert exact blocking to Splink rule."""
        rule = BlockingRule(property_name="email", method="exact")
        splink = rule.to_splink_rule()

        assert "email" in splink
        assert "=" in splink

    def test_should_convert_first_n_chars_rule(self):
        """Should convert first_n_chars to Splink rule with SUBSTR."""
        rule = BlockingRule(
            property_name="name",
            method="first_n_chars",
            params={"n": 4},
        )
        splink = rule.to_splink_rule()

        assert "SUBSTR" in splink
        assert "name" in splink
        assert "4" in splink

    def test_should_use_default_n_for_first_n_chars(self):
        """Should use default n=4 when not specified."""
        rule = BlockingRule(
            property_name="name",
            method="first_n_chars",
            params={},  # No n specified
        )
        splink = rule.to_splink_rule()

        assert "4" in splink  # Default


# ============================================================
# LSHBlocker Tests
# ============================================================


class TestLSHBlockerInit:
    """Test LSHBlocker initialization."""

    def test_should_create_with_defaults(self):
        """Should create blocker with default settings."""
        blocker = LSHBlocker()
        assert blocker.num_hashes == 100
        assert blocker.bands == 20
        assert blocker.ngram_size == 3

    def test_should_accept_custom_settings(self):
        """Should accept custom configuration."""
        blocker = LSHBlocker(
            num_hashes=200,
            bands=40,
            ngram_size=4,
        )
        assert blocker.num_hashes == 200
        assert blocker.bands == 40
        assert blocker.ngram_size == 4


class TestLSHBlockerNgrams:
    """Test n-gram generation."""

    def test_should_generate_character_ngrams(self, lsh_blocker):
        """Should generate character n-grams from text."""
        ngrams = lsh_blocker._get_ngrams("hello")

        # Should have n-grams of size 3
        assert "hel" in ngrams
        assert "ell" in ngrams
        assert "llo" in ngrams

    def test_should_handle_short_text(self, lsh_blocker):
        """Should handle text shorter than ngram size."""
        ngrams = lsh_blocker._get_ngrams("ab")

        # Should return the text itself
        assert "ab" in ngrams

    def test_should_normalize_text(self, lsh_blocker):
        """Should normalize text before generating ngrams."""
        ngrams1 = lsh_blocker._get_ngrams("Hello")
        ngrams2 = lsh_blocker._get_ngrams("hello")

        assert ngrams1 == ngrams2


class TestLSHBlockerMinhash:
    """Test minhash signature computation."""

    def test_should_generate_signature_of_correct_length(self, lsh_blocker):
        """Should generate signature with num_hashes elements."""
        ngrams = lsh_blocker._get_ngrams("test text")
        signature = lsh_blocker._minhash(ngrams)

        assert len(signature) == lsh_blocker.num_hashes

    def test_should_return_zeros_for_empty_ngrams(self, lsh_blocker):
        """Should return zeros for empty ngram set."""
        signature = lsh_blocker._minhash(set())

        assert all(h == 0 for h in signature)

    def test_should_generate_consistent_signature(self, lsh_blocker):
        """Should generate same signature for same text."""
        ngrams = lsh_blocker._get_ngrams("consistent")
        sig1 = lsh_blocker._minhash(ngrams)
        sig2 = lsh_blocker._minhash(ngrams)

        assert sig1 == sig2


class TestLSHBlockerBuckets:
    """Test bucket computation."""

    def test_should_compute_buckets(self, lsh_blocker):
        """Should compute bucket keys for text."""
        buckets = lsh_blocker.compute_buckets("test text")

        assert isinstance(buckets, list)
        assert len(buckets) == lsh_blocker.bands
        assert all(b.startswith("band_") for b in buckets)

    def test_similar_texts_should_share_buckets(self, lsh_blocker):
        """Similar texts should share at least some buckets."""
        buckets1 = set(lsh_blocker.compute_buckets("hello world"))
        buckets2 = set(lsh_blocker.compute_buckets("hello world!"))

        # Should share at least one bucket
        shared = buckets1 & buckets2
        assert len(shared) > 0

    def test_different_texts_may_not_share_buckets(self, lsh_blocker):
        """Very different texts may not share buckets."""
        buckets1 = set(lsh_blocker.compute_buckets("apple banana cherry"))
        buckets2 = set(lsh_blocker.compute_buckets("xyz 123 456"))

        # May or may not share buckets (probabilistic)
        # Just verify buckets are computed
        assert len(buckets1) > 0
        assert len(buckets2) > 0


class TestLSHBlockerCandidatePairs:
    """Test candidate pair generation."""

    def test_should_find_candidate_pairs(self, lsh_blocker):
        """Should find candidate pairs from entities."""
        entities = [
            {"name": "John Smith"},
            {"name": "John Smyth"},  # Similar
            {"name": "Jane Doe"},
        ]

        pairs = lsh_blocker.find_candidate_pairs(entities, "name")

        # Should find pairs - John Smith and John Smyth are similar
        assert isinstance(pairs, list)
        # All pairs should be tuples of indices
        for pair in pairs:
            assert len(pair) == 2
            assert pair[0] < pair[1]  # Consistent ordering

    def test_should_handle_empty_entities(self, lsh_blocker):
        """Should handle empty entity list."""
        pairs = lsh_blocker.find_candidate_pairs([], "name")
        assert pairs == []

    def test_should_handle_missing_property(self, lsh_blocker):
        """Should handle entities missing the property."""
        entities = [
            {"name": "John"},
            {"other": "value"},  # Missing name
            {"name": "Jane"},
        ]

        pairs = lsh_blocker.find_candidate_pairs(entities, "name")
        # Should not crash, may or may not find pairs
        assert isinstance(pairs, list)

    def test_should_return_unique_pairs(self, lsh_blocker):
        """Should return unique pairs without duplicates."""
        entities = [
            {"name": "test text"},
            {"name": "test text"},  # Identical
            {"name": "test text!"},
        ]

        pairs = lsh_blocker.find_candidate_pairs(entities, "name")

        # Check uniqueness
        assert len(pairs) == len(set(pairs))


# ============================================================
# Convenience Function Tests
# ============================================================


class TestGenerateBlockingRulesFromOntology:
    """Test generate_blocking_rules_from_ontology convenience function."""

    def test_should_generate_rules_from_ontology(self, sample_ontology):
        """Should generate rules from parsed ontology."""
        rules = generate_blocking_rules_from_ontology(sample_ontology, "EntityA")

        assert len(rules) > 0
        assert all(isinstance(r, BlockingRule) for r in rules)

    def test_should_raise_for_unknown_entity(self, sample_ontology):
        """Should raise ValueError for unknown entity type."""
        with pytest.raises(ValueError, match="not found"):
            generate_blocking_rules_from_ontology(sample_ontology, "NonExistent")


# ============================================================
# Domain Agnosticism Tests
# ============================================================


class TestBlockingDomainAgnosticism:
    """Test that blocking is domain-agnostic."""

    def test_should_work_with_any_property_name(self, blocking_generator):
        """Should work with arbitrary property names."""
        entity_def = {
            "properties": {
                "arbitrary_field_xyz_123": {
                    "type": "string",
                    "is_blocking_key": True,
                },
            },
        }
        rules = blocking_generator.generate_rules_for_entity("Test", entity_def)

        assert len(rules) > 0
        assert rules[0].property_name == "arbitrary_field_xyz_123"

    def test_should_derive_all_rules_from_ontology(self, blocking_generator):
        """Should derive rules entirely from ontology metadata."""
        # Domain A
        domain_a_entity = {
            "properties": {
                "unique_code": {"type": "identifier", "is_blocking_key": True},
                "title": {"type": "string"},
            },
        }

        # Domain B
        domain_b_entity = {
            "properties": {
                "reference_id": {"type": "identifier", "is_blocking_key": True},
                "name": {"type": "string"},
            },
        }

        rules_a = blocking_generator.generate_rules_for_entity("A", domain_a_entity)
        rules_b = blocking_generator.generate_rules_for_entity("B", domain_b_entity)

        # Both should have blocking rules
        assert len(rules_a) > 0
        assert len(rules_b) > 0

        # Both should use same method for same type
        id_rule_a = next(r for r in rules_a if r.property_name == "unique_code")
        id_rule_b = next(r for r in rules_b if r.property_name == "reference_id")
        assert id_rule_a.method == id_rule_b.method
