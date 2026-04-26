"""Unit tests for Entity Resolution Models.

Tests for data structures used in entity resolution.
Ensures domain-agnostic behavior with ontology-driven configuration.
"""

import pytest

from graphora_server.services.entity_resolution.models import (
    DataType,
    ComparisonMethod,
    ComparisonPrior,
    PropertyMatchingConfig,
    ComparisonRule,
    BlockingRule,
    EntityResolutionConfig,
    get_prior_for_type,
    DEFAULT_PRIORS,
)


# ============================================================
# DataType Enum Tests
# ============================================================


class TestDataType:
    """Test DataType enum."""

    def test_should_have_all_data_types(self):
        """Should include all expected data types."""
        expected_types = [
            "string",
            "text",
            "date",
            "datetime",
            "number",
            "integer",
            "float",
            "boolean",
            "email",
            "url",
            "identifier",
            "phone",
            "list",
        ]
        for type_value in expected_types:
            assert DataType(type_value) is not None

    def test_should_be_string_enum(self):
        """DataType values should be strings."""
        for data_type in DataType:
            assert isinstance(data_type.value, str)


# ============================================================
# ComparisonMethod Enum Tests
# ============================================================


class TestComparisonMethod:
    """Test ComparisonMethod enum."""

    def test_should_have_all_comparison_methods(self):
        """Should include all expected comparison methods."""
        expected_methods = [
            "exact",
            "exact_normalized",
            "jaro_winkler",
            "levenshtein",
            "embedding",
            "numeric_tolerance",
            "date_tolerance",
            "metaphone",
            "soundex",
            "jaccard",
        ]
        for method in expected_methods:
            assert ComparisonMethod(method) is not None


# ============================================================
# ComparisonPrior Tests
# ============================================================


class TestComparisonPrior:
    """Test ComparisonPrior dataclass."""

    def test_should_create_valid_prior(self):
        """Should create prior with valid m/u probabilities."""
        prior = ComparisonPrior(m=(0.9, 0.1), u=(0.1, 0.9))
        assert prior.m == (0.9, 0.1)
        assert prior.u == (0.1, 0.9)

    def test_should_reject_mismatched_tuple_lengths(self):
        """Should reject when m and u have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            ComparisonPrior(m=(0.9, 0.1), u=(0.1, 0.2, 0.7))

    def test_should_reject_probabilities_not_summing_to_one(self):
        """Should reject when probabilities don't sum to 1."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            ComparisonPrior(m=(0.5, 0.3), u=(0.1, 0.9))

    def test_should_accept_four_level_prior(self):
        """Should accept 4-level priors for string similarity."""
        prior = ComparisonPrior(
            m=(0.85, 0.10, 0.04, 0.01),
            u=(0.05, 0.10, 0.15, 0.70),
        )
        assert len(prior.m) == 4
        assert len(prior.u) == 4


# ============================================================
# PropertyMatchingConfig Tests
# ============================================================


class TestPropertyMatchingConfig:
    """Test PropertyMatchingConfig dataclass."""

    def test_should_create_with_defaults(self):
        """Should create config with sensible defaults."""
        config = PropertyMatchingConfig(
            property_name="name",
            data_type=DataType.STRING,
        )
        assert config.property_name == "name"
        assert config.data_type == DataType.STRING
        assert config.is_identifier is False
        assert config.is_blocking_key is False
        assert config.matching_weight == 1.0
        assert config.canonicalization is None

    def test_should_set_default_comparison_methods_for_string(self):
        """Should set jaro_winkler for string type."""
        config = PropertyMatchingConfig(
            property_name="name",
            data_type=DataType.STRING,
        )
        assert ComparisonMethod.JARO_WINKLER in config.comparison_methods

    def test_should_set_default_comparison_methods_for_text(self):
        """Should set embedding for text type."""
        config = PropertyMatchingConfig(
            property_name="description",
            data_type=DataType.TEXT,
        )
        assert ComparisonMethod.EMBEDDING in config.comparison_methods

    def test_should_set_default_comparison_methods_for_identifier(self):
        """Should set exact for identifier type."""
        config = PropertyMatchingConfig(
            property_name="id",
            data_type=DataType.IDENTIFIER,
        )
        assert ComparisonMethod.EXACT in config.comparison_methods

    def test_should_set_default_comparison_methods_for_number(self):
        """Should set numeric_tolerance for number type."""
        config = PropertyMatchingConfig(
            property_name="amount",
            data_type=DataType.NUMBER,
        )
        assert ComparisonMethod.NUMERIC_TOLERANCE in config.comparison_methods

    def test_should_respect_explicit_comparison_methods(self):
        """Should use explicit methods when provided."""
        config = PropertyMatchingConfig(
            property_name="name",
            data_type=DataType.STRING,
            comparison_methods=[ComparisonMethod.EXACT, ComparisonMethod.EMBEDDING],
        )
        assert config.comparison_methods == [
            ComparisonMethod.EXACT,
            ComparisonMethod.EMBEDDING,
        ]

    def test_should_accept_identifier_flag(self):
        """Should accept is_identifier flag."""
        config = PropertyMatchingConfig(
            property_name="email",
            data_type=DataType.EMAIL,
            is_identifier=True,
        )
        assert config.is_identifier is True

    def test_should_accept_matching_weight(self):
        """Should accept custom matching weight."""
        config = PropertyMatchingConfig(
            property_name="name",
            data_type=DataType.STRING,
            matching_weight=2.5,
        )
        assert config.matching_weight == 2.5


# ============================================================
# ComparisonRule Tests
# ============================================================


class TestComparisonRule:
    """Test ComparisonRule dataclass."""

    def test_should_create_comparison_rule(self):
        """Should create comparison rule with required fields."""
        prior = ComparisonPrior(m=(0.9, 0.1), u=(0.1, 0.9))
        rule = ComparisonRule(
            property_name="name",
            comparison_method=ComparisonMethod.JARO_WINKLER,
            prior=prior,
        )
        assert rule.property_name == "name"
        assert rule.comparison_method == ComparisonMethod.JARO_WINKLER
        assert rule.weight == 1.0

    def test_should_identify_high_confidence_rule(self):
        """Should correctly identify high confidence rules."""
        prior = ComparisonPrior(m=(0.95, 0.05), u=(0.02, 0.98))
        rule = ComparisonRule(
            property_name="id",
            comparison_method=ComparisonMethod.EXACT,
            prior=prior,
            weight=2.5,
        )
        assert rule.is_high_confidence is True

    def test_should_identify_low_confidence_rule(self):
        """Should correctly identify low confidence rules."""
        prior = ComparisonPrior(m=(0.7, 0.3), u=(0.3, 0.7))
        rule = ComparisonRule(
            property_name="notes",
            comparison_method=ComparisonMethod.JARO_WINKLER,
            prior=prior,
            weight=0.5,
        )
        assert rule.is_high_confidence is False

    def test_should_accept_thresholds(self):
        """Should accept threshold values for similarity matching."""
        prior = ComparisonPrior(m=(0.85, 0.1, 0.04, 0.01), u=(0.05, 0.1, 0.15, 0.7))
        rule = ComparisonRule(
            property_name="name",
            comparison_method=ComparisonMethod.JARO_WINKLER,
            prior=prior,
            thresholds=[0.95, 0.85, 0.70],
        )
        assert rule.thresholds == [0.95, 0.85, 0.70]


# ============================================================
# BlockingRule Tests
# ============================================================


class TestBlockingRule:
    """Test BlockingRule dataclass."""

    def test_should_create_blocking_rule(self):
        """Should create blocking rule with required fields."""
        rule = BlockingRule(
            property_name="name",
            method="first_n_chars",
            params={"n": 4},
        )
        assert rule.property_name == "name"
        assert rule.method == "first_n_chars"
        assert rule.params == {"n": 4}

    def test_should_convert_exact_to_splink_rule(self):
        """Should convert exact blocking to Splink rule string."""
        rule = BlockingRule(
            property_name="email",
            method="exact",
        )
        splink_rule = rule.to_splink_rule()
        assert 'l."email" = r."email"' in splink_rule

    def test_should_convert_first_n_chars_to_splink_rule(self):
        """Should convert first_n_chars blocking to Splink rule string."""
        rule = BlockingRule(
            property_name="name",
            method="first_n_chars",
            params={"n": 4},
        )
        splink_rule = rule.to_splink_rule()
        assert "SUBSTR" in splink_rule
        assert "1, 4" in splink_rule


# ============================================================
# EntityResolutionConfig Tests
# ============================================================


class TestEntityResolutionConfig:
    """Test EntityResolutionConfig dataclass."""

    def test_should_create_with_defaults(self):
        """Should create config with sensible defaults."""
        config = EntityResolutionConfig(entity_type="Person")
        assert config.entity_type == "Person"
        assert config.match_threshold == 0.7
        assert config.review_threshold == 0.5
        assert config.use_embedding_similarity is True
        assert config.use_lsh_blocking is True

    def test_should_accept_custom_thresholds(self):
        """Should accept custom match thresholds."""
        config = EntityResolutionConfig(
            entity_type="Organization",
            match_threshold=0.85,
            review_threshold=0.6,
        )
        assert config.match_threshold == 0.85
        assert config.review_threshold == 0.6

    def test_should_accept_comparison_rules(self):
        """Should accept list of comparison rules."""
        prior = ComparisonPrior(m=(0.9, 0.1), u=(0.1, 0.9))
        rules = [
            ComparisonRule(
                property_name="name",
                comparison_method=ComparisonMethod.JARO_WINKLER,
                prior=prior,
            ),
        ]
        config = EntityResolutionConfig(
            entity_type="Person",
            comparison_rules=rules,
        )
        assert len(config.comparison_rules) == 1

    def test_should_accept_blocking_rules(self):
        """Should accept list of blocking rules."""
        rules = [
            BlockingRule(property_name="name", method="first_n_chars"),
        ]
        config = EntityResolutionConfig(
            entity_type="Person",
            blocking_rules=rules,
        )
        assert len(config.blocking_rules) == 1

    def test_should_accept_embedding_model_config(self):
        """Should accept embedding model configuration."""
        config = EntityResolutionConfig(
            entity_type="Person",
            embedding_model="all-mpnet-base-v2",
            embedding_cache_enabled=False,
        )
        assert config.embedding_model == "all-mpnet-base-v2"
        assert config.embedding_cache_enabled is False


# ============================================================
# get_prior_for_type Tests
# ============================================================


class TestGetPriorForType:
    """Test get_prior_for_type function."""

    def test_should_return_high_confidence_for_identifier(self):
        """Should return high confidence prior for identifiers."""
        prior = get_prior_for_type(DataType.STRING, is_identifier=True)
        assert prior.m[0] >= 0.95  # High match probability

    def test_should_return_identifier_prior_for_identifier_type(self):
        """Should return appropriate prior for identifier data type."""
        prior = get_prior_for_type(DataType.IDENTIFIER)
        assert prior.m[0] >= 0.95

    def test_should_return_string_prior_for_string_type(self):
        """Should return 4-level prior for string type."""
        prior = get_prior_for_type(DataType.STRING)
        assert len(prior.m) == 4  # 4-level for string similarity

    def test_should_return_email_prior_for_email_type(self):
        """Should return appropriate prior for email type."""
        prior = get_prior_for_type(DataType.EMAIL)
        assert prior.m[0] >= 0.9  # High confidence for emails

    def test_should_return_fallback_for_unknown_type(self):
        """Should return fallback prior for unknown types."""
        # Create a mock unknown type by accessing DEFAULT_PRIORS directly
        # The function should handle types not in DEFAULT_PRIORS
        prior = get_prior_for_type(DataType.LIST)  # Less common type
        assert prior is not None
        assert sum(prior.m) == pytest.approx(1.0, abs=0.01)


# ============================================================
# DEFAULT_PRIORS Tests
# ============================================================


class TestDefaultPriors:
    """Test DEFAULT_PRIORS dictionary."""

    def test_should_have_priors_for_common_types(self):
        """Should have priors for common data types."""
        common_types = [
            DataType.STRING,
            DataType.TEXT,
            DataType.IDENTIFIER,
            DataType.EMAIL,
            DataType.NUMBER,
            DataType.DATE,
        ]
        for data_type in common_types:
            assert data_type in DEFAULT_PRIORS

    def test_all_priors_should_be_valid(self):
        """All default priors should have valid probabilities."""
        for data_type, prior in DEFAULT_PRIORS.items():
            assert abs(sum(prior.m) - 1.0) < 0.01
            assert abs(sum(prior.u) - 1.0) < 0.01
