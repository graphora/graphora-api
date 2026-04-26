"""Unit tests for extraction configuration classes."""

import pytest
from graphora_server.services.extraction.config import (
    ValidationConfig,
    ContextConfig,
    MultiPassConfig,
)


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_creation_with_defaults(self):
        """Test creating ValidationConfig with default values."""
        config = ValidationConfig()

        assert config.min_confidence_threshold == 0.7
        assert config.check_required_properties is True
        assert config.check_orphan_nodes is True
        assert config.check_relationship_completeness is True
        assert config.required_property_weight == 0.3
        assert config.orphan_node_severity == 0.4

    def test_creation_with_custom_values(self):
        """Test creating ValidationConfig with custom values."""
        config = ValidationConfig(
            min_confidence_threshold=0.8,
            check_required_properties=False,
            check_orphan_nodes=False,
            check_relationship_completeness=False,
            required_property_weight=0.5,
            orphan_node_severity=0.6,
        )

        assert config.min_confidence_threshold == 0.8
        assert config.check_required_properties is False
        assert config.check_orphan_nodes is False
        assert config.check_relationship_completeness is False
        assert config.required_property_weight == 0.5
        assert config.orphan_node_severity == 0.6

    def test_invalid_confidence_threshold_high(self):
        """Test validation rejects confidence threshold > 1.0."""
        with pytest.raises(ValueError, match="min_confidence_threshold"):
            ValidationConfig(min_confidence_threshold=1.5)

    def test_invalid_confidence_threshold_low(self):
        """Test validation rejects confidence threshold < 0.0."""
        with pytest.raises(ValueError, match="min_confidence_threshold"):
            ValidationConfig(min_confidence_threshold=-0.1)

    def test_invalid_required_property_weight(self):
        """Test validation rejects invalid required_property_weight."""
        with pytest.raises(ValueError, match="required_property_weight"):
            ValidationConfig(required_property_weight=1.5)

    def test_invalid_orphan_node_severity(self):
        """Test validation rejects invalid orphan_node_severity."""
        with pytest.raises(ValueError, match="orphan_node_severity"):
            ValidationConfig(orphan_node_severity=-0.1)


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_creation_with_defaults(self):
        """Test creating ContextConfig with default values."""
        config = ContextConfig()

        assert config.max_context_chars == 8192
        assert config.include_confidence_scores is True
        assert config.include_validation_feedback is True
        assert config.truncation_strategy == "head_tail"
        assert config.max_entities_in_context == 100
        assert config.max_relationships_in_context == 200
        assert config.prioritize_low_confidence is True

    def test_creation_with_custom_values(self):
        """Test creating ContextConfig with custom values."""
        config = ContextConfig(
            max_context_chars=4096,
            include_confidence_scores=False,
            include_validation_feedback=False,
            truncation_strategy="head",
            max_entities_in_context=50,
            max_relationships_in_context=100,
            prioritize_low_confidence=False,
        )

        assert config.max_context_chars == 4096
        assert config.include_confidence_scores is False
        assert config.include_validation_feedback is False
        assert config.truncation_strategy == "head"
        assert config.max_entities_in_context == 50
        assert config.max_relationships_in_context == 100
        assert config.prioritize_low_confidence is False

    def test_invalid_max_context_chars(self):
        """Test validation rejects non-positive max_context_chars."""
        with pytest.raises(ValueError, match="max_context_chars"):
            ContextConfig(max_context_chars=0)

    def test_invalid_truncation_strategy(self):
        """Test validation rejects invalid truncation_strategy."""
        with pytest.raises(ValueError, match="truncation_strategy"):
            ContextConfig(truncation_strategy="invalid")

    def test_invalid_max_entities_in_context(self):
        """Test validation rejects non-positive max_entities_in_context."""
        with pytest.raises(ValueError, match="max_entities_in_context"):
            ContextConfig(max_entities_in_context=0)

    def test_invalid_max_relationships_in_context(self):
        """Test validation rejects non-positive max_relationships_in_context."""
        with pytest.raises(ValueError, match="max_relationships_in_context"):
            ContextConfig(max_relationships_in_context=-1)


class TestMultiPassConfig:
    """Tests for MultiPassConfig dataclass."""

    def test_creation_with_defaults(self):
        """Test creating MultiPassConfig with default values."""
        config = MultiPassConfig()

        assert config.max_passes == 2
        assert config.gap_severity_threshold == 0.5
        assert config.refinement_batch_size == 5
        assert config.enable_parallel_refinement is True
        assert config.min_improvement_threshold == 0.05
        assert isinstance(config.validation_config, ValidationConfig)
        assert isinstance(config.context_config, ContextConfig)
        assert config.entity_type_priorities == []
        assert config.skip_entity_types == []

    def test_creation_with_custom_values(self):
        """Test creating MultiPassConfig with custom values."""
        validation_config = ValidationConfig(min_confidence_threshold=0.8)
        context_config = ContextConfig(max_context_chars=4096)

        config = MultiPassConfig(
            max_passes=3,
            gap_severity_threshold=0.6,
            refinement_batch_size=10,
            enable_parallel_refinement=False,
            min_improvement_threshold=0.1,
            validation_config=validation_config,
            context_config=context_config,
            entity_type_priorities=["Person", "Organization"],
            skip_entity_types=["Document"],
        )

        assert config.max_passes == 3
        assert config.gap_severity_threshold == 0.6
        assert config.refinement_batch_size == 10
        assert config.enable_parallel_refinement is False
        assert config.min_improvement_threshold == 0.1
        assert config.validation_config.min_confidence_threshold == 0.8
        assert config.context_config.max_context_chars == 4096
        assert config.entity_type_priorities == ["Person", "Organization"]
        assert config.skip_entity_types == ["Document"]

    def test_invalid_max_passes(self):
        """Test validation rejects max_passes < 1."""
        with pytest.raises(ValueError, match="max_passes"):
            MultiPassConfig(max_passes=0)

    def test_invalid_gap_severity_threshold_high(self):
        """Test validation rejects gap_severity_threshold > 1.0."""
        with pytest.raises(ValueError, match="gap_severity_threshold"):
            MultiPassConfig(gap_severity_threshold=1.5)

    def test_invalid_gap_severity_threshold_low(self):
        """Test validation rejects gap_severity_threshold < 0.0."""
        with pytest.raises(ValueError, match="gap_severity_threshold"):
            MultiPassConfig(gap_severity_threshold=-0.1)

    def test_invalid_refinement_batch_size(self):
        """Test validation rejects refinement_batch_size < 1."""
        with pytest.raises(ValueError, match="refinement_batch_size"):
            MultiPassConfig(refinement_batch_size=0)

    def test_invalid_min_improvement_threshold(self):
        """Test validation rejects invalid min_improvement_threshold."""
        with pytest.raises(ValueError, match="min_improvement_threshold"):
            MultiPassConfig(min_improvement_threshold=1.5)

    def test_should_refine_entity_type_included(self):
        """Test should_refine_entity_type returns True for non-skipped types."""
        config = MultiPassConfig(skip_entity_types=["Document"])

        assert config.should_refine_entity_type("Person") is True

    def test_should_refine_entity_type_skipped(self):
        """Test should_refine_entity_type returns False for skipped types."""
        config = MultiPassConfig(skip_entity_types=["Document"])

        assert config.should_refine_entity_type("Document") is False

    def test_get_entity_priority_prioritized(self):
        """Test get_entity_priority returns correct index for prioritized types."""
        config = MultiPassConfig(
            entity_type_priorities=["Person", "Organization", "Location"]
        )

        assert config.get_entity_priority("Person") == 0
        assert config.get_entity_priority("Organization") == 1
        assert config.get_entity_priority("Location") == 2

    def test_get_entity_priority_not_prioritized(self):
        """Test get_entity_priority returns list length for non-prioritized types."""
        config = MultiPassConfig(entity_type_priorities=["Person", "Organization"])

        assert config.get_entity_priority("Document") == 2
