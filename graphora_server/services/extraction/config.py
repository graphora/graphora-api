"""Configuration classes for multi-pass extraction."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationConfig:
    """Configuration for extraction validation.

    Attributes:
        min_confidence_threshold: Minimum confidence score for entities (0.0-1.0).
        check_required_properties: Whether to check for missing required properties.
        check_orphan_nodes: Whether to detect nodes without relationships.
        check_relationship_completeness: Whether to validate relationship coverage.
        required_property_weight: Weight for missing required properties in severity.
        orphan_node_severity: Default severity for orphan node gaps.
    """

    min_confidence_threshold: float = 0.7
    check_required_properties: bool = True
    check_orphan_nodes: bool = True
    check_relationship_completeness: bool = True
    required_property_weight: float = 0.3
    orphan_node_severity: float = 0.4

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.required_property_weight <= 1.0:
            raise ValueError("required_property_weight must be between 0.0 and 1.0")
        if not 0.0 <= self.orphan_node_severity <= 1.0:
            raise ValueError("orphan_node_severity must be between 0.0 and 1.0")


@dataclass
class ContextConfig:
    """Configuration for context building.

    Attributes:
        max_context_chars: Maximum characters for context strings.
        include_confidence_scores: Whether to include confidence in context.
        include_validation_feedback: Whether to include validation feedback.
        truncation_strategy: How to truncate long context ("head", "tail", "head_tail").
        max_entities_in_context: Maximum entities to include in context.
        max_relationships_in_context: Maximum relationships in context.
        prioritize_low_confidence: Whether to prioritize low-confidence entities in context.
    """

    max_context_chars: int = 8192
    include_confidence_scores: bool = True
    include_validation_feedback: bool = True
    truncation_strategy: str = "head_tail"
    max_entities_in_context: int = 100
    max_relationships_in_context: int = 200
    prioritize_low_confidence: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if self.truncation_strategy not in ("head", "tail", "head_tail"):
            raise ValueError(
                "truncation_strategy must be 'head', 'tail', or 'head_tail'"
            )
        if self.max_entities_in_context <= 0:
            raise ValueError("max_entities_in_context must be positive")
        if self.max_relationships_in_context <= 0:
            raise ValueError("max_relationships_in_context must be positive")


@dataclass
class MultiPassConfig:
    """Configuration for multi-pass extraction.

    Attributes:
        max_passes: Maximum number of extraction passes (including initial).
        gap_severity_threshold: Minimum severity for gaps to trigger refinement.
        refinement_batch_size: Maximum gaps to address per refinement pass.
        enable_parallel_refinement: Whether to process refinement chunks in parallel.
        min_improvement_threshold: Minimum improvement to continue refinement.
        validation_config: Configuration for validation.
        context_config: Configuration for context building.
        entity_type_priorities: Priority order for entity types in refinement.
        skip_entity_types: Entity types to skip in refinement.
    """

    max_passes: int = 2
    gap_severity_threshold: float = 0.5
    refinement_batch_size: int = 5
    enable_parallel_refinement: bool = True
    min_improvement_threshold: float = 0.05
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    context_config: ContextConfig = field(default_factory=ContextConfig)
    entity_type_priorities: List[str] = field(default_factory=list)
    skip_entity_types: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_passes < 1:
            raise ValueError("max_passes must be at least 1")
        if not 0.0 <= self.gap_severity_threshold <= 1.0:
            raise ValueError("gap_severity_threshold must be between 0.0 and 1.0")
        if self.refinement_batch_size < 1:
            raise ValueError("refinement_batch_size must be at least 1")
        if not 0.0 <= self.min_improvement_threshold <= 1.0:
            raise ValueError("min_improvement_threshold must be between 0.0 and 1.0")

    def should_refine_entity_type(self, entity_type: str) -> bool:
        """Check if an entity type should be refined."""
        return entity_type not in self.skip_entity_types

    def get_entity_priority(self, entity_type: str) -> int:
        """Get priority for an entity type (lower = higher priority)."""
        if entity_type in self.entity_type_priorities:
            return self.entity_type_priorities.index(entity_type)
        return len(self.entity_type_priorities)
