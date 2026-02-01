"""Data models for extraction quality tracking and validation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class GapType(str, Enum):
    """Types of extraction gaps that can be detected."""

    MISSING_ENTITY = "missing_entity"
    INCOMPLETE_ENTITY = "incomplete_entity"
    MISSING_RELATIONSHIP = "missing_relationship"
    ORPHAN_NODE = "orphan_node"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class ExtractionConfidence:
    """Confidence information for an extracted entity.

    Attributes:
        entity_id: Unique identifier of the entity.
        entity_type: Type of the entity from ontology.
        confidence_score: Overall confidence score (0.0 to 1.0).
        missing_properties: List of required properties that were not extracted.
        uncertain_properties: List of properties with low confidence.
        source_chunks: Indices of chunks where this entity was mentioned.
    """

    entity_id: str
    entity_type: str
    confidence_score: float
    missing_properties: List[str] = field(default_factory=list)
    uncertain_properties: List[str] = field(default_factory=list)
    source_chunks: List[int] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if entity has all required properties."""
        return len(self.missing_properties) == 0

    def needs_refinement(self, threshold: float = 0.7) -> bool:
        """Check if entity needs refinement based on confidence and completeness."""
        return (
            self.confidence_score < threshold
            or len(self.missing_properties) > 0
            or len(self.uncertain_properties) > 0
        )


@dataclass
class ExtractionGap:
    """Represents a gap in extraction that needs to be addressed.

    Attributes:
        gap_type: Type of gap (MISSING_ENTITY, INCOMPLETE_ENTITY, etc.).
        entity_type: Type of entity involved (if applicable).
        entity_id: ID of the entity (for incomplete/orphan gaps).
        chunk_indices: List of chunk indices relevant to this gap.
        description: Human-readable description of the gap.
        severity: Severity score (0.0 to 1.0, higher = more important).
        context: Additional context for addressing the gap.
    """

    gap_type: GapType
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    chunk_indices: List[int] = field(default_factory=list)
    description: str = ""
    severity: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate severity is within bounds."""
        self.severity = max(0.0, min(1.0, self.severity))


@dataclass
class ValidationResult:
    """Result of extraction validation.

    Attributes:
        is_valid: Whether extraction meets minimum quality thresholds.
        gaps: List of identified extraction gaps.
        low_confidence_entities: Entities with confidence below threshold.
        orphan_nodes: Node IDs that have no relationships.
        missing_relationships: Expected relationships that were not found.
        recommendations: List of actionable recommendations.
        overall_confidence: Aggregate confidence score for the extraction.
        property_completeness: Ratio of filled properties to total properties.
    """

    is_valid: bool = True
    gaps: List[ExtractionGap] = field(default_factory=list)
    low_confidence_entities: List[ExtractionConfidence] = field(default_factory=list)
    orphan_nodes: List[str] = field(default_factory=list)
    missing_relationships: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    overall_confidence: float = 1.0
    property_completeness: float = 1.0

    def needs_refinement(self) -> bool:
        """Check if extraction needs refinement pass."""
        return (
            len(self.gaps) > 0
            or len(self.low_confidence_entities) > 0
            or len(self.orphan_nodes) > 0
        )

    def get_high_severity_gaps(self, threshold: float = 0.5) -> List[ExtractionGap]:
        """Get gaps above the severity threshold."""
        return [gap for gap in self.gaps if gap.severity >= threshold]

    def get_refinement_priority_chunks(self) -> List[int]:
        """Get chunk indices that should be prioritized for refinement."""
        chunk_set: set[int] = set()
        for gap in self.gaps:
            chunk_set.update(gap.chunk_indices)
        return sorted(chunk_set)


@dataclass
class RefinementResult:
    """Result of a refinement pass.

    Attributes:
        pass_number: Which refinement pass this represents.
        gaps_addressed: Number of gaps that were addressed.
        new_nodes_count: Number of new nodes extracted.
        updated_nodes_count: Number of existing nodes updated.
        new_relationships_count: Number of new relationships found.
        remaining_gaps: Gaps that still need to be addressed.
        confidence_improvement: How much overall confidence improved.
    """

    pass_number: int
    gaps_addressed: int = 0
    new_nodes_count: int = 0
    updated_nodes_count: int = 0
    new_relationships_count: int = 0
    remaining_gaps: List[ExtractionGap] = field(default_factory=list)
    confidence_improvement: float = 0.0

    def is_improvement(self) -> bool:
        """Check if refinement pass made meaningful improvements."""
        return (
            self.gaps_addressed > 0
            or self.new_nodes_count > 0
            or self.updated_nodes_count > 0
            or self.new_relationships_count > 0
        )
