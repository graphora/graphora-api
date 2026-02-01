"""Unit tests for extraction models."""

from app.services.extraction.models import (
    ExtractionConfidence,
    ExtractionGap,
    GapType,
    ValidationResult,
    RefinementResult,
)


class TestExtractionConfidence:
    """Tests for ExtractionConfidence dataclass."""

    def test_creation_with_defaults(self):
        """Test creating ExtractionConfidence with default values."""
        confidence = ExtractionConfidence(
            entity_id="entity-1",
            entity_type="Person",
            confidence_score=0.8,
        )

        assert confidence.entity_id == "entity-1"
        assert confidence.entity_type == "Person"
        assert confidence.confidence_score == 0.8
        assert confidence.missing_properties == []
        assert confidence.uncertain_properties == []
        assert confidence.source_chunks == []

    def test_creation_with_all_fields(self):
        """Test creating ExtractionConfidence with all fields."""
        confidence = ExtractionConfidence(
            entity_id="entity-1",
            entity_type="Person",
            confidence_score=0.5,
            missing_properties=["email", "phone"],
            uncertain_properties=["address"],
            source_chunks=[0, 1, 2],
        )

        assert confidence.missing_properties == ["email", "phone"]
        assert confidence.uncertain_properties == ["address"]
        assert confidence.source_chunks == [0, 1, 2]

    def test_is_complete_true(self):
        """Test is_complete returns True when no missing properties."""
        confidence = ExtractionConfidence(
            entity_id="entity-1",
            entity_type="Person",
            confidence_score=0.9,
            missing_properties=[],
        )

        assert confidence.is_complete() is True

    def test_is_complete_false(self):
        """Test is_complete returns False when properties are missing."""
        confidence = ExtractionConfidence(
            entity_id="entity-1",
            entity_type="Person",
            confidence_score=0.9,
            missing_properties=["email"],
        )

        assert confidence.is_complete() is False

    def test_needs_refinement_low_confidence(self):
        """Test needs_refinement returns True for low confidence."""
        confidence = ExtractionConfidence(
            entity_id="entity-1",
            entity_type="Person",
            confidence_score=0.5,
        )

        assert confidence.needs_refinement(threshold=0.7) is True

    def test_needs_refinement_missing_properties(self):
        """Test needs_refinement returns True when properties missing."""
        confidence = ExtractionConfidence(
            entity_id="entity-1",
            entity_type="Person",
            confidence_score=0.9,
            missing_properties=["email"],
        )

        assert confidence.needs_refinement(threshold=0.7) is True

    def test_needs_refinement_uncertain_properties(self):
        """Test needs_refinement returns True when properties uncertain."""
        confidence = ExtractionConfidence(
            entity_id="entity-1",
            entity_type="Person",
            confidence_score=0.9,
            uncertain_properties=["address"],
        )

        assert confidence.needs_refinement(threshold=0.7) is True

    def test_needs_refinement_false(self):
        """Test needs_refinement returns False when all criteria met."""
        confidence = ExtractionConfidence(
            entity_id="entity-1",
            entity_type="Person",
            confidence_score=0.9,
        )

        assert confidence.needs_refinement(threshold=0.7) is False


class TestExtractionGap:
    """Tests for ExtractionGap dataclass."""

    def test_creation_with_defaults(self):
        """Test creating ExtractionGap with default values."""
        gap = ExtractionGap(gap_type=GapType.INCOMPLETE_ENTITY)

        assert gap.gap_type == GapType.INCOMPLETE_ENTITY
        assert gap.entity_type is None
        assert gap.entity_id is None
        assert gap.chunk_indices == []
        assert gap.description == ""
        assert gap.severity == 0.5
        assert gap.context == {}

    def test_creation_with_all_fields(self):
        """Test creating ExtractionGap with all fields."""
        gap = ExtractionGap(
            gap_type=GapType.ORPHAN_NODE,
            entity_type="Person",
            entity_id="entity-1",
            chunk_indices=[0, 2],
            description="Node has no relationships",
            severity=0.7,
            context={"node_properties": {"name": "John"}},
        )

        assert gap.gap_type == GapType.ORPHAN_NODE
        assert gap.entity_type == "Person"
        assert gap.entity_id == "entity-1"
        assert gap.chunk_indices == [0, 2]
        assert gap.description == "Node has no relationships"
        assert gap.severity == 0.7
        assert gap.context == {"node_properties": {"name": "John"}}

    def test_severity_clamped_min(self):
        """Test severity is clamped to minimum of 0.0."""
        gap = ExtractionGap(
            gap_type=GapType.LOW_CONFIDENCE,
            severity=-0.5,
        )

        assert gap.severity == 0.0

    def test_severity_clamped_max(self):
        """Test severity is clamped to maximum of 1.0."""
        gap = ExtractionGap(
            gap_type=GapType.LOW_CONFIDENCE,
            severity=1.5,
        )

        assert gap.severity == 1.0


class TestGapType:
    """Tests for GapType enum."""

    def test_gap_types_exist(self):
        """Test all expected gap types exist."""
        assert GapType.MISSING_ENTITY.value == "missing_entity"
        assert GapType.INCOMPLETE_ENTITY.value == "incomplete_entity"
        assert GapType.MISSING_RELATIONSHIP.value == "missing_relationship"
        assert GapType.ORPHAN_NODE.value == "orphan_node"
        assert GapType.LOW_CONFIDENCE.value == "low_confidence"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_creation_with_defaults(self):
        """Test creating ValidationResult with default values."""
        result = ValidationResult()

        assert result.is_valid is True
        assert result.gaps == []
        assert result.low_confidence_entities == []
        assert result.orphan_nodes == []
        assert result.missing_relationships == []
        assert result.recommendations == []
        assert result.overall_confidence == 1.0
        assert result.property_completeness == 1.0

    def test_needs_refinement_with_gaps(self):
        """Test needs_refinement returns True when gaps exist."""
        result = ValidationResult(
            gaps=[ExtractionGap(gap_type=GapType.INCOMPLETE_ENTITY)]
        )

        assert result.needs_refinement() is True

    def test_needs_refinement_with_low_confidence(self):
        """Test needs_refinement returns True with low confidence entities."""
        result = ValidationResult(
            low_confidence_entities=[
                ExtractionConfidence(
                    entity_id="e1",
                    entity_type="Person",
                    confidence_score=0.5,
                )
            ]
        )

        assert result.needs_refinement() is True

    def test_needs_refinement_with_orphans(self):
        """Test needs_refinement returns True with orphan nodes."""
        result = ValidationResult(orphan_nodes=["node-1", "node-2"])

        assert result.needs_refinement() is True

    def test_needs_refinement_false(self):
        """Test needs_refinement returns False when no issues."""
        result = ValidationResult()

        assert result.needs_refinement() is False

    def test_get_high_severity_gaps(self):
        """Test filtering gaps by severity threshold."""
        gaps = [
            ExtractionGap(gap_type=GapType.INCOMPLETE_ENTITY, severity=0.3),
            ExtractionGap(gap_type=GapType.ORPHAN_NODE, severity=0.6),
            ExtractionGap(gap_type=GapType.LOW_CONFIDENCE, severity=0.8),
        ]
        result = ValidationResult(gaps=gaps)

        high_severity = result.get_high_severity_gaps(threshold=0.5)

        assert len(high_severity) == 2
        assert high_severity[0].severity == 0.6
        assert high_severity[1].severity == 0.8

    def test_get_refinement_priority_chunks(self):
        """Test getting unique chunk indices from gaps."""
        gaps = [
            ExtractionGap(gap_type=GapType.INCOMPLETE_ENTITY, chunk_indices=[0, 1]),
            ExtractionGap(gap_type=GapType.ORPHAN_NODE, chunk_indices=[1, 2]),
            ExtractionGap(gap_type=GapType.LOW_CONFIDENCE, chunk_indices=[2, 3]),
        ]
        result = ValidationResult(gaps=gaps)

        priority_chunks = result.get_refinement_priority_chunks()

        assert priority_chunks == [0, 1, 2, 3]


class TestRefinementResult:
    """Tests for RefinementResult dataclass."""

    def test_creation_with_defaults(self):
        """Test creating RefinementResult with default values."""
        result = RefinementResult(pass_number=2)

        assert result.pass_number == 2
        assert result.gaps_addressed == 0
        assert result.new_nodes_count == 0
        assert result.updated_nodes_count == 0
        assert result.new_relationships_count == 0
        assert result.remaining_gaps == []
        assert result.confidence_improvement == 0.0

    def test_is_improvement_true_gaps_addressed(self):
        """Test is_improvement returns True when gaps addressed."""
        result = RefinementResult(pass_number=2, gaps_addressed=3)

        assert result.is_improvement() is True

    def test_is_improvement_true_new_nodes(self):
        """Test is_improvement returns True when new nodes found."""
        result = RefinementResult(pass_number=2, new_nodes_count=2)

        assert result.is_improvement() is True

    def test_is_improvement_true_updated_nodes(self):
        """Test is_improvement returns True when nodes updated."""
        result = RefinementResult(pass_number=2, updated_nodes_count=1)

        assert result.is_improvement() is True

    def test_is_improvement_true_new_relationships(self):
        """Test is_improvement returns True when new relationships found."""
        result = RefinementResult(pass_number=2, new_relationships_count=5)

        assert result.is_improvement() is True

    def test_is_improvement_false(self):
        """Test is_improvement returns False when no improvements."""
        result = RefinementResult(pass_number=2)

        assert result.is_improvement() is False
