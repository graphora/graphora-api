"""Unit tests for ExtractionValidator."""

import pytest
from graphora_server.services.extraction.validator import ExtractionValidator
from graphora_server.services.extraction.config import ValidationConfig
from graphora_server.services.extraction.models import GapType
from graphora_server.services.transform.models import BaseNode, RelationshipInstance


@pytest.fixture
def sample_ontology():
    """Create a sample ontology for testing."""
    return {
        "version": "1.0",
        "entities": {
            "Person": {
                "properties": {
                    "name": {"type": "str", "required": True},
                    "email": {"type": "str", "required": True},
                    "phone": {"type": "str", "required": False},
                    "address": {"type": "str", "required": False},
                },
                "relationships": {
                    "WORKS_AT": {
                        "target": "Organization",
                        "cardinality": "one-to-many",
                    },
                },
            },
            "Organization": {
                "properties": {
                    "name": {"type": "str", "required": True},
                    "industry": {"type": "str", "required": False},
                },
                "relationships": {},
            },
        },
    }


@pytest.fixture
def validator(sample_ontology):
    """Create an ExtractionValidator instance."""
    return ExtractionValidator(sample_ontology)


@pytest.fixture
def complete_node():
    """Create a complete Person node."""
    return BaseNode(
        id="person-1",
        type="Person",
        properties={"name": "John Doe", "email": "john@example.com"},
        confidence_score=0.9,
    )


@pytest.fixture
def incomplete_node():
    """Create an incomplete Person node (missing required property)."""
    return BaseNode(
        id="person-2",
        type="Person",
        properties={"name": "Jane Doe"},  # Missing email
        confidence_score=0.8,
    )


@pytest.fixture
def low_confidence_node():
    """Create a low confidence Person node."""
    return BaseNode(
        id="person-3",
        type="Person",
        properties={"name": "Bob Smith", "email": "bob@example.com"},
        confidence_score=0.5,
    )


@pytest.fixture
def organization_node():
    """Create an Organization node."""
    return BaseNode(
        id="org-1",
        type="Organization",
        properties={"name": "Acme Corp"},
        confidence_score=0.9,
    )


@pytest.fixture
def relationship():
    """Create a WORKS_AT relationship."""
    return RelationshipInstance(
        id="rel-1",
        type="WORKS_AT",
        source_id="person-1",
        target_id="org-1",
        source_type="Person",
        target_type="Organization",
        properties={},
    )


class TestExtractionValidator:
    """Tests for ExtractionValidator class."""

    def test_validator_initialization(self, sample_ontology):
        """Test validator initializes correctly."""
        validator = ExtractionValidator(sample_ontology)

        assert validator.ontology == sample_ontology
        assert isinstance(validator.config, ValidationConfig)

    def test_validator_with_custom_config(self, sample_ontology):
        """Test validator with custom configuration."""
        config = ValidationConfig(
            min_confidence_threshold=0.8,
            check_orphan_nodes=False,
        )
        validator = ExtractionValidator(sample_ontology, config=config)

        assert validator.config.min_confidence_threshold == 0.8
        assert validator.config.check_orphan_nodes is False


class TestValidate:
    """Tests for validate method."""

    def test_validate_complete_extraction(
        self, validator, complete_node, organization_node, relationship
    ):
        """Test validation passes for complete extraction."""
        nodes = [complete_node, organization_node]
        relationships = [relationship]

        result = validator.validate(nodes, relationships)

        assert result.is_valid is True
        assert len(result.low_confidence_entities) == 0

    def test_validate_with_incomplete_entities(
        self, validator, incomplete_node, organization_node
    ):
        """Test validation identifies incomplete entities."""
        nodes = [incomplete_node, organization_node]
        relationships = []

        result = validator.validate(nodes, relationships)

        # Should have gaps for incomplete entity and orphan nodes
        incomplete_gaps = [
            g for g in result.gaps if g.gap_type == GapType.INCOMPLETE_ENTITY
        ]
        assert len(incomplete_gaps) >= 1

    def test_validate_with_low_confidence(
        self, validator, low_confidence_node, organization_node
    ):
        """Test validation identifies low confidence entities."""
        nodes = [low_confidence_node, organization_node]
        relationships = []

        result = validator.validate(nodes, relationships)

        assert len(result.low_confidence_entities) >= 1

    def test_validate_calculates_overall_confidence(
        self, validator, complete_node, low_confidence_node
    ):
        """Test overall confidence is calculated correctly."""
        nodes = [complete_node, low_confidence_node]  # 0.9 and 0.5
        relationships = []

        result = validator.validate(nodes, relationships)

        expected_confidence = (0.9 + 0.5) / 2
        assert result.overall_confidence == expected_confidence


class TestIdentifyGaps:
    """Tests for identify_gaps method."""

    def test_identify_gaps_empty(
        self, validator, complete_node, organization_node, relationship
    ):
        """Test no gaps identified for complete extraction."""
        nodes = [complete_node, organization_node]
        relationships = [relationship]

        gaps = validator.identify_gaps(nodes, relationships)

        # May have some gaps for missing relationships depending on config
        high_severity = [g for g in gaps if g.severity >= 0.7]
        assert len(high_severity) == 0

    def test_identify_gaps_incomplete_entity(self, validator, incomplete_node):
        """Test gaps identified for incomplete entities."""
        gaps = validator.identify_gaps([incomplete_node], [])

        incomplete_gaps = [g for g in gaps if g.gap_type == GapType.INCOMPLETE_ENTITY]
        assert len(incomplete_gaps) >= 1
        assert incomplete_gaps[0].entity_id == "person-2"

    def test_identify_gaps_orphan_nodes(
        self, validator, complete_node, organization_node
    ):
        """Test gaps identified for orphan nodes."""
        nodes = [complete_node, organization_node]
        relationships = []  # No relationships

        gaps = validator.identify_gaps(nodes, relationships)

        orphan_gaps = [g for g in gaps if g.gap_type == GapType.ORPHAN_NODE]
        assert len(orphan_gaps) == 2  # Both nodes are orphans

    def test_identify_gaps_low_confidence(self, validator, low_confidence_node):
        """Test gaps identified for low confidence extractions."""
        gaps = validator.identify_gaps([low_confidence_node], [])

        low_conf_gaps = [g for g in gaps if g.gap_type == GapType.LOW_CONFIDENCE]
        assert len(low_conf_gaps) == 1
        assert low_conf_gaps[0].entity_id == "person-3"


class TestFindLowConfidenceEntities:
    """Tests for find_low_confidence_entities method."""

    def test_find_low_confidence_entities_below_threshold(
        self, validator, low_confidence_node
    ):
        """Test entities below threshold are found."""
        result = validator.find_low_confidence_entities(
            [low_confidence_node], threshold=0.7
        )

        assert len(result) == 1
        assert result[0].entity_id == "person-3"
        assert result[0].confidence_score == 0.5

    def test_find_low_confidence_entities_above_threshold(
        self, validator, complete_node
    ):
        """Test entities above threshold are not included."""
        result = validator.find_low_confidence_entities([complete_node], threshold=0.7)

        assert len(result) == 0

    def test_find_low_confidence_includes_missing_properties(
        self, validator, sample_ontology
    ):
        """Test missing properties are identified for low confidence entities."""
        node = BaseNode(
            id="person-low",
            type="Person",
            properties={"name": "Test"},  # Missing email
            confidence_score=0.5,
        )

        result = validator.find_low_confidence_entities([node], threshold=0.7)

        assert len(result) == 1
        assert "email" in result[0].missing_properties


class TestFindOrphanNodes:
    """Tests for find_orphan_nodes method."""

    def test_find_orphan_nodes_all_orphans(
        self, validator, complete_node, organization_node
    ):
        """Test all nodes are orphans when no relationships."""
        orphans = validator.find_orphan_nodes([complete_node, organization_node], [])

        assert len(orphans) == 2
        assert "person-1" in orphans
        assert "org-1" in orphans

    def test_find_orphan_nodes_none_orphans(
        self, validator, complete_node, organization_node, relationship
    ):
        """Test no orphans when all nodes connected."""
        orphans = validator.find_orphan_nodes(
            [complete_node, organization_node], [relationship]
        )

        assert len(orphans) == 0

    def test_find_orphan_nodes_partial(
        self, validator, complete_node, organization_node, relationship
    ):
        """Test partial orphans detected."""
        extra_node = BaseNode(
            id="person-extra",
            type="Person",
            properties={"name": "Extra", "email": "extra@example.com"},
        )
        nodes = [complete_node, organization_node, extra_node]

        orphans = validator.find_orphan_nodes(nodes, [relationship])

        assert len(orphans) == 1
        assert "person-extra" in orphans


class TestCheckRequiredProperties:
    """Tests for check_required_properties method."""

    def test_check_required_properties_complete(self, validator, complete_node):
        """Test complete node has no missing properties."""
        result = validator.check_required_properties([complete_node])

        assert (
            complete_node.id not in result or len(result.get(complete_node.id, [])) == 0
        )

    def test_check_required_properties_missing(self, validator, incomplete_node):
        """Test incomplete node reports missing properties."""
        result = validator.check_required_properties([incomplete_node])

        assert incomplete_node.id in result
        assert "email" in result[incomplete_node.id]

    def test_check_required_properties_empty_string(self, validator):
        """Test empty string values are treated as missing."""
        node = BaseNode(
            id="person-empty",
            type="Person",
            properties={"name": "Test", "email": ""},
        )

        result = validator.check_required_properties([node])

        assert "email" in result.get(node.id, [])

    def test_check_required_properties_none_value(self, validator):
        """Test None values are treated as missing."""
        node = BaseNode(
            id="person-none",
            type="Person",
            properties={"name": "Test", "email": None},
        )

        result = validator.check_required_properties([node])

        assert "email" in result.get(node.id, [])
