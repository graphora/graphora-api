"""Extraction validator for validating completeness against ontology."""

import logging
from typing import Dict, List, Any, Optional, Set

from graphora_server.services.transform.models import BaseNode, RelationshipInstance
from .models import (
    ExtractionConfidence,
    ExtractionGap,
    GapType,
    ValidationResult,
)
from .config import ValidationConfig

logger = logging.getLogger(__name__)


class ExtractionValidator:
    """Validates extraction completeness and quality against ontology.

    This validator checks extracted entities and relationships against the
    ontology definition to identify gaps, missing properties, orphan nodes,
    and low-confidence extractions.
    """

    def __init__(
        self,
        ontology: Dict[str, Any],
        config: Optional[ValidationConfig] = None,
    ) -> None:
        """Initialize validator with ontology and configuration.

        Args:
            ontology: Parsed ontology dictionary with entities and relationships.
            config: Validation configuration. Uses defaults if not provided.
        """
        self.ontology = ontology
        self.config = config or ValidationConfig()
        self._entity_defs = ontology.get("entities", {})

    def validate(
        self,
        nodes: List[BaseNode],
        relationships: List[RelationshipInstance],
    ) -> ValidationResult:
        """Validate extraction completeness and quality.

        Args:
            nodes: List of extracted nodes.
            relationships: List of extracted relationships.

        Returns:
            ValidationResult with gaps, recommendations, and quality metrics.
        """
        result = ValidationResult()

        # Identify gaps
        result.gaps = self.identify_gaps(nodes, relationships)

        # Find low confidence entities
        if self.config.min_confidence_threshold > 0:
            result.low_confidence_entities = self.find_low_confidence_entities(
                nodes, self.config.min_confidence_threshold
            )

        # Find orphan nodes
        if self.config.check_orphan_nodes:
            result.orphan_nodes = self.find_orphan_nodes(nodes, relationships)

        # Check required properties
        if self.config.check_required_properties:
            missing_props = self.check_required_properties(nodes)
            for entity_id, props in missing_props.items():
                if props:
                    node = next((n for n in nodes if n.id == entity_id), None)
                    if node:
                        result.gaps.append(
                            ExtractionGap(
                                gap_type=GapType.INCOMPLETE_ENTITY,
                                entity_type=node.type,
                                entity_id=entity_id,
                                description=f"Missing required properties: {', '.join(props)}",
                                severity=min(
                                    1.0,
                                    len(props) * self.config.required_property_weight,
                                ),
                                context={"missing_properties": props},
                            )
                        )

        # Calculate overall metrics
        result.overall_confidence = self._calculate_overall_confidence(nodes)
        result.property_completeness = self._calculate_property_completeness(nodes)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        # Determine validity
        result.is_valid = (
            len(result.get_high_severity_gaps(0.7)) == 0
            and result.overall_confidence >= self.config.min_confidence_threshold
        )

        return result

    def identify_gaps(
        self,
        nodes: List[BaseNode],
        relationships: List[RelationshipInstance],
    ) -> List[ExtractionGap]:
        """Identify extraction gaps that need refinement.

        Args:
            nodes: List of extracted nodes.
            relationships: List of extracted relationships.

        Returns:
            List of identified extraction gaps.
        """
        gaps: List[ExtractionGap] = []

        # Check for incomplete entities
        gaps.extend(self._identify_incomplete_entities(nodes))

        # Check for orphan nodes
        if self.config.check_orphan_nodes:
            orphans = self.find_orphan_nodes(nodes, relationships)
            for orphan_id in orphans:
                node = next((n for n in nodes if n.id == orphan_id), None)
                if node:
                    gaps.append(
                        ExtractionGap(
                            gap_type=GapType.ORPHAN_NODE,
                            entity_type=node.type,
                            entity_id=orphan_id,
                            description=f"Node '{orphan_id}' has no relationships",
                            severity=self.config.orphan_node_severity,
                            context={"node_properties": node.properties},
                        )
                    )

        # Check for missing expected relationships
        if self.config.check_relationship_completeness:
            gaps.extend(self._identify_missing_relationships(nodes, relationships))

        # Check for low confidence extractions
        for node in nodes:
            confidence = node.confidence_score or 1.0
            if confidence < self.config.min_confidence_threshold:
                gaps.append(
                    ExtractionGap(
                        gap_type=GapType.LOW_CONFIDENCE,
                        entity_type=node.type,
                        entity_id=node.id,
                        description=f"Low confidence extraction ({confidence:.2f})",
                        severity=1.0 - confidence,
                        context={"confidence_score": confidence},
                    )
                )

        return gaps

    def find_low_confidence_entities(
        self,
        nodes: List[BaseNode],
        threshold: float = 0.7,
    ) -> List[ExtractionConfidence]:
        """Find entities with low confidence scores.

        Args:
            nodes: List of extracted nodes.
            threshold: Confidence threshold below which entities are flagged.

        Returns:
            List of ExtractionConfidence for low-confidence entities.
        """
        low_confidence: List[ExtractionConfidence] = []

        for node in nodes:
            confidence = node.confidence_score or 1.0
            if confidence < threshold:
                entity_def = self._entity_defs.get(node.type, {})
                props_def = entity_def.get("properties", {})

                # Find missing required properties
                missing = []
                uncertain = []
                for prop_name, prop_def in props_def.items():
                    if prop_def.get("required", False):
                        if (
                            prop_name not in node.properties
                            or node.properties.get(prop_name) is None
                        ):
                            missing.append(prop_name)
                        elif confidence < 0.5:
                            uncertain.append(prop_name)

                low_confidence.append(
                    ExtractionConfidence(
                        entity_id=node.id,
                        entity_type=node.type,
                        confidence_score=confidence,
                        missing_properties=missing,
                        uncertain_properties=uncertain,
                        source_chunks=(
                            node.provenance.chunk_ids if node.provenance else []
                        ),
                    )
                )

        return low_confidence

    def find_orphan_nodes(
        self,
        nodes: List[BaseNode],
        relationships: List[RelationshipInstance],
    ) -> List[str]:
        """Find nodes not connected by any relationship.

        Args:
            nodes: List of extracted nodes.
            relationships: List of extracted relationships.

        Returns:
            List of node IDs that have no relationships.
        """
        # Collect all node IDs that are in relationships
        connected_ids: Set[str] = set()
        for rel in relationships:
            connected_ids.add(rel.source_id)
            connected_ids.add(rel.target_id)

        # Find nodes not in any relationship
        orphans = [node.id for node in nodes if node.id not in connected_ids]

        return orphans

    def check_required_properties(
        self,
        nodes: List[BaseNode],
    ) -> Dict[str, List[str]]:
        """Check for missing required properties per entity.

        Args:
            nodes: List of extracted nodes.

        Returns:
            Dictionary mapping entity IDs to lists of missing required properties.
        """
        missing_props: Dict[str, List[str]] = {}

        for node in nodes:
            entity_def = self._entity_defs.get(node.type, {})
            props_def = entity_def.get("properties", {})

            missing = []
            for prop_name, prop_def in props_def.items():
                if prop_def.get("required", False):
                    value = node.properties.get(prop_name)
                    if value is None or (isinstance(value, str) and not value.strip()):
                        missing.append(prop_name)

            if missing:
                missing_props[node.id] = missing

        return missing_props

    def _identify_incomplete_entities(
        self, nodes: List[BaseNode]
    ) -> List[ExtractionGap]:
        """Identify entities with incomplete property coverage."""
        gaps: List[ExtractionGap] = []

        for node in nodes:
            entity_def = self._entity_defs.get(node.type, {})
            props_def = entity_def.get("properties", {})

            if not props_def:
                continue

            # Count filled vs total properties
            total_props = len(props_def)
            filled_props = sum(
                1
                for prop_name in props_def
                if node.properties.get(prop_name) is not None
            )

            fill_rate = filled_props / total_props if total_props > 0 else 1.0

            # Flag if less than 50% filled and has required fields missing
            required_missing = []
            for prop_name, prop_def in props_def.items():
                if prop_def.get("required", False):
                    if node.properties.get(prop_name) is None:
                        required_missing.append(prop_name)

            if required_missing or fill_rate < 0.5:
                severity = 0.3 + (len(required_missing) * 0.2)
                gaps.append(
                    ExtractionGap(
                        gap_type=GapType.INCOMPLETE_ENTITY,
                        entity_type=node.type,
                        entity_id=node.id,
                        description=f"Entity incomplete (fill rate: {fill_rate:.0%})",
                        severity=min(1.0, severity),
                        context={
                            "fill_rate": fill_rate,
                            "missing_required": required_missing,
                            "total_properties": total_props,
                            "filled_properties": filled_props,
                        },
                    )
                )

        return gaps

    def _identify_missing_relationships(
        self,
        nodes: List[BaseNode],
        relationships: List[RelationshipInstance],
    ) -> List[ExtractionGap]:
        """Identify expected relationships that are missing."""
        gaps: List[ExtractionGap] = []

        # Build a map of existing relationships by source type
        rel_by_source: Dict[str, Dict[str, Set[str]]] = {}
        for rel in relationships:
            if rel.source_type not in rel_by_source:
                rel_by_source[rel.source_type] = {}
            if rel.type not in rel_by_source[rel.source_type]:
                rel_by_source[rel.source_type][rel.type] = set()
            rel_by_source[rel.source_type][rel.type].add(rel.source_id)

        # Check each entity type for expected relationships
        for entity_type, entity_def in self._entity_defs.items():
            expected_rels = entity_def.get("relationships", {})

            for rel_name, rel_def in expected_rels.items():
                # Get entities of this type
                entities_of_type = [n for n in nodes if n.type == entity_type]

                # Get entities with this relationship
                entities_with_rel = rel_by_source.get(entity_type, {}).get(
                    rel_name, set()
                )

                # Find entities missing this relationship
                for entity in entities_of_type:
                    if entity.id not in entities_with_rel:
                        # Check if relationship is likely required
                        cardinality = rel_def.get("cardinality", "")
                        is_required = "one" in cardinality.lower() or rel_def.get(
                            "required", False
                        )

                        if is_required:
                            gaps.append(
                                ExtractionGap(
                                    gap_type=GapType.MISSING_RELATIONSHIP,
                                    entity_type=entity_type,
                                    entity_id=entity.id,
                                    description=f"Missing expected relationship '{rel_name}'",
                                    severity=0.6,
                                    context={
                                        "relationship_type": rel_name,
                                        "target_type": rel_def.get("target"),
                                        "cardinality": cardinality,
                                    },
                                )
                            )

        return gaps

    def _calculate_overall_confidence(self, nodes: List[BaseNode]) -> float:
        """Calculate aggregate confidence score across all nodes."""
        if not nodes:
            return 1.0

        total_confidence = sum(node.confidence_score or 1.0 for node in nodes)
        return total_confidence / len(nodes)

    def _calculate_property_completeness(self, nodes: List[BaseNode]) -> float:
        """Calculate ratio of filled properties to total expected properties."""
        if not nodes:
            return 1.0

        total_expected = 0
        total_filled = 0

        for node in nodes:
            entity_def = self._entity_defs.get(node.type, {})
            props_def = entity_def.get("properties", {})

            total_expected += len(props_def)
            total_filled += sum(
                1
                for prop_name in props_def
                if node.properties.get(prop_name) is not None
            )

        return total_filled / total_expected if total_expected > 0 else 1.0

    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate actionable recommendations based on validation result."""
        recommendations: List[str] = []

        # Group gaps by type
        gap_counts: Dict[GapType, int] = {}
        for gap in result.gaps:
            gap_counts[gap.gap_type] = gap_counts.get(gap.gap_type, 0) + 1

        # Generate recommendations based on gap types
        if gap_counts.get(GapType.INCOMPLETE_ENTITY, 0) > 0:
            count = gap_counts[GapType.INCOMPLETE_ENTITY]
            recommendations.append(
                f"Re-extract {count} incomplete entities with focused prompts"
            )

        if gap_counts.get(GapType.ORPHAN_NODE, 0) > 0:
            count = gap_counts[GapType.ORPHAN_NODE]
            recommendations.append(f"Find relationships for {count} orphan nodes")

        if gap_counts.get(GapType.MISSING_RELATIONSHIP, 0) > 0:
            count = gap_counts[GapType.MISSING_RELATIONSHIP]
            recommendations.append(
                f"Re-examine source text for {count} missing relationships"
            )

        if gap_counts.get(GapType.LOW_CONFIDENCE, 0) > 0:
            count = gap_counts[GapType.LOW_CONFIDENCE]
            recommendations.append(
                f"Verify {count} low-confidence extractions against source"
            )

        if result.property_completeness < 0.7:
            recommendations.append(
                f"Property completeness is low ({result.property_completeness:.0%}). "
                "Consider adding more extraction context."
            )

        return recommendations
