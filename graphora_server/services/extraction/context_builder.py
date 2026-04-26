"""Enhanced context builder for multi-pass extraction.

This module provides rich context building for LLM extraction including:
- Relationship schema hints for relationship-aware entity extraction
- Chain-of-thought guidance based on ontology patterns
- Quality indicators and validation feedback
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set

from graphora_server.services.transform.models import BaseNode, RelationshipInstance
from .models import ExtractionGap, ValidationResult, GapType
from .config import ContextConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextEnvelope:
    """Context string plus metadata about truncation.

    Attributes:
        text: The context string.
        truncated: Whether the context was truncated.
        raw_length: Original length before truncation.
        entity_count: Number of entities included.
        relationship_count: Number of relationships included.
    """

    text: str
    truncated: bool
    raw_length: int
    entity_count: int = 0
    relationship_count: int = 0


class EnhancedContextBuilder:
    """Build rich context for LLM extraction with quality indicators.

    This builder creates context strings that include not just entity and
    relationship information, but also quality indicators like confidence
    scores and validation feedback to guide refinement passes.
    """

    TRUNCATION_SENTINEL = "\n...[truncated]...\n"

    def __init__(
        self,
        ontology: Dict[str, Any],
        config: Optional[ContextConfig] = None,
    ) -> None:
        """Initialize context builder.

        Args:
            ontology: Parsed ontology dictionary.
            config: Context building configuration.
        """
        self.ontology = ontology
        self.config = config or ContextConfig()
        self._entity_defs = ontology.get("entities", {})
        self._relationship_defs = ontology.get("relationships", {})
        # Cache relationship schema
        self._relationship_schema_cache: Optional[Dict[str, List[Dict[str, Any]]]] = (
            None
        )

    def build_node_context(
        self,
        nodes: List[BaseNode],
        include_confidence: bool = True,
        include_validation: bool = False,
        validation_result: Optional[ValidationResult] = None,
        include_relationship_hints: bool = False,
    ) -> ContextEnvelope:
        """Build enhanced node context with quality indicators.

        Args:
            nodes: List of extracted nodes.
            include_confidence: Whether to include confidence scores.
            include_validation: Whether to include validation feedback.
            validation_result: Validation result for feedback (if include_validation).
            include_relationship_hints: Whether to include relationship schema hints.

        Returns:
            ContextEnvelope with the built context.
        """
        if not nodes:
            return ContextEnvelope(
                text="No entities extracted yet.",
                truncated=False,
                raw_length=0,
                entity_count=0,
            )

        # Sort nodes by type and confidence for consistency
        sorted_nodes = self._sort_nodes_for_context(nodes)

        # Limit number of nodes
        if len(sorted_nodes) > self.config.max_entities_in_context:
            sorted_nodes = sorted_nodes[: self.config.max_entities_in_context]

        lines: List[str] = []

        # Add validation summary if requested
        if include_validation and validation_result:
            lines.extend(self._build_validation_summary(validation_result))
            lines.append("")

        # Add relationship schema hints if requested
        if include_relationship_hints:
            entity_types = {node.type for node in sorted_nodes}
            hints = self._build_relationship_schema_hints(entity_types)
            if hints:
                lines.extend(hints)
                lines.append("")

        # Build node entries
        for node in sorted_nodes:
            line = self._format_node(node, include_confidence)
            lines.append(line)

        raw_context = "\n".join(lines) + "\n"

        # Apply truncation if needed
        envelope = self._apply_truncation(
            raw_context,
            entity_count=len(sorted_nodes),
        )

        return envelope

    def build_relationship_aware_entity_context(
        self,
        nodes: List[BaseNode],
        include_confidence: bool = True,
    ) -> ContextEnvelope:
        """Build entity context that includes relationship pattern hints.

        This context is designed for entity extraction and includes:
        - Existing entities with their properties
        - Relationship patterns from ontology to guide entity extraction
        - Expected entity types based on relationship definitions

        Args:
            nodes: List of extracted nodes.
            include_confidence: Whether to include confidence scores.

        Returns:
            ContextEnvelope with relationship-aware entity context.
        """
        lines: List[str] = []

        # Section 1: Relationship Schema Hints
        entity_types = {node.type for node in nodes} if nodes else set()
        # Also include all entity types from ontology for comprehensive hints
        entity_types.update(self._entity_defs.keys())

        schema_hints = self._build_relationship_schema_hints(entity_types)
        if schema_hints:
            lines.append(
                "=== RELATIONSHIP PATTERNS (for context during entity extraction) ==="
            )
            lines.extend(schema_hints)
            lines.append("")
            lines.append(
                "NOTE: When extracting entities, consider what relationships they might have."
            )
            lines.append(
                "Missing entities may be implied by relationship patterns above."
            )
            lines.append("")

        # Section 2: Existing Entities
        if nodes:
            lines.append("=== PREVIOUSLY IDENTIFIED ENTITIES ===")
            sorted_nodes = self._sort_nodes_for_context(nodes)
            if len(sorted_nodes) > self.config.max_entities_in_context:
                sorted_nodes = sorted_nodes[: self.config.max_entities_in_context]

            for node in sorted_nodes:
                line = self._format_node(node, include_confidence)
                lines.append(line)
        else:
            lines.append("No entities extracted yet.")

        raw_context = "\n".join(lines) + "\n"

        return self._apply_truncation(
            raw_context,
            entity_count=len(nodes) if nodes else 0,
        )

    def _build_relationship_schema_hints(self, entity_types: Set[str]) -> List[str]:
        """Build relationship schema hints for given entity types.

        Args:
            entity_types: Set of entity types to include hints for.

        Returns:
            List of hint lines.
        """
        if self._relationship_schema_cache is None:
            self._relationship_schema_cache = self._compute_relationship_schema()

        hints: List[str] = []
        seen_patterns: Set[str] = set()

        for entity_type in sorted(entity_types):
            patterns = self._relationship_schema_cache.get(entity_type, [])
            for pattern in patterns:
                # Create a unique key for deduplication
                pattern_key = (
                    f"{pattern['source']}-{pattern['type']}-{pattern['target']}"
                )
                if pattern_key in seen_patterns:
                    continue
                seen_patterns.add(pattern_key)

                direction = pattern.get("direction", "outgoing")
                if direction == "outgoing":
                    hint = f"  ({pattern['source']}) -[:{pattern['type']}]-> ({pattern['target']})"
                else:
                    hint = f"  ({pattern['source']}) <-[:{pattern['type']}]- ({pattern['target']})"

                # Add cardinality hint if available
                cardinality = pattern.get("cardinality", "")
                if cardinality:
                    hint += f"  [{cardinality}]"

                hints.append(hint)

        return hints

    def _compute_relationship_schema(self) -> Dict[str, List[Dict[str, Any]]]:
        """Compute relationship schema from ontology.

        Returns:
            Dictionary mapping entity types to their relationship patterns.
        """
        schema: Dict[str, List[Dict[str, Any]]] = {}

        # Extract from entity definitions (relationships defined per entity)
        for entity_type, entity_def in self._entity_defs.items():
            if entity_type not in schema:
                schema[entity_type] = []

            relationships = entity_def.get("relationships", {})
            for rel_name, rel_def in relationships.items():
                if isinstance(rel_def, dict):
                    target_type = rel_def.get("target", "Unknown")
                    cardinality = rel_def.get("cardinality", "")

                    schema[entity_type].append(
                        {
                            "source": entity_type,
                            "type": rel_name,
                            "target": target_type,
                            "direction": "outgoing",
                            "cardinality": cardinality,
                        }
                    )

                    # Also add reverse mapping for target
                    if target_type not in schema:
                        schema[target_type] = []
                    schema[target_type].append(
                        {
                            "source": entity_type,
                            "type": rel_name,
                            "target": target_type,
                            "direction": "incoming",
                            "cardinality": cardinality,
                        }
                    )

        # Extract from standalone relationship definitions if present
        for rel_name, rel_def in self._relationship_defs.items():
            if not isinstance(rel_def, dict):
                continue

            source_type = rel_def.get("source", "")
            target_type = rel_def.get("target", "")
            cardinality = rel_def.get("cardinality", "")

            if source_type and target_type:
                if source_type not in schema:
                    schema[source_type] = []
                schema[source_type].append(
                    {
                        "source": source_type,
                        "type": rel_name,
                        "target": target_type,
                        "direction": "outgoing",
                        "cardinality": cardinality,
                    }
                )

                if target_type not in schema:
                    schema[target_type] = []
                schema[target_type].append(
                    {
                        "source": source_type,
                        "type": rel_name,
                        "target": target_type,
                        "direction": "incoming",
                        "cardinality": cardinality,
                    }
                )

        return schema

    def get_expected_entity_types_from_relationships(
        self, existing_nodes: List[BaseNode]
    ) -> Set[str]:
        """Get entity types that should exist based on relationship patterns.

        Analyzes existing nodes and their expected relationships to identify
        entity types that should be present but may be missing.

        Args:
            existing_nodes: Currently extracted nodes.

        Returns:
            Set of entity types that are expected based on relationships.
        """
        if self._relationship_schema_cache is None:
            self._relationship_schema_cache = self._compute_relationship_schema()

        expected_types: Set[str] = set()
        existing_types = {node.type for node in existing_nodes}

        for node in existing_nodes:
            patterns = self._relationship_schema_cache.get(node.type, [])
            for pattern in patterns:
                # For outgoing relationships, expect target type
                if pattern["direction"] == "outgoing":
                    target_type = pattern["target"]
                    if target_type not in existing_types:
                        expected_types.add(target_type)

        return expected_types

    def build_relationship_context(
        self,
        nodes: List[BaseNode],
        relationships: List[RelationshipInstance],
        include_orphans: bool = True,
        include_confidence: bool = True,
    ) -> ContextEnvelope:
        """Build relationship context with orphan highlighting.

        Args:
            nodes: List of extracted nodes.
            relationships: List of extracted relationships.
            include_orphans: Whether to highlight orphan nodes.
            include_confidence: Whether to include confidence scores.

        Returns:
            ContextEnvelope with the built context.
        """
        if not relationships and not nodes:
            return ContextEnvelope(
                text="No entities or relationships extracted yet.",
                truncated=False,
                raw_length=0,
            )

        node_map = {node.id: node for node in nodes}
        lines: List[str] = []

        # Sort relationships for consistency
        sorted_rels = sorted(
            relationships,
            key=lambda r: (r.source_type, r.type, r.target_type, r.source_id),
        )

        # Limit relationships
        if len(sorted_rels) > self.config.max_relationships_in_context:
            sorted_rels = sorted_rels[: self.config.max_relationships_in_context]

        # Build relationship entries
        for rel in sorted_rels:
            source_node = node_map.get(rel.source_id)
            target_node = node_map.get(rel.target_id)
            if not source_node or not target_node:
                continue

            line = self._format_relationship(
                source_node, rel, target_node, include_confidence
            )
            lines.append(line)

        # Add orphan nodes section if requested
        if include_orphans:
            connected_ids = {rel.source_id for rel in relationships} | {
                rel.target_id for rel in relationships
            }
            orphans = [n for n in nodes if n.id not in connected_ids]

            if orphans:
                lines.append("")
                lines.append("=== NODES WITHOUT RELATIONSHIPS (need connections) ===")
                for node in sorted(orphans, key=lambda n: (n.type, n.id)):
                    line = self._format_node(node, include_confidence)
                    lines.append(line)

        raw_context = "\n".join(lines)
        if lines:
            raw_context += "\n"

        return self._apply_truncation(
            raw_context,
            entity_count=len(nodes),
            relationship_count=len(sorted_rels),
        )

    def build_refinement_context(
        self,
        gaps: List[ExtractionGap],
        chunk_text: str,
        existing_nodes: List[BaseNode],
        focus_entities: Optional[List[str]] = None,
    ) -> str:
        """Build focused context for gap refinement.

        Args:
            gaps: List of extraction gaps to address.
            chunk_text: The source text chunk being re-examined.
            existing_nodes: Currently extracted nodes for reference.
            focus_entities: Specific entity IDs to focus on.

        Returns:
            Context string for refinement extraction.
        """
        lines: List[str] = []

        # Section 1: Gaps to address
        lines.append("=== EXTRACTION GAPS TO ADDRESS ===")
        for gap in gaps:
            lines.append(self._format_gap(gap))
        lines.append("")

        # Section 2: Relevant existing entities
        if focus_entities:
            relevant_nodes = [n for n in existing_nodes if n.id in focus_entities]
        else:
            # Get entities mentioned in gaps
            gap_entity_ids = {gap.entity_id for gap in gaps if gap.entity_id}
            gap_entity_types = {gap.entity_type for gap in gaps if gap.entity_type}
            relevant_nodes = [
                n
                for n in existing_nodes
                if n.id in gap_entity_ids or n.type in gap_entity_types
            ]

        if relevant_nodes:
            lines.append("=== RELEVANT EXISTING ENTITIES ===")
            for node in relevant_nodes:
                lines.append(self._format_node(node, include_confidence=True))
            lines.append("")

        # Section 3: Instructions based on gap types
        gap_types = {gap.gap_type for gap in gaps}
        instructions = self._get_refinement_instructions(gap_types)
        if instructions:
            lines.append("=== EXTRACTION INSTRUCTIONS ===")
            lines.extend(instructions)
            lines.append("")

        return "\n".join(lines)

    def build_gap_specific_context(
        self,
        gap: ExtractionGap,
        nodes: List[BaseNode],
        relationships: List[RelationshipInstance],
    ) -> str:
        """Build context specific to addressing a single gap.

        Args:
            gap: The specific gap to address.
            nodes: All extracted nodes.
            relationships: All extracted relationships.

        Returns:
            Context string focused on the specific gap.
        """
        lines: List[str] = []

        lines.append(f"=== ADDRESSING: {gap.description} ===")
        lines.append(f"Gap Type: {gap.gap_type.value}")
        lines.append(f"Severity: {gap.severity:.2f}")
        lines.append("")

        # Add gap-specific context
        if gap.gap_type == GapType.INCOMPLETE_ENTITY:
            node = next((n for n in nodes if n.id == gap.entity_id), None)
            if node:
                lines.append("Current Entity State:")
                lines.append(self._format_node(node, include_confidence=True))
                lines.append("")

                missing = gap.context.get("missing_required", [])
                if missing:
                    lines.append(f"Missing Required Properties: {', '.join(missing)}")

                # Add property hints from ontology
                entity_def = self._entity_defs.get(node.type, {})
                props_def = entity_def.get("properties", {})
                if props_def:
                    lines.append("")
                    lines.append("Property Definitions:")
                    for prop_name in missing:
                        prop_info = props_def.get(prop_name, {})
                        desc = prop_info.get("description", "No description")
                        lines.append(f"  - {prop_name}: {desc}")

        elif gap.gap_type == GapType.ORPHAN_NODE:
            node = next((n for n in nodes if n.id == gap.entity_id), None)
            if node:
                lines.append("Orphan Node:")
                lines.append(self._format_node(node, include_confidence=True))
                lines.append("")

                # Show possible relationship types from ontology
                entity_def = self._entity_defs.get(node.type, {})
                rels_def = entity_def.get("relationships", {})
                if rels_def:
                    lines.append("Possible Relationships:")
                    for rel_name, rel_def in rels_def.items():
                        target = rel_def.get("target", "Unknown")
                        lines.append(f"  - {rel_name} -> {target}")

        elif gap.gap_type == GapType.MISSING_RELATIONSHIP:
            source_node = next((n for n in nodes if n.id == gap.entity_id), None)
            if source_node:
                lines.append("Source Entity:")
                lines.append(self._format_node(source_node, include_confidence=True))
                lines.append("")

                rel_type = gap.context.get("relationship_type")
                target_type = gap.context.get("target_type")
                if rel_type and target_type:
                    lines.append(f"Expected Relationship: {rel_type}")
                    lines.append(f"Target Entity Type: {target_type}")

                    # Show potential targets
                    potential_targets = [n for n in nodes if n.type == target_type]
                    if potential_targets:
                        lines.append("")
                        lines.append("Potential Targets:")
                        for target in potential_targets[:5]:
                            lines.append(f"  - {self._format_node_brief(target)}")

        return "\n".join(lines)

    def _sort_nodes_for_context(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Sort nodes for consistent context ordering."""
        if self.config.prioritize_low_confidence:
            # Put low-confidence nodes first for visibility
            return sorted(
                nodes,
                key=lambda n: (
                    n.confidence_score or 1.0,  # Lower confidence first
                    n.type,
                    self._format_properties(n.properties),
                    n.id,
                ),
            )
        else:
            return sorted(
                nodes,
                key=lambda n: (
                    n.type,
                    self._format_properties(n.properties),
                    n.id,
                ),
            )

    def _format_node(self, node: BaseNode, include_confidence: bool = True) -> str:
        """Format a node for context string."""
        props_repr = self._format_properties(node.properties)
        base = f"Node Type: {node.type}, Id: {node.id}, Properties: {props_repr}"

        if include_confidence and self.config.include_confidence_scores:
            confidence = node.confidence_score or 1.0
            base += f" [confidence: {confidence:.2f}]"

        return base

    def _format_node_brief(self, node: BaseNode) -> str:
        """Format a node briefly for lists."""
        # Get a representative property value
        name_props = ["name", "title", "label", "id"]
        display_value = node.id
        for prop in name_props:
            if prop in node.properties and node.properties[prop]:
                display_value = str(node.properties[prop])
                break

        return f"{node.type}:{node.id} ({display_value})"

    def _format_relationship(
        self,
        source: BaseNode,
        rel: RelationshipInstance,
        target: BaseNode,
        include_confidence: bool = True,
    ) -> str:
        """Format a relationship for context string."""
        source_repr = self._format_properties(source.properties)
        target_repr = self._format_properties(target.properties)
        rel_props = self._format_properties(rel.properties)

        line = (
            f"({source.type}:{{'id': '{source.id}', 'properties': {source_repr}}})"
            f"-[:{rel.type}{{'properties': {rel_props}}}]->"
            f"({target.type}:{{'id': '{target.id}', 'properties': {target_repr}}})"
        )

        if include_confidence and self.config.include_confidence_scores:
            confidence = rel.confidence_score or 1.0
            line += f" [confidence: {confidence:.2f}]"

        return line

    def _format_gap(self, gap: ExtractionGap) -> str:
        """Format a gap for context string."""
        parts = [f"- [{gap.gap_type.value}]"]

        if gap.entity_type:
            parts.append(f"Entity Type: {gap.entity_type}")
        if gap.entity_id:
            parts.append(f"Entity ID: {gap.entity_id}")

        parts.append(f"Description: {gap.description}")
        parts.append(f"Severity: {gap.severity:.2f}")

        return " | ".join(parts)

    def _format_properties(self, properties: Optional[Dict[str, Any]]) -> str:
        """Format properties dictionary as JSON string."""
        if not properties:
            return "{}"
        return json.dumps(properties, sort_keys=True, default=str)

    def _build_validation_summary(self, result: ValidationResult) -> List[str]:
        """Build validation summary lines."""
        lines = [
            "=== VALIDATION SUMMARY ===",
            f"Overall Confidence: {result.overall_confidence:.2f}",
            f"Property Completeness: {result.property_completeness:.0%}",
            f"Total Gaps: {len(result.gaps)}",
            f"Orphan Nodes: {len(result.orphan_nodes)}",
        ]

        if result.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in result.recommendations[:3]:
                lines.append(f"  - {rec}")

        return lines

    def _get_refinement_instructions(self, gap_types: set) -> List[str]:
        """Get extraction instructions based on gap types."""
        instructions: List[str] = []

        if GapType.INCOMPLETE_ENTITY in gap_types:
            instructions.append(
                "- Look for missing property values for incomplete entities"
            )
            instructions.append(
                "- Pay attention to implicit information that may fill gaps"
            )

        if GapType.ORPHAN_NODE in gap_types:
            instructions.append("- Find relationships connecting isolated entities")
            instructions.append("- Look for implicit connections mentioned in the text")

        if GapType.MISSING_RELATIONSHIP in gap_types:
            instructions.append("- Focus on extracting the expected relationship types")
            instructions.append("- Check for alternative phrasings of relationships")

        if GapType.LOW_CONFIDENCE in gap_types:
            instructions.append(
                "- Verify uncertain extractions against the source text"
            )
            instructions.append("- Look for additional evidence to increase confidence")

        return instructions

    def _apply_truncation(
        self,
        raw_context: str,
        entity_count: int = 0,
        relationship_count: int = 0,
    ) -> ContextEnvelope:
        """Apply truncation to context if it exceeds the limit."""
        limit = self.config.max_context_chars

        if limit <= 0 or len(raw_context) <= limit:
            return ContextEnvelope(
                text=raw_context,
                truncated=False,
                raw_length=len(raw_context),
                entity_count=entity_count,
                relationship_count=relationship_count,
            )

        # Apply truncation based on strategy
        strategy = self.config.truncation_strategy
        sentinel = self.TRUNCATION_SENTINEL

        if strategy == "head":
            # Reserve 1 char for trailing newline
            truncated_text = raw_context[: limit - 1]
        elif strategy == "tail":
            truncated_text = raw_context[-(limit - 1) :]
        else:  # head_tail
            if limit <= len(sentinel) + 2:
                truncated_text = raw_context[:limit]
            else:
                head_len = (limit - len(sentinel)) // 2
                tail_len = limit - len(sentinel) - head_len
                head = raw_context[:head_len].rstrip("\n")
                tail = raw_context[-tail_len:].lstrip("\n") if tail_len > 0 else ""
                truncated_text = (
                    f"{head}\n{sentinel.strip()}\n{tail}"
                    if tail
                    else f"{head}\n{sentinel.strip()}"
                )
                if len(truncated_text) > limit:
                    truncated_text = truncated_text[:limit]

        logger.warning(
            "Context truncated",
            extra={
                "raw_length": len(raw_context),
                "max_chars": limit,
                "strategy": strategy,
            },
        )

        return ContextEnvelope(
            text=truncated_text.rstrip() + "\n",
            truncated=True,
            raw_length=len(raw_context),
            entity_count=entity_count,
            relationship_count=relationship_count,
        )
