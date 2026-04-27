"""Multi-pass extraction coordinator for validation-driven refinement."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any

from graphora_server.services.transform.models import BaseNode, RelationshipInstance
from graphora_server.services.transform.ontology_helper import OntologyParser
from graphora_server.services.transform.helpers import (
    transform_as_nodes,
    transform_as_relationships,
    merge_nodes,
)
from graphora_server.services.llm.client import LLMClient
from .models import (
    ExtractionGap,
    GapType,
    RefinementResult,
)
from .config import MultiPassConfig
from .prompt_versions import get_prompt_version
from .validator import ExtractionValidator
from .context_builder import EnhancedContextBuilder

logger = logging.getLogger(__name__)


def _backfill_validator_score(
    nodes: List[BaseNode],
    relationships: List[RelationshipInstance],
    score: Optional[float],
) -> None:
    """B0-prov-extend: stamp pass-level validator_score on extracted facts.

    The validator computes one ``overall_confidence`` per pass — not
    per-node. Stamping the same value onto every node/edge that
    survived the pass gives users a coarse "what was the validator
    quality at the pass that produced this fact" anchor, which is
    enough for the Evidence-tab citation surface. Per-node validator
    scores would require Decision Log machinery (Slice 2).

    Setdefault-style — only writes when validator_score is currently
    None on both the NodeProvenance object and the property bag.
    Refinement-pass overrides are deliberately NOT applied here; a
    fact is anchored to the validation that first accepted it.
    """
    if score is None:
        return
    for fact in [*nodes, *relationships]:
        prov = getattr(fact, "provenance", None)
        if prov is not None and prov.validator_score is None:
            prov.validator_score = score
        if "validator_score" not in fact.properties:
            fact.properties["validator_score"] = score


class MultiPassExtractor:
    """Orchestrates multi-pass extraction with validation-driven refinement.

    This extractor performs an initial extraction pass, validates the results
    against the ontology, and then performs targeted refinement passes to
    address identified gaps and improve extraction quality.
    """

    def __init__(
        self,
        ontology_parser: OntologyParser,
        llm_client: Optional[LLMClient] = None,
        config: Optional[MultiPassConfig] = None,
    ) -> None:
        """Initialize multi-pass extractor.

        Args:
            ontology_parser: Parser for the target ontology.
            llm_client: LLM client for extraction. Creates new one if not provided.
            config: Multi-pass extraction configuration.
        """
        self.ontology_parser = ontology_parser
        self.llm_client = llm_client or LLMClient()
        self.config = config or MultiPassConfig()

        self.validator = ExtractionValidator(
            ontology_parser.parsed_ontology,
            self.config.validation_config,
        )
        self.context_builder = EnhancedContextBuilder(
            ontology_parser.parsed_ontology,
            self.config.context_config,
        )

    async def extract(
        self,
        chunks: List[str],
        transform_id: str,
        user_id: Optional[str] = None,
        max_passes: Optional[int] = None,
        progress_callback: Optional[Any] = None,
        chunk_metadatas: Optional[List[Any]] = None,
        extractor_model: Optional[str] = None,
    ) -> Tuple[List[BaseNode], List[RelationshipInstance]]:
        """Multi-pass extraction with validation-driven refinement.

        Pass 1: Initial extraction
        Validation: Identify gaps and low-confidence extractions
        Pass 2+: Targeted re-extraction for gaps

        Args:
            chunks: List of text chunks to extract from.
            transform_id: Unique identifier for this transform.
            user_id: User ID for LLM credentials.
            max_passes: Override max passes from config.
            progress_callback: Optional callback for progress updates.

        Returns:
            Tuple of (nodes, relationships) after all extraction passes.
        """
        max_passes = max_passes or self.config.max_passes

        logger.info(
            "Starting multi-pass extraction",
            extra={
                "transform_id": transform_id,
                "chunk_count": len(chunks),
                "max_passes": max_passes,
            },
        )

        # Pass 1: Initial extraction
        nodes, relationships = await self._initial_extraction_pass(
            chunks,
            transform_id,
            user_id,
            progress_callback,
            chunk_metadatas=chunk_metadatas,
            extractor_model=extractor_model,
        )

        logger.info(
            "Initial extraction complete",
            extra={
                "transform_id": transform_id,
                "node_count": len(nodes),
                "relationship_count": len(relationships),
            },
        )

        # Validation and refinement passes
        for pass_num in range(2, max_passes + 1):
            # Validate current extraction
            validation_result = self.validator.validate(nodes, relationships)

            # B0-prov-extend: stamp the pass-level validator score
            # onto every node/edge that came out of the validated
            # pass. setdefault-style — only writes when absent so a
            # later refinement pass with a higher score doesn't
            # silently override the earlier one.
            _backfill_validator_score(
                nodes,
                relationships,
                validation_result.overall_confidence,
            )

            if not validation_result.needs_refinement():
                logger.info(
                    "Extraction valid, no refinement needed",
                    extra={
                        "transform_id": transform_id,
                        "pass_number": pass_num,
                        "confidence": validation_result.overall_confidence,
                    },
                )
                break

            # Get gaps above threshold
            high_severity_gaps = validation_result.get_high_severity_gaps(
                self.config.gap_severity_threshold
            )

            if not high_severity_gaps:
                logger.info(
                    "No high-severity gaps to address",
                    extra={
                        "transform_id": transform_id,
                        "total_gaps": len(validation_result.gaps),
                        "threshold": self.config.gap_severity_threshold,
                    },
                )
                break

            # Limit gaps to batch size
            gaps_to_address = high_severity_gaps[: self.config.refinement_batch_size]

            logger.info(
                "Starting refinement pass",
                extra={
                    "transform_id": transform_id,
                    "pass_number": pass_num,
                    "gaps_to_address": len(gaps_to_address),
                },
            )

            # Perform refinement pass
            new_nodes, new_relationships, refinement_result = (
                await self._refinement_pass(
                    chunks,
                    nodes,
                    relationships,
                    gaps_to_address,
                    transform_id,
                    user_id,
                    pass_num,
                    chunk_metadatas=chunk_metadatas,
                    extractor_model=extractor_model,
                )
            )

            # Check if refinement made improvements
            if not refinement_result.is_improvement():
                logger.info(
                    "Refinement pass made no improvements, stopping",
                    extra={
                        "transform_id": transform_id,
                        "pass_number": pass_num,
                    },
                )
                break

            # Merge results
            nodes = self._merge_nodes(nodes, new_nodes)
            relationships = self._merge_relationships(relationships, new_relationships)

            logger.info(
                "Refinement pass complete",
                extra={
                    "transform_id": transform_id,
                    "pass_number": pass_num,
                    "new_nodes": refinement_result.new_nodes_count,
                    "updated_nodes": refinement_result.updated_nodes_count,
                    "new_relationships": refinement_result.new_relationships_count,
                    "total_nodes": len(nodes),
                    "total_relationships": len(relationships),
                },
            )

        # Final validator pass + back-fill so any refinement-pass
        # facts created in the last iteration (which the in-loop
        # back-fill couldn't reach) still carry validator_score.
        # Cheap — the validator is in-process, no LLM call.
        # setdefault semantics in _backfill_validator_score mean
        # already-scored facts are left alone.
        final_validation = self.validator.validate(nodes, relationships)
        _backfill_validator_score(
            nodes, relationships, final_validation.overall_confidence
        )

        return nodes, relationships

    async def _initial_extraction_pass(
        self,
        chunks: List[str],
        transform_id: str,
        user_id: Optional[str],
        progress_callback: Optional[Any] = None,
        chunk_metadatas: Optional[List[Any]] = None,
        extractor_model: Optional[str] = None,
    ) -> Tuple[List[BaseNode], List[RelationshipInstance]]:
        """Perform initial extraction pass with relationship-aware entity context.

        This pass uses enhanced context building that includes:
        - Relationship schema hints during entity extraction
        - Expected entity types based on relationship patterns
        - Chain-of-thought guidance in prompts

        Args:
            chunks: Text chunks to extract from.
            transform_id: Transform identifier.
            user_id: User ID for LLM.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (nodes, relationships) from initial extraction.
        """
        nodes_only_ontology = self.ontology_parser.build_entities_only_model()
        nodes: List[BaseNode] = []

        # Build initial relationship-aware context (includes schema hints)
        context_envelope = self.context_builder.build_relationship_aware_entity_context(
            nodes, include_confidence=True
        )
        context_text = context_envelope.text

        logger.debug(
            "Starting entity extraction with relationship-aware context",
            extra={
                "transform_id": transform_id,
                "chunk_count": len(chunks),
                "initial_context_length": len(context_text),
            },
        )

        # Extract entities from each chunk
        for chunk_index, chunk in enumerate(chunks):
            # A1-prov: pull the matching ChunkMetadata for this chunk;
            # transform_as_nodes uses it to stamp source-span properties
            # on each emitted node. None when the caller didn't provide
            # metadata — extraction still succeeds, just without
            # provenance enrichment.
            cm = (
                chunk_metadatas[chunk_index]
                if chunk_metadatas and chunk_index < len(chunk_metadatas)
                else None
            )

            nodes_only_kg = await self.llm_client.extract_nodes_from_chunk(
                chunk,
                response_model=nodes_only_ontology,
                context=context_text,
                ontology_yaml=self.ontology_parser.ontology_yaml,
                user_id=user_id,
                transform_id=transform_id,
            )

            base_nodes = transform_as_nodes(
                self.ontology_parser.parsed_ontology,
                nodes_only_kg,
                transform_id=transform_id,
                chunk_metadata=cm,
                chunk_text=chunk,
                extractor_model=extractor_model,
                prompt_version=get_prompt_version("ExtractNodesFromChunk"),
            )

            # Add chunk index to provenance
            for node in base_nodes:
                if node.provenance:
                    node.provenance.chunk_ids.append(str(chunk_index))

            # Deduplicate and add new nodes
            for new_node in base_nodes:
                existing = next(
                    (n for n in nodes if self._is_same_node(n, new_node)), None
                )
                if existing:
                    # Merge into existing
                    idx = nodes.index(existing)
                    nodes[idx] = merge_nodes(existing, new_node)
                else:
                    nodes.append(new_node)

            # Update context with relationship hints for next chunk
            context_envelope = (
                self.context_builder.build_relationship_aware_entity_context(
                    nodes, include_confidence=True
                )
            )
            context_text = context_envelope.text

            if progress_callback:
                progress_callback(chunk_index + 1, len(chunks) * 2)

        # Check for expected entity types that may be missing
        expected_types = (
            self.context_builder.get_expected_entity_types_from_relationships(nodes)
        )
        existing_types = {node.type for node in nodes}
        missing_types = expected_types - existing_types

        if missing_types:
            logger.info(
                "Entity types expected from relationships but not extracted",
                extra={
                    "transform_id": transform_id,
                    "missing_types": list(missing_types),
                    "existing_types": list(existing_types),
                },
            )

        # Extract relationships
        relationships_only_ontology = (
            self.ontology_parser.build_relationships_only_model()
        )
        relationships: List[RelationshipInstance] = []

        rel_context = self.context_builder.build_relationship_context(
            nodes, relationships, include_orphans=True, include_confidence=True
        )

        for chunk_index, chunk in enumerate(chunks):
            cm = (
                chunk_metadatas[chunk_index]
                if chunk_metadatas and chunk_index < len(chunk_metadatas)
                else None
            )

            relationships_only_kg = (
                await self.llm_client.extract_relationships_from_chunk(
                    chunk,
                    response_model=relationships_only_ontology,
                    context=rel_context.text,
                    ontology_yaml=self.ontology_parser.ontology_yaml,
                    user_id=user_id,
                    transform_id=transform_id,
                )
            )

            base_relationships = transform_as_relationships(
                self.ontology_parser.parsed_ontology,
                nodes,
                relationships_only_kg,
                chunk_metadata=cm,
                chunk_text=chunk,
                extractor_model=extractor_model,
                prompt_version=get_prompt_version("ExtractRelationshipsFromChunk"),
            )

            # Deduplicate relationships
            for new_rel in base_relationships:
                is_dup = any(
                    self._is_same_relationship(r, new_rel) for r in relationships
                )
                if not is_dup:
                    relationships.append(new_rel)

            # Update context with orphan highlighting
            rel_context = self.context_builder.build_relationship_context(
                nodes, relationships, include_orphans=True, include_confidence=True
            )

            if progress_callback:
                progress_callback(len(chunks) + chunk_index + 1, len(chunks) * 2)

        return nodes, relationships

    async def _refinement_pass(
        self,
        chunks: List[str],
        nodes: List[BaseNode],
        relationships: List[RelationshipInstance],
        gaps: List[ExtractionGap],
        transform_id: str,
        user_id: Optional[str],
        pass_number: int,
        chunk_metadatas: Optional[List[Any]] = None,
        extractor_model: Optional[str] = None,
    ) -> Tuple[List[BaseNode], List[RelationshipInstance], RefinementResult]:
        """Perform targeted refinement for identified gaps.

        Args:
            chunks: Original text chunks.
            nodes: Currently extracted nodes.
            relationships: Currently extracted relationships.
            gaps: Gaps to address in this pass.
            transform_id: Transform identifier.
            user_id: User ID for LLM.
            pass_number: Current refinement pass number.

        Returns:
            Tuple of (new_nodes, new_relationships, refinement_result).
        """
        result = RefinementResult(pass_number=pass_number)
        new_nodes: List[BaseNode] = []
        new_relationships: List[RelationshipInstance] = []

        # Group gaps by chunk for efficient processing
        gaps_by_chunk = self._group_gaps_by_chunk(gaps, chunks)

        # Process gaps
        if self.config.enable_parallel_refinement:
            tasks = [
                self._extract_for_gaps(
                    chunk_gaps,
                    chunks,
                    nodes,
                    relationships,
                    transform_id,
                    user_id,
                    chunk_metadatas=chunk_metadatas,
                    extractor_model=extractor_model,
                )
                for chunk_gaps in gaps_by_chunk.values()
            ]
            results = await asyncio.gather(*tasks)

            for chunk_nodes, chunk_rels in results:
                new_nodes.extend(chunk_nodes)
                new_relationships.extend(chunk_rels)
        else:
            for chunk_gaps in gaps_by_chunk.values():
                chunk_nodes, chunk_rels = await self._extract_for_gaps(
                    chunk_gaps,
                    chunks,
                    nodes,
                    relationships,
                    transform_id,
                    user_id,
                    chunk_metadatas=chunk_metadatas,
                    extractor_model=extractor_model,
                )
                new_nodes.extend(chunk_nodes)
                new_relationships.extend(chunk_rels)

        # Count improvements
        result.gaps_addressed = len(gaps)
        result.new_nodes_count = len(new_nodes)
        result.new_relationships_count = len(new_relationships)

        return new_nodes, new_relationships, result

    async def _extract_for_gaps(
        self,
        gaps: List[ExtractionGap],
        chunks: List[str],
        existing_nodes: List[BaseNode],
        existing_relationships: List[RelationshipInstance],
        transform_id: str,
        user_id: Optional[str],
        chunk_metadatas: Optional[List[Any]] = None,
        extractor_model: Optional[str] = None,
    ) -> Tuple[List[BaseNode], List[RelationshipInstance]]:
        """Extract entities/relationships for specific gaps.

        Args:
            gaps: Gaps to address.
            chunks: Original text chunks.
            existing_nodes: Currently extracted nodes.
            existing_relationships: Currently extracted relationships.
            transform_id: Transform identifier.
            user_id: User ID for LLM.

        Returns:
            Tuple of (new_nodes, new_relationships).
        """
        new_nodes: List[BaseNode] = []
        new_relationships: List[RelationshipInstance] = []

        # Determine which chunks to re-examine
        chunk_indices = set()
        for gap in gaps:
            chunk_indices.update(gap.chunk_indices)

        # If no specific chunks, examine all
        if not chunk_indices:
            chunk_indices = set(range(len(chunks)))

        for chunk_idx in chunk_indices:
            if chunk_idx >= len(chunks):
                continue

            chunk_text = chunks[chunk_idx]

            # A1-prov: pull the matching ChunkMetadata for the chunk
            # being re-examined. Refinement-pass nodes/edges need the
            # same source-span properties as initial-pass ones — without
            # this gap, any node introduced during refinement would
            # silently fall out of the Evidence contract.
            cm = (
                chunk_metadatas[chunk_idx]
                if chunk_metadatas and chunk_idx < len(chunk_metadatas)
                else None
            )

            # Build refinement context
            context = self.context_builder.build_refinement_context(
                gaps=gaps,
                chunk_text=chunk_text,
                existing_nodes=existing_nodes,
            )

            # Determine what to extract based on gap types
            gap_types = {gap.gap_type for gap in gaps}

            if (
                GapType.INCOMPLETE_ENTITY in gap_types
                or GapType.LOW_CONFIDENCE in gap_types
            ):
                # Re-extract entities with focused context
                nodes_only_ontology = self.ontology_parser.build_entities_only_model()

                try:
                    nodes_kg = await self.llm_client.extract_nodes_from_chunk(
                        chunk_text,
                        response_model=nodes_only_ontology,
                        context=context,
                        ontology_yaml=self.ontology_parser.ontology_yaml,
                        user_id=user_id,
                        transform_id=transform_id,
                    )

                    extracted_nodes = transform_as_nodes(
                        self.ontology_parser.parsed_ontology,
                        nodes_kg,
                        transform_id=transform_id,
                        chunk_metadata=cm,
                        chunk_text=chunk_text,
                        extractor_model=extractor_model,
                        prompt_version=get_prompt_version("ExtractNodesFromChunk"),
                    )
                    new_nodes.extend(extracted_nodes)
                except Exception as e:
                    logger.warning(
                        "Entity refinement failed for chunk",
                        extra={
                            "transform_id": transform_id,
                            "chunk_index": chunk_idx,
                            "error": str(e),
                        },
                    )

            if (
                GapType.ORPHAN_NODE in gap_types
                or GapType.MISSING_RELATIONSHIP in gap_types
            ):
                # Re-extract relationships with focused context
                rel_context = self.context_builder.build_relationship_context(
                    existing_nodes,
                    existing_relationships,
                    include_orphans=True,
                    include_confidence=True,
                )

                relationships_only_ontology = (
                    self.ontology_parser.build_relationships_only_model()
                )

                try:
                    rels_kg = await self.llm_client.extract_relationships_from_chunk(
                        chunk_text,
                        response_model=relationships_only_ontology,
                        context=rel_context.text,
                        ontology_yaml=self.ontology_parser.ontology_yaml,
                        user_id=user_id,
                        transform_id=transform_id,
                    )

                    extracted_rels = transform_as_relationships(
                        self.ontology_parser.parsed_ontology,
                        existing_nodes + new_nodes,
                        rels_kg,
                        chunk_metadata=cm,
                        chunk_text=chunk_text,
                        extractor_model=extractor_model,
                        prompt_version=get_prompt_version(
                            "ExtractRelationshipsFromChunk"
                        ),
                    )
                    new_relationships.extend(extracted_rels)
                except Exception as e:
                    logger.warning(
                        "Relationship refinement failed for chunk",
                        extra={
                            "transform_id": transform_id,
                            "chunk_index": chunk_idx,
                            "error": str(e),
                        },
                    )

        return new_nodes, new_relationships

    def _group_gaps_by_chunk(
        self, gaps: List[ExtractionGap], chunks: List[str]
    ) -> Dict[int, List[ExtractionGap]]:
        """Group gaps by their associated chunk indices."""
        gaps_by_chunk: Dict[int, List[ExtractionGap]] = {}

        for gap in gaps:
            if gap.chunk_indices:
                for idx in gap.chunk_indices:
                    if idx not in gaps_by_chunk:
                        gaps_by_chunk[idx] = []
                    gaps_by_chunk[idx].append(gap)
            else:
                # No specific chunk - add to all chunks
                for idx in range(len(chunks)):
                    if idx not in gaps_by_chunk:
                        gaps_by_chunk[idx] = []
                    gaps_by_chunk[idx].append(gap)

        return gaps_by_chunk

    def _merge_nodes(
        self, existing: List[BaseNode], new_nodes: List[BaseNode]
    ) -> List[BaseNode]:
        """Merge new nodes into existing list."""
        result = list(existing)

        for new_node in new_nodes:
            existing_node = next(
                (n for n in result if self._is_same_node(n, new_node)), None
            )
            if existing_node:
                # Update existing node with new information
                idx = result.index(existing_node)
                result[idx] = merge_nodes(existing_node, new_node)
            else:
                result.append(new_node)

        return result

    def _merge_relationships(
        self,
        existing: List[RelationshipInstance],
        new_rels: List[RelationshipInstance],
    ) -> List[RelationshipInstance]:
        """Merge new relationships into existing list."""
        result = list(existing)

        for new_rel in new_rels:
            is_dup = any(self._is_same_relationship(r, new_rel) for r in result)
            if not is_dup:
                result.append(new_rel)

        return result

    def _is_same_node(self, node1: BaseNode, node2: BaseNode) -> bool:
        """Check if two nodes represent the same entity."""
        if node1.type != node2.type:
            return False

        # Check by ID
        if node1.id == node2.id:
            return True

        # Check by canonical ID
        if node1.canonical_id and node1.canonical_id == node2.canonical_id:
            return True

        # Check by properties (excluding ID)
        props1 = {k: v for k, v in node1.properties.items() if k != "id"}
        props2 = {k: v for k, v in node2.properties.items() if k != "id"}
        return props1 == props2

    def _is_same_relationship(
        self, rel1: RelationshipInstance, rel2: RelationshipInstance
    ) -> bool:
        """Check if two relationships are the same."""
        return (
            rel1.source_id == rel2.source_id
            and rel1.type == rel2.type
            and rel1.target_id == rel2.target_id
        )
