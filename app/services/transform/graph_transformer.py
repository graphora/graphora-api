import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import List, Callable, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from app.config import settings
from app.services.transform.models import DocumentKnowledgeGraph
from app.services.transform.ontology_helper import OntologyParser
from app.services.transform.helpers import (
    transform_as_nodes,
    resolve_entity_group,
    merge_nodes,
    transform_as_relationships,
    prune_orphaned_nodes,
    deduplicate_entities_with_splink,
)
from app.services.llm.client import LLMClient
from app.services.transform.models import BaseNode, RelationshipInstance
from app.services.entity_ledger_service import entity_ledger_service
from app.utils.logger import logger
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass(frozen=True)
class ContextEnvelope:
    """Context string plus metadata about truncation."""

    text: str
    truncated: bool
    raw_length: int


@dataclass
class ChunkExtractionMetric:
    """Telemetry captured for each chunk/LLM extraction."""

    stage: str
    chunk_index: int
    chunk_chars: int
    context_chars: int
    raw_context_chars: int
    was_context_truncated: bool
    duration_seconds: float


_CONTEXT_TRUNCATION_SENTINEL = "\n...[truncated]...\n"


def _make_context_envelope(raw_context: str, *, stage: str) -> ContextEnvelope:
    """Apply deterministic truncation to context strings and emit metadata."""

    limit = getattr(settings, "MAX_CONTEXT_CHARS", 0) or 0
    if limit <= 0 or len(raw_context) <= limit:
        return ContextEnvelope(text=raw_context, truncated=False, raw_length=len(raw_context))

    # Retain head/tail portions with a sentinel to preserve ordering cues.
    sentinel = _CONTEXT_TRUNCATION_SENTINEL
    # Ensure we have room for the sentinel plus at least one character from the head and tail.
    if limit <= len(sentinel) + 2:
        truncated_text = raw_context[:limit]
    else:
        head_len = (limit - len(sentinel)) // 2
        tail_len = limit - len(sentinel) - head_len
        head = raw_context[:head_len].rstrip("\n")
        tail = raw_context[-tail_len:].lstrip("\n") if tail_len > 0 else ""
        truncated_text = f"{head}\n{sentinel.strip()}\n{tail}" if tail else f"{head}\n{sentinel.strip()}"
        if len(truncated_text) > limit:
            truncated_text = truncated_text[:limit]

    truncated = True
    truncated_text = truncated_text.rstrip() + "\n"
    truncated_text = truncated_text[:limit]
    assert len(truncated_text) <= limit

    logger.warning(
        "Context truncated for %s stage",
        stage,
        extra={
            "raw_length": len(raw_context),
            "max_chars": limit,
        },
    )
    return ContextEnvelope(text=truncated_text, truncated=truncated, raw_length=len(raw_context))


async def _timed_call(func: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
    """Execute an async callable measuring elapsed time."""

    start = perf_counter()
    result = await func(*args, **kwargs)
    duration = perf_counter() - start
    return result, duration


def _chunk_length(chunk: Any) -> int:
    """Approximate chunk size in characters for telemetry."""

    if isinstance(chunk, str):
        return len(chunk)
    try:  # pragma: no cover - defensive for unexpected chunk types
        return len(chunk)
    except TypeError:
        return 0


def _log_chunk_metrics(metrics: List[ChunkExtractionMetric]) -> None:
    """Log summary telemetry for chunk extraction stages."""

    if not metrics:
        return

    by_stage: Dict[str, List[ChunkExtractionMetric]] = {}
    for metric in metrics:
        by_stage.setdefault(metric.stage, []).append(metric)

    for stage, entries in by_stage.items():
        total_duration = sum(item.duration_seconds for item in entries)
        avg_duration = total_duration / len(entries)
        max_duration = max(item.duration_seconds for item in entries)
        max_context = max(item.context_chars for item in entries)
        max_raw_context = max(item.raw_context_chars for item in entries)
        truncated_count = sum(1 for item in entries if item.was_context_truncated)

        logger.info(
            "Chunk extraction telemetry",
            extra={
                "stage": stage,
                "calls": len(entries),
                "avg_duration": round(avg_duration, 3),
                "max_duration": round(max_duration, 3),
                "max_context_chars": max_context,
                "max_raw_context_chars": max_raw_context,
                "truncated_contexts": truncated_count,
            },
        )

        for item in entries:
            logger.debug(
                "Chunk extraction detail",
                extra={
                    "stage": stage,
                    "chunk_index": item.chunk_index,
                    "chunk_chars": item.chunk_chars,
                    "context_chars": item.context_chars,
                    "raw_context_chars": item.raw_context_chars,
                    "truncated": item.was_context_truncated,
                    "duration_seconds": round(item.duration_seconds, 3),
                },
            )


async def build_graph_from_chunks(
    ontology_parser: OntologyParser,
    chunks: List[str],
    transform_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    llm_client = LLMClient()
    return await _build_graph_from(
        ontology_parser,
        chunks,
        transform_id,
        llm_client.extract_nodes_from_chunk,
        llm_client.extract_relationships_from_chunk,
        progress_callback,
        user_id,
        document_usage_id,
    )


async def build_graph_from_pdfs(
    ontology_parser: OntologyParser,
    pdf_paths: List[str],
    transform_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    llm_client = LLMClient()
    return await _build_graph_from(
        ontology_parser,
        pdf_paths,
        transform_id,
        llm_client.extract_nodes_from_pdf,
        llm_client.extract_relationships_from_pdf,
        progress_callback,
        user_id,
        document_usage_id,
    )


async def _build_graph_from(
    ontology_parser: OntologyParser,
    chunks_or_pdf_paths: List[str],
    transform_id: str,
    node_extractor: Callable[[str, BaseModel, str], BaseModel],
    relationship_extractor: Callable[[str, BaseModel, str], BaseModel],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    nodes_only_ontology = ontology_parser.build_entities_only_model()
    nodes: List[BaseNode] = []
    metrics: List[ChunkExtractionMetric] = []
    context_envelope = ContextEnvelope(text="None", truncated=False, raw_length=len("None"))

    # Step 1: LLM-based entity extraction per chunk with deterministic context snapshots.
    for chunk_index, chunk in enumerate(chunks_or_pdf_paths):
        context_used = context_envelope
        nodes_only_kg, duration = await _timed_call(
            node_extractor,
            chunk,
            response_model=nodes_only_ontology,
            context=context_used.text,
            ontology_yaml=ontology_parser.ontology_yaml,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )

        base_nodes = transform_as_nodes(
            ontology_parser.parsed_ontology,
            nodes_only_kg,
            transform_id=transform_id,
        )
        for new_node in base_nodes:
            is_duplicate = any(
                _is_duplicate_node(existing_node, new_node) for existing_node in nodes
            )
            if not is_duplicate:
                nodes.append(new_node)

        metrics.append(
            ChunkExtractionMetric(
                stage="nodes",
                chunk_index=chunk_index,
                chunk_chars=_chunk_length(chunk),
                context_chars=len(context_used.text),
                raw_context_chars=context_used.raw_length,
                was_context_truncated=context_used.truncated,
                duration_seconds=duration,
            )
        )

        context_envelope = _build_nodes_context_envelope(nodes)

    if user_id:
        await entity_ledger_service.hydrate_nodes(user_id, nodes)

    # Step 2: Compare & merge entities, then deduplicate with Splink hints.
    nodes = await _compare_and_merge_nodes(
        nodes,
        user_id=user_id,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
    )
    nodes, _ = await deduplicate_entities_with_splink(
        nodes,
        None,
        parsed_ontology=ontology_parser.parsed_ontology,
        user_id=user_id,
    )
    logger.info(f"Nodes after comparison: {nodes}")

    # Step 3: Relationship inference with bounded concurrency and shared context snapshots.
    relationships_only_ontology = ontology_parser.build_relationships_only_model()
    relationships: List[RelationshipInstance] = []
    group_size = max(getattr(settings, "TRANSFORM_MAX_CONCURRENCY", 1), 1)
    total_chunks = len(chunks_or_pdf_paths)

    for group_start in range(0, total_chunks, group_size):
        group_indices = list(
            range(group_start, min(group_start + group_size, total_chunks))
        )
        rel_context_envelope = _build_relationships_context_envelope(
            nodes, relationships
        )
        context_text = rel_context_envelope.text

        # Batch requests to respect concurrency limits while keeping prompts
        # deterministic. Each batch shares the context built from relationships
        # accepted so far; post-processing deduplicates results across batches.
        tasks = [
            asyncio.create_task(
                _timed_call(
                    relationship_extractor,
                    chunks_or_pdf_paths[idx],
                    response_model=relationships_only_ontology,
                    context=context_text,
                    ontology_yaml=ontology_parser.ontology_yaml,
                    user_id=user_id,
                    transform_id=transform_id,
                    document_usage_id=document_usage_id,
                )
            )
            for idx in group_indices
        ]

        results = await asyncio.gather(*tasks)

        for idx, (relationships_only_kg, duration) in zip(group_indices, results):
            base_relationships = transform_as_relationships(
                ontology_parser.parsed_ontology, nodes, relationships_only_kg
            )
            for new_relationship in base_relationships:
                is_duplicate = any(
                    _is_duplicate_relationship(existing_relationship, new_relationship)
                    for existing_relationship in relationships
                )
                if not is_duplicate:
                    relationships.append(new_relationship)

            metrics.append(
                ChunkExtractionMetric(
                    stage="relationships",
                    chunk_index=idx,
                    chunk_chars=_chunk_length(chunks_or_pdf_paths[idx]),
                    context_chars=len(context_text),
                    raw_context_chars=rel_context_envelope.raw_length,
                    was_context_truncated=rel_context_envelope.truncated,
                    duration_seconds=duration,
                )
            )

    # Step 4: Compare & merge relationships, then run Splink dedup.
    relationships = _compare_and_merge_relationships(relationships)

    nodes, relationships = await deduplicate_entities_with_splink(
        entities=nodes,
        relationships=relationships,
        parsed_ontology=ontology_parser.parsed_ontology,
        user_id=user_id,
    )

    if user_id:
        await entity_ledger_service.record_nodes(user_id, nodes)

    # Step 5: Build final graph and prune orphans.
    kg = DocumentKnowledgeGraph(nodes=nodes, relationships=relationships)
    prune_orphaned_nodes(ontology_parser.parsed_ontology, kg)

    _log_chunk_metrics(metrics)
    return kg


async def _build_nodes_context(
    nodes: List[BaseNode],
) -> str:
    return _build_nodes_context_envelope(nodes).text


async def _build_relationships_context(
    nodes: List[BaseNode],
    relationships: List[RelationshipInstance],
) -> str:
    return _build_relationships_context_envelope(nodes, relationships).text


def _build_nodes_context_envelope(nodes: List[BaseNode]) -> ContextEnvelope:
    if not nodes:
        return ContextEnvelope(text="", truncated=False, raw_length=0)

    sorted_nodes = sorted(nodes, key=_node_context_sort_key)
    lines = []
    for node in sorted_nodes:
        properties_repr = _format_properties(node.properties)
        lines.append(
            f"Node Type: {node.type}, Id: {node.id}, Properties: {properties_repr}"
        )
    raw_context = "\n".join(lines) + "\n"
    return _make_context_envelope(raw_context, stage="nodes")


def _build_relationships_context_envelope(
    nodes: List[BaseNode],
    relationships: List[RelationshipInstance],
) -> ContextEnvelope:
    if not relationships and not nodes:
        return ContextEnvelope(text="", truncated=False, raw_length=0)

    node_map = {node.id: node for node in nodes}
    lines = []

    sorted_relationships = sorted(relationships, key=_relationship_context_sort_key)
    for relationship in sorted_relationships:
        source_node = node_map.get(relationship.source_id)
        target_node = node_map.get(relationship.target_id)
        if not source_node or not target_node:
            continue

        source_repr = _format_properties(source_node.properties)
        target_repr = _format_properties(target_node.properties)
        rel_props = _format_properties(relationship.properties)
        lines.append(
            f"({source_node.type}:{{'id': '{source_node.id}', 'properties': {source_repr}}})"
            f"-[:{relationship.type}{{'properties': {rel_props}}}]->"
            f"({target_node.type}:{{'id': '{target_node.id}', 'properties': {target_repr}}})"
        )

    nodes_in_relationships = {rel.source_id for rel in relationships} | {
        rel.target_id for rel in relationships
    }
    nodes_not_in_relationships = [
        node for node in nodes if node.id not in nodes_in_relationships
    ]

    if nodes_not_in_relationships:
        lines.append("These Nodes without any relationships:")
        for node in sorted(nodes_not_in_relationships, key=_node_context_sort_key):
            node_repr = _format_properties(node.properties)
            lines.append(
                f"({node.type}:{{'id': '{node.id}', 'properties': {node_repr}}})"
            )

    raw_context = "\n".join(lines)
    if lines:
        raw_context += "\n"
    return _make_context_envelope(raw_context, stage="relationships")


def _format_properties(properties: Optional[Dict[str, Any]]) -> str:
    if not properties:
        return "{}"
    return json.dumps(properties, sort_keys=True, default=str)


def _node_context_sort_key(node: BaseNode) -> Tuple[str, str, str]:
    return (
        node.type or "",
        _format_properties(node.properties),
        node.id or "",
    )


def _relationship_context_sort_key(
    relationship: RelationshipInstance,
) -> Tuple[str, str, str, str, str, str]:
    return (
        relationship.source_type or "",
        relationship.type or "",
        relationship.target_type or "",
        relationship.source_id or "",
        relationship.target_id or "",
        _format_properties(relationship.properties),
    )


async def _compare_and_merge_nodes(
    nodes: List[BaseNode],
    user_id: Optional[str] = None,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> List[BaseNode]:
    """Compare all nodes and resolve them using LLM."""
    if not nodes or len(nodes) <= 1:
        return nodes

    entity_groups = {}
    for node in nodes:
        entity_type = node.type
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        if node.id not in [n.id for n in entity_groups[entity_type]]:
            entity_groups[entity_type].append(node)
        else:
            # merge nodes with same id
            base_node = [n for n in entity_groups[entity_type] if n.id == node.id][0]
            entity_groups[entity_type].remove(base_node)
            base_node = merge_nodes(base_node, node)
            entity_groups[entity_type].append(base_node)

    final_nodes = []
    for entity_type, nodes in entity_groups.items():
        if len(nodes) <= 1:
            final_nodes.extend(nodes)
            continue

        # Perform entity resolution for this group
        resolved_groups = await resolve_entity_group(
            entity_type,
            nodes,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )
        # Process resolved groups
        for group in resolved_groups:
            if len(group) == 1:
                # Single node, no merging needed
                final_nodes.append(group[0])
                continue

            # Sort by confidence score to use highest confidence node as base
            sorted_nodes = sorted(
                group,
                key=lambda x: x.confidence_score if x.confidence_score else 0,
                reverse=True,
            )

            # Use highest confidence node as base and merge others into it
            base_node = sorted_nodes[0]
            for other_node in sorted_nodes[1:]:
                base_node = merge_nodes(base_node, other_node)

            final_nodes.append(base_node)
    return final_nodes


def _compare_and_merge_relationships(
    relationships: List[RelationshipInstance],
) -> List[RelationshipInstance]:
    """Compare all relationships and resolve them using LLM."""
    if not relationships or len(relationships) <= 1:
        return relationships

    relationship_groups = {}
    for relationship in relationships:
        relationship_uid = (
            f"{relationship.source_id}-{relationship.type}-{relationship.target_id}"
        )
        if relationship_uid not in relationship_groups:
            relationship_groups[relationship_uid] = relationship
        else:
            rel_props = relationship_groups[relationship_uid].properties
            relationship_groups[relationship_uid].properties = {
                **rel_props,
                **relationship.properties,
            }

    final_relationships = []
    for relationship_uid, relationship in relationship_groups.items():
        final_relationships.append(relationship)

    return final_relationships


def _is_duplicate_node(existing_node: BaseNode, new_node: BaseNode) -> bool:
    """Check if two nodes have the same type and properties (excluding ID)"""
    if existing_node.type != new_node.type:
        return False

    # Compare properties excluding 'id'
    existing_props = {k: v for k, v in existing_node.properties.items() if k != "id"}
    new_props = {k: v for k, v in new_node.properties.items() if k != "id"}
    return existing_props == new_props


def _is_duplicate_relationship(
    existing_relationship: RelationshipInstance, new_relationship: RelationshipInstance
) -> bool:
    """Check if two relationships have the same source, type, and target"""
    return (
        existing_relationship.source_id == new_relationship.source_id
        and existing_relationship.type == new_relationship.type
        and existing_relationship.target_id == new_relationship.target_id
    )
