import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Callable, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from graphora_server.config import settings
from graphora_server.services.transform.models import DocumentKnowledgeGraph
from graphora_server.services.transform.ontology_helper import OntologyParser
from graphora_server.services.transform.helpers import (
    transform_as_nodes,
    resolve_entity_group,
    merge_nodes,
    transform_as_relationships,
    prune_orphaned_nodes,
    deduplicate_entities_with_splink,
)
from graphora_server.services.llm.client import LLMClient
from graphora_server.services.transform.models import BaseNode, RelationshipInstance
from graphora_server.services.entity_ledger_service import entity_ledger_service
from graphora_server.services.extraction.prompt_versions import (
    get_prompt_version as _resolve_prompt_version,
)
from graphora_server.utils.logger import logger
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
        return ContextEnvelope(
            text=raw_context, truncated=False, raw_length=len(raw_context)
        )

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
        truncated_text = (
            f"{head}\n{sentinel.strip()}\n{tail}"
            if tail
            else f"{head}\n{sentinel.strip()}"
        )
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
    return ContextEnvelope(
        text=truncated_text, truncated=truncated, raw_length=len(raw_context)
    )


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
    enable_multi_pass: bool = False,
    max_passes: int = 2,
    chunk_metadatas: Optional[List[Any]] = None,
    extractor_model: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    """Build knowledge graph from text chunks.

    Args:
        ontology_parser: Parser for the target ontology.
        chunks: List of text chunks to extract from.
        transform_id: Unique identifier for this transform.
        progress_callback: Optional callback for progress updates.
        user_id: User ID for LLM credentials.
        document_usage_id: Document usage tracking ID.
        enable_multi_pass: Whether to use multi-pass extraction with validation.
        max_passes: Maximum extraction passes when multi-pass is enabled.
        chunk_metadatas: Optional per-chunk metadata. Same length and
            order as ``chunks``. When provided, A1-prov source-span
            properties (source_chunk_id, source_text, document_name,
            page_number, chunk_offset, extraction_confidence) are
            stamped on every emitted node and edge.

    Returns:
        DocumentKnowledgeGraph with extracted nodes and relationships.
    """
    llm_client = LLMClient()

    # Use multi-pass extraction if enabled
    if enable_multi_pass:
        return await _build_graph_with_multi_pass(
            ontology_parser,
            chunks,
            transform_id,
            llm_client,
            progress_callback,
            user_id,
            document_usage_id,
            max_passes,
            chunk_metadatas=chunk_metadatas,
            extractor_model=extractor_model,
        )

    # Default single-pass extraction
    return await _build_graph_from(
        ontology_parser,
        chunks,
        transform_id,
        llm_client.extract_nodes_from_chunk,
        llm_client.extract_relationships_from_chunk,
        progress_callback,
        user_id,
        document_usage_id,
        chunk_metadatas=chunk_metadatas,
        extractor_model=extractor_model,
        node_baml_function="ExtractNodesFromChunk",
        rel_baml_function="ExtractRelationshipsFromChunk",
    )


async def build_graph_from_pdfs(
    ontology_parser: OntologyParser,
    pdf_paths: List[str],
    transform_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    chunk_metadatas: Optional[List[Any]] = None,
    extractor_model: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    """Build a graph by sending each PDF split file to Gemini's
    multimodal API. Caller may pass per-split ``chunk_metadatas``;
    when present, A1-prov source-span properties (document_name,
    source_chunk_id, page_number, extraction_confidence) are stamped
    on emitted nodes/edges. ``source_text`` is intentionally NOT
    set on this path — Gemini sees the binary, our pipeline does not
    have the text to embed.
    """
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
        chunk_metadatas=chunk_metadatas,
        treat_chunks_as_text=False,
        extractor_model=extractor_model,
        node_baml_function="ExtractNodesFromPdf",
        rel_baml_function="ExtractRelationshipsFromPdf",
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
    chunk_metadatas: Optional[List[Any]] = None,
    treat_chunks_as_text: bool = True,
    extractor_model: Optional[str] = None,
    node_baml_function: Optional[str] = None,
    rel_baml_function: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    nodes_only_ontology = ontology_parser.build_entities_only_model()
    nodes: List[BaseNode] = []
    metrics: List[ChunkExtractionMetric] = []
    context_envelope = ContextEnvelope(
        text="None", truncated=False, raw_length=len("None")
    )

    # Step 1: LLM-based entity extraction per chunk with deterministic context snapshots.
    for chunk_index, chunk in enumerate(chunks_or_pdf_paths):
        context_used = context_envelope
        # A1-prov: per-chunk metadata for source-span stamping. Pulled
        # by index when the caller provided it; None when the caller
        # didn't (older callsites or partial wiring).
        cm = (
            chunk_metadatas[chunk_index]
            if chunk_metadatas and chunk_index < len(chunk_metadatas)
            else None
        )
        # source_text is the chunk's literal text; only the text-chunk
        # path (text/markdown/docx → chunked) supplies it. The PDF-
        # binary path's "chunk" is a filesystem path, so leave
        # source_text unset there — Gemini sees the binary, our
        # pipeline does not have text to embed.
        chunk_text_for_props = (
            chunk if treat_chunks_as_text and isinstance(chunk, str) else None
        )

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
            chunk_metadata=cm,
            chunk_text=chunk_text_for_props,
            extractor_model=extractor_model,
            prompt_version=(
                _resolve_prompt_version(node_baml_function)
                if node_baml_function
                else None
            ),
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
            cm = (
                chunk_metadatas[idx]
                if chunk_metadatas and idx < len(chunk_metadatas)
                else None
            )
            chunk_text_for_props = (
                chunks_or_pdf_paths[idx]
                if treat_chunks_as_text and isinstance(chunks_or_pdf_paths[idx], str)
                else None
            )
            base_relationships = transform_as_relationships(
                ontology_parser.parsed_ontology,
                nodes,
                relationships_only_kg,
                chunk_metadata=cm,
                chunk_text=chunk_text_for_props,
                extractor_model=extractor_model,
                prompt_version=(
                    _resolve_prompt_version(rel_baml_function)
                    if rel_baml_function
                    else None
                ),
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


def _block_keys_for_node(node: BaseNode) -> List[str]:
    """Generate blocking signatures used by ``_compare_and_merge_nodes``
    to bucket candidate match groups before the LLM resolution step.

    Each returned key is type-prefixed so different entity types
    never share blocks. A node may produce multiple keys; nodes
    that share at least one key end up in the same candidate
    group. This is the B2-er slice 1 blocking stage — simple,
    property-based, no Splink/LSH dependency. Slice 2+ can layer
    embedding-based blocking on top.

    Block keys (in order of strength):
      1. canonical_key (when present) — already computed by
         _build_canonical_properties from canonicalization-marked
         properties. Strong "these might be the same entity" signal.
      2. First 3 chars of normalized name/title (lowercased,
         alphanumerics only).
      3. Type-only fallback — preserves the pre-blocking behaviour
         (whole-type group goes to the LLM) for nodes without
         either signature, so blocking can't reduce recall below
         the old code on poor-quality inputs.
    """
    keys: List[str] = []
    type_prefix = node.type or "_"

    if node.canonical_key:
        keys.append(f"{type_prefix}|canon:{node.canonical_key}")

    name_value = None
    if node.properties:
        name_value = node.properties.get("name") or node.properties.get("title")
    if isinstance(name_value, str) and name_value:
        normalized = "".join(c.lower() for c in name_value if c.isalnum())[:3]
        if normalized:
            keys.append(f"{type_prefix}|name3:{normalized}")

    if not keys:
        keys.append(f"{type_prefix}|_all")

    return keys


def _candidate_groups_for_resolution(
    nodes: List[BaseNode],
    max_block_size: int = 50,
) -> Iterable[List[BaseNode]]:
    """Bucket nodes into candidate groups for LLM-based ER.

    Each node is assigned one or more block keys via
    ``_block_keys_for_node``; nodes sharing any key form a
    candidate group. Singletons are not yielded (no resolution
    work to do). Blocks larger than ``max_block_size`` are
    broken into chunks so the LLM call stays bounded — same node
    appearing in multiple chunks is fine because the resolver
    works on each candidate group independently.

    Each node appears in at most one yielded group per call:
    once it's assigned to a group, subsequent blocks that would
    have included it skip it. This keeps the ER work O(n*k)
    rather than O(n²) when nodes share several block keys.

    Caller is responsible for collecting nodes that never appear
    in any yielded group (singletons, blocks of one) and passing
    them through unchanged. ``_compare_and_merge_nodes`` does
    that bookkeeping below.
    """
    blocks: Dict[str, List[BaseNode]] = {}
    for node in nodes:
        for key in _block_keys_for_node(node):
            blocks.setdefault(key, []).append(node)

    seen_ids: set = set()
    # Sort blocks by size descending so the largest blocks fire
    # first — gives us the most "compression" early on. Within a
    # block, sort by id for deterministic chunking.
    for _key, members in sorted(blocks.items(), key=lambda kv: -len(kv[1])):
        unseen = [n for n in members if n.id not in seen_ids]
        if len(unseen) <= 1:
            continue
        # Chunk oversize blocks. The last chunk may be smaller
        # than max_block_size; that's fine.
        for start in range(0, len(unseen), max_block_size):
            chunk = unseen[start : start + max_block_size]
            if len(chunk) <= 1:
                continue
            for n in chunk:
                seen_ids.add(n.id)
            yield chunk


async def _compare_and_merge_nodes(
    nodes: List[BaseNode],
    user_id: Optional[str] = None,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> List[BaseNode]:
    """Compare all nodes and resolve them using LLM, with a
    blocking stage that bounds the LLM input size.

    Pipeline (B2-er slice 1):
      1. Dict-keyed dedup by node id (was O(n²) list scan).
      2. Property-based blocking via ``_block_keys_for_node`` —
         buckets nodes that look related (same canonical_key,
         same name prefix, etc.) into candidate groups.
      3. LLM resolution per candidate group via
         ``resolve_entity_group``. Singletons skip the LLM.
      4. Pass-through for nodes that didn't land in any
         candidate group (e.g. a single Person with no
         lookalikes — no resolution needed).

    Recall tradeoff vs the old all-pairs path: nodes that should
    match but never share a block key get missed. The block
    keys are designed to over-include (canonical_key + name3 +
    type-fallback) so this is rare in practice. Slice 2 layers
    embedding-based blocking on top to catch the long tail.
    """
    if not nodes or len(nodes) <= 1:
        return nodes

    # Step 1 — O(n) dedup by id. The pre-slice-1 code did an
    # O(n²) list scan here.
    by_id: Dict[str, BaseNode] = {}
    for node in nodes:
        if node.id in by_id:
            by_id[node.id] = merge_nodes(by_id[node.id], node)
        else:
            by_id[node.id] = node
    deduped = list(by_id.values())

    # Step 2 + 3 — block, then resolve each candidate group via LLM.
    final_nodes: List[BaseNode] = []
    nodes_in_groups: set = set()

    for candidate_group in _candidate_groups_for_resolution(deduped):
        # All nodes in a candidate group share an entity type
        # (block keys are type-prefixed).
        entity_type = candidate_group[0].type
        resolved_groups = await resolve_entity_group(
            entity_type,
            candidate_group,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )
        for group in resolved_groups:
            if len(group) == 1:
                final_nodes.append(group[0])
                nodes_in_groups.add(group[0].id)
                continue
            # Sort by confidence; merge into the highest-confidence base.
            sorted_nodes = sorted(
                group,
                key=lambda x: x.confidence_score if x.confidence_score else 0,
                reverse=True,
            )
            base_node = sorted_nodes[0]
            for other_node in sorted_nodes[1:]:
                base_node = merge_nodes(base_node, other_node)
            final_nodes.append(base_node)
            for n in group:
                nodes_in_groups.add(n.id)

    # Step 4 — pass-through for nodes that didn't appear in any
    # candidate group (singletons in their block, or in a block
    # that yielded only singletons).
    for node in deduped:
        if node.id not in nodes_in_groups:
            final_nodes.append(node)

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


async def _build_graph_with_multi_pass(
    ontology_parser: OntologyParser,
    chunks: List[str],
    transform_id: str,
    llm_client: LLMClient,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    max_passes: int = 2,
    chunk_metadatas: Optional[List[Any]] = None,
    extractor_model: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    """Build knowledge graph using multi-pass extraction with validation.

    This function uses the MultiPassExtractor to perform validation-driven
    iterative extraction that identifies and addresses gaps in the initial
    extraction through targeted refinement passes.

    Args:
        ontology_parser: Parser for the target ontology.
        chunks: List of text chunks to extract from.
        transform_id: Unique identifier for this transform.
        llm_client: LLM client for extraction.
        progress_callback: Optional callback for progress updates.
        user_id: User ID for LLM credentials.
        document_usage_id: Document usage tracking ID.
        max_passes: Maximum extraction passes.

    Returns:
        DocumentKnowledgeGraph with extracted nodes and relationships.
    """
    from graphora_server.services.extraction import MultiPassExtractor, MultiPassConfig

    logger.info(
        "Starting multi-pass extraction",
        extra={
            "transform_id": transform_id,
            "chunk_count": len(chunks),
            "max_passes": max_passes,
        },
    )

    # Configure multi-pass extraction
    config = MultiPassConfig(
        max_passes=max_passes,
        gap_severity_threshold=0.5,
        enable_parallel_refinement=True,
    )

    # Create multi-pass extractor
    extractor = MultiPassExtractor(
        ontology_parser=ontology_parser,
        llm_client=llm_client,
        config=config,
    )

    # Perform multi-pass extraction
    nodes, relationships = await extractor.extract(
        chunks=chunks,
        transform_id=transform_id,
        user_id=user_id,
        max_passes=max_passes,
        progress_callback=progress_callback,
        chunk_metadatas=chunk_metadatas,
        extractor_model=extractor_model,
    )

    # Hydrate nodes with entity ledger if user_id provided
    if user_id:
        await entity_ledger_service.hydrate_nodes(user_id, nodes)

    # Apply entity resolution and deduplication
    nodes = await _compare_and_merge_nodes(
        nodes,
        user_id=user_id,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
    )
    nodes, relationships = await deduplicate_entities_with_splink(
        entities=nodes,
        relationships=relationships,
        parsed_ontology=ontology_parser.parsed_ontology,
        user_id=user_id,
    )

    # Record nodes to entity ledger
    if user_id:
        await entity_ledger_service.record_nodes(user_id, nodes)

    # Build final graph and prune orphans
    kg = DocumentKnowledgeGraph(nodes=nodes, relationships=relationships)
    prune_orphaned_nodes(ontology_parser.parsed_ontology, kg)

    logger.info(
        "Multi-pass extraction complete",
        extra={
            "transform_id": transform_id,
            "node_count": len(kg.nodes),
            "relationship_count": len(kg.relationships),
        },
    )

    return kg
