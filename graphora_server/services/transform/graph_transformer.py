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
from graphora_server.services.claims_service import (
    Claim,
    ClaimsService,
    TargetKind as ClaimTargetKind,
)
from graphora_server.services.decision_log_service import (
    Decision,
    DecisionLogService,
    DecisionType,
    TargetKind,
)
from graphora_server.utils.constants import SYSTEM_PROPERTIES
from graphora_server.services.disputed_pairs_service import (
    DisputedPair,
    DisputedPairsService,
    SourceStage,
)
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


def _build_orphan_focused_context(
    orphans: List[BaseNode],
    candidates: List[BaseNode],
) -> str:
    """Render a small, labeled context block for the orphan pass.

    The first call's context lists every node + every edge accepted
    so far; on a 40-node doc that is the bulk of the prompt. For the
    second pass we only need the LLM to look at the orphans + their
    same-chunk counterparts, so we drop everything else. Labeling the
    sections makes the ask explicit ("connect THESE if you can").
    """
    orphan_ids = {n.id for n in orphans}
    other_candidates = [c for c in candidates if c.id not in orphan_ids]

    lines: List[str] = []
    lines.append(
        "These entities were extracted from this chunk but have no "
        "relationships yet. Identify any relationships that involve "
        "them, drawing on the entities listed below as candidate "
        "counterparts when applicable."
    )
    lines.append("")
    lines.append("Entities needing relationships:")
    for node in sorted(orphans, key=_node_context_sort_key):
        lines.append(
            f"({node.type}:{{'id': '{node.id}', "
            f"'properties': {_format_properties(node.properties)}}})"
        )

    if other_candidates:
        lines.append("")
        lines.append("Other entities from the same chunk (candidate counterparts):")
        for node in sorted(other_candidates, key=_node_context_sort_key):
            lines.append(
                f"({node.type}:{{'id': '{node.id}', "
                f"'properties': {_format_properties(node.properties)}}})"
            )

    raw = "\n".join(lines) + "\n"
    return _make_context_envelope(raw, stage="relationships").text


async def _orphan_relationship_pass(
    *,
    nodes: List[BaseNode],
    relationships: List[RelationshipInstance],
    chunks_or_pdf_paths: List[Any],
    chunk_metadatas: Optional[List[Any]],
    relationship_extractor: Callable[..., Any],
    relationships_only_ontology: type,
    ontology_parser: OntologyParser,
    parsed_ontology: Dict[str, Any],
    treat_chunks_as_text: bool,
    user_id: Optional[str],
    transform_id: Optional[str],
    document_usage_id: Optional[str],
    extractor_model: Optional[str],
    rel_baml_function: Optional[str],
) -> List[RelationshipInstance]:
    """Second-pass relationship extraction targeting orphan nodes.

    After the per-chunk relationship sweep, some nodes may still
    carry no incoming/outgoing edge — either the LLM missed a
    relationship the source mentions, or the chunk genuinely doesn't
    mention one. We re-call the relationship extractor for each
    chunk that produced an orphan, this time with a *focused*
    context that lists only the orphans plus their same-chunk
    candidate counterparts. Any new edges the LLM emits are deduped
    against the existing set and appended.

    Why per chunk and not whole-graph: the chunk is the only window
    in which the LLM has the original source text in scope. A single
    cross-chunk pass would force us to either resend every chunk
    (expensive) or rely on context strings alone (which is exactly
    what just produced the gap). Re-asking on the chunk that
    produced the orphan is the cheapest way to give the model a
    second look with full context.
    """
    if not nodes or not chunks_or_pdf_paths:
        return relationships

    connected_ids = set()
    for rel in relationships:
        connected_ids.add(rel.source_id)
        connected_ids.add(rel.target_id)

    orphans = [n for n in nodes if n.id not in connected_ids]
    if not orphans:
        return relationships

    chunk_id_to_index: Dict[str, int] = {}
    if chunk_metadatas:
        for idx, cm in enumerate(chunk_metadatas):
            chunk_id = getattr(cm, "chunk_id", None) if cm else None
            if chunk_id:
                chunk_id_to_index[chunk_id] = idx

    orphans_by_chunk_index: Dict[int, List[BaseNode]] = {}
    for orphan in orphans:
        chunk_ids = (orphan.provenance.chunk_ids or []) if orphan.provenance else []
        for chunk_id in chunk_ids:
            idx = chunk_id_to_index.get(chunk_id)
            if idx is None:
                continue
            bucket = orphans_by_chunk_index.setdefault(idx, [])
            if all(existing.id != orphan.id for existing in bucket):
                bucket.append(orphan)

    if not orphans_by_chunk_index:
        logger.info(
            "Orphan re-extraction skipped: %d orphans found but none "
            "could be mapped back to a known chunk",
            len(orphans),
        )
        return relationships

    chunk_ids_for_node: Dict[str, set] = {
        n.id: set((n.provenance.chunk_ids or []) if n.provenance else []) for n in nodes
    }

    new_relationships: List[RelationshipInstance] = []

    for idx, orphan_list in orphans_by_chunk_index.items():
        cm = chunk_metadatas[idx] if chunk_metadatas else None
        chunk_id = getattr(cm, "chunk_id", None) if cm else None

        candidate_ids = {n.id for n in orphan_list}
        candidates: List[BaseNode] = list(orphan_list)
        if chunk_id:
            for n in nodes:
                if n.id in candidate_ids:
                    continue
                if chunk_id in chunk_ids_for_node.get(n.id, set()):
                    candidates.append(n)
                    candidate_ids.add(n.id)

        focused_context = _build_orphan_focused_context(orphan_list, candidates)

        chunk_text_for_props = (
            chunks_or_pdf_paths[idx]
            if treat_chunks_as_text and isinstance(chunks_or_pdf_paths[idx], str)
            else None
        )

        try:
            relationships_only_kg, _duration = await _timed_call(
                relationship_extractor,
                chunks_or_pdf_paths[idx],
                response_model=relationships_only_ontology,
                context=focused_context,
                ontology_yaml=ontology_parser.ontology_yaml,
                user_id=user_id,
                transform_id=transform_id,
                document_usage_id=document_usage_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Orphan re-extraction call failed for chunk_index=%s: %s",
                idx,
                exc,
            )
            continue

        # Pass full node list so transform_as_relationships can resolve
        # any IDs the model might emit (including cross-chunk edges).
        rels = transform_as_relationships(
            parsed_ontology,
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

        for rel in rels:
            if any(
                _is_duplicate_relationship(existing, rel) for existing in relationships
            ):
                continue
            if any(
                _is_duplicate_relationship(existing, rel)
                for existing in new_relationships
            ):
                continue
            new_relationships.append(rel)

    if new_relationships:
        logger.info(
            "Orphan re-extraction added %d new relationship(s) across "
            "%d chunk(s); orphans before=%d",
            len(new_relationships),
            len(orphans_by_chunk_index),
            len(orphans),
        )
    else:
        logger.info(
            "Orphan re-extraction produced no new relationships "
            "(orphans=%d, chunks_revisited=%d)",
            len(orphans),
            len(orphans_by_chunk_index),
        )

    return relationships + new_relationships


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
    decision_log: Optional[DecisionLogService] = None,
    disputed_pairs_service: Optional[DisputedPairsService] = None,
    claims_service: Optional[ClaimsService] = None,
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
            decision_log=decision_log,
            disputed_pairs_service=disputed_pairs_service,
            claims_service=claims_service,
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
        decision_log=decision_log,
        disputed_pairs_service=disputed_pairs_service,
        claims_service=claims_service,
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
    decision_log: Optional[DecisionLogService] = None,
    disputed_pairs_service: Optional[DisputedPairsService] = None,
    claims_service: Optional[ClaimsService] = None,
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
        decision_log=decision_log,
        disputed_pairs_service=disputed_pairs_service,
        claims_service=claims_service,
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
    decision_log: Optional[DecisionLogService] = None,
    disputed_pairs_service: Optional[DisputedPairsService] = None,
    claims_service: Optional[ClaimsService] = None,
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
        # B1-prob slice 2b: emit one claim per (node, property)
        # BEFORE the dedup-by-content loop below. Reason: two
        # chunks extracting the same canonical_id with different
        # property values are exactly the contradiction signal,
        # but the dedup check would drop the second extraction.
        # Emitting first preserves the cross-chunk disagreement
        # so the /contradictions endpoint can surface it.
        await _emit_node_property_claims(
            claims_service,
            base_nodes,
            transform_id=transform_id,
            user_id=user_id,
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

    # Step 2: Compare & merge entities. Splink blocking runs inside
    # _compare_and_merge_nodes as the slice-3 stage of the four-stage
    # pipeline (property → embedding → Splink → LLM). The standalone
    # post-call deduplicate_entities_with_splink invocation that used
    # to live here was removed in slice 3 — its work now happens
    # inside the candidate-group pipeline. The post-relationships
    # call below (Step 4) stays because it does relationship-rewriting
    # which slice 3 doesn't replace.
    nodes = await _compare_and_merge_nodes(
        nodes,
        user_id=user_id,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        parsed_ontology=ontology_parser.parsed_ontology,
        decision_log=decision_log,
        disputed_pairs_service=disputed_pairs_service,
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

    # Step 4.5: Orphan re-extraction. Give the LLM a focused second
    # look at any node that ended up with no edges, scoped to the
    # chunk(s) it was extracted from. Keeps the prune-guard's intent
    # ("don't drop real entities") while plugging the missed-edge
    # failure mode rather than the fabricated-entity one.
    relationships = await _orphan_relationship_pass(
        nodes=nodes,
        relationships=relationships,
        chunks_or_pdf_paths=chunks_or_pdf_paths,
        chunk_metadatas=chunk_metadatas,
        relationship_extractor=relationship_extractor,
        relationships_only_ontology=relationships_only_ontology,
        ontology_parser=ontology_parser,
        parsed_ontology=ontology_parser.parsed_ontology,
        treat_chunks_as_text=treat_chunks_as_text,
        user_id=user_id,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        extractor_model=extractor_model,
        rel_baml_function=rel_baml_function,
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
    """Render node/relationship properties for the LLM context block.

    Filters SYSTEM_PROPERTIES out before serializing — A1-prov and
    B0-prov-extend stamp many provenance fields onto node.properties
    (source_chunk_id, document_name, page_number, source_text up to
    ~1000 chars, extractor_model, prompt_version, validator_score,
    extraction_timestamp, ...). Dumping all of them into the context
    block ballooned the relationship-extraction prompt from ~16k
    tokens to ~81k on the Apple 10K, drowned Gemini in metadata
    noise, and slashed relationship recall (only 2-7 rels surviving
    instead of the historical baseline).

    SYSTEM_PROPERTIES is the centralized "internal/metadata, not
    user-meaningful" list — same set the similarity scorer and
    ontology-validation skip-lists already filter on. Keeping the
    LLM-context formatter aligned with that list restores prompt
    size and extraction quality.
    """
    if not properties:
        return "{}"
    from graphora_server.utils.constants import SYSTEM_PROPERTIES

    user_props = {
        k: v
        for k, v in properties.items()
        if k not in SYSTEM_PROPERTIES and v not in (None, "")
    }
    if not user_props:
        return "{}"
    return json.dumps(user_props, sort_keys=True, default=str)


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


def _node_to_embedding_text(node: BaseNode) -> str:
    """Compose the text representation of a node for embedding-based
    similarity. Mirrors entity_ledger_service._node_to_text:
    canonical properties first (highest signal), then regular
    properties; capped at 5 segments joined by `` | ``.

    Filters SYSTEM_PROPERTIES from the regular-property pass so
    A1-prov source-span fields (source_text, document_name,
    document_id, etc.) and B0-prov-extend decision-trail fields
    (extractor_model, prompt_version, validator_score) don't
    leak into the embedding signal. Without this filter, two
    unrelated nodes from the same document would score
    artificially similar on document_name + source_text overlap,
    and real entity matches would get diluted by metadata-heavy
    text. Same shape of bug the C2-postgres slice 6 review caught
    for the AGE find_similar_nodes scoring path.
    """
    from graphora_server.utils.constants import SYSTEM_PROPERTIES

    parts: List[str] = []
    canonical_props = node.canonical_properties or {}
    props = node.properties or {}
    for value in canonical_props.values():
        if isinstance(value, str) and len(value) > 1:
            parts.append(value)
    for key, value in props.items():
        if (
            key not in canonical_props
            and key not in SYSTEM_PROPERTIES
            and isinstance(value, str)
            and len(value) > 1
        ):
            parts.append(value)
    return " | ".join(parts[:5]) if parts else ""


def _embedding_candidate_groups(
    nodes: List[BaseNode],
    similarity_threshold: Optional[float] = None,
) -> Iterable[List[BaseNode]]:
    """Yield candidate match groups based on embedding similarity.

    B2-er slice 2: closes the recall gap slice 1's property-based
    blocker exposed. Designed to run on nodes that the property
    blocker left unmatched — typically because they share an entity
    type but have varied surface forms ("John Smith" / "Jonathan S."
    / "J. Smith") that don't share a name3 prefix or canonical_key.

    Embedding cost is bounded by the property blocker's miss count,
    not the total node count: ``_compare_and_merge_nodes`` only
    feeds nodes here that didn't land in any property-based
    candidate group.

    Returns nothing (no candidate groups) when:
      - settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED is False
      - the embedding extras aren't installed
        (ImportError on get_embedding_similarity)
      - no nodes have embeddable text (empty
        canonical_properties / properties)
      - no pairs cross the similarity threshold

    The function is deliberately resilient — it never raises into
    the resolution flow. An embedding-side problem degrades to "no
    additional candidate groups" rather than blocking extraction.
    """
    if not nodes or len(nodes) <= 1:
        return

    if not getattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", False):
        return

    # Resolve threshold from the configured ER setting when the
    # caller didn't pin one explicitly. Other embedding-based ER
    # paths (cross_document_service, splink_embedding_comparison)
    # honor ENTITY_RESOLUTION_SIMILARITY_THRESHOLD; this stage
    # joins the convention so operators tuning the setting see
    # consistent behaviour across all four ER stages.
    if similarity_threshold is None:
        similarity_threshold = float(
            getattr(settings, "ENTITY_RESOLUTION_SIMILARITY_THRESHOLD", 0.85)
        )

    try:
        from graphora_server.services.entity_resolution.embedding_similarity import (
            get_embedding_similarity,
        )
    except ImportError:
        logger.debug("Embedding similarity not available; skipping ER embedding stage")
        return

    try:
        embedding_similarity = get_embedding_similarity(
            model_name=settings.ENTITY_RESOLUTION_EMBEDDING_MODEL,
        )
    except Exception as exc:  # pragma: no cover — defensive log
        logger.warning("Embedding similarity init failed: %s", exc)
        return

    # Group by type — semantic similarity is only meaningful within
    # type. Mirrors the type-prefix isolation the property blocker
    # uses.
    by_type: Dict[str, List[BaseNode]] = {}
    for node in nodes:
        by_type.setdefault(node.type, []).append(node)

    for type_nodes in by_type.values():
        if len(type_nodes) <= 1:
            continue

        texts: List[str] = []
        valid_nodes: List[BaseNode] = []
        for node in type_nodes:
            text = _node_to_embedding_text(node)
            if text:
                texts.append(text)
                valid_nodes.append(node)

        if len(valid_nodes) <= 1:
            continue

        try:
            sim_matrix = embedding_similarity.compute_similarity_matrix(texts, texts)
        except Exception as exc:
            logger.warning("Embedding similarity matrix failed: %s", exc)
            continue

        # Union-find: pairs above threshold merge into one group.
        # Path-compression keeps the operation effectively linear.
        parent = list(range(len(valid_nodes)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(valid_nodes)):
            for j in range(i + 1, len(valid_nodes)):
                if float(sim_matrix[i][j]) >= similarity_threshold:
                    _union(i, j)

        groups: Dict[int, List[BaseNode]] = {}
        for i, node in enumerate(valid_nodes):
            groups.setdefault(_find(i), []).append(node)

        for group in groups.values():
            if len(group) > 1:
                yield group


async def _splink_candidate_groups(
    nodes: List[BaseNode],
    parsed_ontology: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> List[List[BaseNode]]:
    """Yield candidate match groups from Splink probabilistic
    record-linkage.

    B2-er slice 3: closes the recall gap that property-based
    blocking (slice 1) and embedding-based blocking (slice 2)
    leave on the table. Splink's m/u-probability scoring with
    learned comparisons (JaroWinkler / Levenshtein on ID-like
    columns, ExactMatch on canonical fields, etc.) catches pairs
    that share neither block keys nor embedding similarity but
    still match according to the probabilistic model.

    Operates on the SUBSET that earlier blockers missed, so the
    cost is bounded relative to total node count. Mirrors the
    ``_embedding_candidate_groups`` contract: returns ``[]`` on
    every failure mode rather than raising into extraction.

    Returns nothing when:
      - the [er] extra (splink + pandas) isn't installed
      - the underlying clustering helper raises mid-flow
      - no Splink clusters of size >= 2 are found

    The function is async because Splink's threshold lookup
    talks to ``merge_learning_service`` (DB-backed in production).
    """
    if not nodes or len(nodes) < 2:
        return []

    try:
        from graphora_server.services.transform.helpers import (
            cluster_entities_with_splink,
        )
    except ImportError:
        logger.debug("Splink extra not installed; skipping ER Splink stage")
        return []

    try:
        id_to_representative = await cluster_entities_with_splink(
            entities=nodes,
            parsed_ontology=parsed_ontology,
            user_id=user_id,
        )
    except Exception as exc:  # pragma: no cover — defensive log
        logger.warning("Splink candidate-group blocker failed: %s", exc)
        return []

    if not id_to_representative:
        return []

    by_id: Dict[str, BaseNode] = {n.id: n for n in nodes}
    by_rep: Dict[str, List[BaseNode]] = {}
    for entity_id, rep_id in id_to_representative.items():
        node = by_id.get(entity_id)
        if node is None:
            # Splink may have seen entities the caller didn't pass
            # in (shouldn't happen with our usage, but be defensive).
            continue
        by_rep.setdefault(rep_id, []).append(node)

    # Ensure each representative is itself in its group — the
    # mapping convention is "duplicate -> representative" with the
    # representative's own self-mapping omitted.
    for rep_id in list(by_rep.keys()):
        rep_node = by_id.get(rep_id)
        if rep_node and rep_node not in by_rep[rep_id]:
            by_rep[rep_id].append(rep_node)

    return [grp for grp in by_rep.values() if len(grp) >= 2]


async def _emit_node_property_claims(
    claims_service: Optional[ClaimsService],
    nodes: List[BaseNode],
    transform_id: Optional[str],
    user_id: Optional[str],
) -> None:
    """B1-prob slice 2b: emit one Claim per (node, property) pair
    so the /contradictions surface can group cross-chunk
    disagreements about the same entity.

    Called from the per-chunk extraction loop BEFORE the
    dedup-by-content check, so claims for "Alice extracted with
    title=Engineer (chunk 1)" and "Alice extracted with
    title=Senior Engineer (chunk 2)" both land — the
    contradiction detector groups them by canonical_id +
    property_key and surfaces the disagreement.

    Target identity: ``canonical_id`` (the stable hash computed
    from "unique"-flagged ontology properties) when available,
    falling back to the per-chunk extraction id. The
    contradictions endpoint groups by target_id, so without
    canonical_id the per-chunk ids stay separate and no
    contradiction surfaces — graceful degrade for ontologies
    without unique properties.

    Failure posture: log-and-swallow. Each claim is its own
    try/except so one bad row doesn't poison the rest of the
    batch. Mirrors B0-log's ``decision_log.append`` posture —
    extraction must never fail because claim-writing failed.

    No-ops when:
      * claims_service is None (callers without claim wiring).
      * transform_id is None (claims are keyed by transform).
      * user_id is None (tenant-scoping invariant — would write
        an orphan row the read endpoints can't surface anyway).
    """
    if not claims_service or not transform_id or not user_id:
        return

    for node in nodes:
        target_id = node.canonical_id or node.id
        # Per-node provenance for the source_* fields. The
        # _attach_provenance_properties helper also mirrors these
        # into node.properties, but reading from the typed model
        # is cheaper than dict lookups + None-coalescing.
        prov = node.provenance
        confidence = 1.0
        if prov is not None and prov.confidence_score is not None:
            confidence = max(0.0, min(1.0, float(prov.confidence_score)))
        source_chunk_id = None
        source_extractor_model = None
        source_prompt_version = None
        if prov is not None:
            # chunk_ids is a list because merged nodes union
            # their sources; pre-merge each node still has one.
            source_chunk_id = prov.chunk_ids[0] if prov.chunk_ids else None
            source_extractor_model = prov.extractor_model
            source_prompt_version = prov.prompt_version

        for property_key, value in (node.properties or {}).items():
            # System properties (source_chunk_id, extractor_model,
            # extraction_confidence, etc.) are observability, not
            # claims. The contradiction detector keyed on them
            # would surface every cross-chunk provenance difference
            # as a "contradiction" — exactly the wrong signal.
            if property_key in SYSTEM_PROPERTIES:
                continue
            # The `id` literal is also in SYSTEM_PROPERTIES, but
            # defensively skip None values too — extraction output
            # can carry None for unset fields and we don't want to
            # emit "claim that X has property K = None."
            if value is None:
                continue
            try:
                await claims_service.append(
                    Claim(
                        transform_id=transform_id,
                        target_id=target_id,
                        target_kind=ClaimTargetKind.NODE,
                        property_key=property_key,
                        value=value,
                        confidence=confidence,
                        user_id=user_id,
                        source_chunk_id=source_chunk_id,
                        source_extractor_model=source_extractor_model,
                        source_prompt_version=source_prompt_version,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                # Log and continue — never let claim-emission
                # block extraction. This mirrors the B0-log
                # posture from commit 8cbc76b. The
                # /contradictions surface gracefully degrades to
                # whatever DID land.
                logger.warning(
                    "Failed to emit claim for node=%s property=%s: %s",
                    target_id,
                    property_key,
                    exc,
                )


async def _emit_entity_merged_decision(
    decision_log: Optional[DecisionLogService],
    transform_id: Optional[str],
    base_node: BaseNode,
    merged_away: List[BaseNode],
    stage: str,
    candidate_group_size: int,
    user_id: Optional[str] = None,
) -> None:
    """B0-log slice 2: emit one entity_merged decision per merge
    event so the Decision Log surface (Evidence tab,
    MCP get_evidence) can render which signal drove the merge.

    No-ops when:
      * decision_log is None (default — preserves the pre-slice-2
        behaviour for callers that don't construct a service).
      * transform_id is None (the log is keyed by transform; without
        it, the row is unfindable).
      * merged_away is empty (singleton groups don't represent a
        merge event — the resolver returned the same node it got).

    ``stage`` is the blocker that surfaced the candidate group
    (``property_blocker``, ``embedding_blocker``, ``splink_blocker``);
    ``candidate_group_size`` is the count fed to the LLM resolver.
    Both end up in ``evidence`` so the rendering layer can show "this
    merge was caught by Splink after property and embedding stages
    missed it" — useful for ER tuning."""
    if not decision_log or not transform_id or not merged_away:
        return
    await decision_log.append(
        Decision(
            transform_id=transform_id,
            target_id=base_node.id,
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason=(
                f"Merged {len(merged_away)} node(s) into {base_node.id} " f"via {stage}"
            ),
            evidence={
                "stage": stage,
                "candidate_group_size": candidate_group_size,
                "merge_group_size": len(merged_away) + 1,
                "node_type": base_node.type,
            },
            alternatives=[
                {
                    "id": n.id,
                    "type": n.type,
                    "canonical_key": n.canonical_key,
                    "confidence_score": n.confidence_score,
                }
                for n in merged_away
            ],
            user_id=user_id,
        )
    )


async def _enqueue_disputed_pair_if_unresolved(
    *,
    disputed_pairs_service: Optional[DisputedPairsService],
    user_id: Optional[str],
    transform_id: Optional[str],
    candidate_group: List[BaseNode],
    resolved_groups: List[List[BaseNode]],
    source_stage: SourceStage,
) -> None:
    """B2-active slice B: enqueue a disputed pair when the LLM
    rejected a 2-node candidate that a blocker had flagged.

    Fires when:
      * The candidate group has exactly 2 nodes (slice B keeps
        the signal tight — larger candidate groups can produce
        combinatorial disputed pairs and are deferred to a
        future slice).
      * The LLM resolver returned BOTH nodes as singletons (no
        merge happened). That's the clearest "blocker said yes,
        LLM said no" signal worth human review.

    Short-circuits when service / user / transform missing — same
    safe-default pattern as ``_emit_entity_merged_decision``.

    Failures during enqueue are logged-and-swallowed. The
    disputed-pairs queue is OBSERVABILITY FOR ER here (the merge
    pipeline already committed to the LLM's verdict); the
    correctness-strict propagation lives in the API path where
    losing a row means losing an explicit user-initiated request.
    Mirrors the same observability-vs-correctness split applied
    to DecisionLogService.append earlier."""
    if not disputed_pairs_service or not user_id or not transform_id:
        return
    if len(candidate_group) != 2:
        return
    # All-singletons check: both nodes came back unmerged.
    if len(resolved_groups) != 2:
        return
    if any(len(g) != 1 for g in resolved_groups):
        return

    a, b = candidate_group
    pair = DisputedPair(
        user_id=user_id,
        transform_id=transform_id,
        node_a_id=a.id,
        node_b_id=b.id,
        entity_type=a.type,
        node_a_canonical_key=a.canonical_key,
        node_b_canonical_key=b.canonical_key,
        source_stage=source_stage,
        # No single similarity score across the blocker stages —
        # property uses canonical_key match, embedding uses
        # cosine, splink uses m/u probability. Leaving None
        # rather than picking a stage-specific number that would
        # surface as misleading.
        similarity_score=None,
    )
    try:
        await disputed_pairs_service.enqueue(pair)
    except Exception as exc:
        logger.warning(
            "Failed to enqueue disputed pair (%s, %s) for " "transform=%s stage=%s: %s",
            a.id,
            b.id,
            transform_id,
            source_stage.value,
            exc,
        )


async def _compare_and_merge_nodes(
    nodes: List[BaseNode],
    user_id: Optional[str] = None,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    parsed_ontology: Optional[Dict[str, Any]] = None,
    decision_log: Optional[DecisionLogService] = None,
    disputed_pairs_service: Optional[DisputedPairsService] = None,
) -> List[BaseNode]:
    """Compare all nodes and resolve them using LLM, with a
    layered blocking stack that bounds the LLM input size.

    Pipeline (B2-er slices 1, 2, 3):
      1. Dict-keyed dedup by node id (was O(n²) list scan).
      2. Property-based blocking via ``_block_keys_for_node`` —
         buckets nodes that look related (same canonical_key,
         same name prefix, etc.) into candidate groups.
      3. LLM resolution per candidate group via
         ``resolve_entity_group``. Singletons skip the LLM.
      4. Embedding-based blocking on slice-2's misses
         (``_embedding_candidate_groups``) — catches semantic
         variants ("John Smith" / "Jonathan S.") that don't
         share property signals. LLM resolves the new groups.
      5. Splink probabilistic blocking on slice-3's misses
         (``_splink_candidate_groups``) — catches pairs the
         property and embedding stages both missed but that
         Splink's m/u-probability comparisons recognize. LLM
         resolves the new groups.
      6. Pass-through for nodes that didn't land in any
         candidate group from any blocker (genuine singletons).

    Each blocker only sees nodes the prior blockers missed, so
    cost stays bounded as we add stages. ``parsed_ontology``
    (slice 3) is needed for Splink's per-type comparison rules;
    earlier blockers are ontology-agnostic.
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
        # B2-active slice B: enqueue if the LLM rejected a 2-node
        # candidate that the property blocker had flagged.
        await _enqueue_disputed_pair_if_unresolved(
            disputed_pairs_service=disputed_pairs_service,
            user_id=user_id,
            transform_id=transform_id,
            candidate_group=candidate_group,
            resolved_groups=resolved_groups,
            source_stage=SourceStage.PROPERTY_BLOCKER,
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
            await _emit_entity_merged_decision(
                decision_log,
                transform_id,
                base_node,
                sorted_nodes[1:],
                stage="property_blocker",
                candidate_group_size=len(candidate_group),
                user_id=user_id,
            )

    # Step 3.5 — B2-er slice 2 embedding-based blocking on the
    # nodes the property blocker missed. Catches semantic variants
    # ("John Smith" / "Jonathan S.") that don't share name3 prefix
    # or canonical_key. Bounded cost: only runs on the unblocked
    # subset.
    unblocked = [n for n in deduped if n.id not in nodes_in_groups]
    for candidate_group in _embedding_candidate_groups(unblocked):
        entity_type = candidate_group[0].type
        resolved_groups = await resolve_entity_group(
            entity_type,
            candidate_group,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )
        # B2-active slice B: enqueue if embedding blocker's
        # 2-node candidate was rejected by the LLM.
        await _enqueue_disputed_pair_if_unresolved(
            disputed_pairs_service=disputed_pairs_service,
            user_id=user_id,
            transform_id=transform_id,
            candidate_group=candidate_group,
            resolved_groups=resolved_groups,
            source_stage=SourceStage.EMBEDDING_BLOCKER,
        )
        for group in resolved_groups:
            if len(group) == 1:
                final_nodes.append(group[0])
                nodes_in_groups.add(group[0].id)
                continue
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
            await _emit_entity_merged_decision(
                decision_log,
                transform_id,
                base_node,
                sorted_nodes[1:],
                stage="embedding_blocker",
                candidate_group_size=len(candidate_group),
                user_id=user_id,
            )

    # Step 3.75 — B2-er slice 3 Splink probabilistic blocking on
    # the nodes both prior blockers missed. Splink's m/u-probability
    # scoring catches pairs that share neither block keys nor
    # embedding similarity but match on learned property
    # comparisons. Same boilerplate as the embedding loop; only
    # the candidate-group source changes.
    unblocked = [n for n in deduped if n.id not in nodes_in_groups]
    splink_groups = await _splink_candidate_groups(
        unblocked,
        parsed_ontology=parsed_ontology,
        user_id=user_id,
    )
    for candidate_group in splink_groups:
        entity_type = candidate_group[0].type
        resolved_groups = await resolve_entity_group(
            entity_type,
            candidate_group,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )
        # B2-active slice B: enqueue if Splink's 2-node candidate
        # was rejected by the LLM. Splink tends to produce the
        # noisiest blocker hits (m/u probability is generous), so
        # this stage may dominate the queue — operators can filter
        # by source_stage in the review UI.
        await _enqueue_disputed_pair_if_unresolved(
            disputed_pairs_service=disputed_pairs_service,
            user_id=user_id,
            transform_id=transform_id,
            candidate_group=candidate_group,
            resolved_groups=resolved_groups,
            source_stage=SourceStage.SPLINK_BLOCKER,
        )
        for group in resolved_groups:
            if len(group) == 1:
                final_nodes.append(group[0])
                nodes_in_groups.add(group[0].id)
                continue
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
            await _emit_entity_merged_decision(
                decision_log,
                transform_id,
                base_node,
                sorted_nodes[1:],
                stage="splink_blocker",
                candidate_group_size=len(candidate_group),
                user_id=user_id,
            )

    # Step 4 — pass-through for nodes that didn't appear in any
    # candidate group (genuine singletons: missed by property,
    # embedding, AND Splink stages).
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
    decision_log: Optional[DecisionLogService] = None,
    disputed_pairs_service: Optional[DisputedPairsService] = None,
    # B1-prob slice 2b: accepted for signature parity but the
    # multi-pass extractor doesn't yet emit claims (each pass
    # has its own internal extraction loop). Wiring the hook
    # into MultiPassExtractor is a slice 2b follow-up.
    claims_service: Optional[ClaimsService] = None,
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

    # Apply entity resolution and deduplication. Splink blocking
    # runs inside _compare_and_merge_nodes (slice 3). The follow-up
    # deduplicate_entities_with_splink call stays — it threads
    # relationships through and rewrites their source/target IDs
    # whenever node merges happen. Slice 3 doesn't replace that
    # relationship-rewriting side-effect.
    nodes = await _compare_and_merge_nodes(
        nodes,
        user_id=user_id,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        parsed_ontology=ontology_parser.parsed_ontology,
        decision_log=decision_log,
        disputed_pairs_service=disputed_pairs_service,
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
