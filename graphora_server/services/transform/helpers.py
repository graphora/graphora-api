from graphora_server.schemas.graph import Edge, Node
from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    DocumentKnowledgeGraph,
)
from graphora_server.services.chunking.models import ChunkMetadata
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Optional, Callable, Set, Union
from dataclasses import dataclass
import json
import uuid
import hashlib
import re
from datetime import datetime, timezone
from collections import defaultdict
from graphora_server.utils.logger import logger
from graphora_server.services.llm.client import LLMClient
from graphora_server.utils.constants import SYSTEM_PROPERTIES
from graphora_server.config import settings
from graphora_server.services.claims_service import (
    Claim,
    ClaimsService,
    TargetKind as ClaimTargetKind,
)

# Entity resolution via Splink is a [er] extra: splink + pandas pull
# duckdb and numpy; heavy for installs that don't use ER. Module scope
# imports are retained but guarded so module load succeeds without [er];
# any code path that actually touches Splink will call
# _require_er_extras() first.
try:
    from splink import block_on, DuckDBAPI, Linker, SettingsCreator
    import splink.comparison_library as cl
    import pandas as pd
except ImportError:  # pragma: no cover — exercised when [er] missing
    block_on = None  # type: ignore
    DuckDBAPI = None  # type: ignore
    Linker = None  # type: ignore
    SettingsCreator = None  # type: ignore
    cl = None  # type: ignore
    pd = None  # type: ignore


def _require_er_extras() -> None:
    if any(
        symbol is None
        for symbol in (block_on, DuckDBAPI, Linker, SettingsCreator, cl, pd)
    ):
        raise ImportError(
            "Entity resolution requires the [er] extra. "
            "Install with: pip install 'graphora-server[er]'"
        )


import logging  # noqa: E402 — preceded by lazy-import guard
import copy  # noqa: E402
import traceback  # noqa: E402
from graphora_server.services.merge.learning import merge_learning_service  # noqa: E402


CANONICAL_COLUMN_PREFIX = "canonical__"


SMALL_ENTITY_GROUP_THRESHOLD = 6

BATCH_SIZE_THRESHOLD = 500


class OntologyPropertyCache:
    """Cache for ontology property lookups to avoid repeated dictionary traversals."""

    def __init__(self, parsed_ontology: Optional[Dict[str, Any]] = None):
        self._ontology = parsed_ontology or {}
        self._entity_defs: Dict[str, Dict[str, Any]] = {}
        self._property_defs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get_entity_def(self, entity_type: str) -> Dict[str, Any]:
        """Get entity definition, cached."""
        if entity_type not in self._entity_defs:
            self._entity_defs[entity_type] = self._ontology.get("entities", {}).get(
                entity_type, {}
            )
        return self._entity_defs[entity_type]

    def get_property_def(self, entity_type: str, prop_name: str) -> Dict[str, Any]:
        """Get property definition for an entity type, cached."""
        key = (entity_type, prop_name)
        if key not in self._property_defs:
            entity_def = self.get_entity_def(entity_type)
            self._property_defs[key] = entity_def.get("properties", {}).get(
                prop_name, {}
            )
        return self._property_defs[key]

    def get_property_type(self, entity_type: str, prop_name: str) -> Optional[str]:
        """Get property type for an entity type, cached."""
        prop_def = self.get_property_def(entity_type, prop_name)
        return prop_def.get("type") if isinstance(prop_def, dict) else None

    def is_property_unique(self, entity_type: str, prop_name: str) -> bool:
        """Check if property is marked as unique, cached."""
        prop_def = self.get_property_def(entity_type, prop_name)
        return bool(prop_def.get("unique")) if isinstance(prop_def, dict) else False

    def is_property_indexed(self, entity_type: str, prop_name: str) -> bool:
        """Check if property is marked as index, cached."""
        prop_def = self.get_property_def(entity_type, prop_name)
        return bool(prop_def.get("index")) if isinstance(prop_def, dict) else False


@dataclass(frozen=True)
class ComparisonPrior:
    m: Tuple[float, ...]
    u: Tuple[float, ...]


UNIQUE_PRIOR = ComparisonPrior(m=(0.97, 0.03), u=(0.02, 0.98))
INDEX_PRIOR = ComparisonPrior(m=(0.92, 0.08), u=(0.08, 0.92))
NUMERIC_PRIOR = ComparisonPrior(m=(0.9, 0.1), u=(0.05, 0.95))
DATETIME_PRIOR = ComparisonPrior(m=(0.9, 0.1), u=(0.05, 0.95))
STRING_PRIOR = ComparisonPrior(m=(0.85, 0.1, 0.04, 0.01), u=(0.05, 0.1, 0.15, 0.7))
FALLBACK_EXACT_PRIOR = ComparisonPrior(m=(0.82, 0.18), u=(0.12, 0.88))

Canonicalizer = Callable[[Any, Dict[str, Any]], Optional[str]]
_CANONICALIZER_REGISTRY: Dict[str, Canonicalizer] = {}


def register_canonicalizer(property_key: str, canonicalizer: Canonicalizer) -> None:
    """Register a custom canonicalizer. Keys can be 'property' or 'Entity.property'."""

    _CANONICALIZER_REGISTRY[property_key.lower()] = canonicalizer


def _exact_match(column: str):
    match_cls = getattr(cl, "ExactMatch", None)
    if match_cls:
        return match_cls(column)
    return cl.LevenshteinAtThresholds(column, [0])


def _with_prior(comparison, prior: ComparisonPrior):
    """Apply heuristic m/u priors to a comparison creator."""

    try:
        non_null_levels = comparison.num_non_null_levels
    except AttributeError:
        return comparison

    if non_null_levels <= 0:
        return comparison

    m_values = list(prior.m[:non_null_levels])
    u_values = list(prior.u[:non_null_levels])

    if len(m_values) == non_null_levels:
        comparison.configure(m_probabilities=m_values)

    if len(u_values) == non_null_levels:
        comparison.configure(u_probabilities=u_values)

    return comparison


COMPANY_SUFFIXES = (
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "plc",
    "gmbh",
)


def _canonicalize_whitespace(value: str) -> str:
    value = value.strip()
    return re.sub(r"\s+", " ", value)


def _canonicalize_company_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", value.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    while words and words[-1] in COMPANY_SUFFIXES:
        words.pop()
    return " ".join(words)


def _basic_canonical_value(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        return _canonicalize_whitespace(value).casefold()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        parts = []
        for item in value:
            canonical_part = _basic_canonical_value(item)
            if canonical_part:
                parts.append(canonical_part)
        return "|".join(parts) if parts else None

    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)

    return str(value)


# A1-prov: cap source-text excerpts written to node/edge properties.
# Chunk text can be 32 KB; embedding it on every node bloats the graph
# payload. The Evidence tab and MCP get_evidence show a quoted span,
# so a 1000-char excerpt is sufficient. Full text remains addressable
# via source_chunk_id when a downstream tool needs it.
_SOURCE_TEXT_PROPERTY_LIMIT = 1000


def _validate_per_fact_excerpt(
    excerpt: Optional[str],
    source_text: Optional[str],
) -> bool:
    """Verify that an LLM-emitted ``source_excerpt`` is actually a
    substring of the source text the model was given.

    Gate 4 follow-up (slice 2): the prompt instructs the LLM to
    emit a verbatim quote, but models occasionally fabricate (or
    paraphrase) — a hallucinated quote on the Evidence tab is worse
    than no quote, since users trust what they read. Validating the
    quote is in the source catches this cheaply.

    Normalizes whitespace and case before comparing — LLMs often
    emit minor differences (newlines vs spaces, multiple spaces
    collapsed, capitalisation drift especially after OCR'd PDFs)
    that aren't fabrication, just lossy round-trip. The substring
    check is intentionally lenient.

    Returns True (validation passes) in two cases:
      1. The normalized excerpt is a substring of the normalized
         source — the LLM quoted faithfully (modulo whitespace).
      2. ``source_text`` is None — the storage layer doesn't have
         access to what the LLM saw (typically PDF-binary path
         with splits too large for the truncation budget). We
         trust the LLM rather than reject all excerpts in those
         cases. ``transform_as_nodes`` logs the validation-skipped
         count separately so operators can see how often this
         happens.

    Returns False when there's source text and the excerpt isn't
    in it — definite hallucination, fall back to chunk-level.
    """
    if not excerpt or not excerpt.strip():
        return False
    if source_text is None:
        return True
    normalized_excerpt = " ".join(excerpt.lower().split())
    normalized_source = " ".join(source_text.lower().split())
    return normalized_excerpt in normalized_source


def _attach_provenance_properties(
    node_or_edge: Union[BaseNode, RelationshipInstance],
    chunk_metadata: Optional[ChunkMetadata] = None,
    chunk_text: Optional[str] = None,
    document_id: Optional[str] = None,
    per_fact_excerpt: Optional[str] = None,
) -> None:
    """Copy provenance fields from a chunk into a node or edge's properties.

    Mirrors ``graphora_server/mcp/server.py::_EVIDENCE_KEYS`` so the
    Explorer Evidence tab and the MCP ``get_evidence`` tool find the
    same set of fields. ``setdefault`` semantics throughout — if an
    LLM extractor already emitted (say) ``document_id``, that value
    wins; we never clobber.

    Gate 4 (per-fact provenance): ``per_fact_excerpt`` is the LLM-
    emitted 1-2 sentence verbatim quote for THIS specific entity /
    relationship. When non-empty, it takes priority over the chunk-
    level fallbacks for ``source_text`` — entities from page 60 of
    a 100-page split get evidence pointing at their sentence,
    rather than the chunk's first 1000 chars (the long-standing
    truncation bias). Falls back to ``chunk_text`` then
    ``chunk_metadata.source_text`` when the LLM omits it (older
    ontologies, model regressions, etc.).

    No-op when none of the optional inputs are set; extraction still
    succeeds and just emits an empty Evidence tab for that node.
    """
    props = node_or_edge.properties
    if chunk_metadata is not None:
        if chunk_metadata.chunk_id:
            props.setdefault("source_chunk_id", chunk_metadata.chunk_id)
        if chunk_metadata.source_file:
            props.setdefault("document_name", chunk_metadata.source_file)
        if chunk_metadata.page_number is not None:
            props.setdefault("page_number", chunk_metadata.page_number)
        # ``start_position`` is always set on ChunkMetadata (default 0)
        # — write it only when non-zero so we don't pollute every node
        # with an uninformative ``chunk_offset: 0``.
        if chunk_metadata.start_position:
            props.setdefault("chunk_offset", chunk_metadata.start_position)

    # source_text resolution priority (highest to lowest):
    #   1. per_fact_excerpt: LLM-emitted Gate-4 per-fact quote.
    #      Replaces chunk-level fallbacks because it points at the
    #      ACTUAL fact's sentence, not the chunk's first ~1000 chars.
    #   2. chunk_text: explicit text-chunk path (caller pre-resolved).
    #   3. chunk_metadata.source_text: PDF-binary path (flows.py
    #      pre-extracted an excerpt at split time, gated on parse
    #      size — see flows._resolve_evidence_source_text).
    #
    # Whitespace-only excerpts are treated as empty so a model that
    # emits "" or " " falls through to the chunk-level fallback
    # rather than wiping out useful evidence.
    if per_fact_excerpt and per_fact_excerpt.strip():
        text_to_write = per_fact_excerpt
    else:
        text_to_write = chunk_text or (
            chunk_metadata.source_text if chunk_metadata else None
        )
    if text_to_write:
        truncated = text_to_write[:_SOURCE_TEXT_PROPERTY_LIMIT]
        props.setdefault("source_text", truncated)

    # Resolve document_id with fallback to the chunk's source_file.
    # In Graphora today the safe_filename written by the upload
    # endpoint is the natural per-file identifier (api/transform.py
    # passes metadata.source, which is the same string the chunker
    # later sees as Path(file_path).name → ChunkMetadata.source_file).
    # Auto-deriving here means every callsite doesn't have to thread
    # an extra parameter just to repeat that fact.
    resolved_doc_id = document_id or (
        chunk_metadata.source_file if chunk_metadata else None
    )
    if resolved_doc_id:
        props.setdefault("document_id", resolved_doc_id)

    prov = getattr(node_or_edge, "provenance", None)
    if prov is not None:
        if prov.confidence_score is not None:
            props.setdefault("extraction_confidence", prov.confidence_score)
        # B0-prov-extend: surface the decision-trail fields onto
        # properties so the Explorer Evidence tab + MCP get_evidence
        # find them via the same _EVIDENCE_KEYS contract used for
        # source-span fields.
        if prov.extractor_model is not None:
            props.setdefault("extractor_model", prov.extractor_model)
        if prov.prompt_version is not None:
            props.setdefault("prompt_version", prov.prompt_version)
        if prov.validator_score is not None:
            props.setdefault("validator_score", prov.validator_score)


def _get_registered_canonicalizer(
    entity_type: str, prop_name: str
) -> Optional[Canonicalizer]:
    key_exact = f"{entity_type}.{prop_name}".lower()
    key_generic = prop_name.lower()
    return _CANONICALIZER_REGISTRY.get(key_exact) or _CANONICALIZER_REGISTRY.get(
        key_generic
    )


def _canonicalize_value(
    entity_type: str,
    prop_name: str,
    value: Any,
    prop_def: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if value is None:
        return None

    prop_def = prop_def or {}

    if not settings.ENTITY_CANONICALIZATION_ENABLED:
        return _basic_canonical_value(value)

    # Custom canonicalizer takes precedence
    custom = _get_registered_canonicalizer(entity_type, prop_name)
    if custom:
        try:
            return custom(value, prop_def)
        except Exception as exc:  # pragma: no cover - defensive log
            logging.warning(
                "Custom canonicalizer failed for %s.%s: %s",
                entity_type,
                prop_name,
                exc,
            )

    canonicalization_cfg = prop_def.get("canonicalization", {})

    if isinstance(value, str):
        string_value = _canonicalize_whitespace(value)
        preserve_case = canonicalization_cfg.get("preserve_case", False)
        working_value = string_value if preserve_case else string_value.casefold()

        if canonicalization_cfg.get("strip_punctuation"):
            working_value = re.sub(r"[^\w\s]", " ", working_value)
            working_value = re.sub(r"\s+", " ", working_value).strip()

        if canonicalization_cfg.get("remove_non_alnum"):
            working_value = re.sub(r"[^0-9a-zA-Z]+", " ", working_value).strip()

        suffixes = canonicalization_cfg.get("strip_suffixes") or []
        suffixes = [str(s).casefold() for s in suffixes]
        if canonicalization_cfg.get("strip_company_suffixes"):
            suffixes = suffixes or list(COMPANY_SUFFIXES)

        if suffixes:
            words = working_value.split()
            while words and words[-1] in suffixes:
                words.pop()
            working_value = " ".join(words)

        if _is_prop_type_datetime(prop_def.get("type", "")):
            return working_value

        return re.sub(r"\s+", " ", working_value).strip()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        parts = []
        for item in value:
            canonical_part = _canonicalize_value(entity_type, prop_name, item, prop_def)
            if canonical_part:
                parts.append(canonical_part)
        return "|".join(parts) if parts else None

    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)

    return str(value)


def _build_canonical_properties(
    parsed_ontology: Dict[str, Any],
    entity_type: str,
    properties: Dict[str, Any],
    raw_properties: Dict[str, Any],
) -> Dict[str, Any]:
    entity_def = parsed_ontology.get("entities", {}).get(entity_type, {})
    property_defs = entity_def.get("properties", {})
    canonical: Dict[str, Any] = {}

    for prop_name, display_value in properties.items():
        prop_def = property_defs.get(prop_name, {})
        raw_value = raw_properties.get(prop_name, display_value)
        canonical_value = _canonicalize_value(
            entity_type, prop_name, raw_value, prop_def
        )
        if canonical_value is None:
            canonical_value = _canonicalize_value(
                entity_type, prop_name, display_value, prop_def
            )
        canonical[prop_name] = canonical_value

    for prop_name, prop_def in property_defs.items():
        if prop_name in canonical:
            continue
        if prop_def.get("unique"):
            raw_value = raw_properties.get(prop_name)
            canonical_value = _canonicalize_value(
                entity_type, prop_name, raw_value, prop_def
            )
            if canonical_value:
                canonical[prop_name] = canonical_value

    return canonical


def transform_as_nodes(
    ontology: Dict[str, Any],
    entity_result: BaseModel,
    transform_id: Optional[str] = None,
    *,
    chunk_metadata: Optional[ChunkMetadata] = None,
    chunk_text: Optional[str] = None,
    document_id: Optional[str] = None,
    extractor_model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    validator_score: Optional[float] = None,
) -> List[BaseNode]:
    """Build BaseNode objects from a BAML extraction result.

    Keyword args:
      ``chunk_metadata`` / ``chunk_text`` / ``document_id``
        A1-prov plumbing — source-span properties on every emitted
        node.
      ``extractor_model`` / ``prompt_version`` / ``validator_score``
        B0-prov-extend (Gate 4 entry) — decision-trail fields. When
        set, populated on NodeProvenance and on node properties via
        ``_attach_provenance_properties``.

    All defaults are None so older callsites stay backward-compatible.
    """
    nodes = []
    chunk_node_registry = {}
    use_deterministic_ids = settings.DETERMINISTIC_MODE and bool(transform_id)

    # Gate 4 slice 2 telemetry: track per-fact excerpt outcomes so
    # operators can see how often the LLM emits a useful excerpt
    # and how often the model is fabricating quotes. Counters reset
    # per call; one INFO line emitted at function exit. Buckets are
    # mutually exclusive over the entities that produced a node.
    excerpt_telemetry = {
        "total_nodes": 0,
        "with_excerpt": 0,
        "validated": 0,
        "rejected": 0,
        "skip_no_source": 0,
    }

    # The source-of-truth text the LLM saw (text-chunk path) or the
    # split-level Markdown if the PDF-binary path managed to capture
    # it (otherwise None). Used as the ground truth for excerpt
    # validation. None disables validation — see
    # _validate_per_fact_excerpt's docstring for why.
    validation_source = chunk_text or (
        chunk_metadata.source_text if chunk_metadata else None
    )

    # Process entities
    for field_name in dir(entity_result):
        if not field_name.endswith("_list") or field_name.startswith("_"):
            continue
        entity_list = getattr(entity_result, field_name)
        if not isinstance(entity_list, list):
            continue
        entity_type = field_name[:-5]
        chunk_node_registry[entity_type] = {}
        for item_index, item in enumerate(entity_list):
            if not item:
                continue
            raw_properties = _extract_properties(item)
            properties = _normalize_entity_properties(
                ontology, entity_type, raw_properties
            )
            if properties is None:
                logger.debug(
                    "Skipping %s node due to ontology validation failure",
                    entity_type,
                )
                continue
            # Capture the LLM-side positional id BEFORE we mint our
            # own UUID. Gemini's extract_relationships_from_pdf
            # references entities by these positional ids
            # ('company_0', 'business_0', ...) instead of honoring
            # the UUIDs we put in the context block. Stamping the
            # original id on the node lets transform_as_relationships
            # bridge the gap. Defensive: only keep non-empty strings
            # — Gemini has been known to emit ints / nulls here.
            #
            # Stored as a list because merge_nodes unions aliases
            # across merges (see BaseNode.original_extraction_ids
            # docstring). At extraction time we have at most one
            # alias per node; the list grows as entity resolution
            # merges duplicates downstream.
            raw_extraction_id = raw_properties.get("id")
            original_extraction_ids = (
                [raw_extraction_id]
                if isinstance(raw_extraction_id, str) and raw_extraction_id
                else []
            )
            fallback_hint = f"{entity_type}:{item_index}"
            canonical_properties = _build_canonical_properties(
                ontology,
                entity_type,
                properties,
                raw_properties,
            )
            node_key = _generate_node_key(
                ontology,
                entity_type,
                properties,
                canonical_properties=canonical_properties,
                raw_properties=raw_properties,
                fallback_hint=fallback_hint,
            )
            canonical_id = _make_canonical_node_id(node_key)
            node_id = (
                _make_deterministic_node_id(transform_id, entity_type, node_key)
                if use_deterministic_ids
                else str(uuid.uuid4())
            )
            node = BaseNode(
                id=node_id,
                type=entity_type,
                properties=properties,
                canonical_properties=canonical_properties,
                canonical_key=node_key,
                canonical_id=canonical_id,
                original_extraction_ids=original_extraction_ids,
                provenance=NodeProvenance(
                    chunk_ids=(
                        [chunk_metadata.chunk_id]
                        if chunk_metadata and chunk_metadata.chunk_id
                        else [node_id]
                    ),
                    extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                    confidence_score=entity_result.confidence_score,
                    source_file=(
                        chunk_metadata.source_file if chunk_metadata else None
                    ),
                    page_number=(
                        chunk_metadata.page_number if chunk_metadata else None
                    ),
                    char_offset=(
                        chunk_metadata.start_position
                        if chunk_metadata and chunk_metadata.start_position
                        else None
                    ),
                    extractor_model=extractor_model,
                    prompt_version=prompt_version,
                    validator_score=validator_score,
                ),
            )
            # Gate 4 per-fact excerpt: the LLM emits source_excerpt
            # alongside the entity properties. Read it from
            # raw_properties (the unfiltered dict — _normalize_entity_
            # properties drops anything not in the ontology, which
            # includes source_excerpt). When set AND validated as a
            # substring of the source the model saw, it lands on
            # node.properties["source_text"] via the helper, replacing
            # the chunk-level fallback. Hallucinated quotes
            # (excerpts not in the source) are dropped — fall back
            # to chunk-level rather than risk showing fabricated
            # evidence to users.
            excerpt_telemetry["total_nodes"] += 1
            raw_excerpt = raw_properties.get("source_excerpt")
            candidate_excerpt = (
                raw_excerpt
                if isinstance(raw_excerpt, str) and raw_excerpt.strip()
                else None
            )
            per_fact_excerpt: Optional[str] = None
            if candidate_excerpt is not None:
                excerpt_telemetry["with_excerpt"] += 1
                if validation_source is None:
                    excerpt_telemetry["skip_no_source"] += 1
                    per_fact_excerpt = candidate_excerpt
                elif _validate_per_fact_excerpt(candidate_excerpt, validation_source):
                    excerpt_telemetry["validated"] += 1
                    per_fact_excerpt = candidate_excerpt
                else:
                    excerpt_telemetry["rejected"] += 1
                    logger.debug(
                        "Rejected per-fact excerpt for %s/%s — not a substring "
                        "of source: %r",
                        entity_type,
                        node_id,
                        candidate_excerpt[:120],
                    )
                    per_fact_excerpt = None  # fall back to chunk-level
            _attach_provenance_properties(
                node,
                chunk_metadata=chunk_metadata,
                chunk_text=chunk_text,
                document_id=document_id,
                per_fact_excerpt=per_fact_excerpt,
            )
            chunk_node_registry[entity_type][node_key] = node_id
            nodes.append(node)

    if excerpt_telemetry["total_nodes"] > 0:
        # One INFO line per transform_as_nodes call. Operators
        # tracking Gate-4 quality use these counters via
        # log-aggregation pipelines (Loki/CloudWatch); the
        # structured ``extra`` keys are the queryable surface,
        # the human-readable message is for tail-watching.
        logger.info(
            "Per-fact excerpt: %d/%d entities, %d validated, %d rejected, "
            "%d skip-no-source",
            excerpt_telemetry["with_excerpt"],
            excerpt_telemetry["total_nodes"],
            excerpt_telemetry["validated"],
            excerpt_telemetry["rejected"],
            excerpt_telemetry["skip_no_source"],
            extra={
                "extraction_total_nodes": excerpt_telemetry["total_nodes"],
                "extraction_with_excerpt": excerpt_telemetry["with_excerpt"],
                "extraction_excerpt_validated": excerpt_telemetry["validated"],
                "extraction_excerpt_rejected": excerpt_telemetry["rejected"],
                "extraction_excerpt_skip_no_source": (
                    excerpt_telemetry["skip_no_source"]
                ),
            },
        )
    return nodes


def transform_as_relationships(
    ontology: Dict[str, Any],
    nodes: List[BaseNode],
    relationship_result: BaseModel,
    *,
    chunk_metadata: Optional[ChunkMetadata] = None,
    chunk_text: Optional[str] = None,
    document_id: Optional[str] = None,
    extractor_model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    validator_score: Optional[float] = None,
) -> List[RelationshipInstance]:
    """Build RelationshipInstance objects from a BAML extraction.

    A1-prov keyword args mirror ``transform_as_nodes``: when set,
    every emitted edge gets source-span properties. B0-prov-extend
    adds the decision-trail kwargs (extractor_model, prompt_version,
    validator_score). Defaults stay backward-compatible.
    """
    logger.debug("%s", "#" * 30)
    logger.debug("relationship_result: %s", relationship_result)
    logger.debug("nodes: %s", nodes)
    logger.debug("%s", "#" * 30)
    relationships = []

    # Gate 4 slice 2 telemetry — same shape as transform_as_nodes.
    excerpt_telemetry = {
        "total_rels": 0,
        "with_excerpt": 0,
        "validated": 0,
        "rejected": 0,
        "skip_no_source": 0,
    }
    validation_source = chunk_text or (
        chunk_metadata.source_text if chunk_metadata else None
    )

    node_by_id = {node.id: node for node in nodes}
    node_by_canonical_id = {
        node.canonical_id: node for node in nodes if node.canonical_id
    }
    node_by_canonical_key = {
        node.canonical_key: node for node in nodes if node.canonical_key
    }

    # Original-extraction-id fallback. Gemini's
    # extract_relationships_from_pdf typically emits positional ids
    # ('company_0', 'business_0') instead of the UUIDs we feed via
    # the relationships-context block. We resolve those by matching
    # against the per-node ``original_extraction_id`` captured during
    # transform_as_nodes.
    #
    # Scoping: positional ids are NOT unique across chunks
    # (chunk-1's 'company_0' might be Apple while chunk-2's
    # 'company_0' is Microsoft). When the caller passed
    # ``chunk_metadata`` we prefer a node whose provenance includes
    # this chunk_id, falling through to a global match only if no
    # in-chunk node carries that original id. The unscoped fallback
    # preserves behaviour for callsites that don't thread chunk
    # metadata through.
    chunk_id_for_scope = (
        chunk_metadata.chunk_id if chunk_metadata and chunk_metadata.chunk_id else None
    )

    def _find_by_original_id(raw_id: Optional[str]) -> Optional[BaseNode]:
        if not raw_id:
            return None
        scoped_match: Optional[BaseNode] = None
        unscoped_match: Optional[BaseNode] = None
        for n in nodes:
            # Membership rather than equality: merge_nodes unions
            # aliases across merged-away duplicates, so a node may
            # carry several positional ids (one per chunk where the
            # same logical entity was extracted under a different
            # `<type>_<n>` name).
            ids = n.original_extraction_ids or []
            if raw_id not in ids:
                continue
            if unscoped_match is None:
                unscoped_match = n
            if (
                chunk_id_for_scope
                and n.provenance
                and chunk_id_for_scope in (n.provenance.chunk_ids or [])
            ):
                scoped_match = n
                break  # in-chunk match always wins
        return scoped_match or unscoped_match

    for field_name in dir(relationship_result):
        if (
            field_name.endswith("_list")
            or field_name.startswith("_")
            or "_" not in field_name
        ):
            continue
        rel_list = getattr(relationship_result, field_name)
        if not isinstance(rel_list, list):
            continue
        # Flexible parsing of relationship keys
        parts = field_name.split("_")
        if len(parts) < 2:
            continue

        # Try to extract source_type, rel_type, and target_type
        source_type = parts[0]
        if source_type not in ontology.get("entities", {}):
            continue

        # Handle cases where target_type might be missing or concatenated
        rel_type_parts = parts[1:-1] if len(parts) > 2 else parts[1:]
        rel_type = (
            "_".join(rel_type_parts) if len(rel_type_parts) > 1 else rel_type_parts[0]
        )
        target_type = parts[-1] if len(parts) > 2 else None

        # Infer target_type from ontology if not in field_name
        relationships_def = (
            ontology["entities"].get(source_type, {}).get("relationships", {})
        )
        logger.info("relationships_def: %s", relationships_def)
        if rel_type in relationships_def:
            target_type = target_type or relationships_def[rel_type].get("target")
        else:
            logger.warning(
                f"Skipping unknown relationship type: {rel_type} for {source_type}"
            )
            continue

        if not target_type or target_type not in ontology.get("entities", {}):
            logger.warning(
                f"Skipping relationship {field_name}: Could not determine valid target_type"
            )
            continue
        logger.info("rel_list: %s", rel_list)
        for rel_item in rel_list:
            if not rel_item:
                continue
            raw_source_id = getattr(rel_item, "source_id", None)
            raw_target_id = getattr(rel_item, "target_id", None)

            if not raw_source_id or not raw_target_id:
                logger.warning(
                    f"Skipping relationship {rel_type}: Missing source_id or target_id"
                )
                continue

            source_node = node_by_id.get(raw_source_id)
            if not source_node:
                source_node = node_by_canonical_id.get(
                    raw_source_id
                ) or node_by_canonical_key.get(raw_source_id)
            if not source_node:
                source_node = _find_by_original_id(raw_source_id)
            target_node = node_by_id.get(raw_target_id)
            if not target_node:
                target_node = node_by_canonical_id.get(
                    raw_target_id
                ) or node_by_canonical_key.get(raw_target_id)
            if not target_node:
                target_node = _find_by_original_id(raw_target_id)

            if not source_node or not target_node:
                logger.warning(
                    f"Skipping relationship {rel_type}: Invalid source_id {raw_source_id} or target_id {raw_target_id}"
                )
                continue

            source_id = source_node.id
            target_id = target_node.id

            raw_properties = _extract_properties(getattr(rel_item, "properties", {}))
            rel_properties = _normalize_relationship_properties(
                ontology, source_type, rel_type, raw_properties
            )
            if rel_properties is None:
                logger.debug(
                    "Skipping relationship %s due to ontology validation failure",
                    rel_type,
                )
                continue
            rel_id = str(uuid.uuid4())
            rel = RelationshipInstance(
                id=rel_id,
                type=rel_type,
                source_id=source_id,
                target_id=target_id,
                source_type=source_type,
                target_type=target_type,
                properties=rel_properties,
                provenance=NodeProvenance(
                    chunk_ids=(
                        [chunk_metadata.chunk_id]
                        if chunk_metadata and chunk_metadata.chunk_id
                        else [rel_id]
                    ),
                    extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                    confidence_score=relationship_result.confidence_score,
                    source_file=(
                        chunk_metadata.source_file if chunk_metadata else None
                    ),
                    page_number=(
                        chunk_metadata.page_number if chunk_metadata else None
                    ),
                    char_offset=(
                        chunk_metadata.start_position
                        if chunk_metadata and chunk_metadata.start_position
                        else None
                    ),
                    extractor_model=extractor_model,
                    prompt_version=prompt_version,
                    validator_score=validator_score,
                ),
            )
            # Gate 4 per-fact excerpt for relationships. Lives
            # directly on the rel_model (next to source_id /
            # target_id) — not inside the nested properties bag —
            # so we read it from rel_item.source_excerpt rather
            # than rel_item.properties.source_excerpt. Validation +
            # telemetry mirrors the entity side; see
            # transform_as_nodes for the rationale.
            excerpt_telemetry["total_rels"] += 1
            raw_rel_excerpt = getattr(rel_item, "source_excerpt", None)
            candidate_rel_excerpt = (
                raw_rel_excerpt
                if isinstance(raw_rel_excerpt, str) and raw_rel_excerpt.strip()
                else None
            )
            per_fact_rel_excerpt: Optional[str] = None
            if candidate_rel_excerpt is not None:
                excerpt_telemetry["with_excerpt"] += 1
                if validation_source is None:
                    excerpt_telemetry["skip_no_source"] += 1
                    per_fact_rel_excerpt = candidate_rel_excerpt
                elif _validate_per_fact_excerpt(
                    candidate_rel_excerpt, validation_source
                ):
                    excerpt_telemetry["validated"] += 1
                    per_fact_rel_excerpt = candidate_rel_excerpt
                else:
                    excerpt_telemetry["rejected"] += 1
                    logger.debug(
                        "Rejected per-fact excerpt for %s relationship — "
                        "not a substring of source: %r",
                        rel_type,
                        candidate_rel_excerpt[:120],
                    )
                    per_fact_rel_excerpt = None
            _attach_provenance_properties(
                rel,
                chunk_metadata=chunk_metadata,
                chunk_text=chunk_text,
                document_id=document_id,
                per_fact_excerpt=per_fact_rel_excerpt,
            )
            relationships.append(rel)

    if excerpt_telemetry["total_rels"] > 0:
        logger.info(
            "Per-fact excerpt (rels): %d/%d relationships, %d validated, "
            "%d rejected, %d skip-no-source",
            excerpt_telemetry["with_excerpt"],
            excerpt_telemetry["total_rels"],
            excerpt_telemetry["validated"],
            excerpt_telemetry["rejected"],
            excerpt_telemetry["skip_no_source"],
            extra={
                "extraction_total_rels": excerpt_telemetry["total_rels"],
                "extraction_rel_with_excerpt": excerpt_telemetry["with_excerpt"],
                "extraction_rel_excerpt_validated": (excerpt_telemetry["validated"]),
                "extraction_rel_excerpt_rejected": (excerpt_telemetry["rejected"]),
                "extraction_rel_excerpt_skip_no_source": (
                    excerpt_telemetry["skip_no_source"]
                ),
            },
        )

    return relationships


async def cluster_entities_with_splink(
    entities: List[BaseNode | Node],
    parsed_ontology: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    threshold: float = 0.95,
    use_embedding_similarity: Optional[bool] = None,
) -> Dict[str, str]:
    """Splink probabilistic-linkage clustering only — does NOT
    merge entities or rewrite relationships.

    Returns ``id_to_representative`` mapping for entities Splink
    decided are duplicates of one another above ``threshold``.
    Singleton entities (and their identity self-mappings) are
    excluded from the result.

    Used by B2-er slice 3's ``_splink_candidate_groups`` in
    graph_transformer.py to feed Splink-discovered candidate
    pairs into the ``resolve_entity_group`` LLM stage inside
    ``_compare_and_merge_nodes``. The slice-1 property blocker
    already covers the ``_deduplicate_small_entity_group``
    heuristic ground, so this function deliberately skips
    heuristics — slice 3 only sees nodes both earlier blockers
    missed, where Splink's m/u-probability scoring is the
    relevant signal.

    Resilient on every standard Splink early-exit branch (no
    comparison columns, no comparisons, no blocking rules, no
    duplicates found): returns ``{}`` and the caller treats
    those nodes as unblocked. Top-level exceptions are logged
    and swallowed so a Splink failure can't break extraction.
    """
    if use_embedding_similarity is None:
        use_embedding_similarity = settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED

    if not entities or len(entities) < 2:
        return {}

    entities_by_type: Dict[str, List[BaseNode | Node]] = {}
    for e in entities:
        if not hasattr(e, "type"):
            continue
        entities_by_type.setdefault(e.type, []).append(e)

    all_mappings: Dict[str, str] = {}

    try:
        for current_type, type_entities in entities_by_type.items():
            if len(type_entities) < 3:
                continue

            effective_threshold = await merge_learning_service.get_threshold(
                user_id, current_type, threshold
            )

            entities_data = _prepare_entities_for_deduplication(
                type_entities, None, parsed_ontology
            )

            entity_def = (
                (parsed_ontology or {}).get("entities", {}).get(current_type, {})
            )
            property_defs = entity_def.get("properties", {}) if entity_def else {}
            allowed_properties = set(property_defs.keys()) if property_defs else None
            if allowed_properties is not None and len(allowed_properties) == 0:
                allowed_properties = None

            df, comparison_columns = _create_splink_dataframe(
                entities_data,
                SYSTEM_PROPERTIES,
                allowed_properties=allowed_properties,
            )
            if not comparison_columns:
                continue

            comparisons, text_columns = _create_splink_comparisons(
                comparison_columns,
                df,
                len(df),
                current_type,
                parsed_ontology,
            )

            if use_embedding_similarity and text_columns:
                try:
                    from graphora_server.services.entity_resolution.splink_embedding_comparison import (
                        EmbeddingAwareComparisonFactory,
                    )

                    embedding_factory = EmbeddingAwareComparisonFactory()
                    df, _ = embedding_factory.prepare_dataframe_with_embeddings(
                        df, text_columns
                    )
                except ImportError as exc:
                    logging.warning(
                        "Embedding similarity not available for clustering: %s",
                        str(exc),
                    )
                except Exception as exc:
                    logging.warning(
                        "Embedding signature step failed for clustering: %s",
                        str(exc),
                    )

            if not comparisons:
                continue

            blocking_rules = _create_blocking_rules(
                comparison_columns,
                df,
                len(df),
                current_type,
                parsed_ontology,
            )
            if not blocking_rules:
                continue

            id_to_representative, _match_scores = _run_splink_deduplication_batched(
                entities_data, df, comparisons, blocking_rules, effective_threshold
            )

            # Deliberately do NOT call merge_learning_service.record_outcome
            # here. This function is a candidate-pair generator for slice 3 —
            # the scores Splink emits are speculative until the LLM stage
            # confirms the merge. Recording them now would
            #   (a) pollute the adaptive threshold with pairs the LLM later
            #       rejects (false positives become "observed outcomes"), and
            #   (b) double-record on every transform, since the post-
            #       relationship deduplicate_entities_with_splink call already
            #       records once on the real (post-LLM, post-relationship-
            #       extraction) population.
            # The real dedup path remains the single source of learning truth.

            if not id_to_representative:
                continue

            for entity_id, rep_id in id_to_representative.items():
                if entity_id != rep_id:
                    all_mappings[entity_id] = rep_id
    except Exception as exc:
        logging.error("Splink clustering failed: %s", str(exc))
        logging.debug("Splink clustering details: %s", traceback.format_exc())
        return {}

    return all_mappings


async def deduplicate_entities_with_splink(
    entities: List[BaseNode | Node],
    relationships: List[RelationshipInstance | Edge] = None,
    entity_type: str = None,
    threshold: float = 0.95,
    parsed_ontology: Dict[str, Any] = None,
    user_id: Optional[str] = None,
    use_embedding_similarity: bool = None,
) -> Tuple[List[BaseNode | Node], List[RelationshipInstance | Edge]]:
    """
    Deduplicate entities using the Splink library.

    Args:
        entities (List[BaseNode]): List of entity nodes to deduplicate
        relationships (List[RelationshipInstance], optional): List of relationships to provide context
        entity_type (str, optional): Type of entity to filter by. If None, all entity types are processed separately.
        threshold (float, optional): Baseline match probability threshold for clustering. Default is 0.95.
        user_id (str, optional): User identifier to scope adaptive thresholds.
        parsed_ontology (dict, optional): Parsed ontology to get property types
        use_embedding_similarity (bool, optional): Enable embedding similarity for TEXT properties.
            Defaults to settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED.

    Returns:
        Tuple[List[BaseNode], List[RelationshipInstance]]: Deduplicated list of entities and updated relationships
    """
    # Resolve embedding similarity flag from settings if not provided
    if use_embedding_similarity is None:
        use_embedding_similarity = settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED
    try:
        logging.info(f"Starting entity deduplication with {len(entities)} entities")
        logger.debug(f"Starting entity deduplication with {len(entities)} entities")
        if relationships:
            logging.info(f"Using {len(relationships)} relationships for context")

        # If no entities to process, return original list
        if not entities:
            return entities, relationships

        # Group entities by type
        entities_by_type = {}
        for e in entities:
            if not hasattr(e, "type"):
                continue

            if e.type not in entities_by_type:
                entities_by_type[e.type] = []
            entities_by_type[e.type].append(e)

        logging.info(
            f"Grouped entities into {len(entities_by_type)} types: {list(entities_by_type.keys())}"
        )

        # If specific entity type is requested, only process that type
        if entity_type and entity_type in entities_by_type:
            types_to_process = [entity_type]
        else:
            types_to_process = list(entities_by_type.keys())

        # Track which nodes were deduplicated (original ID -> representative ID)
        all_node_mappings = {}

        # Process each entity type separately
        all_deduplicated_entities = []
        for current_type in types_to_process:
            type_entities = entities_by_type[current_type]

            heuristic_entities, heuristic_mapping = _deduplicate_small_entity_group(
                current_type, type_entities, parsed_ontology
            )

            if heuristic_mapping:
                logging.info(
                    "Applied heuristic deduplication for type '%s': %d mappings",
                    current_type,
                    len(heuristic_mapping),
                )
                all_node_mappings.update(heuristic_mapping)

            type_entities = heuristic_entities

            if heuristic_mapping and len(type_entities) <= 3:
                logging.info(
                    "Heuristics resolved duplicates for type '%s'; skipping Splink",
                    current_type,
                )
                all_deduplicated_entities.extend(type_entities)
                continue

            if len(type_entities) < 2:
                all_deduplicated_entities.extend(type_entities)
                continue

            if len(type_entities) < 3:
                logging.info(
                    "Skipping Splink for type '%s' - %d entities after heuristics",
                    current_type,
                    len(type_entities),
                )
                all_deduplicated_entities.extend(type_entities)
                continue

            logging.info(
                f"Processing {len(type_entities)} entities of type '{current_type}'"
            )

            effective_threshold = await merge_learning_service.get_threshold(
                user_id, current_type, threshold
            )

            # Prepare entities data for this type
            entities_data = _prepare_entities_for_deduplication(
                type_entities, relationships, parsed_ontology
            )

            entity_def = (
                (parsed_ontology or {}).get("entities", {}).get(current_type, {})
            )
            property_defs = entity_def.get("properties", {}) if entity_def else {}
            allowed_properties = set(property_defs.keys()) if property_defs else None

            if allowed_properties is not None and len(allowed_properties) == 0:
                allowed_properties = None

            # Create DataFrame and prepare for Splink processing
            df, comparison_columns = _create_splink_dataframe(
                entities_data,
                SYSTEM_PROPERTIES,
                allowed_properties=allowed_properties,
            )

            if not comparison_columns:
                logging.warning(
                    f"No comparison columns found for type '{current_type}'"
                )
                all_deduplicated_entities.extend(type_entities)
                continue

            # Create comparisons and blocking rules
            comparisons, text_columns = _create_splink_comparisons(
                comparison_columns,
                df,
                len(df),
                current_type,
                parsed_ontology,
            )

            # Apply embedding similarity for TEXT columns if enabled
            if use_embedding_similarity and text_columns:
                try:
                    from graphora_server.services.entity_resolution.splink_embedding_comparison import (
                        EmbeddingAwareComparisonFactory,
                    )

                    embedding_factory = EmbeddingAwareComparisonFactory()
                    df, _ = embedding_factory.prepare_dataframe_with_embeddings(
                        df, text_columns
                    )
                    logging.info(
                        "Added embedding signatures for %d TEXT columns: %s",
                        len(text_columns),
                        text_columns,
                    )
                except ImportError as e:
                    logging.warning(
                        "Embedding similarity not available: %s. "
                        "Install sentence-transformers for semantic matching.",
                        str(e),
                    )
                except Exception as e:
                    logging.warning(
                        "Failed to compute embeddings for TEXT columns: %s",
                        str(e),
                    )

            if not comparisons:
                logging.warning(
                    f"No comparison rules could be created for type '{current_type}'"
                )
                all_deduplicated_entities.extend(type_entities)
                continue

            # Create blocking rules
            blocking_rules = _create_blocking_rules(
                comparison_columns,
                df,
                len(df),
                current_type,
                parsed_ontology,
            )

            if not blocking_rules:
                logging.warning(
                    f"No blocking rules could be created for type '{current_type}'"
                )
                all_deduplicated_entities.extend(type_entities)
                continue

            # Run Splink deduplication (batched for large datasets)
            logging.info(
                "Running Splink deduplication for type '%s' with threshold %.3f...",
                current_type,
                effective_threshold,
            )
            logger.debug(
                "Running Splink deduplication for type '%s' with adaptive threshold %.3f",
                current_type,
                effective_threshold,
            )
            id_to_representative, match_scores = _run_splink_deduplication_batched(
                entities_data, df, comparisons, blocking_rules, effective_threshold
            )

            if match_scores:
                await merge_learning_service.record_outcome(
                    user_id, current_type, match_scores
                )

            if not id_to_representative:
                logging.info(f"No duplicates found for type '{current_type}'")
                all_deduplicated_entities.extend(type_entities)
                continue

            # Log the resolved nodes (duplicates found)
            resolved_nodes = {}
            for entity_id, rep_id in id_to_representative.items():
                if entity_id != rep_id:  # Only log actual duplicates
                    if rep_id not in resolved_nodes:
                        resolved_nodes[rep_id] = []
                    resolved_nodes[rep_id].append(entity_id)

                    # Add to the global mapping
                    all_node_mappings[entity_id] = rep_id

            # Log the resolved nodes
            if resolved_nodes:
                logging.info(f"Resolved nodes for type '{current_type}':")
                for rep_id, duplicate_ids in resolved_nodes.items():
                    logger.debug(
                        f"  Representative {rep_id} <- duplicates: {duplicate_ids}"
                    )
                    logging.info(
                        f"  Representative {rep_id} <- duplicates: {duplicate_ids}"
                    )

            logging.info(
                f"Found {len(set(id_to_representative.values()))} clusters for type '{current_type}'"
            )

            # Create deduplicated entity list for this type
            type_deduplicated_entities = _create_deduplicated_entities(
                type_entities, id_to_representative
            )
            all_deduplicated_entities.extend(type_deduplicated_entities)

            logging.info(
                f"Reduced {len(type_entities)} entities to {len(type_deduplicated_entities)} for type '{current_type}'"
            )
            logger.debug(
                f"Reduced {len(type_entities)} entities to {len(type_deduplicated_entities)} for type '{current_type}'"
            )

        # Update relationships to use the representative node IDs
        updated_relationships = []
        if relationships and all_node_mappings:
            logging.info(
                f"Updating {len(relationships)} relationships with {len(all_node_mappings)} node mappings"
            )

            for rel in relationships:
                source_id = rel.source_id
                target_id = rel.target_id

                # Check if source or target nodes were deduplicated
                source_changed = source_id in all_node_mappings
                target_changed = target_id in all_node_mappings

                if source_changed or target_changed:
                    # Create a new relationship with updated IDs
                    new_source_id = all_node_mappings.get(source_id, source_id)
                    new_target_id = all_node_mappings.get(target_id, target_id)

                    # Skip self-relationships that might be created by deduplication
                    if new_source_id == new_target_id:
                        logging.info(
                            f"Skipping self-relationship: {rel.type} from {source_id} to {target_id}"
                        )
                        continue

                    # Create a new relationship with the updated IDs
                    new_rel = rel.copy()
                    new_rel.source_id = new_source_id
                    new_rel.target_id = new_target_id

                    logging.debug(
                        f"Updated relationship: {rel.type} from {source_id}->{new_source_id} to {target_id}->{new_target_id}"
                    )
                    updated_relationships.append(new_rel)
                else:
                    # Keep the original relationship
                    updated_relationships.append(rel)
        else:
            updated_relationships = relationships

        # Log summary of deduplication
        if all_node_mappings:
            logging.info(
                f"Deduplication summary: {len(all_node_mappings)} nodes mapped to representatives"
            )
            logging.info(
                f"Completed deduplication: {len(entities)} entities reduced to {len(all_deduplicated_entities)}"
            )
            original_relationship_count = len(relationships) if relationships else 0
            updated_relationship_count = (
                len(updated_relationships) if updated_relationships else 0
            )
            logging.info(
                "Relationships: %d original, %d after updating",
                original_relationship_count,
                updated_relationship_count,
            )
        else:
            logging.info("No duplicates found across any entity types")

        return all_deduplicated_entities, updated_relationships

    except Exception as e:
        logging.error(f"Error in entity deduplication: {str(e)}")
        logging.debug(f"Deduplication error details: {traceback.format_exc()}")
        traceback.print_exc()
        return entities, relationships


async def resolve_entity_group(
    entity_type: str,
    nodes: List[BaseNode],
    user_id: Optional[str] = None,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> List[List[BaseNode]]:
    """
    Use LLM to identify and group matching entities that should be merged.
    Returns list of groups, where each group contains matching nodes.
    """
    if len(nodes) <= 1:
        return [[nodes[0]]] if nodes else []
    llm_client = LLMClient()

    # Convert nodes to simple dict representation for LLM
    node_dicts = []
    for node in nodes:
        node_dict = {
            "id": node.id,
            "properties": node.properties,
            "confidence": node.confidence_score if node.confidence_score else 0.3,
        }
        node_dicts.append(node_dict)

    node_dicts_str = json.dumps(node_dicts, indent=2)

    try:
        # Call LLM for entity resolution
        logger.debug(f"Resolving `{entity_type}` entities")
        results = await llm_client.resolve_entities(
            entity_type=entity_type,
            node_dicts_str=node_dicts_str,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )

        if not results or len(results) == 0:
            # Fallback: treat each node as separate group
            return [[node] for node in nodes]

        # Convert index groups back to node groups
        resolved_groups = []
        node_map = {node.id: node for node in nodes}
        for result in results:
            if result.matching_ids is None or len(result.matching_ids) == 0:
                continue
            node_group = []
            for matching_node_id in result.matching_ids:
                if matching_node_id in node_map:
                    _node = node_map[matching_node_id]
                    _node.confidence_score = result.confidence_score
                    node_group.append(_node)
            if node_group:  # Only add non-empty groups
                resolved_groups.append(node_group)

                # Log explanation if available
                if result.explanation:
                    logger.info(
                        f"Entity resolution group {node_group[0].type}: {result.explanation}"
                    )

        # Check if any nodes were missed (safeguard)
        included_nodes = [node for group in resolved_groups for node in group]

        # Add missed nodes as single-node groups
        for node in nodes:
            if node not in included_nodes:
                resolved_groups.append([node])
        logger.debug("resolved_groups: %s", resolved_groups)
        return resolved_groups

    except Exception as e:
        traceback.print_exc()
        logger.warning(f"Error in LLM entity resolution: {str(e)}")
        # Fallback: treat each node as separate group
        return [[node] for node in nodes]


def merge_nodes(existing_node: BaseNode, new_node: BaseNode) -> BaseNode:
    """Merge properties and provenance from two nodes"""
    # Create a copy of the existing node
    merged_node = copy.deepcopy(existing_node)

    # Merge provenance
    if hasattr(new_node, "provenance") and new_node.provenance:
        if not hasattr(merged_node, "provenance") or not merged_node.provenance:
            merged_node.provenance = new_node.provenance
        else:
            # Combine chunk IDs
            merged_node.provenance.chunk_ids.extend(new_node.provenance.chunk_ids)
            merged_node.provenance.chunk_ids = list(
                set(merged_node.provenance.chunk_ids)
            )

    # Union original_extraction_ids — preserves the alias the
    # merged-away node carried so a later relationship lookup can
    # still resolve `source_id='company_1'` (chunk-2 alias) onto a
    # node merged from chunk-1's `company_0`. Without this, the
    # base node's single alias wins and the merged-away alias is
    # silently dropped — exactly the regression a reviewer would
    # catch with a chunk-1+chunk-2 merge fixture.
    if (
        hasattr(new_node, "original_extraction_ids")
        and new_node.original_extraction_ids
    ):
        if not getattr(merged_node, "original_extraction_ids", None):
            merged_node.original_extraction_ids = []
        seen = set(merged_node.original_extraction_ids)
        for alias in new_node.original_extraction_ids:
            if alias not in seen:
                merged_node.original_extraction_ids.append(alias)
                seen.add(alias)

    # Merge properties
    if hasattr(new_node, "properties") and new_node.properties:
        if not hasattr(merged_node, "properties") or not merged_node.properties:
            merged_node.properties = new_node.properties
        else:
            for key, value in new_node.properties.items():
                if value is not None:
                    # Take new value if not in existing or existing is None
                    if (
                        key not in merged_node.properties
                        or merged_node.properties[key] is None
                    ):
                        merged_node.properties[key] = value
                    # For overlapping values, take the one with higher confidence or longer value
                    elif (
                        hasattr(new_node, "confidence_score")
                        and hasattr(existing_node, "confidence_score")
                        and new_node.confidence_score
                        and existing_node.confidence_score
                        and new_node.confidence_score > existing_node.confidence_score
                    ):
                        merged_node.properties[key] = value
                    elif (
                        isinstance(value, str)
                        and isinstance(merged_node.properties[key], str)
                        and len(value) > len(merged_node.properties[key])
                    ):
                        merged_node.properties[key] = value

    # Update confidence score to max of both
    if (
        hasattr(new_node, "confidence_score")
        and hasattr(existing_node, "confidence_score")
        and new_node.confidence_score
        and existing_node.confidence_score
    ):
        merged_node.confidence_score = max(
            new_node.confidence_score, existing_node.confidence_score
        )

    return merged_node


def prune_orphaned_nodes(
    ontology: Dict[str, Any], graph: DocumentKnowledgeGraph
) -> None:
    """Remove nodes with no ontology-defined properties that aren't referenced in any relationship"""
    # Build set of node IDs used in relationships (both source and target)
    nodes_in_relationships = set()

    # If there are no relationships at all, don't prune any nodes.
    # This preserves entities when relationship extraction returns empty results
    # (e.g., when the ontology doesn't match the document content well).
    has_any_relationships = bool(graph.relationships)
    if not has_any_relationships:
        # Also check dynamic relationship fields for KnowledgeGraph
        for field_name in dir(graph):
            if (
                field_name.endswith("_list")
                or "_" not in field_name
                or field_name.startswith("_")
            ):
                continue
            rel_list = getattr(graph, field_name, None)
            if isinstance(rel_list, list) and len(rel_list) > 0:
                has_any_relationships = True
                break

    if not has_any_relationships:
        logger.info(
            "No relationships found - skipping node pruning to preserve extracted entities"
        )
        return

    # Check relationships list for DocumentKnowledgeGraph
    if graph.relationships:
        for rel in graph.relationships:
            nodes_in_relationships.add(rel.source_id)
            nodes_in_relationships.add(rel.target_id)
    else:
        # Check all relationship fields for KnowledgeGraph
        for field_name in dir(graph):
            if (
                field_name.endswith("_list")
                or "_" not in field_name
                or field_name.startswith("_")
            ):
                continue
            rel_list = getattr(graph, field_name)
            if isinstance(rel_list, list):
                for rel in rel_list:
                    nodes_in_relationships.add(rel.source_id)
                    nodes_in_relationships.add(rel.target_id)

    # Get all nodes
    all_nodes = graph.nodes if hasattr(graph, "nodes") else []
    if not all_nodes:
        for field_name in dir(graph):
            if not field_name.endswith("_list") or field_name.startswith("_"):
                continue
            node_list = getattr(graph, field_name)
            if isinstance(node_list, list):
                all_nodes.extend(node_list)

    # Find orphaned nodes and try to link them
    orphaned_nodes = []
    for node in all_nodes:
        if node.id not in nodes_in_relationships:
            # Match the function's documented contract: keep nodes
            # that carry ontology-defined property values, even when
            # no relationship references them. A node with real
            # extracted data ("name=Apple Inc., ticker=AAPL") is
            # legitimate output regardless of whether the LLM also
            # extracted a relationship for it.
            #
            # Until this guard, the implementation pruned every
            # orphan that couldn't auto-link — silently dropping
            # ~85% of legitimately extracted entities the moment
            # the relationship-extraction pass produced even a few
            # edges. Pre-9191684 this was masked: the relationship
            # bug meant relationships=[] → the early-return at the
            # top of this function fired and nothing got pruned.
            entity_def = ontology.get("entities", {}).get(node.type, {}) or {}
            ontology_prop_names = set(entity_def.get("properties", {}).keys())
            node_props = node.properties or {}
            has_ontology_props = any(
                k in ontology_prop_names
                and k not in SYSTEM_PROPERTIES
                and v not in (None, "", [])
                for k, v in node_props.items()
            )
            if has_ontology_props:
                # Real extracted entity — keep it.
                continue

            # Check if this node type has only one possible relationship type in ontology
            node_type = node.type
            possible_rels = _get_possible_relationships_for_type(ontology, node_type)

            if len(possible_rels) == 1:
                # Get the single relationship type and direction
                rel_type = list(possible_rels.keys())[0]
                rel_info = possible_rels[rel_type]

                # Find potential nodes to link with
                potential_nodes = []
                for other_node in all_nodes:
                    if other_node.id != node.id:
                        # Check both source->target and target->source possibilities
                        if (
                            rel_info["source"] == node_type
                            and rel_info["target"] == other_node.type
                        ) or (
                            rel_info["target"] == node_type
                            and rel_info["source"] == other_node.type
                        ):
                            potential_nodes.append(other_node)

                # If we found exactly one potential node, create the relationship
                if len(potential_nodes) == 1:
                    other_node = potential_nodes[0]
                    # Determine direction based on ontology
                    if rel_info["source"] == node_type:
                        source_id, target_id = node.id, other_node.id
                        source_type, target_type = node_type, other_node.type
                    else:
                        source_id, target_id = other_node.id, node.id
                        source_type, target_type = other_node.type, node_type

                    # Create relationship
                    rel_id = f"{source_id}_{rel_type}_{target_id}"
                    new_rel = RelationshipInstance(
                        id=rel_id,
                        type=rel_type,
                        source_id=source_id,
                        target_id=target_id,
                        source_type=source_type,
                        target_type=target_type,
                        properties={},
                        provenance=NodeProvenance(
                            chunk_ids=[],
                            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                            confidence_score=0.9,
                        ),
                    )

                    # Add to relationships
                    if graph.relationships:
                        graph.relationships.append(new_rel)
                    else:
                        rel_field = f"{source_type.lower()}_{rel_type.lower()}_{target_type.lower()}"
                        if not hasattr(graph, rel_field):
                            setattr(graph, rel_field, [])
                        getattr(graph, rel_field).append(new_rel)

                    # Node is no longer orphaned since it's now in a relationship
                    continue

            # If we couldn't auto-link, add to orphaned list
            orphaned_nodes.append(node)

    # Remove orphaned nodes that couldn't be linked
    if graph.nodes:
        graph.nodes = [n for n in graph.nodes if n not in orphaned_nodes]
    else:
        for field_name in dir(graph):
            if not field_name.endswith("_list") or field_name.startswith("_"):
                continue
            node_list = getattr(graph, field_name)
            if isinstance(node_list, list):
                setattr(
                    graph, field_name, [n for n in node_list if n not in orphaned_nodes]
                )


def _get_possible_relationships_for_type(
    ontology: Dict[str, Any], node_type: str
) -> Dict[str, Dict[str, str]]:
    """Get all possible relationship types for a given node type from ontology"""
    possible_rels = {}

    # Get relationships where this type is either source or target
    entity_def = ontology.get("entities", {}).get(node_type, {})
    if not entity_def:
        return {}

    relationships = entity_def.get("relationships", {})
    for rel_type, rel_info in relationships.items():
        source_type = node_type
        target_type = rel_info.get("target")

        if source_type == node_type or target_type == node_type:
            possible_rels[rel_type] = {"source": source_type, "target": target_type}

    return possible_rels


def _generate_node_key(
    parsed_ontology,
    entity_type: str,
    properties: Dict[str, Any],
    canonical_properties: Optional[Dict[str, Any]] = None,
    raw_properties: Optional[Dict[str, Any]] = None,
    fallback_hint: Optional[str] = None,
) -> str:
    entity_def = parsed_ontology.get("entities", {}).get(entity_type, {})
    candidate_props = canonical_properties or properties
    unique_props = []
    for prop_name, prop_def in entity_def.get("properties", {}).items():
        if (
            prop_def.get("unique", False)
            and prop_name in candidate_props
            and candidate_props[prop_name] is not None
        ):
            value = candidate_props[prop_name]
            if isinstance(value, str):
                value = value.lower().strip()
            unique_props.append((prop_name, value))
    if unique_props:
        sorted_props = sorted(unique_props)
        return f"{entity_type}:" + ":".join(f"{k}={v}" for k, v in sorted_props)
    else:
        non_empty_props = []
        for k, v in candidate_props.items():
            if v is not None:
                if isinstance(v, str):
                    v = v.lower().strip()
                non_empty_props.append((k, v))
        if non_empty_props:
            sorted_props = sorted(non_empty_props)
            return f"{entity_type}:" + ":".join(f"{k}={v}" for k, v in sorted_props)

        if raw_properties:
            raw_repr = json.dumps(raw_properties, sort_keys=True, default=str)
            raw_digest = hashlib.sha1(raw_repr.encode("utf-8")).hexdigest()
            return f"{entity_type}:raw={raw_digest}"

        if fallback_hint:
            return f"{entity_type}:fallback={fallback_hint}"

        return f"{entity_type}:uuid={uuid.uuid4()}"


def _make_deterministic_node_id(
    transform_id: Optional[str], entity_type: str, node_key: str
) -> str:
    """Build a stable node identifier scoped to a transform when enabled."""

    if not transform_id:
        return str(uuid.uuid4())

    namespace_input = f"{transform_id}:{entity_type}:{node_key}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, namespace_input))


_CANONICAL_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "graphora:canonical-node")


def _make_canonical_node_id(node_key: str) -> str:
    """Create a transform-agnostic identifier for ledger lookups."""

    return str(uuid.uuid5(_CANONICAL_NAMESPACE, node_key))


def _extract_properties(item: BaseModel) -> Dict[str, Any]:
    if item is None:
        return {}
    metadata_fields = {
        "model_computed_fields",
        "model_config",
        "model_fields",
        "model_fields_set",
        "__fields__",
        "__annotations__",
        "__field_defaults__",
        "__private_attributes__",
    }
    try:
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        else:
            item_dict = {k: v for k, v in vars(item).items() if not k.startswith("_")}
        for field in metadata_fields:
            item_dict.pop(field, None)
        item_dict.pop("type", None)
        return {k: v for k, v in item_dict.items() if v is not None}
    except Exception:
        properties = {}
        for attr_name in dir(item):
            if (
                attr_name.startswith("_")
                or callable(getattr(item, attr_name))
                or attr_name in metadata_fields
                or attr_name == "type"
            ):
                continue
            try:
                value = getattr(item, attr_name)
                if value is not None:
                    properties[attr_name] = value
            except Exception:
                pass
        return properties


def _normalize_entity_properties(
    parsed_ontology: Dict[str, Any],
    entity_type: str,
    raw_properties: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    entity_def = parsed_ontology.get("entities", {}).get(entity_type)
    if not entity_def:
        return None

    defined_properties = entity_def.get("properties", {})
    if not defined_properties:
        return None

    normalized: Dict[str, Any] = {}
    for prop_name, prop_def in defined_properties.items():
        value = raw_properties.get(prop_name)
        if value is None:
            if prop_def.get("required"):
                logger.debug(
                    "Required property %s missing for %s", prop_name, entity_type
                )
                return None
            continue

        coerced = _coerce_property_value(prop_def, value)
        if coerced is None:
            logger.debug("Failed to coerce property %s for %s", prop_name, entity_type)
            return None

        enum_value = _match_enum_value(prop_def, coerced)
        if enum_value is None:
            logger.debug(
                "Value %s not allowed for %s.%s",
                coerced,
                entity_type,
                prop_name,
            )
            return None

        normalized[prop_name] = enum_value

    if not normalized:
        return None

    return normalized


def _normalize_relationship_properties(
    parsed_ontology: Dict[str, Any],
    source_type: str,
    relationship_type: str,
    raw_properties: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    relationships_def = (
        parsed_ontology.get("entities", {})
        .get(source_type, {})
        .get("relationships", {})
        .get(relationship_type, {})
    )
    property_defs = relationships_def.get("properties", {})
    if not property_defs:
        return {}

    normalized: Dict[str, Any] = {}
    for prop_name, prop_def in property_defs.items():
        value = raw_properties.get(prop_name)
        if value is None:
            if prop_def.get("required"):
                logger.debug(
                    "Required relationship property %s missing for %s",
                    prop_name,
                    relationship_type,
                )
                return None
            continue

        coerced = _coerce_property_value(prop_def, value)
        if coerced is None:
            logger.debug(
                "Failed to coerce relationship property %s for %s",
                prop_name,
                relationship_type,
            )
            return None

        enum_value = _match_enum_value(prop_def, coerced)
        if enum_value is None:
            logger.debug(
                "Value %s not allowed for relationship %s.%s",
                coerced,
                relationship_type,
                prop_name,
            )
            return None

        normalized[prop_name] = enum_value

    return normalized


def _coerce_property_value(prop_def: Dict[str, Any], value: Any) -> Optional[Any]:
    prop_type = (prop_def.get("type") or "string").lower()

    try:
        if prop_type in {"string", "str"}:
            coerced = str(value).strip()
            case_format = (
                prop_def.get("quality", {}).get("format", {}).get("caseFormat")
            )
            if case_format:
                coerced = _apply_case_format(coerced, case_format)
            return coerced

        if prop_type in {"integer", "int"}:
            return int(value)

        if prop_type in {"number", "float", "double"}:
            return float(value)

        if prop_type in {"boolean", "bool"}:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
            return bool(value)

        if prop_type in {"array", "list"}:
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                return [v.strip() for v in value.split(",") if v.strip()]
            return [value]

        return value
    except (ValueError, TypeError):
        return None


def _match_enum_value(prop_def: Dict[str, Any], value: Any) -> Optional[Any]:
    enum_values = prop_def.get("enum") or prop_def.get("allowedValues")
    if not enum_values:
        return value

    if isinstance(value, str):
        for enum_value in enum_values:
            if isinstance(enum_value, str) and enum_value.lower() == value.lower():
                return enum_value

    return value if value in enum_values else None


def _apply_case_format(value: str, case_format: str) -> str:
    formatter = case_format.lower()
    if formatter == "lowercase":
        return value.lower()
    if formatter == "uppercase":
        return value.upper()
    if formatter == "titlecase":
        return value.title()
    return value


def _prepare_entities_for_deduplication(
    entities, relationships=None, parsed_ontology=None
):
    """
    Prepare entities for deduplication by converting to dictionaries and
    enriching with 1-degree related entities where possible.

    Args:
        entities (List[BaseNode]): List of entity nodes
        relationships (List[RelationshipInstance], optional): List of relationships
        parsed_ontology (dict, optional): Parsed ontology to get property types

    Returns:
        List[Dict]: Prepared entity data for deduplication
    """
    # Convert entities to dictionaries
    entities_data = [entity.dict() for entity in entities]

    # Create entity ID to index mapping for quick lookup
    entity_id_to_index = {entity.id: i for i, entity in enumerate(entities)}

    # If relationships are provided, enrich entities with 1-degree connections
    if relationships:
        # Create a graph representation for quick neighbor lookup
        neighbors = {}
        for rel in relationships:
            source_id = rel.source_id
            target_id = rel.target_id
            rel_type = rel.type

            if source_id not in neighbors:
                neighbors[source_id] = []
            if target_id not in neighbors:
                neighbors[target_id] = []

            neighbors[source_id].append(
                {"id": target_id, "type": rel_type, "direction": "outgoing"}
            )
            neighbors[target_id].append(
                {"id": source_id, "type": rel_type, "direction": "incoming"}
            )

        # Enrich entities with neighbor information
        for i, entity_data in enumerate(entities_data):
            entity_id = entity_data["id"]
            if entity_id in neighbors:
                # Instead of creating complex nested structures, create flattened properties
                # with neighbor information that's easier for Splink to process
                neighbor_count = 0

                for neighbor in neighbors[entity_id]:
                    neighbor_id = neighbor["id"]
                    if neighbor_id in entity_id_to_index:
                        try:
                            # Convert complex values to strings
                            neighbor_entity = entities[entity_id_to_index[neighbor_id]]
                            neighbor_data = neighbor_entity.dict()

                            # Add basic neighbor info with a prefix and index to keep them separate
                            prefix = f"neighbor_{neighbor_count}_"

                            # Add basic neighbor metadata
                            if "properties" in entity_data and isinstance(
                                entity_data["properties"], dict
                            ):
                                entity_data["properties"][f"{prefix}type"] = (
                                    neighbor_data.get("type", "")
                                )
                                entity_data["properties"][f"{prefix}rel_type"] = (
                                    neighbor["type"]
                                )
                                entity_data["properties"][f"{prefix}direction"] = (
                                    neighbor["direction"]
                                )

                                # Add selected non-system properties from neighbor
                                if "properties" in neighbor_data and isinstance(
                                    neighbor_data["properties"], dict
                                ):
                                    non_system_properties = {
                                        k: v
                                        for k, v in neighbor_data["properties"].items()
                                        if k not in SYSTEM_PROPERTIES
                                    }
                                    for prop_name in non_system_properties:
                                        prop_value = non_system_properties[prop_name]
                                        entity_data["properties"][
                                            f"{prefix}{prop_name}"
                                        ] = prop_value

                            neighbor_count += 1
                            # Limit the number of neighbors to avoid explosion of properties
                            if neighbor_count >= 3:  # Only include up to 3 neighbors
                                break
                        except Exception as e:
                            logging.warning(
                                f"Error processing neighbor {neighbor_id} for entity {entity_id}: {str(e)}"
                            )

    # Add property types from ontology
    if parsed_ontology:
        for entity_data in entities_data:
            entity_type = entity_data.get("type")
            if entity_type and entity_type in parsed_ontology.get("entities", {}):
                properties = entity_data.get("properties", {})
                # Create a copy of the properties to avoid modifying during iteration
                property_items = list(properties.items())

                for prop_name, prop_value in property_items:
                    if prop_name in parsed_ontology.get("entities", {}).get(
                        entity_type, {}
                    ).get("properties", {}):
                        prop_type = (
                            parsed_ontology.get("entities", {})
                            .get(entity_type, {})
                            .get("properties", {})
                            .get(prop_name, {})
                            .get("type")
                        )
                        if prop_type:
                            entity_data["properties"][f"{prop_name}_type"] = prop_type

    return entities_data


def _deduplicate_small_entity_group(
    entity_type: str,
    entities: List[BaseNode | Node],
    parsed_ontology: Optional[Dict[str, Any]] = None,
) -> Tuple[List[BaseNode | Node], Dict[str, str]]:
    entity_def = (parsed_ontology or {}).get("entities", {}).get(entity_type or "", {})
    property_defs = entity_def.get("properties", {}) if entity_def else {}

    unique_props = sorted(
        prop_name
        for prop_name, prop_def in property_defs.items()
        if isinstance(prop_def, dict) and prop_def.get("unique")
    )
    index_props = sorted(
        prop_name
        for prop_name, prop_def in property_defs.items()
        if isinstance(prop_def, dict) and prop_def.get("index")
    )
    include_index_hints = len(entities) <= 8 and bool(index_props)

    def _get_attr(node: BaseNode | Node, attr: str, default=None):
        if isinstance(node, dict):
            return node.get(attr, default)
        return getattr(node, attr, default)

    def _canonical_value(node: BaseNode | Node, prop: str) -> Optional[str]:
        canonical_props = _get_attr(node, "canonical_properties", {}) or {}
        raw_props = _get_attr(node, "properties", {}) or {}
        value = canonical_props.get(prop)
        if value is None:
            value = raw_props.get(prop)
        if value is None:
            return None
        return value.strip() if isinstance(value, str) else str(value)

    def _signature_candidates(node: BaseNode | Node) -> List[Tuple[str, str]]:
        candidates: List[Tuple[str, str]] = []
        canonical_id = _get_attr(node, "canonical_id")
        if canonical_id:
            candidates.append(("canonical_id", str(canonical_id)))

        canonical_key = _get_attr(node, "canonical_key")
        if canonical_key:
            candidates.append(("canonical_key", str(canonical_key)))

        for prop in unique_props:
            value = _canonical_value(node, prop)
            if value:
                candidates.append((f"unique:{prop}", value))

        if include_index_hints:
            for prop in index_props:
                value = _canonical_value(node, prop)
                if value:
                    candidates.append((f"index:{prop}", value))

        return candidates

    entity_by_id: Dict[str, BaseNode | Node] = {
        _get_attr(entity, "id"): entity for entity in entities
    }

    signature_to_rep: Dict[Tuple[str, str], str] = {}
    mappings: Dict[str, str] = {}
    processing_order: List[str] = []

    for entity in entities:
        entity_id = _get_attr(entity, "id")
        processing_order.append(entity_id)

        candidates = _signature_candidates(entity)
        if not candidates:
            continue

        representative_id = None
        for signature in candidates:
            existing_rep = signature_to_rep.get(signature)
            if existing_rep:
                representative_id = existing_rep
                break

        if representative_id is None:
            representative_id = entity_id

        if representative_id != entity_id:
            mappings[entity_id] = representative_id

        for signature in candidates:
            signature_to_rep.setdefault(signature, representative_id)

    if not mappings:
        return entities, {}

    seen: Set[str] = set()
    deduplicated: List[BaseNode | Node] = []

    for entity_id in processing_order:
        representative_id = mappings.get(entity_id, entity_id)
        if representative_id in seen:
            continue
        seen.add(representative_id)
        deduplicated.append(entity_by_id[representative_id])

    return deduplicated, mappings


def _create_splink_dataframe(
    entities_data,
    system_properties,
    allowed_properties: Optional[Set[str]] = None,
):
    """
    Create a DataFrame for Splink processing from entity data.

    Optimized: Uses vectorized column construction instead of row-by-row df.at[].

    Args:
        entities_data (List[Dict]): List of entity dictionaries
        system_properties (List[str]): List of system properties to exclude

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame and list of comparison columns
    """

    # First create a basic DataFrame with just the IDs
    df = pd.DataFrame(entities_data)

    # Get property columns for comparison (focus on properties dictionary)
    properties_columns: Set[str] = set()
    for entity in entities_data:
        props = (
            entity.get("properties")
            if isinstance(entity.get("properties"), dict)
            else {}
        )
        canonical_props = (
            entity.get("canonical_properties")
            if isinstance(entity.get("canonical_properties"), dict)
            else {}
        )
        for prop in props.keys():
            if prop in system_properties:
                continue
            if allowed_properties is not None and prop not in allowed_properties:
                continue
            properties_columns.add(prop)
        for prop in canonical_props.keys():
            if prop in system_properties:
                continue
            if allowed_properties is not None and prop not in allowed_properties:
                continue
            properties_columns.add(prop)
            properties_columns.add(f"{CANONICAL_COLUMN_PREFIX}{prop}")

    # Get unique property columns
    properties_columns = sorted(properties_columns)

    # Vectorized column construction - build all column data at once
    columns_data: Dict[str, List[Optional[str]]] = {
        col: [] for col in properties_columns
    }

    for entity in entities_data:
        props = (
            entity.get("properties")
            if isinstance(entity.get("properties"), dict)
            else {}
        )
        canonical_props = (
            entity.get("canonical_properties")
            if isinstance(entity.get("canonical_properties"), dict)
            else {}
        )
        combined_props = {
            **props,
            **{k: v for k, v in canonical_props.items() if v is not None},
        }

        for col in properties_columns:
            if col.startswith(CANONICAL_COLUMN_PREFIX):
                # This is a canonical column
                base_prop = col[len(CANONICAL_COLUMN_PREFIX) :]
                value = canonical_props.get(base_prop)
            else:
                value = combined_props.get(col)

            if value is None:
                columns_data[col].append(None)
            elif isinstance(value, (list, dict, tuple)):
                columns_data[col].append(str(value))
            else:
                columns_data[col].append(str(value))

    # Assign all columns at once (vectorized)
    for col_name, col_values in columns_data.items():
        df[col_name] = col_values

    return df, properties_columns


def _is_canonical_column(column_name: str) -> bool:
    return column_name.startswith(CANONICAL_COLUMN_PREFIX)


def _base_property_from_column(column_name: str) -> str:
    return (
        column_name[len(CANONICAL_COLUMN_PREFIX) :]
        if _is_canonical_column(column_name)
        else column_name
    )


def _create_splink_comparisons(
    properties_columns,
    df,
    record_count,
    entity_type,
    parsed_ontology=None,
):
    """
    Create appropriate Splink comparisons based on property columns.

    Args:
        properties_columns (List[str]): List of property column names
        df (pd.DataFrame): DataFrame with entity data
        entity_type (str): Type of entity
        parsed_ontology (dict, optional): Parsed ontology to get property types

    Returns:
        List: List of Splink comparison objects
    """

    entity_prop_defs = (
        (parsed_ontology or {})
        .get("entities", {})
        .get(entity_type, {})
        .get("properties", {})
    )

    column_variants: Dict[str, Dict[str, str]] = defaultdict(dict)
    for col in properties_columns:
        if col not in df.columns:
            continue
        base_prop = _base_property_from_column(col)
        variant_key = "canonical" if _is_canonical_column(col) else "raw"
        column_variants[base_prop][variant_key] = col

    def _column_has_data(column: Optional[str]) -> bool:
        return bool(column) and column in df.columns and df[column].notna().sum() > 0

    def _prefer_column(
        variants: Dict[str, str], prefer_canonical: bool = True
    ) -> Optional[str]:
        order = ["canonical", "raw"] if prefer_canonical else ["raw", "canonical"]
        for variant in order:
            column = variants.get(variant)
            if _column_has_data(column):
                return column
        return None

    property_types: Dict[str, str] = {}
    unique_columns: List[str] = []
    indexed_columns: List[str] = []
    string_columns: List[str] = []
    text_columns: List[str] = []  # TEXT type uses embedding similarity
    numeric_columns: List[str] = []
    datetime_columns: List[str] = []
    fallback_columns: List[str] = []

    for base_prop, variants in column_variants.items():
        prop_def = entity_prop_defs.get(base_prop, {}) if entity_prop_defs else {}
        canonical_col = variants.get("canonical")
        raw_col = variants.get("raw")
        primary_col = _prefer_column(variants) or _prefer_column(
            variants, prefer_canonical=False
        )

        if not _column_has_data(primary_col):
            continue

        prop_type = prop_def.get("type") if isinstance(prop_def, dict) else None
        unique_flag = (
            bool(prop_def.get("unique")) if isinstance(prop_def, dict) else False
        )
        index_flag = (
            bool(prop_def.get("index")) if isinstance(prop_def, dict) else False
        )

        if unique_flag:
            column = canonical_col if _column_has_data(canonical_col) else primary_col
            if column not in unique_columns:
                unique_columns.append(column)
            if prop_type:
                property_types[column] = prop_type
            continue

        if index_flag:
            column = canonical_col if _column_has_data(canonical_col) else primary_col
            if column not in indexed_columns:
                indexed_columns.append(column)
            if prop_type:
                property_types[column] = prop_type
            continue

        # TEXT type columns - for embedding similarity
        if prop_type and _is_prop_type_text(prop_type):
            if primary_col not in text_columns:
                text_columns.append(primary_col)
                property_types[primary_col] = prop_type
            continue

        if prop_type and _is_prop_type_string(prop_type):
            primary = canonical_col if _column_has_data(canonical_col) else primary_col
            if primary not in string_columns:
                string_columns.append(primary)
                property_types[primary] = prop_type
            if (
                canonical_col
                and raw_col
                and canonical_col != raw_col
                and _column_has_data(raw_col)
                and raw_col not in string_columns
            ):
                string_columns.append(raw_col)
                property_types[raw_col] = prop_type
            continue

        if prop_type and _is_prop_type_number(prop_type):
            if primary_col not in numeric_columns:
                numeric_columns.append(primary_col)
                property_types[primary_col] = prop_type
            continue

        if prop_type and _is_prop_type_datetime(prop_type):
            if primary_col not in datetime_columns:
                datetime_columns.append(primary_col)
                property_types[primary_col] = prop_type
            continue

        if primary_col:
            fallback_columns.append(primary_col)
            if prop_type:
                property_types[primary_col] = prop_type
            elif df[primary_col].dtype != "object":
                property_types[primary_col] = "number"
            else:
                property_types[primary_col] = "string"

    comparisons: List[Any] = []
    used_columns: Set[str] = set()

    def _append_exact(columns: List[str], prior: ComparisonPrior) -> None:
        for column in columns:
            if column in used_columns or not _column_has_data(column):
                continue
            comparison = _exact_match(column)
            comparisons.append(_with_prior(comparison, prior))
            used_columns.add(column)

    def _append_string(
        columns: List[str],
        *,
        limit: Optional[int] = None,
        allow_when_prefer_exact: bool = False,
    ) -> None:
        max_entries = limit
        if not allow_when_prefer_exact and comparisons:
            max_entries = 1 if max_entries is None else min(max_entries, 1)
        count = 0
        for column in columns:
            if column in used_columns or not _column_has_data(column):
                continue
            comparison = cl.JaroWinklerAtThresholds(column, [0.95, 0.85])
            comparisons.append(_with_prior(comparison, STRING_PRIOR))
            used_columns.add(column)
            count += 1
            if max_entries is not None and count >= max_entries:
                break

    def _append_numeric(columns: List[str], prior: ComparisonPrior) -> None:
        for column in columns:
            if column in used_columns or not _column_has_data(column):
                continue
            comparison = _exact_match(column)
            comparisons.append(_with_prior(comparison, prior))
            used_columns.add(column)

    _append_exact(unique_columns, UNIQUE_PRIOR)
    _append_exact(indexed_columns, INDEX_PRIOR)

    allow_string_when_prefer_exact = not comparisons
    _append_string(
        string_columns,
        limit=3,
        allow_when_prefer_exact=allow_string_when_prefer_exact,
    )
    _append_numeric(numeric_columns[:2], NUMERIC_PRIOR)
    _append_numeric(datetime_columns[:1], DATETIME_PRIOR)

    if not comparisons:
        for column in fallback_columns:
            if column in used_columns or not _column_has_data(column):
                continue
            prop_type = property_types.get(column, "string")
            if _is_prop_type_string(prop_type):
                comparison = cl.JaroWinklerAtThresholds(column, [0.95, 0.85])
                comparisons.append(_with_prior(comparison, STRING_PRIOR))
            else:
                prior = (
                    NUMERIC_PRIOR
                    if _is_prop_type_number(prop_type)
                    else FALLBACK_EXACT_PRIOR
                )
                comparison = _exact_match(column)
                comparisons.append(_with_prior(comparison, prior))
            used_columns.add(column)
            if len(comparisons) >= 3:
                break

    logging.info(
        "Created %d comparisons for Splink (unique=%d indexed=%d string=%d text=%d numeric=%d)",
        len(comparisons),
        len(unique_columns),
        len(indexed_columns),
        len(string_columns),
        len(text_columns),
        len(numeric_columns),
    )
    return comparisons, text_columns


def _is_prop_type_string(prop_type: str) -> bool:
    return prop_type.lower() in ["str", "string"]


def _is_prop_type_number(prop_type: str) -> bool:
    return prop_type.lower() in ["int", "double", "integer", "float", "number"]


def _is_prop_type_datetime(prop_type: str) -> bool:
    return prop_type.lower() in ["date", "datetime", "timestamp"]


def _is_prop_type_text(prop_type: str) -> bool:
    """Check if property type is TEXT (uses embedding similarity)."""
    return prop_type.lower() == "text"


def _create_blocking_rules(
    properties_columns,
    df,
    record_count,
    entity_type=None,
    parsed_ontology=None,
):
    _require_er_extras()
    """
    Create blocking rules for Splink based on property columns.

    Args:
        properties_columns (List[str]): List of property column names
        df (pd.DataFrame): DataFrame with entity data
        entity_type (str, optional): Type of entity
        parsed_ontology (dict, optional): Parsed ontology to get property types

    Returns:
        List: List of Splink blocking rules
    """

    entity_prop_defs = (
        (parsed_ontology or {})
        .get("entities", {})
        .get(entity_type, {})
        .get("properties", {})
        if entity_type
        else {}
    )

    column_variants: Dict[str, Dict[str, str]] = defaultdict(dict)
    for col in properties_columns:
        if col not in df.columns:
            continue
        base_prop = _base_property_from_column(col)
        variant_key = "canonical" if _is_canonical_column(col) else "raw"
        column_variants[base_prop][variant_key] = col

    def _column_has_data(column: Optional[str]) -> bool:
        return bool(column) and column in df.columns and df[column].notna().sum() > 0

    def _prefer_column(
        variants: Dict[str, str], prefer_canonical: bool = True
    ) -> Optional[str]:
        order = ["canonical", "raw"] if prefer_canonical else ["raw", "canonical"]
        for variant in order:
            column = variants.get(variant)
            if _column_has_data(column):
                return column
        return None

    unique_columns: List[str] = []
    indexed_columns: List[str] = []
    string_columns: List[str] = []
    type_columns: List[str] = []
    fallback_columns: List[str] = []

    for base_prop, variants in column_variants.items():
        prop_def = entity_prop_defs.get(base_prop, {}) if entity_prop_defs else {}
        canonical_col = variants.get("canonical")
        primary_col = _prefer_column(variants) or _prefer_column(
            variants, prefer_canonical=False
        )

        if not _column_has_data(primary_col):
            continue

        if base_prop == "type":
            if primary_col not in type_columns:
                type_columns.append(primary_col)
            continue

        prop_type = prop_def.get("type") if isinstance(prop_def, dict) else None
        unique_flag = (
            bool(prop_def.get("unique")) if isinstance(prop_def, dict) else False
        )
        index_flag = (
            bool(prop_def.get("index")) if isinstance(prop_def, dict) else False
        )

        if unique_flag:
            column = canonical_col if _column_has_data(canonical_col) else primary_col
            if column not in unique_columns:
                unique_columns.append(column)
            continue

        if index_flag:
            column = canonical_col if _column_has_data(canonical_col) else primary_col
            if column not in indexed_columns:
                indexed_columns.append(column)
            continue

        if prop_type and _is_prop_type_string(prop_type):
            preferred = (
                canonical_col if _column_has_data(canonical_col) else primary_col
            )
            if preferred not in string_columns:
                string_columns.append(preferred)
            continue

        if primary_col not in fallback_columns:
            fallback_columns.append(primary_col)

    blocking_rules = []
    seen_sql: Set[str] = set()

    def _append_rule(column: Optional[str]) -> None:
        if not column or not _column_has_data(column):
            return
        rule = block_on(column)
        rule_sql = getattr(rule, "blocking_rule_sql", None)
        if rule_sql and rule_sql in seen_sql:
            return
        blocking_rules.append(rule)
        if rule_sql:
            seen_sql.add(rule_sql)

    if type_columns:
        _append_rule(type_columns[0])

    for column in unique_columns:
        _append_rule(column)

    for column in indexed_columns[:2]:
        _append_rule(column)

    for column in string_columns[:2]:
        _append_rule(column)

    if len(blocking_rules) < 2:
        for column in fallback_columns:
            _append_rule(column)
            if len(blocking_rules) >= 2:
                break

    logging.info(
        "Created %d blocking rules for Splink (unique=%d indexed=%d string=%d)",
        len(blocking_rules),
        len(unique_columns),
        len(indexed_columns),
        len(string_columns),
    )
    return blocking_rules


def _run_splink_deduplication(df, comparisons, blocking_rules, threshold):
    _require_er_extras()
    """
    Run Splink deduplication on the prepared DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with entity data
        comparisons (List): List of Splink comparison objects
        blocking_rules (List): List of Splink blocking rules
        threshold (float): Match probability threshold

    Returns:
        Dict: Mapping of entity IDs to representative entity IDs
    """

    # Ensure 'id' column exists
    if "id" not in df.columns:
        logging.error("DataFrame must have an 'id' column for deduplication")
        return {}, []

    record_count = len(df)

    # If we have too few records or comparisons, skip deduplication
    if record_count < 3 or not comparisons or not blocking_rules:
        logging.warning(
            "Insufficient data for Splink deduplication: %d records, %d comparisons, %d blocking rules",
            record_count,
            len(comparisons),
            len(blocking_rules),
        )
        return {}, []

    # Initialize DuckDB API
    db_api = DuckDBAPI()

    try:
        total_pairs = max((record_count * (record_count - 1)) // 2, 1)
        random_match_probability = max(1e-6, min(0.001, 1.0 / total_pairs))

        # Create settings with the correct unique_id_column_name and simplified parameters
        settings = SettingsCreator(
            link_type="dedupe_only",
            comparisons=comparisons,
            blocking_rules_to_generate_predictions=blocking_rules,
            unique_id_column_name="id",  # Specify the ID column name
            # Use default values for parameters that might be hard to estimate
            probability_two_random_records_match=random_match_probability,
            em_convergence=0.01,
            max_iterations=5,
        )

        # Create linker
        linker = Linker(df, settings, db_api)

        # Simplified training approach
        logging.info("Estimating parameters for deduplication...")

        # Skip complex parameter estimation if we have limited data
        if record_count < 20:
            logging.info("Limited data available, using default parameters")
        else:
            try:
                # Estimate u parameters with a smaller sample to speed things up
                linker.training.estimate_u_using_random_sampling(max_pairs=1e5)

                # Use the first blocking rule for EM estimation with fewer iterations
                if blocking_rules:
                    linker.training.estimate_parameters_using_expectation_maximisation(
                        blocking_rules[0], max_iterations=3
                    )
            except Exception as e:
                logging.warning(
                    f"Parameter estimation failed, using defaults: {str(e)}"
                )

        # Generate predictions with a more lenient threshold for better recall
        logging.info("Generating deduplication predictions...")
        pairwise_predictions = linker.inference.predict(threshold_match_weight=-10)

        # Cluster predictions
        logging.info(f"Clustering predictions with threshold {threshold}...")
        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            pairwise_predictions, threshold
        )

        # Get the cluster results as a dataframe
        df_clusters = clusters.as_pandas_dataframe()

        # Check if we got any clusters
        if df_clusters.empty:
            logging.info("No duplicate clusters found")
            return {}, []

        logging.info(f"Found {len(df_clusters['cluster_id'].unique())} clusters")

        # Create a mapping of entity IDs to their representative entity IDs
        id_to_representative = {}

        # Check if expected columns exist
        id_col = "id"  # This should match unique_id_column_name in settings
        if id_col not in df_clusters.columns or "cluster_id" not in df_clusters.columns:
            logging.warning(
                f"Missing expected columns in cluster dataframe. Available columns: {df_clusters.columns.tolist()}"
            )
            return {}, []

        match_scores: List[float] = []
        try:
            predictions_df = pairwise_predictions.as_pandas_dataframe()
            if (
                not predictions_df.empty
                and "match_probability" in predictions_df.columns
            ):
                probability_series = predictions_df["match_probability"].dropna()
                if not probability_series.empty:
                    match_scores = (
                        probability_series[probability_series >= threshold]
                        .astype(float)
                        .clip(lower=0.0, upper=1.0)
                        .tolist()
                    )
        except Exception:  # pragma: no cover - telemetry only
            logger.debug(
                "Unable to extract match probability metrics from Splink output"
            )

        # Create a mapping of cluster IDs to representative entity IDs
        cluster_to_representative = {}
        for cluster_id in df_clusters["cluster_id"].unique():
            cluster_members = df_clusters[df_clusters["cluster_id"] == cluster_id][
                id_col
            ].tolist()
            if cluster_members:
                representative = cluster_members[
                    0
                ]  # Use first entity as representative
                cluster_to_representative[cluster_id] = representative

                # Log cluster information for debugging
                if len(cluster_members) > 1:
                    logging.info(
                        f"Cluster {cluster_id}: {len(cluster_members)} members, representative: {representative}"
                    )

        # Map each entity to its representative entity - vectorized O(n)
        id_to_representative = dict(
            zip(
                df_clusters[id_col],
                df_clusters["cluster_id"].map(cluster_to_representative),
            )
        )

        return id_to_representative, match_scores

    except Exception as e:
        logging.error(f"Error in Splink deduplication: {str(e)}")
        logging.debug(f"Deduplication error details: {traceback.format_exc()}")
        return {}, []


def _create_deduplicated_entities(entities, id_to_representative):
    """
    Create a deduplicated list of entities based on the deduplication results.

    Optimized from O(n²) to O(n) by building lookup dict once.

    Args:
        entities (List[BaseNode]): Original list of entities
        id_to_representative (Dict): Mapping of entity IDs to representative entity IDs

    Returns:
        List[BaseNode]: Deduplicated list of entities
    """
    # Build entity lookup once - O(n) instead of nested loop
    entity_by_id = {entity.id: entity for entity in entities}
    processed_reps = set()
    deduplicated_entities = []

    for entity in entities:
        entity_id = entity.id
        rep_id = id_to_representative.get(entity_id, entity_id)

        # Skip if we've already added this representative
        if rep_id in processed_reps:
            continue

        processed_reps.add(rep_id)

        # Get the representative entity from lookup - O(1)
        rep_entity = entity_by_id.get(rep_id)
        if rep_entity:
            deduplicated_entities.append(rep_entity)

    return deduplicated_entities


def _run_splink_deduplication_batched(
    entities_data: List[Dict[str, Any]],
    df: "pd.DataFrame",  # quoted: pd is lazy-imported and may be None until [er] is installed
    comparisons: List,
    blocking_rules: List,
    threshold: float,
    batch_size: int = BATCH_SIZE_THRESHOLD,
    overlap_ratio: float = 0.1,
) -> Tuple[Dict[str, str], List[float]]:
    """
    Run Splink deduplication in batches for large datasets.

    Uses overlapping batches to ensure cross-batch matching at boundaries.
    Applies transitive closure to merge clusters across batches.

    Args:
        entities_data: List of entity dictionaries.
        df: DataFrame with entity data.
        comparisons: Splink comparison objects.
        blocking_rules: Splink blocking rules.
        threshold: Match probability threshold.
        batch_size: Maximum entities per batch.
        overlap_ratio: Fraction of batch to overlap with next batch.

    Returns:
        Tuple of (id_to_representative mapping, list of match scores).
    """
    record_count = len(df)

    # If below threshold, use standard processing
    if record_count <= batch_size:
        return _run_splink_deduplication(df, comparisons, blocking_rules, threshold)

    logging.info(
        "Large dataset detected (%d entities). Using batched processing with batch_size=%d",
        record_count,
        batch_size,
    )

    # Calculate overlap size
    overlap_size = max(10, int(batch_size * overlap_ratio))

    # Split into batches with overlap
    batches = []
    start = 0
    while start < record_count:
        end = min(start + batch_size, record_count)
        batches.append((start, end))
        # Move start forward by (batch_size - overlap) for next batch
        start = end - overlap_size if end < record_count else record_count

    logging.info(
        "Processing %d batches with %d entity overlap",
        len(batches),
        overlap_size,
    )

    # Process each batch and collect mappings
    all_mappings: Dict[str, str] = {}
    all_match_scores: List[float] = []

    for batch_idx, (start, end) in enumerate(batches):
        batch_df = df.iloc[start:end].copy().reset_index(drop=True)

        if len(batch_df) < 3:
            continue

        logging.debug(
            "Processing batch %d/%d: entities %d-%d (%d total)",
            batch_idx + 1,
            len(batches),
            start,
            end,
            len(batch_df),
        )

        batch_mappings, batch_scores = _run_splink_deduplication(
            batch_df, comparisons, blocking_rules, threshold
        )

        if batch_mappings:
            all_mappings.update(batch_mappings)
        if batch_scores:
            all_match_scores.extend(batch_scores)

    # Apply transitive closure to merge clusters across batches
    if all_mappings:
        all_mappings = _apply_transitive_closure(all_mappings)
        logging.info(
            "Applied transitive closure: %d final representative mappings",
            len(set(all_mappings.values())),
        )

    return all_mappings, all_match_scores


def _apply_transitive_closure(mappings: Dict[str, str]) -> Dict[str, str]:
    """
    Apply transitive closure to resolve chains of mappings.

    If A -> B and B -> C, this ensures A -> C (the ultimate representative).
    Uses Union-Find algorithm for efficient clustering.

    Args:
        mappings: Initial id -> representative mappings.

    Returns:
        Updated mappings with transitive closure applied.
    """
    if not mappings:
        return mappings

    # Build Union-Find structure
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        """Find root with path compression."""
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        """Union two elements."""
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            # Prefer the representative (value) as the root
            parent[root_x] = root_y

    # Process all mappings
    for entity_id, rep_id in mappings.items():
        union(entity_id, rep_id)

    # Build final mappings with true representatives
    final_mappings: Dict[str, str] = {}
    for entity_id in mappings:
        root = find(entity_id)
        if entity_id != root:
            final_mappings[entity_id] = root

    return final_mappings


async def emit_node_property_claims(
    claims_service: Optional[ClaimsService],
    nodes: List[BaseNode],
    transform_id: Optional[str],
    user_id: Optional[str],
) -> None:
    """B1-prob slice 2b: emit one Claim per (node, property) pair
    so the /contradictions surface can group cross-chunk
    disagreements about the same entity.

    Called from per-chunk extraction loops BEFORE any dedup-by-
    content check, so claims for "Alice extracted with
    title=Engineer (chunk 1)" and "Alice extracted with
    title=Senior Engineer (chunk 2)" both land — the
    contradiction detector groups them by canonical_id +
    property_key and surfaces the disagreement.

    Lives in :mod:`transform.helpers` (slice 2b reviewer-fix,
    moved from ``graph_transformer.py``) so the multi-pass
    extractor can call it without a circular import — both
    paths share the same emission semantics.

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
