from typing import Any, List, Callable, Tuple, Optional, Union
from pathlib import Path
from prefect import task, get_run_logger
from datetime import datetime, timezone
import yaml
import traceback
from graphora_server.services.transform.models import (
    DocumentKnowledgeGraph,
    ExtractionMetrics,
    OntologyDefinition,
)
from graphora_server.services.transform.ontology_helper import OntologyParser
from graphora_server.services.transform.graph_transformer import (
    build_graph_from_chunks,
    build_graph_from_pdfs,
)
from graphora_server.services.claims_service import ClaimsService
from graphora_server.services.decision_log_service import DecisionLogService
from graphora_server.services.disputed_pairs_service import DisputedPairsService
from graphora_server.config import settings


class ExtractionError(RuntimeError):
    """Raised when knowledge graph extraction fails irrecoverably."""

    def __init__(self, message: str, original: Optional[Exception] = None):
        super().__init__(message)
        self.original = original


def log_extraction_metrics(metrics: ExtractionMetrics, transform_id: str) -> None:
    """Log extraction metrics to Prefect"""
    logger = get_run_logger()

    # Calculate summary metrics
    success_rate = (
        metrics.successful_chunks / metrics.total_chunks
        if metrics.total_chunks > 0
        else 0
    )
    avg_extraction_time = (
        sum(metrics.extraction_times) / len(metrics.extraction_times)
        if metrics.extraction_times
        else 0
    )

    logger.info(
        "Knowledge Graph Extraction Metrics",
        extra={
            "transform_id": transform_id,
            "metrics": {
                "success_rate": success_rate,
                "total_chunks": metrics.total_chunks,
                "failed_chunks": len(metrics.failed_chunks),
                "total_nodes": metrics.total_nodes,
                "total_relationships": metrics.total_relationships,
                "performance": {
                    "avg_extraction_time_ms": avg_extraction_time,
                    "peak_memory_mb": metrics.peak_memory_mb,
                },
                "llm_usage": metrics.llm_token_usage,
                "entity_resolution": metrics.entity_resolution_stats,
            },
        },
    )

    # Log failed chunks for investigation
    if metrics.failed_chunks:
        logger.warning(
            f"Failed chunks: {len(metrics.failed_chunks)}",
            extra={
                "transform_id": transform_id,
                "failed_chunks": metrics.failed_chunks,
            },
        )


async def load_and_validate_ontology(ontology_path: Path) -> OntologyDefinition:
    """Load and validate ontology file"""
    try:
        with open(ontology_path) as f:
            ontology_yaml = f.read()

        # Parse and validate
        ontology_dict = yaml.safe_load(ontology_yaml)
        return OntologyDefinition(**ontology_dict)

    except Exception as e:
        raise ValueError(f"Invalid ontology file: {str(e)}")


async def _resolve_extractor_model_name(user_id: Optional[str]) -> Optional[str]:
    """Resolve the LLM model name configured for ``user_id``.

    Used at the top of ``construct_knowledge_graph`` to stamp
    ``extractor_model`` on every emitted node/edge. Resolution
    matches ``get_baml_registry_for_user``'s precedence so the
    value we record matches the model BAML actually invokes:

      1. ``LLM_PROVIDER=ollama`` env var → ``OLLAMA_MODEL`` env value
      2. User's stored provider config → ``model_name`` field

    Returns None when:
      * ``user_id`` is None (test path / no auth)
      * The user has no AI config
      * Lookup fails (DB unavailable, etc.)

    None falls through to a None ``extractor_model`` on properties,
    which the Evidence tab renders as "model unknown" rather than
    failing extraction.
    """
    from graphora_server.config import get_settings

    settings_obj = get_settings()
    if (settings_obj.LLM_PROVIDER or "").lower() == "ollama":
        return settings_obj.OLLAMA_MODEL
    if not user_id:
        return None
    try:
        from graphora_server.services.ai_config_service import AIConfigService

        result = await AIConfigService().get_user_provider_secret(user_id)
        if not result:
            return None
        _provider, _api_key, model_name = result
        return model_name
    except Exception:
        return None


def should_retry_extraction_error(exc: Exception) -> bool:
    """Determine if extraction error should be retried"""
    error_msg = str(exc).lower()

    # Don't retry authentication/configuration errors
    non_retryable_patterns = [
        "api key not valid",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "invalid_argument",
        "permission denied",
        "quota exceeded",
        "billing",
        "api_key_invalid",
        "bamlclienthttperror",  # BAML client errors are often config issues
    ]

    for pattern in non_retryable_patterns:
        if pattern in error_msg:
            return False

    return True


@task(
    name="ontology-extraction",
    retries=settings.TRANSFORM_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
)
async def construct_knowledge_graph(
    ontology_path: Union[str, Path],
    transform_id: str,
    chunks: List[str] = [],
    pdf_paths: List[Path] = [],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    chunk_metadatas: Optional[List[Any]] = None,
) -> Tuple[Optional[DocumentKnowledgeGraph], Optional[ExtractionMetrics]]:
    """Construct knowledge graph from chunks using ontology.

    ``chunk_metadatas`` (optional, A1-prov): per-chunk metadata
    objects same length as ``chunks``. Forwarded to extraction so
    nodes/edges get source-span properties on their property bags.
    The PDF-binary path (``pdf_paths``) does not consume this — its
    chunks are filesystem paths, not text, and per-page metadata is
    derived elsewhere.
    """
    logger = get_run_logger()

    try:
        logger.info(f"Processing {len(chunks)} chunks for transform {transform_id}")

        # Load and validate ontology with user_id for Supabase fallback
        parser = OntologyParser(ontology_path, user_id)

        # B0-prov-extend: resolve the LLM model name once per
        # extraction batch (one DB hit, not per-chunk). The
        # downstream pipeline stamps it onto every emitted
        # NodeProvenance + node/edge property. None when no user
        # config exists (e.g., test mode) — graceful degrade.
        extractor_model = await _resolve_extractor_model_name(user_id)

        # B0-log slice 3b: per-transform Decision Log instance.
        # Constructed unconditionally — the service's dual backend
        # (Postgres when DATABASE_URL is set, in-memory list
        # otherwise) means dev mode runs with zero config and
        # production runs persist decisions. The instance is local
        # to this transform: no singleton, no cross-transform state
        # leak, garbage-collected when the task ends. Any per-row
        # write failure is logged-and-swallowed by append() so the
        # Decision Log can never block extraction itself.
        decision_log = DecisionLogService()

        # B2-active slice B: per-transform DisputedPairsService.
        # Same dual-backend pattern as the Decision Log (Postgres
        # when DATABASE_URL is set, shared in-memory store
        # otherwise). The graph builder's gray-zone hook
        # enqueues 2-node candidates that blockers grouped but
        # the LLM resolver split — "blocker said yes, LLM said
        # no" — for human/agent review. Hook failures are
        # swallowed by the helper so the disputed-pairs queue
        # can never block extraction.
        disputed_pairs_service = DisputedPairsService()

        # B1-prob slice 2b: per-transform ClaimsService for
        # emitting one Claim per (target, property) at extraction
        # time. Same dual-backend + log-and-swallow posture as
        # the Decision Log + DisputedPairsService — claim writes
        # must never block the extraction itself. Without this
        # the /contradictions endpoint shipped in slice 2a stays
        # empty because no writer populates the claims table.
        claims_service = ClaimsService()

        # Process chunks with controlled concurrency
        concurrency = settings.EXTRACTION_CONCURRENCY
        if len(chunks) < concurrency:
            concurrency = len(chunks)
        logger.info(
            f"Large document detected, using parallel processing with concurrency {concurrency}"
        )
        if len(chunks) == 0 and len(pdf_paths) == 0:
            return None, None
        if chunks:
            graph = await build_graph_from_chunks(
                ontology_parser=parser,
                chunks=chunks,
                transform_id=transform_id,
                progress_callback=progress_callback,
                user_id=user_id,
                chunk_metadatas=chunk_metadatas,
                extractor_model=extractor_model,
                decision_log=decision_log,
                disputed_pairs_service=disputed_pairs_service,
                claims_service=claims_service,
            )
        elif pdf_paths:
            graph = await build_graph_from_pdfs(
                ontology_parser=parser,
                pdf_paths=pdf_paths,
                transform_id=transform_id,
                progress_callback=progress_callback,
                user_id=user_id,
                chunk_metadatas=chunk_metadatas,
                extractor_model=extractor_model,
                decision_log=decision_log,
                disputed_pairs_service=disputed_pairs_service,
                claims_service=claims_service,
            )

        metrics = ExtractionMetrics(
            start_time=datetime.now(timezone.utc),
            total_nodes=len(graph.nodes),
            total_relationships=len(graph.relationships),
            merged_nodes=graph.metrics.merged_nodes if graph.metrics else 0,
        )

        # Finalize metrics
        if metrics:
            metrics.finalize()
            # Log metrics
            logger.info(
                f"Extraction completed: "
                f"{metrics.new_nodes} new nodes, {metrics.merged_nodes} merged nodes, "
                f"{metrics.total_relationships} relationships, "
                f"{metrics.failed_chunks}/{metrics.total_chunks} chunks failed"
            )

        return graph, metrics

    except Exception as e:
        logger.error(f"Knowledge graph extraction failed: {str(e)}")
        traceback.print_exc()
        raise ExtractionError(f"Knowledge graph extraction failed: {str(e)}", e)
