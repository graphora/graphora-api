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
            )
        elif pdf_paths:
            graph = await build_graph_from_pdfs(
                ontology_parser=parser,
                pdf_paths=pdf_paths,
                transform_id=transform_id,
                progress_callback=progress_callback,
                user_id=user_id,
                chunk_metadatas=chunk_metadatas,
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
