from typing import List, Callable, Tuple, Optional
from pathlib import Path
from prefect import task, get_run_logger
import yaml
import traceback
from app.services.transform.models import (
    KnowledgeGraph,
    ExtractionMetrics,
    OntologyDefinition
)
from app.services.transform.graph_builder import (
    OntologyParser,
    KnowledgeGraphBuilder
)
from app.services.llm.client import LLMClient
from app.config import settings
from app.utils.logger import logger

def log_extraction_metrics(metrics: ExtractionMetrics, transform_id: str) -> None:
    """Log extraction metrics to Prefect"""
    logger = get_run_logger()
    
    # Calculate summary metrics
    success_rate = (
        metrics.successful_chunks / metrics.total_chunks
        if metrics.total_chunks > 0 else 0
    )
    avg_extraction_time = (
        sum(metrics.extraction_times) / len(metrics.extraction_times)
        if metrics.extraction_times else 0
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
                    "peak_memory_mb": metrics.peak_memory_mb
                },
                "llm_usage": metrics.llm_token_usage,
                "entity_resolution": metrics.entity_resolution_stats
            }
        }
    )
    
    # Log failed chunks for investigation
    if metrics.failed_chunks:
        logger.warning(
            f"Failed chunks: {len(metrics.failed_chunks)}",
            extra={
                "transform_id": transform_id,
                "failed_chunks": metrics.failed_chunks
            }
        )

async def load_and_validate_ontology(
    ontology_path: Path
) -> OntologyDefinition:
    """Load and validate ontology file"""
    try:
        with open(ontology_path) as f:
            ontology_yaml = f.read()
        
        # Parse and validate
        ontology_dict = yaml.safe_load(ontology_yaml)
        return OntologyDefinition(**ontology_dict)
        
    except Exception as e:
        raise ValueError(f"Invalid ontology file: {str(e)}")

@task(
    name="ontology-extraction",
    retries=settings.TRANSFORM_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS
)
async def construct_knowledge_graph(
    chunks: List[str],
    ontology_path: Path,
    transform_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[Optional[KnowledgeGraph], Optional[ExtractionMetrics]]:
    """Construct knowledge graph from chunks using ontology"""
    logger = get_run_logger()
    
    try:
        logger.info(f"Processing {len(chunks)} chunks for transform {transform_id}")
        
        # Load and validate ontology
        parser = OntologyParser(ontology_path)
        parser.validate()
        
        # Initialize builder
        builder = KnowledgeGraphBuilder(parser)
        
        # Process chunks
        graph, metrics = await builder.process_chunks(chunks, progress_callback)
        
        # Log metrics
        logger.info(
            f"Extraction completed: "
            f"{metrics.new_nodes} new nodes, {metrics.merged_nodes} merged nodes, "
            f"{metrics.total_relationships} relationships"
        )
        
        return graph, metrics
        
    except Exception as e:
        logger.error(f"Knowledge graph extraction failed: {str(e)}")
        traceback.print_exc()
        return None, None
