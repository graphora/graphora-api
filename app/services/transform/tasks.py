from typing import List, Tuple
from pathlib import Path
from prefect import task, get_run_logger
import yaml
import json
import asyncio

from app.services.transform.graph_builder import (
    ModelGenerator,
    KnowledgeGraphBuilder
)
from app.services.transform.models import (
    KnowledgeGraph,
    ExtractionMetrics,
    OntologyDefinition
)
from app.config import settings

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
    name="knowledge-graph-construction",
    retries=settings.EXTRACTION_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
    tags=["processing", "knowledge-graph"]
)
async def construct_knowledge_graph(
    chunks: List[str],
    ontology_path: Path,
    transform_id: str,
    chunk_batch_size: int = settings.CHUNK_BATCH_SIZE
) -> Tuple[KnowledgeGraph, ExtractionMetrics]:
    """
    Construct knowledge graph from document chunks
    
    Args:
        chunks: List of text chunks to process
        ontology_path: Path to ontology YAML file
        transform_id: Transform ID for tracking
        chunk_batch_size: Number of chunks to process in parallel
        
    Returns:
        Tuple of (KnowledgeGraph, ExtractionMetrics)
    """
    logger = get_run_logger()
    
    try:
        # Load and validate ontology
        ontology = await load_and_validate_ontology(ontology_path)
        logger.info(
            f"Loaded ontology with {len(ontology.entities)} entities"
        )
        
        # Generate models
        model_gen = ModelGenerator(yaml.dump(ontology.model_dump()))
        models = model_gen.generate_models()
        logger.info(f"Generated {len(models)} Pydantic models")
        
        # Initialize graph builder
        builder = KnowledgeGraphBuilder(models)
        builder.graph.metrics.total_chunks = len(chunks)
        
        # Process chunks in batches
        for i in range(0, len(chunks), chunk_batch_size):
            batch = chunks[i:i + chunk_batch_size]
            
            # Process batch in parallel
            tasks = [
                builder.process_chunk(
                    chunk,
                    f"chunk_{i+j}"
                )
                for j, chunk in enumerate(batch)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Add successful extractions to graph
            for result in results:
                if isinstance(result, list):  # Successful extraction
                    builder.add_nodes_to_graph(result)
                else:  # Exception occurred
                    logger.error(
                        f"Chunk processing failed: {str(result)}"
                    )
        
        # Finalize graph
        final_graph = builder.finalize_graph()
        
        # Log metrics
        log_extraction_metrics(final_graph.metrics, transform_id)
        
        # Save graph state
        graph_path = (
            Path(settings.UPLOAD_DIR) / 
            transform_id / 
            "knowledge_graph.json"
        )
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(graph_path, "w") as f:
            json.dump({
                "nodes": {
                    node_id: node.model_dump()
                    for node_id, node in final_graph.nodes.items()
                },
                "relationships": [
                    rel.model_dump() for rel in final_graph.relationships
                ],
                "metrics": final_graph.metrics.model_dump()
            }, f, indent=2, default=str)
        
        logger.info(
            f"Knowledge graph saved to {graph_path}",
            extra={
                "transform_id": transform_id,
                "node_count": len(final_graph.nodes),
                "relationship_count": len(final_graph.relationships)
            }
        )
        
        return final_graph, final_graph.metrics
        
    except Exception as e:
        logger.error(
            f"Graph construction failed: {str(e)}",
            extra={"transform_id": transform_id}
        )
        raise
