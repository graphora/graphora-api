from typing import List
from datetime import datetime
import psutil
from prefect import task, get_run_logger
import traceback
from app.services.storage.neo4j import Neo4jStorage
from app.services.storage.models import (
    StorageResult,
    StorageStage,
    StorageMetrics,
    StorageError
)
from app.services.transform.models import DocumentKnowledgeGraph
from app.config import settings

def chunk_list(items: List, size: int) -> List[List]:
    """Split list into chunks of specified size"""
    return [
        items[i:i + size]
        for i in range(0, len(items), size)
    ]

def log_storage_metrics(metrics: StorageMetrics, transform_id: str) -> None:
    """Log storage metrics to Prefect"""
    logger = get_run_logger()
    
    logger.info(
        "Storage Metrics",
        extra={
            "transform_id": transform_id,
            "metrics": {
                "nodes": metrics.nodes_processed,
                "relationships": metrics.relationships_processed,
                "time_ms": metrics.storage_time_ms,
                "retries": metrics.retries,
                "errors": len(metrics.errors),
                "avg_batch_time": metrics.avg_batch_time,
                "success_rate": metrics.success_rate,
                "peak_memory_mb": metrics.peak_memory_mb,
                "checkpoint_count": metrics.checkpoint_count
            }
        }
    )
    
    # Log errors if any
    if metrics.errors:
        logger.warning(
            f"Storage errors: {len(metrics.errors)}",
            extra={
                "transform_id": transform_id,
                "errors": metrics.errors
            }
        )

@task(
    name="store-knowledge-graph",
    retries=settings.STORAGE_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
    persist_result=True
)
async def store_knowledge_graph(
    graph: DocumentKnowledgeGraph,
    transform_id: str,
    checkpoint_size: int = settings.STORAGE_BATCH_SIZE
) -> StorageResult:
    """
    Store knowledge graph in Neo4j with checkpointing
    
    Args:
        graph: Knowledge graph to store
        transform_id: Transform ID for tracking
        checkpoint_size: Number of items per batch
        
    Returns:
        StorageResult with metrics
    """
    logger = get_run_logger()
    storage = Neo4jStorage(
        uri=settings.STAGING_NEO4J_URI,
        username=settings.STAGING_NEO4J_USER,
        password=settings.STAGING_NEO4J_PASSWORD,
        database=settings.STAGING_NEO4J_DATABASE
    )
    start_time = datetime.now()
    
    try:
        # Initialize result
        result = StorageResult(
            transform_id=transform_id,
            nodes_stored=0,
            relationships_stored=0,
            start_time=start_time,
            status=StorageStage.NODES,
            metrics=StorageMetrics()
        )
        
        # Get current status
        status = await storage.get_storage_status(transform_id)
        start_from = status.last_processed_index if status else 0
        current_stage = status.stage if status else StorageStage.NODES
        
        # Convert nodes to list if needed
        nodes = graph.nodes
        
        # Process nodes if not completed
        if current_stage == StorageStage.NODES:
            logger.info(
                f"Processing nodes from index {start_from}",
                extra={"transform_id": transform_id}
            )
            node_batches = chunk_list(
                nodes[start_from:],
                checkpoint_size
            )
            
            for batch_idx, node_batch in enumerate(
                node_batches,
                start=start_from
            ):
                try:
                    # Store batch
                    batch_result = await storage.store_nodes(
                        node_batch,
                        batch_idx,
                        transform_id,
                        merge=False
                    )
                    
                    # Update metrics
                    result.nodes_stored += batch_result.items_processed
                    result.metrics.nodes_processed += batch_result.items_processed
                    result.metrics.add_batch_timing(
                        batch_result.processing_time_ms
                    )
                    
                    if batch_result.warnings:
                        for warning in batch_result.warnings:
                            result.metrics.add_error(
                                warning,
                                batch_idx,
                                StorageStage.NODES
                            )
                    
                    # Update checkpoint
                    await storage.update_checkpoint(
                        transform_id,
                        batch_idx * checkpoint_size + len(node_batch),
                        StorageStage.NODES
                    )
                    result.metrics.checkpoint_count += 1
                    
                    logger.info(
                        f"Stored node batch {batch_idx}",
                        extra={
                            "transform_id": transform_id,
                            "processed": batch_result.items_processed,
                            "time_ms": batch_result.processing_time_ms
                        }
                    )
                    
                except Exception as e:
                    traceback.print_exc()
                    result.metrics.add_error(
                        str(e),
                        batch_idx,
                        StorageStage.NODES
                    )
                    raise StorageError(
                        f"Failed at node batch {batch_idx}: {str(e)}"
                    )
            
            # Update stage
            current_stage = StorageStage.RELATIONSHIPS
            await storage.update_checkpoint(
                transform_id,
                0,  # Reset index for relationships
                current_stage
            )
        
        # Process relationships
        if current_stage == StorageStage.RELATIONSHIPS:
            logger.info(
                f"Processing relationships from index {start_from}",
                extra={"transform_id": transform_id}
            )
            
            rel_batches = chunk_list(
                graph.relationships[start_from:],
                checkpoint_size
            )
            
            for batch_idx, rel_batch in enumerate(
                rel_batches,
                start=start_from
            ):
                try:
                    # Store batch
                    batch_result = await storage.store_relationships(
                        rel_batch,
                        batch_idx,
                        transform_id
                    )
                    
                    # Update metrics
                    result.relationships_stored += batch_result.items_processed
                    result.metrics.relationships_processed += (
                        batch_result.items_processed
                    )
                    result.metrics.add_batch_timing(
                        batch_result.processing_time_ms
                    )
                    
                    if batch_result.warnings:
                        for warning in batch_result.warnings:
                            result.metrics.add_error(
                                warning,
                                batch_idx,
                                StorageStage.RELATIONSHIPS
                            )
                    
                    # Update checkpoint
                    await storage.update_checkpoint(
                        transform_id,
                        batch_idx * checkpoint_size + len(rel_batch),
                        StorageStage.RELATIONSHIPS
                    )
                    result.metrics.checkpoint_count += 1
                    
                    logger.info(
                        f"Stored relationship batch {batch_idx}",
                        extra={
                            "transform_id": transform_id,
                            "processed": batch_result.items_processed,
                            "time_ms": batch_result.processing_time_ms
                        }
                    )
                    
                except Exception as e:
                    traceback.print_exc()
                    result.metrics.add_error(
                        str(e),
                        batch_idx,
                        StorageStage.RELATIONSHIPS
                    )
                    raise StorageError(
                        f"Failed at relationship batch {batch_idx}: {str(e)}"
                    )
        
        # Finalize result
        result.finalize(StorageStage.COMPLETED)
        result.metrics.storage_time_ms = (
            datetime.now() - start_time
        ).total_seconds() * 1000
        result.metrics.peak_memory_mb = (
            psutil.Process().memory_info().rss / 1024 / 1024
        )
        
        # Log metrics
        log_storage_metrics(result.metrics, transform_id)
        
        # Verify stored data
        stored_data = await storage.get_transformation_data(transform_id)
        logger.info(
            "Stored data verification",
            extra={
                "transform_id": transform_id,
                "nodes": stored_data.total_nodes,
                "relationships": stored_data.total_edges,
                "node_types": stored_data.nodes,
                "relationship_types": stored_data.edges
            }
        )
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        logger.error(
            f"Storage failed: {str(e)}",
            extra={"transform_id": transform_id}
        )
        raise
    
    finally:
        await storage.close()
