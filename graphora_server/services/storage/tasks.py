from typing import List
from datetime import datetime
import psutil
from prefect import task, get_run_logger
import traceback
from graphora_server.services.storage.models import (
    StorageResult,
    StorageStage,
    StorageMetrics,
    StorageError,
)
from graphora_server.services.storage.factory import (
    create_storage_for_user,
)
from graphora_server.services.transform.models import DocumentKnowledgeGraph
from graphora_server.config import settings


def chunk_list(items: List, size: int) -> List[List]:
    """Split list into chunks of specified size"""
    return [items[i : i + size] for i in range(0, len(items), size)]


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
                "checkpoint_count": metrics.checkpoint_count,
            },
        },
    )

    # Log errors if any
    if metrics.errors:
        logger.warning(
            f"Storage errors: {len(metrics.errors)}",
            extra={"transform_id": transform_id, "errors": metrics.errors},
        )


@task(
    name="store-knowledge-graph",
    retries=settings.STORAGE_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
    persist_result=True,
)
async def store_knowledge_graph(
    graph: DocumentKnowledgeGraph,
    transform_id: str,
    user_id: str,
    checkpoint_size: int = settings.STORAGE_BATCH_SIZE,
) -> StorageResult:
    """
    Store knowledge graph in Neo4j with checkpointing

    Args:
        graph: Knowledge graph to store
        transform_id: Transform ID for tracking
        user_id: User's ID for database configuration
        checkpoint_size: Number of items per batch

    Returns:
        StorageResult with metrics
    """
    logger = get_run_logger()

    # Create storage using factory (Neo4j, Apache AGE, or in-memory)
    storage = await create_storage_for_user(user_id, use_staging=True)
    start_time = datetime.now()

    # Log storage type being used. Read from settings.STORAGE_TYPE
    # rather than the old `memory or neo4j` ternary so that future
    # backends (postgres/AGE — Gate 5) get labelled correctly in
    # metrics without another caller fix.
    storage_type = (settings.STORAGE_TYPE or "neo4j").lower()
    logger.info(
        f"Using {storage_type} storage for transform {transform_id}",
        extra={"transform_id": transform_id, "storage_type": storage_type},
    )

    try:
        # Initialize result
        result = StorageResult(
            transform_id=transform_id,
            nodes_stored=0,
            relationships_stored=0,
            start_time=start_time,
            status=StorageStage.NODES,
            metrics=StorageMetrics(),
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
                extra={"transform_id": transform_id},
            )
            node_batches = chunk_list(nodes[start_from:], checkpoint_size)

            for batch_idx, node_batch in enumerate(node_batches, start=start_from):
                try:
                    # Store batch
                    batch_result = await storage.store_nodes(
                        node_batch, batch_idx, transform_id, merge=False
                    )

                    # Update metrics — items_processed is the actual
                    # count, not the assumed batch size, so partial-
                    # success batches show real numbers in metrics.
                    result.nodes_stored += batch_result.items_processed
                    result.metrics.nodes_processed += batch_result.items_processed
                    result.metrics.add_batch_timing(batch_result.processing_time_ms)

                    if batch_result.warnings:
                        for warning in batch_result.warnings:
                            result.metrics.add_error(
                                warning, batch_idx, StorageStage.NODES
                            )

                    # Honor partial-success contract — adapters return
                    # success=False with items_processed < len(batch)
                    # when a sub-batch fails. Advancing the checkpoint
                    # past the actual write head causes silent data
                    # loss on resume, so refuse to advance and stop
                    # the loop.
                    if not batch_result.success:
                        raise StorageError(
                            f"store_nodes batch {batch_idx} reported failure: "
                            f"{batch_result.error or 'unknown'} "
                            f"({batch_result.items_processed} of "
                            f"{len(node_batch)} stored)"
                        )

                    # Checkpoint at the actual write head, not the
                    # batch tail — keeps resume correct even if the
                    # adapter returns a partial-success state in the
                    # future without raising.
                    advanced_index = (
                        batch_idx * checkpoint_size + batch_result.items_processed
                    )
                    await storage.update_checkpoint(
                        transform_id,
                        advanced_index,
                        StorageStage.NODES,
                    )
                    result.metrics.checkpoint_count += 1

                    logger.info(
                        f"Stored node batch {batch_idx}",
                        extra={
                            "transform_id": transform_id,
                            "processed": batch_result.items_processed,
                            "time_ms": batch_result.processing_time_ms,
                        },
                    )

                except Exception as e:
                    traceback.print_exc()
                    result.metrics.add_error(str(e), batch_idx, StorageStage.NODES)
                    raise StorageError(f"Failed at node batch {batch_idx}: {str(e)}")

            # Update stage
            current_stage = StorageStage.RELATIONSHIPS
            await storage.update_checkpoint(
                transform_id, 0, current_stage  # Reset index for relationships
            )

        # Process relationships
        if current_stage == StorageStage.RELATIONSHIPS:
            logger.info(
                f"Processing relationships from index {start_from}",
                extra={"transform_id": transform_id},
            )

            rel_batches = chunk_list(graph.relationships[start_from:], checkpoint_size)

            for batch_idx, rel_batch in enumerate(rel_batches, start=start_from):
                try:
                    # Store batch
                    batch_result = await storage.store_relationships(
                        rel_batch, batch_idx, transform_id
                    )

                    # Update metrics with actual processed count, not
                    # assumed batch size.
                    result.relationships_stored += batch_result.items_processed
                    result.metrics.relationships_processed += (
                        batch_result.items_processed
                    )
                    result.metrics.add_batch_timing(batch_result.processing_time_ms)

                    if batch_result.warnings:
                        for warning in batch_result.warnings:
                            result.metrics.add_error(
                                warning, batch_idx, StorageStage.RELATIONSHIPS
                            )

                    # Honor partial-success contract — same reason as
                    # the nodes loop above. Advancing the checkpoint
                    # past failed writes causes silent data loss on
                    # resume.
                    if not batch_result.success:
                        raise StorageError(
                            f"store_relationships batch {batch_idx} reported "
                            f"failure: {batch_result.error or 'unknown'} "
                            f"({batch_result.items_processed} of "
                            f"{len(rel_batch)} stored)"
                        )

                    advanced_index = (
                        batch_idx * checkpoint_size + batch_result.items_processed
                    )
                    await storage.update_checkpoint(
                        transform_id,
                        advanced_index,
                        StorageStage.RELATIONSHIPS,
                    )
                    result.metrics.checkpoint_count += 1

                    logger.info(
                        f"Stored relationship batch {batch_idx}",
                        extra={
                            "transform_id": transform_id,
                            "processed": batch_result.items_processed,
                            "time_ms": batch_result.processing_time_ms,
                        },
                    )

                except Exception as e:
                    traceback.print_exc()
                    result.metrics.add_error(
                        str(e), batch_idx, StorageStage.RELATIONSHIPS
                    )
                    raise StorageError(
                        f"Failed at relationship batch {batch_idx}: {str(e)}"
                    )

        # Finalize result
        result.finalize(StorageStage.COMPLETED)
        result.metrics.storage_time_ms = (
            datetime.now() - start_time
        ).total_seconds() * 1000
        result.metrics.peak_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

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
                "relationship_types": stored_data.edges,
            },
        )

        return result

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Storage failed: {str(e)}", extra={"transform_id": transform_id})
        raise

    finally:
        await storage.close()
