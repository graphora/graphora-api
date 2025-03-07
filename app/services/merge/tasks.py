"""Task definitions for merge operations"""
from typing import Dict, Any, Optional, List
from prefect import task
from prefect.tasks import NO_CACHE
from datetime import datetime, timezone
import logging
import time
from redis import Redis
from app.services.merge.models import (
    MergeStage,
    GraphResponse,
    EntityMappingResult,
    EntityMatch,
    Node,
    Edge
)
from app.services.merge.progress import ProgressTracker
from app.services.storage.interface import GraphStorageInterface
from app.services.ontology import load_ontology
from app.schemas.graph import Node as SchemaNode, Edge as SchemaEdge
from app.schemas.conflicts import ConflictType, ConflictSeverity
from app.services.merge.conflict import ConflictDetectionService
from app.config import settings

logger = logging.getLogger(__name__)

async def get_matching_nodes(
    production_storage: GraphStorageInterface,
    node: Node,
    similarity_threshold: float
) -> EntityMatch:
    """Find matching nodes in production graph"""
    # Find similar nodes in production
    similar_nodes = await production_storage.find_similar_nodes(
        node.label,
        node.properties,
        similarity_threshold
    )
    
    # Calculate similarity scores (for now just return 1.0 for all matches)
    similarity_scores = [1.0] * len(similar_nodes)
    
    return EntityMatch(
        staging_node=node,
        production_matches=similar_nodes,
        similarity_scores=similarity_scores
    )

@task(cache_policy=NO_CACHE)
async def start_stage(
    merge_id: str,
    stage: MergeStage,
    progress_tracker: ProgressTracker
) -> None:
    """Start a merge stage"""
    await progress_tracker.start_merge_stage(merge_id, stage)

@task(cache_policy=NO_CACHE)
async def complete_merge_stage(
    merge_id: str,
    stage: MergeStage,
    progress_tracker: ProgressTracker,
    result: Optional[Dict[str, Any]] = None
) -> None:
    """Complete a merge stage"""
    await progress_tracker.complete_merge_stage(merge_id, stage, result)

@task(cache_policy=NO_CACHE)
async def fail_merge(
    merge_id: str,
    error: str,
    progress_tracker: ProgressTracker,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Mark merge as failed"""
    await progress_tracker.fail_merge(merge_id, error, details)

@task(name="map_production_entities", cache_policy=NO_CACHE)
async def map_production_entities(
    staging_storage: GraphStorageInterface,
    production_storage: GraphStorageInterface,
    graph: GraphResponse,
    similarity_threshold: float = 0.7,
    merge_id: str = None,
    progress_tracker: ProgressTracker = None
) -> EntityMappingResult:
    """Map staging entities to production entities"""
    start_time = time.time()
    
    # Start tracking progress if merge_id and progress_tracker are provided
    if merge_id and progress_tracker:
        await progress_tracker.start_merge_stage(merge_id, MergeStage.ANALYZE)
    
    # Get total entities
    total_entities = len(graph.nodes)
    matched_entities = 0
    matches = {}
    
    # Process nodes in batches for better progress tracking
    batch_size = 10
    
    for i in range(0, total_entities, batch_size):
        batch = graph.nodes[i:i + min(batch_size, total_entities - i)]
        batch_matched = 0
        
        for node in batch:
            # Find matching nodes in production
            production_matches = await get_matching_nodes(
                production_storage,
                node,
                similarity_threshold
            )
            
            # If matches found, add to matches dict
            if production_matches:
                matches[node.id] = EntityMatch(
                    staging_id=node.id,
                    staging_type=node.type,
                    production_matches=production_matches
                )
                matched_entities += 1
                batch_matched += 1
        
        # Update progress after each batch
        if merge_id and progress_tracker:
            items_processed = i + len(batch)
            percentage_matched = (matched_entities / total_entities) * 100 if total_entities > 0 else 0
            
            await progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.ANALYZE,
                items_processed,
                total_entities,
                {
                    "matched_entities": matched_entities,
                    "percentage_matched": round(percentage_matched, 2),
                    "current_batch": i // batch_size + 1,
                    "total_batches": (total_entities + batch_size - 1) // batch_size,
                    "batch_matched": batch_matched
                }
            )
    
    # Calculate mapping time
    mapping_time_ms = (time.time() - start_time) * 1000
    
    # Create result
    result = EntityMappingResult(
        matches=matches,
        total_entities=total_entities,
        matched_entities=matched_entities,
        mapping_time_ms=mapping_time_ms
    )
    
    # Complete stage if merge_id and progress_tracker are provided
    if merge_id and progress_tracker:
        percentage_matched = (matched_entities / total_entities) * 100 if total_entities > 0 else 0
        
        await progress_tracker.complete_merge_stage(
            merge_id,
            MergeStage.ANALYZE,
            {
                "total_entities": total_entities,
                "matched_entities": matched_entities,
                "mapping_time_ms": mapping_time_ms,
                "matching_percentage": round(percentage_matched, 2)
            }
        )
    
    return result

@task(cache_policy=NO_CACHE)
async def validate_graph(
    graph: GraphResponse,
    ontology_id: str,
    merge_id: str = None,
    progress_tracker: ProgressTracker = None
) -> bool:
    """Validate graph against ontology"""
    try:
        # Start tracking progress if merge_id and progress_tracker are provided
        if merge_id and progress_tracker:
            await progress_tracker.start_merge_stage(merge_id, MergeStage.VALIDATION)
            
        # Load ontology
        ontology = await load_ontology(ontology_id)
        
        # Update progress after loading ontology
        if merge_id and progress_tracker:
            await progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.VALIDATION,
                1,  # items_processed (ontology loaded)
                len(graph.nodes) + len(graph.edges) + 1,  # items_total (nodes + edges + ontology)
                {"ontology_loaded": True, "ontology_id": ontology_id}
            )
        
        # Validate nodes
        valid_nodes = 0
        invalid_nodes = 0
        validation_issues = []
        
        # Process nodes in batches for better progress tracking
        node_batch_size = 20
        total_nodes = len(graph.nodes)
        
        for i in range(0, total_nodes, node_batch_size):
            batch = graph.nodes[i:i + min(node_batch_size, total_nodes - i)]
            batch_valid = 0
            batch_invalid = 0
            
            for node in batch:
                # Check if node type exists in ontology
                if node.type not in ontology.node_types:
                    logger.error(f"Invalid node type: {node.type}")
                    invalid_nodes += 1
                    batch_invalid += 1
                    validation_issues.append({
                        "entity_id": node.id,
                        "entity_type": "node",
                        "issue_type": "invalid_type",
                        "message": f"Node type '{node.type}' does not exist in ontology"
                    })
                else:
                    valid_nodes += 1
                    batch_valid += 1
            
            # Update progress after each batch of nodes
            if merge_id and progress_tracker:
                items_processed = i + len(batch) + 1  # +1 for ontology
                items_total = total_nodes + len(graph.edges) + 1  # +1 for ontology
                
                await progress_tracker.update_merge_progress(
                    merge_id,
                    MergeStage.VALIDATION,
                    items_processed,
                    items_total,
                    {
                        "valid_nodes": valid_nodes,
                        "invalid_nodes": invalid_nodes,
                        "current_node_batch": i // node_batch_size + 1,
                        "total_node_batches": (total_nodes + node_batch_size - 1) // node_batch_size,
                        "batch_valid_nodes": batch_valid,
                        "batch_invalid_nodes": batch_invalid
                    }
                )
        
        # Validate edges
        valid_edges = 0
        invalid_edges = 0
        
        # Process edges in batches for better progress tracking
        edge_batch_size = 20
        total_edges = len(graph.edges)
        
        for i in range(0, total_edges, edge_batch_size):
            batch = graph.edges[i:i + min(edge_batch_size, total_edges - i)]
            batch_valid = 0
            batch_invalid = 0
            
            for edge in batch:
                # Check if edge type exists in ontology
                if edge.type not in ontology.relationship_types:
                    logger.error(f"Invalid edge type: {edge.type}")
                    invalid_edges += 1
                    batch_invalid += 1
                    validation_issues.append({
                        "entity_id": edge.id,
                        "entity_type": "edge",
                        "issue_type": "invalid_type",
                        "message": f"Edge type '{edge.type}' does not exist in ontology"
                    })
                else:
                    valid_edges += 1
                    batch_valid += 1
            
            # Update progress after each batch of edges
            if merge_id and progress_tracker:
                items_processed = total_nodes + i + len(batch) + 1  # +1 for ontology
                items_total = total_nodes + total_edges + 1  # +1 for ontology
                
                await progress_tracker.update_merge_progress(
                    merge_id,
                    MergeStage.VALIDATION,
                    items_processed,
                    items_total,
                    {
                        "valid_nodes": valid_nodes,
                        "invalid_nodes": invalid_nodes,
                        "valid_edges": valid_edges,
                        "invalid_edges": invalid_edges,
                        "current_edge_batch": i // edge_batch_size + 1,
                        "total_edge_batches": (total_edges + edge_batch_size - 1) // edge_batch_size,
                        "batch_valid_edges": batch_valid,
                        "batch_invalid_edges": batch_invalid
                    }
                )
        
        # Determine if graph is valid
        is_valid = invalid_nodes == 0 and invalid_edges == 0
        
        # Complete stage if tracker is provided
        if merge_id and progress_tracker:
            await progress_tracker.complete_merge_stage(
                merge_id, 
                MergeStage.VALIDATION,
                {
                    "is_valid": is_valid,
                    "total_nodes": len(graph.nodes),
                    "valid_nodes": valid_nodes,
                    "invalid_nodes": invalid_nodes,
                    "total_edges": len(graph.edges),
                    "valid_edges": valid_edges,
                    "invalid_edges": invalid_edges,
                    "validation_issues": validation_issues[:10]  # Limit to first 10 issues
                }
            )
            
        return is_valid
        
    except Exception as e:
        logger.error(f"Failed to validate graph: {str(e)}")
        
        # Fail stage if tracker is provided
        if merge_id and progress_tracker:
            await progress_tracker.fail_merge_stage(
                merge_id,
                MergeStage.VALIDATION,
                f"Failed to validate graph: {str(e)}"
            )
            
        raise

@task(name="detect_conflicts", cache_policy=NO_CACHE)
async def detect_merge_conflicts(
    merge_id: str,
    graph: GraphResponse,
    entity_mapping: EntityMappingResult,
    storage: GraphStorageInterface,
    production_storage: GraphStorageInterface,
    progress_tracker: ProgressTracker
) -> None:
    """Detect conflicts between staging and production graphs"""
    redis_client = None
    conflict_service = None
    
    try:
        # Start tracking progress
        await progress_tracker.start_merge_stage(merge_id, MergeStage.CONFLICT_DETECTION)
        
        # Initialize Redis client
        redis_client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        
        # Initialize conflict detection service
        conflict_service = ConflictDetectionService(production_storage, redis_client)
        
        # Update progress after initialization
        await progress_tracker.update_merge_progress(
            merge_id,
            MergeStage.CONFLICT_DETECTION,
            1,  # items_processed (initialization)
            len(graph.nodes) + len(graph.edges) + 1,  # items_total (nodes + edges + initialization)
            {"initialized": True, "total_entities": len(graph.nodes) + len(graph.edges)}
        )
        
        # Set up progress callback for real-time updates
        async def progress_callback(processed_items, total_items, conflict_count, current_phase=None):
            metrics = {
                "conflicts_detected": conflict_count,
                "processed_percentage": round((processed_items / total_items) * 100, 2) if total_items > 0 else 0
            }
            
            if current_phase:
                metrics["current_phase"] = current_phase
                
            await progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.CONFLICT_DETECTION,
                processed_items,
                total_items,
                metrics
            )
        
        # Detect conflicts with progress tracking
        conflict_batch = await conflict_service.detect_conflicts(
            merge_id,
            graph.nodes,
            graph.edges,
            {
                staging_id: match.production_matches
                for staging_id, match in entity_mapping.matches.items()
            },
            progress_callback=progress_callback
        )
        
        # Update conflicts in progress tracker
        await progress_tracker.update_conflicts(merge_id, conflict_batch)
        
        # Complete stage with detailed metrics
        conflict_severity_counts = {
            severity.value: sum(1 for c in conflict_batch.conflicts if c.severity == severity)
            for severity in ConflictSeverity
        }
        
        conflict_type_counts = {
            conflict_type.value: sum(1 for c in conflict_batch.conflicts if c.conflict_type == conflict_type)
            for conflict_type in ConflictType
        }
        
        await progress_tracker.complete_merge_stage(
            merge_id, 
            MergeStage.CONFLICT_DETECTION,
            {
                "total_conflicts": len(conflict_batch.conflicts),
                "node_conflicts": sum(1 for c in conflict_batch.conflicts if c.entity_type == "node"),
                "edge_conflicts": sum(1 for c in conflict_batch.conflicts if c.entity_type == "edge"),
                "conflict_severity_counts": conflict_severity_counts,
                "conflict_type_counts": conflict_type_counts,
                "conflict_groups": len(conflict_batch.conflict_groups),
                "auto_resolvable": sum(1 for c in conflict_batch.conflicts if c.severity == ConflictSeverity.MINOR)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to detect conflicts: {str(e)}")
        
        # Fail stage
        await progress_tracker.fail_merge_stage(
            merge_id,
            MergeStage.CONFLICT_DETECTION,
            f"Failed to detect conflicts: {str(e)}"
        )
        
        raise
        
    finally:
        # Clean up resources
        if conflict_service and hasattr(conflict_service, 'executor'):
            conflict_service.executor.shutdown(wait=False)
        if redis_client:
            redis_client.close()