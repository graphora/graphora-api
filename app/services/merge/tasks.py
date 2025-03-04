"""Task definitions for merge operations"""
from typing import Dict, Any, Optional, List
from prefect import task
from prefect.tasks import NO_CACHE
from datetime import datetime, timezone
import logging
import time

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

@task(cache_policy=NO_CACHE)
async def extract_staging_graph(
    storage: GraphStorageInterface,
    transform_id: str,
    merge_id: str = None,
    progress_tracker: ProgressTracker = None
) -> GraphResponse:
    """Extract graph from staging storage"""
    start_time = time.time()
    
    # Start tracking progress if merge_id and progress_tracker are provided
    if merge_id and progress_tracker:
        await progress_tracker.start_merge_stage(merge_id, MergeStage.EXTRACT)
        
    # Get nodes from storage
    nodes = await storage.get_nodes_by_transform_id(transform_id)
    
    # If no nodes found, log warning and complete stage
    if not nodes:
        logger.warning(f"No nodes found for transform {transform_id}")
        if merge_id and progress_tracker:
            await progress_tracker.complete_merge_stage(
                merge_id,
                MergeStage.EXTRACT,
                {"nodes": 0, "edges": 0, "extraction_time_ms": 0}
            )
        return GraphResponse(
            nodes=[],
            edges=[],
            total_nodes=0,
            total_edges=0,
            extraction_time_ms=0
        )
    
    # Update progress after retrieving nodes
    if merge_id and progress_tracker:
        await progress_tracker.update_merge_progress(
            merge_id,
            MergeStage.EXTRACT,
            len(nodes),  # items_processed
            len(nodes) * 2,  # items_total (nodes + estimated edges)
            {"nodes_retrieved": len(nodes)}
        )
    
    # Convert storage nodes to schema nodes
    schema_nodes = []
    batch_size = 100
    total_nodes = len(nodes)
    
    for i in range(0, total_nodes, batch_size):
        batch = nodes[i:i + batch_size]
        batch_nodes = []
        
        for node in batch:
            schema_nodes.append(
                Node(
                    id=node.id,
                    type=node.type,
                    properties=node.properties
                )
            )
            batch_nodes.append(node.id)
        
        # Update progress after processing each batch of nodes
        if merge_id and progress_tracker and i + batch_size < total_nodes:
            await progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.EXTRACT,
                i + len(batch),  # items_processed
                total_nodes * 2,  # items_total (nodes + estimated edges)
                {"nodes_processed": i + len(batch)}
            )
    
    # Get relationships between nodes
    edges = await storage.get_edges_by_transform_id(transform_id)
    
    # Update progress after retrieving edges
    if merge_id and progress_tracker:
        await progress_tracker.update_merge_progress(
            merge_id,
            MergeStage.EXTRACT,
            total_nodes + len(edges),  # items_processed (nodes + edges)
            total_nodes + len(edges),  # items_total (actual count)
            {
                "nodes_processed": total_nodes,
                "edges_retrieved": len(edges)
            }
        )
    
    # Convert storage edges to schema edges
    schema_edges = []
    total_edges = len(edges)
    
    for i in range(0, total_edges, batch_size):
        batch = edges[i:i + batch_size]
        
        for edge in batch:
            schema_edges.append(
                Edge(
                    id=edge.id,
                    source=edge.source_id,
                    target=edge.target_id,
                    type=edge.type,
                    properties=edge.properties
                )
            )
        
        # Update progress after processing each batch of edges
        if merge_id and progress_tracker and i + batch_size < total_edges:
            items_processed = total_nodes + i + len(batch)
            total_items = total_nodes + total_edges
            
            await progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.EXTRACT,
                items_processed,
                total_items,
                {
                    "nodes_processed": total_nodes,
                    "edges_processed": i + len(batch)
                }
            )
    
    # Calculate extraction time
    extraction_time_ms = (time.time() - start_time) * 1000
    
    # Create response
    response = GraphResponse(
        nodes=schema_nodes,
        edges=schema_edges,
        total_nodes=len(schema_nodes),
        total_edges=len(schema_edges),
        extraction_time_ms=extraction_time_ms
    )
    
    # Complete stage if merge_id and progress_tracker are provided
    if merge_id and progress_tracker:
        await progress_tracker.complete_merge_stage(
            merge_id,
            MergeStage.EXTRACT,
            {
                "nodes": len(schema_nodes),
                "edges": len(schema_edges),
                "extraction_time_ms": extraction_time_ms
            }
        )
    
    return response

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
    try:
        # Start tracking progress
        await progress_tracker.start_merge_stage(merge_id, MergeStage.CONFLICT_DETECTION)
        
        from app.services.merge.conflict import ConflictDetectionService
        
        # Initialize conflict detection service
        conflict_service = ConflictDetectionService(production_storage)
        
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