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
    transform_id: str
) -> GraphResponse:
    """Extract graph from staging area"""
    try:
        start_time = time.time()
        
        # Get nodes with transform_id
        storage_nodes = await storage.get_nodes_by_property("transform_id", transform_id)
        if not storage_nodes:
            logger.warning(f"No nodes found with transform_id {transform_id}")
            return GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)
            
        # Convert storage nodes to schema nodes
        nodes = [
            SchemaNode(
                id=str(node.id),
                label=node.label,
                type=node.type,
                properties=node.properties
            ) for node in storage_nodes
        ]
        
        # Get relationships between these nodes
        node_ids = [node.id for node in storage_nodes]
        storage_edges = await storage.get_relationships_between_nodes(node_ids)
        
        # Convert storage edges to schema edges
        edges = [
            SchemaEdge(
                id=str(edge.id),
                source=str(edge.source),
                target=str(edge.target),
                type=edge.type,
                properties=edge.properties
            ) for edge in storage_edges
        ]
        
        extraction_time_ms = (time.time() - start_time) * 1000
        
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges),
            extraction_time_ms=extraction_time_ms
        )
        
    except Exception as e:
        logger.error(f"Failed to extract staging graph: {str(e)}")
        raise

@task(name="map_production_entities", cache_policy=NO_CACHE)
async def map_production_entities(
    staging_storage: GraphStorageInterface,
    production_storage: GraphStorageInterface,
    graph: GraphResponse,
    similarity_threshold: float = 0.7
) -> EntityMappingResult:
    """Map staging entities to production entities"""
    start_time = datetime.now(timezone.utc)
    
    # Get production matches for each staging node
    total_entities = len(graph.nodes)
    matched_entities = 0
    matches = {}
    
    for node in graph.nodes:
        match = await get_matching_nodes(production_storage, node, similarity_threshold)
        matches[node.id] = match
        if match.production_matches:
            matched_entities += 1
            
    end_time = datetime.now(timezone.utc)
    mapping_time_ms = int((end_time - start_time).total_seconds() * 1000)
    
    return EntityMappingResult(
        matches=matches,
        total_entities=total_entities,
        matched_entities=matched_entities,
        mapping_time_ms=mapping_time_ms
    )

@task(cache_policy=NO_CACHE)
async def validate_graph(graph: GraphResponse, ontology_id: str) -> bool:
    """Validate graph against ontology"""
    try:
        # Load ontology
        ontology = await load_ontology(ontology_id)
        
        # Validate nodes
        for node in graph.nodes:
            # Check if node type exists in ontology
            if node.type not in ontology.get("entities", {}):
                logger.error(f"Invalid node type: {node.type}")
                return False
                
            # Get entity definition
            entity_def = ontology["entities"][node.type]
            
            # Validate required properties
            for prop_name, prop_def in entity_def.get("properties", {}).items():
                if prop_def.get("required", False):
                    if prop_name not in node.properties:
                        logger.error(f"Missing required property {prop_name} for node {node.id}")
                        return False
                        
        # Validate edges
        for edge in graph.edges:
            # Check if relationship type exists in ontology
            if edge.type not in ontology.get("relationships", {}):
                logger.error(f"Invalid relationship type: {edge.type}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate graph: {str(e)}")
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
    from app.services.merge.conflict import ConflictDetectionService
    
    conflict_service = ConflictDetectionService(production_storage)
    conflict_batch = await conflict_service.detect_conflicts(
        merge_id,
        graph.nodes,
        graph.edges,
        {
            staging_id: match.production_matches
            for staging_id, match in entity_mapping.matches.items()
        }
    )
    
    await progress_tracker.update_conflicts(merge_id, conflict_batch) 