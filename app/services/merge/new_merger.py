from app.baml_client.types import ResolutionStrategy
from app.services.transform.helpers import deduplicate_entities_with_splink
from prefect import flow, task
from app.services.storage.interface import GraphStorageInterface
from app.services.storage.neo4j import Neo4jStorage
from app.config import settings
from app.schemas.graph import GraphResponse, Node, Edge
from app.services.transform.models import RelationshipInstance
from app.services.transform.ontology_helper import OntologyParser
from pathlib import Path
import logging
import time
import traceback
import copy
from app.services.merge.models import EntityMappingResult, EntityMatch, MatchStrategy
from app.services.merge.models import ChangeLog, MergeStatus
from typing import Dict, Any, Optional, Tuple, List
import uuid
import json
from datetime import datetime
from app.utils.constants import VALID_FROM, VALID_TO, PREVIOUS_VERSION_RELATIONSHIP_TYPE, UPDATED, TRANSFORM_ID
from app.baml_client import b
from supabase import create_client, Client
from app.services.storage.interface import StorageBatchResult


logger = logging.getLogger(__name__)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def custom_cache_key_fn(context, parameters):
    # Only include specific parameters in the cache key
    safe_params = {
        # "merge_id": parameters["merge_id"] if "merge_id" in parameters else '',
        "ontology_id": parameters["ontology_id"] if "ontology_id" in parameters else '',
        "transform_id": parameters["transform_id"] if "transform_id" in parameters else '',
    }
    return str(hash(frozenset(safe_params.items())))

@flow(name="graph-merge-flow",
    description="Merge Staging to Production knowledge graph",
    version="1.0.0",
    retries=2,
    retry_delay_seconds=30)
async def merge_flow(
    merge_id: str,
    transform_id: str,
    ontology_id: str
):
    """Merge two graphs"""
    ontology_path = Path(settings.ONTOLOGY_DIR).expanduser() / f"{ontology_id}.yaml"
    ontology_parser = OntologyParser(ontology_path)
    ontology = ontology_parser.parsed_ontology

    # Step-1: Extract Staging Graph
    staging_graph: GraphResponse = await _extract_staging_graph(transform_id)
    merged_graph = copy.deepcopy(staging_graph)

    # Check merge status
    merge_status = supabase.table("merge_status").select("*").eq("merge_id", merge_id).execute()
    if not merge_status.data:
        _start_merge_status(merge_id, transform_id, ontology_id)
    elif merge_status.data[0]['status'] == MergeStatus.READY_TO_MERGE:
        merged_graph = await _complete_prod_merge(merge_id, transform_id, ontology, staging_graph, merged_graph)
        return merged_graph
    elif merge_status.data[0]['status'] == MergeStatus.COMPLETED:
        # For re-merge, load existing prod graph and reconcile
        prod_graph = await _get_prod_graph(merge_id)
        merged_graph = _reconcile_graphs(staging_graph, prod_graph)
        _update_merge_status(merge_id, MergeStatus.STARTED)

    # Step-2: Extract Production Graph
    prod_mapping_result: EntityMappingResult = await _map_production_entities(merged_graph, ontology)
    logger.debug(f"Production mapping result: {prod_mapping_result}")

    # Step-3: Compare Graphs & Identify Conflicts
    change_logs = []
    for node_id, match in prod_mapping_result.matches.items():
        if match.best_match:
            staging_node = next((n for n in merged_graph.nodes if n.id == node_id), None)
            ontology_props = ontology['entities'][staging_node.type]['properties']
            change_log = ChangeLog(
                staging_node=staging_node,
                prod_node=match.best_match,
                prop_changes=_get_prop_changes(staging_node, match.best_match, ontology_props)
            )
            change_logs.append(change_log)

    change_log_by_entity_type = _group_changes_by_entity_type(change_logs)
    high_conf_changes, changes_for_human_review = await _classify_changes(ontology_id, change_log_by_entity_type)

    if changes_for_human_review:
        _update_merge_status(merge_id, MergeStatus.HUMAN_REVIEW)
        for change_log in changes_for_human_review:
            save_change_log(merge_id, change_log, need_human_review=True)
        for change_log in high_conf_changes:
            ontology_props = ontology['entities'][change_log.staging_node.type]['properties']
            change_log = _apply_corrections(ontology_props, change_log)
            save_change_log(merge_id, change_log)
    else:
        _update_merge_status(merge_id, MergeStatus.AUTO_RESOLVE)
        for change_log in high_conf_changes:
            ontology_props = ontology['entities'][change_log.staging_node.type]['properties']
            change_log = _apply_corrections(ontology_props, change_log)
            _merge_nodes(change_log.staging_node, change_log.prod_node, ontology_props, merged_graph)
        _update_merge_status(merge_id, MergeStatus.MERGE_IN_PROGRESS)
        await _persist_to_prod(merged_graph, merge_id, transform_id)
    return merged_graph

def _reconcile_graphs(staging_graph: GraphResponse, prod_graph: GraphResponse) -> GraphResponse:
    """Reconcile staging and production graphs for re-merge"""
    merged_graph = copy.deepcopy(staging_graph)
    prod_node_map = {n.id: n for n in prod_graph.nodes}
    prod_edge_map = {(e.source, e.target, e.type): e for e in prod_graph.edges if e.properties.get(VALID_TO) is None}

    # Update node IDs to match production where applicable
    for staging_node in merged_graph.nodes:
        prod_node = prod_node_map.get(staging_node.id)
        if prod_node:
            staging_node.id = prod_node.id  # Use existing prod ID if matched

    # Reconcile edges
    for edge in merged_graph.edges:
        key = (edge.source, edge.target, edge.type)
        prod_edge = prod_edge_map.get(key)
        if prod_edge:
            edge.id = prod_edge.id  # Use existing prod edge ID
            edge.properties = {**prod_edge.properties, **edge.properties}  # Merge properties
        else:
            edge.id = str(uuid.uuid4())  # New edge gets a new ID

    return merged_graph

def log_merge_failure(merge_id: str, error: str):
    supabase.table("merge_status").update(
        {
            "status": MergeStatus.FAILED,
            "error": error
        }
    ).eq("merge_id", merge_id).execute()

def get_merge_status(merge_id: str) -> MergeStatus:
    merge_status = supabase.table("merge_status").select("*").eq("merge_id", merge_id).execute()
    if not merge_status.data:
        return MergeStatus.NOT_FOUND
    return MergeStatus(merge_status.data[0]['status'])

def get_human_review_items(merge_id: str) -> List[ChangeLog]:
    change_logs_data = supabase.table("change_logs").select("*").eq("merge_id", merge_id).eq("need_human_review", True).execute()
    change_logs = []
    for change_log in change_logs_data.data:
        change_logs.append(ChangeLog(
            id=change_log['id'],
            prop_changes={k: (change_log['changed_props'][k], change_log['previous_props'].get(k, None)) for k in change_log['changed_props']},
            staging_node=Node(
                id=change_log['node_id'],
                label=change_log['node_type'],
                type=change_log['node_type'],
                properties=change_log['changed_props']
            ),
            prod_node=Node(
                id=change_log['prod_node_id'],
                label=change_log['node_type'],
                type=change_log['node_type'],
                properties=change_log['previous_props']
            )
        ))
    return change_logs

async def apply_resolution(
    merge_id: str, change_log_id: str, 
    resolved_props: Dict[str, Any], resolution: ResolutionStrategy, 
    learning_comment: str) -> bool:
    """
    Apply a resolution to a conflict and save the learning for future reference.
    
    Args:
        merge_id: ID of the merge process
        change_log_id: ID of the conflict to resolve
        resolution: The resolution decision : resolved properties
        learning_comment: Comment on the resolution
        
    Returns:
        True if the resolution was applied successfully, False otherwise
    """
    change_log = supabase.table("change_logs").select("*").eq("merge_id", merge_id).eq("id", change_log_id).execute()
    if len(change_log.data) > 0:
        conflict_data = change_log.data[0]
        
        # Get ontology_id from merge_status
        merge_info = supabase.table("merge_status").select("ontology_id").eq("merge_id", merge_id).execute()
        if not merge_info.data:
            logger.error(f"Could not find merge info for merge_id {merge_id}")
            return False
            
        ontology_id = merge_info.data[0].get('ontology_id')
        
        # Save resolution for future learning
        save_resolution(
            merge_id=merge_id,
            change_log_id=change_log_id,
            ontology_id=ontology_id,
            node_id=conflict_data['node_id'],
            node_type=conflict_data['node_type'],
            previous_props=conflict_data['previous_props'],
            changed_props=conflict_data['changed_props'],
            resolved_props=resolved_props,
            resolution=resolution,
            learning_comment=learning_comment
        )

        # Check if all conflicts are resolved
        unresolved_conflicts = supabase.table("change_logs").select("*").eq("merge_id", merge_id).eq("need_human_review", True).execute()
        if len(unresolved_conflicts.data) == 0:
            _update_merge_status(merge_id, MergeStatus.READY_TO_MERGE)
            transform_id = merge_info.data[0].get('transform_id')
            if transform_id and ontology_id:
                await merge_flow(merge_id, transform_id, ontology_id)
        return True
    return False

def get_merge_statistics(merge_id: str) -> Dict[str, Any]:
    merge_status = supabase.table("merge_status").select("statistics").eq("merge_id", merge_id).execute()
    return merge_status.data[0]['statistics']

async def get_merge_graph(merge_id: str, transform_id: str) -> GraphResponse:
    """Get the merged graph for a merge operation"""
    # If merge status is not STARTED then return None
    merge_status = supabase.table("merge_status").select("status").eq("merge_id", merge_id).execute()
    if not merge_status.data:
        return None
    elif merge_status.data[0]['status'] == MergeStatus.COMPLETED:
        return await _get_prod_graph(merge_id)
        
    # Fetch staging graph using transform_id
    staging_graph = await _extract_staging_graph(transform_id)
    logger.debug(f"Retrieved staging graph with {len(staging_graph.nodes)} nodes")
    
    # Fetch change_logs and apply on top of staging graph
    change_logs = supabase.table("change_logs").select("*").eq("merge_id", merge_id).execute()
    if not change_logs.data:
        return staging_graph
        
    # Create a map of node IDs for quick lookup
    node_map = {node.id: node for node in staging_graph.nodes}
    
    for change_log in change_logs.data:
        try:
            # Use 'node_id' instead of 'staging_node_id'
            node_id = change_log.get('node_id')
            if not node_id or node_id not in node_map:
                logger.warning(f"Node ID {node_id} not found in staging graph")
                continue
                
            staging_node = node_map[node_id]
            
            # Apply changes if needed
            if change_log.get('changed_props'):
                # Update the node properties with the resolved properties
                staging_node.properties.update(change_log['changed_props'])
                
            # Mark nodes that need review
            if change_log.get('need_human_review', False):
                staging_node.properties['__NEED_REVIEW'] = True
                
        except Exception as e:
            logger.error(f"Error processing change log: {str(e)}")
            
    return staging_graph

@task(name="extract_staging_graph")
async def _extract_staging_graph(transform_id: str) -> GraphResponse:
    """Extract Staging Graph"""
    start_time = time.time()

    storage = Neo4jStorage(
        uri=settings.STAGING_NEO4J_URI,
        username=settings.STAGING_NEO4J_USER,
        password=settings.STAGING_NEO4J_PASSWORD,
        database=settings.STAGING_NEO4J_DATABASE
    )
    
    try:
        # Extract nodes with transform_id
        storage_nodes = await storage.get_nodes_by_property(
            property_name=TRANSFORM_ID,
            property_value=transform_id
        )
        
        if not storage_nodes:
            logger.warning(f"No nodes found with transform_id {transform_id}")
            return GraphResponse(
                nodes=[],
                edges=[],
                total_nodes=0,
                total_edges=0
            )
        
        # Convert storage nodes to schema nodes
        nodes = [
            Node(
                id=str(node.id),
                label=node.label,
                type=node.type,
                properties=node.properties
            ) for node in storage_nodes
        ]
        
        # Extract relationships between these nodes
        node_ids = [node.id for node in storage_nodes]
        storage_edges = await storage.get_relationships_between_nodes(node_ids)
        
        # Convert storage edges to schema edges
        edges = [
            Edge(
                id=str(edge.id),
                source=str(edge.source),
                target=str(edge.target),
                type=edge.type,
                properties=edge.properties
            ) for edge in storage_edges
        ]
        
        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Extracted {len(nodes)} nodes and {len(edges)} relationships "
            f"in {duration_ms:.2f}ms"
        )
        
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges)
        )
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract staging graph: {str(e)}")
        raise

@task(name="map_production_entities")
async def _map_production_entities(
    staging_graph: GraphResponse,
    ontology: Dict[str, Any],
    similarity_threshold: float = 0.7,
) -> EntityMappingResult:
    try:
        start_time = time.time()
        matches = {}
        matched_count = 0

        storage = Neo4jStorage(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DB
        )

        candidates = []
        
        for node in staging_graph.nodes:
            match = await _get_matching_nodes(
                storage,
                node,
                similarity_threshold
            )
            matches[node.id] = match
            if match.production_matches:
                if match.best_match is None:
                    entity_props = ontology['entities'][node.type]['properties']
                    prod_candidates = "\n".join([_get_node_string(_n, entity_props) for _n in match.production_matches])
                    candidate = f"""
                    <set>
                        Node type: {node.type}
                        Staging Node: {_get_node_string(node, entity_props)}
                        Candidates (one each line): 
                        {prod_candidates}
                    </set>
                    """
                    candidates.append(candidate)
                matched_count += 1

        #analyse candidates and add to matches only if there is high confidence
        if len(candidates) > 0:
            matching_nodes = b.GetMatchingNodes(
                candidate_sets=candidates
            )
            for match in matching_nodes:
                if match.node_id:
                    prod_matches = matches[match.staging_node_id].production_matches
                    best_match = next((n for n in prod_matches if n.id == match.node_id), None)
                    matches[match.staging_node_id].best_match = best_match
            
            # For each staging node, use splink to find the best production match
            for node_id, match_info in matches.items():
                # Skip if we already have a best match from BAML
                if match_info.best_match:
                    continue
                    
                # Get the staging node and its production candidates
                staging_node = next((n for n in staging_graph.nodes if n.id == node_id), None)
                if not staging_node or not match_info.production_matches:
                    continue
                    
                # Use splink to find the best match
                best_match = await _find_best_production_match_with_splink(
                    staging_node=staging_node,
                    production_candidates=match_info.production_matches,
                    ontology=ontology
                )
                
                if best_match:
                    matches[node_id].best_match = best_match
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Mapped {matched_count}/{len(staging_graph.nodes)} entities "
            f"in {duration_ms:.2f}ms"
        )
        
        return EntityMappingResult(
            matches=matches,
            total_entities=len(staging_graph.nodes),
            matched_entities=matched_count,
            mapping_time_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Failed to map production entities: {str(e)}")
        raise

async def _find_best_production_match_with_splink(
    staging_node: Node,
    production_candidates: List[Node],
    ontology: Dict[str, Any],
    threshold: float = 0.95
) -> Optional[Node]:
    """
    Use splink deduplication to find the best matching production node for a staging node.
    
    Args:
        staging_node: The staging node to find a match for
        production_candidates: List of potential production node matches
        ontology: The ontology definition
        threshold: Match probability threshold (default: 0.95)
        
    Returns:
        The best matching production node or None if no match found
    """
    if not production_candidates:
        return None
        
    # Combine staging node with production candidates for deduplication
    all_nodes = [staging_node] + production_candidates
    
    # Run splink deduplication
    deduplicated_nodes, _ = await deduplicate_entities_with_splink(
        entities=all_nodes,
        threshold=threshold,
        parsed_ontology=ontology
    )
    
    # If the number of deduplicated nodes is less than the original count,
    # it means some nodes were considered duplicates
    if len(deduplicated_nodes) < len(all_nodes):
        # Find which production node was merged with the staging node
        staging_node_id = staging_node.id
        
        # Check if the staging node ID is still in the deduplicated results
        staging_node_exists = any(node.id == staging_node_id for node in deduplicated_nodes)
        
        if not staging_node_exists:
            # If staging node was merged with a production node, find which one
            # by checking which production node is still in the results
            for prod_node in production_candidates:
                if any(node.id == prod_node.id for node in deduplicated_nodes):
                    logger.info(f"Splink found match: staging node {staging_node_id} -> production node {prod_node.id}")
                    return prod_node
        else:
            # If staging node still exists, check if any production nodes were removed
            # (meaning they were considered duplicates of something else)
            remaining_prod_ids = {node.id for node in deduplicated_nodes if node.id != staging_node_id}
            removed_prod_ids = {node.id for node in production_candidates} - remaining_prod_ids
            
            if removed_prod_ids:
                # Find the first production node that was removed (considered a duplicate)
                for prod_node in production_candidates:
                    if prod_node.id not in remaining_prod_ids:
                        logger.info(f"Splink found match: staging node {staging_node_id} -> production node {prod_node.id}")
                        return prod_node
    
    return None

async def _get_prod_node(label: str, id: str) -> Node:
    storage = Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )
    prod_node = await storage.find_nodes_by_property_value(
        label=label,
        property_name="id",
        property_value=id
    )
    return prod_node[0] if len(prod_node) > 0 else None

def _merge_nodes(staging_node: Node, prod_node: Node, 
                 ontology: Dict[str, Any], merged_graph: GraphResponse) -> GraphResponse:
    """Merge two nodes and ensure all edges are updated. Only version if properties change."""
    entity_def = ontology
    # Filter properties based on the ontology definition
    staging_props_relevant = {k: v for k, v in staging_node.properties.items() if k in entity_def}
    prod_props_relevant = {k: v for k, v in prod_node.properties.items() if k in entity_def}
    
    # Check if there are any differences or new properties in staging
    properties_changed = False
    for key, value in staging_props_relevant.items():
        if key not in prod_props_relevant or prod_props_relevant[key] != value:
            properties_changed = True
            break

    # Even if no changes in staging props, check if staging *removes* a relevant prop that existed in prod
    # This is unlikely given the merge strategy, but good to consider. Let's assume staging props are the source of truth for now.
    # A more robust check would be: properties_changed = staging_props_relevant != prod_props_relevant
    # Let's use the simpler check first based on the request (new or different in staging)

    current_time_str = str(datetime.now())

    # Merge properties: Staging properties overwrite production properties for defined keys
    # Keep existing prod properties not defined in the ontology (like system properties)
    merged_properties = {**prod_node.properties, **staging_props_relevant}
    merged_properties[VALID_FROM] = current_time_str
    if VALID_TO in merged_properties:
        del merged_properties[VALID_TO]
    
    # Preserve original staging ID before modification
    old_staging_id = staging_node.id

    # Update staging node in-place to reflect the merged state (using prod_node ID)
    staging_node.id = prod_node.id
    staging_node.properties = merged_properties
    # Note: We are modifying the staging_node object that exists within merged_graph.nodes list
    
    # Update edges pointing to the old staging ID to use the prod_node ID
    _update_to_prod_node_id_in_edges(old_id=old_staging_id, new_id=prod_node.id, merged_graph=merged_graph)

    # Only version the production node if properties actually changed
    if properties_changed:
        logger.info(f"Properties changed for node {prod_node.id}. Versioning required.")
        # Version the old production node state
        prod_node.properties[VALID_TO] = current_time_str # Use the same timestamp for consistency
        prev_ver = Node(
            id=str(uuid.uuid4()),
            label=prod_node.label,
            type=prod_node.type,
            properties=prod_node.properties # Properties before merge, now marked with VALID_TO
        )
        merged_graph.nodes.append(prev_ver)
        merged_graph.edges.append(Edge(
            id=str(uuid.uuid4()),
            target=prev_ver.id,
            source=prod_node.id, # Link from the current node ID
            type=PREVIOUS_VERSION_RELATIONSHIP_TYPE,
            properties={UPDATED: current_time_str}
        ))
        logger.debug(f"Merged node: {prod_node.id}, Previous version created: {prev_ver.id}")
    else:
        logger.info(f"No relevant property changes for node {prod_node.id}. Skipping versioning.")
        # Find the actual staging node in the merged_graph.nodes list and update it
        # (since we modified the staging_node object directly earlier)
        # No, the modification above already updates the object in the list.
        # We just need to ensure the old staging node isn't separately added.
        # The logic should handle finding the staging node in merged_graph.nodes and updating it.

    # Ensure the original staging node *object* (with old_staging_id) is removed 
    # if it wasn't the one modified in-place. This depends on how merged_graph was constructed.
    # Assuming merged_graph initially contains the staging_node object reference:
    # The update to staging_node.id and staging_node.properties modifies the object within the list.
    # We might need to remove the prod_node from the list if it was added separately.
    # Let's refine this: The goal is to have ONE node with prod_node.id in the final list.
    
    # Find the index of the node we modified (which now has prod_node.id)
    modified_node_index = -1
    for i, n in enumerate(merged_graph.nodes):
        if n.id == prod_node.id:
            modified_node_index = i
            break
            
    # Remove the original prod_node if it exists separately in the list
    # This assumes prod_node was fetched separately and might be a different object
    # than the one potentially already in merged_graph.nodes with the same ID.
    # A safer approach might be to ensure only ONE node with prod_node.id exists *after* the merge.
    
    final_nodes = []
    ids_seen = set()
    for node in merged_graph.nodes:
        if node.id == prod_node.id:
            if prod_node.id not in ids_seen:
                final_nodes.append(node) # Keep the modified one
                ids_seen.add(prod_node.id)
        elif node.id != old_staging_id: # Avoid adding back the original staging node if it lingered
             if node.id not in ids_seen:
                 final_nodes.append(node)
                 ids_seen.add(node.id)
        # Add the previous version if it was created
        elif properties_changed and node.id == prev_ver.id:
             if node.id not in ids_seen:
                 final_nodes.append(node)
                 ids_seen.add(node.id)
                 
    merged_graph.nodes = final_nodes

    return merged_graph

def _update_to_prod_node_id_in_edges(old_id: str, new_id: str, merged_graph: GraphResponse):
    """Update all edges to use the new production ID"""
    for edge in merged_graph.edges:
        if edge.source == old_id:
            edge.source = new_id
        if edge.target == old_id:
            edge.target = new_id
    logger.debug(f"Updated edges: old_id={old_id} to new_id={new_id}")

async def _get_matching_nodes(
    storage: GraphStorageInterface,
    node: Node,
    similarity_threshold: float
) -> EntityMatch:
    """Find matching nodes in production based on node type and properties"""
    try:
        # Start with most specific matching strategy
        strategy = MatchStrategy.EXACT_NAME
        matches = []
        best_match = None
        logger.info(node)
        # Fall back to name-based matching
        if not matches and "name" in node.properties:
            matches = await storage.find_nodes_by_property_value(
                label=node.label,
                property_name="name",
                property_value=node.properties["name"]
            )
            logger.info(matches)
            if len(matches) > 0:
                best_match = matches[0]
            
        # Use property similarity as last resort
        if not matches:
            strategy = MatchStrategy.PROPERTY_SIMILARITY
            matches = await storage.find_similar_nodes(
                label=node.label,
                properties=node.properties,
                similarity_threshold=similarity_threshold
            )
            logger.info(matches)
        
        # Calculate confidence based on strategy and number of matches
        confidence = 0.8 if strategy == MatchStrategy.EXACT_NAME else 0.2
        
        return EntityMatch(
            staging_id=node.id,
            production_matches=matches,
            best_match = best_match,
            match_confidence=confidence,
            match_strategy=strategy,
            metadata={
                "total_matches": len(matches),
                "node_label": node.label
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get matching nodes: {str(e)}")
        raise

@task(name="persist_to_prod")
async def _persist_to_prod(
    merged_graph: GraphResponse,
    merge_id: str,
    transform_id: str
) -> Tuple[StorageBatchResult, StorageBatchResult]:
    storage = Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )
    
    # Store nodes first
    node_batch_result = await storage.store_nodes(
        merged_graph.nodes,
        0,
        transform_id,
        merge_id
    )
    if not node_batch_result.success:
        logger.error(f"Failed to persist nodes: {node_batch_result.error}")
        raise Exception(f"Node persistence failed: {node_batch_result.error}")

    # Create a node map from persisted nodes
    node_map = {n.id: n for n in merged_graph.nodes}
    
    # Convert edges to RelationshipInstance, ensuring valid IDs
    edges_as_rel_instances = []
    for edge in merged_graph.edges:
        if edge.source in node_map and edge.target in node_map:
            edges_as_rel_instances.append(RelationshipInstance(
                id=edge.id,
                source_id=edge.source,
                target_id=edge.target,
                type=edge.type,
                source_type=node_map[edge.source].type,
                target_type=node_map[edge.target].type,
                properties=edge.properties
            ))
        else:
            logger.warning(f"Skipping edge {edge.id}: Source {edge.source} or Target {edge.target} not in node_map")

    # Store relationships with versioning logic
    edge_batch_result = await storage.store_relationships(
        edges_as_rel_instances,
        0,
        transform_id,
        merge_id,
        merge=True
    )
    if not edge_batch_result.success:
        logger.error(f"Failed to persist edges: {edge_batch_result.error}")
        raise Exception(f"Edge persistence failed: {edge_batch_result.error}")

    # Update merge status
    _add_ingestion_stats(merge_id, node_batch_result, edge_batch_result)
    return node_batch_result, edge_batch_result

def _add_ingestion_stats(merge_id, node_batch_result, edge_batch_result):
    supabase.table("merge_status").update(
        {
            "statistics": {
                "nodes_stored": node_batch_result.model_dump(),
                "edges_stored": edge_batch_result.model_dump()
            },
            "status": MergeStatus.COMPLETED
        }
    ).eq("merge_id", merge_id).execute()

async def _get_prod_graph(merge_id: str) -> GraphResponse:
    storage = Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )
    return await storage.get_merge_data(merge_id)

async def get_past_resolution(ontology_id: str, node_type: str) -> str:
    response = supabase.table("resolutions").select("*").eq("node_type", node_type).execute()
    if not response.data:
        return "None"
    learnings = []
    i = 1
    for log in response.data:
        learning = f"""
        Learning {i}:
            Existing Properties (Production): {log['previous_props']}
            Incoming Properties (Staging): {log['changed_props']}
            Resolution by the User: {log['resolved_props']}
            Resolution Strategy: {"Keep Incoming Properties" if log['resolution'] == ResolutionStrategy.KEEP_BOTH.value else ResolutionStrategy(log['resolution']).value}
            User Comment (rationale): {log['learning_comment']}
        """
        learnings.append(learning)
        i += 1
    return "\n".join(learnings)

def save_change_log(merge_id, change_log, need_human_review: bool = False):
    supabase.table("change_logs").insert(
        {
            "merge_id": merge_id,
            "node_id": change_log.staging_node.id,
            "prod_node_id": change_log.prod_node.id,
            "node_type": change_log.staging_node.type,
            "previous_props": change_log.prod_node.properties,
            "changed_props": {k: v[0] for k, v in change_log.prop_changes.items()},
            "need_human_review": need_human_review
        }
    ).execute()
    
def save_resolution(
    merge_id: str, change_log_id: str, ontology_id: str, 
    node_id: str, node_type: str, previous_props: Dict, 
    changed_props: Dict, resolved_props: Dict, resolution: ResolutionStrategy, learning_comment: str):
    """
    Save a resolution to the resolutions table for future reference.
    
    Args:
        merge_id: ID of the merge operation
        ontology_id: ID of the ontology used for the merge
        node_id: ID of the node that was resolved
        node_type: Type of the node
        previous_props: Properties before the merge
        changed_props: Properties after the merge
        resolution: The resolution decision
        learning_comment: User comment about the resolution decision
    """
    logger.info(f"Saving resolution for node {node_id} of type {node_type}")
    try:
        supabase.table("resolutions").insert(
            {
                "ontology_id": ontology_id,
                "node_type": node_type,
                "previous_props": previous_props,
                "changed_props": changed_props,
                "resolved_props": resolved_props,
                "resolution": resolution.value,
                "learning_comment": learning_comment
            }
        ).execute()
        
        # Update the change log
        supabase.table("change_logs").update(
            {
                "need_human_review": False,
                "changed_props": {} if resolution == ResolutionStrategy.KEEP_BOTH else resolved_props
            }
        ).eq("merge_id", merge_id).eq("id", change_log_id).execute()
        
        logger.info(f"Successfully saved resolution for node {node_id}")
    except Exception as e:
        logger.error(f"Failed to save resolution: {str(e)}")
        traceback.print_exc()
        

def _start_merge_status(merge_id, transform_id, ontology_id):
    supabase.table("merge_status").insert(
        {
            "merge_id": merge_id,
            "transform_id": transform_id,
            "ontology_id": ontology_id,
            "status": MergeStatus.STARTED
        }
    ).execute()
    
def _update_merge_status(merge_id, status):
    supabase.table("merge_status").update(
        {
            "status": status
        }
    ).eq("merge_id", merge_id).execute()

@task(name="complete_prod_merge")
async def _complete_prod_merge(merge_id, transform_id, ontology, staging_graph, merged_graph):
    """Complete the production merge by applying all resolved conflicts"""
    try:
        # Get all change logs for this merge
        change_logs = supabase.table("change_logs").select("*").eq("merge_id", merge_id).execute()
        
        if not change_logs.data:
            logger.info(f"No change logs found for merge {merge_id}, proceeding with direct merge")
            await _persist_to_prod(merged_graph, merge_id, transform_id)
            _update_merge_status(merge_id, MergeStatus.COMPLETED)
            return merged_graph
            
        # Create a map of node IDs for quick lookup
        node_map = {node.id: node for node in staging_graph.nodes}
        
        # Apply all resolved conflicts
        for change_log in change_logs.data:
            try:
                node_id = change_log.get('node_id')
                node_type = change_log.get('node_type')
                
                if not node_id or not node_type or node_id not in node_map:
                    logger.warning(f"Invalid change log entry: node_id={node_id}, node_type={node_type}")
                    continue
                    
                # Get the staging node
                staging_node = node_map[node_id]
                
                # Apply the resolved properties if any
                if change_log.get('changed_props'):
                    if isinstance(change_log['changed_props'], str):
                        # Handle the case where properties are stored as JSON string
                        try:
                            if change_log['changed_props'] == "{}":
                                continue
                            resolved_props = json.loads(change_log['changed_props'])
                            staging_node.properties.update(resolved_props)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse changed_props JSON: {change_log['changed_props']}")
                    else:
                        # Handle the case where properties are stored as dictionary
                        staging_node.properties.update(change_log['changed_props'])
                
            except Exception as e:
                logger.error(f"Error processing change log during prod merge: {str(e)}")
                
        # Persist the merged graph to production
        await _persist_to_prod(merged_graph, merge_id, transform_id)
        _update_merge_status(merge_id, MergeStatus.COMPLETED)
        return merged_graph
        
    except Exception as e:
        logger.error(f"Error in _complete_prod_merge: {str(e)}")
        traceback.print_exc()
        log_merge_failure(merge_id, str(e))
        raise

def _get_node_string(node: Node, properties: Dict[str, Any]) -> str:
    props = { k: v for k, v in node.properties.items() if k in properties }
    return f"(Node Id: {node.id}, properties: {props})"

def _get_prop_changes(staging_node: Node, prod_node: Node, props: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    return { k: (staging_node.properties[k], prod_node.properties.get(k, None)) for k, v in props.items() if k in staging_node.properties and staging_node.properties[k] != prod_node.properties.get(k, None)}

def _group_changes_by_entity_type(change_logs: List[ChangeLog]) -> Dict[str, List[ChangeLog]]:
    change_log_by_entity_type = {}
    for change_log in change_logs:
        if change_log.staging_node.type not in change_log_by_entity_type:
            change_log_by_entity_type[change_log.staging_node.type] = []
        change_log_by_entity_type[change_log.staging_node.type].append(change_log)
    return change_log_by_entity_type

def _get_change_log_string(change_log: ChangeLog) -> str:
    changes = "\n".join([f"{k}: {v[1]} -> {v[0]}" for k, v in change_log.prop_changes.items()])
    return f"""
    Changes (Existing properties [production] -> Incoming properties [staging]; one per line):
    <id>{change_log.id}</id>
    <changes>
    {changes}
    </changes>
    """

@task(name="classify_changes")
async def _classify_changes(
        ontology_id: str, 
        change_log_by_entity_type: Dict[str, List[ChangeLog]]
    ) -> Tuple[List[ChangeLog], List[ChangeLog]]:
    """
    Classify changes into high confidence and low confidence changes
    """
    high_conf_changes = []
    changes_for_human_review = []
    for entity_type, change_logs in change_log_by_entity_type.items():
        #get past resolutions
        past_resolutions = await get_past_resolution(ontology_id, entity_type)
        #get LLM response
        changes = [_get_change_log_string(change_log) for change_log in change_logs if change_log.prop_changes]
        if not changes:
            high_conf_changes.extend(change_logs)
            continue
        change_log_string = "\n".join(changes)
        eval_changes = b.EvalChanges(
            change_logs=change_log_string,
            past_resolutions=past_resolutions
        )
        if len(eval_changes) > 0:
            for change in eval_changes:
                if change.confidence_score > 0.95:
                    if not change.corrections:
                        change_log = next((c for c in change_logs if c.id == change.id), None)
                        high_conf_changes.append(change_log)
                        continue
                    for correction in change.corrections:
                        change_log = next((c for c in change_logs if c.id == change.id), None)
                        change_log.prop_changes[correction.prop_name] = (
                            change_log.prop_changes[correction.prop_name][0], correction.prop_value)
                        high_conf_changes.append(change_log)
                else:
                    change_log = next((c for c in change_logs if c.id == change.id), None)
                    changes_for_human_review.append(change_log)
        else:
            high_conf_changes.extend(change_logs)

    return high_conf_changes, changes_for_human_review

@task(name="apply_auto_corrections")
def _apply_corrections(ontology_props, change_log):
    for prop in ontology_props:
        if change_log.prop_changes and prop in change_log.prop_changes:
            change_log.staging_node.properties[prop] = change_log.prop_changes[prop][1]
    return change_log