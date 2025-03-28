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
from typing import Dict, Any, Tuple, List
import uuid
import json
from datetime import datetime
from app.utils.constants import VALID_FROM, VALID_TO, PREVIOUS_VERSION_RELATIONSHIP_TYPE, UPDATED
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

    # Step-1: Extract Staging Graph
    staging_graph: GraphResponse = await _extract_staging_graph(transform_id)
    merged_graph = copy.deepcopy(staging_graph)

    #if there is a merge status, check if it is already done, merge change_logs and update merge status
    merge_status = supabase.table("merge_status").select("*").eq("merge_id", merge_id).execute()
    if merge_status.data and merge_status.data[0]['status'] == MergeStatus.READY_TO_MERGE:
        merged_graph = await _complete_prod_merge(merge_id, transform_id, ontology, staging_graph, merged_graph)
        return merged_graph
    elif not merge_status.data:
        _start_merge_status(merge_id, transform_id, ontology_id)
    
    # Step-2: Extract Production Graph
    prod_mapping_result: EntityMappingResult = await _map_production_entities(staging_graph, ontology_parser.parsed_ontology)

    # Step-3: Compare Graphs & Identify Conflicts
    ontology = ontology_parser.parsed_ontology  
    change_logs = []
    for node_id, match in prod_mapping_result.matches.items():
        if match.best_match:
            staging_node = next((n for n in staging_graph.nodes if n.id == node_id), None)
            ontology_props = ontology['entities'][staging_node.type]['properties']
            print(staging_node)
            print(match.best_match)
            change_log = ChangeLog(
                staging_node=staging_node,
                prod_node=match.best_match,
                prop_changes=_get_prop_changes(staging_node, ontology_props)
            )
            change_logs.append(change_log)

    change_log_by_entity_type = _group_changes_by_entity_type(change_logs)
    high_conf_changes, changes_for_human_review = await _classify_changes(ontology_id, change_log_by_entity_type)

    if len(changes_for_human_review) > 0:
        _update_merge_status(merge_id, MergeStatus.HUMAN_REVIEW)
        #save all changes in supabase
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

def log_merge_failure(merge_id: str, error: str):
    supabase.table("merge_status").update(
        {
            "status": MergeStatus.FAILED,
            "error": error
        }
    ).eq("merge_id", merge_id).execute()

def get_merge_status(merge_id: str) -> MergeStatus:
    merge_status = supabase.table("merge_status").select("*").eq("merge_id", merge_id).execute()
    return MergeStatus(merge_status.data[0]['status'])

def get_human_review_items(merge_id: str) -> List[ChangeLog]:
    change_logs = supabase.table("change_logs").select("*").eq("merge_id", merge_id).eq("need_human_review", True).execute()
    return change_logs.data

async def apply_resolution(merge_id: str, change_log_id: str, resolution_data: Dict[str, Any], learning_comment: str) -> bool:
    change_log = supabase.table("change_logs").select("*").eq("merge_id", merge_id).eq("id", change_log_id).execute()
    if len(change_log.data) > 0:
        supabase.table("change_logs").update(
            {
                "need_human_review": False,
                "changed_props": json.dumps(resolution_data),
                "learning_comment": learning_comment
            }
        ).eq("merge_id", merge_id).eq("id", change_log_id).execute()

        unresolved_conflicts = supabase.table("change_logs").select("*").eq("merge_id", merge_id).eq("need_human_review", True).execute()
        if len(unresolved_conflicts.data) == 0:
            _update_merge_status(merge_id, MergeStatus.READY_TO_MERGE)
            transform_id, ontology_id = supabase.table("merge_status").select("transform_id, ontology_id").eq("merge_id", merge_id).execute()
            if transform_id and ontology_id:
                await merge_flow(merge_id, transform_id, ontology_id)
        return True
    return False

def get_merge_statistics(merge_id: str) -> Dict[str, Any]:
    merge_status = supabase.table("merge_status").select("statistics").eq("merge_id", merge_id).execute()
    return merge_status.data[0]['statistics']

async def get_merge_graph(merge_id: str, transform_id: str) -> GraphResponse:
    #if merge status is not STARTED then return None
    merge_status = supabase.table("merge_status").select("status").eq("merge_id", merge_id).execute()
    if not merge_status.data:
        return None
    elif merge_status.data[0]['status'] != MergeStatus.COMPLETED:
        return await _get_prod_graph(transform_id)
    #fetch staging graph using transform_id
    staging_graph = await _extract_staging_graph(transform_id) 
    #fetch change_logs and apply on top of staging graph
    change_logs = supabase.table("change_logs").select("*").eq("merge_id", merge_id).execute()
    for change_log in change_logs.data:
        staging_node = staging_graph.nodes[change_log['staging_node_id']]
        prod_node = _get_prod_node(change_log['node_type'], change_log['prod_node_id'])
        staging_graph = _merge_nodes(staging_node, prod_node, staging_node.properties, staging_graph)
        if change_log['need_human_review']:
            staging_node.properties['__NEED_REVIEW'] = True
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
            property_name="transform_id",
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
    """Merge two nodes"""
    entity_def = ontology

    staging_props = { k: v for k, v in staging_node.properties.items() if k in entity_def }
    prod_props = { k: v for k, v in prod_node.properties.items() if k in entity_def }
    
    staging_node.properties = { **staging_props, **prod_props }
    staging_node.properties[VALID_FROM] = str(datetime.now())
    
    _update_to_prod_node_id_in_edges(staging_node, prod_node, merged_graph)
    staging_node.id = prod_node.id

    #Create a previous version for old prod node
    prod_node.properties[VALID_TO] = staging_node.properties[VALID_FROM]
    prev_ver = Node(
        id=str(uuid.uuid4()),
        label=prod_node.label,
        type=prod_node.type,
        properties=prod_node.properties
    )
    merged_graph.nodes.append(prev_ver)
    merged_graph.edges.append(Edge(
        id=str(uuid.uuid4()),
        target=prev_ver.id,
        source=prod_node.id,
        type=PREVIOUS_VERSION_RELATIONSHIP_TYPE,
        properties={UPDATED: staging_node.properties[VALID_FROM]}
    ))

    return merged_graph

def _update_to_prod_node_id_in_edges(staging_node, prod_node, merged_graph):
    for edge in merged_graph.edges:
        if edge.source == staging_node.id:
            edge.source = prod_node.id
        if edge.target == staging_node.id:
            edge.target = prod_node.id

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
    node_batch_result = await storage.store_nodes(
        merged_graph.nodes,
        0,
        transform_id
    )
    if not node_batch_result.success:
        logger.error(f"Failed to persist nodes: {node_batch_result.error}")
        logger.warning(f"Failed to persist nodes: {node_batch_result.error}")
    node_map = {n.id: n for n in merged_graph.nodes}
    print(node_map)
    edges_as_rel_instances = [
        RelationshipInstance(
            id=edge.id,
            source_id=edge.source,
            target_id=edge.target,
            type=edge.type,
            source_type=node_map[edge.source].type,
            target_type=node_map[edge.target].type,
            properties=edge.properties
        ) for edge in merged_graph.edges if edge.source in node_map and edge.target in node_map
    ]
    edge_batch_result = await storage.store_relationships(
        edges_as_rel_instances,
        0,
        transform_id,
        merge=True
    )
    if not edge_batch_result.success:
        logger.error(f"Failed to persist edges: {edge_batch_result.error}")
        logger.warning(f"Failed to persist edges: {edge_batch_result.error}")
    
    #update merge status
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

async def _get_prod_graph(transform_id: str) -> GraphResponse:
    storage = Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )
    return await storage.get_transformation_data(transform_id)

async def get_past_resolution(ontology_id: str, node_type: str) -> str:
    response = supabase.table("resolutions").select("*").eq("ontology_id", ontology_id).eq("node_type", node_type).execute()
    if not response.data:
        return "None"
    learnings = []
    i = 1
    for log in response.data:
        learning = f"""
        Learning {i}:
            Before Merge: {log['previous_props']}
            After Merge: {log['changed_props']}
            User Comment: {log['learning_comment']}
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
                    "previous_props": change_log.staging_node.properties,
                    "changed_props": change_log.prop_changes,
                    "need_human_review": need_human_review
                }
            ).execute()
    
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

async def _complete_prod_merge(merge_id, transform_id, ontology, staging_graph, merged_graph):
    change_logs = supabase.table("change_logs").select("*").eq("merge_id", merge_id).execute()
    if len(change_logs.data) > 0:
        for change_log in change_logs.data:
            ontology_props = ontology['entities'][change_log['node_type']]['properties']
            staging_node = staging_graph.nodes[change_log['staging_node_id']]
            staging_node.properties = json.loads(change_log['previous_props'])
            prod_node = _get_prod_node(change_log['node_type'], change_log['prod_node_id'])
            merged_graph = _merge_nodes(staging_node, prod_node, ontology_props, merged_graph)
            #persist in Graph DB
        await _persist_to_prod(merged_graph, merge_id, transform_id)
    _update_merge_status(merge_id, MergeStatus.COMPLETED)
    return merged_graph

def _get_node_string(node: Node, properties: Dict[str, Any]) -> str:
    props = { k: v for k, v in node.properties.items() if k in properties }
    return f"(Node Id: {node.id}, properties: {props})"

def _get_prop_changes(staging_node: Node, props: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    return { k: (staging_node.properties[k], v) for k, v in props.items() if k in staging_node.properties and staging_node.properties[k] != v}

def _group_changes_by_entity_type(change_logs: List[ChangeLog]) -> Dict[str, List[ChangeLog]]:
    change_log_by_entity_type = {}
    for change_log in change_logs:
        if change_log.staging_node.type not in change_log_by_entity_type:
            change_log_by_entity_type[change_log.staging_node.type] = []
        change_log_by_entity_type[change_log.staging_node.type].append(change_log)
    return change_log_by_entity_type

def _get_change_log_string(change_log: ChangeLog) -> str:
    changes = "\n".join([f"{k}: {v[0]} -> {v[1]}" for k, v in change_log.prop_changes.items()])
    return f"""
    Changes (old prop -> new prop; one per line):
    <id>{change_log.id}</id>
    <changes>
    {changes}
    </changes>
    """
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
        change_log_string = "\n".join([_get_change_log_string(change_log) for change_log in change_logs])
        eval_changes = b.EvalChanges(
            change_logs=change_log_string,
            past_resolutions=past_resolutions
        )
        if len(eval_changes) > 0:
            for change in eval_changes:
                if change.confidence_score > 0.95:
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

def _apply_corrections(ontology_props, change_log):
    for prop in ontology_props:
        if prop in change_log.prop_changes:
            change_log.staging_node.properties[prop] = change_log.prop_changes[prop][1]
    return change_log