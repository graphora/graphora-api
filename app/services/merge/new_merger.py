from app.baml_client.types import ResolutionStrategy
from prefect import flow, task
from app.services.storage.interface import GraphStorageInterface
from app.config import settings
from app.schemas.graph import GraphResponse, Node, Edge
from app.services.transform.models import RelationshipInstance
from app.services.transform.ontology_helper import OntologyParser
from pathlib import Path
import logging
import time
import traceback
import copy
from app.services.merge.models import (
    EntityMappingResult,
    EntityMatch,
    MatchStrategy,
    ChangeLog,
    ChangeLogRecord,
    ChangeLogResolution,
    MergePerformanceMetrics,
    MergeStatus,
)
from typing import Dict, Any, Optional, Tuple, List, Callable, Iterable
import uuid
import json
from datetime import datetime
from app.utils.constants import (
    VALID_FROM,
    VALID_TO,
    PREVIOUS_VERSION_RELATIONSHIP_TYPE,
    UPDATED,
    TRANSFORM_ID,
)
from supabase import create_client, Client
from app.services.storage.interface import StorageBatchResult
from app.services.user_db_service import UserDatabaseService
from app.services.audit_service import audit_service, OperationType


logger = logging.getLogger(__name__)
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def custom_cache_key_fn(context, parameters):
    # Only include specific parameters in the cache key
    safe_params = {
        # "merge_id": parameters["merge_id"] if "merge_id" in parameters else '',
        "ontology_id": parameters["ontology_id"] if "ontology_id" in parameters else "",
        "transform_id": (
            parameters["transform_id"] if "transform_id" in parameters else ""
        ),
    }
    return str(hash(frozenset(safe_params.items())))


def _safe_supabase_call(description: str, func: Callable[[], Any]):
    try:
        return func()
    except Exception as exc:  # pragma: no cover - external service
        logger.error("Supabase %s failed: %s", description, exc)
        raise


@task(name="merge-persist-node-batch", retries=3, retry_delay_seconds=5)
async def persist_node_batch_task(
    uri: str,
    username: str,
    password: str,
    node_batch: List[Node],
    batch_index: int,
    transform_id: str,
    merge_id: str,
) -> StorageBatchResult:
    from app.services.storage.neo4j import Neo4jStorage

    storage = Neo4jStorage(
        uri=uri, username=username, password=password, database="neo4j"
    )
    try:
        return await storage.store_nodes(
            node_batch, batch_index, transform_id, merge_id
        )
    finally:
        await storage.driver.close()


@task(name="merge-persist-relationship-batch", retries=3, retry_delay_seconds=5)
async def persist_relationship_batch_task(
    uri: str,
    username: str,
    password: str,
    relationships: List[RelationshipInstance],
    batch_index: int,
    transform_id: str,
    merge_id: str,
) -> StorageBatchResult:
    from app.services.storage.neo4j import Neo4jStorage

    storage = Neo4jStorage(
        uri=uri, username=username, password=password, database="neo4j"
    )
    try:
        return await storage.store_relationships(
            relationships, batch_index, transform_id, merge_id, merge=True
        )
    finally:
        await storage.driver.close()


@flow(
    name="graph-merge-flow",
    description="Merge Staging to Production knowledge graph",
    version="1.0.0",
    retries=2,
    retry_delay_seconds=30,
)
async def merge_flow(merge_id: str, transform_id: str, ontology_id: str, user_id: str):
    """Merge two graphs"""
    start_time = time.time()
    flow_timer = time.perf_counter()
    metrics = MergePerformanceMetrics()

    # Start audit trail for merge operation
    audit_id = await audit_service.log_operation_start(
        user_id=user_id,
        operation_type=OperationType.MERGE_STARTED,
        operation_id=merge_id,
        resource_name=f"Merge {merge_id[:8]}",
        metadata={"transform_id": transform_id, "ontology_id": ontology_id},
    )

    try:
        ontology_path = Path(settings.ONTOLOGY_DIR).expanduser() / f"{ontology_id}.yaml"
        ontology_parser = OntologyParser(ontology_path, user_id)
        ontology = ontology_parser.parsed_ontology

        # Step-1: Extract Staging Graph (using user's staging database)
        stage_timer = time.perf_counter()
        staging_graph: GraphResponse = await _extract_staging_graph(
            transform_id, user_id
        )
        metrics.record_stage(
            "extract_staging_graph", (time.perf_counter() - stage_timer) * 1000
        )
        metrics.staging_nodes = staging_graph.total_nodes
        metrics.staging_relationships = staging_graph.total_edges
        merged_graph = copy.deepcopy(staging_graph)

        # Check merge status
        merge_status = (
            supabase.table("merge_status")
            .select("*")
            .eq("merge_id", merge_id)
            .execute()
        )
        if not merge_status.data:
            start_merge_status_task(merge_id, transform_id, ontology_id)
        elif merge_status.data[0]["status"] == MergeStatus.READY_TO_MERGE:
            merged_graph = await _complete_prod_merge(
                merge_id,
                transform_id,
                ontology_id,
                ontology,
                staging_graph,
                merged_graph,
                user_id,
            )
            return merged_graph
        elif merge_status.data[0]["status"] == MergeStatus.COMPLETED:
            # For re-merge, load existing prod graph and reconcile
            prod_graph = await _get_prod_graph(merge_id, user_id)
            merged_graph = _reconcile_graphs(staging_graph, prod_graph)
            update_merge_status_task(merge_id, MergeStatus.STARTED)

        # Step-2: Extract Production Graph
        stage_timer = time.perf_counter()
        prod_mapping_result: EntityMappingResult = await _map_production_entities(
            merged_graph,
            ontology,
            user_id,
            merge_id=merge_id,
            transform_id=transform_id,
        )
        metrics.record_stage(
            "map_production_entities", (time.perf_counter() - stage_timer) * 1000
        )
        logger.debug(f"Production mapping result: {prod_mapping_result}")

        stage_timer = time.perf_counter()
        _apply_id_reconciliation(
            merged_graph,
            prod_mapping_result,
            confidence_threshold=settings.MERGE_ID_CONFIDENCE_THRESHOLD,
        )
        metrics.record_stage(
            "id_reconciliation", (time.perf_counter() - stage_timer) * 1000
        )

        # Step-3: Compare Graphs & Identify Conflicts
        change_logs = []
        total_comparisons = 0
        conflicts_detected = 0

        for node_id, match in prod_mapping_result.matches.items():
            if match.best_match:
                staging_node = next(
                    (n for n in merged_graph.nodes if n.id == node_id), None
                )
                if not staging_node:
                    logger.warning(f"Staging node {node_id} not found in merged_graph")
                    continue

                total_comparisons += 1
                ontology_props = ontology["entities"][staging_node.type]["properties"]
                prop_changes = _get_prop_changes(
                    staging_node, match.best_match, ontology_props
                )

                # Only create change log if there are actual property changes
                if prop_changes:
                    conflicts_detected += 1
                    change_log = ChangeLog(
                        staging_node=staging_node,
                        prod_node=match.best_match,
                        prop_changes=prop_changes,
                        match_confidence=match.match_confidence,
                        match_strategy=match.match_strategy,
                    )
                    change_logs.append(change_log)
                    logger.info(
                        f"Conflict detected for node {node_id} ({staging_node.type}): {len(prop_changes)} property changes"
                    )
                else:
                    logger.debug(
                        f"No conflicts detected for node {node_id} ({staging_node.type})"
                    )

        logger.info(
            f"Conflict detection summary: {conflicts_detected} conflicts found in {total_comparisons} node comparisons"
        )
        metrics.conflicts_detected = conflicts_detected

        # Add detailed debugging for mapping results
        _log_mapping_debug_info(prod_mapping_result, merged_graph.nodes, ontology)

        change_log_by_entity_type = _group_changes_by_entity_type(change_logs)
        stage_timer = time.perf_counter()
        high_conf_changes, changes_for_human_review = await _classify_changes(
            ontology_id,
            change_log_by_entity_type,
            merge_id=merge_id,
            transform_id=transform_id,
            user_id=user_id,
        )
        metrics.record_stage(
            "classify_changes", (time.perf_counter() - stage_timer) * 1000
        )

        if changes_for_human_review:
            update_merge_status_task(merge_id, MergeStatus.HUMAN_REVIEW)
            for change_log in changes_for_human_review:
                save_change_log_task(merge_id, change_log, need_human_review=True)
            for change_log in high_conf_changes:
                ontology_props = ontology["entities"][change_log.staging_node.type][
                    "properties"
                ]
                change_log = _apply_corrections(ontology_props, change_log)
                save_change_log_task(merge_id, change_log)

            # Log pending human review - keep using original audit_id for merge_started
            duration_ms = int((time.time() - start_time) * 1000)
            if audit_id:
                await audit_service.log_operation_success(
                    audit_id=audit_id,
                    duration_ms=duration_ms,
                    metadata={
                        "status": "pending_review",
                        "conflicts_count": len(changes_for_human_review),
                        "auto_resolved_count": len(high_conf_changes),
                    },
                )
            metrics.total_duration_ms = (time.perf_counter() - flow_timer) * 1000
            record_merge_metrics_task(merge_id, metrics)
        else:
            update_merge_status_task(merge_id, MergeStatus.AUTO_RESOLVE)
            for change_log in high_conf_changes:
                ontology_props = ontology["entities"][change_log.staging_node.type][
                    "properties"
                ]
                change_log = _apply_corrections(ontology_props, change_log)
                _merge_nodes(
                    change_log.staging_node,
                    change_log.prod_node,
                    ontology_props,
                    merged_graph,
                )
            update_merge_status_task(merge_id, MergeStatus.MERGE_IN_PROGRESS)
            persist_timer = time.perf_counter()
            persistence_summary = await _persist_to_prod(
                merged_graph,
                merge_id,
                transform_id,
                user_id,
                metrics=metrics,
            )
            metrics.record_stage(
                "persist_to_prod", (time.perf_counter() - persist_timer) * 1000
            )
            metrics.total_duration_ms = (time.perf_counter() - flow_timer) * 1000
            add_ingestion_stats_task(
                merge_id,
                persistence_summary["nodes"],
                persistence_summary["edges"],
                metrics,
            )

            # Create separate merge_completed audit record for auto-resolution
            completion_audit_id = await audit_service.log_operation_start(
                user_id=user_id,
                operation_type=OperationType.MERGE_COMPLETED,
                operation_id=merge_id,
                resource_name=f"Merge {merge_id[:8]} Auto-Completed",
                metadata={
                    "transform_id": transform_id,
                    "ontology_id": ontology_id,
                    "stage": "auto_resolve",
                },
            )

            # Log successful auto-resolution and completion
            duration_ms = int((time.time() - start_time) * 1000)
            if completion_audit_id:
                await audit_service.log_operation_success(
                    audit_id=completion_audit_id,
                    duration_ms=duration_ms,
                    metadata={
                        "status": "auto_completed",
                        "auto_resolved_count": len(high_conf_changes),
                        "nodes_count": len(merged_graph.nodes),
                        "edges_count": len(merged_graph.edges),
                    },
                )

        return merged_graph

    except Exception as e:
        # Log failure
        duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=duration_ms
            )

        # Re-raise the exception
        raise e


def _reconcile_graphs(
    staging_graph: GraphResponse, prod_graph: GraphResponse
) -> GraphResponse:
    """Reconcile staging and production graphs for re-merge"""
    merged_graph = copy.deepcopy(staging_graph)
    prod_node_map = {n.id: n for n in prod_graph.nodes}
    prod_edge_map = {
        (e.source, e.target, e.type): e
        for e in prod_graph.edges
        if e.properties.get(VALID_TO) is None
    }

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
            edge.properties = {
                **prod_edge.properties,
                **edge.properties,
            }  # Merge properties
        else:
            edge.id = str(uuid.uuid4())  # New edge gets a new ID

    return merged_graph


def _log_merge_failure(merge_id: str, error: str):
    _safe_supabase_call(
        "log_merge_failure",
        lambda: supabase.table("merge_status")
        .update({"status": MergeStatus.FAILED, "error": error})
        .eq("merge_id", merge_id)
        .execute(),
    )


@task(name="merge-log-failure", retries=3, retry_delay_seconds=10)
def log_merge_failure_task(merge_id: str, error: str):
    _log_merge_failure(merge_id, error)


def get_merge_status(merge_id: str) -> MergeStatus:
    merge_status = _safe_supabase_call(
        "get_merge_status",
        lambda: supabase.table("merge_status")
        .select("*")
        .eq("merge_id", merge_id)
        .execute(),
    )
    if not merge_status.data:
        return MergeStatus.NOT_FOUND
    return MergeStatus(merge_status.data[0]["status"])


def get_human_review_items(merge_id: str) -> List[ChangeLog]:
    change_logs_data = _safe_supabase_call(
        "get_human_review_items",
        lambda: supabase.table("change_logs")
        .select("*")
        .eq("merge_id", merge_id)
        .eq("need_human_review", True)
        .execute(),
    )
    change_logs = []
    for change_log in change_logs_data.data:
        record = ChangeLogRecord.from_supabase(change_log)
        prop_changes = {
            key: (record.changed_props.get(key), record.previous_props.get(key))
            for key in record.changed_props
        }

        change_logs.append(
            ChangeLog(
                id=record.id,
                prop_changes=prop_changes,
                staging_node=Node(
                    id=record.node_id,
                    label=record.node_type,
                    type=record.node_type,
                    properties=record.changed_props,
                ),
                prod_node=Node(
                    id=record.prod_node_id,
                    label=record.node_type,
                    type=record.node_type,
                    properties=record.previous_props,
                ),
                created_at=record.created_at or datetime.utcnow(),
                need_human_review=record.need_human_review,
                match_confidence=record.match_confidence,
                match_strategy=record.match_strategy,
            )
        )
    return change_logs


async def apply_resolution(
    merge_id: str,
    change_log_id: str,
    resolved_props: Dict[str, Any],
    resolution: ResolutionStrategy,
    learning_comment: str,
    user_id: str,
) -> bool:
    """
    Apply a resolution to a conflict and save the learning for future reference.

    Args:
        merge_id: ID of the merge process
        change_log_id: ID of the conflict to resolve
        resolution: The resolution decision : resolved properties
        learning_comment: Comment on the resolution
        user_id: User's ID

    Returns:
        True if the resolution was applied successfully, False otherwise
    """
    change_log = _safe_supabase_call(
        "fetch_change_log",
        lambda: supabase.table("change_logs")
        .select("*")
        .eq("merge_id", merge_id)
        .eq("id", change_log_id)
        .execute(),
    )
    if len(change_log.data) > 0:
        conflict_record = ChangeLogRecord.from_supabase(change_log.data[0])

        # Get ontology_id from merge_status
        merge_info = _safe_supabase_call(
            "fetch_merge_info",
            lambda: supabase.table("merge_status")
            .select("ontology_id")
            .eq("merge_id", merge_id)
            .execute(),
        )
        if not merge_info.data:
            logger.error(f"Could not find merge info for merge_id {merge_id}")
            return False

        ontology_id = merge_info.data[0].get("ontology_id")

        # Save resolution for future learning
        save_resolution_task(
            merge_id=merge_id,
            change_log_id=change_log_id,
            ontology_id=ontology_id,
            node_id=conflict_record.node_id,
            node_type=conflict_record.node_type,
            previous_props=conflict_record.previous_props,
            changed_props=conflict_record.changed_props,
            resolved_props=resolved_props,
            resolution=resolution,
            learning_comment=learning_comment,
        )

        # Check if all conflicts are resolved
        unresolved_conflicts = _safe_supabase_call(
            "fetch_unresolved_conflicts",
            lambda: supabase.table("change_logs")
            .select("*")
            .eq("merge_id", merge_id)
            .eq("need_human_review", True)
            .execute(),
        )
        if len(unresolved_conflicts.data) == 0:
            update_merge_status_task(merge_id, MergeStatus.READY_TO_MERGE)
            transform_id = merge_info.data[0].get("transform_id")
            if transform_id and ontology_id:
                await merge_flow(merge_id, transform_id, ontology_id, user_id)
        return True
    return False


async def get_merge_statistics(merge_id: str) -> Dict[str, Any]:
    merge_status = (
        supabase.table("merge_status")
        .select("statistics")
        .eq("merge_id", merge_id)
        .execute()
    )
    if not merge_status.data:
        return None

    statistics = merge_status.data[0].get("statistics")
    return statistics if statistics else None


async def get_merge_graph(
    merge_id: str, transform_id: str, user_id: str
) -> GraphResponse:
    """Get the merged graph for a merge operation"""
    logger.info(
        f"get_merge_graph called with merge_id: {merge_id}, transform_id: {transform_id}, user_id: {user_id}"
    )

    # Check merge status
    merge_status = (
        supabase.table("merge_status")
        .select("status")
        .eq("merge_id", merge_id)
        .execute()
    )
    if not merge_status.data:
        logger.warning(f"No merge status found for merge_id: {merge_id}")
        return None

    status = merge_status.data[0]["status"]
    logger.info(f"Merge status for {merge_id}: {status}")

    # Always fetch staging graph and apply resolved changes
    # This ensures we show the merged result even after completion
    logger.info(f"Fetching staging graph for transform_id: {transform_id}")
    staging_graph = await _extract_staging_graph(transform_id, user_id)
    logger.info(
        f"Retrieved staging graph with {len(staging_graph.nodes)} nodes and {len(staging_graph.edges)} edges"
    )

    # Fetch change_logs and apply resolved changes on top of staging graph
    change_logs = (
        supabase.table("change_logs").select("*").eq("merge_id", merge_id).execute()
    )
    logger.info(
        f"Found {len(change_logs.data) if change_logs.data else 0} change logs for merge_id: {merge_id}"
    )

    if not change_logs.data:
        logger.info("No change logs found, returning original staging graph")
        return staging_graph

    # Create a map of node IDs for quick lookup
    node_map = {node.id: node for node in staging_graph.nodes}
    applied_changes = 0

    for change_log in change_logs.data:
        try:
            node_id = change_log.get("node_id")
            if not node_id or node_id not in node_map:
                logger.warning(f"Node ID {node_id} not found in staging graph")
                continue

            staging_node = node_map[node_id]

            # Apply resolved changes (both auto-resolved and manually resolved)
            if change_log.get("changed_props") and not change_log.get(
                "need_human_review", False
            ):
                # Only apply if the conflict has been resolved (not needing human review)
                staging_node.properties.update(change_log["changed_props"])
                applied_changes += 1
                logger.debug(f"Applied resolved changes to node {node_id}")

            # Mark nodes that still need review (shouldn't happen after all conflicts are resolved)
            if change_log.get("need_human_review", False):
                staging_node.properties["__NEED_REVIEW"] = True
                logger.debug(f"Marked node {node_id} as needing review")

        except Exception as e:
            logger.error(f"Error processing change log: {str(e)}")

    logger.info(
        f"Applied {applied_changes} resolved changes. Returning staging graph with modifications: {len(staging_graph.nodes)} nodes, {len(staging_graph.edges)} edges"
    )
    return staging_graph


@task(name="extract_staging_graph")
async def _extract_staging_graph(transform_id: str, user_id: str) -> GraphResponse:
    """Extract Staging Graph"""
    start_time = time.time()

    # Get user's staging database configuration
    user_config = await UserDatabaseService.get_user_config(user_id)

    from app.services.storage.neo4j import Neo4jStorage

    storage = Neo4jStorage(
        uri=user_config.stagingDb.uri,
        username=user_config.stagingDb.username,
        password=user_config.stagingDb.password,
        database="neo4j",  # Default database name
    )

    try:
        # Extract nodes with transform_id
        storage_nodes = await storage.get_nodes_by_property(
            property_name=TRANSFORM_ID, property_value=transform_id
        )

        if not storage_nodes:
            logger.warning(f"No nodes found with transform_id {transform_id}")
            return GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)

        # Convert storage nodes to schema nodes
        nodes = [
            Node(
                id=str(node.id),
                label=node.label,
                type=node.type,
                properties=node.properties,
            )
            for node in storage_nodes
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
                properties=edge.properties,
            )
            for edge in storage_edges
        ]

        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Extracted {len(nodes)} nodes and {len(edges)} relationships "
            f"in {duration_ms:.2f}ms"
        )

        return GraphResponse(
            nodes=nodes, edges=edges, total_nodes=len(nodes), total_edges=len(edges)
        )

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract staging graph: {str(e)}")
        raise


@task(name="map_production_entities")
async def _map_production_entities(
    staging_graph: GraphResponse,
    ontology: Dict[str, Any],
    user_id: str,
    similarity_threshold: float = 0.7,
    merge_id: Optional[str] = None,
    transform_id: Optional[str] = None,
) -> EntityMappingResult:
    try:
        start_time = time.time()
        matches = {}
        matched_count = 0

        # Get user's production database configuration
        user_config = await UserDatabaseService.get_user_config(user_id)

        from app.services.storage.neo4j import Neo4jStorage

        storage = Neo4jStorage(
            uri=user_config.prodDb.uri,
            username=user_config.prodDb.username,
            password=user_config.prodDb.password,
            database="neo4j",  # Default database name
        )

        candidates = []

        for node in staging_graph.nodes:
            match = await _get_matching_nodes(
                storage, node, ontology, similarity_threshold
            )
            matches[node.id] = match
            if match.production_matches:
                if match.best_match is None:
                    entity_props = ontology["entities"][node.type]["properties"]
                    prod_candidates = "\n".join(
                        [
                            _get_node_string(_n, entity_props)
                            for _n in match.production_matches
                        ]
                    )
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

        # analyse candidates and add to matches only if there is high confidence
        if len(candidates) > 0:
            from app.utils.baml_usage_tracker import track_baml_get_matching_nodes
            from app.utils.llm_helper import (
                get_user_llm_credentials,
                create_baml_client_registry,
            )

            # Get user's LLM credentials and create client registry
            api_key, model_name = await get_user_llm_credentials(user_id)
            client_registry = create_baml_client_registry(api_key, model_name)

            matching_nodes = await track_baml_get_matching_nodes(
                user_id=user_id,
                candidate_sets=candidates,
                merge_id=merge_id,
                transform_id=transform_id,
                client_registry=client_registry,
            )
            for match in matching_nodes:
                if match.node_id:
                    prod_matches = matches[match.staging_node_id].production_matches
                    best_match = next(
                        (n for n in prod_matches if n.id == match.node_id), None
                    )
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
            mapping_time_ms=duration_ms,
        )

    except Exception as e:
        logger.error(f"Failed to map production entities: {str(e)}")
        raise


def _merge_nodes(
    staging_node: Node,
    prod_node: Node,
    ontology: Dict[str, Any],
    merged_graph: GraphResponse,
) -> GraphResponse:
    """Merge two nodes and ensure all edges are updated. Only version if properties change."""
    entity_def = ontology
    # Filter properties based on the ontology definition
    staging_props_relevant = {
        k: v for k, v in staging_node.properties.items() if k in entity_def
    }
    prod_props_relevant = {
        k: v for k, v in prod_node.properties.items() if k in entity_def
    }

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
    _update_to_prod_node_id_in_edges(
        old_id=old_staging_id, new_id=prod_node.id, merged_graph=merged_graph
    )

    # Only version the production node if properties actually changed
    if properties_changed:
        logger.info(f"Properties changed for node {prod_node.id}. Versioning required.")
        # Version the old production node state
        prod_node.properties[VALID_TO] = (
            current_time_str  # Use the same timestamp for consistency
        )
        prev_ver = Node(
            id=str(uuid.uuid4()),
            label=prod_node.label,
            type=prod_node.type,
            properties=prod_node.properties,  # Properties before merge, now marked with VALID_TO
        )
        merged_graph.nodes.append(prev_ver)
        merged_graph.edges.append(
            Edge(
                id=str(uuid.uuid4()),
                target=prev_ver.id,
                source=prod_node.id,  # Link from the current node ID
                type=PREVIOUS_VERSION_RELATIONSHIP_TYPE,
                properties={UPDATED: current_time_str},
            )
        )
        logger.debug(
            f"Merged node: {prod_node.id}, Previous version created: {prev_ver.id}"
        )
    else:
        logger.info(
            f"No relevant property changes for node {prod_node.id}. Skipping versioning."
        )
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

    # Remove the original prod_node if it exists separately in the list
    # This assumes prod_node was fetched separately and might be a different object
    # than the one potentially already in merged_graph.nodes with the same ID.
    # A safer approach might be to ensure only ONE node with prod_node.id exists *after* the merge.

    final_nodes = []
    ids_seen = set()
    for node in merged_graph.nodes:
        if node.id == prod_node.id:
            if prod_node.id not in ids_seen:
                final_nodes.append(node)  # Keep the modified one
                ids_seen.add(prod_node.id)
        elif (
            node.id != old_staging_id
        ):  # Avoid adding back the original staging node if it lingered
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


def _apply_id_reconciliation(
    merged_graph: GraphResponse,
    mapping_result: EntityMappingResult,
    confidence_threshold: float,
) -> None:
    """Update staging node IDs to production IDs where matches are confident."""

    id_map: Dict[str, str] = {}
    target_claims: Dict[str, str] = {}

    for staging_id, match in mapping_result.matches.items():
        if not match.best_match:
            continue
        if match.match_confidence < confidence_threshold:
            logger.debug(
                "Skipping ID reconciliation for %s: confidence %.2f below threshold %.2f",
                staging_id,
                match.match_confidence,
                confidence_threshold,
            )
            continue

        target_id = match.best_match.id
        previous_staging = target_claims.get(target_id)
        if previous_staging:
            prev_conf = mapping_result.matches[previous_staging].match_confidence
            if match.match_confidence > prev_conf:
                logger.warning(
                    "Reassigning production node %s from staging %s to %s based on higher confidence %.2f > %.2f",
                    target_id,
                    previous_staging,
                    staging_id,
                    match.match_confidence,
                    prev_conf,
                )
                id_map.pop(previous_staging, None)
                id_map[staging_id] = target_id
                target_claims[target_id] = staging_id
            else:
                logger.warning(
                    "Multiple staging nodes map to production %s; keeping %s (confidence %.2f) over %s (confidence %.2f)",
                    target_id,
                    previous_staging,
                    prev_conf,
                    staging_id,
                    match.match_confidence,
                )
                continue
        else:
            id_map[staging_id] = target_id
            target_claims[target_id] = staging_id

        match.metadata["id_reconciled"] = True
        match.metadata["mapped_to"] = target_id
    for staging_id, match in mapping_result.matches.items():
        if staging_id not in id_map:
            match.metadata.setdefault("id_reconciled", False)

    if not id_map:
        return

    logger.info("Reconciling %s staging IDs with production IDs", len(id_map))

    # Update nodes
    reconciled_nodes: Dict[str, Node] = {}
    updated_nodes: List[Node] = []
    for node in merged_graph.nodes:
        original_id = node.id
        if original_id in id_map:
            node.properties.setdefault("_staging_id", original_id)
            node.id = id_map[original_id]

        existing = reconciled_nodes.get(node.id)
        if existing:
            existing.properties = {**node.properties, **existing.properties}
        else:
            reconciled_nodes[node.id] = node
            updated_nodes.append(node)

    merged_graph.nodes = updated_nodes

    # Update edges to reference reconciled IDs
    for edge in merged_graph.edges:
        if edge.source in id_map:
            edge.source = id_map[edge.source]
        if edge.target in id_map:
            edge.target = id_map[edge.target]

    # Deduplicate edges after ID updates
    edge_seen: Dict[Tuple[str, str, str], Edge] = {}
    deduped_edges: List[Edge] = []
    for edge in merged_graph.edges:
        key = (edge.source, edge.target, edge.type)
        existing = edge_seen.get(key)
        if existing:
            existing.properties = {**edge.properties, **existing.properties}
        else:
            edge_seen[key] = edge
            deduped_edges.append(edge)

    merged_graph.edges = deduped_edges


def _update_to_prod_node_id_in_edges(
    old_id: str, new_id: str, merged_graph: GraphResponse
):
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
    ontology: Dict[str, Any],
    similarity_threshold: float,
) -> EntityMatch:
    """Find matching nodes in production based on node type and properties"""
    try:
        # Start with most specific matching strategy
        strategy = MatchStrategy.EXACT_NAME
        matches = []
        best_match = None

        logger.debug(
            f"Finding matches for staging node {node.id} ({node.type}) with properties: {node.properties}"
        )

        entity_props = (
            ontology.get("entities", {}).get(node.type, {}).get("properties", {})
        )
        matched_property = None

        # Strategy 1: Try unique property matching first
        unique_props = [
            (prop_name, node.properties.get(prop_name))
            for prop_name, prop_def in entity_props.items()
            if prop_def.get("unique") and node.properties.get(prop_name)
        ]
        for prop_name, prop_value in unique_props:
            matches = await storage.find_nodes_by_property_value(
                label=node.label,
                property_name=prop_name,
                property_value=prop_value,
            )
            logger.debug(
                "Unique-property matching on %s=%s found %s candidate(s)",
                prop_name,
                prop_value,
                len(matches),
            )
            if matches:
                strategy = MatchStrategy.UNIQUE_PROPERTY
                best_match = matches[0] if len(matches) == 1 else None
                matched_property = prop_name
                break

        # If no unique property match, try exact name
        if not matches:
            # Strategy 1: Try exact name matching first
            if "name" in node.properties and node.properties["name"]:
                matches = await storage.find_nodes_by_property_value(
                    label=node.label,
                    property_name="name",
                    property_value=node.properties["name"],
                )
                logger.debug(
                    f"Name-based matching found {len(matches)} candidates for '{node.properties['name']}'"
                )
                if matches:
                    best_match = matches[0] if len(matches) == 1 else None
                    strategy = MatchStrategy.EXACT_NAME
                    matched_property = "name"

        # Strategy 2: Try ID-based matching if no name matches
        if not matches and "id" in node.properties and node.properties["id"]:
            matches = await storage.find_nodes_by_property_value(
                label=node.label,
                property_name="id",
                property_value=node.properties["id"],
            )
            logger.debug(
                f"ID-based matching found {len(matches)} candidates for ID '{node.properties['id']}'"
            )
            if matches:
                best_match = matches[0] if len(matches) == 1 else None
                strategy = MatchStrategy.EXACT_NAME
                matched_property = "id"

        # Strategy 3: Use property similarity as fallback
        if not matches:
            strategy = MatchStrategy.PROPERTY_SIMILARITY
            matches = await storage.find_similar_nodes(
                label=node.label,
                properties=node.properties,
                similarity_threshold=similarity_threshold,
            )
            logger.debug(
                f"Similarity-based matching found {len(matches)} candidates (threshold: {similarity_threshold})"
            )

        # Calculate confidence based on strategy and number of matches
        if strategy == MatchStrategy.EXACT_NAME:
            confidence = (
                0.9 if len(matches) == 1 else 0.7
            )  # Lower confidence if multiple exact matches
        else:
            confidence = 0.3  # Lower confidence for similarity matches

        if matches:
            logger.info(
                f"Found {len(matches)} potential matches for staging node {node.id} using {strategy.value}"
            )
        else:
            logger.info(
                f"No matches found for staging node {node.id} - will be treated as new node"
            )

        return EntityMatch(
            staging_id=node.id,
            production_matches=matches,
            best_match=best_match,
            match_confidence=confidence,
            match_strategy=strategy.value,
            metadata={
                "total_matches": len(matches),
                "node_label": node.label,
                "matching_property": matched_property
                or (
                    "name"
                    if "name" in node.properties and node.properties.get("name")
                    else "similarity"
                ),
            },
        )

    except Exception as e:
        logger.error(f"Failed to get matching nodes for {node.id}: {str(e)}")
        raise


@task(name="persist_to_prod")
async def _persist_to_prod(
    merged_graph: GraphResponse,
    merge_id: str,
    transform_id: str,
    user_id: str,
    metrics: Optional[MergePerformanceMetrics] = None,
) -> Dict[str, Any]:
    # Get user's production database configuration
    user_config = await UserDatabaseService.get_user_config(user_id)

    node_results: List[StorageBatchResult] = []
    for batch_index, chunk in enumerate(
        _iter_chunks(merged_graph.nodes, settings.MERGE_NODE_BATCH_SIZE)
    ):
        result = await persist_node_batch_task(
            uri=user_config.prodDb.uri,
            username=user_config.prodDb.username,
            password=user_config.prodDb.password,
            node_batch=chunk,
            batch_index=batch_index,
            transform_id=transform_id,
            merge_id=merge_id,
        )
        node_results.append(result)

        if metrics:
            metrics.record_node_batch(result.processing_time_ms, result.items_processed)

        if not result.success:
            logger.error(f"Failed to persist node batch {batch_index}: {result.error}")
            raise Exception(f"Node persistence failed: {result.error}")

    # Create a node map from persisted nodes
    node_map = {n.id: n for n in merged_graph.nodes}

    # Convert edges to RelationshipInstance, ensuring valid IDs
    edges_as_rel_instances = []
    for edge in merged_graph.edges:
        if edge.source in node_map and edge.target in node_map:
            edges_as_rel_instances.append(
                RelationshipInstance(
                    id=edge.id,
                    source_id=edge.source,
                    target_id=edge.target,
                    type=edge.type,
                    source_type=node_map[edge.source].type,
                    target_type=node_map[edge.target].type,
                    properties=edge.properties,
                )
            )
        else:
            logger.warning(
                f"Skipping edge {edge.id}: Source {edge.source} or Target {edge.target} not in node_map"
            )

    # Store relationships with versioning logic
    edge_results: List[StorageBatchResult] = []
    for batch_index, chunk in enumerate(
        _iter_chunks(edges_as_rel_instances, settings.MERGE_REL_BATCH_SIZE)
    ):
        result = await persist_relationship_batch_task(
            uri=user_config.prodDb.uri,
            username=user_config.prodDb.username,
            password=user_config.prodDb.password,
            relationships=chunk,
            batch_index=batch_index,
            transform_id=transform_id,
            merge_id=merge_id,
        )
        edge_results.append(result)

        if metrics:
            metrics.record_relationship_batch(
                result.processing_time_ms, result.items_processed
            )

        if not result.success:
            logger.error(
                f"Failed to persist relationship batch {batch_index}: {result.error}"
            )
            raise Exception(f"Edge persistence failed: {result.error}")

    node_summary = _summarize_batch_results(node_results)
    edge_summary = _summarize_batch_results(edge_results)

    return {
        "nodes": node_summary,
        "edges": edge_summary,
    }


def _iter_chunks(items: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    """Yield fixed-size chunks from a list."""
    if chunk_size <= 0:
        yield items
        return

    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


def _summarize_batch_results(results: List[StorageBatchResult]) -> Dict[str, Any]:
    """Create a consolidated view of batch execution results."""

    if not results:
        return {
            "total_items": 0,
            "total_time_ms": 0.0,
            "batches": [],
            "success": True,
            "warnings": [],
        }

    total_items = sum(result.items_processed for result in results)
    total_time_ms = sum(result.processing_time_ms for result in results)
    success = all(result.success for result in results)
    warnings: List[str] = []
    for result in results:
        warnings.extend(result.warnings)

    return {
        "total_items": total_items,
        "total_time_ms": total_time_ms,
        "batches": [result.model_dump() for result in results],
        "success": success,
        "warnings": warnings,
    }


def _add_ingestion_stats(
    merge_id: str,
    node_summary: Dict[str, Any],
    edge_summary: Dict[str, Any],
    metrics: Optional[MergePerformanceMetrics] = None,
) -> None:
    statistics = {
        "nodes_stored": node_summary,
        "edges_stored": edge_summary,
    }
    if metrics:
        statistics["performance"] = metrics.model_dump(mode="json")
    else:
        existing_stats = _safe_supabase_call(
            "fetch_existing_statistics",
            lambda: supabase.table("merge_status")
            .select("statistics")
            .eq("merge_id", merge_id)
            .execute(),
        )
        if existing_stats.data:
            performance = (
                existing_stats.data[0].get("statistics", {}).get("performance")
            )
            if performance is not None:
                statistics["performance"] = performance

    _safe_supabase_call(
        "update_merge_statistics",
        lambda: supabase.table("merge_status")
        .update({"statistics": statistics, "status": MergeStatus.COMPLETED})
        .eq("merge_id", merge_id)
        .execute(),
    )


@task(name="merge-add-ingestion-stats", retries=3, retry_delay_seconds=10)
def add_ingestion_stats_task(
    merge_id: str,
    node_summary: Dict[str, Any],
    edge_summary: Dict[str, Any],
    metrics: Optional[MergePerformanceMetrics] = None,
) -> None:
    _add_ingestion_stats(merge_id, node_summary, edge_summary, metrics)


def _record_merge_metrics(merge_id: str, metrics: MergePerformanceMetrics) -> None:
    """Persist performance metrics for visibility in the merge dashboard."""
    existing_stats = _safe_supabase_call(
        "fetch_existing_statistics",
        lambda: supabase.table("merge_status")
        .select("statistics")
        .eq("merge_id", merge_id)
        .execute(),
    )

    statistics: Dict[str, Any] = {}
    if existing_stats.data:
        existing_statistics = existing_stats.data[0].get("statistics") or {}
        if isinstance(existing_statistics, dict):
            statistics.update(existing_statistics)

    statistics["performance"] = metrics.model_dump(mode="json")

    _safe_supabase_call(
        "record_merge_metrics",
        lambda: supabase.table("merge_status")
        .update({"statistics": statistics})
        .eq("merge_id", merge_id)
        .execute(),
    )


@task(name="merge-record-metrics", retries=3, retry_delay_seconds=10)
def record_merge_metrics_task(merge_id: str, metrics: MergePerformanceMetrics) -> None:
    _record_merge_metrics(merge_id, metrics)


async def _get_prod_graph(merge_id: str, user_id: str) -> GraphResponse:
    logger.info(f"_get_prod_graph called with merge_id: {merge_id}, user_id: {user_id}")

    # Get user's production database configuration
    user_config = await UserDatabaseService.get_user_config(user_id)
    logger.info(
        f"Retrieved user config for production database: {user_config.prodDb.uri}"
    )

    from app.services.storage.neo4j import Neo4jStorage

    storage = Neo4jStorage(
        uri=user_config.prodDb.uri,
        username=user_config.prodDb.username,
        password=user_config.prodDb.password,
        database="neo4j",  # Default database name
    )

    try:
        graph_data = await storage.get_merge_data(merge_id)
        logger.info(
            f"Retrieved production graph data: {len(graph_data.nodes) if graph_data else 0} nodes, {len(graph_data.edges) if graph_data else 0} edges"
        )
        return graph_data
    except Exception as e:
        logger.error(f"Error retrieving production graph data: {str(e)}")
        raise


async def get_past_resolution(ontology_id: str, node_type: str) -> str:
    response = (
        supabase.table("resolutions").select("*").eq("node_type", node_type).execute()
    )
    if not response.data:
        return "None"
    learnings = []
    i = 1
    for log in response.data:
        # Handle different resolution strategies
        if log["resolution"] == ResolutionStrategy.KEEP_BOTH.value:
            resolution_description = "Keep Both (Staging and Production)"
        else:
            try:
                resolution_description = ResolutionStrategy(log["resolution"]).value
            except ValueError:
                # Fallback for unknown resolution values
                resolution_description = log["resolution"]

        learning = f"""
        Learning {i}:
            Existing Properties (Production): {log['previous_props']}
            Incoming Properties (Staging): {log['changed_props']}
            Resolution by the User: {log['resolved_props']}
            Resolution Strategy: {resolution_description}
            User Comment (rationale): {log['learning_comment']}
        """
        learnings.append(learning)
        i += 1
    return "\n".join(learnings)


def _save_change_log(merge_id, change_log, need_human_review: bool = False):
    record = change_log.to_record(merge_id, need_human_review=need_human_review)
    payload = record.to_supabase_payload()

    _safe_supabase_call(
        "save_change_log",
        lambda: supabase.table("change_logs")
        .upsert(payload, on_conflict="id")
        .execute(),
    )


@task(name="save-change-log", retries=3, retry_delay_seconds=10)
def save_change_log_task(
    merge_id: str, change_log: ChangeLog, need_human_review: bool = False
):
    _save_change_log(merge_id, change_log, need_human_review)


def _save_resolution(
    merge_id: str,
    change_log_id: str,
    ontology_id: str,
    node_id: str,
    node_type: str,
    previous_props: Dict,
    changed_props: Dict,
    resolved_props: Dict,
    resolution: ResolutionStrategy,
    learning_comment: str,
):
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
    resolution_record = ChangeLogResolution(
        merge_id=merge_id,
        ontology_id=ontology_id,
        node_id=node_id,
        node_type=node_type,
        previous_props=previous_props,
        changed_props=changed_props,
        resolved_props=resolved_props,
        resolution=resolution,
        learning_comment=learning_comment or None,
    )

    payload = resolution_record.to_supabase_payload()

    _safe_supabase_call(
        "save_resolution",
        lambda: supabase.table("resolutions")
        .upsert(payload, on_conflict="id")
        .execute(),
    )

    updated_props = {} if resolution == ResolutionStrategy.KEEP_BOTH else resolved_props

    _safe_supabase_call(
        "mark_change_log_resolved",
        lambda: supabase.table("change_logs")
        .update(
            {
                "need_human_review": False,
                "changed_props": updated_props,
            }
        )
        .eq("merge_id", merge_id)
        .eq("id", change_log_id)
        .execute(),
    )

    logger.info(f"Successfully saved resolution for node {node_id}")


@task(name="save-resolution", retries=3, retry_delay_seconds=10)
def save_resolution_task(
    merge_id: str,
    change_log_id: str,
    ontology_id: str,
    node_id: str,
    node_type: str,
    previous_props: Dict,
    changed_props: Dict,
    resolved_props: Dict,
    resolution: ResolutionStrategy,
    learning_comment: str,
):
    _save_resolution(
        merge_id,
        change_log_id,
        ontology_id,
        node_id,
        node_type,
        previous_props,
        changed_props,
        resolved_props,
        resolution,
        learning_comment,
    )


def _start_merge_status_impl(merge_id, transform_id, ontology_id):
    _safe_supabase_call(
        "start_merge_status",
        lambda: supabase.table("merge_status")
        .insert(
            {
                "merge_id": merge_id,
                "transform_id": transform_id,
                "ontology_id": ontology_id,
                "status": MergeStatus.STARTED,
            }
        )
        .execute(),
    )


@task(name="merge-start-status", retries=3, retry_delay_seconds=10)
def start_merge_status_task(merge_id, transform_id, ontology_id):
    _start_merge_status_impl(merge_id, transform_id, ontology_id)


def _update_merge_status_impl(merge_id, status):
    _safe_supabase_call(
        "update_merge_status",
        lambda: supabase.table("merge_status")
        .update({"status": status})
        .eq("merge_id", merge_id)
        .execute(),
    )


@task(name="merge-update-status", retries=3, retry_delay_seconds=10)
def update_merge_status_task(merge_id, status):
    _update_merge_status_impl(merge_id, status)


@task(name="complete_prod_merge")
async def _complete_prod_merge(
    merge_id, transform_id, ontology_id, ontology, staging_graph, merged_graph, user_id
):
    """Complete the production merge by applying all resolved conflicts"""
    start_time = time.time()

    # Create audit trail for merge completion
    completion_audit_id = await audit_service.log_operation_start(
        user_id=user_id,
        operation_type=OperationType.MERGE_COMPLETED,
        operation_id=merge_id,
        resource_name=f"Merge {merge_id[:8]} Completion",
        metadata={
            "transform_id": transform_id,
            "ontology_id": ontology_id,
            "stage": "post_human_review",
        },
    )

    try:
        # Get all change logs for this merge
        change_logs = _safe_supabase_call(
            "fetch_change_logs",
            lambda: supabase.table("change_logs")
            .select("*")
            .eq("merge_id", merge_id)
            .execute(),
        )

        if not change_logs.data:
            logger.info(
                f"No change logs found for merge {merge_id}, proceeding with direct merge"
            )
            summary = await _persist_to_prod(
                merged_graph, merge_id, transform_id, user_id
            )
            add_ingestion_stats_task(
                merge_id, summary["nodes"], summary["edges"], metrics=None
            )
            update_merge_status_task(merge_id, MergeStatus.COMPLETED)

            # Log completion
            duration_ms = int((time.time() - start_time) * 1000)
            if completion_audit_id:
                await audit_service.log_operation_success(
                    audit_id=completion_audit_id,
                    duration_ms=duration_ms,
                    metadata={
                        "nodes_count": len(merged_graph.nodes),
                        "edges_count": len(merged_graph.edges),
                        "conflicts_resolved": 0,
                    },
                )

            return merged_graph

        # Create a map of node IDs for quick lookup
        node_map = {node.id: node for node in staging_graph.nodes}
        conflicts_resolved = 0

        # Apply all resolved conflicts
        for change_log in change_logs.data:
            try:
                node_id = change_log.get("node_id")
                node_type = change_log.get("node_type")

                if not node_id or not node_type or node_id not in node_map:
                    logger.warning(
                        f"Invalid change log entry: node_id={node_id}, node_type={node_type}"
                    )
                    continue

                # Get the staging node
                staging_node = node_map[node_id]

                # Apply the resolved properties if any
                if change_log.get("changed_props"):
                    conflicts_resolved += 1
                    if isinstance(change_log["changed_props"], str):
                        # Handle the case where properties are stored as JSON string
                        try:
                            if change_log["changed_props"] == "{}":
                                continue
                            resolved_props = json.loads(change_log["changed_props"])
                            staging_node.properties.update(resolved_props)
                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed to parse changed_props JSON: {change_log['changed_props']}"
                            )
                    else:
                        # Handle the case where properties are stored as dictionary
                        staging_node.properties.update(change_log["changed_props"])

            except Exception as e:
                logger.error(f"Error processing change log during prod merge: {str(e)}")

        # Persist the merged graph to production
        summary = await _persist_to_prod(merged_graph, merge_id, transform_id, user_id)
        add_ingestion_stats_task(
            merge_id, summary["nodes"], summary["edges"], metrics=None
        )
        update_merge_status_task(merge_id, MergeStatus.COMPLETED)

        # Log successful completion
        duration_ms = int((time.time() - start_time) * 1000)
        if completion_audit_id:
            await audit_service.log_operation_success(
                audit_id=completion_audit_id,
                duration_ms=duration_ms,
                metadata={
                    "nodes_count": len(merged_graph.nodes),
                    "edges_count": len(merged_graph.edges),
                    "conflicts_resolved": conflicts_resolved,
                },
            )

        return merged_graph

    except Exception as e:
        logger.error(f"Error in _complete_prod_merge: {str(e)}")

        # Log failure
        duration_ms = int((time.time() - start_time) * 1000)
        if completion_audit_id:
            await audit_service.log_operation_failure(
                audit_id=completion_audit_id,
                error_message=str(e),
                duration_ms=duration_ms,
            )

        traceback.print_exc()
        log_merge_failure_task(merge_id, str(e))
        raise


def _get_node_string(node: Node, properties: Dict[str, Any]) -> str:
    props = {k: v for k, v in node.properties.items() if k in properties}
    return f"(Node Id: {node.id}, properties: {props})"


def _validate_node_comparison(staging_node: Node, prod_node: Node) -> bool:
    """Validate that two nodes can be meaningfully compared"""
    if not staging_node or not prod_node:
        logger.warning("One of the nodes is None")
        return False

    if staging_node.type != prod_node.type:
        logger.warning(
            f"Node types don't match: staging={staging_node.type}, prod={prod_node.type}"
        )
        return False

    return True


def _log_mapping_debug_info(
    mapping_result: "EntityMappingResult",
    staging_nodes: List[Node],
    ontology: Dict[str, Any],
):
    """Log detailed debugging information about entity mapping and potential conflicts"""
    logger.info("=== MAPPING DEBUG INFO ===")
    logger.info(f"Total staging nodes: {len(staging_nodes)}")
    logger.info(f"Total mapped entities: {mapping_result.matched_entities}")
    logger.info(
        f"Mapping success rate: {mapping_result.matched_entities}/{mapping_result.total_entities}"
    )

    # Log unmapped staging nodes
    unmapped_count = 0
    for node in staging_nodes:
        if (
            node.id not in mapping_result.matches
            or not mapping_result.matches[node.id].best_match
        ):
            unmapped_count += 1
            logger.debug(
                f"Unmapped staging node: {node.id} ({node.type}) - {node.properties}"
            )

    logger.info(f"Unmapped staging nodes: {unmapped_count}")

    # Log nodes with matches but no conflicts detected
    matched_no_conflicts = 0
    for node_id, match in mapping_result.matches.items():
        if match.best_match:
            staging_node = next((n for n in staging_nodes if n.id == node_id), None)
            if staging_node:
                ontology_props = ontology["entities"][staging_node.type]["properties"]
                prop_changes = _get_prop_changes(
                    staging_node, match.best_match, ontology_props
                )
                if not prop_changes:
                    matched_no_conflicts += 1
                    logger.debug(
                        f"Matched node with no conflicts: {node_id} ({staging_node.type})"
                    )
                    logger.debug(f"  Staging props: {staging_node.properties}")
                    logger.debug(f"  Production props: {match.best_match.properties}")

    logger.info(f"Matched nodes with no conflicts: {matched_no_conflicts}")
    logger.info("=== END MAPPING DEBUG INFO ===")


def _get_prop_changes(
    staging_node: Node, prod_node: Node, props: Dict[str, Any]
) -> Dict[str, Tuple[Any, Any]]:
    """
    Detect property changes between staging and production nodes.
    Returns a dictionary with property_name -> (staging_value, prod_value) for all changes.
    """
    if not _validate_node_comparison(staging_node, prod_node):
        return {}

    changes = {}

    logger.debug(
        f"Comparing nodes - Staging: {staging_node.id} vs Production: {prod_node.id}"
    )
    logger.debug(f"Staging properties: {staging_node.properties}")
    logger.debug(f"Production properties: {prod_node.properties}")
    logger.debug(f"Ontology properties to check: {list(props.keys())}")

    # Check all properties defined in the ontology
    for prop_name in props.keys():
        staging_value = staging_node.properties.get(prop_name)
        prod_value = prod_node.properties.get(prop_name)

        logger.debug(
            f"Checking property '{prop_name}': staging='{staging_value}' vs prod='{prod_value}'"
        )

        # Case 1: Property exists in both - check for value differences
        if staging_value is not None and prod_value is not None:
            # Convert to strings for comparison to handle type differences
            staging_str = str(staging_value).strip()
            prod_str = str(prod_value).strip()
            logger.debug(f"  String comparison: '{staging_str}' vs '{prod_str}'")
            if staging_str != prod_str:
                changes[prop_name] = (staging_value, prod_value)
                logger.info(
                    f"Property conflict detected - {prop_name}: '{prod_str}' -> '{staging_str}'"
                )
            else:
                logger.debug(f"  No change detected for property '{prop_name}'")

        # Case 2: Property exists only in staging (new property)
        elif staging_value is not None and prod_value is None:
            changes[prop_name] = (staging_value, None)
            logger.info(
                f"New property detected - {prop_name}: None -> '{staging_value}'"
            )

        # Case 3: Property exists only in production (property deletion)
        elif staging_value is None and prod_value is not None:
            changes[prop_name] = (None, prod_value)
            logger.info(
                f"Property deletion detected - {prop_name}: '{prod_value}' -> None"
            )

        # Case 4: Property exists in neither (skip)
        else:
            logger.debug(f"  Property '{prop_name}' exists in neither node")

    logger.debug(f"Total changes detected: {len(changes)}")
    return changes


def _group_changes_by_entity_type(
    change_logs: List[ChangeLog],
) -> Dict[str, List[ChangeLog]]:
    change_log_by_entity_type = {}
    for change_log in change_logs:
        if change_log.staging_node.type not in change_log_by_entity_type:
            change_log_by_entity_type[change_log.staging_node.type] = []
        change_log_by_entity_type[change_log.staging_node.type].append(change_log)
    return change_log_by_entity_type


def _get_change_log_string(change_log: ChangeLog) -> str:
    changes = "\n".join(
        [f"{k}: {v[1]} -> {v[0]}" for k, v in change_log.prop_changes.items()]
    )
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
    change_log_by_entity_type: Dict[str, List[ChangeLog]],
    merge_id: Optional[str] = None,
    transform_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Tuple[List[ChangeLog], List[ChangeLog]]:
    """
    Classify changes into high confidence and low confidence changes
    """
    high_conf_changes = []
    changes_for_human_review = []
    for entity_type, change_logs in change_log_by_entity_type.items():
        # get past resolutions
        past_resolutions = await get_past_resolution(ontology_id, entity_type)
        # get LLM response
        changes = [
            _get_change_log_string(change_log)
            for change_log in change_logs
            if change_log.prop_changes
        ]
        if not changes:
            high_conf_changes.extend(change_logs)
            continue
        change_log_string = "\n".join(changes)
        from app.utils.baml_usage_tracker import track_baml_eval_changes
        from app.utils.llm_helper import (
            get_user_llm_credentials,
            create_baml_client_registry,
        )

        # Get user's LLM credentials and create client registry
        api_key, model_name = await get_user_llm_credentials(user_id)
        client_registry = create_baml_client_registry(api_key, model_name)

        eval_changes = await track_baml_eval_changes(
            user_id=user_id,
            change_logs=change_log_string,
            past_resolutions=past_resolutions,
            merge_id=merge_id,
            transform_id=transform_id,
            ontology_id=ontology_id,
            client_registry=client_registry,
        )
        if len(eval_changes) > 0:
            for change in eval_changes:
                if change.confidence_score > 0.95:
                    if not change.corrections:
                        change_log = next(
                            (c for c in change_logs if c.id == change.id), None
                        )
                        if change_log:
                            high_conf_changes.append(change_log)
                        continue
                    for correction in change.corrections:
                        change_log = next(
                            (c for c in change_logs if c.id == change.id), None
                        )
                        if (
                            change_log
                            and correction.prop_name in change_log.prop_changes
                        ):
                            change_log.prop_changes[correction.prop_name] = (
                                change_log.prop_changes[correction.prop_name][0],
                                correction.prop_value,
                            )
                        elif change_log:
                            # If property doesn't exist in prop_changes, create it with empty string as original value
                            change_log.prop_changes[correction.prop_name] = (
                                "",
                                correction.prop_value,
                            )
                        if change_log:
                            high_conf_changes.append(change_log)
                else:
                    change_log = next(
                        (c for c in change_logs if c.id == change.id), None
                    )
                    if change_log:
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


def test_conflict_detection(
    staging_node_props: Dict, prod_node_props: Dict, ontology_props: Dict
) -> Dict:
    """
    Test function to manually check conflict detection logic.
    Useful for debugging specific property conflict scenarios.

    Example usage:
    staging_props = {"A": "456", "name": "test_node"}
    prod_props = {"A": "123", "name": "test_node"}
    ontology_props = {"A": {"type": "string"}, "name": {"type": "string"}}
    conflicts = test_conflict_detection(staging_props, prod_props, ontology_props)
    """
    from app.schemas.graph import Node

    staging_node = Node(
        id="test_staging",
        label="TestNode",
        type="TestType",
        properties=staging_node_props,
    )

    prod_node = Node(
        id="test_prod", label="TestNode", type="TestType", properties=prod_node_props
    )

    return _get_prop_changes(staging_node, prod_node, ontology_props)
