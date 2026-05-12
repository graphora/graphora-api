from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict, List, Optional
from graphora_server.schemas.graph import GraphResponse
from graphora_server.schemas.graph_changes import SaveGraphRequest, SaveGraphResponse
from graphora_server.services.user_db_service import (
    UserDatabaseService,
    is_memory_storage_enabled,
)
from graphora_server.services.decision_log_service import (
    DecisionLogService,
    DecisionType,
)
from graphora_server.services.diff_service import DiffService
from graphora_server.services.usage_tracking import UsageTrackingService
from graphora_server.utils.logger import logger
import traceback
from graphora_server.auth import AuthContext, get_current_auth

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])


@router.get(
    "/{transform_id}",
    response_model=GraphResponse,
    description="Retrieve nodes by transform ID and their relationships",
)
async def get_graph_by_transform_id(
    transform_id: str,
    limit: Optional[int] = 1000,
    skip: Optional[int] = 0,
    auth: AuthContext = Depends(get_current_auth),
) -> GraphResponse:
    """
    Retrieve nodes by transform ID and their relationships from user's staging database

    Parameters:
    - transform_id: Transform ID to query
    - user_id: User's ID (from header)
    - limit: Maximum number of nodes to return (default: 1000)
    - skip: Number of nodes to skip for pagination (default: 0)

    Returns:
    - GraphResponse containing:
        - nodes: Array of nodes with properties
        - edges: Array of relationships between nodes
        - total_nodes: Total count of nodes with this label
        - total_edges: Total count of relationships
    """
    graph_service = None
    try:
        # Validate inputs
        if limit < 0 or skip < 0:
            raise HTTPException(
                status_code=400, detail="Limit and skip must be non-negative"
            )

        if limit > 10000:
            raise HTTPException(status_code=400, detail="Maximum limit is 10000 nodes")

        # Check if user has staging DB configured or if memory storage is globally enabled
        from graphora_server.services.storage.factory import user_has_staging_db

        use_in_memory = is_memory_storage_enabled() or not await user_has_staging_db(
            auth.user_id
        )

        if use_in_memory:
            from graphora_server.services.storage.memory import InMemoryStorage

            storage = InMemoryStorage(user_id=auth.user_id)
            response = await storage.get_transformation_data(transform_id)

            # Apply pagination
            nodes = response.nodes[skip : skip + limit]
            response = GraphResponse(
                nodes=nodes,
                edges=response.edges,
                total_nodes=response.total_nodes,
                total_edges=response.total_edges,
                metadata=response.metadata,
            )

            logger.info(
                "Retrieved %s nodes and %s edges for user %s from in-memory storage",
                len(response.nodes),
                len(response.edges),
                auth.user_id,
            )
            return response

        # Get user's staging database (graph operations always use staging)
        graph_service = await UserDatabaseService.get_staging_graph_service(
            auth.user_id
        )

        # Get graph data
        response = graph_service.get_graph_by_transform_id(
            transform_id=transform_id, limit=limit, skip=skip
        )

        if not response.nodes:
            logger.warning(
                "No nodes found with transform_id %s for user %s",
                transform_id,
                auth.user_id,
            )

        logger.info(
            "Retrieved %s nodes and %s edges for user %s from staging database",
            len(response.nodes),
            len(response.edges),
            auth.user_id,
        )
        return response

    except ValueError as e:
        logger.error("Configuration error for user %s: %s", auth.user_id, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        logger.error(
            "Error retrieving graph data for user %s: %s", auth.user_id, str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Error retrieving graph data: {str(e)}"
        )
    finally:
        if graph_service:
            graph_service.close()


def _decision_to_dict(decision: Any) -> Dict[str, Any]:
    """Serialize a Decision dataclass into the JSON shape the
    /decisions response contract specifies. Enums are serialized
    to string values so the response survives JSON transport
    without requiring a Python-typed consumer.

    Mirror of the helper in graphora_server/mcp/server.py — kept
    duplicated rather than abstracted because the two surfaces
    have different consumers (REST API vs MCP tool) and the
    serialization choice could legitimately diverge later."""
    return {
        "id": decision.id,
        "transform_id": decision.transform_id,
        "target_id": decision.target_id,
        "target_kind": decision.target_kind.value,
        "decision_type": decision.decision_type.value,
        "reason": decision.reason,
        "evidence": decision.evidence,
        "alternatives": decision.alternatives,
        "created_at": decision.created_at,
    }


@router.get(
    "/{transform_id}/decisions",
    description=(
        "Decision Log entries for a transform. With ``node_id`` set, "
        "returns schema-level decisions plus the per-node decisions "
        "for that node, schema-first. Without ``node_id``, returns "
        "only schema-level decisions."
    ),
)
async def get_decisions_by_transform_id(
    transform_id: str,
    node_id: Optional[str] = None,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    """B0-explain (reviewer-flagged on commit 9ac9bb5): the Decision
    Log lives on the API side, not in MCP. MCP is documented and
    implemented as an HTTP client; reading decisions directly from
    a local DecisionLogService inside the MCP process would either
    fall into an empty in-memory store (no DATABASE_URL) or open a
    new direct DB dependency / secret surface (DATABASE_URL set).

    This endpoint owns the read so MCP stays a pure HTTP client.

    Schema-level decisions (target_kind=schema, decision_type=
    schema_inferred) are always included for context — they're
    transform-level prerequisites for any per-node "why is this
    here?" answer. Node decisions follow if ``node_id`` is supplied.

    Performance pin: schema decisions are fetched via
    ``for_decision_type`` (uses the (transform_id, decision_type)
    index from migration 14) so node-evidence lookups don't scale
    with the total decision count for the transform.

    Returns:
        decision_log (list): Schema decisions first, then node
            decisions if ``node_id`` is supplied. Each entry: ``{id,
            transform_id, target_id, target_kind, decision_type,
            reason, evidence, alternatives, created_at}``.
        alternatives (list): Aggregated candidate entities the
            pipeline considered for ``node_id`` across all merge
            decisions. Empty when ``node_id`` is None or the node
            had no merge events.
    """
    try:
        log = DecisionLogService()

        # P1 follow-up (commit eb22a79): scope reads by auth.user_id
        # so authenticated user A can't fetch user B's transform
        # decisions just by knowing the transform_id. Writers
        # (entity-merge hook, schema-inference hook) stamp the
        # row with their user_id; reads filter on it. Rows with
        # NULL user_id (legacy / pre-migration-15) won't match
        # any specific user — they're orphaned by design.
        schema_decisions = await log.for_decision_type(
            transform_id,
            DecisionType.SCHEMA_INFERRED,
            user_id=auth.user_id,
        )

        node_decisions: List[Any] = []
        alternatives: List[Dict[str, Any]] = []
        if node_id:
            node_decisions = await log.for_target(
                transform_id, node_id, user_id=auth.user_id
            )
            for d in node_decisions:
                alternatives.extend(d.alternatives)

        # Schema first (causation chain: schema is the prerequisite
        # for the node merges that followed). The Evidence tab
        # renders this as a top-down narrative; flattening to
        # walltime would mis-narrate causation for re-extractions
        # where node merges land before a schema decision in
        # walltime.
        decision_dicts = [
            _decision_to_dict(d) for d in schema_decisions + node_decisions
        ]

        return {
            "decision_log": decision_dicts,
            "alternatives": alternatives,
        }
    except Exception as e:
        traceback.print_exc()
        logger.error(
            "Error fetching decisions for transform %s, user %s: %s",
            transform_id,
            auth.user_id,
            str(e),
        )
        raise HTTPException(
            status_code=500, detail=f"Error fetching decisions: {str(e)}"
        )


@router.get(
    "/{transform_id}/cost",
    description=(
        "Per-transform LLM cost / token report. Aggregates llm_usage "
        "rows for the transform: total calls, input/output/total "
        "tokens, estimated cost in USD, distinct models used, and "
        "a per-operation-type breakdown."
    ),
)
async def get_cost_by_transform_id(
    transform_id: str,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    """B5-obs: agent-facing cost surface.

    Tenant-scoped via auth.user_id (same pattern as /decisions —
    a request for another user's transform_id returns the zero-row
    aggregate). Surfaces existing llm_usage data without forcing
    callers to scrape the dashboard endpoint.

    Returns:
        transform_id (str): Echo of the input.
        total_calls (int): Count of LLM invocations on this transform.
        input_tokens / output_tokens / total_tokens (int): Sums.
        estimated_cost_usd (str | None): Sum of estimated_cost_usd
            values as a string for JSON precision. None when no
            row had pricing (e.g., the model wasn't in the
            model_pricing table) — distinguishes "cost is zero"
            from "cost is unknown".
        models_used (list[str]): Distinct ``"<provider>:<model>"``.
        by_operation_type (dict): Per-op breakdown, same shape as
            the top-level totals.
    """
    try:
        service = UsageTrackingService()
        return await service.get_transform_cost_report(
            transform_id=transform_id,
            user_id=auth.user_id,
        )
    except Exception as e:
        traceback.print_exc()
        logger.error(
            "Error fetching cost report for transform %s, user %s: %s",
            transform_id,
            auth.user_id,
            str(e),
        )
        raise HTTPException(
            status_code=500, detail=f"Error fetching cost report: {str(e)}"
        )


async def _load_graph_for_diff(transform_id: str, user_id: str) -> GraphResponse:
    """Helper: fetch one transform's full graph via the same
    backend-selection logic the main /graph/{transform_id}
    endpoint uses (in-memory vs staging DB). Pagination is NOT
    applied — diffs need the whole graph, not a page slice.

    The hard cap of 10k matches the user-facing endpoint's max
    limit; transforms bigger than that need a different surface
    (streaming diff) that's out of scope for this slice."""
    from graphora_server.services.storage.factory import user_has_staging_db

    use_in_memory = is_memory_storage_enabled() or not await user_has_staging_db(
        user_id
    )

    if use_in_memory:
        from graphora_server.services.storage.memory import InMemoryStorage

        storage = InMemoryStorage(user_id=user_id)
        return await storage.get_transformation_data(transform_id)

    graph_service = await UserDatabaseService.get_staging_graph_service(user_id)
    try:
        return graph_service.get_graph_by_transform_id(
            transform_id=transform_id, limit=10000, skip=0
        )
    finally:
        graph_service.close()


@router.get(
    "/{base_transform_id}/diff/{compare_transform_id}",
    description=(
        "B3-diff: structured graph-state diff between two transforms "
        "(same user). Returns added / removed / changed nodes and "
        "edges, plus a summary the rendering layer can use without "
        "walking the full payload. Node identity matches across "
        "transforms by canonical_id (Gate 4 ER), falling back to "
        "type:canonical_key, falling back to per-transform id."
    ),
)
async def diff_transforms(
    base_transform_id: str,
    compare_transform_id: str,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    """B3-diff endpoint. Both transforms must belong to the
    authenticated user — tenant scoping follows the same pattern
    as /decisions and /cost (the graph reader at the storage
    layer scopes by user_id via the staging DB lookup).

    Returns:
        base_transform_id (str): Echo.
        compare_transform_id (str): Echo.
        summary (dict): ``{nodes: {added, removed, changed,
            unchanged}, edges: {...}}``.
        added_nodes (list): Nodes in compare but not base.
        removed_nodes (list): Nodes in base but not compare.
        changed_nodes (list): Nodes in both with property changes.
            Each: ``{canonical_id, type, base_id, compare_id,
            property_changes: {<key>: {base, compare}}}``.
        added_edges / removed_edges / changed_edges (lists):
            Same shape, edge-level.
    """
    try:
        base_graph = await _load_graph_for_diff(base_transform_id, auth.user_id)
        compare_graph = await _load_graph_for_diff(compare_transform_id, auth.user_id)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(
            "Error loading graphs for diff (base=%s, compare=%s, user=%s): %s",
            base_transform_id,
            compare_transform_id,
            auth.user_id,
            str(e),
        )
        raise HTTPException(
            status_code=500, detail=f"Error loading graphs for diff: {str(e)}"
        )

    # Reviewer-flagged P2 on commit a75cd73: the 10k node loader
    # cap meant a transform with more nodes than that returned a
    # confident-looking but silently incomplete diff. Fail loud
    # (413) when either side was truncated rather than expose a
    # ``truncated: true`` flag — diffs MUST be correct to be
    # useful, and a flag can be missed by agents/CLI consumers.
    # Streaming for >10k transforms is a future slice.
    _check_truncated(base_graph, "base", base_transform_id)
    _check_truncated(compare_graph, "compare", compare_transform_id)

    diff = DiffService().diff(
        base_graph=base_graph,
        compare_graph=compare_graph,
        base_transform_id=base_transform_id,
        compare_transform_id=compare_transform_id,
    )
    return _diff_to_dict(diff)


def _check_truncated(graph: GraphResponse, side: str, transform_id: str) -> None:
    """Raise HTTPException(413) when the loaded graph is smaller
    than its reported total_nodes. ``total_nodes`` is populated
    by both backends (in-memory + Neo4j) from a count query that
    is independent of the LIMIT applied to the data fetch, so a
    mismatch is a reliable truncation signal."""
    if graph.total_nodes is None:
        return
    if len(graph.nodes) < graph.total_nodes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "transform_too_large_to_diff",
                "side": side,
                "transform_id": transform_id,
                "returned_nodes": len(graph.nodes),
                "total_nodes": graph.total_nodes,
                "reason": (
                    f"The {side} transform contains {graph.total_nodes} "
                    f"nodes but the diff loader caps reads at "
                    f"{len(graph.nodes)}. Streaming diff for large "
                    "transforms is not yet implemented; refusing to "
                    "return a silent partial diff."
                ),
            },
        )


def _diff_to_dict(diff: Any) -> Dict[str, Any]:
    """Serialize the GraphDiff dataclass for JSON transport.

    Nodes and edges are already Pydantic models; ``.model_dump()``
    produces JSON-clean dicts. NodeDelta / EdgeDelta /
    PropertyChange are plain dataclasses — convert via
    dataclasses.asdict so nested ``PropertyChange(base=..., compare=...)``
    instances render as ``{"base": ..., "compare": ...}``."""
    from dataclasses import asdict

    return {
        "base_transform_id": diff.base_transform_id,
        "compare_transform_id": diff.compare_transform_id,
        "summary": {
            "nodes": {
                "added": diff.summary.nodes_added,
                "removed": diff.summary.nodes_removed,
                "changed": diff.summary.nodes_changed,
                "unchanged": diff.summary.nodes_unchanged,
            },
            "edges": {
                "added": diff.summary.edges_added,
                "removed": diff.summary.edges_removed,
                "changed": diff.summary.edges_changed,
                "unchanged": diff.summary.edges_unchanged,
            },
        },
        "added_nodes": [n.model_dump(mode="json") for n in diff.added_nodes],
        "removed_nodes": [n.model_dump(mode="json") for n in diff.removed_nodes],
        "changed_nodes": [asdict(n) for n in diff.changed_nodes],
        "added_edges": [e.model_dump(mode="json") for e in diff.added_edges],
        "removed_edges": [e.model_dump(mode="json") for e in diff.removed_edges],
        "changed_edges": [asdict(e) for e in diff.changed_edges],
    }


@router.put(
    "/{transform_id}",
    response_model=SaveGraphResponse,
    description="Save graph changes in a single transaction",
)
async def save_graph_changes(
    transform_id: str,
    changes: SaveGraphRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> SaveGraphResponse:
    """
    Save bulk modifications to the user's staging graph database

    Parameters:
    - transform_id: Transformation ID
    - changes: Batch of modifications to apply
    - user_id: User's ID (from header)

    Returns:
    - Updated graph data
    - New version
    - Warning/info messages
    """
    graph_service = None
    try:
        # Check if user has staging DB configured or if memory storage is globally enabled
        from graphora_server.services.storage.factory import user_has_staging_db

        use_in_memory = is_memory_storage_enabled() or not await user_has_staging_db(
            auth.user_id
        )

        if use_in_memory:
            # Use in-memory storage for saving changes
            from graphora_server.services.storage.memory import InMemoryStorage

            storage = InMemoryStorage(user_id=auth.user_id)
            result = await storage.save_graph_changes(transform_id, changes)

            logger.info(
                "Saved graph changes for user %s in in-memory storage",
                auth.user_id,
            )
            return result

        # Get user's staging database (graph operations always use staging)
        graph_service = await UserDatabaseService.get_staging_graph_service(
            auth.user_id
        )

        # Save changes
        result = graph_service.save_graph_changes(transform_id, changes)

        logger.info(
            "Saved graph changes for user %s in staging database",
            auth.user_id,
        )
        return result

    except ValueError as e:
        logger.error("Configuration error for user %s: %s", auth.user_id, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error saving graph changes for user %s: %s", auth.user_id, str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error saving changes: {str(e)}")
    finally:
        if graph_service:
            graph_service.close()
