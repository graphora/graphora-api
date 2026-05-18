from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict, List, Optional
from graphora_server.schemas.claims import ContradictionsResponse
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
from graphora_server.services.claims_service import ClaimsService
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
        "Decision Log entries for a transform. Optionally narrow to "
        "a single target by setting ``node_id`` OR ``edge_id`` "
        "(mutually exclusive). Schema-level decisions are always "
        "returned first as the causation prefix. Without either id, "
        "returns only schema-level decisions."
    ),
)
async def get_decisions_by_transform_id(
    transform_id: str,
    node_id: Optional[str] = None,
    edge_id: Optional[str] = None,
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
    transform-level prerequisites for any per-node/per-edge
    "why is this here?" answer. Per-target decisions follow if
    ``node_id`` OR ``edge_id`` is supplied.

    Edge support (reviewer-flagged on slice C wrap-up): the closed
    set of DecisionTypes already includes RELATIONSHIP_ACCEPTED and
    RELATIONSHIP_REJECTED, and ``DecisionLogService.for_target`` is
    generic over target_id (works for any target_kind). What was
    missing pre-fix was the query-path that lets callers ASK for a
    specific edge's decisions — without it, ``graphora explain
    <edge>`` couldn't return source text. The ``edge_id`` query
    param closes that gap. ``node_id`` and ``edge_id`` are mutually
    exclusive because a single Decision target_id belongs to one
    kind only; mixing them in a single request would either
    over-fetch or surface ambiguous results.

    Performance pin: schema decisions are fetched via
    ``for_decision_type`` (uses the (transform_id, decision_type)
    index from migration 14) so target-evidence lookups don't scale
    with the total decision count for the transform.

    Returns:
        decision_log (list): Schema decisions first, then per-target
            decisions if a node_id/edge_id is supplied. Each entry:
            ``{id, transform_id, target_id, target_kind,
            decision_type, reason, evidence, alternatives,
            created_at}``.
        alternatives (list): Aggregated candidate entities the
            pipeline considered for the target across all merge
            decisions. Empty when no node_id/edge_id is supplied or
            the target had no merge events.
    """
    # Mutex check: node and edge target_ids occupy different identity
    # namespaces in real data (UUIDs are globally unique in practice)
    # but the SEMANTIC contract is one-target-per-request. Returning
    # 400 surfaces the ambiguity at the API surface instead of
    # silently picking one.
    if node_id is not None and edge_id is not None:
        raise HTTPException(
            status_code=400,
            detail="node_id and edge_id are mutually exclusive; pass exactly one.",
        )

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

        target_decisions: List[Any] = []
        alternatives: List[Dict[str, Any]] = []
        target_id = node_id or edge_id
        if target_id:
            target_decisions = await log.for_target(
                transform_id, target_id, user_id=auth.user_id
            )
            for d in target_decisions:
                alternatives.extend(d.alternatives)

        # Schema first (causation chain: schema is the prerequisite
        # for the node/edge merges that followed). The Evidence tab
        # renders this as a top-down narrative; flattening to
        # walltime would mis-narrate causation for re-extractions
        # where target merges land before a schema decision in
        # walltime.
        decision_dicts = [
            _decision_to_dict(d) for d in schema_decisions + target_decisions
        ]

        return {
            "decision_log": decision_dicts,
            "alternatives": alternatives,
        }
    except HTTPException:
        # Don't swallow the mutex 400 with the broad-Exception 500.
        raise
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
    "/{transform_id}/contradictions",
    response_model=ContradictionsResponse,
    description=(
        "B1-prob slice 2a: surface (target, property) pairs where the "
        "extraction pipeline emitted multiple distinct claimed values. "
        "Each contradiction carries its competing claims sorted by "
        "confidence DESC — the 'winning' value is the first entry; "
        "the rest are alternatives the contradiction detector wants "
        "you to know about. ``min_confidence`` filters low-confidence "
        "noise out of the result. Returns an empty list until B1-prob "
        "slice 2b's pipeline hooks emit claims at extraction time; "
        "the surface is live so CLI/MCP callers can build against "
        "the wire shape today."
    ),
)
async def get_contradictions_by_transform_id(
    transform_id: str,
    min_confidence: float = 0.0,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    """B1-prob slice 2a: agent-facing contradictions surface.

    Tenant-scoped via auth.user_id (same pattern as /decisions
    and /cost — a request for another user's transform_id
    returns an empty contradictions list, never leaking
    existence). The underlying read goes through
    ``ClaimsService.contradictions_for_transform`` which
    groups claims by (target_id, target_kind, property_key)
    and counts distinct JSON-equal values.

    Reviewer-flagged Medium on commit 66987b2: the route now
    declares ``response_model=ContradictionsResponse`` so
    OpenAPI exposes the typed Claim/Contradiction contract to
    generated clients. Pre-fix the 200 response was a generic
    object — schema documents existed but the wire surface
    didn't claim them, so the OpenAPI snapshot test pinned a
    permissive shape rather than the real one.

    Returns:
        transform_id (str): Echo.
        min_confidence (float): The applied floor (default 0.0).
        contradictions (list): Per-(target, property) group with
            competing_claims + severity.
        total_claims_scanned (int): Total claims at/above
            min_confidence for this transform — NOT just the
            ones inside contradiction groups. Lets callers
            distinguish "no writer yet" (count=0) from "writer
            healthy, consistent data" (count>0, empty
            contradictions list). Reviewer-flagged Medium on
            commit 66987b2: pre-fix this counted only claims
            INSIDE contradiction groups, which collapses both
            states to 0 once slice 2b's writer lands.
    """
    # Pydantic's ge/le on the query parameter would catch this
    # too, but the explicit early-return gives a clearer error
    # message than the framework's "Input should be greater
    # than or equal to 0".
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise HTTPException(
            status_code=400,
            detail=("min_confidence must be in [0.0, 1.0]; got " f"{min_confidence!r}"),
        )

    service = ClaimsService()
    contradictions = await service.contradictions_for_transform(
        transform_id=transform_id,
        user_id=auth.user_id,
        min_confidence=min_confidence,
    )
    # Reviewer-flagged Medium on commit 66987b2: the
    # ``total_claims_scanned`` field documents itself as
    # "claims considered for contradiction detection (post-
    # confidence-filter)" — counting only claims inside
    # contradiction groups breaks that contract. A clean
    # 100-claim transform would report 0, indistinguishable
    # from the "no writer yet" empty state. The new
    # ``count_claims_for_transform`` does the true count via
    # ``SELECT COUNT(*)`` so we don't load every claim just to
    # count them.
    total_claims_scanned = await service.count_claims_for_transform(
        transform_id=transform_id,
        user_id=auth.user_id,
        min_confidence=min_confidence,
    )

    # Project service-layer dataclasses into wire dicts. The
    # claims_service dataclass uses TargetKind enum members; the
    # wire shape uses the string value. Project + adapt here so
    # the service stays Pydantic-free.
    contradiction_dicts: List[Dict[str, Any]] = []
    for c in contradictions:
        claims_list = [
            {
                "id": claim.id,
                "transform_id": claim.transform_id,
                "target_id": claim.target_id,
                "target_kind": claim.target_kind.value,
                "property_key": claim.property_key,
                "value": claim.value,
                "confidence": claim.confidence,
                "source_chunk_id": claim.source_chunk_id,
                "source_extractor_model": claim.source_extractor_model,
                "source_prompt_version": claim.source_prompt_version,
                "user_id": claim.user_id,
                "created_at": claim.created_at,
            }
            for claim in c.competing_claims
        ]
        contradiction_dicts.append(
            {
                "target_id": c.target_id,
                "target_kind": c.target_kind.value,
                "property_key": c.property_key,
                "competing_claims": claims_list,
                "severity": c.severity,
            }
        )

    return {
        "transform_id": transform_id,
        "min_confidence": min_confidence,
        "contradictions": contradiction_dicts,
        "total_claims_scanned": total_claims_scanned,
    }


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


# Diff loader's main-node cap. Shared between the LIMIT in the
# staging query and the upfront total_nodes check in
# _check_truncated. Must stay in lockstep — the check assumes a
# transform whose total_nodes exceeds this value would have its
# main-node page truncated by the LIMIT below.
_DIFF_NODE_LIMIT = 10000


async def _load_graph_for_diff(transform_id: str, user_id: str) -> GraphResponse:
    """Helper: fetch one transform's full graph via the same
    backend-selection logic the main /graph/{transform_id}
    endpoint uses (in-memory vs staging DB). Pagination is NOT
    applied — diffs need the whole graph, not a page slice.

    The hard cap of ``_DIFF_NODE_LIMIT`` matches the user-facing
    endpoint's max limit; transforms bigger than that need a
    different surface (streaming diff) that's out of scope for
    this slice. ``_check_truncated`` enforces the cap upfront
    using ``total_nodes`` — see that function's docstring for
    why ``len(graph.nodes)`` alone isn't a trustworthy signal."""
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
            transform_id=transform_id, limit=_DIFF_NODE_LIMIT, skip=0
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
    """Raise HTTPException(413) when the loaded graph might be
    incomplete. Three triggers, in priority order:

      1. ``total_nodes > _DIFF_NODE_LIMIT`` — the upfront cap
         check. The staging reader's query is
         ``MATCH (n) ... SKIP $skip LIMIT $limit OPTIONAL MATCH
         (n)-[r]-(m) ... collect ... connected_nodes``. The
         response.nodes list combines the paged main nodes with
         the connected nodes pulled via edge endpoints, so its
         length can REACH OR EXCEED total_nodes even when the
         main-node page was truncated. That makes ``len(nodes)``
         alone an unreliable truncation signal. The upfront
         total_nodes vs cap comparison closes that hole — see
         reviewer P2 on commit fae7e91.

      2. ``len(nodes) < total_nodes`` — catches non-staging-path
         truncation (in-memory backend, future readers) where
         the cap-check isn't load-bearing but the loader still
         returned fewer than expected.

      3. ``len(edges) < total_edges`` — closes the edge-only
         truncation hole (reviewer P2 on commit a261321).

    All three return the same structured detail body with a
    ``truncated_dimension`` field so operators see exactly
    which dimension fell over."""
    if graph.total_nodes is not None and graph.total_nodes > _DIFF_NODE_LIMIT:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "transform_too_large_to_diff",
                "side": side,
                "transform_id": transform_id,
                "truncated_dimension": "nodes",
                "total_nodes": graph.total_nodes,
                "diff_node_cap": _DIFF_NODE_LIMIT,
                "reason": (
                    f"The {side} transform contains {graph.total_nodes} "
                    f"nodes — exceeding the diff loader's cap of "
                    f"{_DIFF_NODE_LIMIT}. The staging reader merges "
                    "connected nodes into the response so len(nodes) "
                    "alone can mask main-node truncation. Streaming "
                    "diff for large transforms is not yet implemented; "
                    "refusing to return a silent partial diff."
                ),
            },
        )
    if graph.total_nodes is not None and len(graph.nodes) < graph.total_nodes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "transform_too_large_to_diff",
                "side": side,
                "transform_id": transform_id,
                "truncated_dimension": "nodes",
                "returned_nodes": len(graph.nodes),
                "total_nodes": graph.total_nodes,
                "reason": (
                    f"The {side} transform contains {graph.total_nodes} "
                    f"nodes but the diff loader returned only "
                    f"{len(graph.nodes)}. Refusing to return a silent "
                    "partial diff."
                ),
            },
        )
    if graph.total_edges is not None and len(graph.edges) < graph.total_edges:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "transform_too_large_to_diff",
                "side": side,
                "transform_id": transform_id,
                "truncated_dimension": "edges",
                "returned_edges": len(graph.edges),
                "total_edges": graph.total_edges,
                "reason": (
                    f"The {side} transform contains {graph.total_edges} "
                    f"edges but the diff loader returned only "
                    f"{len(graph.edges)} — edges between nodes excluded "
                    "by the node-pagination cap are missing. Refusing "
                    "to return a silent partial diff."
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
