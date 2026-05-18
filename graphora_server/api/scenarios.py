"""B6-scenario slice 1: REST surface for scenario snapshots.

Endpoints (all tenant-scoped by auth.user_id):

  POST   /api/v1/scenarios          — create from transform.
  GET    /api/v1/scenarios          — list scenarios for the user.
  GET    /api/v1/scenarios/{id}     — fetch one scenario incl. graph.
  DELETE /api/v1/scenarios/{id}     — delete one scenario.

Slice 1 covers create / read / delete. Scenario mutations and
copy-on-write storage land in slice 2.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.schemas.graph import GraphResponse
from graphora_server.schemas.graph_changes import SaveGraphRequest
from graphora_server.schemas.scenario import (
    Scenario as ScenarioResponse,
    ScenarioCreateRequest,
    ScenarioSummary,
)
from graphora_server.services.scenario_service import (
    Scenario as ScenarioRecord,
    ScenarioConflictError,
    ScenarioMutationError,
    ScenarioNotFoundError,
    ScenarioService,
)
from graphora_server.utils.logger import logger

# Lazy import via attribute lookup at call time — same pattern
# golden.py uses to avoid pulling the loader's transitive deps
# at module load and to keep the truncation contract in lockstep
# with /diff.
from graphora_server.api.graph import _check_truncated, _load_graph_for_diff


router = APIRouter(prefix="/api/v1/scenarios", tags=["Scenarios"])


def _service() -> ScenarioService:
    """Thin factory so tests can patch the service into the route
    layer without monkeypatching the import. The service is
    cheap to construct (just reads settings + an optional memory
    store) so per-request instantiation is fine."""
    return ScenarioService()


def _to_summary(record: ScenarioRecord) -> ScenarioSummary:
    """Project a service-layer record into the wire summary.

    The list view never needs the full graph payload — counting
    nodes/edges from the resolved view (base + diff) is enough
    for the operator to decide which scenario to drill into.
    Keeping the projection here (not in the service) means the
    service stays free of Pydantic-vs-dataclass ceremony.

    Slice 2c: uses ``record.resolved_graph()`` not
    ``record.graph_snapshot`` directly. Pre-slice-2c the
    snapshot field held the current view; post-2c it holds the
    immutable base. Reading the resolved view is the only way
    to get accurate counts after mutations land.
    """
    resolved = record.resolved_graph()
    nodes = resolved.get("nodes") or []
    edges = resolved.get("edges") or []
    return ScenarioSummary(
        id=record.id,
        transform_id=record.transform_id,
        parent_scenario_id=record.parent_scenario_id,
        name=record.name,
        description=record.description,
        created_at=record.created_at,
        node_count=len(nodes),
        edge_count=len(edges),
    )


def _to_response(record: ScenarioRecord) -> ScenarioResponse:
    """Project a service-layer record into the full wire shape
    (summary + embedded graph). The embedded graph is the
    resolved view (base + diff), so callers see the same
    contract whether the scenario has been mutated or not.

    Slice 2c: ``record.graph_snapshot`` is now the immutable
    base; the embedded ``graph`` field on the response is the
    resolved view computed at access time. Wire shape is
    unchanged from pre-2c — clients can't tell whether storage
    is materialized or CoW."""
    summary = _to_summary(record)
    resolved = record.resolved_graph()
    return ScenarioResponse(
        **summary.model_dump(),
        graph=GraphResponse.model_validate(resolved),
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    description=(
        "Create a new scenario by snapshotting the named transform's "
        "current graph. Scenario name must be unique per (user, "
        "transform_id) — a duplicate returns 409."
    ),
)
async def create_scenario(
    body: ScenarioCreateRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    """B6-scenario slice 1: snapshot a transform graph into a new
    scenario row.

    The transform graph is loaded via the same per-user storage
    helper /diff and /golden/score use, so a request for another
    user's transform_id surfaces as an empty actual graph (and a
    permission-flavored 404). The truncation guard mirrors /diff:
    a transform exceeding the 10k-node cap returns 413 here
    rather than silently snapshotting a partial graph.
    """
    try:
        graph = await _load_graph_for_diff(body.transform_id, auth.user_id)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        logger.error(
            "Error loading graph for scenario create (transform=%s, user=%s): %s",
            body.transform_id,
            auth.user_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error loading graph for scenario: {exc}",
        )

    # Truncation guard runs FIRST so a transform with total_nodes
    # above the cap surfaces as 413 even when its returned nodes
    # list came back empty. Reviewer-flagged Medium on commit
    # d7a1f6e: pre-fix the empty-graph 404 fired ahead of this
    # check, so a transform with total_nodes=15000 and nodes=[]
    # (the staging reader's "cap truncated everything" case)
    # returned 404 instead of the intended 413 — a confusingly
    # wrong status for the caller, who can't tell "this transform
    # doesn't exist" from "this transform is too big to scenario."
    _check_truncated(graph, "scenario_source", body.transform_id)

    if not graph or not graph.nodes:
        # Empty / missing / cross-tenant — collapses to the same
        # "transform not available" treatment /golden/score had
        # pre-fix. For scenarios this is the right posture: you
        # can't snapshot a graph that doesn't exist for you.
        # Distinct from /golden/score because the use case is
        # different — there, an all-FN report is a valid signal;
        # here, snapshotting nothing is a bug.
        raise HTTPException(
            status_code=404,
            detail=(
                f"No graph found for transform_id {body.transform_id!r}. "
                "Either the transform doesn't exist, it belongs to "
                "another user, or it produced zero entities."
            ),
        )

    try:
        record = await _service().create_from_transform(
            user_id=auth.user_id,
            transform_id=body.transform_id,
            name=body.name,
            description=body.description,
            graph=graph,
        )
    except ScenarioConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return _to_response(record).model_dump(mode="json")


@router.get(
    "",
    description=(
        "List scenarios owned by the authenticated user, newest "
        "first. Returns lightweight summaries (no embedded graph) "
        "so the list stays fast even when scenarios carry large "
        "materialized snapshots — use GET /api/v1/scenarios/{id} "
        "to fetch the full graph for one scenario."
    ),
)
async def list_scenarios(
    auth: AuthContext = Depends(get_current_auth),
) -> List[Dict[str, Any]]:
    records = await _service().list_for_user(auth.user_id)
    return [_to_summary(r).model_dump(mode="json") for r in records]


@router.get(
    "/{scenario_id}",
    description=(
        "Fetch one scenario by id, including the materialized "
        "graph snapshot. Tenant-scoped: cross-tenant or missing "
        "ids return 404 without distinguishing the two."
    ),
)
async def get_scenario(
    scenario_id: str,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    try:
        record = await _service().get(scenario_id, auth.user_id)
    except ScenarioNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _to_response(record).model_dump(mode="json")


@router.patch(
    "/{scenario_id}",
    description=(
        "B6-scenario slice 2b: apply node + edge mutations to "
        "the scenario's graph snapshot. The request body reuses "
        "the same SaveGraphRequest shape /graph/{tx} accepts — "
        "nodes / edges each with created / updated / deleted "
        "lists. Property updates use the same merge semantics "
        "(set a value to None to delete that key from the "
        "property bag). Mutations are validated atomically: a "
        "request that would leave any dangling edge (source or "
        "target node missing from the post-mutation state) is "
        "rejected with 422; nothing is persisted."
    ),
)
async def patch_scenario(
    scenario_id: str,
    body: SaveGraphRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, Any]:
    """Mutate a scenario's graph snapshot. Returns the full
    updated scenario record (incl. the mutated graph payload)
    on success."""
    try:
        record = await _service().apply_mutations(
            scenario_id=scenario_id,
            user_id=auth.user_id,
            changes=body,
        )
    except ScenarioNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ScenarioMutationError as exc:
        # 422 = "well-formed request, semantically invalid."
        # Distinct from 404 (scenario doesn't exist) and 400
        # (Pydantic validation failure on the request body).
        raise HTTPException(status_code=422, detail=str(exc))
    return _to_response(record).model_dump(mode="json")


@router.delete(
    "/{scenario_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    description=(
        "Delete one scenario by id. Tenant-scoped. 404 when the "
        "scenario doesn't exist for the caller (does not "
        "distinguish 'never existed' from 'belongs to another "
        "user' — see the service docstring for why)."
    ),
)
async def delete_scenario(
    scenario_id: str,
    auth: AuthContext = Depends(get_current_auth),
) -> None:
    try:
        await _service().delete(scenario_id, auth.user_id)
    except ScenarioNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return None
