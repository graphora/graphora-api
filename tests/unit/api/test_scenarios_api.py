"""Unit tests for the /api/v1/scenarios endpoints (B6-scenario slice 1).

Tests cover the four endpoints (create / list / get / delete)
across their happy paths plus the contract pins that prevent
cross-tenant leakage and silent truncation. Service is patched
at the route layer so we exercise the wire shape + tenant
threading without standing up real storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.main import app
from graphora_server.schemas.graph import GraphResponse, Node
from graphora_server.services.scenario_service import (
    Scenario,
    ScenarioConflictError,
    ScenarioNotFoundError,
)


@pytest.fixture
def test_client():
    def fake_auth():
        return AuthContext(user_id="user-1", token="t", claims={})

    app.dependency_overrides[get_current_auth] = fake_auth
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def _record(
    *,
    scenario_id: str = "sc-1",
    user_id: str = "user-1",
    transform_id: str = "tx-1",
    name: str = "baseline",
    description: str = None,
    snapshot_nodes: int = 1,
    snapshot_edges: int = 0,
) -> Scenario:
    """Build a service-layer Scenario record. Helper keeps the
    test bodies focused on the API contract instead of dataclass
    field plumbing."""
    return Scenario(
        id=scenario_id,
        user_id=user_id,
        transform_id=transform_id,
        name=name,
        description=description,
        graph_snapshot={
            "nodes": [
                {
                    "id": f"n{i}",
                    "label": f"Node{i}",
                    "type": "Person",
                    "properties": {},
                }
                for i in range(snapshot_nodes)
            ],
            "edges": [
                {
                    "id": f"e{i}",
                    "source": f"n{i}",
                    "target": f"n{i+1}",
                    "type": "KNOWS",
                    "properties": {},
                }
                for i in range(snapshot_edges)
            ],
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _graph_response(*, n_nodes: int = 1) -> GraphResponse:
    """Server-side graph response — what _load_graph_for_diff returns."""
    return GraphResponse(
        nodes=[
            Node(id=f"n{i}", label=f"N{i}", type="Person", properties={})
            for i in range(n_nodes)
        ],
        edges=[],
        total_nodes=n_nodes,
        total_edges=0,
    )


# ============================================================
# POST /api/v1/scenarios — create
# ============================================================


def test_create_scenario_snapshots_transform_graph(test_client):
    """POST /api/v1/scenarios with a valid transform_id must
    snapshot the graph and return the full record (incl. graph
    payload) on 201. Pin the wire shape so a refactor that drops
    the embedded graph or returns a summary instead regresses."""
    graph = _graph_response(n_nodes=2)
    expected = _record(name="my-scenario", snapshot_nodes=2)

    with (
        patch(
            "graphora_server.api.scenarios._load_graph_for_diff",
            new=AsyncMock(return_value=graph),
        ),
        patch(
            "graphora_server.api.scenarios._service",
        ) as service_factory,
    ):
        service = service_factory.return_value
        service.create_from_transform = AsyncMock(return_value=expected)

        response = test_client.post(
            "/api/v1/scenarios",
            json={"transform_id": "tx-1", "name": "my-scenario"},
        )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] == "sc-1"
    assert body["name"] == "my-scenario"
    assert body["transform_id"] == "tx-1"
    assert body["node_count"] == 2
    # Full graph payload embedded on the create response.
    assert "graph" in body
    assert len(body["graph"]["nodes"]) == 2


def test_create_scenario_passes_auth_user_id_to_service(test_client):
    """The user_id from the auth context must flow into the
    service call. Tenant scoping pin: a refactor that drops the
    arg would silently let users create scenarios under a
    different tenant — same severity as the loader-tenant
    regression caught for /golden/score."""
    graph = _graph_response()
    with (
        patch(
            "graphora_server.api.scenarios._load_graph_for_diff",
            new=AsyncMock(return_value=graph),
        ),
        patch(
            "graphora_server.api.scenarios._service",
        ) as service_factory,
    ):
        service = service_factory.return_value
        service.create_from_transform = AsyncMock(return_value=_record())

        test_client.post(
            "/api/v1/scenarios",
            json={"transform_id": "tx-1", "name": "x"},
        )

    service.create_from_transform.assert_awaited_once()
    kwargs = service.create_from_transform.await_args.kwargs
    assert kwargs["user_id"] == "user-1"
    assert kwargs["transform_id"] == "tx-1"
    assert kwargs["name"] == "x"


def test_create_scenario_409_on_duplicate_name(test_client):
    """Service raises ScenarioConflictError on a (user_id,
    transform_id, name) collision — must surface as 409. Pin
    so a refactor that lets the conflict bubble as 500 (or, worse,
    silently 201s a second scenario) regresses."""
    graph = _graph_response()
    with (
        patch(
            "graphora_server.api.scenarios._load_graph_for_diff",
            new=AsyncMock(return_value=graph),
        ),
        patch(
            "graphora_server.api.scenarios._service",
        ) as service_factory,
    ):
        service = service_factory.return_value
        service.create_from_transform = AsyncMock(
            side_effect=ScenarioConflictError("dup")
        )

        response = test_client.post(
            "/api/v1/scenarios",
            json={"transform_id": "tx-1", "name": "dup"},
        )

    assert response.status_code == 409
    assert "dup" in response.text


def test_create_scenario_404_when_transform_graph_empty(test_client):
    """Empty / missing / cross-tenant transform → 404. For
    /golden/score this is "score against empty actual"; for
    scenarios it's "can't snapshot nothing." Pin the distinction
    so the two endpoints' postures don't accidentally converge."""
    empty = GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)
    with patch(
        "graphora_server.api.scenarios._load_graph_for_diff",
        new=AsyncMock(return_value=empty),
    ):
        response = test_client.post(
            "/api/v1/scenarios",
            json={"transform_id": "missing-tx", "name": "x"},
        )

    assert response.status_code == 404
    assert "missing-tx" in response.text


def test_create_scenario_413_when_source_truncated(test_client):
    """Mirror /diff and /golden/score's truncation guard: an
    oversized source transform must 413, not silently snapshot
    a partial graph. Pin so a refactor that drops the
    _check_truncated call regresses noisily (same class of bug
    e21c1f5 fixed on /golden/score)."""
    oversized = GraphResponse(
        nodes=[Node(id="n1", label="x", type="t", properties={})],
        edges=[],
        total_nodes=15000,  # > 10k cap
        total_edges=0,
    )
    with patch(
        "graphora_server.api.scenarios._load_graph_for_diff",
        new=AsyncMock(return_value=oversized),
    ):
        response = test_client.post(
            "/api/v1/scenarios",
            json={"transform_id": "tx-huge", "name": "x"},
        )

    assert response.status_code == 413
    body = response.json()["detail"]
    assert body["error"] == "transform_too_large_to_diff"
    assert body["side"] == "scenario_source"


def test_create_scenario_413_when_source_truncated_with_empty_nodes(
    test_client,
):
    """Reviewer-flagged Medium on commit d7a1f6e. Pre-fix the
    empty-graph 404 check fired BEFORE _check_truncated, so a
    transform with total_nodes=15000 + nodes=[] (the staging
    reader's "cap truncated everything" outcome) returned 404
    instead of 413. The status confusion makes the failure
    debuggable only via logs — the caller sees "no such
    transform" but the transform exists and is just too big.

    Pin the corrected ordering: truncation runs first, so any
    transform exceeding the cap surfaces as 413 regardless of
    whether the loader returned nodes."""
    truncated_to_empty = GraphResponse(
        nodes=[],  # cap fired, returned empty
        edges=[],
        total_nodes=15000,  # > 10k cap
        total_edges=0,
    )
    with patch(
        "graphora_server.api.scenarios._load_graph_for_diff",
        new=AsyncMock(return_value=truncated_to_empty),
    ):
        response = test_client.post(
            "/api/v1/scenarios",
            json={"transform_id": "tx-truncated-empty", "name": "x"},
        )

    assert response.status_code == 413, (
        "Oversized + empty-nodes must surface as 413 (the truncation "
        f"signal), not 404. Got {response.status_code}: "
        f"{response.text}"
    )
    body = response.json()["detail"]
    assert body["error"] == "transform_too_large_to_diff"
    assert body["side"] == "scenario_source"


def test_create_scenario_rejects_missing_name(test_client):
    """Pydantic validation: name is required + min-length 1.
    422 before any service call. Pin so a refactor that loosens
    the schema doesn't accidentally let empty-named scenarios
    in (where the list view would render a blank row)."""
    response = test_client.post(
        "/api/v1/scenarios",
        json={"transform_id": "tx-1"},
    )
    assert response.status_code == 422


# ============================================================
# GET /api/v1/scenarios — list
# ============================================================


def test_list_scenarios_returns_summaries_without_graph(test_client):
    """List endpoint returns ScenarioSummary shape — counts only,
    no embedded graph. Pin so a refactor that returns full
    Scenario records would blow out response sizes for users
    with many large scenarios."""
    records = [
        _record(scenario_id="sc-1", name="a", snapshot_nodes=3),
        _record(scenario_id="sc-2", name="b", snapshot_nodes=1, snapshot_edges=2),
    ]
    with patch("graphora_server.api.scenarios._service") as service_factory:
        service = service_factory.return_value
        service.list_for_user = AsyncMock(return_value=records)

        response = test_client.get("/api/v1/scenarios")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    # Summary shape: counts present, graph absent.
    assert body[0]["node_count"] == 3
    assert "graph" not in body[0]
    assert body[1]["edge_count"] == 2


def test_list_scenarios_passes_auth_user_id(test_client):
    """Tenant scoping at the API layer: list must be called with
    auth.user_id. Same severity as the create test."""
    with patch("graphora_server.api.scenarios._service") as service_factory:
        service = service_factory.return_value
        service.list_for_user = AsyncMock(return_value=[])

        test_client.get("/api/v1/scenarios")

    service.list_for_user.assert_awaited_once_with("user-1")


# ============================================================
# GET /api/v1/scenarios/{id} — get
# ============================================================


def test_get_scenario_returns_full_record_with_graph(test_client):
    """Detail endpoint returns the full Scenario including graph
    payload. Pin the wire shape so a refactor that drops the
    graph (or moves it to a separate endpoint) regresses."""
    record = _record(name="detail", snapshot_nodes=2, snapshot_edges=1)
    with patch("graphora_server.api.scenarios._service") as service_factory:
        service = service_factory.return_value
        service.get = AsyncMock(return_value=record)

        response = test_client.get("/api/v1/scenarios/sc-1")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "sc-1"
    assert "graph" in body
    assert len(body["graph"]["nodes"]) == 2
    assert len(body["graph"]["edges"]) == 1


def test_get_scenario_404_when_service_raises_not_found(test_client):
    """ScenarioNotFoundError → 404. Pin so a regression that
    surfaces as 500 (or leaks cross-tenant existence via a
    different status) is caught."""
    with patch("graphora_server.api.scenarios._service") as service_factory:
        service = service_factory.return_value
        service.get = AsyncMock(side_effect=ScenarioNotFoundError("not found"))

        response = test_client.get("/api/v1/scenarios/nope")

    assert response.status_code == 404


# ============================================================
# DELETE /api/v1/scenarios/{id}
# ============================================================


def test_delete_scenario_returns_204_on_success(test_client):
    """Delete returns 204 No Content on success — pin so a
    refactor that returns 200 with a body would break the
    contract."""
    with patch("graphora_server.api.scenarios._service") as service_factory:
        service = service_factory.return_value
        service.delete = AsyncMock(return_value=None)

        response = test_client.delete("/api/v1/scenarios/sc-1")

    assert response.status_code == 204
    # 204 must have no body — verify the response is empty.
    assert response.content == b""


def test_delete_scenario_404_when_unknown_or_cross_tenant(test_client):
    """Delete of a nonexistent / cross-tenant id raises NotFound
    from the service → 404 from the route. Pin so a refactor
    that silently 204s the missing case (which would let
    attackers probe existence via timing) regresses noisily."""
    with patch("graphora_server.api.scenarios._service") as service_factory:
        service = service_factory.return_value
        service.delete = AsyncMock(side_effect=ScenarioNotFoundError("not found"))

        response = test_client.delete("/api/v1/scenarios/nope")

    assert response.status_code == 404
