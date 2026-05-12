"""Unit tests for the /api/v1/graph/{base}/diff/{compare} endpoint.

B3-diff backend. Three concerns pinned:
  * Tenant scoping: the graph reader uses the user's staging DB
    via the same backend-selection logic the main /graph
    endpoint uses (in-memory vs Postgres+Neo4j). Auth threads
    through.
  * Wire shape: response includes summary + added/removed/
    changed lists for both nodes and edges. Property changes
    serialize as ``{base, compare}`` dicts.
  * Loading both transforms before computing the diff fails
    cleanly: a load error returns 500, not a half-computed diff.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.main import app
from graphora_server.schemas.graph import Edge, GraphResponse, Node


@pytest.fixture
def test_client():
    def fake_auth():
        return AuthContext(user_id="test-user-1", token="t", claims={})

    app.dependency_overrides[get_current_auth] = fake_auth
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


def _node(id: str, type: str = "Person", canonical_id: str = None, **props) -> Node:
    if canonical_id:
        props["canonical_id"] = canonical_id
    return Node(id=id, label=props.get("name", id), type=type, properties=props)


def _edge(id: str, source: str, target: str, type: str = "WORKS_AT") -> Edge:
    return Edge(id=id, source=source, target=target, type=type, properties={})


def test_diff_endpoint_returns_added_removed_summary(test_client):
    """Happy path: base has alice; compare has alice + bob;
    diff surfaces bob as added."""
    base = GraphResponse(
        nodes=[_node("a", canonical_id="cid-alice")],
        edges=[],
    )
    compare = GraphResponse(
        nodes=[
            _node("a2", canonical_id="cid-alice"),
            _node("b", canonical_id="cid-bob"),
        ],
        edges=[],
    )

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-1" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-1/diff/tx-2")

    assert response.status_code == 200
    body = response.json()
    assert body["base_transform_id"] == "tx-1"
    assert body["compare_transform_id"] == "tx-2"
    assert body["summary"]["nodes"]["added"] == 1
    assert body["summary"]["nodes"]["removed"] == 0
    assert len(body["added_nodes"]) == 1
    assert body["added_nodes"][0]["id"] == "b"


def test_diff_endpoint_loads_both_transforms_with_user_id(test_client):
    """Pin: the load helper receives auth.user_id for BOTH
    transforms. Without this scoping, a malicious request for
    ``/api/v1/graph/some-other-users-tx/diff/my-tx`` would
    succeed and leak the diff of another user's transform.

    The tenant-scoping defense is the same as /decisions and
    /cost — at the loader layer, not the service layer."""
    base = GraphResponse(nodes=[], edges=[])
    compare = GraphResponse(nodes=[], edges=[])
    seen_calls: list = []

    async def fake_load(transform_id: str, user_id: str):
        seen_calls.append((transform_id, user_id))
        return base if transform_id == "tx-base" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-base/diff/tx-cmp")

    assert response.status_code == 200
    # Both loads scoped to the authenticated user. Pin via
    # exact sequence — if a future refactor accidentally loads
    # one transform without user_id, this fires.
    assert seen_calls == [
        ("tx-base", "test-user-1"),
        ("tx-cmp", "test-user-1"),
    ]


def test_diff_endpoint_surfaces_property_changes_with_base_compare_shape(
    test_client,
):
    """Wire-shape pin: a ``changed_node``'s ``property_changes``
    entry serializes as ``{base, compare}`` — not ``{from, to}``
    or ``{old, new}``. The rendering layer (UI / agent / CLI)
    keys on these names; a future refactor that renames them
    would silently break consumers."""
    alice_base = _node("a1", canonical_id="cid-alice", role="engineer")
    alice_compare = _node("a2", canonical_id="cid-alice", role="principal engineer")
    base_graph = GraphResponse(nodes=[alice_base], edges=[])
    compare_graph = GraphResponse(nodes=[alice_compare], edges=[])

    async def fake_load(transform_id: str, user_id: str):
        return base_graph if transform_id == "tx-1" else compare_graph

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-1/diff/tx-2")

    body = response.json()
    assert body["summary"]["nodes"]["changed"] == 1
    [changed] = body["changed_nodes"]
    assert changed["canonical_id"] == "cid-alice"
    assert changed["base_id"] == "a1"
    assert changed["compare_id"] == "a2"
    # The contract: each property delta is {"base": ..., "compare": ...}.
    role_change = changed["property_changes"]["role"]
    assert role_change == {"base": "engineer", "compare": "principal engineer"}


def test_diff_endpoint_returns_500_when_loading_fails(test_client):
    """A load failure (network blip, DB down, transform_id
    doesn't exist) bubbles to 500 — not a half-computed diff
    showing one side as 'fully removed' because the other side
    was empty due to the failure.

    This is the right default: the alternative (silently
    treating a load failure as 'empty graph') would let a
    flaky storage layer mis-report 'every node removed' on
    every diff."""

    async def fake_load(transform_id: str, user_id: str):
        raise RuntimeError("staging DB unreachable")

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-1/diff/tx-2")

    assert response.status_code == 500


def test_diff_endpoint_413s_when_base_truncated(test_client):
    """Reviewer-flagged P2 on commit a75cd73. The diff loader
    caps reads at 10k nodes, but ``GraphResponse.total_nodes``
    can exceed the returned list — pre-fix the endpoint returned
    a confident-looking but silently incomplete diff for larger
    transforms.

    Pin: when the base graph reports more total_nodes than were
    returned, the endpoint fails loud with 413 carrying the
    structured detail body so the agent / CLI can render a
    meaningful "this transform is too big for diff" message
    rather than misinterpret a partial diff."""
    # Returned 5 nodes but the count query said there are 15000.
    base = GraphResponse(
        nodes=[_node(f"b-{i}") for i in range(5)],
        edges=[],
        total_nodes=15000,
        total_edges=0,
    )
    compare = GraphResponse(
        nodes=[_node(f"c-{i}") for i in range(5)],
        edges=[],
        total_nodes=5,
        total_edges=0,
    )

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-big" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-big/diff/tx-small")

    assert response.status_code == 413
    body = response.json()["detail"]
    assert body["error"] == "transform_too_large_to_diff"
    assert body["side"] == "base"
    assert body["transform_id"] == "tx-big"
    assert body["returned_nodes"] == 5
    assert body["total_nodes"] == 15000


def test_diff_endpoint_413s_when_compare_truncated(test_client):
    """Mirror pin: truncation on the compare side trips the same
    413 with side='compare'. Both sides must pass the check for
    the diff to compute."""
    base = GraphResponse(
        nodes=[_node("b-1")],
        edges=[],
        total_nodes=1,
        total_edges=0,
    )
    compare = GraphResponse(
        nodes=[_node("c-1")],
        edges=[],
        total_nodes=20000,  # truncated
        total_edges=0,
    )

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-small" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-small/diff/tx-big")

    assert response.status_code == 413
    body = response.json()["detail"]
    assert body["side"] == "compare"
    assert body["transform_id"] == "tx-big"


def test_diff_endpoint_succeeds_when_counts_match(test_client):
    """Sanity check: when total_nodes == len(nodes) on both sides
    (the common case), the 413 doesn't fire. Pin to guard against
    the truncation check accidentally rejecting fully-loaded
    graphs."""
    base = GraphResponse(
        nodes=[_node("b-1"), _node("b-2")],
        edges=[],
        total_nodes=2,
        total_edges=0,
    )
    compare = GraphResponse(
        nodes=[_node("c-1"), _node("c-2")],
        edges=[],
        total_nodes=2,
        total_edges=0,
    )

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-1" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-1/diff/tx-2")

    assert response.status_code == 200


def test_diff_endpoint_handles_edge_property_change(test_client):
    """End-to-end edge-level diff: same endpoint identities,
    different edge properties. The endpoint surfaces it as a
    changed_edge with the {source_key, target_key, type,
    property_changes} shape."""
    alice_b = _node("a1", canonical_id="cid-alice")
    acme_b = _node("b1", canonical_id="cid-acme")
    alice_c = _node("a2", canonical_id="cid-alice")
    acme_c = _node("b2", canonical_id="cid-acme")

    base_edge = Edge(
        id="e-base",
        source="a1",
        target="b1",
        type="WORKS_AT",
        properties={"role": "engineer"},
    )
    compare_edge = Edge(
        id="e-compare",
        source="a2",
        target="b2",
        type="WORKS_AT",
        properties={"role": "principal engineer"},
    )

    base = GraphResponse(nodes=[alice_b, acme_b], edges=[base_edge])
    compare = GraphResponse(nodes=[alice_c, acme_c], edges=[compare_edge])

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-1" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-1/diff/tx-2")

    body = response.json()
    assert body["summary"]["edges"]["changed"] == 1
    [delta] = body["changed_edges"]
    # Endpoint identity composed of canonical_ids — survives the
    # storage-layer id churn.
    assert delta["source_key"] == "cid-alice"
    assert delta["target_key"] == "cid-acme"
    assert delta["type"] == "WORKS_AT"
    assert delta["property_changes"]["role"] == {
        "base": "engineer",
        "compare": "principal engineer",
    }
