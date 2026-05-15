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
    """Reviewer-flagged P2 on commit a75cd73 — pin retained
    after the commit-fae7e91 fix that added the upfront cap
    check. Same scenario (large base transform), just hits the
    cap-check trigger now instead of the len-vs-total trigger.

    Pin: total_nodes far above the cap → 413, side=base."""
    # Returned 5 nodes but the count query said there are
    # 15000 — exceeds the diff cap. Hits trigger 1 (cap).
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
    assert body["truncated_dimension"] == "nodes"
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


def test_diff_endpoint_413s_when_edges_truncated_even_with_full_nodes(
    test_client,
):
    """Reviewer-flagged P2 on commit a261321. The pre-fix
    truncation guard only inspected total_nodes. The Neo4j loader
    paginates by main node; edges between paginated-out nodes
    are silently omitted from the fetched edges. A transform
    that happens to fit within the 10k node cap (so
    total_nodes == len(nodes)) could still have its edges
    truncated and the diff would silently return a partial
    answer.

    Pin: total_edges > len(edges) trips 413 with
    ``truncated_dimension: "edges"`` so the operator knows
    which side and which dimension fell over."""
    base = GraphResponse(
        nodes=[_node("b-1"), _node("b-2")],
        edges=[
            # Only 1 edge returned but the count says 500.
            Edge(id="e-1", source="b-1", target="b-2", type="WORKS_AT", properties={}),
        ],
        total_nodes=2,
        total_edges=500,  # truncated
    )
    compare = GraphResponse(
        nodes=[_node("c-1")],
        edges=[],
        total_nodes=1,
        total_edges=0,
    )

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-many-edges" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-many-edges/diff/tx-small")

    assert response.status_code == 413
    body = response.json()["detail"]
    assert body["error"] == "transform_too_large_to_diff"
    assert body["side"] == "base"
    assert body["truncated_dimension"] == "edges"
    assert body["returned_edges"] == 1
    assert body["total_edges"] == 500


def test_diff_endpoint_413s_when_total_nodes_exceeds_cap_even_if_list_full(
    test_client,
):
    """Reviewer-flagged P2 on commit fae7e91. The staging reader's
    query is::

        MATCH (n) WHERE n.__tid = $tid
        WITH n ORDER BY n.id SKIP $skip LIMIT $limit
        OPTIONAL MATCH (n)-[r]-(m) WHERE r.__tid = $tid
        RETURN nodes, relationships, connected_nodes

    The response merges paged main nodes with connected nodes
    pulled via edge endpoints. So ``len(graph.nodes)`` can
    REACH OR EXCEED ``total_nodes`` (the main-node count) even
    when the main-node page was capped — connected nodes pad
    the list back up. The earlier ``len(nodes) < total_nodes``
    check was a false-negative for this case.

    Pin: when ``total_nodes`` itself exceeds the diff loader's
    cap, the endpoint MUST 413 regardless of how ``len(nodes)``
    looks. The exact scenario the reviewer described."""
    # 10001 main nodes — exceeds the 10000 cap. The staging
    # reader would page out 1 main node BUT then connected_nodes
    # could pad len(nodes) up to 10001 or higher. Pre-fix:
    # len(nodes) (10001) < total_nodes (10001) is False →
    # truncation check passes → silent partial diff. Post-fix:
    # total_nodes (10001) > _DIFF_NODE_LIMIT (10000) → 413.
    base = GraphResponse(
        # Simulate connected-node padding: 10001 nodes returned
        # via the mocked loader even though total_nodes is 10001.
        nodes=[_node(f"b-{i}") for i in range(10001)],
        edges=[],
        total_nodes=10001,
        total_edges=0,
    )
    compare = GraphResponse(
        nodes=[_node("c-1")],
        edges=[],
        total_nodes=1,
        total_edges=0,
    )

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-too-big" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-too-big/diff/tx-small")

    assert response.status_code == 413, (
        "Pre-fix: connected_nodes padding masked the main-node "
        "truncation. Got "
        f"{response.status_code} instead of 413. The diff would "
        "have silently returned a partial answer."
    )
    body = response.json()["detail"]
    assert body["side"] == "base"
    assert body["truncated_dimension"] == "nodes"
    assert body["total_nodes"] == 10001
    # The new detail field that exposes the cap value so
    # operators can correlate "the transform is bigger than X".
    assert body["diff_node_cap"] == 10000


def test_diff_endpoint_413s_when_compare_total_nodes_exceeds_cap(
    test_client,
):
    """Mirror pin for the compare side. Same scenario: a
    transform whose total_nodes exceeds the cap can pad its
    response.nodes via connected nodes; the upfront cap check
    must fire on either side."""
    base = GraphResponse(
        nodes=[_node("b-1")],
        edges=[],
        total_nodes=1,
        total_edges=0,
    )
    compare = GraphResponse(
        nodes=[_node(f"c-{i}") for i in range(50000)],  # padded list
        edges=[],
        total_nodes=20000,  # but the count says 20000 main nodes
        total_edges=0,
    )

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-small" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-small/diff/tx-huge")

    assert response.status_code == 413
    body = response.json()["detail"]
    assert body["side"] == "compare"
    assert body["truncated_dimension"] == "nodes"


def test_diff_endpoint_413s_when_compare_edges_truncated(test_client):
    """Mirror pin for the compare side."""
    base = GraphResponse(
        nodes=[_node("b-1")],
        edges=[],
        total_nodes=1,
        total_edges=0,
    )
    compare = GraphResponse(
        nodes=[_node("c-1"), _node("c-2")],
        edges=[],
        total_nodes=2,
        total_edges=99,  # truncated
    )

    async def fake_load(transform_id: str, user_id: str):
        return base if transform_id == "tx-small" else compare

    with patch(
        "graphora_server.api.graph._load_graph_for_diff",
        new=AsyncMock(side_effect=fake_load),
    ):
        response = test_client.get("/api/v1/graph/tx-small/diff/tx-many-edges")

    assert response.status_code == 413
    body = response.json()["detail"]
    assert body["side"] == "compare"
    assert body["truncated_dimension"] == "edges"


def test_diff_endpoint_handles_edge_property_change(test_client):
    """End-to-end edge-level diff: same endpoint identities,
    different edge properties. The endpoint surfaces it as a
    changed_edge with the {source_key, target_key, type,
    property_changes} shape."""
    # Type-prefixed canonical_id keys (commit 48dbe0a Medium fix)
    # mean the endpoint identity carries the type up to the wire.
    alice_b = _node("a1", type="Person", canonical_id="cid-alice")
    acme_b = _node("b1", type="Organization", canonical_id="cid-acme")
    alice_c = _node("a2", type="Person", canonical_id="cid-alice")
    acme_c = _node("b2", type="Organization", canonical_id="cid-acme")

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
    # storage-layer id churn. Type-prefixed since 48dbe0a so a
    # stale cid shared across types can't collapse a Person and
    # an Organization into one edge endpoint.
    assert delta["source_key"] == "Person:cid-alice"
    assert delta["target_key"] == "Organization:cid-acme"
    assert delta["type"] == "WORKS_AT"
    assert delta["property_changes"]["role"] == {
        "base": "engineer",
        "compare": "principal engineer",
    }
