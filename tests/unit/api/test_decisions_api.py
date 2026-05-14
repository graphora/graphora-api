"""Unit tests for the /api/v1/graph/{transform_id}/decisions endpoint.

Reviewer-flagged on commit 9ac9bb5 (B0-explain): the MCP server is
documented + implemented as a pure HTTP client and must not touch
the DB directly. The /decisions endpoint owns the read so MCP stays
decoupled. These tests pin the endpoint contract:

  * Schema-level decisions (decision_type=schema_inferred) are
    fetched via the indexed for_decision_type method, not the
    full-transform-then-Python-filter pattern that scaled with the
    total decision count.
  * Per-node decisions return alongside schema decisions, schema
    first (causation chain).
  * Alternatives are aggregated only from node-level decisions
    (schema decisions don't contribute candidates).
  * Empty-state response shape is stable: decision_log + alternatives
    keys present even when there's nothing to show.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from graphora_server.main import app
from graphora_server.auth import AuthContext, get_current_auth


@pytest.fixture
def test_client():
    """Test client with auth bypass."""

    def fake_auth():
        return AuthContext(user_id="test-user-1", token="test-token", claims={})

    app.dependency_overrides[get_current_auth] = fake_auth
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_auth, None)


@pytest.fixture(autouse=True)
def _force_memory_mode(monkeypatch):
    """Force the DecisionLogService memory backend so the endpoint
    doesn't try to talk to a real Postgres. Same pattern as the
    other Decision Log test fixtures (commit 82eaaba)."""
    from graphora_server.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)


def test_decisions_endpoint_uses_for_decision_type_for_schema_query(test_client):
    """Reviewer-flagged P3 (commit 9ac9bb5): the previous
    implementation called for_transform() and filtered for
    schema-level decisions in Python — fetching every decision in
    the transform just to surface 1-2 schema rows.

    Pin: the endpoint must call ``for_decision_type(transform_id,
    DecisionType.SCHEMA_INFERRED)`` so the (transform_id,
    decision_type) index from migration 14 covers the read."""
    from graphora_server.services.decision_log_service import (
        Decision,
        DecisionType,
        TargetKind,
    )

    schema_decision = Decision(
        id="d-schema",
        transform_id="tx1",
        target_id=None,
        target_kind=TargetKind.SCHEMA,
        decision_type=DecisionType.SCHEMA_INFERRED,
        reason="auto-inferred from chunks",
        evidence={"ontology_id": "auto_abc", "entities_count": 3},
        alternatives=[],
        created_at="2026-05-09T10:00:00+00:00",
    )

    with patch("graphora_server.api.graph.DecisionLogService") as mock_log_class:
        mock_log = AsyncMock()
        mock_log.for_decision_type = AsyncMock(return_value=[schema_decision])
        mock_log.for_target = AsyncMock(return_value=[])
        mock_log_class.return_value = mock_log

        response = test_client.get("/api/v1/graph/tx1/decisions")

    assert response.status_code == 200
    # Pin the indexed-read contract: the endpoint must call
    # for_decision_type with SCHEMA_INFERRED, NOT for_transform.
    # The user_id kwarg lands as part of the P1 follow-up
    # (commit eb22a79); pinned separately in
    # test_decisions_endpoint_passes_auth_user_id_to_service_filters.
    mock_log.for_decision_type.assert_awaited_once_with(
        "tx1", DecisionType.SCHEMA_INFERRED, user_id="test-user-1"
    )
    # for_transform must NOT be invoked — the bug-fix collapses
    # the read to the indexed type query only.
    assert (
        not hasattr(mock_log, "for_transform") or not mock_log.for_transform.await_count
    )


def test_decisions_endpoint_returns_empty_arrays_when_no_node_id(test_client):
    """Without ``node_id``, the endpoint returns schema-level
    decisions only — no node lookup happens. ``alternatives`` is
    empty because schema decisions don't contribute candidates."""
    with patch("graphora_server.api.graph.DecisionLogService") as mock_log_class:
        mock_log = AsyncMock()
        mock_log.for_decision_type = AsyncMock(return_value=[])
        mock_log.for_target = AsyncMock(return_value=[])
        mock_log_class.return_value = mock_log

        response = test_client.get("/api/v1/graph/tx1/decisions")

    assert response.status_code == 200
    body = response.json()
    assert body == {"decision_log": [], "alternatives": []}
    # for_target must not be called when node_id isn't provided —
    # callers asking for transform-level context shouldn't pay an
    # extra DB roundtrip for a node lookup they didn't request.
    mock_log.for_target.assert_not_awaited()


def test_decisions_endpoint_orders_schema_first_then_node_decisions(test_client):
    """Narrative-ordering pin (matches the pre-fix MCP-side test).
    Schema-level decisions render before node-level ones so the
    Evidence tab shows a top-down causation chain ("schema chosen
    → these per-entity merges followed"). Walltime ordering would
    mis-narrate causation in re-extractions where node merges
    happen to land before a schema decision in walltime."""
    from graphora_server.services.decision_log_service import (
        Decision,
        DecisionType,
        TargetKind,
    )

    schema_decision = Decision(
        id="d-schema",
        transform_id="tx1",
        target_id=None,
        target_kind=TargetKind.SCHEMA,
        decision_type=DecisionType.SCHEMA_INFERRED,
        reason="schema",
        evidence={},
        alternatives=[],
        # Schema decision walltime is AFTER the node decision
        # below — proves ordering is intentional, not an artifact
        # of timestamp sort.
        created_at="2026-05-09T11:00:00+00:00",
    )
    node_decision = Decision(
        id="d-node",
        transform_id="tx1",
        target_id="n1",
        target_kind=TargetKind.NODE,
        decision_type=DecisionType.ENTITY_MERGED,
        reason="merge",
        evidence={},
        alternatives=[{"id": "n1-alias"}],
        created_at="2026-05-09T10:00:00+00:00",
    )

    with patch("graphora_server.api.graph.DecisionLogService") as mock_log_class:
        mock_log = AsyncMock()
        mock_log.for_decision_type = AsyncMock(return_value=[schema_decision])
        mock_log.for_target = AsyncMock(return_value=[node_decision])
        mock_log_class.return_value = mock_log

        response = test_client.get("/api/v1/graph/tx1/decisions?node_id=n1")

    assert response.status_code == 200
    body = response.json()
    kinds = [d["target_kind"] for d in body["decision_log"]]
    assert kinds == ["schema", "node"], (
        f"Expected schema-first causation chain ordering; got {kinds}. "
        f"Walltime sorting would mis-narrate causation for "
        f"re-extractions."
    )

    # Per-node alternatives aggregated at the top level. Schema
    # decision contributed none; node decision contributed one.
    assert body["alternatives"] == [{"id": "n1-alias"}]


def test_decisions_endpoint_passes_auth_user_id_to_service_filters(test_client):
    """Reviewer-flagged P1 (commit eb22a79): the endpoint went
    straight to the decision table which was keyed only by
    transform_id/target_id, so any authenticated user could fetch
    any other tenant's decision log just by knowing the transform
    ID. Migration 15 added a user_id column; the endpoint must now
    pass auth.user_id to every read so the WHERE clause filters
    on it.

    Pin: every service call out of the endpoint receives
    user_id=auth.user_id. If a future refactor drops the kwarg
    on any read path, this fires."""
    from graphora_server.services.decision_log_service import DecisionType

    with patch("graphora_server.api.graph.DecisionLogService") as mock_log_class:
        mock_log = AsyncMock()
        mock_log.for_decision_type = AsyncMock(return_value=[])
        mock_log.for_target = AsyncMock(return_value=[])
        mock_log_class.return_value = mock_log

        response = test_client.get("/api/v1/graph/tx1/decisions?node_id=n1")

    assert response.status_code == 200

    # Both read methods invoked with user_id from the auth context.
    mock_log.for_decision_type.assert_awaited_once_with(
        "tx1", DecisionType.SCHEMA_INFERRED, user_id="test-user-1"
    )
    mock_log.for_target.assert_awaited_once_with("tx1", "n1", user_id="test-user-1")


def test_decisions_endpoint_does_not_leak_other_tenants_decisions(test_client):
    """End-to-end pin for the cross-tenant fix. Two users seeded
    decisions for the same transform_id (a synthetic edge case
    that can occur if two users somehow share a transform id, OR
    if a malicious user crafts a request with another user's
    transform_id). Auth as user-1 must only see user-1's row.

    Memory backend equivalence pin — the same filter must apply
    to both backends so dev-mode behaves like production."""
    from graphora_server.services.decision_log_service import (
        Decision,
        DecisionLogService,
        DecisionType,
        TargetKind,
    )

    # Seed a real DecisionLogService memory store with rows from
    # both users. This exercises the actual filtering, not just
    # mock contracts.
    store: list = []
    log = DecisionLogService(memory_store=store)

    user_a_decision = Decision(
        transform_id="tx1",
        target_id=None,
        target_kind=TargetKind.SCHEMA,
        decision_type=DecisionType.SCHEMA_INFERRED,
        reason="user-a's schema",
        evidence={"ontology_id": "auto_a"},
        user_id="test-user-1",
    )
    user_b_decision = Decision(
        transform_id="tx1",  # SAME transform id, different user
        target_id=None,
        target_kind=TargetKind.SCHEMA,
        decision_type=DecisionType.SCHEMA_INFERRED,
        reason="user-b's schema — must not leak to user-1",
        evidence={"ontology_id": "auto_b"},
        user_id="test-user-2",
    )

    import asyncio

    asyncio.get_event_loop().run_until_complete(log.append(user_a_decision))
    asyncio.get_event_loop().run_until_complete(log.append(user_b_decision))

    # Patch DecisionLogService class so the endpoint's
    # `DecisionLogService()` call returns OUR seeded instance.
    with patch(
        "graphora_server.api.graph.DecisionLogService",
        return_value=log,
    ):
        response = test_client.get("/api/v1/graph/tx1/decisions")

    assert response.status_code == 200
    body = response.json()

    # The auth context says user-1; only user-1's decision is
    # returned. user-2's row stays invisible.
    assert len(body["decision_log"]) == 1
    only = body["decision_log"][0]
    assert only["reason"] == "user-a's schema", (
        f"Cross-tenant leak: got reason {only['reason']!r} "
        f"(expected user-a's row only). Pre-fix this would have "
        f"returned BOTH rows because the filter ignored user_id."
    )


def test_decisions_endpoint_serializes_enums_to_strings(test_client):
    """Enums must serialize to their string values for JSON transport
    (``target_kind: "node"``, not ``target_kind: TargetKind.NODE``).
    Pin so a future refactor that returns the dataclass directly
    without conversion is caught by the response shape assertion."""
    from graphora_server.services.decision_log_service import (
        Decision,
        DecisionType,
        TargetKind,
    )

    node_decision = Decision(
        id="d1",
        transform_id="tx1",
        target_id="n1",
        target_kind=TargetKind.NODE,
        decision_type=DecisionType.ENTITY_MERGED,
        reason="merge",
        evidence={"stage": "property_blocker"},
        alternatives=[],
        created_at="2026-05-09T10:00:00+00:00",
    )

    with patch("graphora_server.api.graph.DecisionLogService") as mock_log_class:
        mock_log = AsyncMock()
        mock_log.for_decision_type = AsyncMock(return_value=[])
        mock_log.for_target = AsyncMock(return_value=[node_decision])
        mock_log_class.return_value = mock_log

        response = test_client.get("/api/v1/graph/tx1/decisions?node_id=n1")

    assert response.status_code == 200
    [d] = response.json()["decision_log"]
    assert d["target_kind"] == "node"
    assert d["decision_type"] == "entity_merged"


# ============================================================
# Gate-4-wrap edge-evidence backend (reviewer flagged on slice C
# wrap-up): /decisions must accept edge_id so ``graphora explain
# <edge>`` can return source text + decision log. The
# DecisionLogService is already generic over target_id; the gap
# was the query-path. These tests pin the edge_id surface and the
# node_id/edge_id mutex.
# ============================================================


def test_decisions_endpoint_accepts_edge_id(test_client):
    """Edge-targeted decisions return alongside the schema prefix.
    Same response shape as the node case — only the
    DecisionType.RELATIONSHIP_ACCEPTED / RELATIONSHIP_REJECTED
    target_kind=edge rows distinguish the result."""
    from graphora_server.services.decision_log_service import (
        Decision,
        DecisionType,
        TargetKind,
    )

    edge_decision = Decision(
        id="d-edge",
        transform_id="tx1",
        target_id="e1",
        target_kind=TargetKind.EDGE,
        decision_type=DecisionType.RELATIONSHIP_ACCEPTED,
        reason="validator confirmed",
        evidence={"stage": "validator"},
        alternatives=[],
        created_at="2026-05-14T00:00:00+00:00",
    )

    with patch("graphora_server.api.graph.DecisionLogService") as mock_log_class:
        mock_log = AsyncMock()
        mock_log.for_decision_type = AsyncMock(return_value=[])
        mock_log.for_target = AsyncMock(return_value=[edge_decision])
        mock_log_class.return_value = mock_log

        response = test_client.get("/api/v1/graph/tx1/decisions?edge_id=e1")

    assert response.status_code == 200
    body = response.json()
    [d] = body["decision_log"]
    assert d["target_kind"] == "edge"
    assert d["target_id"] == "e1"
    assert d["decision_type"] == "relationship_accepted"

    # Pin: for_target was called with the edge_id, not a node_id.
    mock_log.for_target.assert_awaited_once()
    call_args = mock_log.for_target.await_args
    assert call_args.args[1] == "e1", (
        "for_target should have been called with edge_id='e1'. "
        "If the endpoint accidentally swapped to node_id we'd "
        "fetch the wrong target's decisions."
    )


def test_decisions_endpoint_aggregates_edge_alternatives(test_client):
    """Like the node path, edge decisions contribute their
    alternatives to the aggregate. Pin so a refactor that
    forgets to thread the alternatives list (e.g., by introducing
    a separate edge code path that doesn't aggregate) fails
    loud."""
    from graphora_server.services.decision_log_service import (
        Decision,
        DecisionType,
        TargetKind,
    )

    edge_decision = Decision(
        id="d-edge",
        transform_id="tx1",
        target_id="e1",
        target_kind=TargetKind.EDGE,
        decision_type=DecisionType.RELATIONSHIP_REJECTED,
        reason="confidence below threshold",
        evidence={},
        alternatives=[
            {"reason": "below validator score 0.6"},
            {"reason": "no source span match"},
        ],
        created_at="2026-05-14T00:00:00+00:00",
    )

    with patch("graphora_server.api.graph.DecisionLogService") as mock_log_class:
        mock_log = AsyncMock()
        mock_log.for_decision_type = AsyncMock(return_value=[])
        mock_log.for_target = AsyncMock(return_value=[edge_decision])
        mock_log_class.return_value = mock_log

        response = test_client.get("/api/v1/graph/tx1/decisions?edge_id=e1")

    assert response.status_code == 200
    body = response.json()
    assert len(body["alternatives"]) == 2


def test_decisions_endpoint_rejects_both_node_id_and_edge_id(test_client):
    """Mutex pin: setting both ids in one request is ambiguous —
    a Decision's target_id belongs to one kind only, and mixing
    them would either over-fetch (do both queries) or surface
    ambiguous results. Returning 400 at the API surface is the
    right shape; the alternative (server silently picks one) is
    a footgun for clients."""
    response = test_client.get(
        "/api/v1/graph/tx1/decisions?node_id=n1&edge_id=e1",
    )
    assert response.status_code == 400
    assert "mutually exclusive" in response.json()["detail"]


def test_decisions_endpoint_returns_only_schema_when_neither_id(test_client):
    """No node_id and no edge_id is a legitimate request — return
    the schema prefix only. Pin so a future tightening that
    rejects "neither" surfaces the design choice intentionally."""
    from graphora_server.services.decision_log_service import (
        Decision,
        DecisionType,
        TargetKind,
    )

    schema_decision = Decision(
        id="d-schema",
        transform_id="tx1",
        target_id=None,
        target_kind=TargetKind.SCHEMA,
        decision_type=DecisionType.SCHEMA_INFERRED,
        reason="inferred from chunk 0",
        evidence={},
        alternatives=[],
        created_at="2026-05-14T00:00:00+00:00",
    )

    with patch("graphora_server.api.graph.DecisionLogService") as mock_log_class:
        mock_log = AsyncMock()
        mock_log.for_decision_type = AsyncMock(return_value=[schema_decision])
        mock_log.for_target = AsyncMock()
        mock_log_class.return_value = mock_log

        response = test_client.get("/api/v1/graph/tx1/decisions")

    assert response.status_code == 200
    body = response.json()
    assert len(body["decision_log"]) == 1
    assert body["decision_log"][0]["target_kind"] == "schema"
    # for_target was NOT called when neither id is set — the
    # schema prefix is all the caller asked for.
    mock_log.for_target.assert_not_awaited()
