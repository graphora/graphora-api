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
    mock_log.for_decision_type.assert_awaited_once_with(
        "tx1", DecisionType.SCHEMA_INFERRED
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
