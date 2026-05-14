"""Unit tests for the /api/v1/disputed-pairs endpoints.

Concerns pinned:
  * Tenant scoping at the endpoint boundary: every route passes
    auth.user_id into the service. A request with another user's
    pair_id returns 404 (the service returns None; the endpoint
    translates without leaking).
  * Wire shape: list endpoints return arrays of pair dicts; label
    endpoint accepts {decision, reason} body and returns the
    updated pair.
  * Closed-set decision enum: invalid decisions return 400 before
    the service is touched.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.main import app
from graphora_server.services.disputed_pairs_service import (
    DisputedPair,
    SourceStage,
    Status,
)


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


def _pair(
    pair_id: str = "p-1",
    user_id: str = "test-user-1",
    status: Status = Status.PENDING,
    similarity: float | None = 0.85,
) -> DisputedPair:
    return DisputedPair(
        id=pair_id,
        user_id=user_id,
        transform_id="tx-1",
        node_a_id="n-a",
        node_b_id="n-b",
        entity_type="Person",
        source_stage=SourceStage.EMBEDDING_BLOCKER,
        status=status,
        node_a_canonical_key="alice",
        node_b_canonical_key="alicia",
        similarity_score=(Decimal(str(similarity)) if similarity is not None else None),
        created_at="2026-05-14T00:00:00+00:00",
    )


# ---- GET /disputed-pairs ----------------------------------------------------


def test_list_pending_passes_auth_user_id_to_service(test_client):
    """Tenant-scoping pin: the endpoint must thread auth.user_id
    into list_pending. Without this, an authenticated user could
    accidentally list pairs that belong to another tenant. Same
    contract pattern as /budgets and /decisions."""
    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.list_pending = AsyncMock(return_value=[])
        mock_class.return_value = mock

        response = test_client.get("/api/v1/disputed-pairs")

    assert response.status_code == 200
    assert response.json() == []
    mock.list_pending.assert_awaited_once()
    kwargs = mock.list_pending.await_args.kwargs
    assert kwargs["user_id"] == "test-user-1"


def test_list_pending_supports_transform_id_filter(test_client):
    """The optional transform_id query param threads through to
    the service. Pin so a future refactor doesn't drop the
    filter (collapsing the per-transform review surface into
    the global queue)."""
    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.list_pending = AsyncMock(return_value=[])
        mock_class.return_value = mock

        test_client.get("/api/v1/disputed-pairs?transform_id=tx-abc")

    kwargs = mock.list_pending.await_args.kwargs
    assert kwargs["transform_id"] == "tx-abc"


def test_list_pending_returns_wire_shape_with_string_decimal(test_client):
    """similarity_score is a string on the wire (Decimal
    precision). Pin via a fixture pair with 0.85 — the response
    must echo as ``"0.85"`` not 0.85 (float)."""
    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.list_pending = AsyncMock(return_value=[_pair(similarity=0.85)])
        mock_class.return_value = mock

        response = test_client.get("/api/v1/disputed-pairs")

    [body] = response.json()
    assert body["similarity_score"] == "0.85"
    assert body["status"] == "pending"
    assert body["source_stage"] == "embedding_blocker"


# ---- GET /disputed-pairs/transform/{transform_id} ---------------------------


def test_list_for_transform_returns_all_statuses(test_client):
    """Per-transform view returns labeled + skipped rows too,
    not just pending. Pin the endpoint's contract — labeled rows
    are the audit trail for that run."""
    pairs = [
        _pair(pair_id="p-1", status=Status.PENDING),
        _pair(pair_id="p-2", status=Status.LABELED_MATCH),
        _pair(pair_id="p-3", status=Status.SKIPPED),
    ]
    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.list_for_transform = AsyncMock(return_value=pairs)
        mock_class.return_value = mock

        response = test_client.get("/api/v1/disputed-pairs/transform/tx-1")

    body = response.json()
    statuses = {p["status"] for p in body}
    assert statuses == {"pending", "labeled_match", "skipped"}


# ---- GET /disputed-pairs/{pair_id} ------------------------------------------


def test_get_pair_returns_404_when_missing_or_cross_tenant(test_client):
    """The service returns None for both 'doesn't exist' and
    'belongs to another tenant'. The endpoint maps both to 404
    so the existence-vs-ownership distinction stays internal."""
    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock_class.return_value = mock

        response = test_client.get("/api/v1/disputed-pairs/missing-id")

    assert response.status_code == 404


# ---- POST /disputed-pairs/{pair_id}/label -----------------------------------


def test_label_pair_happy_path(test_client):
    """Match decision flows through: service called with the
    typed Decision.MATCH; response carries the updated status."""
    from graphora_server.services.disputed_pairs_service import Decision

    labeled = _pair(status=Status.LABELED_MATCH)
    labeled.labeled_at = "2026-05-14T01:00:00+00:00"
    labeled.labeled_by_user_id = "test-user-1"

    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.label = AsyncMock(return_value=labeled)
        mock_class.return_value = mock

        response = test_client.post(
            "/api/v1/disputed-pairs/p-1/label",
            json={"decision": "match", "reason": "same person"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "labeled_match"
    assert body["labeled_by_user_id"] == "test-user-1"

    # Service called with the typed Decision.MATCH, not the raw
    # string. The endpoint's job is to validate + convert.
    call_kwargs = mock.label.await_args.kwargs
    assert call_kwargs["decision"] == Decision.MATCH
    assert call_kwargs["reason"] == "same person"
    assert call_kwargs["user_id"] == "test-user-1"


def test_label_pair_rejects_invalid_decision_with_422(test_client):
    """Closed-set enum at the wire layer, exposed via Pydantic
    so generated clients / OpenAPI docs see the closed set
    (match / not_match / skip) instead of a plain ``string``
    type. An unrecognized decision string is rejected BEFORE
    the service is touched — Pydantic returns 422 with a
    structured validation-error body. Pre-fix the manual
    check returned 400 with a free-form detail; switching to
    the typed field exposes the enum to schema consumers."""
    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.label = AsyncMock()
        mock_class.return_value = mock

        response = test_client.post(
            "/api/v1/disputed-pairs/p-1/label",
            json={"decision": "maybe"},  # not in {match, not_match, skip}
        )

    assert response.status_code == 422, (
        "Pydantic validation should reject invalid decision "
        "BEFORE the handler runs (returns 422). Pre-fix the "
        "manual check returned 400 but didn't surface the enum "
        "in the OpenAPI schema."
    )
    # Service was NOT touched.
    mock.label.assert_not_awaited()


def test_label_pair_returns_404_on_cross_tenant_attempt(test_client):
    """Cross-tenant label attempts return 404 — same as get.
    The service returns None when the (pair_id, user_id) tuple
    doesn't match; the endpoint maps to 404 without leaking the
    cross-tenant existence."""
    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.label = AsyncMock(return_value=None)
        mock_class.return_value = mock

        response = test_client.post(
            "/api/v1/disputed-pairs/p-1/label",
            json={"decision": "match"},
        )

    assert response.status_code == 404


def test_label_pair_accepts_skip_decision(test_client):
    """SKIP is a valid decision; not all reviews end in
    match/not_match. Pin so a future refactor that limits
    decisions to binary fails loud."""
    from graphora_server.services.disputed_pairs_service import Decision

    labeled = _pair(status=Status.SKIPPED)
    labeled.labeled_at = "2026-05-14T02:00:00+00:00"
    labeled.labeled_by_user_id = "test-user-1"

    with patch("graphora_server.api.disputed_pairs.DisputedPairsService") as mock_class:
        mock = AsyncMock()
        mock.label = AsyncMock(return_value=labeled)
        mock_class.return_value = mock

        response = test_client.post(
            "/api/v1/disputed-pairs/p-1/label",
            json={"decision": "skip"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert mock.label.await_args.kwargs["decision"] == Decision.SKIP
