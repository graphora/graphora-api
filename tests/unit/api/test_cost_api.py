"""Unit tests for the /api/v1/graph/{transform_id}/cost endpoint.

B5-obs: per-transform LLM cost / token aggregation surface. Mirror
the architectural pattern of the /decisions endpoint (commit
65fceac) — tenant-scoped via auth.user_id, owns the DB read so MCP
stays a pure HTTP client.

These tests pin:
  * Service is invoked with both transform_id AND auth.user_id —
    no cross-tenant leak.
  * Empty transform returns zero-aggregate, not 404.
  * Estimated cost serializes as a string (Decimal precision) or
    None (model not priced).
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


def test_cost_endpoint_passes_auth_user_id_to_service(test_client):
    """Reviewer-flagged equivalent of the /decisions tenant pin
    (commit 65fceac, P1): the cost endpoint must pass auth.user_id
    to the aggregator so authenticated user A can't fetch user B's
    transform cost just by knowing the transform_id.

    The aggregator's WHERE clause includes ``user_id = %s`` — pinned
    at the service level in test_usage_tracking — but the endpoint
    is the layer that supplies it from the auth context."""
    with patch("graphora_server.api.graph.UsageTrackingService") as mock_service_class:
        mock_service = AsyncMock()
        mock_service.get_transform_cost_report = AsyncMock(
            return_value={
                "transform_id": "tx-1",
                "total_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": None,
                "models_used": [],
                "by_operation_type": {},
            }
        )
        mock_service_class.return_value = mock_service

        response = test_client.get("/api/v1/graph/tx-1/cost")

    assert response.status_code == 200
    mock_service.get_transform_cost_report.assert_awaited_once_with(
        transform_id="tx-1",
        user_id="test-user-1",
    )


def test_cost_endpoint_returns_zero_aggregate_for_unknown_transform(test_client):
    """A transform with no llm_usage rows must return the
    zero-aggregate shape, NOT 404. The endpoint is informational —
    callers can render "0 calls, no cost recorded" without a
    conditional. Distinguishing "transform doesn't exist" from
    "transform exists but had no LLM calls" isn't this endpoint's
    job; the /graph endpoint is where existence is authoritative."""
    with patch("graphora_server.api.graph.UsageTrackingService") as mock_service_class:
        mock_service = AsyncMock()
        mock_service.get_transform_cost_report = AsyncMock(
            return_value={
                "transform_id": "tx-empty",
                "total_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": None,
                "models_used": [],
                "by_operation_type": {},
            }
        )
        mock_service_class.return_value = mock_service

        response = test_client.get("/api/v1/graph/tx-empty/cost")

    assert response.status_code == 200
    body = response.json()
    assert body["total_calls"] == 0
    assert body["estimated_cost_usd"] is None
    # None ≠ "0" — distinguishes "no LLM was invoked" from "LLM
    # ran with an unpriced model" (where the row would have None
    # estimated_cost_usd but total_calls > 0).
