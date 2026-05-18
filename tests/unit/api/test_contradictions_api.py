"""Unit tests for the /api/v1/graph/{transform_id}/contradictions endpoint.

B1-prob slice 2a. The endpoint owns the read so MCP stays a
pure HTTP client (same architectural pattern /decisions and
/diff use). Tests pin:

  * Empty-state response shape (the live state until slice 2b
    hooks emit claims).
  * Wire shape when claims exist: contradictions array,
    per-entry competing_claims sorted by confidence DESC,
    severity = distinct-value count.
  * Tenant scoping — auth.user_id reaches the service.
  * Query param validation (min_confidence in [0.0, 1.0]).
  * Service-layer integration via the in-memory backend so the
    endpoint exercises real claim grouping.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.config import settings
from graphora_server.main import app
from graphora_server.services.claims_service import (
    Claim,
    ClaimsService,
    TargetKind,
    _reset_default_memory_store_for_tests,
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


@pytest.fixture(autouse=True)
def _force_memory_mode(monkeypatch):
    """Force ClaimsService into memory mode + clear the shared
    dev-mode store between tests. The endpoint constructs its
    own service instance via ``ClaimsService()`` — without the
    shared store the test setup wouldn't persist across the
    request boundary. See commit 088692b for the shared-store
    rationale."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    _reset_default_memory_store_for_tests()
    yield
    _reset_default_memory_store_for_tests()


def _seed_claim(
    *,
    transform_id: str = "tx-1",
    target_id: str = "node-alice",
    target_kind: TargetKind = TargetKind.NODE,
    property_key: str = "title",
    value: Any = "Engineer",
    confidence: float = 0.9,
    user_id: str = "user-1",
) -> Claim:
    """Direct service-layer append: the endpoint reads from
    the same shared dev-mode store the service writes to, so
    pre-loading the store is the fastest way to drive the
    endpoint into a specific state."""
    return Claim(
        transform_id=transform_id,
        target_id=target_id,
        target_kind=target_kind,
        property_key=property_key,
        value=value,
        confidence=confidence,
        user_id=user_id,
        source_chunk_id="chunk-1",
        source_extractor_model="gemini-2.5-flash",
        source_prompt_version="v1.0",
    )


# ============================================================
# Empty-state surface
# ============================================================


def test_contradictions_empty_state_returns_stable_envelope(test_client):
    """Pin the empty-state response — the live state until slice
    2b's pipeline hooks emit claims. Callers (Explorer
    Contradictions tab, CLI runner) should be able to render
    "no contradictions" without any conditional access pattern;
    the envelope shape is stable whether the list is empty or
    not."""
    response = test_client.get("/api/v1/graph/tx-empty/contradictions")
    assert response.status_code == 200

    body = response.json()
    assert body["transform_id"] == "tx-empty"
    assert body["min_confidence"] == 0.0
    assert body["contradictions"] == []
    assert body["total_claims_scanned"] == 0


# ============================================================
# Wire shape when contradictions exist
# ============================================================


@pytest.mark.asyncio
async def test_contradictions_surface_competing_claims_sorted_desc(test_client):
    """The endpoint must return claims sorted by confidence DESC
    within each competing_claims list — the rendering layer
    relies on the "winning value first" invariant.

    Pin via service-layer seed: two claims about the same
    (target, property) with different values + different
    confidences. Endpoint should group them, sort by
    confidence, and surface as one contradiction."""
    service = ClaimsService()
    await service.append(_seed_claim(value="Senior Engineer", confidence=0.95))
    await service.append(_seed_claim(value="Staff Engineer", confidence=0.7))

    response = test_client.get("/api/v1/graph/tx-1/contradictions")
    assert response.status_code == 200

    body = response.json()
    assert len(body["contradictions"]) == 1
    c = body["contradictions"][0]
    assert c["target_id"] == "node-alice"
    assert c["target_kind"] == "node"
    assert c["property_key"] == "title"
    assert c["severity"] == 2

    claims = c["competing_claims"]
    assert len(claims) == 2
    # Winning value first.
    assert claims[0]["value"] == "Senior Engineer"
    assert claims[1]["value"] == "Staff Engineer"
    # Confidence is preserved verbatim.
    assert claims[0]["confidence"] == 0.95
    assert claims[1]["confidence"] == 0.7
    # total_claims_scanned reflects what surfaced (claims
    # inside contradictions, not the global table count).
    assert body["total_claims_scanned"] == 2


@pytest.mark.asyncio
async def test_contradictions_min_confidence_filter_via_query_param(test_client):
    """``?min_confidence=0.5`` filters out low-confidence claims
    before the contradiction grouping. Pin so a refactor that
    silently drops the parameter regresses — CI scripts that
    gate on high-confidence contradictions rely on this filter."""
    service = ClaimsService()
    await service.append(_seed_claim(value="A", confidence=0.9))
    await service.append(_seed_claim(value="B", confidence=0.1))  # below floor

    # No filter: contradiction surfaces (2 distinct values).
    response = test_client.get("/api/v1/graph/tx-1/contradictions?min_confidence=0.0")
    assert response.status_code == 200
    assert len(response.json()["contradictions"]) == 1

    # Floor at 0.5: low-confidence claim drops out, only one
    # claim survives → no contradiction.
    response = test_client.get("/api/v1/graph/tx-1/contradictions?min_confidence=0.5")
    assert response.status_code == 200
    body = response.json()
    assert body["contradictions"] == []
    assert body["min_confidence"] == 0.5


# ============================================================
# Tenant scoping
# ============================================================


@pytest.mark.asyncio
async def test_contradictions_filter_by_authenticated_user(test_client):
    """The endpoint must thread auth.user_id into the service —
    a request for tx-1 must NOT surface another user's
    contradictions even if they share the transform_id. Pin so
    a refactor that drops the user_id arg silently leaks
    cross-tenant data, same risk class /decisions had
    pre-fix."""
    service = ClaimsService()
    # user-1's contradicting claims on tx-1.
    await service.append(_seed_claim(user_id="user-1", value="A", confidence=0.9))
    await service.append(_seed_claim(user_id="user-1", value="B", confidence=0.8))
    # user-2 has only one claim on tx-1 — no contradiction.
    await service.append(_seed_claim(user_id="user-2", value="X"))

    # Test client authenticates as user-1.
    response = test_client.get("/api/v1/graph/tx-1/contradictions")
    body = response.json()
    assert len(body["contradictions"]) == 1
    # All competing claims should be user-1's.
    for claim in body["contradictions"][0]["competing_claims"]:
        assert claim["user_id"] == "user-1"


# ============================================================
# Query param validation
# ============================================================


def test_contradictions_400_when_min_confidence_above_one(test_client):
    """``min_confidence`` is a probability — out-of-range values
    must reject at the endpoint boundary. Pin so a refactor
    that drops the explicit check regresses (the service-side
    validation is on Claim.confidence, not on the query
    parameter)."""
    response = test_client.get("/api/v1/graph/tx-1/contradictions?min_confidence=1.5")
    assert response.status_code == 400
    assert "min_confidence" in response.json()["detail"]


def test_contradictions_400_when_min_confidence_negative(test_client):
    """Symmetric pin for the lower bound."""
    response = test_client.get("/api/v1/graph/tx-1/contradictions?min_confidence=-0.1")
    assert response.status_code == 400


# ============================================================
# Service integration
# ============================================================


@pytest.mark.asyncio
async def test_contradictions_endpoint_threads_user_id_to_service(
    test_client,
):
    """The service call must receive auth.user_id explicitly.
    Mock the service's method to capture the kwargs. Pin so a
    refactor that drops the kwarg silently breaks tenant
    scoping (same severity class as test_apply_mutations
    passes_auth_user_id_to_service in test_scenarios_api.py)."""
    from graphora_server.services.claims_service import ClaimsService

    captured: dict = {}

    async def fake_contradictions(self, transform_id, user_id, min_confidence=0.0):
        # ``patch.object`` with new= binds as an unbound method,
        # so self lands as the first positional arg. Same shape
        # the real method has.
        captured["transform_id"] = transform_id
        captured["user_id"] = user_id
        captured["min_confidence"] = min_confidence
        return []

    with patch.object(
        ClaimsService,
        "contradictions_for_transform",
        new=fake_contradictions,
    ):
        response = test_client.get(
            "/api/v1/graph/tx-1/contradictions?min_confidence=0.5"
        )

    assert response.status_code == 200
    assert captured == {
        "transform_id": "tx-1",
        "user_id": "user-1",
        "min_confidence": 0.5,
    }
