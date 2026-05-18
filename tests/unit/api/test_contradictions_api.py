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


# ============================================================
# Reviewer-fix: total_claims_scanned semantic + response_model
# (Medium x2 on 66987b2)
# ============================================================


@pytest.mark.asyncio
async def test_total_claims_scanned_counts_consistent_claims_too(test_client):
    """Reviewer-flagged Medium on commit 66987b2. Pre-fix the
    endpoint summed claims INSIDE contradiction groups, so a
    transform with N consistent claims (no conflicts) would
    report ``total_claims_scanned: 0`` — indistinguishable
    from the "writer hasn't emitted any claims yet" state.

    Pin: when N claims exist for a transform but they all
    agree (one distinct value per property), the response's
    contradictions list is empty AND total_claims_scanned
    equals N. This lets CLI / dashboard callers tell the two
    states apart."""
    service = ClaimsService()
    # Three claims, all consistent (same value across all
    # three chunks) — should NOT surface as a contradiction.
    for chunk in ("a", "b", "c"):
        c = _seed_claim(value="Engineer", confidence=0.9)
        c.source_chunk_id = f"chunk-{chunk}"
        await service.append(c)

    response = test_client.get("/api/v1/graph/tx-1/contradictions")
    assert response.status_code == 200
    body = response.json()
    # No contradictions (consistent values).
    assert body["contradictions"] == []
    # But the count reflects the claims that DID exist + were
    # scanned. Pre-fix this would have been 0, masking the
    # consistent-data case as "no data."
    assert body["total_claims_scanned"] == 3, (
        "total_claims_scanned must count ALL claims above the "
        "confidence floor, not just the ones inside "
        "contradictions. Got "
        f"{body['total_claims_scanned']} alongside 3 stored claims."
    )


@pytest.mark.asyncio
async def test_total_claims_scanned_honors_min_confidence_filter(test_client):
    """The count must respect the same ``min_confidence`` floor
    the contradictions detector does — otherwise the count
    would overstate what's actually being considered. Pin so
    a refactor that splits the floor across the two methods
    regresses."""
    service = ClaimsService()
    await service.append(_seed_claim(value="A", confidence=0.95))
    await service.append(_seed_claim(value="B", confidence=0.1))

    # No filter: both claims contribute.
    response = test_client.get("/api/v1/graph/tx-1/contradictions?min_confidence=0.0")
    assert response.json()["total_claims_scanned"] == 2

    # Floor 0.5: low-confidence claim drops out of the count too.
    response = test_client.get("/api/v1/graph/tx-1/contradictions?min_confidence=0.5")
    assert response.json()["total_claims_scanned"] == 1


def test_contradictions_openapi_uses_typed_response_model():
    """Reviewer-flagged Medium on commit 66987b2. Pre-fix the
    route returned ``Dict[str, Any]`` so OpenAPI exposed a
    generic object as the 200 response — generated clients
    (e.g., openapi-generator) wouldn't see the Claim /
    Contradiction shape at all. The wire-shape pin in the
    OpenAPI snapshot was pinning a permissive shape rather
    than the real one.

    Pin: the endpoint's OpenAPI spec must reference the
    ContradictionsResponse schema (via $ref) on its 200
    response. A refactor that drops ``response_model`` would
    silently downgrade the wire contract back to ``object``."""
    from graphora_server.main import app

    spec = app.openapi()
    route_spec = (
        spec.get("paths", {})
        .get("/api/v1/graph/{transform_id}/contradictions", {})
        .get("get", {})
    )
    response_200 = (
        route_spec.get("responses", {})
        .get("200", {})
        .get("content", {})
        .get("application/json", {})
    )
    schema = response_200.get("schema", {})
    # FastAPI represents response_model as a $ref to a named
    # component schema; the ref string ends with the class name.
    ref = schema.get("$ref", "")
    assert ref.endswith("ContradictionsResponse"), (
        "Route must declare response_model=ContradictionsResponse "
        "so OpenAPI exposes the typed Claim/Contradiction "
        f"contract. Got schema={schema!r}"
    )
    # And the referenced component must actually exist with the
    # expected fields. Pin so a partial removal (response_model
    # declared but model deleted) regresses.
    components = spec.get("components", {}).get("schemas", {})
    assert "ContradictionsResponse" in components
    crefs = components["ContradictionsResponse"]
    assert "contradictions" in crefs.get("properties", {})
    assert "total_claims_scanned" in crefs.get("properties", {})


def test_contradictions_response_required_fields_include_total_claims_scanned():
    """Reviewer-flagged Low on commit 86c1dbd. ``total_claims_scanned``
    is the load-bearing signal for distinguishing "no writer
    yet" (count=0) from "writer healthy, consistent data"
    (count>0, empty contradictions list). Pre-fix the schema
    declared ``Field(default=0, ...)`` so OpenAPI marked it
    optional — generated clients would have written
    conditional access logic that hides the signal.

    Pin: ``total_claims_scanned`` (alongside ``transform_id``
    and ``contradictions``) must appear in the schema's
    ``required`` list so consumers know they can always rely
    on it. ``min_confidence`` legitimately has a default
    (0.0), so it stays optional."""
    from graphora_server.main import app

    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    required = set(components.get("ContradictionsResponse", {}).get("required", []))

    assert "transform_id" in required
    assert "contradictions" in required
    assert "total_claims_scanned" in required, (
        "total_claims_scanned must be required so OpenAPI "
        "consumers can rely on it as the empty-state "
        f"distinguishing signal. Required list: {required!r}"
    )
    # min_confidence is genuinely optional (defaults to 0.0)
    # and should stay that way.
    assert "min_confidence" not in required
