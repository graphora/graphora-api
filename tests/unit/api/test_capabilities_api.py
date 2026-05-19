"""Unit tests for ``GET /api/v1/capabilities/matrix`` (M-matrix).

The endpoint surfaces the static matrix from
``services.storage.capabilities.BACKEND_MATRIX``. Tests pin the
HTTP surface:
  * 200 with the wire envelope and one row per backend.
  * Unauthenticated — the matrix is repo-level documentation.
  * Round-trips through the Pydantic ``CapabilityMatrixResponse``
    schema so OpenAPI consumers get the typed shape.
  * Per-row content fidelity — Neo4j caps come through, AGE's
    dynamic_flags surfaces, memory's dev/demo note is present.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graphora_server.main import app
from graphora_server.schemas.capabilities import CapabilityMatrixResponse


@pytest.fixture
def client():
    return TestClient(app)


def test_matrix_endpoint_returns_200_with_envelope(client):
    resp = client.get("/api/v1/capabilities/matrix")
    assert resp.status_code == 200
    body = resp.json()
    assert "backends" in body
    assert isinstance(body["backends"], list)
    assert len(body["backends"]) >= 3  # neo4j + postgres + memory


def test_matrix_endpoint_is_unauthenticated(client):
    """The matrix is repo-level documentation — no tenant
    scoping needed. Pin so a future 'auth all the things' sweep
    that catches /capabilities/matrix accidentally would fail
    here before merging."""
    # No Authorization header set.
    resp = client.get("/api/v1/capabilities/matrix")
    assert resp.status_code == 200


def test_matrix_response_validates_against_pydantic_schema(client):
    """Round-trip the live HTTP response through the Pydantic
    schema. A future refactor that breaks the dataclass →
    Pydantic projection (renames a field, drops a required key,
    etc.) fails here with a clear ValidationError rather than
    silently shipping a broken contract.

    The Pydantic models are also what OpenAPI generates client
    stubs from; if validate fails here, generated clients fail
    too."""
    resp = client.get("/api/v1/capabilities/matrix")
    assert resp.status_code == 200
    validated = CapabilityMatrixResponse.model_validate(resp.json())
    assert len(validated.backends) >= 3


def test_neo4j_row_carries_full_feature_set(client):
    """Neo4j is the reference backend with every flag True (no
    dynamic). Pin the headline numbers so a regression on the
    matrix-as-truth view (renames in capabilities.py that
    accidentally flip a flag, etc.) surfaces here."""
    resp = client.get("/api/v1/capabilities/matrix")
    body = resp.json()
    neo4j = next(b for b in body["backends"] if b["name"] == "neo4j")
    caps = neo4j["default_capabilities"]
    assert caps["persistent"] is True
    assert caps["full_text_indexes"] is True
    assert caps["similarity_search"] is True
    assert caps["per_user_routing"] is True
    # No flags are runtime-detected.
    assert neo4j["dynamic_flags"] == []


def test_postgres_row_marks_full_text_indexes_dynamic(client):
    """The reviewer-flagged design choice: AGE's
    ``full_text_indexes`` depends on pg_trgm being installed.
    The matrix reports the conservative default (False) and
    advertises full_text_indexes in dynamic_flags so the
    frontend can render the cell with a "depends on runtime"
    annotation."""
    resp = client.get("/api/v1/capabilities/matrix")
    body = resp.json()
    pg = next(b for b in body["backends"] if b["name"] == "postgres")
    assert pg["default_capabilities"]["full_text_indexes"] is False
    assert "full_text_indexes" in pg["dynamic_flags"]


def test_memory_row_carries_dev_demo_note(client):
    """The in-memory backend is dev/demo only — operators
    picking STORAGE_TYPE=memory for production deployments
    are reading the matrix page hoping to see this caveat.
    Pin that the warning lands in the rendered notes."""
    resp = client.get("/api/v1/capabilities/matrix")
    body = resp.json()
    memory = next(b for b in body["backends"] if b["name"] == "memory")
    joined = " ".join(memory["notes"]).lower()
    assert "dev" in joined or "demo" in joined


def test_render_order_is_preserved(client):
    """Frontend matrix renderers depend on the order: reference
    backend first (Neo4j), production alternatives next
    (Postgres), then dev/demo (memory). Pin so a refactor that
    accidentally sorts alphabetically would regress the
    operator-mental-anchor order."""
    resp = client.get("/api/v1/capabilities/matrix")
    body = resp.json()
    names = [b["name"] for b in body["backends"]]
    assert names[0] == "neo4j"
    assert names[-1] == "memory"
