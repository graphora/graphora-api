"""Unit tests for the /api/v1/golden/score endpoint (B4-test).

The endpoint takes a caller-supplied expected graph + a live
transform_id and returns the P/R/F1 ScoringReport. Tests cover:
  * Happy path: matching expected + actual → perfect scores
  * Mismatched extraction → expected FP/FN counts surface
  * Tenant scoping: loader is called with auth.user_id
  * Truncation guard: oversized actual → 413 (mirrors /diff;
    reviewer-flagged High on commit 9fb4909)
  * Empty actual scores as all-FN, not 404 (reviewer-flagged
    Medium on commit 9fb4909 — a real "extractor found nothing"
    outcome is a domain result, not an infrastructure error)
  * Malformed request → 422 (Pydantic validation)
  * The seed corpus expected.json roundtrips cleanly when paired
    with a mock-graph identical to it
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.main import app
from graphora_server.schemas.graph import GraphResponse, Node


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


def _node(
    *,
    id: str,
    type: str,
    canonical_id: str,
    canonical_key: str,
    name: str,
) -> dict[str, Any]:
    """JSON-shaped node payload (POST body uses raw JSON, not
    Pydantic objects)."""
    return {
        "id": id,
        "type": type,
        "label": name,
        "properties": {
            "name": name,
            "canonical_id": canonical_id,
            "canonical_key": canonical_key,
        },
    }


def _node_obj(
    *,
    id: str,
    type: str,
    canonical_id: str,
    canonical_key: str,
    name: str,
) -> Node:
    """Server-side Node (returned from the load helper mock)."""
    return Node(
        id=id,
        label=name,
        type=type,
        properties={
            "name": name,
            "canonical_id": canonical_id,
            "canonical_key": canonical_key,
        },
    )


def test_score_endpoint_perfect_match(test_client):
    """Identical expected + actual → P/R/F1 all 1.0 with the
    correct TP counts. Pin so a regression that scrambles the
    expected/actual orientation (which would give zero TPs)
    surfaces immediately."""
    actual_graph = GraphResponse(
        nodes=[
            _node_obj(
                id="n1",
                type="Person",
                canonical_id="cid-alice",
                canonical_key="Person:name=alice",
                name="Alice",
            )
        ],
        edges=[],
        total_nodes=1,
        total_edges=0,
    )

    expected_payload = {
        "nodes": [
            _node(
                id="n1",
                type="Person",
                canonical_id="cid-alice",
                canonical_key="Person:name=alice",
                name="Alice",
            )
        ],
        "edges": [],
    }

    with patch(
        "graphora_server.api.golden._load_graph_for_diff",
        new=AsyncMock(return_value=actual_graph),
    ):
        response = test_client.post(
            "/api/v1/golden/score",
            json={
                "expected": expected_payload,
                "transform_id": "tx-1",
                "corpus_slug": "single_person",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["corpus_slug"] == "single_person"
    assert body["nodes"]["precision"] == 1.0
    assert body["nodes"]["recall"] == 1.0
    assert body["nodes"]["f1"] == 1.0
    assert body["nodes"]["by_type"]["Person"]["true_positives"] == 1


def test_score_endpoint_surfaces_fp_fn_on_mismatch(test_client):
    """Expected has Alice; actual has Bob (different canonical_id).
    Person gets 1 FN (Alice missing) + 1 FP (Bob hallucinated).
    Pin so the FP/FN bookkeeping flows through the wire layer."""
    actual_graph = GraphResponse(
        nodes=[
            _node_obj(
                id="n2",
                type="Person",
                canonical_id="cid-bob",
                canonical_key="Person:name=bob",
                name="Bob",
            )
        ],
        edges=[],
        total_nodes=1,
        total_edges=0,
    )
    expected_payload = {
        "nodes": [
            _node(
                id="n1",
                type="Person",
                canonical_id="cid-alice",
                canonical_key="Person:name=alice",
                name="Alice",
            )
        ],
        "edges": [],
    }

    with patch(
        "graphora_server.api.golden._load_graph_for_diff",
        new=AsyncMock(return_value=actual_graph),
    ):
        response = test_client.post(
            "/api/v1/golden/score",
            json={"expected": expected_payload, "transform_id": "tx-1"},
        )

    assert response.status_code == 200
    body = response.json()
    person = body["nodes"]["by_type"]["Person"]
    assert person["true_positives"] == 0
    assert person["false_positives"] == 1, (
        "Bob (in actual but not expected) should land as FP, but "
        f"the endpoint reported {person['false_positives']} FPs."
    )
    assert person["false_negatives"] == 1


def test_score_endpoint_passes_auth_user_id_to_loader(test_client):
    """Tenant scoping pin: the load helper must receive auth.user_id.
    Pin so a refactor that drops the user-isolation arg silently
    succeeds (it would silently leak other users' scores)."""
    actual_graph = GraphResponse(
        nodes=[
            _node_obj(
                id="n1",
                type="Person",
                canonical_id="cid-alice",
                canonical_key="Person:name=alice",
                name="Alice",
            )
        ],
        edges=[],
    )
    load_mock = AsyncMock(return_value=actual_graph)

    expected_payload = {"nodes": [], "edges": []}

    with patch("graphora_server.api.golden._load_graph_for_diff", new=load_mock):
        test_client.post(
            "/api/v1/golden/score",
            json={"expected": expected_payload, "transform_id": "tx-1"},
        )

    load_mock.assert_awaited_once()
    # Positional args: (transform_id, user_id)
    args = load_mock.await_args.args
    assert args[0] == "tx-1"
    assert args[1] == "test-user-1", (
        "Loader didn't receive the authenticated user_id. " f"Got: {args!r}"
    )


def test_score_endpoint_413s_when_actual_truncated(test_client):
    """Reviewer-flagged High on commit 9fb4909: mirror /diff's
    truncation guard. If the loader returns a graph with
    ``total_nodes`` exceeding the 10k diff cap, scoring against
    a partial actual would silently produce a confident-but-wrong
    report (typically all-FN for the missing tail). Pin so a
    refactor that drops _check_truncated regresses noisily.

    The 413 detail body shape (truncated_dimension, side,
    transform_id) is inherited from /diff — re-checking those
    fields catches contract drift, not just status code drift."""
    oversized_actual = GraphResponse(
        nodes=[
            _node_obj(
                id="n1",
                type="Person",
                canonical_id="cid-a",
                canonical_key="Person:name=a",
                name="A",
            )
        ],
        edges=[],
        total_nodes=15000,  # exceeds _DIFF_NODE_LIMIT (10000)
        total_edges=0,
    )
    with patch(
        "graphora_server.api.golden._load_graph_for_diff",
        new=AsyncMock(return_value=oversized_actual),
    ):
        response = test_client.post(
            "/api/v1/golden/score",
            json={
                "expected": {"nodes": [], "edges": []},
                "transform_id": "tx-too-big",
            },
        )

    assert response.status_code == 413, (
        "/golden/score must 413 on truncated actual graphs, same "
        "as /diff. Got "
        f"{response.status_code} — the truncation guard is missing."
    )
    body = response.json()["detail"]
    assert body["error"] == "transform_too_large_to_diff"
    assert body["side"] == "actual"
    assert body["transform_id"] == "tx-too-big"
    assert body["truncated_dimension"] == "nodes"


def test_score_endpoint_scores_empty_actual_as_all_fn(test_client):
    """Reviewer-flagged Medium on commit 9fb4909: empty actual
    graphs are a legitimate "extractor found nothing" outcome,
    not a 404. The CLI runner needs to see this as a regression
    signal — every expected node becomes FN, recall drops to 0.
    Pin so a future refactor that re-adds the 404 branch
    regresses noisily.

    Note: this also covers the "missing or cross-tenant
    transform_id" case (loader returns empty for both). The
    trade-off is documented inline in golden.py: a malformed
    transform_id no longer 404s, but the CLI runner controls
    transform_id (it just uploaded the doc) so this is
    acceptable."""
    empty_graph = GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)
    expected_payload = {
        "nodes": [
            _node(
                id="n1",
                type="Person",
                canonical_id="cid-alice",
                canonical_key="Person:name=alice",
                name="Alice",
            )
        ],
        "edges": [],
    }
    with patch(
        "graphora_server.api.golden._load_graph_for_diff",
        new=AsyncMock(return_value=empty_graph),
    ):
        response = test_client.post(
            "/api/v1/golden/score",
            json={
                "expected": expected_payload,
                "transform_id": "tx-zero",
                "corpus_slug": "regression-doc",
            },
        )

    assert response.status_code == 200, (
        "Empty actual must score, not 404. Got "
        f"{response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["corpus_slug"] == "regression-doc"
    person = body["nodes"]["by_type"]["Person"]
    assert person["true_positives"] == 0
    assert person["false_positives"] == 0
    assert person["false_negatives"] == 1, (
        "Empty actual + 1 expected Person must produce 1 FN, but "
        f"got {person['false_negatives']}. The all-FN contract is "
        "what the CLI runner reads to flag the corpus doc as a "
        "regression."
    )
    # Recall must be 0 (0 TP out of 1 expected); precision is 0/0
    # which the scorer maps to 0.0 (not NaN) for JSON cleanliness.
    assert body["nodes"]["recall"] == 0.0
    assert body["nodes"]["precision"] == 0.0


def test_score_endpoint_rejects_malformed_expected(test_client):
    """Pydantic validation on the request body: an expected
    payload missing required Node fields (e.g., ``type``)
    returns 422 BEFORE the loader is touched. Pin to catch a
    future refactor that loosens the schema."""
    bad_payload = {
        "expected": {"nodes": [{"id": "x", "label": "x"}], "edges": []},
        "transform_id": "tx-1",
    }
    response = test_client.post("/api/v1/golden/score", json=bad_payload)
    assert response.status_code == 422


def test_score_endpoint_with_seed_corpus_roundtrips_clean(test_client):
    """End-to-end pin: load the seed corpus expected.json, POST
    it with a mock actual graph constructed from the same JSON,
    expect a perfect-score report. This confirms the corpus
    shape is wire-compatible with the endpoint — a future doc
    that drifts from the shape would fail this assertion.

    The B4-corpus seed expected.json was reviewer-aligned with
    the live extraction helpers (commit a1505e0), so this test
    also implicitly confirms that path."""
    repo_root = Path(__file__).resolve().parents[3]
    seed_dir = repo_root / "golden" / "single_person_works_at_org"
    expected_payload = json.loads((seed_dir / "expected.json").read_text())

    # Use the same payload as the "actual" extraction (perfect
    # match) — the loader returns the same content as a
    # GraphResponse object.
    actual_graph = GraphResponse.model_validate(expected_payload)

    with patch(
        "graphora_server.api.golden._load_graph_for_diff",
        new=AsyncMock(return_value=actual_graph),
    ):
        response = test_client.post(
            "/api/v1/golden/score",
            json={
                "expected": expected_payload,
                "transform_id": "tx-seed",
                "corpus_slug": "single_person_works_at_org",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["corpus_slug"] == "single_person_works_at_org"
    # Both Person and Organization should score perfect.
    assert body["nodes"]["precision"] == 1.0
    assert body["nodes"]["recall"] == 1.0
    assert body["nodes"]["f1"] == 1.0
    assert body["edges"]["precision"] == 1.0
    assert body["edges"]["recall"] == 1.0
    assert body["edges"]["f1"] == 1.0
