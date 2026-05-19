"""Unit tests for the GET /api/v1/bench/run endpoint (B4-bench).

The endpoint reads ``bench/results/`` and ``golden/`` from the
repo root and returns the aggregated benchmark report.
Underlying runner logic is exercised in
``tests/unit/services/bench/test_runner.py``; this test surface
just pins the HTTP shape:
  * Endpoint is reachable (200) and serializable as JSON.
  * Endpoint is unauthenticated — the bench is public.
  * The response shape carries the BenchRunReport.to_dict() keys
    the frontend will read.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from graphora_server.main import app


@pytest.fixture
def test_client():
    """Plain TestClient — bench endpoint takes no auth dep, so
    nothing needs overriding."""
    return TestClient(app)


def _fake_runner_factory(tmp_path: Path):
    """Build a BenchRunner pointing at ``tmp_path`` instead of the
    real repo root. The endpoint imports its own runner at request
    time; we monkeypatch the constructor to swap repo_root in the
    test."""
    from graphora_server.services.bench import BenchRunner as _BenchRunner

    def factory(*args, **kwargs):
        kwargs["repo_root"] = tmp_path
        return _BenchRunner(*args, **kwargs)

    return factory


def _seed_corpus(repo_root: Path, slug: str = "alpha") -> None:
    entry = repo_root / "golden" / slug
    entry.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": [
            {
                "id": "cid-a",
                "type": "Person",
                "label": "A",
                "properties": {
                    "canonical_id": "cid-a",
                    "canonical_key": "Person:name=a",
                    "name": "A",
                },
            }
        ],
        "edges": [],
    }
    (entry / "document.txt").write_text("doc")
    (entry / "ontology.yaml").write_text("version: '0.1.0'\n")
    (entry / "expected.json").write_text(json.dumps(payload))
    return payload


def _seed_extractor_output(repo_root: Path, extractor: str, slug: str, payload) -> None:
    out_dir = repo_root / "bench" / "results" / extractor
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(payload))


# ============================================================
# Happy path
# ============================================================


def test_run_endpoint_returns_aggregated_report(test_client, tmp_path):
    """End-to-end through HTTP: seed a corpus entry + one extractor
    with an identity-copy output, hit the endpoint, assert the
    response carries the expected shape with non-zero scores."""
    payload = _seed_corpus(tmp_path)
    _seed_extractor_output(tmp_path, "smoke_self", "alpha", payload)

    with patch(
        "graphora_server.api.bench.BenchRunner",
        side_effect=_fake_runner_factory(tmp_path),
    ):
        resp = test_client.get("/api/v1/bench/run")

    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus_size"] == 1
    assert body["extractor_count"] == 1
    extractor = body["extractors"][0]
    assert extractor["extractor_name"] == "smoke_self"
    assert extractor["scored_count"] == 1
    assert extractor["errored_count"] == 0
    assert extractor["micro_node_f1"] == 1.0
    assert extractor["macro_node_f1"] == 1.0


def test_run_endpoint_is_unauthenticated(test_client, tmp_path):
    """The bench is public by design — no Authorization header in
    the request, and the endpoint should respond 200 without an
    auth dependency overriding it.

    Pin so a future "make all endpoints authenticated" sweep
    that catches /api/v1/bench/run accidentally would fail this
    test before merging — the reproducibility claim depends on
    anonymous fetchability."""
    _seed_corpus(tmp_path)
    with patch(
        "graphora_server.api.bench.BenchRunner",
        side_effect=_fake_runner_factory(tmp_path),
    ):
        resp = test_client.get("/api/v1/bench/run")
    # No Authorization header was set, no auth override.
    assert resp.status_code == 200


def test_run_endpoint_response_serializes_per_entry_detail(test_client, tmp_path):
    """The frontend renders per-entry detail (which corpus slugs
    were scored vs. errored). Pin that the response shape includes
    the ``entries`` array with per-slug breakdowns."""
    payload = _seed_corpus(tmp_path)
    _seed_corpus(tmp_path, slug="beta")
    _seed_extractor_output(tmp_path, "partial", "alpha", payload)
    # No beta.json — should surface as errored entry.

    with patch(
        "graphora_server.api.bench.BenchRunner",
        side_effect=_fake_runner_factory(tmp_path),
    ):
        resp = test_client.get("/api/v1/bench/run")

    assert resp.status_code == 200
    body = resp.json()
    extractor = body["extractors"][0]
    entries = extractor["entries"]
    assert len(entries) == 2
    alpha_entry = next(e for e in entries if e["corpus_slug"] == "alpha")
    beta_entry = next(e for e in entries if e["corpus_slug"] == "beta")
    assert alpha_entry["error"] is None
    assert alpha_entry["node_f1"] == 1.0
    # Beta is errored — extractor lacked output for this slug.
    assert beta_entry["error"] is not None
    assert "beta.json" in beta_entry["error"]


def test_run_endpoint_empty_bench_returns_zero_extractors(test_client, tmp_path):
    """Slice 1 ships with no extractor outputs committed. The
    endpoint must respond 200 with an empty extractors list,
    NOT 404 or 500. Pin the empty state for the frontend to
    render an "awaiting bench data" placeholder."""
    _seed_corpus(tmp_path)
    # No bench/results/<extractor>/ directories at all.

    with patch(
        "graphora_server.api.bench.BenchRunner",
        side_effect=_fake_runner_factory(tmp_path),
    ):
        resp = test_client.get("/api/v1/bench/run")

    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus_size"] == 1
    assert body["extractor_count"] == 0
    assert body["extractors"] == []
