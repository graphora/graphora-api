"""Unit tests for the sample-gallery endpoints (demo surface).

The endpoints serve the on-disk ``golden/<slug>/`` entries as a
public, no-auth gallery. Tests pin the HTTP shape against the
real on-disk corpus (50 entries as of 2026-05-20) so a corpus
add/rename surfaces here too.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graphora_server.api.samples import _load_all_samples
from graphora_server.main import app
from graphora_server.schemas.samples import (
    SampleDetail,
    SamplesListResponse,
)


@pytest.fixture(autouse=True)
def _clear_sample_cache():
    """The endpoint caches the samples index via ``lru_cache``.
    Clear before each test so a corpus-on-disk change between
    tests doesn't read a stale cache."""
    _load_all_samples.cache_clear()
    yield
    _load_all_samples.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# List endpoint
# ============================================================


def test_list_endpoint_returns_200_with_envelope(client):
    resp = client.get("/api/v1/samples")
    assert resp.status_code == 200
    body = resp.json()
    assert "samples" in body
    assert "domains" in body


def test_list_endpoint_is_unauthenticated(client):
    """Demo gallery is public by design — no Authorization
    header sent, and no auth dependency override needed. Pin
    so a future 'auth all endpoints' sweep that catches
    /samples accidentally would fail here before merging."""
    resp = client.get("/api/v1/samples")
    assert resp.status_code == 200


def test_list_endpoint_returns_full_corpus(client):
    """Pin against the live golden directory. As of 2026-05-20
    the corpus is 50 entries; this assertion floors at 10 so a
    deploy on a checkout with fewer entries still passes, but
    a corpus regression that drops below 10 would fail here."""
    resp = client.get("/api/v1/samples")
    body = resp.json()
    # Match the corpus floor pin from
    # tests/unit/services/test_golden_corpus_invariants.py.
    assert len(body["samples"]) >= 10


def test_list_endpoint_carries_per_sample_counts(client):
    """Pin that each summary item has the load-bearing fields
    the gallery card needs to render (no extra fetch required
    to show node/edge counts on the grid)."""
    resp = client.get("/api/v1/samples")
    body = resp.json()
    sample = body["samples"][0]
    # Required fields per the SampleSummary schema.
    for required in ("slug", "display_name", "domain", "node_count", "edge_count"):
        assert required in sample, f"missing field {required!r} on sample card"
    assert sample["node_count"] >= 0
    assert sample["edge_count"] >= 0


def test_list_endpoint_includes_known_domain(client):
    """The Domain column in golden/README.md drives the filter
    chips. Pin Healthcare since it's one of the most populated
    domains — if the parser ever silently drops the domain
    mapping, this fails."""
    resp = client.get("/api/v1/samples")
    body = resp.json()
    assert "Healthcare" in body["domains"]


def test_list_endpoint_response_validates_against_pydantic_schema(client):
    """Round-trip through the schema — a future refactor that
    breaks the dataclass → wire projection fails here rather
    than as a runtime ValidationError on first user request."""
    resp = client.get("/api/v1/samples")
    SamplesListResponse.model_validate(resp.json())


# ============================================================
# Detail endpoint
# ============================================================


def test_detail_endpoint_returns_known_sample(client):
    """``healthcare_clinical_note`` is one of the seed entries
    and has been stable across the corpus growth. Pin against
    its presence + payload shape."""
    resp = client.get("/api/v1/samples/healthcare_clinical_note")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "healthcare_clinical_note"
    assert body["domain"] == "Healthcare"
    # Each detail must carry the four heavy-content fields.
    assert body["document"]  # non-empty
    assert body["ontology_yaml"]
    assert body["expected_graph"]
    assert body["readme_markdown"]


def test_detail_endpoint_404_on_unknown_slug(client):
    resp = client.get("/api/v1/samples/not-a-real-slug")
    assert resp.status_code == 404
    assert (
        "not a real slug" in resp.json()["detail"].lower()
        or "no sample with slug" in resp.json()["detail"].lower()
    )


def test_detail_endpoint_expected_graph_has_nodes_and_edges(client):
    """The detail page renders the graph via the frontend's
    graph-viz component, which expects the GraphResponse shape
    (``nodes`` + ``edges``). Pin the wire structure so a
    silent reformat at the storage layer (e.g., wrapping in
    ``data: {...}``) breaks here."""
    resp = client.get("/api/v1/samples/healthcare_clinical_note")
    graph = resp.json()["expected_graph"]
    assert isinstance(graph["nodes"], list)
    assert isinstance(graph["edges"], list)
    assert len(graph["nodes"]) > 0


def test_detail_endpoint_validates_against_pydantic_schema(client):
    resp = client.get("/api/v1/samples/healthcare_clinical_note")
    SampleDetail.model_validate(resp.json())


# ============================================================
# Roster parser
# ============================================================


def test_every_sample_has_a_domain(client):
    """Every sample SHOULD have a domain from the roster table.
    Samples that don't appear in golden/README.md fall back to
    'Other' — flag them as data hygiene if more than a few
    exist. This is a soft pin: the gallery still works with
    'Other', but a fresh slug without a roster row means the
    gallery's filter UX is degraded for it."""
    resp = client.get("/api/v1/samples")
    body = resp.json()
    other_slugs = [s["slug"] for s in body["samples"] if s["domain"] == "Other"]
    # As of 2026-05-20 every corpus slug has a roster entry;
    # the float tolerance is just defensive.
    assert len(other_slugs) <= 1, (
        f"{len(other_slugs)} samples have no roster domain — add them "
        f"to golden/README.md: {other_slugs}"
    )
