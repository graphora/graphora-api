"""Unit tests for the golden-corpus scorer (B4-corpus).

The scorer wraps ``DiffService``'s identity-matching logic so the
test surface is small: build two ``GraphResponse`` fixtures
(expected + actual), run them through ``CorpusScorer.score()``,
assert the P/R/F1 numbers + per-type breakdown.

Identity matching depends on ``canonical_id`` / ``canonical_key``
read from the node properties bag (see
``diff_service._canonical_id_or_none``), so test fixtures stamp
those onto ``properties`` rather than as top-level fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from graphora_server.schemas.graph import Edge, GraphResponse, Node
from graphora_server.services.golden_corpus import CorpusScorer


def _node(
    *,
    id: str,
    type: str,
    canonical_id: str,
    canonical_key: str,
    name: str,
    extra_props: Dict[str, Any] | None = None,
) -> Node:
    """Build a Node with canonical_id/canonical_key in properties
    (the shape DiffService's _canonical_id_or_none reads from)."""
    props: Dict[str, Any] = {
        "name": name,
        "canonical_id": canonical_id,
        "canonical_key": canonical_key,
    }
    if extra_props:
        props.update(extra_props)
    return Node(id=id, label=name, type=type, properties=props)


def _edge(
    *,
    id: str,
    type: str,
    source: str,
    target: str,
    properties: Dict[str, Any] | None = None,
) -> Edge:
    return Edge(
        id=id,
        type=type,
        source=source,
        target=target,
        properties=properties or {},
    )


# ============================================================
# Aggregate + per-type scoring
# ============================================================


def test_perfect_match_yields_full_scores():
    """Identical expected vs actual → every node/edge is a TP.
    Precision, recall, and F1 all 1.0. Pin so a refactor that
    flips the diff orientation (base/compare swap) immediately
    surfaces as zero scores."""
    nodes = [
        _node(
            id="n1",
            type="Person",
            canonical_id="alice",
            canonical_key="alice",
            name="Alice",
        ),
        _node(
            id="n2",
            type="Organization",
            canonical_id="acme",
            canonical_key="acme",
            name="Acme",
        ),
    ]
    edges = [_edge(id="e1", type="WORKS_AT", source="n1", target="n2")]
    expected = GraphResponse(nodes=nodes, edges=edges, total_nodes=2, total_edges=1)
    actual = GraphResponse(nodes=nodes, edges=edges, total_nodes=2, total_edges=1)

    report = CorpusScorer().score(expected, actual)

    assert report.nodes.precision == 1.0
    assert report.nodes.recall == 1.0
    assert report.nodes.f1 == 1.0
    assert report.edges.precision == 1.0
    assert report.edges.recall == 1.0
    assert report.edges.f1 == 1.0
    # Per-type breakdown also surfaces TP for each present type.
    assert report.nodes.by_type["Person"].true_positives == 1
    assert report.nodes.by_type["Organization"].true_positives == 1
    assert report.edges.by_type["WORKS_AT"].true_positives == 1


def test_missing_node_drops_recall_keeps_precision():
    """Expected has 2 Persons; actual has 1. The missing Person
    is an FN — recall drops; precision stays at 1.0 because
    everything the extractor produced was correct."""
    alice = _node(
        id="n1",
        type="Person",
        canonical_id="alice",
        canonical_key="alice",
        name="Alice",
    )
    bob = _node(
        id="n2",
        type="Person",
        canonical_id="bob",
        canonical_key="bob",
        name="Bob",
    )
    expected = GraphResponse(nodes=[alice, bob], edges=[], total_nodes=2, total_edges=0)
    actual = GraphResponse(nodes=[alice], edges=[], total_nodes=1, total_edges=0)

    report = CorpusScorer().score(expected, actual)

    assert report.nodes.precision == 1.0
    # 1 TP / (1 TP + 1 FN) = 0.5
    assert report.nodes.recall == pytest.approx(0.5)
    assert report.nodes.f1 == pytest.approx(2 / 3)
    # Person breakdown: 1 TP (alice), 0 FP, 1 FN (bob).
    person = report.nodes.by_type["Person"]
    assert person.true_positives == 1
    assert person.false_positives == 0
    assert person.false_negatives == 1


def test_extra_node_drops_precision_keeps_recall():
    """Mirror case: actual has an extra Person not in expected.
    That's an FP — precision drops; recall stays at 1.0 because
    everything in the ground truth was found."""
    alice = _node(
        id="n1",
        type="Person",
        canonical_id="alice",
        canonical_key="alice",
        name="Alice",
    )
    bob = _node(
        id="n2",
        type="Person",
        canonical_id="bob",
        canonical_key="bob",
        name="Bob",
    )
    expected = GraphResponse(nodes=[alice], edges=[], total_nodes=1, total_edges=0)
    actual = GraphResponse(nodes=[alice, bob], edges=[], total_nodes=2, total_edges=0)

    report = CorpusScorer().score(expected, actual)

    # 1 TP / (1 TP + 1 FP) = 0.5
    assert report.nodes.precision == pytest.approx(0.5)
    assert report.nodes.recall == 1.0
    assert report.nodes.f1 == pytest.approx(2 / 3)
    person = report.nodes.by_type["Person"]
    assert person.true_positives == 1
    assert person.false_positives == 1
    assert person.false_negatives == 0


def test_changed_node_counts_as_identity_match_tp():
    """When a node identity-matches but its properties differ,
    the diff returns it as a ``changed_node``. The scorer counts
    it as a TP at the identity level — partial-credit property
    scoring is a future slice. Pin the contract."""
    alice_expected = _node(
        id="n1",
        type="Person",
        canonical_id="alice",
        canonical_key="alice",
        name="Alice",
        extra_props={"title": "Engineer"},
    )
    alice_actual = _node(
        id="n1",
        type="Person",
        canonical_id="alice",
        canonical_key="alice",
        name="Alice",
        extra_props={"title": "Senior Engineer"},  # changed property
    )
    expected = GraphResponse(
        nodes=[alice_expected], edges=[], total_nodes=1, total_edges=0
    )
    actual = GraphResponse(nodes=[alice_actual], edges=[], total_nodes=1, total_edges=0)

    report = CorpusScorer().score(expected, actual)

    # The matched-but-property-changed pair counts as TP at the
    # identity level → no FP/FN, full P/R/F1.
    assert report.nodes.precision == 1.0
    assert report.nodes.recall == 1.0
    assert report.nodes.f1 == 1.0
    assert report.nodes.by_type["Person"].true_positives == 1


def test_per_type_breakdown_isolates_failures():
    """Mixed-type cases: Person extraction is perfect, but the
    Organization extraction missed a node. Pin that scoring is
    per-type so a partial failure in one type doesn't drag the
    other type's score."""
    alice = _node(
        id="n1",
        type="Person",
        canonical_id="alice",
        canonical_key="alice",
        name="Alice",
    )
    acme = _node(
        id="n2",
        type="Organization",
        canonical_id="acme",
        canonical_key="acme",
        name="Acme",
    )
    initech = _node(
        id="n3",
        type="Organization",
        canonical_id="initech",
        canonical_key="initech",
        name="Initech",
    )
    expected = GraphResponse(
        nodes=[alice, acme, initech], edges=[], total_nodes=3, total_edges=0
    )
    actual = GraphResponse(nodes=[alice, acme], edges=[], total_nodes=2, total_edges=0)

    report = CorpusScorer().score(expected, actual)

    # Person: perfect.
    person = report.nodes.by_type["Person"]
    assert person.precision == 1.0
    assert person.recall == 1.0
    # Organization: 1 TP (acme), 1 FN (initech).
    org = report.nodes.by_type["Organization"]
    assert org.precision == 1.0
    assert org.recall == pytest.approx(0.5)
    # Aggregate: 2 TP, 0 FP, 1 FN.
    assert report.nodes.precision == 1.0
    assert report.nodes.recall == pytest.approx(2 / 3)


def test_empty_graphs_yield_zero_not_nan():
    """Edge case: both expected and actual are empty. There are
    no instances to score against. The scorer returns 0.0
    everywhere rather than NaN — JSON-serializable, predictable
    for batch aggregation."""
    empty = GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)
    report = CorpusScorer().score(empty, empty)
    assert report.nodes.precision == 0.0
    assert report.nodes.recall == 0.0
    assert report.nodes.f1 == 0.0


def test_type_mismatch_with_same_canonical_id_scores_as_fp_plus_fn():
    """Reviewer-flagged Medium on commit 48dbe0a: a wrong-typed
    extraction that shares a canonical_id with the expected node
    used to score as a full TP for the EXPECTED type — masking
    a real failure mode (the extractor produced an Organization
    where a Person was expected, but downstream cost/aggregate
    surfaces showed "Person extraction perfect").

    Post-fix: the diff service keys canonical_id matching by
    (cid, type), so type-mismatched nodes never collapse. The
    scorer correctly records the expected-side Person as FN and
    the actual-side Organization as FP — Person scores plummet,
    Organization shows up as a hallucinated extraction."""
    alice_person = _node(
        id="n1",
        type="Person",
        canonical_id="shared-cid",
        canonical_key="alice",
        name="Alice",
    )
    alice_org = _node(
        id="n2",
        type="Organization",
        canonical_id="shared-cid",  # same cid — pre-fix this matched
        canonical_key="alice",
        name="Alice",
    )
    expected = GraphResponse(
        nodes=[alice_person], edges=[], total_nodes=1, total_edges=0
    )
    actual = GraphResponse(nodes=[alice_org], edges=[], total_nodes=1, total_edges=0)

    report = CorpusScorer().score(expected, actual)

    # Person on expected side: 0 TP, 0 FP, 1 FN (missing).
    person = report.nodes.by_type["Person"]
    assert person.true_positives == 0, (
        f"Wrong-typed actual was scored as a Person TP — pre-fix "
        f"this happened because the diff service matched on bare "
        f"canonical_id. Got {person.true_positives} TP for Person."
    )
    assert person.false_negatives == 1
    # Organization on actual side: 0 TP, 1 FP, 0 FN.
    org = report.nodes.by_type["Organization"]
    assert org.false_positives == 1
    assert org.true_positives == 0


def test_edges_scored_by_type():
    """Edges score the same way as nodes — per-type breakdown
    keyed on edge type. Pin a mixed case so a future refactor
    that keys edges differently from nodes surfaces here."""
    alice = _node(
        id="n1",
        type="Person",
        canonical_id="alice",
        canonical_key="alice",
        name="Alice",
    )
    acme = _node(
        id="n2",
        type="Organization",
        canonical_id="acme",
        canonical_key="acme",
        name="Acme",
    )
    works_at = _edge(id="e1", type="WORKS_AT", source="n1", target="n2")
    knows = _edge(id="e2", type="KNOWS", source="n1", target="n2")
    expected = GraphResponse(
        nodes=[alice, acme], edges=[works_at, knows], total_nodes=2, total_edges=2
    )
    # Actual is missing the KNOWS edge entirely.
    actual = GraphResponse(
        nodes=[alice, acme], edges=[works_at], total_nodes=2, total_edges=1
    )

    report = CorpusScorer().score(expected, actual)

    assert report.edges.by_type["WORKS_AT"].true_positives == 1
    assert report.edges.by_type["WORKS_AT"].precision == 1.0
    assert report.edges.by_type["WORKS_AT"].recall == 1.0
    # KNOWS: 0 TP, 0 FP, 1 FN → P=0, R=0, F1=0.
    knows_scores = report.edges.by_type["KNOWS"]
    assert knows_scores.true_positives == 0
    assert knows_scores.false_negatives == 1
    assert knows_scores.precision == 0.0
    assert knows_scores.recall == 0.0


def test_to_dict_serializes_full_report():
    """The serialized report includes the derived P/R/F1 numbers
    so report consumers don't recompute. Pin the shape — JSON
    breakage here breaks CI dashboards that read it."""
    alice = _node(
        id="n1",
        type="Person",
        canonical_id="alice",
        canonical_key="alice",
        name="Alice",
    )
    expected = GraphResponse(nodes=[alice], edges=[], total_nodes=1, total_edges=0)
    actual = GraphResponse(nodes=[alice], edges=[], total_nodes=1, total_edges=0)

    report = CorpusScorer().score(expected, actual, corpus_slug="my-slug")
    payload = report.to_dict()

    assert payload["corpus_slug"] == "my-slug"
    assert payload["nodes"]["true_positives"] == 1
    assert payload["nodes"]["precision"] == 1.0
    assert payload["nodes"]["recall"] == 1.0
    assert payload["nodes"]["f1"] == 1.0
    assert payload["nodes"]["by_type"]["Person"]["true_positives"] == 1
    # Roundtrips through json.dumps cleanly (no NaN, no dataclasses).
    assert json.loads(json.dumps(payload)) == payload


# ============================================================
# Seed corpus structural integrity
# ============================================================


def test_seed_corpus_doc_parses_into_graph_response():
    """The B4-corpus seed document's expected.json must validate
    as a GraphResponse. Pin so the corpus contract — drop in
    {document, ontology, expected, README} — stays mechanically
    enforceable. New docs will fail this test if their
    expected.json drifts from the live API shape."""
    # Walk up from the test file to the repo root, then into
    # golden/. Avoids hardcoding absolute paths.
    repo_root = Path(__file__).resolve().parents[3]
    seed_dir = repo_root / "golden" / "single_person_works_at_org"
    assert seed_dir.is_dir(), f"Seed corpus dir missing: {seed_dir}"
    for required in ("document.txt", "ontology.yaml", "expected.json", "README.md"):
        assert (seed_dir / required).is_file(), (
            f"Seed corpus missing {required} — the B4-corpus contract "
            "requires all four files per doc."
        )

    raw = json.loads((seed_dir / "expected.json").read_text())
    expected = GraphResponse.model_validate(raw)
    # Sanity: the doc is named after this exact pattern.
    assert any(n.type == "Person" for n in expected.nodes)
    assert any(n.type == "Organization" for n in expected.nodes)
    assert any(e.type == "WORKS_AT" for e in expected.edges)

    # Reviewer-flagged High: the Node schema only declares
    # id/label/type/properties — top-level canonical_id /
    # canonical_key are silently dropped by GraphResponse
    # .model_validate. The diff service reads identity from
    # properties, so canonical fields MUST live in properties
    # for cross-extraction matching to work. Pin so a future doc
    # author who puts them at top level (as the seed originally
    # did) fails loud rather than silently mis-scoring.
    from graphora_server.services.diff_service import (
        _canonical_id_or_none,
        _canonical_key_or_none,
    )

    for n in expected.nodes:
        assert _canonical_id_or_none(n) is not None, (
            f"Node {n.id!r} (type={n.type}) has no canonical_id "
            "visible to the diff service. The seed/expected.json "
            "shape requires canonical_id INSIDE properties — not "
            "at top level, where Pydantic drops it."
        )
        assert _canonical_key_or_none(n) is not None, (
            f"Node {n.id!r} (type={n.type}) has no canonical_key "
            "visible to the diff service. Same property-bag "
            "shape rule applies."
        )


def test_seed_expected_canonical_values_match_helper_derivation():
    """Reviewer-flagged High on commit 9e1cd30: hand-written
    canonical_id / canonical_key values in expected.json must
    match what the live extraction helpers actually produce —
    otherwise a CORRECT extraction would diff against the
    expected fixture as all FP/FN, because DiffService's
    "conflicting canonical IDs stay unmatched" contract
    (asymmetric ER constraint, commit a261321) refuses the
    canonical_key fallback when both sides have explicit-but-
    different canonical_ids.

    This test loads the ontology + expected.json, recomputes the
    canonical fields from each node's properties via the SAME
    helpers used at extraction time (``_generate_node_key`` +
    ``_make_canonical_node_id``), and asserts the values stored
    in expected.json match. Drift here means the seed is broken
    against a real extractor — surface it loud."""
    import yaml

    from graphora_server.services.transform.helpers import (
        _generate_node_key,
        _make_canonical_node_id,
    )

    repo_root = Path(__file__).resolve().parents[3]
    seed_dir = repo_root / "golden" / "single_person_works_at_org"
    parsed_ontology = yaml.safe_load((seed_dir / "ontology.yaml").read_text())
    expected_raw = json.loads((seed_dir / "expected.json").read_text())

    for node in expected_raw["nodes"]:
        props = node["properties"]
        # The helpers don't read canonical_id / canonical_key from
        # the property bag — they only consult schema-declared
        # fields (filtered by ``unique: true`` when any are
        # marked). Passing the full property bag here mirrors what
        # the live extractor does after parsing the LLM output.
        recomputed_key = _generate_node_key(
            parsed_ontology=parsed_ontology,
            entity_type=node["type"],
            properties=props,
        )
        recomputed_id = _make_canonical_node_id(recomputed_key)

        assert props["canonical_key"] == recomputed_key, (
            f"Node {node['id']!r} (type={node['type']}) has a "
            f"hand-written canonical_key that doesn't match the "
            f"helper derivation. A live extraction would produce "
            f"{recomputed_key!r} but expected.json says "
            f"{props['canonical_key']!r}. Either update the "
            f"expected to match, or change the ontology's "
            f"``unique: true`` declarations to drive the helper "
            f"toward the value you want."
        )
        assert props["canonical_id"] == recomputed_id, (
            f"Node {node['id']!r} (type={node['type']}) has a "
            f"hand-written canonical_id that doesn't match the "
            f"helper derivation. Expected.json says "
            f"{props['canonical_id']!r} but the helpers produce "
            f"{recomputed_id!r}. canonical_id is a deterministic "
            f"UUID5 over the canonical_key — fix the key first "
            f"and the id follows."
        )
