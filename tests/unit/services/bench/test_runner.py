"""Unit tests for the B4-bench runner.

The runner is filesystem-bound (discovers ``golden/<slug>/`` and
``bench/results/<extractor>/<slug>.json``), so each test builds a
fake repo root with ``tmp_path`` and asserts the resulting
:class:`BenchRunReport` shape.

Pinned invariants:
  * Self-score smoke: identity copy of expected.json → perfect P/R/F1.
  * Empty extractor (no output files): all entries errored; aggregate
    is 0 across the board but report still surfaces with the right
    extractor count.
  * Missing extractor output for one slug: that slug is errored;
    others scored; aggregate excludes the missing one.
  * Malformed JSON: error reported per entry, run completes.
  * Empty bench/results/ dir: report has 0 extractors, corpus_size
    still populated.
  * Micro vs macro divergence: a fixture that has both a high-density
    perfectly-scored entry AND a low-density zero-scored entry
    shows a clear gap between micro and macro F1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from graphora_server.services.bench import BenchRunner


def _make_corpus_entry(
    repo_root: Path,
    slug: str,
    expected: Dict[str, Any],
) -> None:
    """Drop a corpus entry with the trio under repo_root/golden/<slug>/."""
    entry_dir = repo_root / "golden" / slug
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "document.txt").write_text(f"document for {slug}")
    (entry_dir / "ontology.yaml").write_text("version: '0.1.0'\n")
    (entry_dir / "expected.json").write_text(json.dumps(expected, indent=2))


def _make_extractor_output(
    repo_root: Path,
    extractor: str,
    slug: str,
    actual: Dict[str, Any],
) -> None:
    out_dir = repo_root / "bench" / "results" / extractor
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(actual, indent=2))


def _alice_node_payload() -> Dict[str, Any]:
    """The same Alice node shape both expected.json and actual files
    use. Both carry canonical_id/canonical_key in properties so the
    DiffService's identity-matching path lights up."""
    return {
        "id": "cid-alice",
        "type": "Person",
        "label": "Alice",
        "properties": {
            "name": "Alice",
            "canonical_id": "cid-alice",
            "canonical_key": "Person:name=alice",
        },
    }


def _bob_node_payload() -> Dict[str, Any]:
    return {
        "id": "cid-bob",
        "type": "Person",
        "label": "Bob",
        "properties": {
            "name": "Bob",
            "canonical_id": "cid-bob",
            "canonical_key": "Person:name=bob",
        },
    }


def _alice_to_bob_edge() -> Dict[str, Any]:
    return {
        "id": "alice-knows-bob",
        "type": "KNOWS",
        "source": "cid-alice",
        "target": "cid-bob",
        "properties": {},
    }


def _alice_bob_graph() -> Dict[str, Any]:
    return {
        "nodes": [_alice_node_payload(), _bob_node_payload()],
        "edges": [_alice_to_bob_edge()],
    }


# ============================================================
# Discovery
# ============================================================


def test_discover_corpus_slugs_returns_only_complete_trios(tmp_path: Path):
    """A directory under ``golden/`` counts as a corpus entry only
    when all three of document.txt + ontology.yaml + expected.json
    are present. Mirrors the invariant-test discoverer's contract."""
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    # Incomplete entry — missing expected.json.
    (tmp_path / "golden" / "beta").mkdir(parents=True)
    (tmp_path / "golden" / "beta" / "document.txt").write_text("doc")
    (tmp_path / "golden" / "beta" / "ontology.yaml").write_text("v")
    # No expected.json — should be excluded.

    runner = BenchRunner(repo_root=tmp_path)
    slugs = runner.discover_corpus_slugs()
    assert slugs == ["alpha"]


def test_discover_extractors_skips_hidden_dirs(tmp_path: Path):
    """``.gitkeep`` and similar hidden entries shouldn't surface as
    extractors. Empty extractor directories DO count — they
    represent reserved slots."""
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    (tmp_path / "bench" / "results" / ".hidden").mkdir(parents=True)
    (tmp_path / "bench" / "results" / "real_extractor").mkdir(parents=True)
    (tmp_path / "bench" / "results" / "empty_slot").mkdir(parents=True)

    runner = BenchRunner(repo_root=tmp_path)
    names = runner.discover_extractors()
    assert ".hidden" not in names
    assert "real_extractor" in names
    assert "empty_slot" in names


# ============================================================
# Self-score smoke
# ============================================================


def test_identity_copy_produces_perfect_scores(tmp_path: Path):
    """The load-bearing smoke test: when the extractor output is an
    identity copy of expected.json, every score is 1.0. If this
    test fails, the scoring pipeline is broken — anchor case for
    every other assertion in this module."""
    graph = _alice_bob_graph()
    _make_corpus_entry(tmp_path, "alpha", graph)
    _make_extractor_output(tmp_path, "identity_self", "alpha", graph)

    runner = BenchRunner(repo_root=tmp_path)
    score = runner.score_entry("identity_self", "alpha")

    assert score.error is None
    assert score.node_precision == pytest.approx(1.0)
    assert score.node_recall == pytest.approx(1.0)
    assert score.node_f1 == pytest.approx(1.0)
    assert score.edge_precision == pytest.approx(1.0)
    assert score.edge_recall == pytest.approx(1.0)
    assert score.edge_f1 == pytest.approx(1.0)


# ============================================================
# Error handling per entry
# ============================================================


def test_missing_extractor_output_surfaces_as_error(tmp_path: Path):
    """An extractor that didn't produce output for a corpus entry
    must surface as an errored entry — NOT a zero F1. Otherwise
    the aggregate would silently deflate from missing coverage."""
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    (tmp_path / "bench" / "results" / "incomplete").mkdir(parents=True)

    runner = BenchRunner(repo_root=tmp_path)
    score = runner.score_entry("incomplete", "alpha")

    assert score.error is not None
    assert "alpha.json" in score.error
    assert score.node_f1 == 0.0


def test_malformed_actual_json_surfaces_as_error(tmp_path: Path):
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    out_dir = tmp_path / "bench" / "results" / "broken"
    out_dir.mkdir(parents=True)
    (out_dir / "alpha.json").write_text("{not valid json")

    runner = BenchRunner(repo_root=tmp_path)
    score = runner.score_entry("broken", "alpha")

    assert score.error is not None
    assert "JSON parse" in score.error


def test_actual_failing_pydantic_validation_surfaces_as_error(
    tmp_path: Path,
):
    """Valid JSON but wrong field types — surface the validation
    failure rather than crashing the whole run. Empty/permissive
    payloads ({} or {"foo":"bar"}) are NOT validation failures —
    GraphResponse has default-empty ``nodes`` and ``edges``, so
    those score as legitimate "extractor returned nothing"
    coverage gaps. The error path is reserved for shapes that
    actually break Pydantic — e.g., ``nodes`` carrying a
    non-list value."""
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    out_dir = tmp_path / "bench" / "results" / "shape_drift"
    out_dir.mkdir(parents=True)
    # ``nodes`` must be a list; this passes a string instead so
    # the model_validate raises.
    (out_dir / "alpha.json").write_text(
        json.dumps({"nodes": "not-a-list", "edges": []})
    )

    runner = BenchRunner(repo_root=tmp_path)
    score = runner.score_entry("shape_drift", "alpha")

    assert score.error is not None
    assert "validation failed" in score.error


def test_empty_actual_graph_scores_as_zero_recall(tmp_path: Path):
    """The complement to the validation-failure test: a structurally
    valid GraphResponse that's just empty (``{"nodes":[],
    "edges":[]}`` or even ``{}``) is NOT an error — it's a
    legitimate "extractor returned nothing" output that scores
    as zero recall. Pin this case so a future refactor that
    over-eagerly tags empty payloads as errors doesn't mistakenly
    inflate the errored_count and hide real-vs-bad-data
    distinction."""
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    out_dir = tmp_path / "bench" / "results" / "empty_output"
    out_dir.mkdir(parents=True)
    (out_dir / "alpha.json").write_text(json.dumps({}))

    runner = BenchRunner(repo_root=tmp_path)
    score = runner.score_entry("empty_output", "alpha")

    # NOT an error — empty is a legitimate result.
    assert score.error is None
    # Everything is FN (the extractor missed all expected nodes/edges).
    assert score.node_true_positives == 0
    assert score.node_false_positives == 0
    assert score.node_false_negatives == 2  # alice + bob
    assert score.node_recall == 0.0
    assert score.edge_true_positives == 0
    assert score.edge_false_negatives == 1  # the KNOWS edge
    assert score.edge_recall == 0.0


# ============================================================
# Aggregate behavior
# ============================================================


def test_run_extractor_includes_all_slugs_with_per_entry_errors(
    tmp_path: Path,
):
    """``run_extractor`` produces a BenchExtractorReport with one
    entry per corpus slug. Missing outputs become errored entries.
    Aggregate properties exclude errored entries (visible coverage
    gap rather than silent denominator deflation)."""
    graph = _alice_bob_graph()
    _make_corpus_entry(tmp_path, "alpha", graph)
    _make_corpus_entry(tmp_path, "beta", graph)
    # Only produce output for alpha.
    _make_extractor_output(tmp_path, "partial", "alpha", graph)

    runner = BenchRunner(repo_root=tmp_path)
    report = runner.run_extractor("partial")

    assert report.total_entries == 2
    assert len(report.scored_entries) == 1
    assert len(report.errored_entries) == 1
    # Micro F1 reflects only the scored alpha entry.
    assert report.micro_node_f1 == pytest.approx(1.0)
    # Macro F1 also excludes errored entries.
    assert report.macro_node_f1 == pytest.approx(1.0)


def test_micro_vs_macro_diverges_when_density_varies(tmp_path: Path):
    """Pin the micro-vs-macro distinction. The fixture has:
      * "alpha" — 5-node graph extracted perfectly → P=R=F1=1.0
      * "beta"  — 1-node graph extracted as empty → P=R=F1=0.0

    Macro F1 is the average of [1.0, 0.0] = 0.5.
    Micro F1 weights by entity count: alpha contributes 5 TP, beta
    contributes 1 FN, so node TP=5, FN=1, FP=0 →
        precision = 5/(5+0) = 1.0
        recall    = 5/(5+1) ≈ 0.833
        f1        = 2*1.0*0.833/(1.0+0.833) ≈ 0.909

    A consumer choosing micro vs macro gets meaningfully different
    headline numbers. Pin so the divergence stays visible."""

    def _node_pair(n: int) -> Dict[str, Any]:
        return {
            "id": f"cid-n{n}",
            "type": "Person",
            "label": f"Person {n}",
            "properties": {
                "canonical_id": f"cid-n{n}",
                "canonical_key": f"Person:name=person {n}",
                "name": f"Person {n}",
            },
        }

    alpha_graph = {
        "nodes": [_node_pair(i) for i in range(5)],
        "edges": [],
    }
    beta_graph = {
        "nodes": [_node_pair(99)],
        "edges": [],
    }

    _make_corpus_entry(tmp_path, "alpha", alpha_graph)
    _make_corpus_entry(tmp_path, "beta", beta_graph)
    _make_extractor_output(tmp_path, "varied", "alpha", alpha_graph)
    # Beta: extractor returns empty graph (zero recall).
    _make_extractor_output(
        tmp_path,
        "varied",
        "beta",
        {"nodes": [], "edges": []},
    )

    runner = BenchRunner(repo_root=tmp_path)
    report = runner.run_extractor("varied")

    # Macro average across two entries.
    assert report.macro_node_f1 == pytest.approx(0.5)
    # Micro is weighted by entity count.
    # TP=5, FP=0, FN=1 → P=1.0, R=5/6, F1=2*1*5/6 / (1 + 5/6)
    expected_micro = 2 * 1.0 * (5 / 6) / (1.0 + 5 / 6)
    assert report.micro_node_f1 == pytest.approx(expected_micro)


# ============================================================
# Top-level run
# ============================================================


def test_run_returns_one_report_per_extractor(tmp_path: Path):
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    _make_extractor_output(tmp_path, "extractor_a", "alpha", _alice_bob_graph())
    _make_extractor_output(tmp_path, "extractor_b", "alpha", _alice_bob_graph())

    runner = BenchRunner(repo_root=tmp_path)
    report = runner.run()

    assert report.corpus_size == 1
    assert len(report.extractor_reports) == 2
    names = sorted(r.extractor_name for r in report.extractor_reports)
    assert names == ["extractor_a", "extractor_b"]


def test_empty_bench_results_dir_returns_zero_extractors(tmp_path: Path):
    """Slice 1 ships an empty bench/results/ — pin the
    "no extractors yet" case produces a clean empty report,
    not a crash."""
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    # No bench/results/ directory at all.

    runner = BenchRunner(repo_root=tmp_path)
    report = runner.run()

    assert report.corpus_size == 1
    assert report.extractor_reports == []


def test_run_serializes_to_dict_cleanly(tmp_path: Path):
    """The to_dict() output must round-trip through json.dumps
    without TypeError — pin so a dataclass field that's not
    JSON-serializable slips through here, not at the API surface."""
    _make_corpus_entry(tmp_path, "alpha", _alice_bob_graph())
    _make_extractor_output(tmp_path, "extractor_a", "alpha", _alice_bob_graph())

    runner = BenchRunner(repo_root=tmp_path)
    report = runner.run()
    payload = report.to_dict()

    # Must JSON-serialize without raising.
    json.dumps(payload)

    # Sanity on the top-level shape.
    assert payload["corpus_size"] == 1
    assert payload["extractor_count"] == 1
    assert payload["extractors"][0]["extractor_name"] == "extractor_a"
    assert payload["extractors"][0]["total_entries"] == 1
    assert payload["extractors"][0]["scored_count"] == 1
    assert payload["extractors"][0]["errored_count"] == 0
