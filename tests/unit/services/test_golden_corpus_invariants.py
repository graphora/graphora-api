"""Invariant tests across every golden/<slug>/ corpus entry.

The corpus is editorial work that's easy to drift over time —
hand-typed canonical_ids that don't match the helper output,
schemas with missing required fields, expected.json shapes that
don't parse against GraphResponse. These tests walk every
corpus subdirectory and pin the per-entry contract so a typo
or copy-paste mistake surfaces as a hard test failure rather
than as a confusing 0-recall score during a real benchmark run.

Reviewer-flagged on commit 9e1cd30: the pre-fix seed used
hand-written canonical_ids that diverged from the helper-
computed UUID5 values. Every node landed as FP+FN because the
DiffService's "conflicting canonical IDs stay unmatched" rule
refused the canonical_key fallback. This test catches that
class of bug at corpus-load time.

What's pinned per corpus entry:
  * The required trio (document.txt + ontology.yaml +
    expected.json) all exist.
  * The README.md is present.
  * expected.json parses cleanly into GraphResponse.
  * Each expected node's canonical_id matches what the helper
    would produce given the ontology + the node's properties
    (canonical_id is helper-derived, not hand-written).
  * Each expected edge points at a node id that exists in the
    nodes list (no dangling edges in ground-truth).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from graphora_server.schemas.graph import GraphResponse
from graphora_server.services.transform.helpers import (
    _generate_node_key,
    _make_canonical_node_id,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GOLDEN_DIR = _REPO_ROOT / "golden"


def _discover_corpus_dirs() -> list[Path]:
    """Walk golden/ for subdirectories containing the trio.

    Matches the discoverer in graphora-client's
    ``graphora test`` CLI (see test_cmd.py:_discover_corpus_entries):
    a corpus entry is any subdir with all three of document.txt,
    ontology.yaml, expected.json. Sorted for stable test ids."""
    if not _GOLDEN_DIR.is_dir():
        return []
    entries: list[Path] = []
    for child in sorted(_GOLDEN_DIR.iterdir()):
        if not child.is_dir():
            continue
        if (
            (child / "document.txt").exists()
            and (child / "ontology.yaml").exists()
            and (child / "expected.json").exists()
        ):
            entries.append(child)
    return entries


_CORPUS_DIRS = _discover_corpus_dirs()


@pytest.fixture(scope="module")
def all_corpus_dirs() -> list[Path]:
    """Module-scoped so the discovery cost is paid once.

    Fails the entire module if zero corpus entries are found —
    that's almost certainly a setup error (working dir wrong,
    corpus moved, etc.) rather than an intentional empty
    state."""
    if not _CORPUS_DIRS:
        pytest.fail(
            f"No corpus entries found under {_GOLDEN_DIR}. The "
            "test discoverer looks for subdirs containing all "
            "three of document.txt + ontology.yaml + expected.json. "
            "If you deliberately emptied the corpus, delete this "
            "module too."
        )
    return _CORPUS_DIRS


def test_corpus_is_not_empty(all_corpus_dirs: list[Path]) -> None:
    """Reviewer-flagged Low on commit a68225f. The parametrized
    tests below get their parameter list from ``_CORPUS_DIRS``,
    which is captured at module import time. If ``golden/``
    disappears or every entry loses one of the trio files
    (document.txt + ontology.yaml + expected.json), the
    discoverer returns ``[]`` and the parametrized tests run
    with zero parameters — pytest **skips** them silently
    rather than failing.

    That's the dangerous-mute-pass shape: a corpus regression
    that wipes out coverage would let the suite report green.
    This test pulls the ``all_corpus_dirs`` fixture, which
    fails the whole module on empty, so an empty corpus
    surfaces as a hard test failure with the diagnostic the
    fixture writes.

    The check is also non-parametrized, so it runs even when
    the parametrized tests skip — closing the only gap where
    silent-skip could hide regression."""
    assert len(all_corpus_dirs) > 0
    # Belt-and-suspenders: the fixture would have already
    # called pytest.fail() if the list were empty, so reaching
    # here means at least one entry. Also pin a numeric floor
    # so a "regression that wipes out N-1 entries" surfaces —
    # the corpus shouldn't shrink below the current count
    # without an explicit removal.
    assert len(all_corpus_dirs) >= 50, (
        f"Corpus shrank to {len(all_corpus_dirs)} entries — current "
        "floor is 50 (the post-slice-3 growth set; Gate-4 exit "
        "target of 50+). If you're intentionally removing entries, "
        "lower the floor in this test in the same commit. Slugs "
        f"found: {sorted(d.name for d in all_corpus_dirs)!r}"
    )


@pytest.mark.parametrize("corpus_dir", _CORPUS_DIRS, ids=[d.name for d in _CORPUS_DIRS])
def test_corpus_entry_has_readme(corpus_dir: Path):
    """Every entry must document what it tests. README.md is
    where reviewers learn the per-entry test signal — "this
    entry exercises Patient/Doctor dedup with honorific
    normalization." Pin so a copy-paste new entry without a
    README surfaces immediately."""
    readme = corpus_dir / "README.md"
    assert readme.exists(), (
        f"Corpus entry {corpus_dir.name!r} is missing README.md. "
        "Every entry must document its test signal so a reviewer "
        "can tell what'll change when the score moves."
    )
    # And the README must actually have content. Pin so a
    # zero-byte placeholder doesn't slip through.
    assert (
        readme.stat().st_size > 0
    ), f"Corpus entry {corpus_dir.name!r}'s README.md is empty."


@pytest.mark.parametrize("corpus_dir", _CORPUS_DIRS, ids=[d.name for d in _CORPUS_DIRS])
def test_expected_json_parses_as_graph_response(corpus_dir: Path):
    """expected.json must parse cleanly into the same Pydantic
    GraphResponse model the API surface uses. Pin so a typo in
    JSON syntax or a schema-drift refactor surfaces here, not
    during a real benchmark run."""
    payload = json.loads((corpus_dir / "expected.json").read_text())
    # Use model_validate so Pydantic surfaces a clear error path
    # on mismatch — "expected_payload[nodes][2].properties:
    # missing field name" is easier to debug than a downstream
    # KeyError.
    GraphResponse.model_validate(payload)


@pytest.mark.parametrize("corpus_dir", _CORPUS_DIRS, ids=[d.name for d in _CORPUS_DIRS])
def test_expected_canonical_ids_match_helper_recomputation(corpus_dir: Path):
    """The load-bearing pin: every expected node's
    ``canonical_id`` (and ``canonical_key``) must equal what
    ``_generate_node_key`` + ``_make_canonical_node_id`` would
    produce given the node's type and the unique-flagged
    properties on the ontology.

    Reviewer-flagged on commit 9e1cd30: pre-fix the seed
    expected.json carried hand-written canonical_ids like
    ``"alice-martinez"`` that diverged from the helper-derived
    UUID5 values. DiffService's "conflicting canonical IDs stay
    unmatched" rule then refused to fall back to canonical_key
    matching, and every node landed as FP+FN. The corpus looked
    correct but every benchmark run reported 0/0 P/R/F1 — a
    silent failure that wasted a lot of debugging time.

    Pin so the next person hand-typing a canonical_id immediately
    sees the test fail with the expected vs computed values
    printed for comparison."""
    ontology = yaml.safe_load((corpus_dir / "ontology.yaml").read_text())
    expected = json.loads((corpus_dir / "expected.json").read_text())

    for node in expected.get("nodes", []):
        node_type = node.get("type")
        properties = node.get("properties", {}) or {}
        stored_canonical_id = properties.get("canonical_id")
        stored_canonical_key = properties.get("canonical_key")

        # Compute what the helper WOULD produce given this
        # node's user-facing properties (canonical_id /
        # canonical_key themselves filtered out so we don't
        # circular-reference into the input).
        identity_props = {
            k: v
            for k, v in properties.items()
            if k not in {"canonical_id", "canonical_key"}
        }
        recomputed_key = _generate_node_key(ontology, node_type, identity_props)
        recomputed_id = _make_canonical_node_id(recomputed_key)

        assert stored_canonical_key == recomputed_key, (
            f"In corpus {corpus_dir.name!r}, node id={node.get('id')!r} "
            f"(type={node_type!r}) has canonical_key="
            f"{stored_canonical_key!r} but the helper would produce "
            f"{recomputed_key!r}. Hand-typing canonical_key drifts "
            "from the live extractor's output; recompute via "
            "_generate_node_key (lowercased, unique-only properties)."
        )
        assert stored_canonical_id == recomputed_id, (
            f"In corpus {corpus_dir.name!r}, node id={node.get('id')!r} "
            f"(type={node_type!r}) has canonical_id="
            f"{stored_canonical_id!r} but the helper would produce "
            f"{recomputed_id!r}. canonical_id is a UUID5 of "
            "canonical_key under the canonical-node namespace; "
            "compute it from the key, don't hand-write a "
            "human-readable form."
        )


@pytest.mark.parametrize("corpus_dir", _CORPUS_DIRS, ids=[d.name for d in _CORPUS_DIRS])
def test_expected_edges_reference_existing_nodes(corpus_dir: Path):
    """No dangling edges in the ground-truth. Pin so a typo in
    a source/target id (referencing a node id that doesn't
    exist) is caught at corpus-load time. Dangling expected
    edges would silently land as FN in a benchmark run — the
    extractor would have to extract them but couldn't possibly
    match, since the expected target doesn't resolve to a real
    expected node."""
    expected = json.loads((corpus_dir / "expected.json").read_text())
    node_ids = {n["id"] for n in expected.get("nodes", [])}
    dangling: list[tuple[str, str]] = []
    for edge in expected.get("edges", []):
        if edge.get("source") not in node_ids:
            dangling.append((edge.get("id"), edge.get("source")))
        if edge.get("target") not in node_ids:
            dangling.append((edge.get("id"), edge.get("target")))
    assert not dangling, (
        f"Corpus {corpus_dir.name!r} has dangling expected edges — "
        "source/target ids reference nodes that aren't in the "
        f"nodes list: {dangling!r}. Either add the missing nodes "
        "or fix the edge endpoints."
    )
