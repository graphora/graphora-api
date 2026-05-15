"""Precision / Recall / F1 scoring for golden-corpus extractions.

Wraps the existing ``DiffService`` (B3-diff) so identity matching
between expected and actual graphs uses the SAME canonical_id /
type+canonical_key cascade the live diff surface uses — no
duplicate matching logic. The scorer then converts the diff
summary into standard P/R/F1 numbers per entity/edge type and in
aggregate.

The diff perspective:
  * ``base`` = expected graph (ground truth from expected.json)
  * ``compare`` = actual graph (live extraction output)
  * ``added`` (compare \\ base)  → false positives (FP)
  * ``removed`` (base \\ compare) → false negatives (FN)
  * ``unchanged`` (in both, same props)   → true positives (TP)
  * ``changed`` (in both, different props) → matched-identity but
    partial credit; the aggregate counts them as TP for identity
    P/R, and the per-property scoring lands in a future slice.

Aggregate scoring follows the conventional micro-F1 definition:
``precision = TP / (TP + FP)``, ``recall = TP / (TP + FN)``,
``F1 = 2PR / (P + R)``. Zero denominators yield ``0.0`` rather
than NaN so the report serializes cleanly.

The scorer is deliberately stateless and IO-free — it takes two
``GraphResponse`` payloads and returns a structured ``ScoringReport``.
Loading expected.json from disk + running an extraction belongs to
B4-test's runner (separate slice).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

from graphora_server.schemas.graph import GraphResponse
from graphora_server.services.diff_service import DiffService


@dataclass
class TypeScores:
    """P/R/F1 for a single entity or edge type."""

    type_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        if denom == 0:
            # No predictions for this type — by convention precision
            # is undefined; return 0.0 to keep the report numeric.
            # Callers can distinguish "0.0 because everything wrong"
            # from "0.0 because nothing predicted" by also reading
            # true_positives + false_positives directly.
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            # No ground-truth instances of this type — recall is
            # undefined. Same convention as precision.
            return 0.0
        return self.true_positives / denom

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class GraphScores:
    """Aggregate + per-type scores for one side of the graph
    (nodes or edges)."""

    by_type: Dict[str, TypeScores] = field(default_factory=dict)

    @property
    def true_positives(self) -> int:
        return sum(t.true_positives for t in self.by_type.values())

    @property
    def false_positives(self) -> int:
        return sum(t.false_positives for t in self.by_type.values())

    @property
    def false_negatives(self) -> int:
        return sum(t.false_negatives for t in self.by_type.values())

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class ScoringReport:
    """Top-level scoring report for a corpus document.

    ``corpus_slug`` is the golden/<slug> directory name; the runner
    populates it. The scorer itself doesn't read disk and so leaves
    this empty by default — callers tag the report with their slug
    when emitting batch results."""

    nodes: GraphScores = field(default_factory=GraphScores)
    edges: GraphScores = field(default_factory=GraphScores)
    corpus_slug: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Plain-dict serialization for JSON output. Includes the
        derived P/R/F1 fields alongside the raw counts so report
        consumers (CI, dashboards) don't need to recompute them."""
        return {
            "corpus_slug": self.corpus_slug,
            "nodes": _graph_scores_to_dict(self.nodes),
            "edges": _graph_scores_to_dict(self.edges),
        }


def _graph_scores_to_dict(gs: GraphScores) -> Dict[str, object]:
    return {
        "true_positives": gs.true_positives,
        "false_positives": gs.false_positives,
        "false_negatives": gs.false_negatives,
        "precision": gs.precision,
        "recall": gs.recall,
        "f1": gs.f1,
        "by_type": {
            name: {
                "true_positives": t.true_positives,
                "false_positives": t.false_positives,
                "false_negatives": t.false_negatives,
                "precision": t.precision,
                "recall": t.recall,
                "f1": t.f1,
            }
            for name, t in gs.by_type.items()
        },
    }


class CorpusScorer:
    """Compute P/R/F1 for an extraction against a golden expected
    graph.

    Stateless — every call goes through ``score()`` which wraps a
    fresh ``DiffService`` invocation. The diff service is the
    identity-matching authority; this class is only responsible
    for turning matched/added/removed counts into scores.
    """

    def __init__(self, diff_service: Optional[DiffService] = None) -> None:
        # Injectable for tests, defaults to a fresh instance.
        # DiffService itself is stateless, so a per-call new()
        # would also work — the kwarg is purely for test seam
        # symmetry with other services in this codebase.
        self._diff = diff_service or DiffService()

    def score(
        self,
        expected: GraphResponse,
        actual: GraphResponse,
        *,
        corpus_slug: Optional[str] = None,
    ) -> ScoringReport:
        """Score ``actual`` against ``expected``.

        Returns a :class:`ScoringReport` carrying aggregate and
        per-type P/R/F1 numbers plus the underlying TP/FP/FN
        counts. ``corpus_slug`` is echoed onto the report for
        downstream batch aggregation.
        """
        diff = self._diff.diff(
            base_graph=expected,
            compare_graph=actual,
            base_transform_id="<expected>",
            compare_transform_id="<actual>",
        )

        # NODES -----------------------------------------------------
        node_scores: Dict[str, TypeScores] = defaultdict(
            lambda: TypeScores(type_name="")
        )

        # added_nodes are in compare but not base → FP per type.
        for n in diff.added_nodes:
            t = node_scores[n.type]
            t.type_name = n.type
            t.false_positives += 1
        # removed_nodes are in base but not compare → FN per type.
        for n in diff.removed_nodes:
            t = node_scores[n.type]
            t.type_name = n.type
            t.false_negatives += 1
        # changed_nodes were identity-matched on both sides — count
        # as TP for the IDENTITY level. Property-level scoring is
        # a future refinement; for now the diff already exposes the
        # property_changes for downstream consumers.
        for delta in diff.changed_nodes:
            t = node_scores[delta.type]
            t.type_name = delta.type
            t.true_positives += 1
        # unchanged_nodes don't appear in the diff payload at the
        # detail level — only as a count in the summary. We can
        # split them per-type because identity matching went through
        # ``_match_nodes`` and the matched-and-property-identical
        # pairs are exactly the ones absent from changed_nodes. The
        # summary already tracks this:
        #   total unchanged = summary.nodes_unchanged
        # but type breakdown isn't surfaced by the diff payload.
        # Recover per-type unchanged by subtracting:
        #   unchanged[type] = (instances in expected of type)
        #                     - removed[type] - changed[type]
        # which equals (instances in actual of type)
        #                     - added[type] - changed[type]. We use
        # the expected-side count so unchanged tracks ground truth
        # (consistent with how recall denominators are computed).
        expected_by_type: Dict[str, int] = defaultdict(int)
        for n in expected.nodes:
            expected_by_type[n.type] += 1
        for type_name, expected_count in expected_by_type.items():
            t = node_scores[type_name]
            t.type_name = type_name
            removed = sum(1 for n in diff.removed_nodes if n.type == type_name)
            changed = sum(1 for d in diff.changed_nodes if d.type == type_name)
            unchanged = expected_count - removed - changed
            t.true_positives += unchanged

        # EDGES -----------------------------------------------------
        edge_scores: Dict[str, TypeScores] = defaultdict(
            lambda: TypeScores(type_name="")
        )

        for e in diff.added_edges:
            edge_type = getattr(e, "type", "unknown") or "unknown"
            t = edge_scores[edge_type]
            t.type_name = edge_type
            t.false_positives += 1
        for e in diff.removed_edges:
            edge_type = getattr(e, "type", "unknown") or "unknown"
            t = edge_scores[edge_type]
            t.type_name = edge_type
            t.false_negatives += 1
        for delta in diff.changed_edges:
            t = edge_scores[delta.type]
            t.type_name = delta.type
            t.true_positives += 1
        # Same unchanged-recovery trick for edges, keyed on edge type.
        expected_edges_by_type: Dict[str, int] = defaultdict(int)
        for e in expected.edges:
            edge_type = getattr(e, "type", "unknown") or "unknown"
            expected_edges_by_type[edge_type] += 1
        for type_name, expected_count in expected_edges_by_type.items():
            t = edge_scores[type_name]
            t.type_name = type_name
            removed = sum(
                1
                for e in diff.removed_edges
                if (getattr(e, "type", "unknown") or "unknown") == type_name
            )
            changed = sum(1 for d in diff.changed_edges if d.type == type_name)
            unchanged = expected_count - removed - changed
            t.true_positives += unchanged

        return ScoringReport(
            nodes=GraphScores(by_type=dict(node_scores)),
            edges=GraphScores(by_type=dict(edge_scores)),
            corpus_slug=corpus_slug,
        )
