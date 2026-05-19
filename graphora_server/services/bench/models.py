"""Dataclasses for the B4-bench surface.

Three nested levels:
  * :class:`BenchEntryScore` — one extractor's score on one corpus entry.
  * :class:`BenchExtractorReport` — aggregated scores across all corpus
    entries for one extractor (one column in the benchmark report).
  * :class:`BenchRunReport` — the full report covering every extractor
    discovered under ``bench/results/``.

The dataclasses are plain Python (no Pydantic) to keep them
unit-testable without framework ceremony and to mirror the
ScoringReport convention in
``graphora_server.services.golden_corpus.scorer``. The API layer
projects to JSON dicts via ``to_dict()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchEntryScore:
    """One extractor's score on one corpus entry.

    ``error`` is set when the runner couldn't score this entry (e.g.,
    missing actual.json or malformed shape). When non-None, the
    numeric fields are zeroed and the entry is excluded from the
    extractor's aggregate. Surfacing per-entry errors lets the
    report distinguish "extractor failed on this doc" from
    "extractor scored 0 on this doc" — both look like zero F1 if
    silently bucketed together.
    """

    corpus_slug: str
    node_precision: float = 0.0
    node_recall: float = 0.0
    node_f1: float = 0.0
    edge_precision: float = 0.0
    edge_recall: float = 0.0
    edge_f1: float = 0.0
    node_true_positives: int = 0
    node_false_positives: int = 0
    node_false_negatives: int = 0
    edge_true_positives: int = 0
    edge_false_positives: int = 0
    edge_false_negatives: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "corpus_slug": self.corpus_slug,
            "node_precision": self.node_precision,
            "node_recall": self.node_recall,
            "node_f1": self.node_f1,
            "edge_precision": self.edge_precision,
            "edge_recall": self.edge_recall,
            "edge_f1": self.edge_f1,
            "node_true_positives": self.node_true_positives,
            "node_false_positives": self.node_false_positives,
            "node_false_negatives": self.node_false_negatives,
            "edge_true_positives": self.edge_true_positives,
            "edge_false_positives": self.edge_false_positives,
            "edge_false_negatives": self.edge_false_negatives,
            "error": self.error,
        }


@dataclass
class BenchExtractorReport:
    """Aggregated scores for one extractor across the full corpus.

    The ``entries`` list carries per-entry detail; aggregate
    properties compute micro-averaged P/R/F1 across all entries
    that produced a score (entries with errors are excluded from
    the denominators per the surfacing-vs-burying principle in
    BenchEntryScore).

    Macro-averaged F1 (mean of per-entry F1) is reported alongside
    micro for the headline number — micro weights by entity count
    (a 100-node doc dominates a 5-node doc), macro treats each doc
    equally. Both are useful; consumers can pick.
    """

    extractor_name: str
    entries: List[BenchEntryScore] = field(default_factory=list)

    @property
    def scored_entries(self) -> List[BenchEntryScore]:
        return [e for e in self.entries if e.error is None]

    @property
    def errored_entries(self) -> List[BenchEntryScore]:
        return [e for e in self.entries if e.error is not None]

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    # ---- Micro-averaged scores (weighted by entity count) ----

    @property
    def micro_node_precision(self) -> float:
        tp = sum(e.node_true_positives for e in self.scored_entries)
        fp = sum(e.node_false_positives for e in self.scored_entries)
        denom = tp + fp
        return tp / denom if denom > 0 else 0.0

    @property
    def micro_node_recall(self) -> float:
        tp = sum(e.node_true_positives for e in self.scored_entries)
        fn = sum(e.node_false_negatives for e in self.scored_entries)
        denom = tp + fn
        return tp / denom if denom > 0 else 0.0

    @property
    def micro_node_f1(self) -> float:
        p = self.micro_node_precision
        r = self.micro_node_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def micro_edge_precision(self) -> float:
        tp = sum(e.edge_true_positives for e in self.scored_entries)
        fp = sum(e.edge_false_positives for e in self.scored_entries)
        denom = tp + fp
        return tp / denom if denom > 0 else 0.0

    @property
    def micro_edge_recall(self) -> float:
        tp = sum(e.edge_true_positives for e in self.scored_entries)
        fn = sum(e.edge_false_negatives for e in self.scored_entries)
        denom = tp + fn
        return tp / denom if denom > 0 else 0.0

    @property
    def micro_edge_f1(self) -> float:
        p = self.micro_edge_precision
        r = self.micro_edge_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # ---- Macro-averaged scores (mean across entries) ----

    @property
    def macro_node_f1(self) -> float:
        scored = self.scored_entries
        if not scored:
            return 0.0
        return sum(e.node_f1 for e in scored) / len(scored)

    @property
    def macro_edge_f1(self) -> float:
        scored = self.scored_entries
        if not scored:
            return 0.0
        return sum(e.edge_f1 for e in scored) / len(scored)

    def to_dict(self) -> Dict[str, object]:
        return {
            "extractor_name": self.extractor_name,
            "total_entries": self.total_entries,
            "scored_count": len(self.scored_entries),
            "errored_count": len(self.errored_entries),
            "micro_node_precision": self.micro_node_precision,
            "micro_node_recall": self.micro_node_recall,
            "micro_node_f1": self.micro_node_f1,
            "micro_edge_precision": self.micro_edge_precision,
            "micro_edge_recall": self.micro_edge_recall,
            "micro_edge_f1": self.micro_edge_f1,
            "macro_node_f1": self.macro_node_f1,
            "macro_edge_f1": self.macro_edge_f1,
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass
class BenchRunReport:
    """Top-level report. One row per extractor under bench/results/."""

    corpus_size: int
    extractor_reports: List[BenchExtractorReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "corpus_size": self.corpus_size,
            "extractor_count": len(self.extractor_reports),
            "extractors": [r.to_dict() for r in self.extractor_reports],
        }
