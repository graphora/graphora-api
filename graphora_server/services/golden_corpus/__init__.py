"""Golden corpus utilities for B4-corpus / B4-test (Gate 4).

This package owns the scoring side of the corpus: given an
``expected.json`` ground truth and an ``actual`` graph from a live
extraction, produce per-type and aggregate precision/recall/F1
scores. The CLI/runner side (B4-test) lives elsewhere — this
package is intentionally pure-Python with no IO beyond reading
the corpus JSON.
"""

from .scorer import (
    CorpusScorer,
    GraphScores,
    ScoringReport,
    TypeScores,
)

__all__ = [
    "CorpusScorer",
    "GraphScores",
    "ScoringReport",
    "TypeScores",
]
