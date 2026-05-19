"""B4-bench (Gate 4) — benchmark runner over the golden corpus.

Reads per-extractor output files from ``bench/results/<extractor>/<slug>.json``,
scores each against the matching ``golden/<slug>/expected.json`` via
:class:`CorpusScorer`, and aggregates the per-entry reports into a
single :class:`BenchRunReport`.

See ``bench/README.md`` for the file-format spec.
"""

from graphora_server.services.bench.models import (
    BenchEntryScore,
    BenchExtractorReport,
    BenchRunReport,
)
from graphora_server.services.bench.runner import BenchRunner

__all__ = [
    "BenchEntryScore",
    "BenchExtractorReport",
    "BenchRunReport",
    "BenchRunner",
]
