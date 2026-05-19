"""B4-bench runner.

Discovers extractor outputs under ``bench/results/<extractor>/<slug>.json``,
loads matching ``golden/<slug>/expected.json`` ground truth, and scores
each pair via :class:`CorpusScorer`. Aggregates per-extractor into a
:class:`BenchRunReport`.

The runner is filesystem-bound by design — the benchmark is meant to
be reproducible by anyone with a checkout of the repo. Production
storage for extractor outputs lives in the repo (not a DB) so the
review-able artifact for a benchmark claim is a commit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from graphora_server.schemas.graph import GraphResponse
from graphora_server.services.bench.models import (
    BenchEntryScore,
    BenchExtractorReport,
    BenchRunReport,
)
from graphora_server.services.golden_corpus import CorpusScorer
from graphora_server.utils.logger import logger


class BenchRunner:
    """Scores each entry under ``bench/results/<extractor>/`` against
    the matching ``golden/<slug>/expected.json``.

    Stateless apart from the injected scorer. The runner accepts an
    explicit ``repo_root`` so tests can point at a synthetic
    filesystem layout without touching the real corpus.
    """

    def __init__(
        self,
        repo_root: Path,
        scorer: Optional[CorpusScorer] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self._scorer = scorer or CorpusScorer()

    # ---- Discovery ----

    def discover_corpus_slugs(self) -> List[str]:
        """Walk ``golden/`` for subdirs containing the trio (matches
        the invariant test's discoverer).
        """
        golden_dir = self.repo_root / "golden"
        if not golden_dir.is_dir():
            return []
        slugs: List[str] = []
        for child in sorted(golden_dir.iterdir()):
            if not child.is_dir():
                continue
            if (
                (child / "document.txt").exists()
                and (child / "ontology.yaml").exists()
                and (child / "expected.json").exists()
            ):
                slugs.append(child.name)
        return slugs

    def discover_extractors(self) -> List[str]:
        """Walk ``bench/results/`` for extractor subdirectories.

        An extractor directory is any non-hidden subdir of
        ``bench/results/``. Empty extractor directories are allowed
        (returns a report with 0 entries) so a slot can be reserved
        before any outputs land — useful for keeping the dashboard
        column visible across runs.
        """
        results_dir = self.repo_root / "bench" / "results"
        if not results_dir.is_dir():
            return []
        names: List[str] = []
        for child in sorted(results_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("."):
                # Skip .gitkeep-like sentinel dirs.
                continue
            names.append(child.name)
        return names

    # ---- Scoring ----

    def score_entry(
        self,
        extractor_name: str,
        corpus_slug: str,
    ) -> BenchEntryScore:
        """Score one (extractor, corpus) pair.

        Loads ``bench/results/<extractor>/<slug>.json`` and
        ``golden/<slug>/expected.json``, then routes through
        :meth:`CorpusScorer.score`. When the extractor output is
        missing or malformed, returns a score with ``error`` set —
        the per-entry error keeps the report informative rather than
        collapsing the failure into a zero F1.
        """
        actual_path = (
            self.repo_root
            / "bench"
            / "results"
            / extractor_name
            / f"{corpus_slug}.json"
        )
        expected_path = self.repo_root / "golden" / corpus_slug / "expected.json"

        if not actual_path.exists():
            return BenchEntryScore(
                corpus_slug=corpus_slug,
                error=f"missing actual output at {actual_path.name}",
            )
        if not expected_path.exists():
            return BenchEntryScore(
                corpus_slug=corpus_slug,
                error=(
                    f"missing expected ground-truth at "
                    f"golden/{corpus_slug}/expected.json"
                ),
            )

        try:
            actual_data = json.loads(actual_path.read_text())
        except json.JSONDecodeError as exc:
            return BenchEntryScore(
                corpus_slug=corpus_slug,
                error=f"actual JSON parse failed: {exc.msg}",
            )
        try:
            expected_data = json.loads(expected_path.read_text())
        except json.JSONDecodeError as exc:
            return BenchEntryScore(
                corpus_slug=corpus_slug,
                error=f"expected JSON parse failed: {exc.msg}",
            )

        try:
            actual = GraphResponse.model_validate(actual_data)
            expected = GraphResponse.model_validate(expected_data)
        except Exception as exc:  # pragma: no cover - schema-shape failures
            # Pydantic ValidationError or similar. Surface the failure
            # rather than crashing the full bench run — a single
            # malformed extractor output shouldn't take down the
            # whole report.
            return BenchEntryScore(
                corpus_slug=corpus_slug,
                error=f"GraphResponse validation failed: {type(exc).__name__}",
            )

        try:
            report = self._scorer.score(
                expected=expected, actual=actual, corpus_slug=corpus_slug
            )
        except Exception as exc:  # pragma: no cover - scorer crash isolation
            logger.exception(
                "Scorer raised for extractor=%s slug=%s: %s",
                extractor_name,
                corpus_slug,
                exc,
            )
            return BenchEntryScore(
                corpus_slug=corpus_slug,
                error=f"scorer raised {type(exc).__name__}",
            )

        return BenchEntryScore(
            corpus_slug=corpus_slug,
            node_precision=report.nodes.precision,
            node_recall=report.nodes.recall,
            node_f1=report.nodes.f1,
            edge_precision=report.edges.precision,
            edge_recall=report.edges.recall,
            edge_f1=report.edges.f1,
            node_true_positives=report.nodes.true_positives,
            node_false_positives=report.nodes.false_positives,
            node_false_negatives=report.nodes.false_negatives,
            edge_true_positives=report.edges.true_positives,
            edge_false_positives=report.edges.false_positives,
            edge_false_negatives=report.edges.false_negatives,
        )

    def run_extractor(self, extractor_name: str) -> BenchExtractorReport:
        """Score one extractor against every corpus entry.

        Iterates every corpus slug. An extractor that lacks output
        for a slug surfaces as an errored entry — the report shows
        the gap rather than silently shrinking the denominator. The
        BenchExtractorReport's aggregate properties only consider
        ``scored_entries`` so missing outputs don't deflate the
        average; coverage is visible via ``errored_count``.
        """
        slugs = self.discover_corpus_slugs()
        entries = [self.score_entry(extractor_name, slug) for slug in slugs]
        return BenchExtractorReport(extractor_name=extractor_name, entries=entries)

    def run(self) -> BenchRunReport:
        """Run the full bench: every extractor × every corpus entry."""
        slugs = self.discover_corpus_slugs()
        extractors = self.discover_extractors()
        reports = [self.run_extractor(name) for name in extractors]
        return BenchRunReport(corpus_size=len(slugs), extractor_reports=reports)
