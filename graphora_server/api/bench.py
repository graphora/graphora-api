"""B4-bench (Gate 4) — public benchmark report endpoint.

Returns the latest bench run, scored from extractor outputs
committed to ``bench/results/<extractor>/<slug>.json`` against the
golden corpus's ``golden/<slug>/expected.json`` ground truth.

The benchmark is public-by-design — the numbers are derived from
files committed to the repo, so anyone with a checkout can
reproduce them. This is the load-bearing claim for the B4-bench
exit signal: external reviewers can run the same calculation and
get matching scores. The endpoint just renders what's on disk; it
doesn't run extractions.

Auth posture: the endpoint is **unauthenticated**. Bench numbers
are public information — the marketing claim is that anyone can
verify them — so there's nothing tenant-scoped to enforce. The
underlying files are repo artifacts, not user data.

Wire shape: the route declares ``response_model=BenchRunReport``
(Pydantic) so the OpenAPI snapshot exposes the actual response
fields — corpus_size, extractor aggregates, per-entry detail —
to generated clients. Reviewer-flagged Medium on commit 06fc210:
pre-fix the route returned ``Dict[str, Any]`` with no
``response_model``, so OpenAPI consumers saw a permissive
``{"additionalProperties": true, "type": "object"}`` shape that
couldn't catch wire-shape regressions.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from graphora_server.schemas.bench import BenchRunReport
from graphora_server.services.bench import BenchRunner

router = APIRouter(prefix="/api/v1/bench", tags=["Bench"])

# The repo root is two levels up from this file
# (graphora_server/api/bench.py → graphora_server/api → graphora_server →
# repo root). Computed once at import time; the runner reads from
# disk on every request so swapping out files between requests works
# without a process restart.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@router.get(
    "/run",
    response_model=BenchRunReport,
    description=(
        "B4-bench public benchmark report. Returns per-extractor "
        "aggregate scores plus per-corpus-entry breakdowns. The "
        "scores are computed from extractor outputs committed under "
        "``bench/results/`` against the golden corpus ground truth — "
        "anyone with a checkout can re-run the same computation "
        "locally and get matching numbers. The response is "
        "unauthenticated because the data is by definition public."
    ),
)
async def run_bench() -> BenchRunReport:
    """Score every (extractor, corpus entry) pair on disk.

    Discovery walks ``bench/results/`` for extractor subdirectories
    and ``golden/`` for corpus entries. An extractor that lacks
    output for a given entry shows up as an ``errored`` per-entry
    record with a human-readable reason — keeps coverage visible
    instead of silently shrinking the denominator. Aggregate
    micro/macro F1 only counts the entries that produced a score.
    """
    runner = BenchRunner(repo_root=_REPO_ROOT)
    report = runner.run()
    # The service-layer dataclass's ``to_dict()`` produces the
    # same shape as the Pydantic model's field set, so
    # ``model_validate`` cleanly projects between the two without
    # requiring a manual field-by-field rebuild. Pin the service
    # layer staying dataclass-based (no Pydantic import there) so
    # tests don't pay Pydantic-construction cost.
    return BenchRunReport.model_validate(report.to_dict())
