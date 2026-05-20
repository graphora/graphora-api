"""Sample gallery — public demo surface (no auth).

Renders the on-disk ``golden/<slug>/`` entries as a browsable
gallery. The point of the surface is to give visitors a
production-quality preview of what Graphora extracts WITHOUT
requiring them to sign up + provide an LLM API key — the
single biggest friction point on the dashboard signup flow
(2026-05-20 conversation: visitors sign up but bounce when
asked for a key).

Same auth posture as ``/api/v1/bench/run`` and
``/api/v1/capabilities/matrix``: content is public
documentation (the golden corpus is committed in the repo),
no tenant scoping needed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from fastapi import APIRouter, HTTPException

from graphora_server.schemas.samples import (
    SampleDetail,
    SampleSummary,
    SamplesListResponse,
)

router = APIRouter(prefix="/api/v1/samples", tags=["Samples"])

# Repo root: graphora_server/api/samples.py → graphora_server/api →
# graphora_server → repo. parents[2] = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _REPO_ROOT / "golden"
_ROSTER_FILE = _GOLDEN_DIR / "README.md"


def _parse_roster_domains(roster_path: Path) -> Dict[str, str]:
    """Walk the golden/README.md roster table and extract the
    slug → domain mapping.

    The roster is a markdown table:
        | `slug` | Domain | Pattern | Entity types | Edge types |
        |---|---|---|---|---|
        | `single_person_works_at_org` | Business | ... |

    We pick rows where the first cell starts with a backticked
    identifier. Domain is the second column. Robust to the
    "table border with hyphens" row and any rows that don't
    match the expected shape.

    Cached because the README is a static repo file — only
    re-read on process restart.
    """
    if not roster_path.is_file():
        return {}
    mapping: Dict[str, str] = {}
    for line in roster_path.read_text().splitlines():
        # Skip non-data rows (headers, separators, surrounding
        # prose). The data rows we want have a backticked slug
        # in the first cell.
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        # cells[0] is empty (leading |), cells[-1] is empty
        # (trailing |). Skip rows that don't have at least
        # 5 cells (slug + 4 attrs).
        if len(cells) < 6:
            continue
        slug_cell = cells[1]
        # Backticked slug — strip the backticks. The "---"
        # separator row's first cell is just hyphens; falls
        # through here as it won't have backticks.
        if not (slug_cell.startswith("`") and slug_cell.endswith("`")):
            continue
        slug = slug_cell.strip("`").strip()
        domain = cells[2]
        if slug and domain:
            mapping[slug] = domain
    return mapping


def _slug_to_display(slug: str) -> str:
    """``single_person_works_at_org`` → ``Single Person Works At Org``."""
    return slug.replace("_", " ").title()


@lru_cache(maxsize=1)
def _load_all_samples() -> Tuple[Dict[str, SampleSummary], Dict[str, SampleDetail]]:
    """Walk ``golden/`` once, build the summary index + detail
    cache.

    Cached for process lifetime. Samples are committed to the
    repo and don't change at runtime; rebuilding the cache
    requires a process restart. If a sample fails to load
    (bad JSON, missing trio member), it's skipped with a
    silent omission — better to ship a smaller gallery than
    to 500 the endpoint.
    """
    summaries: Dict[str, SampleSummary] = {}
    details: Dict[str, SampleDetail] = {}

    if not _GOLDEN_DIR.is_dir():
        return summaries, details

    domain_map = _parse_roster_domains(_ROSTER_FILE)

    for entry in sorted(_GOLDEN_DIR.iterdir()):
        if not entry.is_dir():
            continue
        slug = entry.name
        document_path = entry / "document.txt"
        ontology_path = entry / "ontology.yaml"
        expected_path = entry / "expected.json"
        readme_path = entry / "README.md"

        # Skip incomplete corpus entries — the invariant tests
        # pin the trio existence, but the API should be
        # defensive in case a future refactor drops a file.
        if not all(p.is_file() for p in (document_path, ontology_path, expected_path)):
            continue

        try:
            document = document_path.read_text()
            ontology_yaml = ontology_path.read_text()
            expected = json.loads(expected_path.read_text())
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue

        nodes = expected.get("nodes") or []
        edges = expected.get("edges") or []
        domain = domain_map.get(slug, "Other")
        display_name = _slug_to_display(slug)

        summaries[slug] = SampleSummary(
            slug=slug,
            display_name=display_name,
            domain=domain,
            node_count=len(nodes),
            edge_count=len(edges),
        )
        readme_content: Optional[str] = None
        if readme_path.is_file():
            try:
                readme_content = readme_path.read_text()
            except OSError:
                readme_content = None
        details[slug] = SampleDetail(
            slug=slug,
            display_name=display_name,
            domain=domain,
            document=document,
            ontology_yaml=ontology_yaml,
            expected_graph=expected,
            readme_markdown=readme_content,
        )

    return summaries, details


@router.get(
    "",
    response_model=SamplesListResponse,
    description=(
        "Lightweight gallery list — slug + display_name + "
        "domain + node/edge counts per sample. Use "
        "``GET /samples/{slug}`` for the full document + "
        "ontology + graph payload. Unauthenticated."
    ),
)
async def list_samples() -> SamplesListResponse:
    summaries_map, _ = _load_all_samples()
    # Preserve directory order (matches sorted slug order).
    # Domain order: alphabetical for predictable filter-chip
    # rendering.
    samples = sorted(summaries_map.values(), key=lambda s: s.slug)
    domains = sorted({s.domain for s in samples})
    return SamplesListResponse(samples=samples, domains=domains)


@router.get(
    "/{slug}",
    response_model=SampleDetail,
    description=(
        "Full payload for a single sample: source document, "
        "ontology yaml, extracted graph (in GraphResponse "
        "shape — frontend renders via the same graph-viz "
        "component the Explorer uses), and the README "
        "markdown explaining what extraction patterns the "
        "sample exercises. Unauthenticated."
    ),
    responses={404: {"description": "No sample with that slug"}},
)
async def get_sample(slug: str) -> SampleDetail:
    _, details_map = _load_all_samples()
    if slug not in details_map:
        raise HTTPException(status_code=404, detail=f"No sample with slug {slug!r}")
    return details_map[slug]
