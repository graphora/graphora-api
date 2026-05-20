"""Sample gallery wire models — public demo surface.

Renders the golden-corpus entries as a public, no-auth gallery
so visitors can see what Graphora extracts before committing to
sign up + provide an API key. Sample content is the same
``golden/<slug>/`` entries the benchmark uses; this surface
re-projects them for operator-facing browsing.

Auth posture: **unauthenticated**. Same convention as
``/bench/run`` and ``/capabilities/matrix`` — content is public
documentation, no tenant scoping needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SampleSummary(BaseModel):
    """Lightweight entry for the gallery grid.

    The list endpoint returns these without the heavyweight
    document/ontology/graph content so an unauthed gallery view
    can render fast and small. ``GET /samples/{slug}`` fetches
    the full payload for the detail page.
    """

    slug: str = Field(
        ...,
        description=(
            "Stable identifier matching the ``golden/<slug>/`` "
            "directory. Used in /samples/{slug} URLs and as the "
            "react ``key`` on the gallery grid."
        ),
    )
    display_name: str = Field(
        ...,
        description=(
            "Human-readable title derived from the slug " "(snake_case → Title Case)."
        ),
    )
    domain: str = Field(
        ...,
        description=(
            "Subject domain — used by the gallery's filter "
            "chips. Examples: Healthcare, Legal, Finance, "
            "Software. Sourced from the golden/README.md "
            "roster table."
        ),
    )
    node_count: int = Field(..., ge=0)
    edge_count: int = Field(..., ge=0)

    model_config = ConfigDict(from_attributes=True)


class SampleDetail(BaseModel):
    """Full sample payload for the detail-page view.

    Carries the source document, the ontology used for
    extraction, the resulting graph, and the README content
    (which explains what the sample exercises — useful as the
    "what this demonstrates" panel on the detail page).
    """

    slug: str
    display_name: str
    domain: str

    document: str = Field(
        ..., description="Source document text the extraction ran against."
    )
    ontology_yaml: str = Field(
        ...,
        description=(
            "Ontology used for this extraction — the schema "
            "the extractor was given. Rendered in a code block "
            "alongside the graph so visitors can see the "
            "input-side of the contract."
        ),
    )
    expected_graph: Dict[str, Any] = Field(
        ...,
        description=(
            "Extracted graph in GraphResponse shape (nodes + "
            "edges). Frontend renders via the same graph-viz "
            "component the real Explorer uses."
        ),
    )
    readme_markdown: Optional[str] = Field(
        default=None,
        description=(
            "Markdown content of the sample's README — explains "
            "what extraction patterns this entry exercises. "
            "Rendered on the detail page as the 'about this "
            "sample' panel."
        ),
    )

    model_config = ConfigDict(from_attributes=True)


class SamplesListResponse(BaseModel):
    """Wire envelope for ``GET /api/v1/samples``."""

    samples: List[SampleSummary]
    domains: List[str] = Field(
        ...,
        description=(
            "Unique domain names across all samples — for "
            "populating the gallery's filter chip list without "
            "the frontend having to derive them from the "
            "samples array."
        ),
    )

    model_config = ConfigDict(from_attributes=True)
