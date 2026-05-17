"""B6-scenario slice 1: Pydantic models for the scenario surface.

A scenario is a named, point-in-time snapshot of a transform's
graph. Slice 1 materializes the full graph in ``graph_snapshot``;
slice 2 may swap to a diff-from-parent layout (CoW) once the
read/write API shape stabilizes — these Pydantic models are
designed to be stable across that storage migration (the wire
shape doesn't change).

The split between :class:`ScenarioSummary` and :class:`Scenario`
mirrors the diff-vs-extraction pattern elsewhere: list endpoints
return summaries (no graph payload — keeps the response small);
the detail endpoint returns the full :class:`Scenario` with the
embedded graph for rendering.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from graphora_server.schemas.graph import GraphResponse


class ScenarioCreateRequest(BaseModel):
    """Request body for ``POST /api/v1/scenarios``.

    ``transform_id`` is the live transform whose current graph is
    snapshotted into the scenario at create time. ``name`` is
    unique per (user, transform) — the DB enforces this so a
    repeat create with the same name surfaces as a 409 rather
    than a silent second row.
    """

    transform_id: str = Field(
        ...,
        description=(
            "Source transform whose current graph is snapshotted "
            "into the new scenario. Must belong to the authenticated "
            "user."
        ),
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Human-readable scenario name. Unique per (user, "
            "transform) — repeat creates with the same name "
            "surface as 409 Conflict."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        max_length=2000,
        description=(
            "Optional free-form description of why this scenario "
            "was created (e.g., 'what-if Alice and Alicia are the "
            "same entity'). Surfaced in list view + detail."
        ),
    )


class ScenarioSummary(BaseModel):
    """Lightweight scenario shape for list endpoints — no graph
    payload so the list view stays fast even when scenarios carry
    large materialized snapshots."""

    id: str
    transform_id: str
    parent_scenario_id: Optional[str] = Field(
        default=None,
        description=(
            "Reserved for slice 2 (branching from another scenario). "
            "Always None in slice 1."
        ),
    )
    name: str
    description: Optional[str] = None
    created_at: datetime
    node_count: int = Field(
        ...,
        description=(
            "Cached node count from the materialized snapshot so "
            "the list view can show graph size without loading the "
            "full snapshot."
        ),
    )
    edge_count: int

    model_config = ConfigDict(from_attributes=True)


class Scenario(ScenarioSummary):
    """Full scenario shape — extends :class:`ScenarioSummary` with
    the embedded graph snapshot for the detail endpoint."""

    graph: GraphResponse = Field(
        ...,
        description="Materialized graph at the time the scenario was created.",
    )
