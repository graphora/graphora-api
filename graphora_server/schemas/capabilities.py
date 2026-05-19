"""M-matrix Pydantic wire models for the public capability matrix.

Mirrors the service-layer dataclasses in
``graphora_server.services.storage.capabilities`` field-for-field
so the OpenAPI schema exposes the matrix shape to generated
clients (frontend dashboard, docs site renderer, third-party
integrations).

Mirror-the-dataclass split convention — same as the bench and
contradictions surfaces — keeps the service layer Pydantic-free
(cheap test construction, no framework coupling) while the wire
contract still carries typed field constraints.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class BackendCapabilities(BaseModel):
    """Runtime-or-static feature flags for a graph storage backend.

    Boolean flags only — each one answers "would calling this
    capability today succeed?" with True/False. The matrix view
    reports the static default; live instances may diverge for
    flags listed in the parent ``BackendMatrixEntry.dynamic_flags``.
    """

    persistent: bool = Field(
        ...,
        description=(
            "True if the backend survives process restart. False "
            "for in-memory; True for any backend that writes to "
            "disk or a remote database."
        ),
    )
    full_text_indexes: bool = Field(
        ...,
        description=(
            "True if create_or_replace_ft_index_* creates a real "
            "index. False means the methods are no-op or "
            "unavailable; CONTAINS-style query correctness is "
            "preserved (scan fallback), only performance differs."
        ),
    )
    similarity_search: bool = Field(
        ...,
        description=(
            "True if find_similar_nodes returns real candidate "
            "matches. False means it returns an empty list — "
            "callers should treat as 'no fuzzy fallback'."
        ),
    )
    embedding_similarity: bool = Field(
        ...,
        description=(
            "True if vector-distance similarity is wired (not "
            "just property-string fuzzy match). False on every "
            "backend today; flips True per-backend as embedding "
            "wiring lands upstream."
        ),
    )
    per_user_routing: bool = Field(
        ...,
        description=(
            "True if UserDatabaseService carries per-user "
            "staging/prod connection details for this backend. "
            "False on single-tenant Postgres and in-memory."
        ),
    )

    model_config = ConfigDict(from_attributes=True)


class BackendMatrixEntry(BaseModel):
    """One row in the M-matrix.

    ``name`` matches the ``STORAGE_TYPE`` setting that selects the
    backend. ``dynamic_flags`` lists which capability fields are
    NOT guaranteed by the static default — the runtime adapter
    instance's ``capabilities`` property may report higher than
    the matrix-reported default. ``extras`` is the pip extra to
    install (empty tuple = no extra needed).
    """

    name: str = Field(
        ...,
        description=(
            "STORAGE_TYPE value that selects this backend "
            "(e.g., 'neo4j', 'postgres', 'memory')."
        ),
    )
    display_name: str = Field(
        ...,
        description="Operator-facing label for the backend.",
    )
    default_capabilities: BackendCapabilities = Field(
        ...,
        description=(
            "Conservative/static capability set — the safe floor. "
            "Live instances may report higher values for fields "
            "listed in ``dynamic_flags``."
        ),
    )
    dynamic_flags: List[str] = Field(
        default_factory=list,
        description=(
            "Capability field names whose runtime value depends "
            "on instance detection (e.g., AGE's "
            "``full_text_indexes`` depends on pg_trgm install). "
            "Frontend matrix renderers should annotate these "
            "cells as 'depends on runtime detection'."
        ),
    )
    notes: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable caveats that don't fit in boolean "
            "flags (deployment requirements, multi-tenant gaps, "
            "etc.). Rendered as footnotes on the matrix page."
        ),
    )
    extras: List[str] = Field(
        default_factory=list,
        description=(
            "Pip extras required to install this backend. "
            "Compose into ``pip install 'graphora-server[X]'`` "
            "for the install hint. Empty list = no extra needed."
        ),
    )

    model_config = ConfigDict(from_attributes=True)


class CapabilityMatrixResponse(BaseModel):
    """Wire envelope for ``GET /api/v1/capabilities/matrix``.

    The list is ordered the way the matrix table should render —
    reference backend (Neo4j) first, then alternatives, then
    dev/demo. Frontend consumers should preserve order rather
    than re-sort.
    """

    backends: List[BackendMatrixEntry] = Field(
        ...,
        description="Per-backend capability rows; render order matters.",
    )

    model_config = ConfigDict(from_attributes=True)
