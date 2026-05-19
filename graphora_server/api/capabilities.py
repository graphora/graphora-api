"""M-matrix (Gate 5) — public backend capability matrix endpoint.

Surfaces ``BACKEND_MATRIX`` as a typed API response. The matrix
is the operator-facing "what does each backend support?" view
that lets users pick STORAGE_TYPE without reading source.

Auth posture: **unauthenticated**. The matrix is repo-level
documentation — every value comes from a frozen module-level
constant. No tenant data, no install-specific state. Same
posture as ``/api/v1/bench/run`` for the same reason: the
public reproducibility claim depends on anonymous fetchability.

Wire shape via Pydantic models in ``schemas/capabilities.py``
so OpenAPI exposes the actual matrix shape (per-row capability
flags + dynamic_flags + notes + extras) to generated clients.
"""

from __future__ import annotations

from fastapi import APIRouter

from graphora_server.schemas.capabilities import (
    BackendCapabilities as WireBackendCapabilities,
    BackendMatrixEntry as WireBackendMatrixEntry,
    CapabilityMatrixResponse,
)
from graphora_server.services.storage.capabilities import BACKEND_MATRIX

router = APIRouter(prefix="/api/v1/capabilities", tags=["Capabilities"])


@router.get(
    "/matrix",
    response_model=CapabilityMatrixResponse,
    description=(
        "M-matrix (Gate 5) — feature-compatibility matrix across "
        "all supported storage backends. Returns per-backend "
        "default capabilities + the list of dynamic flags whose "
        "runtime value depends on instance detection (e.g., "
        "Postgres+AGE's ``full_text_indexes`` depends on pg_trgm "
        "being installed). Unauthenticated — the matrix is "
        "repo-level documentation."
    ),
)
async def get_capability_matrix() -> CapabilityMatrixResponse:
    """Project ``BACKEND_MATRIX`` into the wire shape.

    The service-layer constants are frozen dataclasses; the API
    constructs Pydantic models from them. Order is preserved —
    matrix renderers depend on it (reference backend first,
    then alternatives, then dev/demo).
    """
    backends = [
        WireBackendMatrixEntry(
            name=entry.name,
            display_name=entry.display_name,
            default_capabilities=WireBackendCapabilities(
                persistent=entry.default_capabilities.persistent,
                full_text_indexes=entry.default_capabilities.full_text_indexes,
                similarity_search=entry.default_capabilities.similarity_search,
                embedding_similarity=entry.default_capabilities.embedding_similarity,
                per_user_routing=entry.default_capabilities.per_user_routing,
            ),
            dynamic_flags=list(entry.dynamic_flags),
            notes=list(entry.notes),
            extras=list(entry.extras),
        )
        for entry in BACKEND_MATRIX
    ]
    return CapabilityMatrixResponse(backends=backends)
