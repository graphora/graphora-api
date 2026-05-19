"""Runtime capability flags per storage backend.

The C2-postgres review process surfaced a recurring pattern: the
adapter contract has methods that are real on one backend, degraded
to no-op on another, and outright unavailable on a third. Without
a programmatic way to ask "does this backend support X?", callers
either:

  - call blindly and get a no-op (FT indexes on AGE without pg_trgm,
    find_similar_nodes on InMemoryStorage)
  - branch on ``settings.STORAGE_TYPE.lower()`` strings (the pattern
    in build_full_text_indexes_for_user pre-slice-8 review)
  - hit ``NotImplementedError`` at runtime for methods that aren't
    on a critical path but happen to be reached

``BackendCapabilities`` formalizes "what would happen if I called
this method right now?" as a typed surface. Adapters report their
capability set; callers can branch on the flags to skip work that
would degrade or short-circuit, without coupling to backend-name
strings.

Capability set is deliberately small — only flags with actual caller
decisions today. Add new ones when a caller needs them, not
speculatively. The matrix in
``work/Graphora/technical-notes.md::Storage backends`` documents
the human-readable view; this file is the runtime mirror.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class BackendCapabilities:
    """Runtime feature-detection for a graph storage backend.

    Frozen so a backend can return its capabilities directly without
    callers worrying about mutation. Each flag is documented at
    declaration; pick the conservative value when the answer is
    "depends" — callers can opt into the optimistic path.
    """

    persistent: bool
    """Survives process restart. False for in-memory backends; True
    for any backend that writes to disk or a remote database."""

    full_text_indexes: bool
    """``create_or_replace_ft_index_*`` creates a real index that
    accelerates substring search. False means the methods are no-op
    or unavailable; correctness of CONTAINS-style queries is
    preserved (they fall back to scans), only performance differs.

    AGE reports this dynamically from ``_has_pg_trgm`` (set during
    ``_bootstrap_schema``). Before bootstrap runs the flag is False
    — the conservative default so callers that branch on it don't
    create a real index just to discover it was never built. After
    bootstrap the flag matches whether pg_trgm is actually
    installed in the target Postgres instance."""

    similarity_search: bool
    """``find_similar_nodes`` returns real candidate matches. False
    means it returns ``[]`` (callers should treat as "no fuzzy
    fallback")."""

    embedding_similarity: bool
    """Vector-distance similarity is wired (not just property-string
    fuzzy match). False today on every backend; flips True on AGE
    when pgvector wiring + embedding-generation upstream lands, and
    on Neo4j when the vector-index integration ships."""

    per_user_routing: bool
    """``UserDatabaseService`` carries per-user staging/prod
    connection details for this backend (the stagingDb / prodDb
    shape Neo4j uses today). False on AGE (single global
    POSTGRES_AGE_DSN per process) and on in-memory storage."""


# Static capability sets per backend. Adapters return these
# directly via their ``capabilities`` property — keeping them as
# module-level constants makes the capability surface easy to
# read at a glance and trivially diff-able when a backend gains
# or loses a feature.

NEO4J_CAPABILITIES = BackendCapabilities(
    persistent=True,
    full_text_indexes=True,
    similarity_search=True,
    embedding_similarity=False,
    per_user_routing=True,
)

# AGE: ``full_text_indexes`` is runtime-derived from ``_has_pg_trgm``
# (set by _bootstrap_schema based on whether pg_trgm is installed
# in the target Postgres). The PostgresAGEStorage.capabilities
# property constructs the BackendCapabilities instance dynamically
# rather than returning a static constant — see the property impl
# in graphora_server/services/storage/postgres_age.py. Tests verify
# both branches: pg_trgm available → full_text_indexes=True,
# unavailable / pre-bootstrap → False.
#
# Other AGE flags ARE static today: persistence is inherent to
# Postgres, similarity_search via Cypher CONTAINS scoring (slice 6)
# always works, embedding_similarity waits for pgvector wiring
# upstream, per_user_routing waits for multi-tenant Postgres
# extensions to UserDatabaseService.
AGE_STATIC_CAPABILITIES = {
    "persistent": True,
    "similarity_search": True,
    "embedding_similarity": False,
    "per_user_routing": False,
}

# In-memory storage is the dev/demo path. No persistence (lost on
# restart) and no full-text index concept. similarity_search IS
# True — InMemoryStorage.find_similar_nodes runs a real fuzzy
# match via _calculate_similarity, which makes the dev/demo path
# behave like prod for the merge flow's similarity fallback.
MEMORY_CAPABILITIES = BackendCapabilities(
    persistent=False,
    full_text_indexes=False,
    similarity_search=True,
    embedding_similarity=False,
    per_user_routing=False,
)


# ============================================================
# M-matrix — static aggregate view over the per-backend caps.
# ============================================================
#
# The runtime ``capabilities`` property on each adapter answers
# "what does THIS instance support?" (with AGE's pg_trgm-derived
# full_text_indexes flag flipping based on the live Postgres
# state). The matrix below answers the prior question: "what do
# I get if I pick this backend type?" — the operator-facing view
# before any instance exists.
#
# Conservative defaults for dynamic flags. The ``dynamic_flags``
# list tells consumers (frontend matrix page, docs renderer)
# which capability values are NOT guaranteed at runtime; they're
# the safe-floor value, and the actual instance may report
# higher capability after bootstrap. Without this annotation,
# the matrix would either lie (claim AGE supports FT indexes
# unconditionally — wrong when pg_trgm isn't installed) or be
# uselessly pessimistic (claim it never does).
#
# Per-backend ``notes`` carry the human-readable caveats that
# can't be encoded in flags — "dev/demo only", "requires extra
# X", etc. The docs page renders them as footnotes on the
# matrix cells.


@dataclass(frozen=True)
class BackendMatrixEntry:
    """One row in the public capability matrix.

    ``name`` is the ``STORAGE_TYPE`` value that selects the
    backend in ``settings``. ``display_name`` is the
    operator-facing label. ``default_capabilities`` is the
    conservative/static capability set — the safe floor.
    ``dynamic_flags`` lists capability names whose runtime value
    depends on instance detection (currently only AGE's
    ``full_text_indexes``). ``extras`` is the pip extra to
    install, useful for ``pip install 'graphora-server[<extra>]'``
    install hints.
    """

    name: str
    display_name: str
    default_capabilities: BackendCapabilities
    dynamic_flags: Tuple[str, ...] = field(default_factory=tuple)
    notes: Tuple[str, ...] = field(default_factory=tuple)
    extras: Tuple[str, ...] = field(default_factory=tuple)


# AGE's static (conservative) capability snapshot. Mirrors
# AGE_STATIC_CAPABILITIES above but typed as a full
# BackendCapabilities for the matrix view. ``full_text_indexes``
# defaults False (pre-bootstrap or pg_trgm-not-installed); the
# runtime PostgresAGEStorage.capabilities property may flip it
# True. The ``dynamic_flags`` annotation surfaces that to
# matrix consumers.
_AGE_DEFAULT_CAPABILITIES = BackendCapabilities(
    persistent=AGE_STATIC_CAPABILITIES["persistent"],
    full_text_indexes=False,
    similarity_search=AGE_STATIC_CAPABILITIES["similarity_search"],
    embedding_similarity=AGE_STATIC_CAPABILITIES["embedding_similarity"],
    per_user_routing=AGE_STATIC_CAPABILITIES["per_user_routing"],
)


BACKEND_MATRIX: Tuple[BackendMatrixEntry, ...] = (
    BackendMatrixEntry(
        name="neo4j",
        display_name="Neo4j",
        default_capabilities=NEO4J_CAPABILITIES,
        extras=("neo4j",),
        notes=(
            "Reference backend. Full feature support; "
            "per-user staging/prod routing via Neo4jStorage's "
            "stagingDb/prodDb configuration.",
        ),
    ),
    BackendMatrixEntry(
        name="postgres",
        display_name="Postgres + Apache AGE",
        default_capabilities=_AGE_DEFAULT_CAPABILITIES,
        dynamic_flags=("full_text_indexes",),
        extras=("postgres",),
        notes=(
            "full_text_indexes depends on pg_trgm being installed "
            "in the target Postgres instance — the matrix reports "
            "the conservative (False) default; PostgresAGEStorage's "
            "runtime capabilities property reports the live state.",
            "Per-user staging/prod routing isn't wired yet — all "
            "users share POSTGRES_AGE_DSN.",
        ),
    ),
    BackendMatrixEntry(
        name="memory",
        display_name="In-memory (dev/demo)",
        default_capabilities=MEMORY_CAPABILITIES,
        extras=(),
        notes=(
            "Dev/demo path only — process-local store, lost on "
            "restart. Use for quickstart and integration tests; "
            "STORAGE_TYPE=neo4j or =postgres for production.",
        ),
    ),
)


def matrix_entry_by_name(name: str) -> BackendMatrixEntry:
    """Lookup helper used by the API + tests. Raises ValueError on
    unknown names so a typo surfaces rather than returning a
    fallback row that would silently lie about capabilities.
    """
    for entry in BACKEND_MATRIX:
        if entry.name == name:
            return entry
    known = ", ".join(repr(e.name) for e in BACKEND_MATRIX)
    raise ValueError(f"Unknown backend name {name!r}. Known: {known}")
