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

from dataclasses import dataclass


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
