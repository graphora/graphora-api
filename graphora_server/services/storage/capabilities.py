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

    On AGE this is conservatively reported as the static class
    capability; the method itself runtime-checks
    ``_has_pg_trgm`` (set during ``_bootstrap_schema``) and
    degrades gracefully when pg_trgm isn't installed."""

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

# AGE matches Neo4j on persistence + similarity (CONTAINS scoring,
# slice 6) + full-text (GIN/pg_trgm polyfill, slice 8). It diverges
# on per-user routing (single global DSN until the multi-tenant
# Postgres routing in UserDatabaseService gets wired) and on
# embedding similarity (waiting for pgvector + embedding-generation
# upstream).
AGE_CAPABILITIES = BackendCapabilities(
    persistent=True,
    full_text_indexes=True,
    similarity_search=True,
    embedding_similarity=False,
    per_user_routing=False,
)

# In-memory storage is the dev/demo path. No persistence, no
# indexes, no similarity. Reads + writes work for the duration of
# the process.
MEMORY_CAPABILITIES = BackendCapabilities(
    persistent=False,
    full_text_indexes=False,
    similarity_search=False,
    embedding_similarity=False,
    per_user_routing=False,
)
