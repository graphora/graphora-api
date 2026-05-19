"""Unit tests for the M-matrix data layer.

Pins the per-row contracts on ``BACKEND_MATRIX``:
  * Every supported STORAGE_TYPE has exactly one matrix row.
  * Per-backend default capabilities match the constants the
    runtime adapters return (so the matrix can't drift from
    what the live ``capabilities`` property reports).
  * ``dynamic_flags`` are real BackendCapabilities field names —
    catches typos that would silently render as "no dynamic
    flags" on the frontend.
  * AGE specifically has ``full_text_indexes`` in dynamic_flags
    and the static default is the conservative False.
  * Lookup helper raises on unknown names rather than returning
    a fallback.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from graphora_server.services.storage.capabilities import (
    AGE_STATIC_CAPABILITIES,
    BACKEND_MATRIX,
    BackendCapabilities,
    BackendMatrixEntry,
    MEMORY_CAPABILITIES,
    NEO4J_CAPABILITIES,
    matrix_entry_by_name,
)


# ============================================================
# Coverage invariants
# ============================================================


def test_matrix_covers_every_storage_type():
    """Every value the factory accepts as STORAGE_TYPE must have
    a matrix row. Pin so a new backend that lands without a
    matrix entry regresses noisily — the matrix is the
    operator-facing "what's available?" view; a missing row
    means the new backend ships invisible."""
    names = {entry.name for entry in BACKEND_MATRIX}
    # The current factory supports: memory, neo4j, postgres.
    # Kuzu lands in A5-kuzu and adds a row at that time.
    expected = {"neo4j", "postgres", "memory"}
    assert names == expected


def test_matrix_rows_are_unique_by_name():
    """Two rows with the same ``name`` would silently shadow
    each other under ``matrix_entry_by_name``. Pin uniqueness
    so a copy-paste bug surfaces here."""
    names = [entry.name for entry in BACKEND_MATRIX]
    assert len(names) == len(set(names))


def test_render_order_starts_with_reference_backend():
    """Frontend matrix renderers display rows in matrix order;
    Neo4j (the reference backend) should be the first row so
    the matrix-page mental anchor matches the documentation
    convention. Memory (dev/demo) should be LAST so operators
    see production options first."""
    assert BACKEND_MATRIX[0].name == "neo4j"
    assert BACKEND_MATRIX[-1].name == "memory"


# ============================================================
# Per-backend capability fidelity
# ============================================================


def test_neo4j_default_matches_runtime_constant():
    """The matrix's Neo4j row must report the same capabilities
    as the runtime ``NEO4J_CAPABILITIES`` constant — i.e., the
    matrix isn't lying about what the live backend supports."""
    entry = matrix_entry_by_name("neo4j")
    assert entry.default_capabilities == NEO4J_CAPABILITIES
    # Neo4j has no runtime-detected flags — every capability is
    # statically determined by the backend.
    assert entry.dynamic_flags == ()


def test_memory_default_matches_runtime_constant():
    entry = matrix_entry_by_name("memory")
    assert entry.default_capabilities == MEMORY_CAPABILITIES
    assert entry.dynamic_flags == ()


def test_postgres_default_uses_conservative_full_text_indexes():
    """AGE's ``full_text_indexes`` depends on pg_trgm being
    installed in the target Postgres. The matrix's static
    default has to commit to one value — pick the
    conservative False, with the dynamic_flags annotation
    telling consumers the live value may flip True after
    bootstrap. Pin both halves of the contract:
      * default_capabilities.full_text_indexes is False
      * dynamic_flags lists full_text_indexes by name

    Without dynamic_flags annotating it, a matrix consumer
    would treat the False as final and misrepresent the
    backend's full capability."""
    entry = matrix_entry_by_name("postgres")
    # Conservative static default.
    assert entry.default_capabilities.full_text_indexes is False
    # ...and dynamic_flags advertises that it may flip at runtime.
    assert "full_text_indexes" in entry.dynamic_flags
    # Other AGE flags match the AGE_STATIC_CAPABILITIES constants
    # — those flags ARE static today, no runtime detection
    # involved.
    assert entry.default_capabilities.persistent == (
        AGE_STATIC_CAPABILITIES["persistent"]
    )
    assert entry.default_capabilities.similarity_search == (
        AGE_STATIC_CAPABILITIES["similarity_search"]
    )
    assert entry.default_capabilities.embedding_similarity == (
        AGE_STATIC_CAPABILITIES["embedding_similarity"]
    )
    assert entry.default_capabilities.per_user_routing == (
        AGE_STATIC_CAPABILITIES["per_user_routing"]
    )


# ============================================================
# dynamic_flags integrity
# ============================================================


def test_dynamic_flag_names_are_real_capability_fields():
    """A typo in dynamic_flags (e.g., 'fulltext_indexes') would
    silently render as "no dynamic flags" on the frontend
    matrix because the lookup wouldn't match. Pin that every
    advertised dynamic flag is an actual
    ``BackendCapabilities`` field name."""
    capability_fields = {f.name for f in fields(BackendCapabilities)}
    for entry in BACKEND_MATRIX:
        for flag in entry.dynamic_flags:
            assert flag in capability_fields, (
                f"{entry.name}: dynamic_flags references "
                f"unknown capability {flag!r}. Known fields: "
                f"{sorted(capability_fields)}"
            )


# ============================================================
# Lookup helper
# ============================================================


def test_lookup_returns_matching_entry():
    entry = matrix_entry_by_name("postgres")
    assert isinstance(entry, BackendMatrixEntry)
    assert entry.name == "postgres"


def test_lookup_raises_on_unknown_name():
    """Pin so a typo surfaces as a hard error rather than a
    silently-returned fallback row. The matrix is small and
    enumerable; an unknown name is always a caller bug."""
    with pytest.raises(ValueError, match="Unknown backend"):
        matrix_entry_by_name("falkordb-typo")


# ============================================================
# Notes + extras
# ============================================================


def test_postgres_notes_mention_pg_trgm_dependency():
    """The dynamic_flags annotation tells callers WHICH flags
    are runtime-detected; the notes should explain WHY for
    each. Pin that the pg_trgm dependency is documented in
    the notes — operators reading the matrix page need to
    know the AGE row's full_text_indexes story has more
    nuance than a single boolean."""
    entry = matrix_entry_by_name("postgres")
    joined = " ".join(entry.notes)
    assert "pg_trgm" in joined


def test_memory_marks_itself_dev_demo():
    """The in-memory backend is for dev/demo only — that's a
    deployment-decision-affecting fact that has to be in the
    notes. Pin so the operator who picks STORAGE_TYPE=memory
    in production doesn't first learn it's wrong from a
    process restart wiping their data."""
    entry = matrix_entry_by_name("memory")
    joined = " ".join(entry.notes).lower()
    assert "dev" in joined or "demo" in joined


def test_extras_align_with_pyproject_extras():
    """The matrix advertises pip extras that operators
    compose into ``pip install 'graphora-server[<extra>]'``.
    Memory needs no extra (empty); Neo4j needs [neo4j];
    Postgres needs [postgres]. Pin these so the install
    hints in any downstream consumer (docs, CLI install
    command, error messages) stay aligned with reality."""
    expected = {
        "neo4j": ("neo4j",),
        "postgres": ("postgres",),
        "memory": (),
    }
    for name, want in expected.items():
        entry = matrix_entry_by_name(name)
        assert (
            entry.extras == want
        ), f"{name}: expected extras={want!r}, got {entry.extras!r}"
