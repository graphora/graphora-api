"""Tests for the per-backend capability flags (M-matrix).

The capability set is small and the values are static per backend,
so the tests are mostly contract pins: each adapter exposes the
``capabilities`` property and returns the documented constants.
Tests fail loudly if a backend's capability set drifts from the
matrix in graphora_server/services/storage/capabilities.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphora_server.services.storage.capabilities import (
    AGE_CAPABILITIES,
    MEMORY_CAPABILITIES,
    NEO4J_CAPABILITIES,
    BackendCapabilities,
)


class TestBackendCapabilitiesShape:
    """Pin the dataclass shape — adding a field elsewhere without
    updating these tests + the per-backend constants is the kind
    of drift the matrix exists to prevent."""

    def test_is_frozen(self) -> None:
        # Frozen dataclass — backends return shared constants.
        # Mutating one would silently change behaviour for every
        # callsite using that backend.
        caps = BackendCapabilities(
            persistent=True,
            full_text_indexes=True,
            similarity_search=True,
            embedding_similarity=False,
            per_user_routing=True,
        )
        with pytest.raises(Exception):  # FrozenInstanceError on Pydantic / dataclass
            caps.persistent = False  # type: ignore[misc]

    def test_required_fields(self) -> None:
        # Forces every backend to think about every flag — the
        # whole point of the matrix is "no defaults, no implicit
        # answers." Adding a new field to the dataclass MUST come
        # with explicit values on every constant, which this test
        # implicitly enforces (any backend missing the field
        # raises TypeError on construction).
        with pytest.raises(TypeError):
            BackendCapabilities(persistent=True)  # type: ignore[call-arg]


class TestPerBackendConstants:
    """Pin the per-backend capability sets. Updating a constant
    must come with an explicit test change here so the matrix
    stays auditable in code review."""

    def test_neo4j_capabilities(self) -> None:
        assert NEO4J_CAPABILITIES == BackendCapabilities(
            persistent=True,
            full_text_indexes=True,
            similarity_search=True,
            embedding_similarity=False,
            per_user_routing=True,
        )

    def test_age_capabilities(self) -> None:
        # AGE matches Neo4j on persistence + similarity (CONTAINS
        # scoring, slice 6) + full-text (GIN/pg_trgm polyfill,
        # slice 8). Diverges on per-user routing (single global
        # DSN until UserDatabaseService gets the multi-tenant
        # Postgres extension) and embedding_similarity (waiting
        # for pgvector + embedding-generation upstream).
        assert AGE_CAPABILITIES == BackendCapabilities(
            persistent=True,
            full_text_indexes=True,
            similarity_search=True,
            embedding_similarity=False,
            per_user_routing=False,
        )

    def test_memory_capabilities(self) -> None:
        # In-memory: no persistence, no indexes, no similarity.
        assert MEMORY_CAPABILITIES == BackendCapabilities(
            persistent=False,
            full_text_indexes=False,
            similarity_search=False,
            embedding_similarity=False,
            per_user_routing=False,
        )


class TestAdaptersExposeCapabilities:
    """Each concrete adapter exposes ``capabilities`` and returns
    the matching constant. Caught backend drift would surface
    here when an adapter forgets to wire the property."""

    def test_in_memory_storage_returns_memory_capabilities(self) -> None:
        from graphora_server.services.storage.memory import InMemoryStorage

        storage = InMemoryStorage(user_id="test")
        assert storage.capabilities is MEMORY_CAPABILITIES

    def test_postgres_age_storage_returns_age_capabilities(self) -> None:
        from graphora_server.services.storage.postgres_age import PostgresAGEStorage

        storage = PostgresAGEStorage(dsn="postgresql://localhost/test")
        assert storage.capabilities is AGE_CAPABILITIES

    def test_neo4j_storage_returns_neo4j_capabilities(self) -> None:
        # Neo4jStorage's __init__ opens a sync connection to verify
        # auth, which we don't want in a unit test. Mock the driver
        # construction so we can instantiate without a live Neo4j.
        from graphora_server.services.storage import neo4j as neo4j_mod

        with patch.object(neo4j_mod, "AsyncGraphDatabase") as mock_async:
            with patch.object(neo4j_mod, "GraphDatabase") as mock_sync:
                mock_async.driver = MagicMock()
                mock_sync.driver = MagicMock()
                # The __init__ does `with sync_driver.session() as session: session.run("RETURN 1")` —
                # context-manager protocol on MagicMock returns a MagicMock by default,
                # which has a .run() method, so that's enough.
                storage = neo4j_mod.Neo4jStorage(
                    uri="bolt://localhost:7687",
                    username="u",
                    password="p",
                )
        assert storage.capabilities is NEO4J_CAPABILITIES


class TestCapabilityBranching:
    """Demonstrate the intended caller pattern: branch on a flag
    instead of on a STORAGE_TYPE string. These tests don't add new
    dispatch logic to the codebase — they pin the contract that
    callers can use."""

    def test_branch_on_full_text_indexes_flag(self) -> None:
        # Idiomatic caller pattern: check the flag instead of
        # introspecting the backend's class or settings.STORAGE_TYPE.
        for caps in (NEO4J_CAPABILITIES, AGE_CAPABILITIES):
            assert caps.full_text_indexes is True
        assert MEMORY_CAPABILITIES.full_text_indexes is False

    def test_branch_on_persistent_flag(self) -> None:
        # The "this won't survive restart" decision shouldn't
        # require knowing whether the backend is in-memory by name.
        assert NEO4J_CAPABILITIES.persistent is True
        assert AGE_CAPABILITIES.persistent is True
        assert MEMORY_CAPABILITIES.persistent is False

    def test_branch_on_per_user_routing(self) -> None:
        # Important for the eventual UserDatabaseService refactor:
        # AGE doesn't carry per-user staging/prod connections today.
        assert NEO4J_CAPABILITIES.per_user_routing is True
        assert AGE_CAPABILITIES.per_user_routing is False
        assert MEMORY_CAPABILITIES.per_user_routing is False
