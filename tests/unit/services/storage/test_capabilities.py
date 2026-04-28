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
    AGE_STATIC_CAPABILITIES,
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

    def test_age_static_capabilities(self) -> None:
        # AGE's full_text_indexes is runtime-derived (depends on
        # whether pg_trgm got installed during _bootstrap_schema),
        # so it's NOT in AGE_STATIC_CAPABILITIES. The constant
        # exposes only the truly static flags; the adapter's
        # capabilities property fills in full_text_indexes from
        # ``self._has_pg_trgm`` per call.
        assert AGE_STATIC_CAPABILITIES == {
            "persistent": True,
            "similarity_search": True,
            "embedding_similarity": False,
            "per_user_routing": False,
        }
        assert "full_text_indexes" not in AGE_STATIC_CAPABILITIES

    def test_memory_capabilities(self) -> None:
        # In-memory: no persistence, no FT indexes. similarity_search
        # is True — InMemoryStorage.find_similar_nodes runs a real
        # fuzzy match via _calculate_similarity, so the dev/demo
        # path keeps the merge flow's similarity fallback working.
        assert MEMORY_CAPABILITIES == BackendCapabilities(
            persistent=False,
            full_text_indexes=False,
            similarity_search=True,
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

    def test_postgres_age_pre_bootstrap_reports_full_text_false(self) -> None:
        # Before _bootstrap_schema runs, ``_has_pg_trgm`` is unset.
        # capabilities defaults full_text_indexes to False so a
        # caller branching on the flag doesn't fire an FT index
        # call that would just no-op.
        from graphora_server.services.storage.postgres_age import PostgresAGEStorage

        storage = PostgresAGEStorage(dsn="postgresql://localhost/test")
        caps = storage.capabilities
        assert caps.full_text_indexes is False
        # Static fields stay matched to AGE_STATIC_CAPABILITIES.
        assert caps.persistent is True
        assert caps.similarity_search is True
        assert caps.embedding_similarity is False
        assert caps.per_user_routing is False

    def test_postgres_age_with_pg_trgm_reports_full_text_true(self) -> None:
        # After _bootstrap_schema runs and pg_trgm is available,
        # ``_has_pg_trgm`` is True and capabilities flips
        # full_text_indexes to match. Simulate the post-bootstrap
        # state directly without spinning up a container.
        from graphora_server.services.storage.postgres_age import PostgresAGEStorage

        storage = PostgresAGEStorage(dsn="postgresql://localhost/test")
        storage._has_pg_trgm = True
        assert storage.capabilities.full_text_indexes is True

    def test_postgres_age_without_pg_trgm_reports_full_text_false(self) -> None:
        # pg_trgm absent → full_text_indexes False. Mirrors the
        # bootstrap path that warns once and continues.
        from graphora_server.services.storage.postgres_age import PostgresAGEStorage

        storage = PostgresAGEStorage(dsn="postgresql://localhost/test")
        storage._has_pg_trgm = False
        assert storage.capabilities.full_text_indexes is False

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
        # Neo4j is statically True; AGE is runtime-derived so we
        # exercise it via the adapter property (covered by the
        # TestAdaptersExposeCapabilities class above). Memory is
        # statically False.
        assert NEO4J_CAPABILITIES.full_text_indexes is True
        assert MEMORY_CAPABILITIES.full_text_indexes is False

    def test_branch_on_persistent_flag(self) -> None:
        # The "this won't survive restart" decision shouldn't
        # require knowing whether the backend is in-memory by name.
        assert NEO4J_CAPABILITIES.persistent is True
        assert MEMORY_CAPABILITIES.persistent is False
        assert AGE_STATIC_CAPABILITIES["persistent"] is True

    def test_branch_on_per_user_routing(self) -> None:
        # Important for the eventual UserDatabaseService refactor:
        # AGE doesn't carry per-user staging/prod connections today.
        assert NEO4J_CAPABILITIES.per_user_routing is True
        assert MEMORY_CAPABILITIES.per_user_routing is False
        assert AGE_STATIC_CAPABILITIES["per_user_routing"] is False
