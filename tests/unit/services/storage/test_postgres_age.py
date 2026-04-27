"""Unit tests for the Apache AGE storage adapter foundation.

Slice 1 covers the connection layer + cypher helper + agtype parser.
Method-body tests for store_nodes / get_node_by_id / etc. land
alongside their implementations in subsequent slices.

These tests deliberately avoid a real Postgres connection — agtype
parsing and identifier validation are pure-function logic, and the
connection lifecycle is exercised via mocked psycopg pools. The
integration test that round-trips through a real AGE instance is
gated behind a testcontainers fixture and lives in
tests/integration/ when it lands.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphora_server.services.storage.postgres_age import (
    CypherInjectionError,
    PostgresAGEStorage,
    _validate_identifier,
)


class TestIdentifierValidation:
    """Cypher identifier safety — labels and graph names get
    interpolated into AGE's cypher() body as raw strings, so the
    validator is the only thing standing between a malicious caller
    and SQL/Cypher injection."""

    def test_accepts_simple_alpha(self) -> None:
        assert _validate_identifier("Person") == "Person"

    def test_accepts_underscore_prefix(self) -> None:
        assert _validate_identifier("_internal") == "_internal"

    def test_accepts_alphanumeric_with_underscore(self) -> None:
        assert _validate_identifier("Person_42") == "Person_42"

    def test_rejects_empty(self) -> None:
        with pytest.raises(CypherInjectionError, match="Empty"):
            _validate_identifier("")

    def test_rejects_leading_digit(self) -> None:
        with pytest.raises(CypherInjectionError, match="Invalid"):
            _validate_identifier("42Person")

    def test_rejects_special_chars(self) -> None:
        # Each one is a real injection attempt vector for the
        # cypher() interpolation. None should pass.
        for bad in ["Per;son", "Person'", "Person--", "Per son", "Per`son"]:
            with pytest.raises(CypherInjectionError):
                _validate_identifier(bad)

    def test_rejects_overlong(self) -> None:
        with pytest.raises(CypherInjectionError, match="exceeds 256"):
            _validate_identifier("a" * 300)

    def test_includes_kind_in_error(self) -> None:
        # The kind label flows into error messages so the operator
        # can tell which validation tripped.
        with pytest.raises(CypherInjectionError, match="graph name"):
            _validate_identifier("", kind="graph name")


class TestAgtypeParser:
    """AGE returns vertex/edge values as ``{...}::<type>`` strings.
    The parser strips the trailing type tag and json-decodes the
    body. Scalars round-trip through json.loads."""

    def test_parses_typed_vertex(self) -> None:
        raw = '{"id": 0, "label": "Person", "properties": {"name": "Alice"}}::vertex'
        parsed = PostgresAGEStorage._parse_agtype(raw)
        assert parsed == {
            "id": 0,
            "label": "Person",
            "properties": {"name": "Alice"},
        }

    def test_parses_typed_edge(self) -> None:
        raw = '{"id": 1, "label": "KNOWS", "start_id": 0, "end_id": 2}::edge'
        parsed = PostgresAGEStorage._parse_agtype(raw)
        assert parsed["label"] == "KNOWS"
        assert parsed["start_id"] == 0

    def test_parses_scalar_string(self) -> None:
        # AGE returns scalar strings json-quoted.
        assert PostgresAGEStorage._parse_agtype('"hello"') == "hello"

    def test_parses_scalar_number(self) -> None:
        assert PostgresAGEStorage._parse_agtype("42") == 42

    def test_parses_null(self) -> None:
        assert PostgresAGEStorage._parse_agtype(None) is None

    def test_passes_through_already_parsed(self) -> None:
        # If psycopg's agtype adapter returns a dict directly (newer
        # versions), the parser should be a no-op.
        already_parsed = {"id": 0}
        assert PostgresAGEStorage._parse_agtype(already_parsed) == already_parsed

    def test_invalid_json_returns_raw(self) -> None:
        # Defensive: don't raise on unexpected agtype shapes — the
        # method is best-effort. Caller can inspect the raw value.
        assert (
            PostgresAGEStorage._parse_agtype("not json::vertex") == "not json::vertex"
        )

    def test_scalar_string_with_embedded_double_colon(self) -> None:
        # Regression for slice-1 review: the earlier implementation
        # stripped on the rightmost ``::`` unconditionally, so
        # ``"a::b"`` (a JSON-quoted scalar containing ``::``) parsed
        # as ``"a"`` and then crashed back to the raw string. Anchor
        # on the known-tag suffix list so this case round-trips.
        assert PostgresAGEStorage._parse_agtype('"a::b"') == "a::b"

    def test_scalar_string_with_unknown_type_tag_suffix(self) -> None:
        # ``"foo"::custom`` is not an AGE-recognized tag — the parser
        # should not strip ``::custom`` and should return the raw
        # value rather than mis-decoding. Defensive against future
        # AGE versions that introduce tags we don't know about.
        assert PostgresAGEStorage._parse_agtype('"foo"::custom') == '"foo"::custom'

    def test_path_type_tag(self) -> None:
        # AGE returns paths as ``[v, e, v, ...]::path`` — same strip
        # logic applies as vertex/edge.
        raw = '[{"id": 0}, {"id": 1}]::path'
        parsed = PostgresAGEStorage._parse_agtype(raw)
        assert parsed == [{"id": 0}, {"id": 1}]


class TestPostgresAGEStorageInitialization:
    """Class construction — no I/O, just config validation and pool
    placeholder setup."""

    def test_defaults_graph_name_to_graphora(self) -> None:
        storage = PostgresAGEStorage(dsn="postgresql://localhost/test")
        assert storage.graph_name == "graphora"
        assert storage._pool is None  # Lazy — no connection yet.

    def test_validates_graph_name_at_construction(self) -> None:
        # Invalid graph names short-circuit before any I/O.
        with pytest.raises(CypherInjectionError):
            PostgresAGEStorage(
                dsn="postgresql://localhost/test",
                graph_name="bad name",
            )

    def test_carries_pool_size_config(self) -> None:
        storage = PostgresAGEStorage(
            dsn="postgresql://localhost/test",
            min_pool_size=2,
            max_pool_size=20,
        )
        assert storage.min_pool_size == 2
        assert storage.max_pool_size == 20


class TestPostgresAGEStorageMethodStubs:
    """Stubbed GraphStorageInterface methods raise NotImplementedError
    with a clear pointer to the slice that owns the implementation.

    These tests pin the pending-method contract — when a slice lands,
    the corresponding test here flips from "raises" to "round-trips."
    The naming pattern lets future contributors find the stubs by
    grepping for the slice number.
    """

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_store_nodes_pending_slice_2(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Note: store_nodes / store_relationships migrated to slice 3
        # alongside per-user factory dispatch. The error message still
        # points at the next-up slice, the marker just shifted.
        with pytest.raises(NotImplementedError, match="slice"):
            await storage.store_nodes([], batch_index=0, transform_id="t")

    @pytest.mark.asyncio
    async def test_get_node_by_id_pending_slice_3(
        self, storage: PostgresAGEStorage
    ) -> None:
        with pytest.raises(NotImplementedError, match="slice 3"):
            await storage.get_node_by_id("nid")

    @pytest.mark.asyncio
    async def test_find_similar_nodes_pending_slice_4(
        self, storage: PostgresAGEStorage
    ) -> None:
        with pytest.raises(NotImplementedError, match="slice 4"):
            await storage.find_similar_nodes(
                label="Person", properties={"name": "Alice"}
            )

    @pytest.mark.asyncio
    async def test_create_ft_index_pending_slice_5(
        self, storage: PostgresAGEStorage
    ) -> None:
        with pytest.raises(NotImplementedError, match="slice 5"):
            await storage.create_or_replace_ft_index_for_node("ix", "Person", ["name"])


class TestCheckpointRoundTrip:
    """Slice 2: ``update_checkpoint`` and ``get_storage_status`` move
    from stubs to real Cypher.

    These tests mock ``_execute_cypher`` (the boundary between the
    adapter and psycopg) so we don't need a live Postgres+AGE for
    the unit suite. The full DB round-trip lands as an integration
    test alongside the testcontainers fixture in slice 3.
    """

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_update_checkpoint_emits_merge_cypher(
        self, storage: PostgresAGEStorage
    ) -> None:
        from graphora_server.services.storage.models import StorageStage

        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            result = await storage.update_checkpoint(
                transform_id="t-42", last_index=7, stage=StorageStage.NODES
            )

        assert result.success is True
        assert result.items_processed == 1
        assert result.batch_index == 7
        assert result.error is None
        # _execute_cypher should have been called once with a MERGE
        # on the :_Checkpoint label keyed by transform_id, plus
        # parameters carrying the stage value (.value, not the enum).
        assert mock_exec.call_count == 1
        cypher_body = mock_exec.call_args.args[0]
        params = mock_exec.call_args.kwargs["params"]
        assert "MERGE (c:_Checkpoint" in cypher_body
        assert params["transform_id"] == "t-42"
        assert params["last_index"] == 7
        assert params["stage"] == "nodes"  # StorageStage.NODES.value
        # Timestamp must be ISO-8601 — get_storage_status parses it
        # with fromisoformat on the read side.
        datetime.fromisoformat(params["timestamp"])

    @pytest.mark.asyncio
    async def test_update_checkpoint_reports_failure_on_exception(
        self, storage: PostgresAGEStorage
    ) -> None:
        from graphora_server.services.storage.models import StorageStage

        with patch.object(
            storage,
            "_execute_cypher",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            result = await storage.update_checkpoint(
                transform_id="t-42", last_index=7, stage=StorageStage.NODES
            )

        assert result.success is False
        assert result.items_processed == 0
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_get_storage_status_returns_none_when_absent(
        self, storage: PostgresAGEStorage
    ) -> None:
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=[])):
            assert await storage.get_storage_status("missing-tx") is None

    @pytest.mark.asyncio
    async def test_get_storage_status_parses_vertex_payload(
        self, storage: PostgresAGEStorage
    ) -> None:
        from graphora_server.services.storage.models import StorageStage

        # Shape mirrors what AGE returns for a vertex agtype value:
        # parsed _parse_agtype output is a dict with "label" and
        # "properties" keys.
        vertex = (
            '{"id": 0, "label": "_Checkpoint", "properties": '
            '{"transform_id": "t-42", "last_processed_index": 7, '
            '"stage": "relationships", '
            '"timestamp": "2026-04-27T15:00:00+00:00"}}::vertex'
        )
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[(vertex,)])
        ):
            checkpoint = await storage.get_storage_status("t-42")

        assert checkpoint is not None
        assert checkpoint.transform_id == "t-42"
        assert checkpoint.last_processed_index == 7
        assert checkpoint.stage == StorageStage.RELATIONSHIPS
        assert checkpoint.timestamp.year == 2026

    @pytest.mark.asyncio
    async def test_get_storage_status_falls_back_on_malformed_timestamp(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Defensive: if AGE's timestamp round-tripped into something
        # fromisoformat can't parse, we want a checkpoint with the
        # current wall clock rather than a 500.
        vertex = (
            '{"id": 0, "label": "_Checkpoint", "properties": '
            '{"transform_id": "t-42", "last_processed_index": 0, '
            '"stage": "nodes", "timestamp": "not-a-date"}}::vertex'
        )
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[(vertex,)])
        ):
            checkpoint = await storage.get_storage_status("t-42")

        assert checkpoint is not None
        assert checkpoint.transform_id == "t-42"
        # Wall-clock fallback — no exception raised.

    @pytest.mark.asyncio
    async def test_get_storage_status_match_orders_by_timestamp_desc(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Pin the read-side Cypher shape — store_nodes resumes from
        # the most recent checkpoint, so the ORDER BY direction is
        # load-bearing.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.get_storage_status("t-42")

        cypher_body = mock_exec.call_args.args[0]
        assert "MATCH (c:_Checkpoint" in cypher_body
        assert "ORDER BY c.timestamp DESC" in cypher_body
        assert "LIMIT 1" in cypher_body


class TestFactoryDispatch:
    """STORAGE_TYPE='postgres' goes from reserved-error (slice 1) to
    actual dispatch (slice 2) for the global ``create_storage()`` entry
    point. Per-user dispatch (``create_storage_for_user``) stays
    reserved until slice 3 wires per-user Postgres config.
    """

    @pytest.mark.asyncio
    async def test_create_storage_constructs_postgres_age_storage(self) -> None:
        from graphora_server.services.storage import factory
        from graphora_server.services.storage.factory import (
            StorageConfig,
            create_storage,
        )

        # Minimal env: explicit DSN set at the global config level.
        # The constructor does no I/O (lazy pool open), so a fake DSN
        # is enough to exercise the dispatch path.
        with patch.object(factory.settings, "POSTGRES_AGE_DSN", "postgresql://fake/db"):
            with patch.object(factory.settings, "POSTGRES_AGE_GRAPH_NAME", "graphora"):
                config = StorageConfig(storage_type="postgres")
                storage = await create_storage(config)

        assert isinstance(storage, PostgresAGEStorage)
        assert storage.dsn == "postgresql://fake/db"
        assert storage.graph_name == "graphora"

    @pytest.mark.asyncio
    async def test_create_storage_falls_back_to_database_url(self) -> None:
        # When POSTGRES_AGE_DSN is unset, the AGE adapter rides on
        # the application Postgres connection (resolved_database_url).
        from graphora_server.services.storage import factory
        from graphora_server.services.storage.factory import (
            StorageConfig,
            create_storage,
        )

        with patch.object(factory.settings, "POSTGRES_AGE_DSN", None):
            with patch.object(
                type(factory.settings),
                "resolved_database_url",
                new_callable=lambda: property(
                    lambda self: "postgresql://app-db/graphora"
                ),
            ):
                config = StorageConfig(storage_type="postgres")
                storage = await create_storage(config)

        assert isinstance(storage, PostgresAGEStorage)
        assert storage.dsn == "postgresql://app-db/graphora"

    @pytest.mark.asyncio
    async def test_create_storage_rejects_when_no_dsn_available(self) -> None:
        from graphora_server.services.storage import factory
        from graphora_server.services.storage.factory import (
            StorageConfig,
            create_storage,
        )

        with patch.object(factory.settings, "POSTGRES_AGE_DSN", None):
            # Make resolved_database_url empty too — operator forgot
            # to configure both. Should fail loud with a clear msg.
            with patch.object(
                type(factory.settings),
                "resolved_database_url",
                new_callable=lambda: property(lambda self: ""),
            ):
                config = StorageConfig(storage_type="postgres")
                with pytest.raises(ValueError, match="POSTGRES_AGE_DSN"):
                    await create_storage(config)

    @pytest.mark.asyncio
    async def test_create_storage_for_user_rejects_postgres_pending_slice_3(
        self,
    ) -> None:
        # Per-user Postgres config doesn't exist yet — keep this
        # branch raising NotImplementedError until slice 3 lands the
        # UserDatabaseService extension.
        from graphora_server.services.storage import factory

        with patch.object(factory.settings, "STORAGE_TYPE", "postgres"):
            with pytest.raises(NotImplementedError, match="slice 3"):
                await factory.create_storage_for_user(user_id="u1", use_staging=True)


class TestExecuteCypher:
    """Foundation _execute_cypher uses a mocked psycopg pool to avoid
    needing a real Postgres. Verifies query shape, parameter
    serialization, and graph-name interpolation."""

    @pytest.mark.asyncio
    async def test_wraps_query_in_cypher_function(self) -> None:
        storage = PostgresAGEStorage(dsn="postgresql://localhost/test")
        cur = AsyncMock()
        cur.fetchall = AsyncMock(return_value=[("result_row",)])
        cur.execute = AsyncMock()
        cur.__aenter__ = AsyncMock(return_value=cur)
        cur.__aexit__ = AsyncMock(return_value=None)
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=cur)
        conn.commit = AsyncMock()

        @asynccontextmanager_helper()
        async def fake_get_connection():
            yield conn

        with patch.object(storage, "_get_connection", fake_get_connection):
            rows = await storage._execute_cypher(
                "MATCH (n) RETURN n",
                params={"foo": "bar"},
            )

        # Last execute() call was the cypher() wrapper — earlier
        # calls were LOAD 'age' / SET search_path emitted by
        # _get_connection. We assert on the cypher() wrapper.
        cypher_calls = [
            call
            for call in cur.execute.call_args_list
            if call.args and "cypher(" in call.args[0]
        ]
        assert cypher_calls, "expected a cypher() SQL wrapper to be issued"
        sql, sql_params = cypher_calls[0].args
        assert "cypher('graphora'" in sql
        assert "$cypher$MATCH (n) RETURN n$cypher$" in sql
        assert sql_params == ('{"foo": "bar"}',)
        assert rows == [("result_row",)]


def asynccontextmanager_helper():
    """Tiny shim — returns a decorator that wraps an async generator
    into an async context manager so we can patch.object() the
    _get_connection attribute. ``contextlib.asynccontextmanager`` is
    the right tool but it doesn't compose well with patch.object's
    callable expectation in MagicMock land.
    """
    from contextlib import asynccontextmanager

    return asynccontextmanager
