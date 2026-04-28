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
    _coerce_for_age,
    _validate_identifier,
)
from graphora_server.services.storage.models import StorageStage
from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
)
from graphora_server.utils.constants import MERGE_ID, TRANSFORM_ID


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
    """Slice 3: STORAGE_TYPE='postgres' wired at BOTH factory entry
    points, sharing the ``_build_age_storage`` helper so they stay
    in lockstep — the asymmetric wiring slice 2 first attempted was
    the contract bug flagged in review.

    Per-user Postgres routing (parallel to Neo4j's stagingDb / prodDb)
    is intentionally out of scope for slice 3; both entry points use
    the same global ``POSTGRES_AGE_DSN``. ``use_staging`` is a no-op
    for postgres and just emits a debug warning. Slice 4 wires
    multi-tenant Postgres routing if/when there's actual demand.
    """

    @pytest.mark.asyncio
    async def test_create_storage_constructs_postgres_age_storage(self) -> None:
        from graphora_server.services.storage import factory
        from graphora_server.services.storage.factory import (
            StorageConfig,
            create_storage,
        )

        with patch.object(factory.settings, "POSTGRES_AGE_DSN", "postgresql://fake/db"):
            with patch.object(factory.settings, "POSTGRES_AGE_GRAPH_NAME", "graphora"):
                config = StorageConfig(storage_type="postgres")
                storage = await create_storage(config)

        assert isinstance(storage, PostgresAGEStorage)
        assert storage.dsn == "postgresql://fake/db"
        assert storage.graph_name == "graphora"

    @pytest.mark.asyncio
    async def test_create_storage_for_user_constructs_postgres_age_storage(
        self,
    ) -> None:
        # Real-app paths (tasks.py, api/quality.py, services/merge/...)
        # enter through create_storage_for_user, so this is the
        # entry point that has to actually work for STORAGE_TYPE=postgres
        # to be usable end-to-end. Pinned here.
        from graphora_server.services.storage import factory

        with patch.object(factory.settings, "STORAGE_TYPE", "postgres"):
            with patch.object(
                factory.settings, "POSTGRES_AGE_DSN", "postgresql://fake/db"
            ):
                storage = await factory.create_storage_for_user(
                    user_id="u1", use_staging=True
                )

        assert isinstance(storage, PostgresAGEStorage)

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
            with patch.object(
                type(factory.settings),
                "resolved_database_url",
                new_callable=lambda: property(lambda self: ""),
            ):
                config = StorageConfig(storage_type="postgres")
                with pytest.raises(ValueError, match="POSTGRES_AGE_DSN"):
                    await create_storage(config)

    @pytest.mark.asyncio
    async def test_create_storage_for_user_use_staging_false_still_works(
        self,
    ) -> None:
        # Per-user Postgres routing is slice 4 work — for now
        # use_staging=False on postgres returns the same shared AGE
        # storage and emits a warning rather than refusing. Pin
        # the behaviour so callers (merge flow) keep working.
        from graphora_server.services.storage import factory

        with patch.object(factory.settings, "STORAGE_TYPE", "postgres"):
            with patch.object(
                factory.settings, "POSTGRES_AGE_DSN", "postgresql://fake/db"
            ):
                storage = await factory.create_storage_for_user(
                    user_id="u1", use_staging=False
                )

        assert isinstance(storage, PostgresAGEStorage)


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


class TestCoerceForAge:
    """``_coerce_for_age`` is the value-side hardening between
    Python land and AGE's jsonb-only param channel — datetime,
    Pydantic models, enums, and nested dicts/lists all get
    flattened to JSON-compatible values."""

    def test_passes_through_primitives(self) -> None:
        for v in [None, "x", 42, 3.14, True]:
            assert _coerce_for_age(v) == v

    def test_datetime_to_isoformat(self) -> None:
        from datetime import timezone as tz

        dt = datetime(2026, 4, 27, 15, 0, 0, tzinfo=tz.utc)
        assert _coerce_for_age(dt) == "2026-04-27T15:00:00+00:00"

    def test_enum_to_value(self) -> None:
        assert _coerce_for_age(StorageStage.NODES) == "nodes"

    def test_pydantic_model_to_dict(self) -> None:
        prov = NodeProvenance(
            chunk_ids=["c1"],
            extraction_timestamp="2026-04-27",
            confidence_score=0.9,
        )
        out = _coerce_for_age(prov)
        assert isinstance(out, dict)
        assert out["confidence_score"] == 0.9
        assert out["chunk_ids"] == ["c1"]

    def test_nested_dict_recursion(self) -> None:
        out = _coerce_for_age({"stage": StorageStage.NODES, "nested": {"score": 0.5}})
        assert out == {"stage": "nodes", "nested": {"score": 0.5}}

    def test_list_recursion(self) -> None:
        assert _coerce_for_age([StorageStage.NODES, StorageStage.RELATIONSHIPS]) == [
            "nodes",
            "relationships",
        ]


def _ok_checkpoint_result():
    """Tiny helper — returns a fake StorageBatchResult-shaped object
    so update_checkpoint mocks satisfy ``.success`` / ``.error``
    accesses without bringing in the full pydantic model."""
    return type("R", (), {"success": True, "error": None})()


class TestStoreNodes:
    """Slice 3: store_nodes ships UNWIND-batched writes grouped
    by entity type. Tests mock ``_execute_cypher`` so we can pin
    the Cypher shape, batch grouping, property coercion, and
    partial-batch failure semantics without a live AGE."""

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    def _make_node(
        self, type_: str, name: str, *, with_provenance: bool = False
    ) -> BaseNode:
        prov = (
            NodeProvenance(
                chunk_ids=["c1"],
                extraction_timestamp="2026-04-27",
                confidence_score=0.9,
                source_file="doc.pdf",
            )
            if with_provenance
            else None
        )
        return BaseNode(
            type=type_,
            properties={"name": name},
            provenance=prov,
        )

    @pytest.mark.asyncio
    async def test_buckets_by_type_one_unwind_per_bucket(
        self, storage: PostgresAGEStorage
    ) -> None:
        nodes = [
            self._make_node("Person", "Alice"),
            self._make_node("Person", "Bob"),
            self._make_node("Company", "Acme"),
        ]
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            with patch.object(
                storage,
                "update_checkpoint",
                new=AsyncMock(return_value=_ok_checkpoint_result()),
            ):
                result = await storage.store_nodes(
                    nodes, batch_index=0, transform_id="t-42"
                )

        # Two type buckets in input → two cypher calls (Person + Company).
        assert mock_exec.call_count == 2
        cyphers = [c.args[0] for c in mock_exec.call_args_list]
        assert any("MERGE (n:Person {id: row.id})" in c for c in cyphers)
        assert any("MERGE (n:Company {id: row.id})" in c for c in cyphers)
        assert all("UNWIND $batch AS row" in c for c in cyphers)
        assert result.success is True
        assert result.items_processed == 3

    @pytest.mark.asyncio
    async def test_create_mode_uses_create_keyword(
        self, storage: PostgresAGEStorage
    ) -> None:
        # merge=False maps to CREATE; pin to prevent silent flips.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            with patch.object(
                storage,
                "update_checkpoint",
                new=AsyncMock(return_value=_ok_checkpoint_result()),
            ):
                await storage.store_nodes(
                    [self._make_node("Person", "Alice")],
                    batch_index=0,
                    transform_id="t",
                    merge=False,
                )
        cypher = mock_exec.call_args.args[0]
        assert "CREATE (n:Person {id: row.id})" in cypher

    @pytest.mark.asyncio
    async def test_skips_nodes_with_invalid_type(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Injection-attempt entity type — should be skipped and
        # warned, not crash the whole batch.
        nodes = [
            BaseNode(type="Per;son", properties={"name": "Bad"}),
            self._make_node("Person", "Good"),
        ]
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=[])):
            with patch.object(
                storage,
                "update_checkpoint",
                new=AsyncMock(return_value=_ok_checkpoint_result()),
            ):
                result = await storage.store_nodes(
                    nodes, batch_index=0, transform_id="t"
                )
        assert result.items_processed == 1
        assert any("Per;son" in w or "alphanumerics" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_partial_batch_failure_reports_warnings(
        self, storage: PostgresAGEStorage
    ) -> None:
        # First _execute_cypher call succeeds, second raises.
        nodes = [
            self._make_node("Person", "Alice"),
            self._make_node("Company", "Acme"),
        ]
        responses = [iter([[], RuntimeError("boom")])]

        async def flaky(*args, **kwargs):
            v = next(responses[0])
            if isinstance(v, Exception):
                raise v
            return v

        with patch.object(storage, "_execute_cypher", new=flaky):
            with patch.object(
                storage,
                "update_checkpoint",
                new=AsyncMock(return_value=_ok_checkpoint_result()),
            ):
                result = await storage.store_nodes(
                    nodes, batch_index=0, transform_id="t"
                )
        assert result.success is False
        assert result.items_processed == 1  # First bucket succeeded
        assert "boom" in (result.error or "")
        assert any("Partial batch" in w for w in result.warnings)


class TestExtractNodeWritepathFields:
    """Static helper extracted from store_nodes so the value-prep
    logic (metadata strip + provenance fold + JSON coerce) can be
    pinned without faking AGE."""

    def test_basenode_input(self) -> None:
        node = BaseNode(type="Person", properties={"name": "Alice"})
        node_id, node_type, props = PostgresAGEStorage._extract_node_writepath_fields(
            node, transform_id="t-42", merge_id=None
        )
        assert node_type == "Person"
        assert props["name"] == "Alice"
        assert props[TRANSFORM_ID] == "t-42"
        assert MERGE_ID not in props

    def test_dict_input_with_uuid_fallback(self) -> None:
        node_id, _, _ = PostgresAGEStorage._extract_node_writepath_fields(
            {"type": "Person", "properties": {}},
            transform_id="t",
            merge_id=None,
        )
        # Auto-assigned a UUID when the dict didn't carry an id.
        assert isinstance(node_id, str) and len(node_id) >= 32

    def test_provenance_setdefault_doesnt_clobber(self) -> None:
        # LLM-emitted source_file should win over provenance's value.
        node = BaseNode(
            type="Person",
            properties={"name": "Alice", "source_file": "manual.pdf"},
            provenance=NodeProvenance(
                chunk_ids=["c1"],
                extraction_timestamp="2026-04-27",
                source_file="auto.pdf",
            ),
        )
        _, _, props = PostgresAGEStorage._extract_node_writepath_fields(
            node, transform_id="t", merge_id=None
        )
        assert props["source_file"] == "manual.pdf"

    def test_no_type_raises(self) -> None:
        with pytest.raises(ValueError, match="no type"):
            PostgresAGEStorage._extract_node_writepath_fields(
                {"id": "abc", "type": "", "properties": {}},
                transform_id="t",
                merge_id=None,
            )


class TestStoreRelationships:
    """Slice 3: per-rel MERGE Cypher with static identifier
    interpolation. Versioning + bucketed UNWIND batching lands in
    slice 4."""

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    def _make_rel(self, rel_type: str = "WORKS_AT") -> RelationshipInstance:
        return RelationshipInstance(
            type=rel_type,
            source_id="s1",
            target_id="t1",
            source_type="Person",
            target_type="Company",
            properties={"role": "engineer"},
        )

    @pytest.mark.asyncio
    async def test_emits_match_match_merge_cypher(
        self, storage: PostgresAGEStorage
    ) -> None:
        rel = self._make_rel()
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            with patch.object(
                storage,
                "update_checkpoint",
                new=AsyncMock(return_value=_ok_checkpoint_result()),
            ):
                result = await storage.store_relationships(
                    [rel], batch_index=0, transform_id="t-42"
                )
        cypher = mock_exec.call_args.args[0]
        assert "MATCH (s:Person {id: $source_id})" in cypher
        assert "MATCH (t:Company {id: $target_id})" in cypher
        assert "MERGE (s)-[r:WORKS_AT]->(t)" in cypher
        assert "SET r += $props" in cypher
        # Source/target ids passed through params, not interpolated.
        params = mock_exec.call_args.kwargs["params"]
        assert params["source_id"] == "s1"
        assert params["target_id"] == "t1"
        assert params["props"]["role"] == "engineer"
        assert params["props"][TRANSFORM_ID] == "t-42"
        assert result.items_processed == 1

    @pytest.mark.asyncio
    async def test_skips_invalid_identifier(self, storage: PostgresAGEStorage) -> None:
        # Injection attempt in rel type — must skip with warning,
        # not interpolate the bad string into Cypher.
        rel = self._make_rel(rel_type="WORKS;DROP")
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            with patch.object(
                storage,
                "update_checkpoint",
                new=AsyncMock(return_value=_ok_checkpoint_result()),
            ):
                result = await storage.store_relationships(
                    [rel], batch_index=0, transform_id="t"
                )
        # _execute_cypher never reached because of the skip.
        assert mock_exec.call_count == 0
        assert result.items_processed == 0
        assert any("Skipping relationship" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_dedupes_on_relationship_id(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Caller may pass the same RelationshipInstance twice (e.g.
        # from a duplicated extraction). Skip the second occurrence.
        rel = self._make_rel()
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            with patch.object(
                storage,
                "update_checkpoint",
                new=AsyncMock(return_value=_ok_checkpoint_result()),
            ):
                result = await storage.store_relationships(
                    [rel, rel], batch_index=0, transform_id="t"
                )
        assert mock_exec.call_count == 1
        assert result.items_processed == 1


class TestReadPath:
    """Slice 4: read-path methods that the app's verification +
    HTTP read endpoints + merge flow all depend on. Tests mock
    the Cypher boundary and pin the agtype-vertex-to-Node /
    agtype-edge-to-Edge mappings."""

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_get_transformation_data_two_round_trips(
        self, storage: PostgresAGEStorage
    ) -> None:
        responses = [
            [
                (
                    '{"id": 0, "label": "Person", "properties": '
                    '{"id": "n1", "name": "Alice", "__tid": "t-42"}}::vertex',
                ),
                (
                    '{"id": 1, "label": "Company", "properties": '
                    '{"id": "n2", "name": "Acme", "__tid": "t-42"}}::vertex',
                ),
            ],
            [
                (
                    '{"id": 100, "label": "WORKS_AT", "start_id": 0, '
                    '"end_id": 1, "properties": {"id": "e1", '
                    '"role": "engineer", "__tid": "t-42"}}::edge',
                    '"n1"',
                    '"n2"',
                ),
            ],
        ]
        call_log = []

        async def stub_run(query, params=None, return_columns=None):
            call_log.append((query, params))
            return responses[len(call_log) - 1]

        with patch.object(storage, "_execute_cypher", new=stub_run):
            response = await storage.get_transformation_data("t-42")

        assert len(call_log) == 2
        assert "MATCH (n)" in call_log[0][0]
        assert "n.__tid = $transform_id" in call_log[0][0]
        assert "MATCH (s)-[r]->(t)" in call_log[1][0]
        assert "r.__tid = $transform_id" in call_log[1][0]
        assert response.total_nodes == 2
        assert response.total_edges == 1
        assert {n.type for n in response.nodes} == {"Person", "Company"}
        assert response.edges[0].source == "n1"
        assert response.edges[0].target == "n2"
        assert response.edges[0].type == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_get_transformation_data_returns_empty_response(
        self, storage: PostgresAGEStorage
    ) -> None:
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=[])):
            response = await storage.get_transformation_data("missing-tx")
        assert response.total_nodes == 0
        assert response.total_edges == 0
        assert response.nodes == []
        assert response.edges == []

    @pytest.mark.asyncio
    async def test_get_transformation_data_dedupes_nodes(
        self, storage: PostgresAGEStorage
    ) -> None:
        dup = (
            '{"id": 0, "label": "Person", "properties": '
            '{"id": "n1", "name": "Alice"}}::vertex'
        )
        responses = [[(dup,), (dup,)], []]
        call_count = [0]

        async def stub_run(*args, **kwargs):
            r = responses[call_count[0]]
            call_count[0] += 1
            return r

        with patch.object(storage, "_execute_cypher", new=stub_run):
            response = await storage.get_transformation_data("t-42")
        assert response.total_nodes == 1
        assert response.nodes[0].id == "n1"

    @pytest.mark.asyncio
    async def test_get_merge_data_keys_on_merge_id(
        self, storage: PostgresAGEStorage
    ) -> None:
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.get_merge_data("m-1")
        for call in mock_exec.call_args_list:
            cypher = call.args[0]
            assert "__mid" in cypher
            assert "__tid" not in cypher

    @pytest.mark.asyncio
    async def test_get_nodes_by_property_validates_property_name(
        self, storage: PostgresAGEStorage
    ) -> None:
        with pytest.raises(CypherInjectionError):
            await storage.get_nodes_by_property("name; DROP", "anything")

    @pytest.mark.asyncio
    async def test_get_nodes_by_property_returns_matching_nodes(
        self, storage: PostgresAGEStorage
    ) -> None:
        rows = [
            (
                '{"id": 0, "label": "Person", "properties": '
                '{"id": "n1", "name": "Alice"}}::vertex',
            ),
        ]
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=rows)
        ) as mock_exec:
            nodes = await storage.get_nodes_by_property("name", "Alice")
        cypher = mock_exec.call_args.args[0]
        assert "n.name = $value" in cypher
        assert nodes[0].id == "n1"
        assert nodes[0].properties["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_relationships_between_nodes_uses_in_clause(
        self, storage: PostgresAGEStorage
    ) -> None:
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.get_relationships_between_nodes(["n1", "n2", "n3"])
        cypher = mock_exec.call_args.args[0]
        params = mock_exec.call_args.kwargs["params"]
        assert "s.id IN $node_ids" in cypher
        assert "t.id IN $node_ids" in cypher
        assert params["node_ids"] == ["n1", "n2", "n3"]

    @pytest.mark.asyncio
    async def test_get_relationships_between_with_type_filter(
        self, storage: PostgresAGEStorage
    ) -> None:
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.get_relationships_between(
                "n1", "n2", relationship_type="WORKS_AT"
            )
        cypher = mock_exec.call_args.args[0]
        assert "[r:WORKS_AT]" in cypher

    @pytest.mark.asyncio
    async def test_get_relationships_between_no_type_filter(
        self, storage: PostgresAGEStorage
    ) -> None:
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.get_relationships_between("n1", "n2")
        cypher = mock_exec.call_args.args[0]
        assert "-[r]->" in cypher
        assert "[r:" not in cypher

    @pytest.mark.asyncio
    async def test_get_all_node_properties_skips_metadata(
        self, storage: PostgresAGEStorage
    ) -> None:
        rows = [('["id", "name", "__tid", "extractor_model"]',)]
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=rows)):
            keys = await storage.get_all_node_properties("Person")
        assert "__tid" not in keys
        assert "id" in keys
        assert "name" in keys
        assert "extractor_model" in keys


class TestVertexToNode:
    """Pin the AGE-vertex to schema-Node mapping. The vertex_id from
    AGE is its internal numeric id; the user-facing id is in the
    property bag."""

    def test_pulls_user_id_from_properties_not_age_internal(self) -> None:
        parsed = {
            "id": 0,
            "label": "Person",
            "properties": {"id": "user-uuid", "name": "Alice"},
        }
        node = PostgresAGEStorage._vertex_to_node(parsed)
        assert node.id == "user-uuid"
        assert node.type == "Person"
        assert node.label == "Person"

    def test_strips_age_internal_id_from_properties(self) -> None:
        parsed = {
            "id": 99999,
            "label": "Person",
            "properties": {"id": "user-uuid", "name": "Alice"},
        }
        node = PostgresAGEStorage._vertex_to_node(parsed)
        assert "id" not in node.properties
        assert node.properties["name"] == "Alice"

    def test_returns_none_for_malformed_input(self) -> None:
        assert (
            PostgresAGEStorage._vertex_to_node({"label": "X", "properties": "weird"})
            is None
        )


class TestAdapterDoesNotWriteCheckpoint:
    """Regression for the slice-3 round-2 review finding:
    ``store_nodes`` / ``store_relationships`` used to write the
    checkpoint INTERNALLY before returning. That value (the batch
    INDEX, not items_processed) conflicted with the task layer's
    partial-failure contract: on success=False the task raises
    before its own checkpoint write, leaving the adapter's bogus
    value persisted and causing duplicate CREATEs on resume.

    Pin that the adapter never calls update_checkpoint from inside
    the write methods — the task layer is the canonical owner.
    """

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_store_nodes_does_not_call_update_checkpoint(
        self, storage: PostgresAGEStorage
    ) -> None:
        node = BaseNode(type="Person", properties={"name": "Alice"})
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=[])):
            with patch.object(
                storage, "update_checkpoint", new=AsyncMock()
            ) as mock_ckpt:
                await storage.store_nodes([node], batch_index=0, transform_id="t-42")
        mock_ckpt.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_relationships_does_not_call_update_checkpoint(
        self, storage: PostgresAGEStorage
    ) -> None:
        rel = RelationshipInstance(
            type="WORKS_AT",
            source_id="s1",
            target_id="t1",
            source_type="Person",
            target_type="Company",
        )
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=[])):
            with patch.object(
                storage, "update_checkpoint", new=AsyncMock()
            ) as mock_ckpt:
                await storage.store_relationships(
                    [rel], batch_index=0, transform_id="t-42"
                )
        mock_ckpt.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_nodes_does_not_checkpoint_on_partial_failure(
        self, storage: PostgresAGEStorage
    ) -> None:
        # The exact scenario the review flagged: first bucket succeeds,
        # second raises. The adapter must not stamp a checkpoint at
        # batch_index — the task layer raises on success=False and
        # owns the checkpoint write.
        node_p = BaseNode(type="Person", properties={"name": "Alice"})
        node_c = BaseNode(type="Company", properties={"name": "Acme"})
        responses = [iter([[], RuntimeError("second bucket failed")])]

        async def flaky(*args, **kwargs):
            v = next(responses[0])
            if isinstance(v, Exception):
                raise v
            return v

        with patch.object(storage, "_execute_cypher", new=flaky):
            with patch.object(
                storage, "update_checkpoint", new=AsyncMock()
            ) as mock_ckpt:
                result = await storage.store_nodes(
                    [node_p, node_c], batch_index=7, transform_id="t-42"
                )
        assert result.success is False
        # The bug-shaped behaviour we're guarding against: writing
        # checkpoint(7) here would mislead a resume into skipping
        # nodes 0..6.
        mock_ckpt.assert_not_called()


class TestFindNodesByPropertyValue:
    """Slice 5: ``find_nodes_by_property_value`` is the merge-flow
    pre-similarity lookup (services/merge/new_merger.py:1016, :1037,
    :1052). Different from get_nodes_by_property: adds a label
    filter and an exact_match toggle. Identifier validation guards
    the label + property name interpolations."""

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_exact_match_default(self, storage: PostgresAGEStorage) -> None:
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.find_nodes_by_property_value("Person", "name", "Alice")
        cypher = mock_exec.call_args.args[0]
        params = mock_exec.call_args.kwargs["params"]
        assert "MATCH (n:Person)" in cypher
        assert "n.name = $value" in cypher
        assert "CONTAINS" not in cypher
        assert params["value"] == "Alice"

    @pytest.mark.asyncio
    async def test_fuzzy_match_uses_contains(self, storage: PostgresAGEStorage) -> None:
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.find_nodes_by_property_value(
                "Person", "name", "alic", exact_match=False
            )
        cypher = mock_exec.call_args.args[0]
        # toLower wrapping on both sides for case-insensitive match.
        assert "toLower(n.name) CONTAINS toLower($value)" in cypher

    @pytest.mark.asyncio
    async def test_validates_label(self, storage: PostgresAGEStorage) -> None:
        # Injection attempts in either identifier must be refused.
        with pytest.raises(CypherInjectionError):
            await storage.find_nodes_by_property_value("Per;son", "name", "Alice")

    @pytest.mark.asyncio
    async def test_validates_property_name(self, storage: PostgresAGEStorage) -> None:
        with pytest.raises(CypherInjectionError):
            await storage.find_nodes_by_property_value("Person", "name; DROP", "Alice")

    @pytest.mark.asyncio
    async def test_returns_mapped_nodes(self, storage: PostgresAGEStorage) -> None:
        rows = [
            (
                '{"id": 0, "label": "Person", "properties": '
                '{"id": "n1", "name": "Alice"}}::vertex',
            ),
        ]
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=rows)):
            nodes = await storage.find_nodes_by_property_value(
                "Person", "name", "Alice"
            )
        assert len(nodes) == 1
        assert nodes[0].id == "n1"
        assert nodes[0].type == "Person"


class TestFullTextIndexDegradation:
    """Slice 5: AGE has no native CREATE FULLTEXT INDEX; the GIN /
    tsvector polyfill ships in slice 6. Until then both index
    methods degrade to no-op + warning so extraction
    (ontology_helper.py:539, :543, :589, :593 — 4+ callsites each)
    doesn't crash on STORAGE_TYPE=postgres."""

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_node_index_no_op_returns_none(
        self, storage: PostgresAGEStorage
    ) -> None:
        # No raise, returns None (the abstract method is -> None).
        result = await storage.create_or_replace_ft_index_for_node(
            "ix_person_name", "Person", ["name"]
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_relationship_index_no_op_returns_none(
        self, storage: PostgresAGEStorage
    ) -> None:
        result = await storage.create_or_replace_ft_index_for_relationship(
            "ix_works_at", "Person", "WORKS_AT", "Company", ["role"]
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_node_index_does_not_call_cypher(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Pin that the no-op doesn't accidentally hit the database;
        # extraction calls these 4+ times per ontology and a stray
        # round-trip per call adds up.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.create_or_replace_ft_index_for_node("ix", "Person", ["name"])
        mock_exec.assert_not_called()


class TestFindSimilarNodes:
    """Slice 6: real find_similar_nodes via AGE Cypher CONTAINS
    scoring. Replaces the slice-3 round-2 degraded-to-empty
    fallback. Merge flow's similarity fallback
    (services/merge/new_merger.py:1068) now actually returns
    candidates instead of always falling through to "keep unmatched."
    """

    @pytest.fixture
    def storage(self) -> PostgresAGEStorage:
        return PostgresAGEStorage(dsn="postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_scoring_properties(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Empty properties → no signal to score on → return empty
        # without hitting the database.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            result = await storage.find_similar_nodes(label="Person", properties={})
        assert result == []
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_filters_out_system_properties(
        self, storage: PostgresAGEStorage
    ) -> None:
        # SYSTEM_PROPERTIES values must not enter the scoring
        # expression — id, __tid, etc. are metadata, not signal.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.find_similar_nodes(
                label="Person",
                properties={
                    "name": "Alice",
                    "id": "some-uuid",
                    TRANSFORM_ID: "t-42",
                    "confidence_score": 0.9,
                },
            )
        cypher = mock_exec.call_args.args[0]
        assert "n.name" in cypher
        assert "n.id" not in cypher
        assert "n.__tid" not in cypher

    @pytest.mark.asyncio
    async def test_filters_out_a1_prov_source_span_properties(
        self, storage: PostgresAGEStorage
    ) -> None:
        # A1-prov source-span fields stamped on every node by
        # services/transform/helpers.py::_attach_provenance_properties
        # are metadata, not entity signal. Without filtering, two
        # nodes extracted from the same document would score
        # artificially similar on source_text / document_name
        # overlap — that's the merge-contract bug the slice-6 review
        # caught.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.find_similar_nodes(
                label="Person",
                properties={
                    "name": "Alice",
                    "source_chunk_id": "chunk-7",
                    "document_name": "report.pdf",
                    "source_text": "Alice is a software engineer at Acme.",
                    "document_id": "doc-42",
                    "chunk_offset": 1024,
                    "page_number": 3,
                    "extraction_confidence": 0.87,
                },
            )
        cypher = mock_exec.call_args.args[0]
        assert "n.name" in cypher
        for leaked in (
            "n.source_chunk_id",
            "n.document_name",
            "n.source_text",
            "n.document_id",
            "n.chunk_offset",
            "n.page_number",
            "n.extraction_confidence",
        ):
            assert leaked not in cypher, f"{leaked} should not enter scoring"
        # Score normalizes by 1 (only "name" is a real entity field).
        assert "/ 1.0" in cypher

    @pytest.mark.asyncio
    async def test_filters_out_b0_decision_trail_properties(
        self, storage: PostgresAGEStorage
    ) -> None:
        # B0-prov-extend decision-trail fields are LLM telemetry, not
        # entity signal. Two nodes extracted by the same model would
        # otherwise score artificially similar on extractor_model /
        # prompt_version overlap.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.find_similar_nodes(
                label="Person",
                properties={
                    "name": "Alice",
                    "extractor_model": "gemini-1.5-pro",
                    "prompt_version": "v1.0.0",
                    "validator_score": 0.92,
                },
            )
        cypher = mock_exec.call_args.args[0]
        assert "n.name" in cypher
        for leaked in (
            "n.extractor_model",
            "n.prompt_version",
            "n.validator_score",
        ):
            assert leaked not in cypher, f"{leaked} should not enter scoring"
        assert "n.confidence_score" not in cypher

    @pytest.mark.asyncio
    async def test_skips_when_only_system_properties_supplied(
        self, storage: PostgresAGEStorage
    ) -> None:
        # All-system-properties input → nothing to score → return
        # empty without a DB hit.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            result = await storage.find_similar_nodes(
                label="Person",
                properties={"id": "x", TRANSFORM_ID: "t"},
            )
        assert result == []
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_emits_case_insensitive_contains_score_expr(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Pin the score-expression shape — scoring is the contract,
        # so a refactor that drops toLower or CONTAINS should fail
        # the test.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.find_similar_nodes(
                label="Person", properties={"name": "Alice"}
            )
        cypher = mock_exec.call_args.args[0]
        params = mock_exec.call_args.kwargs["params"]
        assert "MATCH (n:Person)" in cypher
        assert "toLower(toString(coalesce(n.name, '')))" in cypher
        assert "CONTAINS toLower($value0)" in cypher
        assert "WHERE similarity_score >= $threshold" in cypher
        assert "ORDER BY similarity_score DESC" in cypher
        assert "LIMIT $max_results" in cypher
        assert params["value0"] == "Alice"
        assert params["threshold"] == 0.7  # default

    @pytest.mark.asyncio
    async def test_score_normalizes_by_property_count(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Three scoring properties → score = sum / 3.0; pin the
        # denominator so a future tweak doesn't quietly break the
        # threshold's meaning.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.find_similar_nodes(
                label="Person",
                properties={
                    "name": "Alice",
                    "email": "alice@example.com",
                    "title": "engineer",
                },
            )
        cypher = mock_exec.call_args.args[0]
        # Score sum is divided by 3.0 (one per property).
        assert "/ 3.0" in cypher

    @pytest.mark.asyncio
    async def test_returns_mapped_nodes_in_score_order(
        self, storage: PostgresAGEStorage
    ) -> None:
        # AGE returns rows ordered by the ORDER BY; the adapter
        # preserves that order in the returned list.
        rows = [
            (
                '{"id": 0, "label": "Person", "properties": '
                '{"id": "n1", "name": "Alice"}}::vertex',
                "1.0",
            ),
            (
                '{"id": 1, "label": "Person", "properties": '
                '{"id": "n2", "name": "Alic"}}::vertex',
                "0.85",
            ),
        ]
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=rows)):
            nodes = await storage.find_similar_nodes(
                label="Person", properties={"name": "Alice"}
            )
        assert [n.id for n in nodes] == ["n1", "n2"]

    @pytest.mark.asyncio
    async def test_dedupes_by_node_id(self, storage: PostgresAGEStorage) -> None:
        dup = (
            '{"id": 0, "label": "Person", "properties": '
            '{"id": "n1", "name": "Alice"}}::vertex'
        )
        rows = [(dup, "1.0"), (dup, "1.0")]
        with patch.object(storage, "_execute_cypher", new=AsyncMock(return_value=rows)):
            nodes = await storage.find_similar_nodes(
                label="Person", properties={"name": "Alice"}
            )
        assert len(nodes) == 1

    @pytest.mark.asyncio
    async def test_validates_label(self, storage: PostgresAGEStorage) -> None:
        with pytest.raises(CypherInjectionError):
            await storage.find_similar_nodes(
                label="Per;son", properties={"name": "Alice"}
            )

    @pytest.mark.asyncio
    async def test_validates_property_names(self, storage: PostgresAGEStorage) -> None:
        with pytest.raises(CypherInjectionError):
            await storage.find_similar_nodes(
                label="Person", properties={"name; DROP": "Alice"}
            )

    @pytest.mark.asyncio
    async def test_passes_threshold_and_max_results_through(
        self, storage: PostgresAGEStorage
    ) -> None:
        # Caller-overridable parameters reach the params payload —
        # merge flow tunes both.
        with patch.object(
            storage, "_execute_cypher", new=AsyncMock(return_value=[])
        ) as mock_exec:
            await storage.find_similar_nodes(
                label="Person",
                properties={"name": "Alice"},
                similarity_threshold=0.85,
                max_results=25,
            )
        params = mock_exec.call_args.kwargs["params"]
        assert params["threshold"] == 0.85
        assert params["max_results"] == 25
