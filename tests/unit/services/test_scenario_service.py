"""Unit tests for ScenarioService (B6-scenario slice 1).

Scenarios are named, point-in-time snapshots of a transform's
graph. Slice 1 lands the foundation (create / list / get /
delete) with dual-backend storage. These tests pin both directions:

  * Memory backend (zero-config dev mode): full round-trip,
    tenant scoping, name-unique conflict, delete idempotency
    posture (404 not 204 for non-existent ids).
  * Postgres-shape concerns are mocked at the db helper layer
    so we can pin SQL parameter binding + the conflict-error
    mapping without standing up a real Postgres instance.
"""

from __future__ import annotations


import pytest
from unittest.mock import AsyncMock, patch

from graphora_server.config import settings
from graphora_server.schemas.graph import Edge, GraphResponse, Node
from graphora_server.services.scenario_service import (
    ScenarioConflictError,
    ScenarioNotFoundError,
    ScenarioService,
    _DEFAULT_MEMORY_STORE,
    _reset_default_memory_store_for_tests,
)


def _graph(*, n_nodes: int = 1, n_edges: int = 0) -> GraphResponse:
    """Build a minimum-viable GraphResponse for snapshotting."""
    return GraphResponse(
        nodes=[
            Node(
                id=f"n{i}",
                label=f"Node{i}",
                type="Person",
                properties={"name": f"Node{i}"},
            )
            for i in range(n_nodes)
        ],
        edges=[
            Edge(
                id=f"e{i}",
                source=f"n{i}",
                target=f"n{i+1}",
                type="KNOWS",
                properties={},
            )
            for i in range(n_edges)
        ],
        total_nodes=n_nodes,
        total_edges=n_edges,
    )


@pytest.fixture
def memory_service(monkeypatch):
    """Force memory mode by disabling DATABASE_URL on the settings
    module. Mirrors the decision_log / entity_ledger test
    fixtures so the fixture shape is portable across services."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    return ScenarioService(memory_store=[])


# ============================================================
# Memory backend — create round-trip
# ============================================================


@pytest.mark.asyncio
async def test_create_from_transform_materializes_snapshot(memory_service):
    """create_from_transform(graph) must store the graph as the
    snapshot, populate id + created_at, and surface it via
    list/get without modification. Pin the minimal round-trip
    so a refactor that drops the snapshot or scrambles fields
    regresses noisily."""
    record = await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-1",
        name="baseline",
        graph=_graph(n_nodes=2, n_edges=1),
        description="initial snapshot",
    )

    assert record.id is not None
    assert record.created_at is not None
    assert record.user_id == "user-1"
    assert record.name == "baseline"
    assert record.description == "initial snapshot"
    # Snapshot must round-trip the full graph shape.
    assert len(record.graph_snapshot["nodes"]) == 2
    assert len(record.graph_snapshot["edges"]) == 1

    # list_for_user surfaces it.
    listed = await memory_service.list_for_user("user-1")
    assert len(listed) == 1
    assert listed[0].id == record.id

    # get(id, user_id) returns the same record.
    fetched = await memory_service.get(record.id, "user-1")
    assert fetched.id == record.id
    assert fetched.name == "baseline"


# ============================================================
# Memory backend — uniqueness
# ============================================================


@pytest.mark.asyncio
async def test_create_duplicate_name_for_same_transform_conflicts(
    memory_service,
):
    """The (user_id, transform_id, name) tuple is the uniqueness
    boundary. Pin so a regression that drops the pre-flight
    check (or the underlying DB constraint) surfaces here
    instead of producing two scenarios that race the get-by-name
    case to follow."""
    await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-1",
        name="dup",
        graph=_graph(),
    )
    with pytest.raises(ScenarioConflictError):
        await memory_service.create_from_transform(
            user_id="user-1",
            transform_id="tx-1",
            name="dup",
            graph=_graph(),
        )


@pytest.mark.asyncio
async def test_same_name_different_transform_does_not_conflict(
    memory_service,
):
    """The unique key includes transform_id — the same scenario
    name across different transforms is legitimate (e.g.,
    'baseline' for both tx-A and tx-B). Pin so the constraint
    isn't accidentally widened to (user_id, name)."""
    await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-A",
        name="baseline",
        graph=_graph(),
    )
    # Different transform_id — must NOT conflict.
    await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-B",
        name="baseline",
        graph=_graph(),
    )
    listed = await memory_service.list_for_user("user-1")
    assert len(listed) == 2


@pytest.mark.asyncio
async def test_same_name_different_user_does_not_conflict(memory_service):
    """The unique key also pins user_id — two users naming their
    scenarios identically is fine and expected. Pin so the
    constraint isn't accidentally lifted to (transform_id, name)
    only, which would leak across tenants."""
    await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-1",
        name="baseline",
        graph=_graph(),
    )
    await memory_service.create_from_transform(
        user_id="user-2",
        transform_id="tx-1",
        name="baseline",
        graph=_graph(),
    )
    listed_1 = await memory_service.list_for_user("user-1")
    listed_2 = await memory_service.list_for_user("user-2")
    assert len(listed_1) == 1
    assert len(listed_2) == 1


# ============================================================
# Tenant scoping
# ============================================================


@pytest.mark.asyncio
async def test_list_returns_only_callers_scenarios(memory_service):
    """list_for_user must filter on user_id. Cross-tenant leak
    is the worst failure for an observability-flavored API
    surface — pin so a regression that drops the filter is
    caught at the boundary."""
    await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-1",
        name="a",
        graph=_graph(),
    )
    await memory_service.create_from_transform(
        user_id="user-2",
        transform_id="tx-2",
        name="b",
        graph=_graph(),
    )

    user1_list = await memory_service.list_for_user("user-1")
    user2_list = await memory_service.list_for_user("user-2")

    assert {s.name for s in user1_list} == {"a"}
    assert {s.name for s in user2_list} == {"b"}


@pytest.mark.asyncio
async def test_get_with_wrong_user_raises_not_found(memory_service):
    """get(id, user_id) must reject cross-tenant access with the
    SAME error a missing id produces — never leak existence.
    Pin so a regression that returns the record (or surfaces a
    different exception type) is caught here."""
    record = await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-1",
        name="a",
        graph=_graph(),
    )
    with pytest.raises(ScenarioNotFoundError):
        await memory_service.get(record.id, "user-2")


@pytest.mark.asyncio
async def test_get_unknown_id_raises_not_found(memory_service):
    """The same exception type fires for a nonexistent id.
    Pin so the two cases (cross-tenant + nonexistent) remain
    indistinguishable to callers."""
    with pytest.raises(ScenarioNotFoundError):
        await memory_service.get("not-a-real-id", "user-1")


# ============================================================
# Delete
# ============================================================


@pytest.mark.asyncio
async def test_delete_removes_scenario(memory_service):
    """delete(id, user_id) must remove the row so a subsequent
    get raises NotFound. Pin the minimal delete contract."""
    record = await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-1",
        name="a",
        graph=_graph(),
    )
    await memory_service.delete(record.id, "user-1")
    with pytest.raises(ScenarioNotFoundError):
        await memory_service.get(record.id, "user-1")


@pytest.mark.asyncio
async def test_delete_cross_tenant_raises_not_found(memory_service):
    """delete with the wrong user_id must reject and leave the
    record intact. Combined with the get-cross-tenant test, this
    pins the full 'no cross-tenant access' invariant."""
    record = await memory_service.create_from_transform(
        user_id="user-1",
        transform_id="tx-1",
        name="a",
        graph=_graph(),
    )
    with pytest.raises(ScenarioNotFoundError):
        await memory_service.delete(record.id, "user-2")
    # Owner can still see it.
    fetched = await memory_service.get(record.id, "user-1")
    assert fetched.id == record.id


@pytest.mark.asyncio
async def test_delete_unknown_id_raises_not_found_not_silent_204(
    memory_service,
):
    """Deleting a nonexistent id must raise, not silently 'succeed'.
    Same posture as get — a silent 204 would let an attacker
    probe existence by timing the response. Pin so a future
    refactor that 'fixes' the behavior to be idempotent
    regresses noisily."""
    with pytest.raises(ScenarioNotFoundError):
        await memory_service.delete("not-a-real-id", "user-1")


# ============================================================
# Postgres backend — SQL parameter binding pins
# ============================================================


@pytest.mark.asyncio
async def test_postgres_create_binds_user_id_and_snapshot(monkeypatch):
    """Pin the Postgres write path: the INSERT must bind user_id +
    transform_id + a JSON snapshot of the graph (via psycopg's
    Json wrapper). A refactor that drops user_id or sends the
    graph as text would silently break tenant scoping or storage
    compatibility — keep the contract pinned at the SQL layer."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://fake/url")
    service = ScenarioService()
    assert service._enabled

    fake_execute = AsyncMock()
    fake_fetch = AsyncMock(return_value=[])  # no name conflict

    with (
        patch("graphora_server.services.scenario_service.db.execute", new=fake_execute),
        patch("graphora_server.services.scenario_service.db.fetch", new=fake_fetch),
    ):
        await service.create_from_transform(
            user_id="user-pg",
            transform_id="tx-pg",
            name="pg-baseline",
            graph=_graph(n_nodes=1),
        )

    fake_execute.assert_called_once()
    # Args are (sql, id, user_id, transform_id, parent, name,
    # description, json_snapshot, created_at). Pin the SQL +
    # positional params.
    args = fake_execute.await_args.args
    sql = args[0]
    assert "INSERT INTO scenarios" in sql
    assert args[2] == "user-pg"  # user_id position
    assert args[3] == "tx-pg"  # transform_id position
    assert args[5] == "pg-baseline"  # name position
    # graph_snapshot is wrapped in Json() — confirm the type so
    # a refactor passing a raw dict (which psycopg would reject)
    # would fail here.
    from psycopg.types.json import Json

    assert isinstance(args[7], Json)


@pytest.mark.asyncio
async def test_postgres_create_maps_unique_violation_to_conflict(
    monkeypatch,
):
    """Even with the pre-flight check, a concurrent INSERT race
    can land both writes into the same uniqueness slot — the DB
    raises a UniqueViolation. The service must catch that and
    re-raise ScenarioConflictError so the API still returns 409.
    Pin so a refactor that strips the error-string sniff regresses."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://fake/url")
    service = ScenarioService()

    # Pre-flight: no row found.
    fake_fetch = AsyncMock(return_value=[])
    # INSERT: raise a Postgres-flavored unique violation.
    fake_execute = AsyncMock(
        side_effect=Exception(
            "duplicate key value violates unique constraint "
            '"scenarios_name_unique_per_transform"'
        )
    )

    with (
        patch("graphora_server.services.scenario_service.db.execute", new=fake_execute),
        patch("graphora_server.services.scenario_service.db.fetch", new=fake_fetch),
    ):
        with pytest.raises(ScenarioConflictError):
            await service.create_from_transform(
                user_id="user-pg",
                transform_id="tx-pg",
                name="race-name",
                graph=_graph(),
            )


# ============================================================
# Reviewer-fix: shared dev-mode store across instances
# ============================================================


@pytest.fixture
def shared_store_reset(monkeypatch):
    """Force memory mode AND clear the module-level store before
    + after each test. The shared store is process-wide by design
    (mirrors DisputedPairsService), so tests that exercise it
    must isolate themselves explicitly."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    _reset_default_memory_store_for_tests()
    yield
    _reset_default_memory_store_for_tests()


@pytest.mark.asyncio
async def test_default_constructor_shares_memory_across_instances(
    shared_store_reset,
):
    """Reviewer-flagged High on commit d7a1f6e. Default-constructed
    services (no ``memory_store=`` arg) must share the same
    underlying list so dev-mode CRUD persists across API
    requests — each request gets its own service instance.

    Pre-fix each instance allocated a fresh empty list, so a
    POST landed data in one and a follow-up GET read from
    another. Pin so a refactor that re-introduces per-instance
    allocation regresses noisily."""
    writer = ScenarioService()
    reader = ScenarioService()

    record = await writer.create_from_transform(
        user_id="user-shared",
        transform_id="tx-shared",
        name="visible-everywhere",
        graph=_graph(),
    )

    # A different service instance must see the same record —
    # this is the load-bearing assertion.
    listed = await reader.list_for_user("user-shared")
    assert len(listed) == 1
    assert listed[0].id == record.id, (
        "Default-constructed service instances must share the "
        "dev-mode memory store. The reader saw an empty list, "
        "which means writes are lost between API requests."
    )

    # get also resolves cross-instance.
    fetched = await reader.get(record.id, "user-shared")
    assert fetched.id == record.id


@pytest.mark.asyncio
async def test_explicit_memory_store_isolates_from_shared(
    shared_store_reset,
):
    """The ``memory_store=[]`` test escape hatch must NOT share
    state with the module-level dev store — otherwise tests
    pollute the shared store and break later runs. Pin so a
    refactor that removes the per-instance override regresses."""
    isolated = ScenarioService(memory_store=[])
    shared = ScenarioService()

    await isolated.create_from_transform(
        user_id="user-x",
        transform_id="tx-x",
        name="in-isolated",
        graph=_graph(),
    )

    # Shared store must NOT see the write.
    shared_listed = await shared.list_for_user("user-x")
    assert shared_listed == []
    # Module-level store stays empty.
    assert _DEFAULT_MEMORY_STORE == []


# ============================================================
# Reviewer-fix: DB exceptions propagate (don't masquerade as
# empty / not-found)
# ============================================================


@pytest.mark.asyncio
async def test_postgres_list_propagates_db_exception(monkeypatch):
    """Reviewer-flagged High on commit d7a1f6e. Scenarios are
    user-owned data, not best-effort observability — a DB outage
    must NOT masquerade as "you have no scenarios" (empty list).
    Pin that list_for_user propagates DB exceptions so the API
    surfaces them as 5xx."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://fake/url")
    service = ScenarioService()

    fake_fetch = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("graphora_server.services.scenario_service.db.fetch", new=fake_fetch):
        with pytest.raises(RuntimeError, match="connection refused"):
            await service.list_for_user("user-x")


@pytest.mark.asyncio
async def test_postgres_get_propagates_db_exception(monkeypatch):
    """Same posture as list: a "DB on fire" outcome must not
    surface as 404. The API maps 5xx → user "the system is
    broken"; the get-NotFound mapping is only for actual missing
    rows (rows=[] after a successful query)."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://fake/url")
    service = ScenarioService()

    fake_fetch = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("graphora_server.services.scenario_service.db.fetch", new=fake_fetch):
        with pytest.raises(RuntimeError, match="connection refused"):
            await service.get("sc-1", "user-x")


@pytest.mark.asyncio
async def test_postgres_delete_propagates_db_exception(monkeypatch):
    """Same posture as list + get. Delete must propagate DB
    failures so the API can return 5xx — a 404 here would tell
    a malicious caller "the row doesn't exist" even when the
    real answer is "the DB is unreachable.""" ""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://fake/url")
    service = ScenarioService()

    fake_fetch = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("graphora_server.services.scenario_service.db.fetch", new=fake_fetch):
        with pytest.raises(RuntimeError, match="connection refused"):
            await service.delete("sc-1", "user-x")
