"""Pin the transform-scoped read-query contract on the two
read paths the FE consumes:

- ``GraphService.get_graph_by_transform_id`` (sync driver)
- ``Neo4jStorage.get_transformation_data``     (async driver)

The pre-fix queries (both files) did:

    MATCH (n) WHERE n.__tid = $transform_id
    WITH count(n) as node_count
    OPTIONAL MATCH (n)-[r]-()
    RETURN node_count, count(DISTINCT r) as edge_count

The ``WITH`` clause drops ``n`` from scope; the OPTIONAL MATCH
rebinds it to ANY node in the database, so the relationship
counter ends up summing every edge in Neo4j — not just the
edges the current transform produced. Users observed
``total_edges: 204`` on a 49-edge transform because the figure
spilled across transform boundaries. The data queries had a
mirror gap: ``OPTIONAL MATCH (n)-[r]-(m)`` collected edges
from other transforms that happened to touch one of the
selected nodes, so the relationships list disagreed with the
total count.

These tests don't run against a live Neo4j (no integration
harness for either reader); they capture the query string
passed to ``session.run`` and pin the relationship-side filter
clauses at the cheapest level that catches the regression.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


# ----------------------------------------------------------------
# GraphService (sync driver) helpers
# ----------------------------------------------------------------


def _make_graph_service_with_mock_driver():
    """Build a GraphService with the Neo4j driver replaced by a mock.

    GraphDatabase is local-imported inside ``__init__`` to avoid
    pulling in the driver at module load. Patching it at the
    ``neo4j`` module level catches the import path the constructor
    actually uses."""
    from graphora_server.services.graph_service import GraphService

    fake_driver = MagicMock()
    with patch("neo4j.GraphDatabase") as graph_db_cls:
        graph_db_cls.driver.return_value = fake_driver
        service = GraphService("bolt://stub", "user", "pw")
    return service, fake_driver


def _stub_graph_service_session(fake_driver):
    """Wire up a sync session that returns empty results for both
    the count query and the data query. Returns the inner session
    mock so callers can introspect ``session.run.call_args_list``."""
    session_cm = MagicMock()
    session = MagicMock()
    session_cm.__enter__.return_value = session
    session_cm.__exit__.return_value = None
    fake_driver.session.return_value = session_cm

    count_result = MagicMock()
    count_result.single.return_value = {"node_count": 0, "edge_count": 0}
    data_result = MagicMock()
    data_result.single.return_value = {
        "nodes": [],
        "relationships": [],
        "connected_nodes": [],
    }
    session.run.side_effect = [count_result, data_result]
    return session


class TestGraphServiceCountQuery:
    """Pin the count-query contract in graph_service.py."""

    def test_count_query_applies_transform_id_filter_on_relationship(self) -> None:
        service, fake_driver = _make_graph_service_with_mock_driver()
        session = _stub_graph_service_session(fake_driver)

        service.get_graph_by_transform_id("tx-fixed-bug", limit=10, skip=0)

        # First positional arg of the first run() call is the count query.
        count_query = session.run.call_args_list[0].args[0]

        # The relationship match must filter by the relationship's own
        # transform-id property; without this filter, the count spans
        # the whole database. ``__tid`` is the canonical TRANSFORM_ID
        # constant used everywhere in the codebase.
        assert "r.__tid = $transform_id" in count_query, (
            "Count query no longer filters relationships by transform_id; "
            "regression risk — the FE will see cross-transform edge counts."
        )

    def test_count_query_does_not_use_undefined_n_after_with(self) -> None:
        """The pre-fix query referenced ``n`` after a ``WITH count(n)``
        that drops it from scope, which Neo4j happily reinterprets as
        a fresh anonymous node. Pin that this exact shape is gone so
        a refactor can't quietly reintroduce it."""
        service, fake_driver = _make_graph_service_with_mock_driver()
        session = _stub_graph_service_session(fake_driver)

        service.get_graph_by_transform_id("tx-fixed-bug", limit=10, skip=0)

        count_query = session.run.call_args_list[0].args[0]
        assert "OPTIONAL MATCH (n)-[r]-()" not in count_query, (
            "Count query is back to the n-after-WITH form that scopes "
            "across the whole DB; restore the relationship-side filter."
        )


class TestGraphServiceDataQuery:
    """Pin the relationship-side filter on the paginated data
    fetch. Without it the response returns cross-transform edges
    that the count query (now correctly scoped) doesn't see, so
    ``len(edges) != total_edges`` and the FE shows a mismatch."""

    def test_data_query_applies_transform_id_filter_on_relationship(self) -> None:
        service, fake_driver = _make_graph_service_with_mock_driver()
        session = _stub_graph_service_session(fake_driver)

        service.get_graph_by_transform_id("tx-fixed-bug", limit=10, skip=0)

        # Second run() call is the data query; its query is the
        # first positional arg.
        data_query = session.run.call_args_list[1].args[0]
        assert "r.__tid = $transform_id" in data_query, (
            "Paginated data query no longer filters relationships by "
            "transform_id; len(edges) will diverge from total_edges."
        )


# ----------------------------------------------------------------
# Neo4jStorage (async driver) helpers
# ----------------------------------------------------------------


def _stub_neo4j_storage_session(storage):
    """Wire an async session onto a Neo4jStorage instance and stub
    both run() calls. Returns the inner session mock for assertions.

    Neo4jStorage uses ``async with self.driver.session() as session``
    and ``await session.run(...)`` — both halves need AsyncMock
    behaviour. driver.session() itself is sync (returns the async
    context manager), so it stays a regular MagicMock."""
    session = MagicMock()
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    fake_driver = MagicMock()
    fake_driver.session = MagicMock(return_value=session_cm)
    storage.driver = fake_driver

    count_result = MagicMock()
    count_result.single = AsyncMock(
        return_value={"node_count": 0, "edge_count": 0}
    )
    data_result = MagicMock()
    data_result.single = AsyncMock(
        return_value={
            "nodes": [],
            "relationships": [],
            "connected_nodes": [],
        }
    )
    session.run = AsyncMock(side_effect=[count_result, data_result])
    return session


class TestNeo4jStorageTransformationDataCountQuery:
    """Same regression as graph_service.py, mirrored in the storage
    adapter. Anything calling ``get_transformation_data`` directly
    (legacy paths, the test harness, anything bypassing
    GraphService) was hitting the same DB-wide leak."""

    async def test_count_query_applies_transform_id_filter_on_relationship(
        self,
    ) -> None:
        from graphora_server.services.storage.neo4j import Neo4jStorage

        storage = Neo4jStorage.__new__(Neo4jStorage)
        session = _stub_neo4j_storage_session(storage)

        await storage.get_transformation_data("tx-fixed-bug")

        count_query = session.run.call_args_list[0].args[0]
        assert "r.__tid = $transform_id" in count_query, (
            "Neo4jStorage count query no longer filters relationships "
            "by transform_id; total_edges spans the whole DB."
        )

    async def test_count_query_does_not_use_undefined_n_after_with(self) -> None:
        from graphora_server.services.storage.neo4j import Neo4jStorage

        storage = Neo4jStorage.__new__(Neo4jStorage)
        session = _stub_neo4j_storage_session(storage)

        await storage.get_transformation_data("tx-fixed-bug")

        count_query = session.run.call_args_list[0].args[0]
        assert "OPTIONAL MATCH (n)-[r]-()" not in count_query, (
            "Neo4jStorage count query is back to the n-after-WITH form; "
            "restore the relationship-side filter."
        )

    async def test_data_query_applies_transform_id_filter_on_relationship(
        self,
    ) -> None:
        from graphora_server.services.storage.neo4j import Neo4jStorage

        storage = Neo4jStorage.__new__(Neo4jStorage)
        session = _stub_neo4j_storage_session(storage)

        await storage.get_transformation_data("tx-fixed-bug")

        data_query = session.run.call_args_list[1].args[0]
        assert "r.__tid = $transform_id" in data_query, (
            "Neo4jStorage data query no longer filters relationships by "
            "transform_id; the relationships collection will include "
            "cross-transform edges."
        )
