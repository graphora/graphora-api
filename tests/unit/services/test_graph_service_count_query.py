"""Pin the count-query contract on
``GraphService.get_graph_by_transform_id``.

The pre-fix query did:

    MATCH (n) WHERE n.__tid = $transform_id
    WITH count(n) as node_count
    OPTIONAL MATCH (n)-[r]-()
    RETURN node_count, count(DISTINCT r) as edge_count

The ``WITH`` clause drops ``n`` from scope, so the OPTIONAL MATCH
rebinds ``n`` to ANY node in the database and the relationship
counter ends up summing every edge in Neo4j — not just the edges
the current transform produced. Users observed ``total_edges:
204`` on a 49-edge transform because the figure spilled across
transform boundaries.

These tests don't run against a live Neo4j (we have no GraphService
integration harness yet); they capture the query string passed to
``session.run`` and pin the relationship-side filter clause. The
integration suite still exists at the storage layer to verify the
write side; this test pins the read side at the cheapest level
that catches the regression.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_service_with_mock_driver():
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


class TestCountQueryFiltersRelationshipsByTransformId:
    """Pin the relationship-side filter so the FE never sees a
    cross-transform edge count again."""

    def test_count_query_applies_transform_id_filter_on_relationship(self) -> None:
        service, fake_driver = _make_service_with_mock_driver()

        session_cm = MagicMock()
        session = MagicMock()
        session_cm.__enter__.return_value = session
        session_cm.__exit__.return_value = None
        fake_driver.session.return_value = session_cm

        # Two run() calls happen: count query, then the data query.
        # Stub both so we don't crash before assertions land.
        count_result = MagicMock()
        count_result.single.return_value = {"node_count": 0, "edge_count": 0}
        data_result = MagicMock()
        data_result.single.return_value = {
            "nodes": [],
            "relationships": [],
            "connected_nodes": [],
        }
        session.run.side_effect = [count_result, data_result]

        service.get_graph_by_transform_id("tx-fixed-bug", limit=10, skip=0)

        # First positional arg of the first run() call is the count query.
        first_call = session.run.call_args_list[0]
        count_query = first_call.args[0]

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
        service, fake_driver = _make_service_with_mock_driver()

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

        service.get_graph_by_transform_id("tx-fixed-bug", limit=10, skip=0)

        count_query = session.run.call_args_list[0].args[0]

        # The buggy substring. If a future refactor re-introduces the
        # n-after-WITH-count pattern, this fails loud.
        assert "OPTIONAL MATCH (n)-[r]-()" not in count_query, (
            "Count query is back to the n-after-WITH form that scopes "
            "across the whole DB; restore the relationship-side filter."
        )
