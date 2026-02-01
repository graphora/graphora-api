"""Mock graph storage interface for London School TDD unit tests.

These mocks implement the GraphStorageInterface contract without
requiring a real database. They focus on verifying interactions
and returning configured responses.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class MockStorageBatchResult:
    """Mock result from a batch storage operation."""

    batch_index: int
    items_processed: int
    items_failed: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class MockNode:
    """Mock node for storage responses."""

    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockEdge:
    """Mock edge for storage responses."""

    id: str
    type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockGraphResponse:
    """Mock graph query response."""

    nodes: List[MockNode] = field(default_factory=list)
    edges: List[MockEdge] = field(default_factory=list)


class MockGraphStorage:
    """Configurable mock graph storage implementation.

    Tracks all method calls and returns configured responses.
    Implements the GraphStorageInterface contract for testing.

    Example:
        ```python
        storage = MockGraphStorage()
        storage.configure_node_lookup({
            "node-123": MockNode(id="node-123", type="Company", properties={"name": "Acme"})
        })

        node = await storage.get_node_by_id("node-123")
        assert node.properties["name"] == "Acme"
        storage.assert_called("get_node_by_id")
        ```
    """

    def __init__(self):
        # Configured responses
        self._nodes: Dict[str, MockNode] = {}
        self._edges: Dict[str, MockEdge] = {}
        self._similar_nodes: Dict[str, List[MockNode]] = {}
        self._transformation_data: Dict[str, MockGraphResponse] = {}
        self._query_results: Dict[str, List[Dict[str, Any]]] = {}

        # Call tracking
        self._call_counts: Dict[str, int] = {
            "store_nodes": 0,
            "store_relationships": 0,
            "get_node_by_id": 0,
            "get_transformation_data": 0,
            "find_similar_nodes": 0,
            "execute_query": 0,
            "create_ft_index": 0,
            "drop_ft_index": 0,
        }

        self._call_args: Dict[str, List[Dict[str, Any]]] = {
            method: [] for method in self._call_counts.keys()
        }

        # Error configuration
        self._raise_on_method: Dict[str, Exception] = {}

        # Stored data tracking (what was "stored")
        self._stored_nodes: List[Any] = []
        self._stored_relationships: List[Any] = []

    def configure_node_lookup(self, nodes: Dict[str, MockNode]):
        """Configure nodes to return on get_node_by_id."""
        self._nodes = nodes

    def configure_similar_nodes(self, results: Dict[str, List[MockNode]]):
        """Configure results for find_similar_nodes by node type."""
        self._similar_nodes = results

    def configure_transformation_data(self, data: Dict[str, MockGraphResponse]):
        """Configure results for get_transformation_data by transform_id."""
        self._transformation_data = data

    def configure_query_results(self, results: Dict[str, List[Dict[str, Any]]]):
        """Configure results for execute_query by query pattern."""
        self._query_results = results

    def configure_error(self, method: str, exception: Exception):
        """Configure an error to be raised on method call."""
        self._raise_on_method[method] = exception

    def _check_error(self, method: str):
        """Check and raise configured error for method."""
        if method in self._raise_on_method:
            raise self._raise_on_method[method]

    async def store_nodes(
        self,
        nodes: List[Any],
        batch_index: int = 0,
        transform_id: Optional[str] = None,
        merge: bool = True,
        user_id: Optional[str] = None,
    ) -> MockStorageBatchResult:
        """Mock node storage operation."""
        self._check_error("store_nodes")
        self._call_counts["store_nodes"] += 1
        self._call_args["store_nodes"].append(
            {
                "nodes": nodes,
                "batch_index": batch_index,
                "transform_id": transform_id,
                "merge": merge,
                "user_id": user_id,
            }
        )
        self._stored_nodes.extend(nodes)

        return MockStorageBatchResult(
            batch_index=batch_index,
            items_processed=len(nodes),
            items_failed=0,
        )

    async def store_relationships(
        self,
        relationships: List[Any],
        batch_index: int = 0,
        transform_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> MockStorageBatchResult:
        """Mock relationship storage operation."""
        self._check_error("store_relationships")
        self._call_counts["store_relationships"] += 1
        self._call_args["store_relationships"].append(
            {
                "relationships": relationships,
                "batch_index": batch_index,
                "transform_id": transform_id,
                "user_id": user_id,
            }
        )
        self._stored_relationships.extend(relationships)

        return MockStorageBatchResult(
            batch_index=batch_index,
            items_processed=len(relationships),
            items_failed=0,
        )

    async def get_node_by_id(
        self,
        node_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[MockNode]:
        """Mock node lookup by ID."""
        self._check_error("get_node_by_id")
        self._call_counts["get_node_by_id"] += 1
        self._call_args["get_node_by_id"].append(
            {
                "node_id": node_id,
                "user_id": user_id,
            }
        )

        return self._nodes.get(node_id)

    async def get_transformation_data(
        self,
        transform_id: str,
        user_id: Optional[str] = None,
    ) -> MockGraphResponse:
        """Mock transformation data retrieval."""
        self._check_error("get_transformation_data")
        self._call_counts["get_transformation_data"] += 1
        self._call_args["get_transformation_data"].append(
            {
                "transform_id": transform_id,
                "user_id": user_id,
            }
        )

        return self._transformation_data.get(transform_id, MockGraphResponse())

    async def find_similar_nodes(
        self,
        node_type: str,
        properties: Dict[str, Any],
        threshold: float = 0.8,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[MockNode]:
        """Mock similar node search."""
        self._check_error("find_similar_nodes")
        self._call_counts["find_similar_nodes"] += 1
        self._call_args["find_similar_nodes"].append(
            {
                "node_type": node_type,
                "properties": properties,
                "threshold": threshold,
                "limit": limit,
                "user_id": user_id,
            }
        )

        return self._similar_nodes.get(node_type, [])[:limit]

    async def execute_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Mock query execution."""
        self._check_error("execute_query")
        self._call_counts["execute_query"] += 1
        self._call_args["execute_query"].append(
            {
                "query": query,
                "parameters": parameters or {},
                "user_id": user_id,
            }
        )

        # Find matching result by query pattern
        for pattern, results in self._query_results.items():
            if pattern in query:
                return results

        return []

    async def create_ft_index(
        self,
        index_name: str,
        label_or_type: str,
        properties: List[str],
        is_relationship: bool = False,
        user_id: Optional[str] = None,
    ) -> bool:
        """Mock full-text index creation."""
        self._check_error("create_ft_index")
        self._call_counts["create_ft_index"] += 1
        self._call_args["create_ft_index"].append(
            {
                "index_name": index_name,
                "label_or_type": label_or_type,
                "properties": properties,
                "is_relationship": is_relationship,
                "user_id": user_id,
            }
        )
        return True

    async def drop_ft_index(
        self,
        index_name: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Mock full-text index deletion."""
        self._check_error("drop_ft_index")
        self._call_counts["drop_ft_index"] += 1
        self._call_args["drop_ft_index"].append(
            {
                "index_name": index_name,
                "user_id": user_id,
            }
        )
        return True

    @property
    def call_counts(self) -> Dict[str, int]:
        """Get method call counts."""
        return self._call_counts.copy()

    @property
    def stored_nodes(self) -> List[Any]:
        """Get all nodes that were "stored"."""
        return self._stored_nodes.copy()

    @property
    def stored_relationships(self) -> List[Any]:
        """Get all relationships that were "stored"."""
        return self._stored_relationships.copy()

    def get_call_args(self, method: str) -> List[Dict[str, Any]]:
        """Get all call arguments for a method."""
        return self._call_args.get(method, [])

    def assert_called(self, method: str, times: int = None):
        """Assert that a method was called.

        Args:
            method: Method name to check.
            times: Expected call count. If None, asserts at least once.
        """
        count = self._call_counts.get(method, 0)
        if times is not None:
            assert (
                count == times
            ), f"Expected {method} to be called {times} times, but was called {count} times"
        else:
            assert count > 0, f"Expected {method} to be called at least once"

    def assert_not_called(self, method: str):
        """Assert that a method was not called."""
        count = self._call_counts.get(method, 0)
        assert (
            count == 0
        ), f"Expected {method} to not be called, but was called {count} times"

    def assert_nodes_stored(self, count: int):
        """Assert that specific number of nodes were stored."""
        actual = len(self._stored_nodes)
        assert (
            actual == count
        ), f"Expected {count} nodes to be stored, but {actual} were stored"

    def assert_relationships_stored(self, count: int):
        """Assert that specific number of relationships were stored."""
        actual = len(self._stored_relationships)
        assert (
            actual == count
        ), f"Expected {count} relationships to be stored, but {actual} were stored"

    def reset(self):
        """Reset all call tracking and stored data."""
        self._stored_nodes = []
        self._stored_relationships = []
        self._raise_on_method = {}

        for method in self._call_counts:
            self._call_counts[method] = 0
            self._call_args[method] = []
