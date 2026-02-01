"""Mock Neo4j driver and session for London School TDD unit tests.

These mocks allow testing of Neo4jStorage without a real database connection.
They focus on verifying interactions (queries executed, parameters passed)
rather than internal state.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class MockNeo4jRecord:
    """Mock Neo4j record for test data.

    Provides dict-like access to record data, matching the real
    Neo4j record interface.
    """

    data: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def keys(self) -> List[str]:
        return list(self.data.keys())

    def values(self) -> List[Any]:
        return list(self.data.values())

    def items(self):
        return self.data.items()


@dataclass
class MockNeo4jNode:
    """Mock Neo4j node with labels and properties.

    Simulates a Neo4j graph node for query result testing.
    """

    labels: List[str]
    _properties: Dict[str, Any] = field(default_factory=dict)
    element_id: str = "node-mock-id"

    def items(self):
        return self._properties.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._properties.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self._properties


@dataclass
class MockNeo4jRelationship:
    """Mock Neo4j relationship.

    Simulates a Neo4j graph relationship for query result testing.
    """

    type: str
    start_node: MockNeo4jNode
    end_node: MockNeo4jNode
    _properties: Dict[str, Any] = field(default_factory=dict)
    element_id: str = "rel-mock-id"

    def items(self):
        return self._properties.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._properties.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._properties[key]


class MockNeo4jResult:
    """Mock Neo4j query result.

    Provides an async iterable over mock records, matching the
    real Neo4j Result interface for async operations.
    """

    def __init__(self, records: List[MockNeo4jRecord] = None):
        self._records = records or []
        self._index = 0

    async def single(self, default=None):
        """Get single record or default if empty."""
        return self._records[0] if self._records else default

    async def values(self):
        """Get all record values."""
        return [[r.data for r in self._records]]

    async def consume(self):
        """Consume remaining records."""
        pass

    async def data(self):
        """Get all records as list of dicts."""
        return [r.data for r in self._records]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index < len(self._records):
            record = self._records[self._index]
            self._index += 1
            return record
        raise StopAsyncIteration


class MockNeo4jSession:
    """Mock Neo4j async session.

    Captures executed queries for verification and returns
    configured results. This is the primary interaction point
    for testing query execution.

    Attributes:
        _query_results: Dict mapping query patterns to expected results.
        _executed_queries: List of queries executed during test.
        _raise_on_query: Optional exception to raise on query execution.
    """

    def __init__(
        self,
        query_results: Dict[str, List[MockNeo4jRecord]] = None,
        raise_on_query: Optional[Exception] = None,
    ):
        self._query_results = query_results or {}
        self._executed_queries: List[Dict[str, Any]] = []
        self._raise_on_query = raise_on_query
        self._closed = False

    async def run(
        self, query: str, parameters: Dict = None, **kwargs
    ) -> MockNeo4jResult:
        """Execute a Cypher query and return mock results.

        Records the query and parameters for later verification.
        """
        if self._raise_on_query:
            raise self._raise_on_query

        combined_params = {**(parameters or {}), **kwargs}
        self._executed_queries.append(
            {
                "query": query,
                "parameters": combined_params,
            }
        )

        # Find matching result by checking if pattern is in query
        for pattern, records in self._query_results.items():
            if pattern in query:
                return MockNeo4jResult(records)

        return MockNeo4jResult([])

    async def close(self):
        """Close the session."""
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    def executed_queries(self) -> List[Dict[str, Any]]:
        """Get list of executed queries for verification."""
        return self._executed_queries

    def get_queries_matching(self, pattern: str) -> List[Dict[str, Any]]:
        """Get queries containing the specified pattern."""
        return [q for q in self._executed_queries if pattern in q["query"]]

    def assert_query_executed(self, pattern: str, times: int = None):
        """Assert that a query matching pattern was executed.

        Args:
            pattern: Substring to match in query.
            times: Expected execution count. If None, asserts at least once.
        """
        matching = self.get_queries_matching(pattern)
        if times is not None:
            assert (
                len(matching) == times
            ), f"Expected query '{pattern}' to be executed {times} times, but was executed {len(matching)} times"
        else:
            assert (
                len(matching) > 0
            ), f"Expected query '{pattern}' to be executed at least once"

    def assert_query_not_executed(self, pattern: str):
        """Assert that no query matching pattern was executed."""
        matching = self.get_queries_matching(pattern)
        assert (
            len(matching) == 0
        ), f"Expected query '{pattern}' to not be executed, but found {len(matching)} executions"


class MockNeo4jDriver:
    """Mock Neo4j async driver.

    Provides session factory that returns configured mock sessions.
    """

    def __init__(
        self,
        session: MockNeo4jSession = None,
        session_factory: Callable[[], MockNeo4jSession] = None,
    ):
        self._session = session or MockNeo4jSession()
        self._session_factory = session_factory
        self._closed = False

    def session(self, database: str = None, **kwargs) -> MockNeo4jSession:
        """Get a mock session."""
        if self._session_factory:
            return self._session_factory()
        return self._session

    async def close(self):
        """Close the driver."""
        self._closed = True

    async def verify_connectivity(self):
        """Verify driver connectivity (always succeeds for mock)."""
        pass


@dataclass
class MockNeo4jStorage:
    """Container for mock Neo4j components.

    Provides convenient access to all mock components for test setup
    and verification.
    """

    driver: MockNeo4jDriver
    session: MockNeo4jSession

    @property
    def executed_queries(self) -> List[Dict[str, Any]]:
        """Get all executed queries from the session."""
        return self.session.executed_queries


def create_mock_neo4j_storage(
    query_results: Dict[str, List[MockNeo4jRecord]] = None,
    raise_on_query: Optional[Exception] = None,
) -> MockNeo4jStorage:
    """Factory function to create a complete mock storage setup.

    Args:
        query_results: Dict mapping query patterns to expected results.
        raise_on_query: Optional exception to raise on any query.

    Returns:
        MockNeo4jStorage with configured driver and session.

    Example:
        ```python
        mock_storage = create_mock_neo4j_storage(
            query_results={
                "MATCH (n:Company)": [
                    MockNeo4jRecord({"name": "Acme Corp"})
                ]
            }
        )
        ```
    """
    session = MockNeo4jSession(query_results or {}, raise_on_query)
    driver = MockNeo4jDriver(session)
    return MockNeo4jStorage(driver, session)
