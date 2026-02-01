"""Mock objects for London School TDD tests."""

from .neo4j_mock import (
    MockNeo4jRecord,
    MockNeo4jNode,
    MockNeo4jRelationship,
    MockNeo4jResult,
    MockNeo4jSession,
    MockNeo4jDriver,
    MockNeo4jStorage,
    create_mock_neo4j_storage,
)
from .llm_client_mock import (
    MockLLMResponse,
    MockLLMClient,
)
from .storage_mock import (
    MockStorageBatchResult,
    MockGraphStorage,
)
from .auth_mock import (
    MockAuthContext,
    create_mock_auth_dependency,
)

__all__ = [
    # Neo4j mocks
    "MockNeo4jRecord",
    "MockNeo4jNode",
    "MockNeo4jRelationship",
    "MockNeo4jResult",
    "MockNeo4jSession",
    "MockNeo4jDriver",
    "MockNeo4jStorage",
    "create_mock_neo4j_storage",
    # LLM mocks
    "MockLLMResponse",
    "MockLLMClient",
    # Storage mocks
    "MockStorageBatchResult",
    "MockGraphStorage",
    # Auth mocks
    "MockAuthContext",
    "create_mock_auth_dependency",
]
