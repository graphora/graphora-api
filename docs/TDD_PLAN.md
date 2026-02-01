# Graphora API - London School TDD Implementation Plan

## Executive Summary

This document outlines a comprehensive Test-Driven Development plan following the London School (mockist) approach for the Graphora API codebase. The goal is to achieve comprehensive test coverage that enables safe refactoring while preserving API contracts.

## Current State Analysis

### Existing Test Coverage

The codebase has **20 test files** with the following patterns:

| Directory | Files | Focus |
|-----------|-------|-------|
| `tests/` | 3 | Client auth, settings |
| `tests/api/` | 3 | OpenAPI schema, quality API, dashboard |
| `tests/integration/` | 1 | App endpoints |
| `tests/services/` | 2 | LLM cache, merge learning |
| `tests/quality/` | 2 | Quality service, validator |
| `tests/transform/` | 7 | Validators, status models, chunking, canonicalization, etc. |
| `tests/scripts/` | 1 | Migration scripts |

### Key Observations

1. **Good patterns already established:**
   - Test environment configuration in `conftest.py`
   - Neo4j, Splink, and LangChain stubs for isolation
   - Monkeypatch-based mocking
   - Async test support with `pytest-asyncio`

2. **Coverage gaps identified:**
   - No dedicated unit tests for `Neo4jStorage` implementation
   - No tests for `GraphTransformer` orchestration logic
   - Limited LLM client extraction testing
   - No tests for authentication dependencies
   - Missing API router unit tests (most endpoints)
   - No entity resolution helper tests

---

## London School TDD Methodology

### Core Principles Applied

1. **Outside-In Development**: Start with behavior at the API boundary, work inward
2. **Mock Collaborators**: Isolate units by mocking external dependencies
3. **Verify Interactions**: Focus on HOW objects collaborate, not internal state
4. **Contract Definition**: Use mocks to define clear interfaces

### Mock Strategy Overview

```
+------------------------+      +------------------------+
|   API Layer (Routers)  | ---> |   Mocked Services      |
+------------------------+      +------------------------+
            |                              |
            v                              v
+------------------------+      +------------------------+
|   Service Layer        | ---> |   Mocked Storage/LLM   |
+------------------------+      +------------------------+
            |                              |
            v                              v
+------------------------+      +------------------------+
|   Storage Layer        | ---> |   Mocked Neo4j Driver  |
+------------------------+      +------------------------+
```

---

## Test Infrastructure Setup

### 1. Directory Structure

```
tests/
  conftest.py                    # Global fixtures and configuration

  unit/
    __init__.py

    services/
      __init__.py
      storage/
        __init__.py
        test_neo4j_storage.py
        test_storage_interface.py
        conftest.py              # Storage-specific fixtures

      transform/
        __init__.py
        test_graph_transformer.py
        test_helpers.py
        test_ontology_helper.py
        conftest.py

      llm/
        __init__.py
        test_llm_client.py
        test_cache.py
        conftest.py

      quality/
        __init__.py
        test_quality_service.py
        test_validator.py
        conftest.py

      audit/
        __init__.py
        test_audit_service.py
        conftest.py

    api/
      __init__.py
      test_transform_router.py
      test_graph_router.py
      test_ontology_router.py
      test_quality_router.py
      test_chat_router.py
      conftest.py                # API-specific fixtures

    auth/
      __init__.py
      test_dependencies.py
      test_models.py
      conftest.py

  integration/
    __init__.py
    test_app_endpoints.py
    test_transform_flow.py
    conftest.py

  fixtures/
    __init__.py
    ontologies.py                # Sample ontology fixtures
    nodes.py                     # Sample node fixtures
    relationships.py             # Sample relationship fixtures
    extraction_results.py        # Sample LLM extraction fixtures

  factories/
    __init__.py
    node_factory.py
    relationship_factory.py
    ontology_factory.py
    user_factory.py

  mocks/
    __init__.py
    neo4j_mock.py
    llm_client_mock.py
    storage_mock.py
    auth_mock.py
    postgres_mock.py
```

### 2. Core Fixtures (tests/conftest.py)

```python
# Add to existing conftest.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

# ============================================================
# Mock Factories
# ============================================================

@pytest.fixture
def mock_neo4j_session():
    """Create a mock Neo4j async session."""
    session = AsyncMock()
    session.run = AsyncMock(return_value=AsyncMock())
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_neo4j_driver(mock_neo4j_session):
    """Create a mock Neo4j async driver."""
    driver = AsyncMock()
    driver.session = MagicMock(return_value=mock_neo4j_session)
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client with configurable responses."""
    client = AsyncMock()
    client.extract_nodes_from_chunk = AsyncMock()
    client.extract_relationships_from_chunk = AsyncMock()
    client.extract_nodes_from_pdf = AsyncMock()
    client.extract_relationships_from_pdf = AsyncMock()
    client.resolve_entities = AsyncMock(return_value=[])
    client.infer_relationship = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_storage():
    """Create a mock GraphStorageInterface implementation."""
    storage = AsyncMock()
    storage.store_nodes = AsyncMock()
    storage.store_relationships = AsyncMock()
    storage.get_transformation_data = AsyncMock()
    storage.get_node_by_id = AsyncMock()
    storage.find_similar_nodes = AsyncMock(return_value=[])
    storage.execute_query = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_audit_service():
    """Create a mock audit service."""
    service = AsyncMock()
    service.log_operation_start = AsyncMock(return_value="audit-123")
    service.log_operation_success = AsyncMock(return_value=True)
    service.log_operation_failure = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_postgres_db():
    """Create a mock Postgres DB module."""
    db = AsyncMock()
    db.fetchrow = AsyncMock(return_value={"id": "row-123"})
    db.fetch = AsyncMock(return_value=[])
    db.execute = AsyncMock()
    return db


# ============================================================
# Sample Data Fixtures
# ============================================================

@pytest.fixture
def sample_ontology() -> Dict[str, Any]:
    """Standard test ontology with Company and Person entities."""
    return {
        "entities": {
            "Company": {
                "properties": {
                    "name": {"type": "string", "required": True, "unique": True},
                    "ticker": {"type": "string", "required": False, "index": True},
                    "industry": {"type": "string", "required": False},
                },
                "relationships": {
                    "EMPLOYS": {
                        "target": "Person",
                        "properties": {
                            "role": {"type": "string"},
                            "start_date": {"type": "string"},
                        },
                    },
                },
            },
            "Person": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string", "unique": True},
                    "age": {"type": "integer"},
                },
            },
        },
    }


@pytest.fixture
def sample_base_node():
    """Create a sample BaseNode."""
    from app.services.transform.models import BaseNode
    return BaseNode(
        id="node-123",
        type="Company",
        properties={"name": "Acme Corp", "ticker": "ACM"},
        canonical_properties={"name": "acme corp", "ticker": "acm"},
        canonical_key="Company:name=acme corp",
        canonical_id="canonical-123",
    )


@pytest.fixture
def sample_relationship():
    """Create a sample RelationshipInstance."""
    from app.services.transform.models import RelationshipInstance
    return RelationshipInstance(
        id="rel-123",
        type="EMPLOYS",
        source_id="company-1",
        target_id="person-1",
        source_type="Company",
        target_type="Person",
        properties={"role": "CEO"},
    )


@pytest.fixture
def sample_auth_context():
    """Create a sample AuthContext."""
    from app.auth.models import AuthContext
    return AuthContext(
        user_id="user-123",
        session_id="session-456",
        token="test-token",
        claims={"sub": "user-123", "email": "test@example.com"},
    )


# ============================================================
# Test User Fixture
# ============================================================

@pytest.fixture
def test_user_id() -> str:
    """Standard test user ID."""
    return "test-user-123"
```

### 3. Mock Objects (tests/mocks/)

#### tests/mocks/neo4j_mock.py

```python
"""Mock Neo4j driver and session for unit tests."""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field


@dataclass
class MockNeo4jRecord:
    """Mock Neo4j record for test data."""
    data: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass
class MockNeo4jNode:
    """Mock Neo4j node with labels and properties."""
    labels: List[str]
    _properties: Dict[str, Any] = field(default_factory=dict)

    def items(self):
        return self._properties.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._properties.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._properties[key]


@dataclass
class MockNeo4jRelationship:
    """Mock Neo4j relationship."""
    type: str
    start_node: MockNeo4jNode
    end_node: MockNeo4jNode
    _properties: Dict[str, Any] = field(default_factory=dict)

    def items(self):
        return self._properties.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._properties.get(key, default)


class MockNeo4jResult:
    """Mock Neo4j query result."""

    def __init__(self, records: List[MockNeo4jRecord] = None):
        self._records = records or []
        self._index = 0

    async def single(self, default=None):
        return self._records[0] if self._records else default

    async def values(self):
        return [[r.data for r in self._records]]

    async def consume(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index < len(self._records):
            record = self._records[self._index]
            self._index += 1
            return record
        raise StopAsyncIteration


class MockNeo4jSession:
    """Mock Neo4j async session."""

    def __init__(self, query_results: Dict[str, List[MockNeo4jRecord]] = None):
        self._query_results = query_results or {}
        self._executed_queries: List[Dict[str, Any]] = []

    async def run(self, query: str, parameters: Dict = None, **kwargs) -> MockNeo4jResult:
        self._executed_queries.append({
            "query": query,
            "parameters": parameters or kwargs,
        })

        # Find matching result or return empty
        for pattern, records in self._query_results.items():
            if pattern in query:
                return MockNeo4jResult(records)
        return MockNeo4jResult([])

    async def close(self):
        pass

    @property
    def executed_queries(self) -> List[Dict[str, Any]]:
        return self._executed_queries


class MockNeo4jDriver:
    """Mock Neo4j async driver."""

    def __init__(self, session: MockNeo4jSession = None):
        self._session = session or MockNeo4jSession()

    def session(self, database: str = None) -> MockNeo4jSession:
        return self._session

    async def close(self):
        pass


def create_mock_neo4j_storage(
    query_results: Dict[str, List[MockNeo4jRecord]] = None
) -> "MockNeo4jStorage":
    """Factory function to create a complete mock storage setup."""
    session = MockNeo4jSession(query_results or {})
    driver = MockNeo4jDriver(session)
    return MockNeo4jStorage(driver, session)


@dataclass
class MockNeo4jStorage:
    """Container for mock Neo4j components."""
    driver: MockNeo4jDriver
    session: MockNeo4jSession
```

#### tests/mocks/llm_client_mock.py

```python
"""Mock LLM client for unit tests."""

from typing import Any, Dict, List, Optional, Type
from unittest.mock import AsyncMock
from pydantic import BaseModel


class MockLLMResponse:
    """Mock LLM extraction response."""

    def __init__(self, entities: Dict[str, List[Any]], confidence: float = 0.9):
        self._entities = entities
        self.confidence_score = confidence

    def __getattr__(self, name: str) -> Any:
        if name.endswith("_list"):
            entity_type = name[:-5]
            return self._entities.get(entity_type, [])
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class MockLLMClient:
    """Configurable mock LLM client for testing."""

    def __init__(self):
        self._node_extractions: List[MockLLMResponse] = []
        self._relationship_extractions: List[MockLLMResponse] = []
        self._entity_resolutions: List[Any] = []
        self._call_counts = {
            "extract_nodes_from_chunk": 0,
            "extract_relationships_from_chunk": 0,
            "extract_nodes_from_pdf": 0,
            "extract_relationships_from_pdf": 0,
            "resolve_entities": 0,
        }

    def configure_node_extraction(self, *responses: MockLLMResponse):
        """Configure sequential node extraction responses."""
        self._node_extractions = list(responses)

    def configure_relationship_extraction(self, *responses: MockLLMResponse):
        """Configure sequential relationship extraction responses."""
        self._relationship_extractions = list(responses)

    def configure_entity_resolution(self, *resolutions: List[Any]):
        """Configure entity resolution responses."""
        self._entity_resolutions = list(resolutions)

    async def extract_nodes_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> Any:
        self._call_counts["extract_nodes_from_chunk"] += 1
        idx = min(
            self._call_counts["extract_nodes_from_chunk"] - 1,
            len(self._node_extractions) - 1,
        )
        if idx >= 0 and self._node_extractions:
            return self._node_extractions[idx]
        return MockLLMResponse({})

    async def extract_relationships_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> Any:
        self._call_counts["extract_relationships_from_chunk"] += 1
        idx = min(
            self._call_counts["extract_relationships_from_chunk"] - 1,
            len(self._relationship_extractions) - 1,
        )
        if idx >= 0 and self._relationship_extractions:
            return self._relationship_extractions[idx]
        return MockLLMResponse({})

    async def resolve_entities(
        self,
        entity_type: str,
        node_dicts_str: str,
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> List[Any]:
        self._call_counts["resolve_entities"] += 1
        idx = min(
            self._call_counts["resolve_entities"] - 1,
            len(self._entity_resolutions) - 1,
        )
        if idx >= 0 and self._entity_resolutions:
            return self._entity_resolutions[idx]
        return []

    @property
    def call_counts(self) -> Dict[str, int]:
        return self._call_counts.copy()
```

---

## Component Test Plans

### Tier 1: Critical Path (Must Have)

#### 1. Storage Layer: Neo4jStorage

**File:** `tests/unit/services/storage/test_neo4j_storage.py`

**Collaborators to Mock:**
- Neo4j AsyncGraphDatabase driver
- Neo4j session

**Test Cases:**

```python
# ============================================================
# Neo4jStorage Unit Tests
# ============================================================

class TestNeo4jStorageInitialization:
    """Test Neo4jStorage constructor and connection handling."""

    async def test_should_create_driver_with_credentials(self):
        """When initializing, should create Neo4j driver with provided credentials."""
        pass

    async def test_should_raise_storage_auth_error_on_auth_failure(self):
        """When authentication fails, should raise StorageAuthError."""
        pass

    async def test_should_raise_storage_connection_error_when_unavailable(self):
        """When service unavailable, should raise StorageConnectionError."""
        pass


class TestNeo4jStorageNodeOperations:
    """Test node storage operations."""

    async def test_store_nodes_should_execute_merge_query(self):
        """When storing nodes with merge=True, should execute MERGE query."""
        pass

    async def test_store_nodes_should_execute_create_query_when_merge_disabled(self):
        """When storing nodes with merge=False, should execute CREATE query."""
        pass

    async def test_store_nodes_should_add_transform_id_to_properties(self):
        """When storing nodes, should include transform_id in properties."""
        pass

    async def test_store_nodes_should_update_checkpoint_on_success(self):
        """After successful node storage, should update checkpoint."""
        pass

    async def test_store_nodes_should_return_partial_result_on_failure(self):
        """When some nodes fail to store, should return partial success result."""
        pass

    async def test_store_nodes_should_retry_on_transient_error(self):
        """When transient error occurs, should retry with exponential backoff."""
        pass

    async def test_store_nodes_should_validate_labels_for_cypher_injection(self):
        """When node type contains injection characters, should raise error."""
        pass


class TestNeo4jStorageRelationshipOperations:
    """Test relationship storage operations."""

    async def test_store_relationships_should_check_existing_relationship(self):
        """When storing relationship, should first check for existing."""
        pass

    async def test_store_relationships_should_version_on_property_change(self):
        """When properties differ from existing, should close old and create new."""
        pass

    async def test_store_relationships_should_skip_duplicate_ids(self):
        """When relationship ID already processed, should skip."""
        pass

    async def test_store_relationships_should_add_versioning_properties(self):
        """Should add valid_from, valid_to, transform_id to relationships."""
        pass


class TestNeo4jStorageQueryOperations:
    """Test query and retrieval operations."""

    async def test_get_transformation_data_should_return_graph_response(self):
        """Should return GraphResponse with nodes and edges."""
        pass

    async def test_get_node_by_id_should_return_node_when_exists(self):
        """When node exists, should return Node object."""
        pass

    async def test_get_node_by_id_should_return_none_when_not_exists(self):
        """When node does not exist, should return None."""
        pass

    async def test_find_similar_nodes_should_combine_exact_and_fuzzy_results(self):
        """Should combine exact match, similarity, and full-text results."""
        pass


class TestNeo4jStorageIndexOperations:
    """Test full-text index operations."""

    async def test_create_ft_index_for_node_should_drop_existing_first(self):
        """Should drop existing index before creating new one."""
        pass

    async def test_create_ft_index_for_relationship_should_use_correct_syntax(self):
        """Should use correct Cypher syntax for relationship index."""
        pass

    async def test_create_ft_index_should_validate_identifiers(self):
        """Should validate all identifiers for Cypher injection."""
        pass
```

**Expected Assertions:**
- Query execution verification
- Parameter passing verification
- Return value transformation
- Error handling behavior
- Retry mechanism activation

---

#### 2. Transform Pipeline: GraphTransformer

**File:** `tests/unit/services/transform/test_graph_transformer.py`

**Collaborators to Mock:**
- LLMClient
- OntologyParser
- Entity ledger service
- Merge learning service

**Test Cases:**

```python
# ============================================================
# GraphTransformer Unit Tests
# ============================================================

class TestBuildGraphFromChunks:
    """Test the main graph building orchestration."""

    async def test_should_extract_nodes_for_each_chunk(self):
        """Should call node extraction for each input chunk."""
        pass

    async def test_should_pass_accumulated_context_to_subsequent_extractions(self):
        """Should build context from previous nodes for next extraction."""
        pass

    async def test_should_deduplicate_nodes_within_chunk(self):
        """Should not add duplicate nodes from same chunk."""
        pass

    async def test_should_call_entity_resolution_when_multiple_nodes(self):
        """Should resolve entities when group has multiple nodes."""
        pass

    async def test_should_extract_relationships_with_bounded_concurrency(self):
        """Should respect TRANSFORM_MAX_CONCURRENCY setting."""
        pass

    async def test_should_prune_orphaned_nodes_at_end(self):
        """Should remove nodes with no relationships and no properties."""
        pass

    async def test_should_hydrate_nodes_from_ledger_when_user_provided(self):
        """When user_id provided, should hydrate from entity ledger."""
        pass

    async def test_should_record_nodes_to_ledger_after_processing(self):
        """When user_id provided, should record nodes to entity ledger."""
        pass


class TestContextBuilding:
    """Test context envelope construction."""

    async def test_nodes_context_should_be_deterministically_sorted(self):
        """Nodes context should sort by type, properties, id."""
        pass

    async def test_relationships_context_should_include_orphan_nodes(self):
        """Should list nodes not in any relationship."""
        pass

    async def test_context_should_truncate_when_exceeds_limit(self):
        """When context exceeds MAX_CONTEXT_CHARS, should truncate."""
        pass

    async def test_truncated_context_should_include_sentinel(self):
        """Truncated context should include truncation sentinel."""
        pass


class TestNodeComparison:
    """Test node comparison and merging."""

    async def test_compare_and_merge_should_group_by_entity_type(self):
        """Should process nodes grouped by their entity type."""
        pass

    async def test_merge_nodes_should_combine_provenance(self):
        """Merged node should have combined chunk IDs."""
        pass

    async def test_merge_nodes_should_take_higher_confidence_values(self):
        """Should prefer property values from higher confidence node."""
        pass

    async def test_merge_nodes_should_take_longer_string_values(self):
        """When confidence equal, should prefer longer string values."""
        pass


class TestRelationshipProcessing:
    """Test relationship extraction and merging."""

    async def test_should_deduplicate_relationships_by_unique_id(self):
        """Should merge relationships with same source-type-target."""
        pass

    async def test_should_skip_invalid_source_or_target(self):
        """Should skip relationships with unknown node IDs."""
        pass
```

---

#### 3. LLM Client

**File:** `tests/unit/services/llm/test_llm_client.py`

**Collaborators to Mock:**
- User LLM credentials service
- BAML client
- Gemini client
- Cache (Redis/LRU)
- Usage tracker

**Test Cases:**

```python
# ============================================================
# LLMClient Unit Tests
# ============================================================

class TestLLMClientNodeExtraction:
    """Test node extraction from chunks."""

    async def test_should_require_user_id(self):
        """Should raise ValueError when user_id not provided."""
        pass

    async def test_should_get_credentials_for_user(self):
        """Should fetch LLM credentials for the specified user."""
        pass

    async def test_should_return_cached_result_when_available(self):
        """When cache hit, should return cached result without LLM call."""
        pass

    async def test_should_cache_result_after_extraction(self):
        """Should store result in cache after successful extraction."""
        pass

    async def test_cache_key_should_include_user_and_model(self):
        """Cache key should incorporate user_id and model name."""
        pass

    async def test_should_track_usage_when_user_provided(self):
        """Should call usage tracker after extraction."""
        pass


class TestLLMClientRelationshipExtraction:
    """Test relationship extraction from chunks."""

    async def test_should_use_separate_cache_namespace(self):
        """Relationship extraction should use different cache than nodes."""
        pass

    async def test_should_pass_context_to_extraction(self):
        """Should include provided context in extraction prompt."""
        pass


class TestLLMClientPDFExtraction:
    """Test PDF-based extraction."""

    async def test_should_read_pdf_bytes_and_hash(self):
        """Should read file bytes and compute content hash for caching."""
        pass

    async def test_should_handle_gemini_response(self):
        """Should parse Gemini structured response correctly."""
        pass

    async def test_should_retry_on_failure(self):
        """Should retry extraction up to max_attempts on error."""
        pass


class TestLLMClientEntityResolution:
    """Test entity resolution calls."""

    async def test_should_format_nodes_as_json(self):
        """Should convert node dicts to JSON string for LLM."""
        pass

    async def test_should_return_grouped_entities(self):
        """Should return list of resolved entity groups."""
        pass


class TestLLMCache:
    """Test caching behavior."""

    def test_should_use_redis_when_url_configured(self):
        """When LLM_CACHE_URL set, should create Redis cache."""
        pass

    def test_should_fallback_to_lru_when_redis_fails(self):
        """When Redis init fails, should fallback to LRU cache."""
        pass

    def test_lru_cache_should_evict_oldest_on_overflow(self):
        """LRU cache should evict least recently used entries."""
        pass
```

---

#### 4. Entity Resolution Helpers

**File:** `tests/unit/services/transform/test_helpers.py`

**Test Cases:**

```python
# ============================================================
# Transform Helpers Unit Tests
# ============================================================

class TestTransformAsNodes:
    """Test node transformation from extraction results."""

    def test_should_extract_properties_from_entity_list(self):
        """Should process fields ending with _list as entity lists."""
        pass

    def test_should_normalize_properties_per_ontology(self):
        """Should coerce types and validate against ontology."""
        pass

    def test_should_build_canonical_properties(self):
        """Should create canonical versions of properties."""
        pass

    def test_should_generate_deterministic_id_when_enabled(self):
        """When DETERMINISTIC_MODE=True, should generate stable ID."""
        pass

    def test_should_generate_uuid_when_deterministic_disabled(self):
        """When DETERMINISTIC_MODE=False, should generate random UUID."""
        pass

    def test_should_skip_nodes_failing_validation(self):
        """Should skip nodes missing required properties."""
        pass


class TestTransformAsRelationships:
    """Test relationship transformation."""

    def test_should_resolve_node_ids_by_canonical_id(self):
        """Should match source/target by canonical_id when available."""
        pass

    def test_should_resolve_node_ids_by_canonical_key(self):
        """Should match source/target by canonical_key as fallback."""
        pass

    def test_should_infer_target_type_from_ontology(self):
        """Should lookup target type from ontology relationships."""
        pass

    def test_should_skip_unknown_relationship_types(self):
        """Should skip relationships not defined in ontology."""
        pass


class TestSplinkDeduplication:
    """Test Splink-based entity deduplication."""

    async def test_should_group_entities_by_type(self):
        """Should process each entity type separately."""
        pass

    async def test_should_apply_heuristic_dedup_for_small_groups(self):
        """For small groups, should use canonical key matching."""
        pass

    async def test_should_skip_splink_when_fewer_than_three_entities(self):
        """Should not run Splink when < 3 entities after heuristics."""
        pass

    async def test_should_use_adaptive_threshold_from_learning(self):
        """Should get threshold from merge_learning_service."""
        pass

    async def test_should_update_relationships_with_representative_ids(self):
        """Should remap relationship source/target to representatives."""
        pass

    async def test_should_skip_self_relationships_after_dedup(self):
        """Should not create self-referential relationships."""
        pass


class TestCanonicalisation:
    """Test property canonicalisation."""

    def test_should_lowercase_strings(self):
        """Should lowercase string values by default."""
        pass

    def test_should_strip_whitespace(self):
        """Should collapse and trim whitespace."""
        pass

    def test_should_strip_company_suffixes_when_configured(self):
        """Should remove Inc, LLC, etc. when strip_company_suffixes=True."""
        pass

    def test_should_preserve_case_when_configured(self):
        """Should not lowercase when preserve_case=True."""
        pass

    def test_should_use_registered_canonicalizer(self):
        """Should apply custom canonicalizer when registered."""
        pass


class TestNodeKeyGeneration:
    """Test canonical key and ID generation."""

    def test_should_use_unique_properties_first(self):
        """Should prioritize unique properties for key."""
        pass

    def test_should_fallback_to_all_properties(self):
        """When no unique props, should use all non-empty properties."""
        pass

    def test_should_hash_raw_properties_when_no_canonical(self):
        """Should hash raw properties when nothing else available."""
        pass

    def test_should_include_fallback_hint(self):
        """Should use fallback_hint when all else fails."""
        pass
```

---

### Tier 2: Important (Should Have)

#### 5. API Endpoints

**File:** `tests/unit/api/test_transform_router.py`

**Collaborators to Mock:**
- Auth dependencies
- Transform flow
- Progress tracker
- Audit service
- File validator

**Test Cases:**

```python
# ============================================================
# Transform API Unit Tests
# ============================================================

class TestTransformUpload:
    """Test POST /transform/{ontology_id}/upload endpoint."""

    async def test_should_require_authentication(self):
        """Should return 401 when no auth token provided."""
        pass

    async def test_should_validate_file_mime_type(self):
        """Should reject files with disallowed MIME types."""
        pass

    async def test_should_validate_file_size(self):
        """Should reject files exceeding size limit."""
        pass

    async def test_should_initialize_progress_tracker(self):
        """Should call progress_tracker.initialize_transform."""
        pass

    async def test_should_start_audit_trail(self):
        """Should log operation start to audit service."""
        pass

    async def test_should_start_background_flow(self):
        """Should trigger run_transform_flow in background."""
        pass

    async def test_should_return_transform_id_and_status(self):
        """Should return pending status with transform ID."""
        pass


class TestTransformStatus:
    """Test GET /transform/{transform_id}/status endpoint."""

    async def test_should_return_current_stage(self):
        """Should return transformation's current stage."""
        pass

    async def test_should_return_progress_percentage(self):
        """Should calculate and return progress percentage."""
        pass

    async def test_should_return_error_on_failure(self):
        """When transform failed, should include error details."""
        pass
```

---

#### 6. Authentication

**File:** `tests/unit/auth/test_dependencies.py`

**Collaborators to Mock:**
- PyJWKClient
- JWT decode

**Test Cases:**

```python
# ============================================================
# Auth Dependencies Unit Tests
# ============================================================

class TestGetCurrentAuth:
    """Test authentication dependency."""

    async def test_should_raise_401_when_no_credentials(self):
        """Should raise HTTPException 401 when Authorization header missing."""
        pass

    async def test_should_decode_valid_jwt(self):
        """Should decode and validate JWT from bearer token."""
        pass

    async def test_should_extract_user_id_from_sub_claim(self):
        """Should use 'sub' claim as user_id."""
        pass

    async def test_should_raise_401_on_expired_token(self):
        """Should raise HTTPException 401 when token expired."""
        pass

    async def test_should_raise_401_on_invalid_issuer(self):
        """Should raise HTTPException 401 when issuer mismatch."""
        pass

    async def test_should_raise_500_when_jwks_not_configured(self):
        """Should raise HTTPException 500 when CLERK_JWKS_URL missing."""
        pass

    async def test_should_require_issuer_in_production(self):
        """In production, should fail when CLERK_ISSUER not set."""
        pass

    async def test_should_warn_when_audience_not_configured(self):
        """Should log warning when CLERK_AUDIENCE not configured."""
        pass
```

---

#### 7. Audit Service

**File:** `tests/unit/services/audit/test_audit_service.py`

**Collaborators to Mock:**
- Postgres DB module

**Test Cases:**

```python
# ============================================================
# Audit Service Unit Tests
# ============================================================

class TestAuditServiceOperationLogging:
    """Test operation logging methods."""

    async def test_log_operation_start_should_insert_record(self):
        """Should insert audit_trail record with IN_PROGRESS status."""
        pass

    async def test_log_operation_start_should_return_audit_id(self):
        """Should return the created audit record ID."""
        pass

    async def test_log_operation_success_should_update_status(self):
        """Should update record status to SUCCESS."""
        pass

    async def test_log_operation_success_should_merge_metadata(self):
        """Should merge new metadata with existing."""
        pass

    async def test_log_operation_failure_should_include_error(self):
        """Should update record with error_message."""
        pass

    async def test_should_handle_db_errors_gracefully(self):
        """Should return empty/false on database errors, not raise."""
        pass


class TestAuditServiceQueries:
    """Test audit trail query methods."""

    async def test_get_user_audit_trail_should_filter_by_user(self):
        """Should only return records for specified user."""
        pass

    async def test_get_user_audit_trail_should_support_pagination(self):
        """Should apply limit and offset to results."""
        pass

    async def test_get_audit_summary_should_count_by_type(self):
        """Should aggregate counts by operation_type."""
        pass
```

---

#### 8. User Database Service

**File:** `tests/unit/services/test_user_db_service.py`

**Test Cases:**

```python
# ============================================================
# User Database Service Unit Tests
# ============================================================

class TestUserDatabaseService:
    """Test user database operations."""

    async def test_should_create_user_database_on_first_access(self):
        """Should create Neo4j database for new user."""
        pass

    async def test_should_isolate_user_data(self):
        """Should ensure queries scoped to user's database."""
        pass

    async def test_should_cache_database_connections(self):
        """Should reuse existing connections for same user."""
        pass
```

---

### Tier 3: Nice to Have

#### 9. Chunking Services

**File:** `tests/unit/services/chunking/test_chunker.py`

**Test Cases:**

```python
class TestChunker:
    """Test document chunking service."""

    async def test_should_split_by_semantic_boundaries(self):
        """Should use semantic chunker when available."""
        pass

    async def test_should_fallback_to_character_splitter(self):
        """Should use character splitter as fallback."""
        pass

    async def test_should_respect_chunk_size_limits(self):
        """Should not exceed max_chunk_size setting."""
        pass
```

#### 10. Quality Validation

**File:** `tests/unit/services/quality/test_validator.py`

**Test Cases:**

```python
class TestQualityValidator:
    """Test quality validation rules."""

    async def test_should_validate_required_properties(self):
        """Should flag missing required properties."""
        pass

    async def test_should_validate_property_formats(self):
        """Should check format constraints."""
        pass

    async def test_should_calculate_overall_score(self):
        """Should compute weighted quality score."""
        pass
```

---

## Implementation Order

The implementation should follow a bottom-up approach, starting with the lowest-level dependencies:

### Phase 1: Foundations (Week 1)

1. **Test Infrastructure Setup**
   - Create directory structure
   - Write mock objects
   - Create factory functions
   - Set up fixtures

2. **Storage Layer Tests**
   - `test_neo4j_storage.py` - All node and relationship operations
   - Verify Cypher query generation
   - Test retry logic

### Phase 2: Core Services (Week 2)

3. **Transform Helpers Tests**
   - `test_helpers.py` - Canonicalisation, node generation
   - Entity deduplication logic
   - Relationship transformation

4. **LLM Client Tests**
   - `test_llm_client.py` - Extraction methods
   - `test_cache.py` - Caching behavior
   - Credential handling

### Phase 3: Orchestration (Week 3)

5. **Graph Transformer Tests**
   - `test_graph_transformer.py` - Full orchestration flow
   - Context building
   - Node/relationship processing coordination

6. **Audit Service Tests**
   - `test_audit_service.py` - Logging operations
   - Query methods

### Phase 4: API Layer (Week 4)

7. **Authentication Tests**
   - `test_dependencies.py` - JWT validation
   - Error handling

8. **API Router Tests**
   - Transform endpoints
   - Graph endpoints
   - Quality endpoints

### Phase 5: Integration & Polish (Week 5)

9. **Integration Tests**
   - Full flow tests
   - Error scenario tests

10. **Coverage Analysis & Gap Filling**
    - Run coverage report
    - Add missing edge cases

---

## Coverage Targets

| Component | Target | Critical Path |
|-----------|--------|---------------|
| Storage Layer | 90%+ | Yes |
| Transform Pipeline | 90%+ | Yes |
| LLM Client | 85%+ | Yes |
| Entity Resolution | 90%+ | Yes |
| API Routers | 80%+ | No |
| Authentication | 85%+ | No |
| Audit Service | 80%+ | No |
| **Overall** | **80%+** | - |

---

## Test Execution Configuration

### pytest.ini additions

```ini
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "-ra",
]
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

### Running Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=app --cov-report=html --cov-report=term-missing

# Run specific tier
pytest tests/unit/services/storage/ -v

# Run excluding slow tests
pytest tests/unit/ -v -m "not slow"

# Run integration tests only
pytest tests/integration/ -v -m integration
```

---

## Appendix: Mock Contract Examples

### Storage Interface Contract

```python
# All storage implementations must satisfy these interaction contracts:

async def test_storage_contract_store_nodes(storage):
    """Storage must accept nodes and return batch result."""
    nodes = [sample_node]
    result = await storage.store_nodes(nodes, batch_index=0, transform_id="tx-1")

    assert isinstance(result, StorageBatchResult)
    assert result.batch_index == 0
    assert result.items_processed >= 0


async def test_storage_contract_get_node_by_id(storage):
    """Storage must return Node or None for ID lookup."""
    result = await storage.get_node_by_id("some-id")

    assert result is None or isinstance(result, Node)
```

### LLM Client Contract

```python
# LLM client must satisfy these interaction contracts:

async def test_llm_contract_extract_nodes(llm_client):
    """LLM client must return model matching response_model schema."""
    result = await llm_client.extract_nodes_from_chunk(
        chunk="Sample text",
        response_model=SampleModel,
        user_id="user-123",
    )

    assert hasattr(result, "confidence_score")
    # Result should have *_list attributes for entities


async def test_llm_contract_requires_user_id(llm_client):
    """LLM client must require user_id for credential lookup."""
    with pytest.raises(ValueError, match="user_id is required"):
        await llm_client.extract_nodes_from_chunk(
            chunk="Sample",
            response_model=SampleModel,
            user_id=None,
        )
```

---

## Conclusion

This TDD plan provides a comprehensive roadmap for achieving robust test coverage of the Graphora API using London School (mockist) principles. By focusing on:

1. **Clear collaborator contracts** through mock definitions
2. **Interaction verification** over state inspection
3. **Outside-in development** from API to storage
4. **Bounded mock scope** to ensure tests remain focused

The resulting test suite will enable safe refactoring, preserve API contracts, and provide confidence for future architecture improvements.
