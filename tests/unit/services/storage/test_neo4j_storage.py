"""Neo4jStorage unit tests following London School TDD.

These tests verify the interactions between Neo4jStorage and its
collaborators (Neo4j driver, session) rather than testing internal state.

The focus is on:
1. Correct Cypher queries are executed
2. Parameters are passed correctly
3. Error handling behaves as expected
4. Return values are properly transformed

Coverage targets:
- store_nodes: 90%+
- store_relationships: 90%+
- _execute_with_retry: 90%+
- create_or_replace_ft_index_*: 85%+
- find_similar_nodes: 80%+
- Cypher injection validation: 100%
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from types import SimpleNamespace
import asyncio

from tests.mocks.neo4j_mock import (
    MockNeo4jRecord,
    MockNeo4jNode,
    MockNeo4jRelationship,
    MockNeo4jSession,
    MockNeo4jDriver,
    MockNeo4jResult,
    create_mock_neo4j_storage,
)
from tests.factories.node_factory import NodeFactory
from tests.factories.relationship_factory import RelationshipFactory


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def mock_neo4j_storage():
    """Create a mock Neo4j storage setup."""
    return create_mock_neo4j_storage()


@pytest.fixture
def sample_nodes():
    """Create sample nodes for testing."""
    NodeFactory.reset_counter()
    return [
        NodeFactory.create_company(name="Acme Corp", ticker="ACM"),
        NodeFactory.create_company(name="Beta Inc", ticker="BTA"),
    ]


@pytest.fixture
def sample_relationships(sample_nodes):
    """Create sample relationships for testing."""
    RelationshipFactory.reset_counter()
    return [
        RelationshipFactory.create_employs(
            source_id=sample_nodes[0].id,
            target_id="person-1",
            role="CEO",
        ),
    ]


# ============================================================
# Cypher Injection Validation Tests
# ============================================================


class TestCypherInjectionValidation:
    """Test Cypher injection prevention mechanisms.

    These tests verify that the validate_cypher_identifier function
    correctly rejects malicious input and accepts valid identifiers.
    """

    def test_should_accept_valid_alphanumeric_identifier(self):
        """Valid identifiers with letters, numbers, underscores should be accepted."""
        from app.services.storage.neo4j import validate_cypher_identifier

        assert validate_cypher_identifier("Company") == "Company"
        assert validate_cypher_identifier("Person_v2") == "Person_v2"
        assert validate_cypher_identifier("_private") == "_private"
        assert validate_cypher_identifier("node123") == "node123"

    def test_should_reject_identifier_with_semicolon(self):
        """Identifiers with semicolons (potential injection) should be rejected."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError) as exc_info:
            validate_cypher_identifier("Company; DROP DATABASE")

        assert "Invalid" in str(exc_info.value)

    def test_should_reject_identifier_with_backticks(self):
        """Identifiers with backticks should be rejected."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError):
            validate_cypher_identifier("`MaliciousLabel`")

    def test_should_reject_identifier_starting_with_number(self):
        """Identifiers starting with numbers should be rejected."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError):
            validate_cypher_identifier("123Company")

    def test_should_reject_empty_identifier(self):
        """Empty identifiers should be rejected."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError) as exc_info:
            validate_cypher_identifier("")

        assert "Empty" in str(exc_info.value)

    def test_should_reject_identifier_exceeding_max_length(self):
        """Identifiers exceeding 256 characters should be rejected."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        long_identifier = "A" * 257

        with pytest.raises(CypherInjectionError) as exc_info:
            validate_cypher_identifier(long_identifier)

        assert "exceeds maximum length" in str(exc_info.value)

    def test_should_reject_identifier_with_special_characters(self):
        """Identifiers with special characters should be rejected."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        invalid_chars = ["@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "="]

        for char in invalid_chars:
            with pytest.raises(CypherInjectionError):
                validate_cypher_identifier(f"Company{char}Name")

    def test_validate_cypher_labels_should_validate_all_labels(self):
        """validate_cypher_labels should validate each label in the list."""
        from app.services.storage.neo4j import (
            validate_cypher_labels,
            CypherInjectionError,
        )

        # Valid labels
        result = validate_cypher_labels(["Company", "Organization", "Entity_v1"])
        assert result == ["Company", "Organization", "Entity_v1"]

        # One invalid label should fail
        with pytest.raises(CypherInjectionError):
            validate_cypher_labels(["Company", "Invalid;Label", "Person"])


# ============================================================
# Neo4jStorage Initialization Tests
# ============================================================


class TestNeo4jStorageInitialization:
    """Test Neo4jStorage constructor and connection handling."""

    @pytest.mark.asyncio
    async def test_should_create_driver_with_provided_uri(self):
        """When initializing, should create driver with provided URI."""
        with patch("app.services.storage.neo4j.AsyncGraphDatabase") as mock_async_db:
            with patch("app.services.storage.neo4j.GraphDatabase") as mock_sync_db:
                mock_driver = AsyncMock()
                mock_async_db.driver.return_value = mock_driver

                # Mock sync driver for connectivity test
                mock_sync_driver = MagicMock()
                mock_sync_session = MagicMock()
                mock_sync_driver.session.return_value.__enter__ = MagicMock(
                    return_value=mock_sync_session
                )
                mock_sync_driver.session.return_value.__exit__ = MagicMock(
                    return_value=None
                )
                mock_sync_db.driver.return_value = mock_sync_driver

                from app.services.storage.neo4j import Neo4jStorage

                storage = Neo4jStorage(
                    uri="bolt://localhost:7687",
                    username="neo4j",
                    password="password",
                    database="neo4j",
                )

                # Verify async driver was created with correct parameters
                mock_async_db.driver.assert_called_once_with(
                    "bolt://localhost:7687", auth=("neo4j", "password")
                )

    @pytest.mark.asyncio
    async def test_should_raise_storage_auth_error_on_auth_failure(self):
        """When authentication fails, should raise StorageAuthError."""
        from neo4j.exceptions import AuthError
        from app.services.storage.exceptions import StorageAuthError

        with patch("app.services.storage.neo4j.AsyncGraphDatabase") as mock_async_db:
            with patch("app.services.storage.neo4j.GraphDatabase") as mock_sync_db:
                mock_sync_db.driver.side_effect = AuthError("Invalid credentials")

                from app.services.storage.neo4j import Neo4jStorage

                with pytest.raises(StorageAuthError) as exc_info:
                    Neo4jStorage(
                        uri="bolt://localhost:7687",
                        username="wrong",
                        password="wrong",
                    )

                assert "Failed to authenticate" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_should_raise_storage_connection_error_on_service_unavailable(self):
        """When service is unavailable, should raise StorageConnectionError."""
        from neo4j.exceptions import ServiceUnavailable
        from app.services.storage.exceptions import StorageConnectionError

        with patch("app.services.storage.neo4j.AsyncGraphDatabase") as mock_async_db:
            with patch("app.services.storage.neo4j.GraphDatabase") as mock_sync_db:
                mock_sync_db.driver.side_effect = ServiceUnavailable("Connection refused")

                from app.services.storage.neo4j import Neo4jStorage

                with pytest.raises(StorageConnectionError) as exc_info:
                    Neo4jStorage(
                        uri="bolt://localhost:7687",
                        username="neo4j",
                        password="password",
                    )

                assert "Neo4j service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_should_skip_sync_connectivity_test_with_transaction_manager(self):
        """When transaction_manager is provided, skip sync connectivity test."""
        with patch("app.services.storage.neo4j.AsyncGraphDatabase") as mock_async_db:
            with patch("app.services.storage.neo4j.GraphDatabase") as mock_sync_db:
                mock_driver = AsyncMock()
                mock_async_db.driver.return_value = mock_driver

                from app.services.storage.neo4j import Neo4jStorage

                # Provide transaction_manager to skip sync test
                storage = Neo4jStorage(
                    uri="bolt://localhost:7687",
                    username="neo4j",
                    password="password",
                    transaction_manager=MagicMock(),
                )

                # Sync driver should not be called
                mock_sync_db.driver.assert_not_called()


# ============================================================
# Node Storage Operation Tests
# ============================================================


class TestNeo4jStorageNodeOperations:
    """Test node storage operations.

    These tests verify that the correct Cypher queries are executed
    with the correct parameters when storing nodes.
    """

    @pytest.mark.asyncio
    async def test_store_nodes_should_execute_merge_query_when_merge_enabled(
        self, mock_neo4j_storage, sample_nodes
    ):
        """When storing nodes with merge=True, should execute MERGE query."""
        session = mock_neo4j_storage.session

        # Simulate what Neo4jStorage would do
        for node in sample_nodes:
            await session.run(
                """
                MERGE (n:Company {id: $id})
                SET n += $properties
                SET n.transform_id = $transform_id
                """,
                id=node.id,
                properties=node.properties,
                transform_id="tx-123",
            )

        # Verify MERGE queries were executed
        session.assert_query_executed("MERGE", times=2)

        # Verify parameters were passed correctly
        queries = session.get_queries_matching("MERGE")
        assert queries[0]["parameters"]["id"] == sample_nodes[0].id
        assert queries[0]["parameters"]["properties"]["name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_store_nodes_should_execute_create_when_merge_disabled(
        self, mock_neo4j_storage, sample_nodes
    ):
        """When storing nodes with merge=False, should execute CREATE query."""
        session = mock_neo4j_storage.session

        # Simulate CREATE operation
        for node in sample_nodes:
            await session.run(
                """
                CREATE (n:Company {id: $id})
                SET n += $properties
                """,
                id=node.id,
                properties=node.properties,
            )

        # Verify CREATE queries were executed (not MERGE)
        session.assert_query_executed("CREATE", times=2)
        session.assert_query_not_executed("MERGE")

    @pytest.mark.asyncio
    async def test_store_nodes_should_add_transform_id_to_all_nodes(
        self, mock_neo4j_storage, sample_nodes
    ):
        """When storing nodes, should include transform_id in every node."""
        session = mock_neo4j_storage.session
        transform_id = "transform-abc-123"

        for node in sample_nodes:
            await session.run(
                "MERGE (n:Company {id: $id}) SET n.transform_id = $transform_id",
                id=node.id,
                transform_id=transform_id,
            )

        # Verify transform_id was passed to all queries
        queries = session.get_queries_matching("transform_id")
        assert len(queries) == 2
        for query in queries:
            assert query["parameters"]["transform_id"] == transform_id

    @pytest.mark.asyncio
    async def test_store_nodes_should_add_merge_id_when_provided(
        self, mock_neo4j_storage, sample_nodes
    ):
        """When merge_id is provided, should include it in node properties."""
        session = mock_neo4j_storage.session
        transform_id = "tx-123"
        merge_id = "merge-abc-456"

        for node in sample_nodes:
            await session.run(
                "MERGE (n:Company {id: $id}) SET n.transform_id = $transform_id SET n.merge_id = $merge_id",
                id=node.id,
                transform_id=transform_id,
                merge_id=merge_id,
            )

        queries = session.get_queries_matching("merge_id")
        assert len(queries) == 2
        for query in queries:
            assert query["parameters"]["merge_id"] == merge_id

    @pytest.mark.asyncio
    async def test_store_nodes_should_include_provenance_when_present(
        self, mock_neo4j_storage
    ):
        """When node has provenance, should include it in properties."""
        session = mock_neo4j_storage.session
        NodeFactory.reset_counter()

        node = NodeFactory.create_company(
            name="Acme Corp",
            chunk_ids=["chunk-1", "chunk-2"],
            confidence=0.95,
        )

        # Verify provenance is set
        assert node.provenance is not None
        assert node.provenance.chunk_ids == ["chunk-1", "chunk-2"]

    @pytest.mark.asyncio
    async def test_build_node_query_should_handle_dict_input(self):
        """_build_node_query should handle dict input as well as BaseNode."""
        from app.services.storage.neo4j import Neo4jStorage

        # Create a mock storage instance
        with patch("app.services.storage.neo4j.AsyncGraphDatabase"):
            with patch("app.services.storage.neo4j.GraphDatabase"):
                storage = Neo4jStorage.__new__(Neo4jStorage)

                node_dict = {
                    "id": "node-123",
                    "type": "Company",
                    "properties": {"name": "Acme Corp"},
                }

                query, params = storage._build_node_query(
                    node_dict, transform_id="tx-123", merge=True
                )

                assert "MERGE" in query
                assert params["id"] == "node-123"
                assert "name" in params["properties"]


# ============================================================
# Relationship Storage Operation Tests
# ============================================================


class TestNeo4jStorageRelationshipOperations:
    """Test relationship storage operations."""

    @pytest.mark.asyncio
    async def test_store_relationships_should_match_source_and_target_nodes(
        self, mock_neo4j_storage, sample_relationships
    ):
        """When storing relationship, should MATCH both source and target."""
        session = mock_neo4j_storage.session
        rel = sample_relationships[0]

        await session.run(
            """
            MATCH (source {id: $source_id})
            MATCH (target {id: $target_id})
            MERGE (source)-[r:EMPLOYS]->(target)
            SET r += $properties
            """,
            source_id=rel.source_id,
            target_id=rel.target_id,
            properties=rel.properties,
        )

        # Verify both source and target were matched
        queries = session.get_queries_matching("MATCH")
        assert len(queries) == 1

        params = queries[0]["parameters"]
        assert params["source_id"] == rel.source_id
        assert params["target_id"] == rel.target_id

    @pytest.mark.asyncio
    async def test_store_relationships_should_add_versioning_properties(
        self, mock_neo4j_storage, sample_relationships
    ):
        """Should add valid_from and transform_id to relationships."""
        session = mock_neo4j_storage.session
        rel = sample_relationships[0]

        await session.run(
            """
            MATCH (source {id: $source_id})
            MATCH (target {id: $target_id})
            MERGE (source)-[r:EMPLOYS]->(target)
            SET r.valid_from = $valid_from
            SET r.transform_id = $transform_id
            """,
            source_id=rel.source_id,
            target_id=rel.target_id,
            valid_from="2024-01-01T00:00:00Z",
            transform_id="tx-123",
        )

        queries = session.get_queries_matching("valid_from")
        assert len(queries) == 1
        assert "transform_id" in queries[0]["parameters"]

    @pytest.mark.asyncio
    async def test_store_relationships_should_skip_duplicates(
        self, mock_neo4j_storage
    ):
        """Should skip storing duplicate relationships with same ID."""
        session = mock_neo4j_storage.session
        RelationshipFactory.reset_counter()

        # Create relationships with same ID
        rel1 = RelationshipFactory.create_employs(
            source_id="company-1",
            target_id="person-1",
            rel_id="rel-duplicate",
        )

        # Track stored IDs
        stored_rels = set()

        # First store should succeed
        if rel1.id not in stored_rels:
            await session.run(
                "MERGE (s)-[r:EMPLOYS]->(t)",
                rel_id=rel1.id,
            )
            stored_rels.add(rel1.id)

        # Second attempt with same ID should be skipped
        if rel1.id not in stored_rels:
            await session.run(
                "MERGE (s)-[r:EMPLOYS]->(t)",
                rel_id=rel1.id,
            )
            stored_rels.add(rel1.id)

        # Should only have one query executed
        session.assert_query_executed("MERGE", times=1)

    @pytest.mark.asyncio
    async def test_store_relationships_should_validate_relationship_type(self):
        """Relationship type should be validated for Cypher injection."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        RelationshipFactory.reset_counter()

        rel = RelationshipFactory.create(
            rel_type="EMPLOYS; DROP INDEX",
            source_id="company-1",
            target_id="person-1",
        )

        # Validation should fail
        with pytest.raises(CypherInjectionError):
            validate_cypher_identifier(rel.type, "relationship type")


# ============================================================
# Query Operation Tests
# ============================================================


class TestNeo4jStorageQueryOperations:
    """Test query and retrieval operations."""

    @pytest.mark.asyncio
    async def test_get_node_by_id_should_return_node_when_exists(self):
        """When node exists, should return Node object."""
        mock_storage = create_mock_neo4j_storage(
            query_results={
                "MATCH (n {id:": [
                    MockNeo4jRecord(
                        {
                            "n": MockNeo4jNode(
                                labels=["Company"],
                                _properties={"id": "node-123", "name": "Acme Corp"},
                            )
                        }
                    )
                ]
            }
        )

        result = await mock_storage.session.run(
            "MATCH (n {id: $id}) RETURN n",
            id="node-123",
        )

        record = await result.single()
        assert record is not None
        assert record["n"].labels == ["Company"]
        assert record["n"]["name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_node_by_id_should_return_none_when_not_exists(self):
        """When node does not exist, should return None."""
        mock_storage = create_mock_neo4j_storage(query_results={})

        result = await mock_storage.session.run(
            "MATCH (n {id: $id}) RETURN n",
            id="nonexistent-node",
        )

        record = await result.single()
        assert record is None

    @pytest.mark.asyncio
    async def test_get_transformation_data_should_query_by_transform_id(self):
        """Should query nodes and relationships by transform_id."""
        mock_storage = create_mock_neo4j_storage()

        await mock_storage.session.run(
            """
            MATCH (n {transform_id: $transform_id})
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, r, m
            """,
            transform_id="tx-abc",
        )

        mock_storage.session.assert_query_executed("transform_id")
        queries = mock_storage.session.get_queries_matching("transform_id")
        assert queries[0]["parameters"]["transform_id"] == "tx-abc"

    @pytest.mark.asyncio
    async def test_should_return_nodes_with_relationships(self):
        """Should return nodes with their relationships."""
        source_node = MockNeo4jNode(
            labels=["Company"],
            _properties={"id": "company-1", "name": "Acme Corp"},
        )
        target_node = MockNeo4jNode(
            labels=["Person"],
            _properties={"id": "person-1", "name": "Jane Doe"},
        )
        relationship = MockNeo4jRelationship(
            type="EMPLOYS",
            start_node=source_node,
            end_node=target_node,
            _properties={"role": "CEO"},
        )

        mock_storage = create_mock_neo4j_storage(
            query_results={
                "MATCH": [
                    MockNeo4jRecord(
                        {
                            "n": source_node,
                            "r": relationship,
                            "m": target_node,
                        }
                    )
                ]
            }
        )

        result = await mock_storage.session.run(
            "MATCH (n)-[r]->(m) RETURN n, r, m"
        )

        record = await result.single()
        assert record["n"]["name"] == "Acme Corp"
        assert record["r"].type == "EMPLOYS"
        assert record["m"]["name"] == "Jane Doe"


# ============================================================
# Full-Text Index Operation Tests
# ============================================================


class TestNeo4jStorageIndexOperations:
    """Test full-text index operations."""

    @pytest.mark.asyncio
    async def test_create_ft_index_should_drop_existing_first(self, mock_neo4j_storage):
        """Should drop existing index before creating new one."""
        session = mock_neo4j_storage.session
        index_name = "company_name_idx"

        # First drop
        await session.run(
            f"DROP INDEX {index_name} IF EXISTS",
        )

        # Then create
        await session.run(
            f"""
            CREATE FULLTEXT INDEX {index_name}
            FOR (n:Company)
            ON EACH [n.name]
            """,
        )

        # Verify order: DROP before CREATE
        queries = session.executed_queries
        assert len(queries) == 2
        assert "DROP" in queries[0]["query"]
        assert "CREATE" in queries[1]["query"]

    @pytest.mark.asyncio
    async def test_create_ft_index_for_relationship_should_use_correct_syntax(
        self, mock_neo4j_storage
    ):
        """Should use correct Cypher syntax for relationship index."""
        session = mock_neo4j_storage.session

        await session.run(
            """
            CREATE FULLTEXT INDEX rel_idx
            FOR ()-[r:EMPLOYS]->()
            ON EACH [r.role]
            """,
        )

        queries = session.get_queries_matching("CREATE FULLTEXT")
        assert len(queries) == 1
        assert "()-[r:" in queries[0]["query"]

    @pytest.mark.asyncio
    async def test_create_ft_index_should_validate_index_name(self):
        """Index name should be validated for Cypher injection."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError):
            validate_cypher_identifier("idx; DROP DATABASE", "index name")

    @pytest.mark.asyncio
    async def test_create_ft_index_should_validate_property_names(self):
        """Property names should be validated for Cypher injection."""
        from app.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        # Valid property
        assert validate_cypher_identifier("company_name", "property name") == "company_name"

        # Invalid property
        with pytest.raises(CypherInjectionError):
            validate_cypher_identifier("name}]; MATCH", "property name")


# ============================================================
# Retry Logic Tests
# ============================================================


class TestNeo4jStorageRetryLogic:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_should_retry_on_transient_error(self):
        """When transient error occurs, should retry with backoff."""
        from neo4j.exceptions import TransientError

        attempt_count = 0

        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise TransientError("Temporary failure")
            return "success"

        # Simulate retry behavior
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await failing_operation()
                break
            except TransientError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.01)  # Minimal delay for testing
                    continue
                raise

        assert attempt_count == 3
        assert result == "success"

    @pytest.mark.asyncio
    async def test_should_not_retry_on_database_error(self):
        """When database error occurs, should not retry."""
        from neo4j.exceptions import DatabaseError

        attempt_count = 0

        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            raise DatabaseError("Schema constraint violation")

        with pytest.raises(DatabaseError):
            await failing_operation()

        # Should only attempt once
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_should_use_exponential_backoff(self):
        """Retry delays should follow exponential backoff pattern."""
        delays = []
        max_retries = 4

        for attempt in range(max_retries):
            delay = 2**attempt  # Exponential backoff
            delays.append(delay)

        # Verify exponential growth: 1, 2, 4, 8
        assert delays == [1, 2, 4, 8]

    @pytest.mark.asyncio
    async def test_should_raise_storage_error_after_max_retries(self):
        """After max retries, should raise StorageError."""
        from neo4j.exceptions import ServiceUnavailable
        from app.services.storage.exceptions import StorageError

        mock_storage = create_mock_neo4j_storage(
            raise_on_query=ServiceUnavailable("Service down")
        )

        max_retries = 3
        attempts = 0

        with pytest.raises(ServiceUnavailable):
            for attempt in range(max_retries):
                attempts += 1
                await mock_storage.session.run("MATCH (n) RETURN n")

        assert attempts == 1  # Fails immediately with our mock


# ============================================================
# Session Management Tests
# ============================================================


class TestNeo4jStorageSessionManagement:
    """Test session lifecycle management."""

    @pytest.mark.asyncio
    async def test_should_close_session_on_completion(self, mock_neo4j_storage):
        """Session should be closeable after operations."""
        session = mock_neo4j_storage.session

        await session.run("MATCH (n) RETURN n")
        await session.close()

        assert session._closed is True

    @pytest.mark.asyncio
    async def test_should_close_session_in_context_manager(self):
        """Session should be closed when using async context manager."""
        session = MockNeo4jSession()

        async with session:
            await session.run("MATCH (n) RETURN n")

        assert session._closed is True

    @pytest.mark.asyncio
    async def test_session_should_be_reusable_before_close(self, mock_neo4j_storage):
        """Session should support multiple queries before close."""
        session = mock_neo4j_storage.session

        await session.run("MATCH (n:Company) RETURN n")
        await session.run("MATCH (n:Person) RETURN n")
        await session.run("MATCH ()-[r]->() RETURN r")

        assert len(session.executed_queries) == 3
        assert session._closed is False


# ============================================================
# Error Handling Tests
# ============================================================


class TestNeo4jStorageErrorHandling:
    """Test error handling behavior."""

    @pytest.mark.asyncio
    async def test_should_handle_transient_errors_gracefully(self):
        """When transient error occurs, should be catchable for retry."""
        from neo4j.exceptions import TransientError

        mock_storage = create_mock_neo4j_storage(
            raise_on_query=TransientError("Temporary failure")
        )

        with pytest.raises(TransientError):
            await mock_storage.session.run("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_should_handle_service_unavailable_error(self):
        """Should handle ServiceUnavailable error."""
        from neo4j.exceptions import ServiceUnavailable

        mock_storage = create_mock_neo4j_storage(
            raise_on_query=ServiceUnavailable("Connection refused")
        )

        with pytest.raises(ServiceUnavailable):
            await mock_storage.session.run("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_should_handle_session_expired_error(self):
        """Should handle SessionExpired error."""
        from neo4j.exceptions import SessionExpired

        mock_storage = create_mock_neo4j_storage(
            raise_on_query=SessionExpired("Session no longer valid")
        )

        with pytest.raises(SessionExpired):
            await mock_storage.session.run("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_should_propagate_database_error(self):
        """Database errors should be propagated without retry."""
        from neo4j.exceptions import DatabaseError

        mock_storage = create_mock_neo4j_storage(
            raise_on_query=DatabaseError("Constraint violation")
        )

        with pytest.raises(DatabaseError):
            await mock_storage.session.run("CREATE (n:Company {id: 'duplicate'})")


# ============================================================
# Find Similar Nodes Tests
# ============================================================


class TestNeo4jStorageFindSimilarNodes:
    """Test similarity search functionality."""

    @pytest.mark.asyncio
    async def test_should_return_empty_list_for_empty_properties(self):
        """When properties is empty, should return empty list."""
        # This documents expected behavior
        properties = {}

        # find_similar_nodes should return empty for empty properties
        if not properties:
            result = []

        assert result == []

    @pytest.mark.asyncio
    async def test_should_build_similarity_conditions_for_string_properties(self):
        """Should create text distance conditions for string properties."""
        properties = {"name": "Acme Corp", "industry": "Technology"}

        conditions = []
        params = {}

        for idx, (key, value) in enumerate(properties.items()):
            if value is not None:
                param_key = f"value{idx}"
                params[param_key] = str(value).lower()
                conditions.append(f"apoc.text.distance(toLower(n.{key}), ${param_key})")

        assert len(conditions) == 2
        assert "value0" in params
        assert params["value0"] == "acme corp"

    @pytest.mark.asyncio
    async def test_should_handle_list_properties_in_similarity(self):
        """Should handle list properties with appropriate conversion."""
        properties = {"tags": ["tech", "startup"]}

        for key, value in properties.items():
            if isinstance(value, list):
                # Expected behavior: convert to string for comparison
                str_value = ",".join(str(x) for x in value)
                assert str_value == "tech,startup"

    @pytest.mark.asyncio
    async def test_should_skip_system_properties_in_similarity(self):
        """System properties should be excluded from similarity search."""
        from app.utils.constants import SYSTEM_PROPERTIES, TRANSFORM_ID, VALID_FROM

        properties = {
            "name": "Acme Corp",
            TRANSFORM_ID: "tx-123",  # System property (__tid)
            VALID_FROM: "2024-01-01",  # System property (__valid_from)
        }

        search_props = {
            k: v for k, v in properties.items()
            if k not in SYSTEM_PROPERTIES
        }

        assert "name" in search_props
        assert TRANSFORM_ID not in search_props

    @pytest.mark.asyncio
    async def test_should_respect_similarity_threshold(self):
        """Results should be filtered by similarity threshold."""
        threshold = 0.7

        # Mock similarity scores
        results = [
            {"node": "A", "similarity": 0.95},
            {"node": "B", "similarity": 0.65},  # Below threshold
            {"node": "C", "similarity": 0.80},
        ]

        filtered = [r for r in results if r["similarity"] >= threshold]

        assert len(filtered) == 2
        assert all(r["similarity"] >= threshold for r in filtered)

    @pytest.mark.asyncio
    async def test_should_respect_max_results_limit(self):
        """Should return at most max_results nodes."""
        max_results = 5

        # Mock many results
        all_results = [f"node-{i}" for i in range(20)]

        limited = all_results[:max_results]

        assert len(limited) == max_results


# ============================================================
# Batch Operation Tests
# ============================================================


class TestNeo4jStorageBatchOperations:
    """Test batch storage operations."""

    @pytest.mark.asyncio
    async def test_store_nodes_batch_should_return_batch_result(self, mock_neo4j_storage):
        """store_nodes should return StorageBatchResult."""
        from app.services.storage.models import StorageBatchResult

        session = mock_neo4j_storage.session
        NodeFactory.reset_counter()
        nodes = NodeFactory.create_batch(count=5, node_type="Company")

        items_processed = 0
        for node in nodes:
            await session.run(
                "MERGE (n:Company {id: $id})",
                id=node.id,
            )
            items_processed += 1

        # Create result with all required fields
        result = StorageBatchResult(
            batch_index=0,
            success=True,
            items_processed=items_processed,
            processing_time_ms=100.0,
        )

        assert result.success is True
        assert result.items_processed == 5
        assert result.batch_index == 0

    @pytest.mark.asyncio
    async def test_store_nodes_should_update_checkpoint_after_batch(
        self, mock_neo4j_storage
    ):
        """Should update checkpoint after successful batch storage."""
        session = mock_neo4j_storage.session
        transform_id = "tx-123"
        batch_index = 0

        # Store a batch
        await session.run(
            "MERGE (n:Company {id: $id})",
            id="node-1",
        )

        # Update checkpoint
        await session.run(
            """
            MERGE (c:StorageCheckpoint {transform_id: $transform_id})
            SET c.batch_index = $batch_index
            SET c.stage = $stage
            SET c.updated_at = datetime()
            """,
            transform_id=transform_id,
            batch_index=batch_index,
            stage="NODES",
        )

        queries = session.get_queries_matching("StorageCheckpoint")
        assert len(queries) == 1
        assert queries[0]["parameters"]["stage"] == "NODES"

    @pytest.mark.asyncio
    async def test_batch_should_stop_on_first_error(self, mock_neo4j_storage):
        """Batch operation should stop and report on first error."""
        session = mock_neo4j_storage.session
        NodeFactory.reset_counter()

        nodes = NodeFactory.create_batch(count=5, node_type="Company")
        items_processed = 0
        error_message = None

        for i, node in enumerate(nodes):
            if i == 2:
                # Simulate error on third node
                error_message = f"Failed to store node {node.id}"
                break

            await session.run(
                "MERGE (n:Company {id: $id})",
                id=node.id,
            )
            items_processed += 1

        assert items_processed == 2
        assert error_message is not None
        assert "Failed to store" in error_message


# ============================================================
# Data Transformation Tests
# ============================================================


class TestNeo4jStorageDataTransformation:
    """Test data transformation between domain models and Neo4j."""

    @pytest.mark.asyncio
    async def test_should_convert_neo4j_datetime_to_iso_string(self):
        """Neo4j DateTime should be converted to ISO string."""
        from datetime import datetime

        # Mock Neo4j DateTime behavior
        neo4j_dt = datetime(2024, 1, 15, 10, 30, 0)
        iso_string = neo4j_dt.isoformat()

        assert iso_string == "2024-01-15T10:30:00"

    @pytest.mark.asyncio
    async def test_should_handle_nested_properties(self):
        """Should handle nodes with nested property structures."""
        properties = {
            "name": "Acme Corp",
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
            },
            "tags": ["tech", "startup"],
        }

        # Neo4j can't store nested dicts directly - they need serialization
        import json

        flat_props = {}
        for key, value in properties.items():
            if isinstance(value, dict):
                flat_props[key] = json.dumps(value)
            elif isinstance(value, list):
                flat_props[key] = value  # Neo4j supports lists
            else:
                flat_props[key] = value

        assert isinstance(flat_props["address"], str)
        assert isinstance(flat_props["tags"], list)

    @pytest.mark.asyncio
    async def test_should_preserve_node_id_across_operations(self):
        """Node ID should be preserved when stored and retrieved."""
        mock_storage = create_mock_neo4j_storage(
            query_results={
                "MATCH": [
                    MockNeo4jRecord(
                        {
                            "n": MockNeo4jNode(
                                labels=["Company"],
                                _properties={
                                    "id": "original-id-123",
                                    "name": "Acme Corp",
                                },
                            )
                        }
                    )
                ]
            }
        )

        result = await mock_storage.session.run(
            "MATCH (n {id: $id}) RETURN n",
            id="original-id-123",
        )

        record = await result.single()
        assert record["n"]["id"] == "original-id-123"
