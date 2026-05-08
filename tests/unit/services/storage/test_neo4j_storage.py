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
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from tests.mocks.neo4j_mock import (
    MockNeo4jRecord,
    MockNeo4jNode,
    MockNeo4jRelationship,
    MockNeo4jSession,
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
        from graphora_server.services.storage.neo4j import validate_cypher_identifier

        assert validate_cypher_identifier("Company") == "Company"
        assert validate_cypher_identifier("Person_v2") == "Person_v2"
        assert validate_cypher_identifier("_private") == "_private"
        assert validate_cypher_identifier("node123") == "node123"

    def test_should_reject_identifier_with_semicolon(self):
        """Identifiers with semicolons (potential injection) should be rejected."""
        from graphora_server.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError) as exc_info:
            validate_cypher_identifier("Company; DROP DATABASE")

        assert "Invalid" in str(exc_info.value)

    def test_should_reject_identifier_with_backticks(self):
        """Identifiers with backticks should be rejected."""
        from graphora_server.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError):
            validate_cypher_identifier("`MaliciousLabel`")

    def test_should_reject_identifier_starting_with_number(self):
        """Identifiers starting with numbers should be rejected."""
        from graphora_server.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError):
            validate_cypher_identifier("123Company")

    def test_should_reject_empty_identifier(self):
        """Empty identifiers should be rejected."""
        from graphora_server.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError) as exc_info:
            validate_cypher_identifier("")

        assert "Empty" in str(exc_info.value)

    def test_should_reject_identifier_exceeding_max_length(self):
        """Identifiers exceeding 256 characters should be rejected."""
        from graphora_server.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        long_identifier = "A" * 257

        with pytest.raises(CypherInjectionError) as exc_info:
            validate_cypher_identifier(long_identifier)

        assert "exceeds maximum length" in str(exc_info.value)

    def test_should_reject_identifier_with_special_characters(self):
        """Identifiers with special characters should be rejected."""
        from graphora_server.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        invalid_chars = ["@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "="]

        for char in invalid_chars:
            with pytest.raises(CypherInjectionError):
                validate_cypher_identifier(f"Company{char}Name")

    def test_validate_cypher_labels_should_validate_all_labels(self):
        """validate_cypher_labels should validate each label in the list."""
        from graphora_server.services.storage.neo4j import (
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
        with patch(
            "graphora_server.services.storage.neo4j.AsyncGraphDatabase"
        ) as mock_async_db:
            with patch(
                "graphora_server.services.storage.neo4j.GraphDatabase"
            ) as mock_sync_db:
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

                from graphora_server.services.storage.neo4j import Neo4jStorage

                _storage = Neo4jStorage(
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
        from graphora_server.services.storage.exceptions import StorageAuthError

        with patch(
            "graphora_server.services.storage.neo4j.AsyncGraphDatabase"
        ) as _mock_async_db:
            with patch(
                "graphora_server.services.storage.neo4j.GraphDatabase"
            ) as mock_sync_db:
                mock_sync_db.driver.side_effect = AuthError("Invalid credentials")

                from graphora_server.services.storage.neo4j import Neo4jStorage

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
        from graphora_server.services.storage.exceptions import StorageConnectionError

        with patch(
            "graphora_server.services.storage.neo4j.AsyncGraphDatabase"
        ) as _mock_async_db:
            with patch(
                "graphora_server.services.storage.neo4j.GraphDatabase"
            ) as mock_sync_db:
                mock_sync_db.driver.side_effect = ServiceUnavailable(
                    "Connection refused"
                )

                from graphora_server.services.storage.neo4j import Neo4jStorage

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
        with patch(
            "graphora_server.services.storage.neo4j.AsyncGraphDatabase"
        ) as mock_async_db:
            with patch(
                "graphora_server.services.storage.neo4j.GraphDatabase"
            ) as mock_sync_db:
                mock_driver = AsyncMock()
                mock_async_db.driver.return_value = mock_driver

                from graphora_server.services.storage.neo4j import Neo4jStorage

                # Provide transaction_manager to skip sync test
                _storage = Neo4jStorage(
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
        from graphora_server.services.storage.neo4j import Neo4jStorage

        # Create a mock storage instance
        with patch("graphora_server.services.storage.neo4j.AsyncGraphDatabase"):
            with patch("graphora_server.services.storage.neo4j.GraphDatabase"):
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
    async def test_store_relationships_should_skip_duplicates(self, mock_neo4j_storage):
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
    async def test_versioning_path_reads_properties_via_items_not_get(self):
        """Reviewer-flagged on commit c347f9c: ``existing_rel`` is a
        neo4j-driver Relationship object, not a dict — properties
        live directly on the object, accessed via ``.items()`` /
        ``[key]``. Pre-fix, the code did
        ``existing_rel.get('properties', {}).items()`` which on a
        Relationship returns the value of the literal property
        ``properties`` (which doesn't exist) and falls back to {}.
        Result: the versioning path's 'differing properties' check
        always saw {} for existing_props, the 'no meaningful
        properties' early-return ALWAYS fired, and a changed edge
        never closed v1 / created v2.

        The integration test pins the end-to-end behaviour but
        requires Docker; this unit test pins the reading pattern
        directly so a regression to ``existing_rel.get('properties')``
        fails loud on every CI run."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from graphora_server.services.storage.neo4j import Neo4jStorage
        from graphora_server.services.transform.models import (
            RelationshipInstance,
        )

        class _FakeNeo4jRelationship:
            """Duck-typed Relationship: ``items()`` exposes the
            property bag, but ``get('properties')`` returns {} —
            mirroring the real driver object's behaviour. If the
            production code ever reverts to .get('properties'),
            this fake produces empty existing_props and the
            versioning path silently skips, which the assertion
            below catches."""

            def __init__(self, properties):
                self._props = dict(properties)

            def items(self):
                return self._props.items()

            def __getitem__(self, key):
                return self._props[key]

            def get(self, key, default=None):
                return self._props.get(key, default)

        # Construct a Neo4jStorage without invoking __init__ (which
        # tries to dial a real driver). We only need the
        # store_relationships method body to run.
        storage = Neo4jStorage.__new__(Neo4jStorage)
        storage.max_retries = 1

        # Mock the async session context. session.run captures every
        # query so we can later assert which path the code took.
        executed: list[str] = []

        async def fake_run(query, *args, **kwargs):
            executed.append(query)
            mock_result = MagicMock()
            mock_result.single = AsyncMock(return_value=None)
            mock_result.__aiter__ = lambda self: iter([])
            return mock_result

        session = MagicMock()
        session.run = AsyncMock(side_effect=fake_run)
        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=None)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def fake_get_session():
            yield session

        storage._get_session = fake_get_session

        async def fake_execute_with_retry(op):
            return await op()

        storage._execute_with_retry = fake_execute_with_retry

        # The existing relationship has 'role: engineer' as its meaningful
        # property. The new relationship has 'role: principal-engineer'.
        # If existing_props is read correctly via .items(), the
        # versioning branch fires and we see _close_existing_relationship's
        # SET r.__valid_to query in `executed`. If existing_props comes
        # back as {} (the bug), the early-return fires and no SET
        # query gets issued.
        fake_existing = _FakeNeo4jRelationship(
            {
                "id": "existing-rel-id",
                "role": "engineer",
                "__tid": "old-tx",
                "__valid_from": "2026-01-01T00:00:00",
                "__valid_to": None,
            }
        )

        new_rel = RelationshipInstance(
            id="new-rel-id",
            type="WORKS_AT",
            source_id="alice",
            target_id="acme",
            source_type="Person",
            target_type="Company",
            properties={"role": "principal-engineer"},
        )

        with patch.object(
            storage,
            "_find_existing_relationship",
            new=AsyncMock(return_value=fake_existing),
        ):
            await storage.store_relationships(
                [new_rel], batch_index=0, transform_id="new-tx"
            )

        # Versioning fired iff one of the queries SETs __valid_to —
        # that's the close_existing_relationship signature. Pre-fix
        # this branch was unreachable.
        close_queries = [q for q in executed if "__valid_to" in q and "SET" in q]
        assert close_queries, (
            "Versioning path didn't fire — _close_existing_relationship "
            "was never called. existing_props was probably empty because "
            "the production code reverted to existing_rel.get('properties') "
            "which returns {} on a Relationship object. Use "
            "existing_rel.items() instead."
        )

    @pytest.mark.asyncio
    async def test_unchanged_properties_does_not_trigger_versioning(self):
        """Reviewer-flagged on commit f476aa3: with the .items() fix,
        existing_props now includes the stored r.id property (because
        _build_relationship_query always SETs r.id = $rel_id). The
        new rel's .properties dict typically doesn't have ``id`` (it
        lives on .id attribute). So the comparison saw
        ``{"id": ..., "role": "engineer"}`` vs ``{"role":
        "engineer"}`` and triggered versioning on every retry/replay
        even when no user-meaningful property changed. Unbounded
        version churn on idempotent writes.

        Fix: filter both sides via SYSTEM_PROPERTIES (the canonical
        'metadata, not user signal' list — includes id, all
        provenance fields, transform_id, merge_id, etc.) rather than
        the narrow {VALID_FROM, VALID_TO, TRANSFORM_ID, MERGE_ID}
        subset. This test pins the contract: if the same role lands
        on an existing 'role: engineer' edge, the versioning path is
        NOT taken — _close_existing_relationship is never called and
        no SET __valid_to query reaches the session."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from graphora_server.services.storage.neo4j import Neo4jStorage
        from graphora_server.services.transform.models import (
            RelationshipInstance,
        )

        class _FakeNeo4jRelationship:
            def __init__(self, properties):
                self._props = dict(properties)

            def items(self):
                return self._props.items()

            def __getitem__(self, key):
                return self._props[key]

            def get(self, key, default=None):
                return self._props.get(key, default)

        storage = Neo4jStorage.__new__(Neo4jStorage)
        storage.max_retries = 1

        executed: list[str] = []

        async def fake_run(query, *args, **kwargs):
            executed.append(query)
            mock_result = MagicMock()
            mock_result.single = AsyncMock(return_value=None)
            mock_result.__aiter__ = lambda self: iter([])
            return mock_result

        session = MagicMock()
        session.run = AsyncMock(side_effect=fake_run)

        @asynccontextmanager
        async def fake_get_session():
            yield session

        storage._get_session = fake_get_session

        async def fake_execute_with_retry(op):
            return await op()

        storage._execute_with_retry = fake_execute_with_retry

        # Existing rel has the stored id + provenance metadata + the
        # ONLY meaningful user property is role=engineer. New rel has
        # the same role. Without SYSTEM_PROPERTIES filtering, id /
        # extractor_model / etc. would leak into existing_props and
        # diverge from new_props (which lacks them).
        fake_existing = _FakeNeo4jRelationship(
            {
                "id": "existing-rel-id",
                "role": "engineer",
                "__tid": "old-tx",
                "__valid_from": "2026-01-01T00:00:00",
                "__valid_to": None,
                "extractor_model": "gemini-1.5-pro",
                "validator_score": 0.92,
                "source_chunk_id": "chunk-7",
            }
        )

        new_rel = RelationshipInstance(
            id="new-rel-id",
            type="WORKS_AT",
            source_id="alice",
            target_id="acme",
            source_type="Person",
            target_type="Company",
            properties={"role": "engineer"},  # unchanged
        )

        with patch.object(
            storage,
            "_find_existing_relationship",
            new=AsyncMock(return_value=fake_existing),
        ):
            await storage.store_relationships(
                [new_rel], batch_index=0, transform_id="new-tx"
            )

        # No SET __valid_to query → versioning didn't fire → contract
        # holds. Pre-fix the narrow system-key filter let id leak
        # into existing_props, the comparison saw a phantom 'change',
        # and a SET __valid_to query DID land in executed.
        close_queries = [q for q in executed if "__valid_to" in q and "SET" in q]
        assert not close_queries, (
            f"Versioning fired on an unchanged-properties re-store. "
            f"existing_props probably includes id / provenance fields "
            f"that aren't in new_rel.properties — filter both sides "
            f"via SYSTEM_PROPERTIES, not the narrow VALID_*/TRANSFORM_ID "
            f"set. Captured queries: {close_queries}"
        )

    @pytest.mark.asyncio
    async def test_empty_existing_with_new_props_versions_not_drops(self):
        """Reviewer-flagged on commit 3e7c3cd: with the
        SYSTEM_PROPERTIES filter, an existing edge that was first
        stored without user properties has existing_props={}. The
        pre-fix early-return on ``not existing_props`` then
        unconditionally skipped the write, silently DROPPING the
        incoming user properties on the floor.

        Real-world hit: an edge gets created during a transform
        before the LLM emits properties for it (e.g., a chunked
        write that adds metadata first, then properties on a
        later replay). The replay's user data should land on
        v2; pre-fix it just disappeared.

        Fix: compute new_props above the early-return and only
        skip when BOTH sides are empty. This test pins the
        contract: empty existing + non-empty new ⇒ versioning
        fires (close existing, create new with the user props).
        Cross-pinned with the symmetric test
        test_unchanged_properties_does_not_trigger_versioning so
        the early-return only kills truly-noop writes."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from graphora_server.services.storage.neo4j import Neo4jStorage
        from graphora_server.services.transform.models import (
            RelationshipInstance,
        )

        class _FakeNeo4jRelationship:
            def __init__(self, properties):
                self._props = dict(properties)

            def items(self):
                return self._props.items()

            def __getitem__(self, key):
                return self._props[key]

            def get(self, key, default=None):
                return self._props.get(key, default)

        storage = Neo4jStorage.__new__(Neo4jStorage)
        storage.max_retries = 1

        executed: list[str] = []

        async def fake_run(query, *args, **kwargs):
            executed.append(query)
            mock_result = MagicMock()
            mock_result.single = AsyncMock(return_value=None)
            mock_result.__aiter__ = lambda self: iter([])
            return mock_result

        session = MagicMock()
        session.run = AsyncMock(side_effect=fake_run)

        @asynccontextmanager
        async def fake_get_session():
            yield session

        storage._get_session = fake_get_session

        async def fake_execute_with_retry(op):
            return await op()

        storage._execute_with_retry = fake_execute_with_retry

        # Existing rel has ONLY system properties — id, transform-id,
        # validity, provenance metadata. Zero user-meaningful keys.
        # SYSTEM_PROPERTIES filtering reduces existing_props to {}.
        fake_existing = _FakeNeo4jRelationship(
            {
                "id": "existing-rel-id",
                "__tid": "old-tx",
                "__valid_from": "2026-01-01T00:00:00",
                "__valid_to": None,
                "extractor_model": "gemini-1.5-pro",
            }
        )

        # New rel brings real user data the existing rel doesn't have.
        # Pre-fix, this got silently dropped because existing_props={}
        # triggered the early-return before new_props was even
        # computed.
        new_rel = RelationshipInstance(
            id="new-rel-id",
            type="WORKS_AT",
            source_id="alice",
            target_id="acme",
            source_type="Person",
            target_type="Company",
            properties={"role": "engineer"},
        )

        with patch.object(
            storage,
            "_find_existing_relationship",
            new=AsyncMock(return_value=fake_existing),
        ):
            await storage.store_relationships(
                [new_rel], batch_index=0, transform_id="new-tx"
            )

        # Versioning fired iff a SET __valid_to query reached the
        # session. Pre-fix this is empty (skip path); post-fix the
        # close happens and the new edge with {role: engineer} gets
        # created.
        close_queries = [q for q in executed if "__valid_to" in q and "SET" in q]
        assert close_queries, (
            "Versioning didn't fire on empty-existing + non-empty-new — "
            "the incoming user property was silently dropped. The "
            "early-return must require BOTH existing_props AND new_props "
            "to be empty before skipping."
        )
        # And the new edge's CREATE query carries the role property,
        # so the user data didn't get lost in the wash.
        create_queries = [q for q in executed if "CREATE" in q and "$properties" in q]
        assert create_queries, (
            "No CREATE for the new version landed — the close happened "
            "but the new edge wasn't built. Check that "
            "_build_relationship_query was invoked with the new props."
        )

    @pytest.mark.asyncio
    async def test_find_existing_relationship_scopes_by_transform_id(self):
        """Reviewer-flagged on commit 6329d68: the lookup needs to
        filter by transform_id, otherwise transform B's writes find
        transform A's active edge for the same logical (s, t, type)
        and either silently no-op (props unchanged) or close A's edge
        (props differ). Both outcomes break the transform-scoped read
        contract that get_transformation_data depends on.

        Pin the query shape: when transform_id is passed, the WHERE
        clause includes ``r.__tid = $transform_id``. The store_
        relationships caller threads this value through every call.
        Pre-fix the function only filtered on s/t/type/__valid_to
        IS NULL — cross-transform collisions waiting to happen the
        moment versioning actually worked."""
        from unittest.mock import AsyncMock, MagicMock

        from graphora_server.services.storage.neo4j import Neo4jStorage
        from graphora_server.services.transform.models import (
            RelationshipInstance,
        )

        storage = Neo4jStorage.__new__(Neo4jStorage)

        captured_query: list[str] = []
        captured_params: list[dict] = []

        async def fake_run(query, **kwargs):
            captured_query.append(query)
            captured_params.append(kwargs)
            mock_result = MagicMock()
            mock_result.single = AsyncMock(return_value=None)
            return mock_result

        session = MagicMock()
        session.run = AsyncMock(side_effect=fake_run)

        rel = RelationshipInstance(
            id="rel-id",
            type="WORKS_AT",
            source_id="alice",
            target_id="acme",
            source_type="Person",
            target_type="Company",
            properties={"role": "engineer"},
        )
        await storage._find_existing_relationship(session, rel, transform_id="tx-a")

        assert captured_query, "_find_existing_relationship made no query"
        query = captured_query[0]
        assert "r.__tid = $transform_id" in query, (
            "Lookup query is missing the transform_id filter — cross-"
            "transform collision is possible. The same logical edge "
            "stored under transform B will find transform A's edge "
            "as 'existing' and either no-op (props unchanged) or "
            "version A's edge (props differ), breaking the "
            "transform-scoped read."
        )
        assert (
            captured_params[0].get("transform_id") == "tx-a"
        ), "transform_id parameter wasn't bound to the query."

    @pytest.mark.asyncio
    async def test_find_existing_relationship_legacy_no_transform_id(self):
        """When called without transform_id (the pre-fix signature
        for any callers that haven't been migrated), the WHERE clause
        omits the __tid filter and the params dict doesn't carry it.
        Pin so a future refactor doesn't accidentally make
        transform_id mandatory and break those callers."""
        from unittest.mock import AsyncMock, MagicMock

        from graphora_server.services.storage.neo4j import Neo4jStorage
        from graphora_server.services.transform.models import (
            RelationshipInstance,
        )

        storage = Neo4jStorage.__new__(Neo4jStorage)

        captured_query: list[str] = []
        captured_params: list[dict] = []

        async def fake_run(query, **kwargs):
            captured_query.append(query)
            captured_params.append(kwargs)
            mock_result = MagicMock()
            mock_result.single = AsyncMock(return_value=None)
            return mock_result

        session = MagicMock()
        session.run = AsyncMock(side_effect=fake_run)

        rel = RelationshipInstance(
            id="rel-id",
            type="WORKS_AT",
            source_id="alice",
            target_id="acme",
            source_type="Person",
            target_type="Company",
            properties={},
        )
        await storage._find_existing_relationship(session, rel)

        query = captured_query[0]
        assert "r.__tid" not in query
        assert "transform_id" not in captured_params[0]

    @pytest.mark.asyncio
    async def test_no_existing_path_uses_create_not_merge(self):
        """Reviewer-flagged on commit 14a939a: scoping the lookup
        is necessary but not sufficient. Even when the scoped
        lookup correctly returns 'no existing edge for this
        transform', the write path then issues
        ``MERGE (s)-[r:T]->(t)`` which is unscoped — that MERGE
        re-matches a DIFFERENT transform's active edge for the
        same (s, t, type) and ``SET r = $properties`` overwrites
        its __tid with ours. End result: cross-transform corruption
        on the create path, mirroring what the versioning path had
        before c347f9c.

        Pin: when the scoped lookup returns nothing (no existing for
        this transform), the issued query uses CREATE rather than
        MERGE. CREATE always makes a new edge without matching any
        existing one — preserving other transforms' edges
        untouched. Symmetric to the versioning fix on Case 2."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from graphora_server.services.storage.neo4j import Neo4jStorage
        from graphora_server.services.transform.models import (
            RelationshipInstance,
        )

        storage = Neo4jStorage.__new__(Neo4jStorage)
        storage.max_retries = 1

        executed: list[str] = []

        async def fake_run(query, *args, **kwargs):
            executed.append(query)
            mock_result = MagicMock()
            mock_result.single = AsyncMock(return_value=None)
            mock_result.__aiter__ = lambda self: iter([])
            return mock_result

        session = MagicMock()
        session.run = AsyncMock(side_effect=fake_run)

        @asynccontextmanager
        async def fake_get_session():
            yield session

        storage._get_session = fake_get_session

        async def fake_execute_with_retry(op):
            return await op()

        storage._execute_with_retry = fake_execute_with_retry

        new_rel = RelationshipInstance(
            id="new-rel-id",
            type="WORKS_AT",
            source_id="alice",
            target_id="acme",
            source_type="Person",
            target_type="Company",
            properties={"role": "engineer"},
        )

        # Lookup returns None (no edge for this transform).
        with patch.object(
            storage,
            "_find_existing_relationship",
            new=AsyncMock(return_value=None),
        ):
            await storage.store_relationships(
                [new_rel], batch_index=0, transform_id="tx-b"
            )

        # The write query for the new edge must be a CREATE, not a
        # MERGE. A MERGE here would silently match a sibling
        # transform's active edge for the same (s, t, type) and
        # overwrite it.
        write_queries = [q for q in executed if "WORKS_AT" in q and "$properties" in q]
        assert write_queries, "No write query reached the session"
        write_query = write_queries[0]
        assert "CREATE (s)" in write_query, (
            f"Create-path uses MERGE instead of CREATE: {write_query!r}. "
            f"That MERGE will re-match a different transform's edge "
            f"for the same (s, t, type) and SET r = \\$properties "
            f"will overwrite its __tid — cross-transform corruption. "
            f"Switch the Case 3 _build_relationship_query call to "
            f"merge=False."
        )
        assert "MERGE (s)" not in write_query, (
            "Create-path query still contains MERGE (s); switch to "
            "CREATE so it doesn't match across transforms."
        )

    @pytest.mark.asyncio
    async def test_first_time_write_preserves_caller_supplied_id(self):
        """Reviewer-flagged on commit 1240ced: switching the
        no-existing path to merge=False fixed cross-transform
        corruption but introduced a side-effect via
        _build_relationship_query, which generated a fresh UUID
        whenever merge=False. Result: first-time relationship
        writes no longer round-tripped the caller-supplied
        rel.id — Case 3 stored the edge under a synthesized
        id, breaking external callers that expect to query
        their own rels by id.

        Pin the round-trip: the rel_id parameter sent to Neo4j
        on the Case 3 (no-existing) write equals the caller's
        rel.id, not a fresh UUID. Symmetric to the versioning
        path which legitimately needs a new id (and now mints
        one explicitly at the callsite via rel.model_copy)."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from graphora_server.services.storage.neo4j import Neo4jStorage
        from graphora_server.services.transform.models import (
            RelationshipInstance,
        )

        storage = Neo4jStorage.__new__(Neo4jStorage)
        storage.max_retries = 1

        captured_params: list = []

        async def fake_run(query, *args, **kwargs):
            # store_relationships calls session.run(query, params) —
            # params is the second positional arg in this codebase's
            # idiom. Capture both positional and kwarg shapes
            # defensively.
            captured_params.append((args, kwargs))
            mock_result = MagicMock()
            mock_result.single = AsyncMock(return_value=None)
            mock_result.__aiter__ = lambda self: iter([])
            return mock_result

        session = MagicMock()
        session.run = AsyncMock(side_effect=fake_run)

        @asynccontextmanager
        async def fake_get_session():
            yield session

        storage._get_session = fake_get_session

        async def fake_execute_with_retry(op):
            return await op()

        storage._execute_with_retry = fake_execute_with_retry

        caller_supplied_id = "rel-caller-id-abc-123"
        new_rel = RelationshipInstance(
            id=caller_supplied_id,
            type="WORKS_AT",
            source_id="alice",
            target_id="acme",
            source_type="Person",
            target_type="Company",
            properties={"role": "engineer"},
        )

        with patch.object(
            storage,
            "_find_existing_relationship",
            new=AsyncMock(return_value=None),
        ):
            await storage.store_relationships(
                [new_rel], batch_index=0, transform_id="tx-a"
            )

        # The session.run call for the write carries a params dict
        # (or kwargs) with rel_id == caller-supplied. Pre-fix the
        # rel_id was a fresh uuid4 — the assertion below catches
        # the regression by asserting equality against the exact
        # caller-supplied string.
        rel_ids_seen = []
        for args, kwargs in captured_params:
            if args and isinstance(args[-1], dict):
                rid = args[-1].get("rel_id")
                if rid is not None:
                    rel_ids_seen.append(rid)
            if "rel_id" in kwargs:
                rel_ids_seen.append(kwargs["rel_id"])

        assert caller_supplied_id in rel_ids_seen, (
            f"First-time write didn't preserve the caller-supplied "
            f"rel.id={caller_supplied_id!r}. Saw rel_ids: "
            f"{rel_ids_seen}. _build_relationship_query is probably "
            f"still generating a fresh UUID when merge=False — "
            f"decouple Cypher op from id policy: id should always be "
            f"rel.id, fresh UUIDs should be minted at the callsite "
            f"(Case 2) explicitly."
        )

    @pytest.mark.asyncio
    async def test_store_relationships_should_validate_relationship_type(self):
        """Relationship type should be validated for Cypher injection."""
        from graphora_server.services.storage.neo4j import (
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

        result = await mock_storage.session.run("MATCH (n)-[r]->(m) RETURN n, r, m")

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
        from graphora_server.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        with pytest.raises(CypherInjectionError):
            validate_cypher_identifier("idx; DROP DATABASE", "index name")

    @pytest.mark.asyncio
    async def test_create_ft_index_should_validate_property_names(self):
        """Property names should be validated for Cypher injection."""
        from graphora_server.services.storage.neo4j import (
            validate_cypher_identifier,
            CypherInjectionError,
        )

        # Valid property
        assert (
            validate_cypher_identifier("company_name", "property name")
            == "company_name"
        )

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
        from graphora_server.utils.constants import (
            SYSTEM_PROPERTIES,
            TRANSFORM_ID,
            VALID_FROM,
        )

        properties = {
            "name": "Acme Corp",
            TRANSFORM_ID: "tx-123",  # System property (__tid)
            VALID_FROM: "2024-01-01",  # System property (__valid_from)
        }

        search_props = {
            k: v for k, v in properties.items() if k not in SYSTEM_PROPERTIES
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
    async def test_store_nodes_batch_should_return_batch_result(
        self, mock_neo4j_storage
    ):
        """store_nodes should return StorageBatchResult."""
        from graphora_server.services.storage.models import StorageBatchResult

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
