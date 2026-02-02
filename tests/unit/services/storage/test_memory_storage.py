"""Unit tests for in-memory graph storage."""

import pytest
from datetime import datetime, timezone

from app.services.storage.memory import (
    InMemoryStorage,
    InMemoryGraphStore,
    get_memory_store,
    clear_memory_store,
    clear_all_memory_stores,
)
from app.services.storage.models import StorageStage
from app.services.transform.models import BaseNode, RelationshipInstance
from app.schemas.graph import Node, Edge


@pytest.fixture
def storage():
    """Create a fresh in-memory storage instance."""
    clear_all_memory_stores()
    return InMemoryStorage(user_id="test-user")


@pytest.fixture
def sample_nodes():
    """Create sample BaseNode instances for testing."""
    return [
        BaseNode(
            id="node-1",
            type="Person",
            properties={"name": "Alice", "age": 30},
            canonical_properties={"canonical_name": "alice"},
        ),
        BaseNode(
            id="node-2",
            type="Person",
            properties={"name": "Bob", "age": 25},
            canonical_properties={"canonical_name": "bob"},
        ),
        BaseNode(
            id="node-3",
            type="Company",
            properties={"name": "Acme Corp"},
            canonical_properties={},
        ),
    ]


@pytest.fixture
def sample_relationships():
    """Create sample RelationshipInstance instances for testing."""
    return [
        RelationshipInstance(
            id="rel-1",
            type="WORKS_AT",
            source_id="node-1",
            target_id="node-3",
            source_type="Person",
            target_type="Company",
            properties={"since": 2020},
        ),
        RelationshipInstance(
            id="rel-2",
            type="KNOWS",
            source_id="node-1",
            target_id="node-2",
            source_type="Person",
            target_type="Person",
            properties={"since": 2015},
        ),
    ]


class TestInMemoryStorage:
    """Tests for InMemoryStorage class."""

    @pytest.mark.asyncio
    async def test_store_nodes(self, storage, sample_nodes):
        """Test storing nodes in memory."""
        result = await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )

        assert result.success
        assert result.items_processed == 3
        assert result.batch_index == 0
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_store_relationships(self, storage, sample_nodes, sample_relationships):
        """Test storing relationships in memory."""
        # First store nodes
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )

        # Then store relationships
        result = await storage.store_relationships(
            relationships=sample_relationships,
            batch_index=0,
            transform_id="test-transform-1",
        )

        assert result.success
        assert result.items_processed == 2
        assert len(result.warnings) == 0

    @pytest.mark.asyncio
    async def test_store_relationships_missing_nodes(self, storage):
        """Test storing relationships when nodes don't exist."""
        relationships = [
            RelationshipInstance(
                id="rel-1",
                type="KNOWS",
                source_id="nonexistent-1",
                target_id="nonexistent-2",
                source_type="Person",
                target_type="Person",
            )
        ]

        result = await storage.store_relationships(
            relationships=relationships,
            batch_index=0,
            transform_id="test-transform-1",
        )

        assert result.success
        assert result.items_processed == 0
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_get_transformation_data(self, storage, sample_nodes, sample_relationships):
        """Test retrieving transformation data."""
        transform_id = "test-transform-1"

        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id=transform_id,
        )
        await storage.store_relationships(
            relationships=sample_relationships,
            batch_index=0,
            transform_id=transform_id,
        )

        response = storage.get_transformation_data(transform_id)

        assert response.total_nodes == 3
        assert response.total_edges == 2
        assert len(response.nodes) == 3
        assert len(response.edges) == 2

    @pytest.mark.asyncio
    async def test_get_node_by_id(self, storage, sample_nodes):
        """Test retrieving a node by ID."""
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )

        node = await storage.get_node_by_id("node-1")

        assert node is not None
        assert node.id == "node-1"
        assert node.label == "Person"
        assert node.properties["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_node_by_id_not_found(self, storage):
        """Test retrieving a non-existent node."""
        node = await storage.get_node_by_id("nonexistent")
        assert node is None

    @pytest.mark.asyncio
    async def test_find_nodes_by_property_value(self, storage, sample_nodes):
        """Test finding nodes by property value."""
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )

        # Exact match
        nodes = await storage.find_nodes_by_property_value(
            label="Person",
            property_name="name",
            property_value="Alice",
            exact_match=True,
        )

        assert len(nodes) == 1
        assert nodes[0].properties["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_find_nodes_by_property_value_partial(self, storage, sample_nodes):
        """Test finding nodes by partial property value."""
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )

        # Partial match
        nodes = await storage.find_nodes_by_property_value(
            label="Person",
            property_name="name",
            property_value="ali",
            exact_match=False,
        )

        assert len(nodes) == 1
        assert nodes[0].properties["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_relationships_between(self, storage, sample_nodes, sample_relationships):
        """Test getting relationships between two nodes."""
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )
        await storage.store_relationships(
            relationships=sample_relationships,
            batch_index=0,
            transform_id="test-transform-1",
        )

        edges = await storage.get_relationships_between(
            source_id="node-1",
            target_id="node-2",
        )

        assert len(edges) == 1
        assert edges[0].type == "KNOWS"

    @pytest.mark.asyncio
    async def test_get_relationships_between_with_type(self, storage, sample_nodes, sample_relationships):
        """Test getting relationships between two nodes filtered by type."""
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )
        await storage.store_relationships(
            relationships=sample_relationships,
            batch_index=0,
            transform_id="test-transform-1",
        )

        edges = await storage.get_relationships_between(
            source_id="node-1",
            target_id="node-3",
            relationship_type="WORKS_AT",
        )

        assert len(edges) == 1
        assert edges[0].type == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_create_node(self, storage):
        """Test creating a new node."""
        node = await storage.create_node(
            label="Product",
            properties={"name": "Widget", "price": 99.99},
        )

        assert node.id is not None
        assert node.label == "Product"
        assert node.properties["name"] == "Widget"

    @pytest.mark.asyncio
    async def test_update_node(self, storage, sample_nodes):
        """Test updating an existing node."""
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )

        updated = await storage.update_node(
            node_id="node-1",
            properties={"age": 31, "city": "NYC"},
        )

        assert updated.properties["age"] == 31
        assert updated.properties["city"] == "NYC"
        assert updated.properties["name"] == "Alice"  # Original preserved

    @pytest.mark.asyncio
    async def test_update_node_not_found(self, storage):
        """Test updating a non-existent node."""
        with pytest.raises(ValueError, match="not found"):
            await storage.update_node(
                node_id="nonexistent",
                properties={"foo": "bar"},
            )

    @pytest.mark.asyncio
    async def test_create_relationship(self, storage, sample_nodes):
        """Test creating a new relationship."""
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )

        edge = await storage.create_relationship(
            source_id="node-2",
            target_id="node-3",
            rel_type="WORKS_AT",
            properties={"position": "Engineer"},
        )

        assert edge.id is not None
        assert edge.source == "node-2"
        assert edge.target == "node-3"
        assert edge.type == "WORKS_AT"
        assert edge.properties["position"] == "Engineer"

    @pytest.mark.asyncio
    async def test_checkpoint_operations(self, storage):
        """Test checkpoint get/update operations."""
        transform_id = "test-transform-1"

        # Initially no checkpoint
        status = await storage.get_storage_status(transform_id)
        assert status is None

        # Update checkpoint
        await storage.update_checkpoint(
            transform_id=transform_id,
            last_index=10,
            stage=StorageStage.NODES,
        )

        # Retrieve checkpoint
        status = await storage.get_storage_status(transform_id)
        assert status is not None
        assert status.last_processed_index == 10
        assert status.stage == StorageStage.NODES

    @pytest.mark.asyncio
    async def test_find_similar_nodes(self, storage, sample_nodes):
        """Test finding similar nodes."""
        await storage.store_nodes(
            nodes=sample_nodes,
            batch_index=0,
            transform_id="test-transform-1",
        )

        similar = await storage.find_similar_nodes(
            label="Person",
            properties={"name": "Alice"},
            similarity_threshold=0.5,
            max_results=5,
        )

        assert len(similar) >= 1
        # Alice should be the most similar
        assert similar[0].properties["name"] == "Alice"


class TestInMemoryGraphStore:
    """Tests for the underlying InMemoryGraphStore."""

    def test_isolation_between_users(self):
        """Test that different users have isolated stores."""
        clear_all_memory_stores()

        store1 = get_memory_store("user-1")
        store2 = get_memory_store("user-2")

        node = Node(id="test-node", label="Test", type="Test", properties={})
        store1.add_node(node, "transform-1")

        # User 1 should see the node
        assert store1.get_node("test-node") is not None

        # User 2 should not see the node
        assert store2.get_node("test-node") is None

    def test_clear_memory_store(self):
        """Test clearing a user's store."""
        clear_all_memory_stores()

        store = get_memory_store("user-1")
        node = Node(id="test-node", label="Test", type="Test", properties={})
        store.add_node(node, "transform-1")

        assert store.get_node("test-node") is not None

        clear_memory_store("user-1")

        # Get fresh store reference
        store = get_memory_store("user-1")
        assert store.get_node("test-node") is None

    def test_clear_all_memory_stores(self):
        """Test clearing all stores."""
        store1 = get_memory_store("user-1")
        store2 = get_memory_store("user-2")

        node1 = Node(id="node-1", label="Test", type="Test", properties={})
        node2 = Node(id="node-2", label="Test", type="Test", properties={})

        store1.add_node(node1, "transform-1")
        store2.add_node(node2, "transform-2")

        clear_all_memory_stores()

        # Both stores should be empty
        new_store1 = get_memory_store("user-1")
        new_store2 = get_memory_store("user-2")

        assert new_store1.get_node("node-1") is None
        assert new_store2.get_node("node-2") is None
