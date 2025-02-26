"""
Unit tests for the graph storage interface and implementations.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional

from app.services.storage.interface import GraphStorageInterface
from app.services.storage.models import (
    StorageBatchResult,
    StorageCheckpoint,
    StorageStage,
    DatabaseError,
    TransformationResult,
    Node,
    Edge
)

# Test fixture for mock Neo4j driver
@pytest.fixture
def mock_neo4j_driver():
    driver = MagicMock()
    session = MagicMock()
    result = MagicMock()
    
    # Configure mocks
    driver.session.return_value.__enter__.return_value = session
    session.run.return_value = result
    
    return driver

# Mock implementation for testing
class MockGraphStorage(GraphStorageInterface):
    """Mock implementation of GraphStorageInterface for testing"""
    
    def __init__(self):
        self.nodes = []
        self.relationships = []
        self.checkpoints = {}
        self.transform_data = {}
    
    def add_test_node(self, node: Node):
        self.nodes.append(node)
    
    def add_test_relationship(self, edge: Edge):
        self.relationships.append(edge)
    
    async def store_nodes(
        self,
        nodes: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store nodes in batch"""
        try:
            for node_data in nodes:
                node = Node(
                    id=node_data.get("id", str(uuid.uuid4())),
                    label=node_data.get("label", "TestNode"),
                    type=node_data.get("type", "TestNode"),
                    properties={
                        **node_data.get("properties", {}),
                        "transform_id": transform_id
                    }
                )
                self.nodes.append(node)
            
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=len(nodes),
                processing_time_ms=10.0,
                success=True
            )
        except Exception as e:
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=0,
                processing_time_ms=10.0,
                success=False,
                error=str(e)
            )
    
    async def store_relationships(
        self,
        relationships: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store relationships in batch"""
        try:
            for rel_data in relationships:
                edge = Edge(
                    id=rel_data.get("id", str(uuid.uuid4())),
                    source=rel_data.get("source"),
                    target=rel_data.get("target"),
                    type=rel_data.get("type", "TEST_REL"),
                    properties={
                        **rel_data.get("properties", {}),
                        "transform_id": transform_id
                    }
                )
                self.relationships.append(edge)
            
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=len(relationships),
                processing_time_ms=10.0,
                success=True
            )
        except Exception as e:
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=0,
                processing_time_ms=10.0,
                success=False,
                error=str(e)
            )
    
    async def get_storage_status(
        self,
        transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status"""
        return self.checkpoints.get(transform_id)
    
    async def update_checkpoint(
        self,
        transform_id: str,
        last_index: int,
        stage: StorageStage
    ) -> None:
        """Update storage checkpoint"""
        self.checkpoints[transform_id] = StorageCheckpoint(
            transform_id=transform_id,
            last_processed_index=last_index,
            stage=stage,
            timestamp=datetime.now()
        )
    
    async def get_transformation_data(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships for a transformation"""
        if transform_id in self.transform_data:
            return self.transform_data[transform_id]
        
        nodes = [
            node.dict() for node in self.nodes 
            if node.properties.get("transform_id") == transform_id
        ]
        
        relationships = [
            rel.dict() for rel in self.relationships 
            if rel.properties.get("transform_id") == transform_id
        ]
        
        result = TransformationResult(
            transform_id=transform_id,
            nodes=nodes,
            relationships=relationships,
            timestamp=datetime.now()
        )
        
        self.transform_data[transform_id] = result
        return result
    
    async def get_nodes_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Node]:
        """Get all nodes with the specified property value"""
        return [
            node for node in self.nodes 
            if node.properties.get(property_name) == property_value
        ]
    
    async def get_relationships_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Edge]:
        """Get all relationships between two nodes"""
        filtered = [
            rel for rel in self.relationships 
            if rel.source == source_id and rel.target == target_id
        ]
        
        if relationship_type:
            filtered = [rel for rel in filtered if rel.type == relationship_type]
            
        return filtered
    
    async def get_relationships_between_nodes(
        self,
        node_ids: List[str]
    ) -> List[Edge]:
        """Get all relationships between a set of nodes"""
        return [
            rel for rel in self.relationships 
            if rel.source in node_ids and rel.target in node_ids
        ]
        
    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True
    ) -> List[Node]:
        """Find nodes with the specified label and property value"""
        filtered = [
            node for node in self.nodes 
            if node.label == label
        ]
        
        if exact_match:
            return [
                node for node in filtered
                if node.properties.get(property_name) == property_value
            ]
        else:
            # Simple case-insensitive contains for string values
            if isinstance(property_value, str):
                return [
                    node for node in filtered
                    if isinstance(node.properties.get(property_name), str) and
                    property_value.lower() in node.properties.get(property_name, "").lower()
                ]
            return []
    
    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True
    ) -> List[Dict[str, Any]]:
        """Find nodes with similar properties using fuzzy matching"""
        # Simple mock implementation - just return nodes with the same label
        # that match at least one property exactly
        filtered = [
            node for node in self.nodes 
            if node.label == label
        ]
        
        results = []
        for node in filtered:
            # Check if any property matches
            for prop_name, prop_value in properties.items():
                if node.properties.get(prop_name) == prop_value:
                    node_dict = node.dict()
                    node_dict["similarity_score"] = 1.0  # Mock score
                    
                    if include_relationships:
                        # Add related edges
                        node_dict["relationships"] = [
                            rel.dict() for rel in self.relationships
                            if rel.source == node.id or rel.target == node.id
                        ]
                    
                    results.append(node_dict)
                    break
        
        return results[:max_results]
    
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Create a new node"""
        node_id = properties.get("id", str(uuid.uuid4()))
        node = Node(
            id=node_id,
            label=label,
            type=label,
            properties=properties
        )
        self.nodes.append(node)
        return node
    
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> Optional[Node]:
        """Update an existing node"""
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                # Update properties
                updated_props = {**node.properties, **properties}
                updated_node = Node(
                    id=node_id,
                    label=node.label,
                    type=node.type,
                    properties=updated_props
                )
                self.nodes[i] = updated_node
                return updated_node
        return None
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> Optional[Edge]:
        """Create a relationship between nodes"""
        if properties is None:
            properties = {}
        
        # Check if source and target nodes exist
        source_exists = any(node.id == source_id for node in self.nodes)
        target_exists = any(node.id == target_id for node in self.nodes)
        
        if not source_exists or not target_exists:
            return None
        
        edge_id = properties.get("id", str(uuid.uuid4()))
        edge = Edge(
            id=edge_id,
            source=source_id,
            target=target_id,
            type=rel_type,
            properties=properties or {}
        )
        self.relationships.append(edge)
        return edge
    
    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    async def get_edges_between(
        self,
        source_id: str,
        target_id: str
    ) -> List[Edge]:
        """Get all edges between two nodes"""
        return [
            rel for rel in self.relationships 
            if rel.source == source_id and rel.target == target_id
        ]


# Test for mock implementation
class TestMockGraphStorage:
    """Tests for the MockGraphStorage implementation"""
    
    @pytest.mark.asyncio
    async def test_store_nodes(self):
        """Test storing nodes"""
        # Arrange
        storage = MockGraphStorage()
        nodes = [
            {"id": "1", "label": "Person", "properties": {"name": "Alice"}},
            {"id": "2", "label": "Person", "properties": {"name": "Bob"}}
        ]
        
        # Act
        result = await storage.store_nodes(nodes, 0, "test_transform")
        
        # Assert
        assert result.success is True
        assert result.items_processed == 2
        assert len(storage.nodes) == 2
        assert storage.nodes[0].id == "1"
        assert storage.nodes[0].properties["name"] == "Alice"
        assert storage.nodes[0].properties["transform_id"] == "test_transform"
    
    @pytest.mark.asyncio
    async def test_store_relationships(self):
        """Test storing relationships"""
        # Arrange
        storage = MockGraphStorage()
        relationships = [
            {
                "source": "1", 
                "target": "2", 
                "type": "KNOWS", 
                "properties": {"since": 2020}
            }
        ]
        
        # Act
        result = await storage.store_relationships(relationships, 0, "test_transform")
        
        # Assert
        assert result.success is True
        assert result.items_processed == 1
        assert len(storage.relationships) == 1
        assert storage.relationships[0].source == "1"
        assert storage.relationships[0].target == "2"
        assert storage.relationships[0].type == "KNOWS"
        assert storage.relationships[0].properties["since"] == 2020
        assert storage.relationships[0].properties["transform_id"] == "test_transform"
    
    @pytest.mark.asyncio
    async def test_get_nodes_by_property(self):
        """Test retrieving nodes by property"""
        # Arrange
        storage = MockGraphStorage()
        storage.add_test_node(Node(
            id="1", 
            label="Person", 
            type="Person",
            properties={"name": "Alice", "transform_id": "test_id"}
        ))
        storage.add_test_node(Node(
            id="2", 
            label="Person", 
            type="Person",
            properties={"name": "Bob", "transform_id": "test_id"}
        ))
        
        # Act
        nodes = await storage.get_nodes_by_property("transform_id", "test_id")
        
        # Assert
        assert len(nodes) == 2
        assert nodes[0].id == "1"
        assert nodes[1].id == "2"
        
        # Test filtering by another property
        nodes = await storage.get_nodes_by_property("name", "Alice")
        assert len(nodes) == 1
        assert nodes[0].id == "1"
    
    @pytest.mark.asyncio
    async def test_get_relationships_between(self):
        """Test retrieving relationships between nodes"""
        # Arrange
        storage = MockGraphStorage()
        storage.add_test_relationship(Edge(
            id="rel1",
            source="1",
            target="2",
            type="KNOWS",
            properties={"since": 2020}
        ))
        storage.add_test_relationship(Edge(
            id="rel2",
            source="1",
            target="2",
            type="WORKS_WITH",
            properties={"since": 2021}
        ))
        
        # Act - Get all relationships
        relationships = await storage.get_relationships_between("1", "2")
        
        # Assert
        assert len(relationships) == 2
        
        # Act - Filter by type
        relationships = await storage.get_relationships_between("1", "2", "KNOWS")
        
        # Assert
        assert len(relationships) == 1
        assert relationships[0].id == "rel1"
        assert relationships[0].type == "KNOWS"
    
    @pytest.mark.asyncio
    async def test_get_relationships_between_nodes(self):
        """Test retrieving relationships between a set of nodes"""
        # Arrange
        storage = MockGraphStorage()
        storage.add_test_relationship(Edge(
            id="rel1",
            source="1",
            target="2",
            type="KNOWS",
            properties={"since": 2020}
        ))
        storage.add_test_relationship(Edge(
            id="rel2",
            source="2",
            target="3",
            type="KNOWS",
            properties={"since": 2021}
        ))
        storage.add_test_relationship(Edge(
            id="rel3",
            source="3",
            target="4",
            type="KNOWS",
            properties={"since": 2022}
        ))
        
        # Act
        relationships = await storage.get_relationships_between_nodes(["1", "2", "3"])
        
        # Assert
        assert len(relationships) == 2
        assert "rel1" in [rel.id for rel in relationships]
        assert "rel2" in [rel.id for rel in relationships]
        assert "rel3" not in [rel.id for rel in relationships]
    
    @pytest.mark.asyncio
    async def test_update_and_get_checkpoint(self):
        """Test updating and retrieving checkpoints"""
        # Arrange
        storage = MockGraphStorage()
        transform_id = "test_transform"
        
        # Act - Initially no checkpoint
        checkpoint = await storage.get_storage_status(transform_id)
        
        # Assert
        assert checkpoint is None
        
        # Act - Update checkpoint
        await storage.update_checkpoint(
            transform_id, 
            last_index=10, 
            stage=StorageStage.NODES
        )
        
        # Get updated checkpoint
        checkpoint = await storage.get_storage_status(transform_id)
        
        # Assert
        assert checkpoint is not None
        assert checkpoint.transform_id == transform_id
        assert checkpoint.last_processed_index == 10
        assert checkpoint.stage == StorageStage.NODES
    
    @pytest.mark.asyncio
    async def test_get_transformation_data(self):
        """Test retrieving transformation data"""
        # Arrange
        storage = MockGraphStorage()
        transform_id = "test_transform"
        
        # Add some test data
        storage.add_test_node(Node(
            id="1", 
            label="Person", 
            type="Person",
            properties={"name": "Alice", "transform_id": transform_id}
        ))
        storage.add_test_relationship(Edge(
            id="rel1",
            source="1",
            target="2",
            type="KNOWS",
            properties={"since": 2020, "transform_id": transform_id}
        ))
        
        # Act
        result = await storage.get_transformation_data(transform_id)
        
        # Assert
        assert result.transform_id == transform_id
        assert len(result.nodes) == 1
        assert len(result.relationships) == 1
        assert result.nodes[0]["id"] == "1"
        assert result.relationships[0]["id"] == "rel1"


# Test Neo4j implementation with mocks
@patch('app.services.storage.interface.GraphDatabase')
class TestNeo4jStorage:
    """Tests for the Neo4jStorage implementation using mocks"""
    
    def test_init_connection(self, mock_graph_db):
        """Test initialization and connection"""
        from app.services.storage.interface import Neo4jStorage
        
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Act
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Assert
        mock_graph_db.driver.assert_called_once_with(
            "bolt://localhost:7687", 
            auth=("neo4j", "password")
        )
        mock_driver.session.assert_called()
    
    def test_init_connection_failure(self, mock_graph_db):
        """Test handling of connection failures"""
        from app.services.storage.interface import Neo4jStorage
        
        # Arrange
        mock_graph_db.driver.side_effect = Exception("Connection failed")
        
        # Act & Assert
        with pytest.raises(DatabaseError):
            Neo4jStorage(
                uri="bolt://localhost:7687", 
                username="neo4j", 
                password="password"
            )
    
    @pytest.mark.asyncio
    async def test_get_nodes_by_property(self, mock_graph_db):
        """Test retrieving nodes by property"""
        from app.services.storage.interface import Neo4jStorage
        
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        # Mock node data
        mock_node = MagicMock()
        mock_node.id = "1"
        mock_node.labels = ["Person"]  # Ensure labels is a non-empty list
        mock_node.items.return_value = [
            ("name", "Alice"), 
            ("transform_id", "test_id")
        ]
        
        # Configure the mock record to properly return the mock node
        mock_record.__getitem__.return_value = mock_node
        mock_result.__iter__.return_value = [mock_record]
        
        # Mock the _node_from_record method to avoid Neo4j specific implementation details
        with patch.object(Neo4jStorage, '_node_from_record', return_value=Node(
            id="1",
            label="Person",
            type="Person",
            properties={"name": "Alice", "transform_id": "test_id"}
        )):
            storage = Neo4jStorage(
                uri="bolt://localhost:7687", 
                username="neo4j", 
                password="password"
            )
            
            # Reset the mock to clear the connection check call
            mock_session.run.reset_mock()
            
            # Act
            nodes = await storage.get_nodes_by_property("transform_id", "test_id")
            
            # Assert
            assert len(nodes) == 1
            assert nodes[0].id == "1"
            assert nodes[0].properties["name"] == "Alice"
            mock_session.run.assert_called_once()
            query = mock_session.run.call_args[0][0]
            assert "MATCH (n)" in query
            assert "WHERE n.transform_id = $value" in query
