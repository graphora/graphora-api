"""
Unit tests for the Neo4j implementation of the graph storage interface.
"""
import pytest
from unittest.mock import MagicMock, patch
from app.services.storage.interface import Neo4jStorage
from app.services.storage.models import Node, StorageBatchResult, DatabaseError
from app.services.transform.models import RelationshipInstance

# Test fixture for sample nodes
@pytest.fixture
def sample_nodes():
    return [
        Node(
            id="node1",
            label="Entity",
            type="Person",
            properties={
                "name": "Alice",
                "age": 30
            }
        ),
        Node(
            id="node2",
            label="Entity",
            type="Person",
            properties={
                "name": "Bob",
                "age": 25
            }
        )
    ]

# Test fixture for sample relationships
@pytest.fixture
def sample_relationships():
    return [
        RelationshipInstance(
            id="rel1",
            source_id="node1",
            source_type="Person",  
            target_id="node2",
            target_type="Person",  
            type="KNOWS",
            properties={
                "since": 2020
            }
        )
    ]

class TestNeo4jStorage:
    """Tests for the Neo4jStorage implementation using mocks"""
    
    @patch('app.services.storage.interface.GraphDatabase')  
    def test_init_connection(self, mock_graph_db):
        """Test initialization and connection"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Act
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password",
            database="neo4j"
        )
        
        # Assert
        mock_graph_db.driver.assert_called_once_with(
            "bolt://localhost:7687", 
            auth=("neo4j", "password")
        )
        mock_driver.session.assert_called()
        assert storage.database == "neo4j"
    
    def test_init_connection_failure(self, mock_graph_db):
        """Test handling of connection failures"""
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
    async def test_store_nodes(self, mock_graph_db, sample_nodes):
        """Test storing nodes"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_summary = MagicMock()
        mock_counters = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.consume.return_value = mock_summary
        mock_summary.counters = mock_counters
        mock_counters.nodes_created = 2
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Reset mock to clear connection check call
        mock_session.run.reset_mock()
        
        # Act
        result = await storage.store_nodes(sample_nodes, 0, "test_transform")
        
        # Assert
        assert result.success is True
        assert result.items_processed == 2
        assert result.batch_index == 0
        mock_session.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_nodes_error(self, mock_graph_db, sample_nodes):
        """Test error handling when storing nodes"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Database error")
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        result = await storage.store_nodes(sample_nodes, 0, "test_transform")
        
        # Assert
        assert result.success is False
        assert "Database error" in result.error
    
    @pytest.mark.asyncio
    async def test_store_relationships(self, mock_graph_db, sample_relationships):
        """Test storing relationships"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_summary = MagicMock()
        mock_counters = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.consume.return_value = mock_summary
        mock_summary.counters = mock_counters
        mock_counters.relationships_created = 1
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Reset mock to clear connection check call
        mock_session.run.reset_mock()
        
        # Act
        result = await storage.store_relationships(sample_relationships, 0, "test_transform")
        
        # Assert
        assert result.success is True
        assert result.items_processed == 1
        assert result.batch_index == 0
        mock_session.run.assert_called_once_with("MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id}) CREATE (source)-[:KNOWS {since: $properties.since, transform_id: $transform_id}]->(target)", source_id="node1", target_id="node2", properties={"since": 2020}, transform_id="test_transform")
    
    @pytest.mark.asyncio
    async def test_store_relationships_error(self, mock_graph_db, sample_relationships):
        """Test error handling when storing relationships"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Database error")
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        result = await storage.store_relationships(sample_relationships, 0, "test_transform")
        
        # Assert
        assert result.success is False
        assert "Database error" in result.error
    
    @pytest.mark.asyncio
    async def test_get_storage_status(self, mock_graph_db):
        """Test retrieving storage status"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        # Mock checkpoint data
        mock_record.get.side_effect = lambda key: {
            "transform_id": "test_transform",
            "last_processed_index": 10,
            "stage": "nodes",
            "timestamp": datetime.now(),
            "error": None,
            "metadata": None
        }.get(key)
        
        mock_result.__iter__.return_value = [mock_record]
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        checkpoint = await storage.get_storage_status("test_transform")
        
        # Assert
        assert checkpoint is not None
        assert checkpoint.transform_id == "test_transform"
        assert checkpoint.last_processed_index == 10
        assert checkpoint.stage == StorageStage.NODES
        mock_session.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_storage_status_not_found(self, mock_graph_db):
        """Test retrieving non-existent storage status"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        # Empty result
        mock_result.__iter__.return_value = []
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        checkpoint = await storage.get_storage_status("test_transform")
        
        # Assert
        assert checkpoint is None
        mock_session.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_checkpoint(self, mock_graph_db):
        """Test updating checkpoint"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        await storage.update_checkpoint(
            "test_transform", 
            last_index=10, 
            stage=StorageStage.NODES
        )
        
        # Assert
        mock_session.run.assert_called_once()
        query = mock_session.run.call_args[0][0]
        assert "MERGE" in query
        assert "SET" in query
    
    @pytest.mark.asyncio
    async def test_get_transformation_data(self, mock_graph_db):
        """Test retrieving transformation data"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_node_result = MagicMock()
        mock_rel_result = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Mock session to return different results for different queries
        def mock_run(query, **params):
            if "MATCH (n)" in query:
                return mock_node_result
            elif "MATCH (source)-[r]->(target)" in query:
                return mock_rel_result
            return MagicMock()
        
        mock_session.run.side_effect = mock_run
        
        # Mock node data
        mock_node_record = MagicMock()
        mock_node = MagicMock()
        mock_node.id = "node1"
        mock_node.labels = ["Person"]
        mock_node.items.return_value = [
            ("name", "Alice"), 
            ("transform_id", "test_transform")
        ]
        mock_node_record.get.return_value = mock_node
        mock_node_result.__iter__.return_value = [mock_node_record]
        
        # Mock relationship data
        mock_rel_record = MagicMock()
        mock_rel = MagicMock()
        mock_rel.id = "rel1"
        mock_rel.type = "KNOWS"
        mock_rel.start_node.id = "node1"
        mock_rel.end_node.id = "node2"
        mock_rel.items.return_value = [
            ("since", 2020), 
            ("transform_id", "test_transform")
        ]
        mock_rel_record.get.return_value = mock_rel
        mock_rel_result.__iter__.return_value = [mock_rel_record]
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        result = await storage.get_transformation_data("test_transform")
        
        # Assert
        assert result.transform_id == "test_transform"
        assert len(result.nodes) == 1
        assert len(result.relationships) == 1
        mock_session.run.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_nodes_by_property(self, mock_graph_db):
        """Test retrieving nodes by property"""
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
        mock_node.id = "node1"
        mock_node.labels = ["Person"]
        mock_node.items.return_value = [
            ("name", "Alice"), 
            ("transform_id", "test_transform")
        ]
        
        mock_record.get.return_value = mock_node
        mock_result.__iter__.return_value = [mock_record]
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        nodes = await storage.get_nodes_by_property("transform_id", "test_transform")
        
        # Assert
        assert len(nodes) == 1
        assert nodes[0].id == "node1"
        assert nodes[0].properties["name"] == "Alice"
        mock_session.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_relationships_between(self, mock_graph_db):
        """Test retrieving relationships between nodes"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        # Mock relationship data
        mock_rel = MagicMock()
        mock_rel.id = "rel1"
        mock_rel.type = "KNOWS"
        mock_rel.start_node.id = "node1"
        mock_rel.end_node.id = "node2"
        mock_rel.items.return_value = [
            ("since", 2020)
        ]
        
        mock_record.get.return_value = mock_rel
        mock_result.__iter__.return_value = [mock_record]
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        relationships = await storage.get_relationships_between("node1", "node2", "KNOWS")
        
        # Assert
        assert len(relationships) == 1
        assert relationships[0].id == "rel1"
        assert relationships[0].type == "KNOWS"
        assert relationships[0].source == "node1"
        assert relationships[0].target == "node2"
        mock_session.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_relationships_between_nodes(self, mock_graph_db):
        """Test retrieving relationships between a set of nodes"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        # Mock relationship data
        mock_rel = MagicMock()
        mock_rel.id = "rel1"
        mock_rel.type = "KNOWS"
        mock_rel.start_node.id = "node1"
        mock_rel.end_node.id = "node2"
        mock_rel.items.return_value = [
            ("since", 2020)
        ]
        
        mock_record.get.return_value = mock_rel
        mock_result.__iter__.return_value = [mock_record]
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        relationships = await storage.get_relationships_between_nodes(["node1", "node2", "node3"])
        
        # Assert
        assert len(relationships) == 1
        assert relationships[0].id == "rel1"
        mock_session.run.assert_called_once()
    
    def test_close(self, mock_graph_db):
        """Test closing the connection"""
        # Arrange
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        storage.close()
        
        # Assert
        mock_driver.close.assert_called_once()
    
    def test_build_node_query(self, mock_graph_db, sample_nodes):
        """Test building Cypher query for node creation"""
        # Arrange
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        query, params = storage._build_node_query(sample_nodes[0], "test_transform")
        
        # Assert
        assert "CREATE" in query
        assert "Person" in query
        assert "properties" in params
        assert params["node_id"] == "node1"
        assert params["transform_id"] == "test_transform"
        assert params["properties"]["name"] == "Alice"
    
    def test_build_relationship_query(self, mock_graph_db, sample_relationships):
        """Test building Cypher query for relationship creation"""
        # Arrange
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        query, params = storage._build_relationship_query(sample_relationships[0], "test_transform")
        
        # Assert
        assert "MATCH" in query
        assert "CREATE" in query
        assert "KNOWS" in query
        assert params["source_id"] == "node1"
        assert params["target_id"] == "node2"
        assert params["transform_id"] == "test_transform"
        assert params["properties"]["since"] == 2020
