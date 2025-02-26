"""
Unit tests for edge cases and error handling in the storage layer.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional

from app.services.storage.interface import GraphStorageInterface
from app.services.storage.neo4j import Neo4jStorage
from app.services.storage.models import (
    StorageBatchResult,
    StorageCheckpoint,
    StorageStage,
    DatabaseError,
    TransformationResult,
    Node,
    Edge,
    StorageError,
    CheckpointError
)
from app.services.transform.models import BaseNode, RelationshipInstance

# Create a test implementation that simulates various error conditions
class ErrorProneStorage(GraphStorageInterface):
    """Mock implementation that simulates various error conditions"""
    
    def __init__(self, error_mode=None):
        self.error_mode = error_mode
        self.nodes = []
        self.relationships = []
        self.checkpoints = {}
    
    async def store_nodes(
        self,
        nodes: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store nodes with simulated errors"""
        if self.error_mode == "connection_error":
            raise DatabaseError("Connection to database failed")
        elif self.error_mode == "timeout":
            raise DatabaseError("Database operation timed out")
        elif self.error_mode == "partial_failure" and batch_index > 0:
            # Simulate partial success
            processed = len(nodes) // 2
            for i in range(processed):
                node_data = nodes[i]
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
                items_processed=processed,
                processing_time_ms=10.0,
                success=False,
                error="Partial batch failure",
                warnings=["Some nodes could not be processed"]
            )
        
        # Normal operation
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
    
    async def store_relationships(
        self,
        relationships: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store relationships with simulated errors"""
        if self.error_mode == "connection_error":
            raise DatabaseError("Connection to database failed")
        elif self.error_mode == "timeout":
            raise DatabaseError("Database operation timed out")
        elif self.error_mode == "missing_nodes":
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=0,
                processing_time_ms=10.0,
                success=False,
                error="Referenced nodes not found",
                warnings=["Some relationships reference non-existent nodes"]
            )
        
        # Normal operation
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
    
    async def get_storage_status(
        self,
        transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status with simulated errors"""
        if self.error_mode == "connection_error":
            raise DatabaseError("Connection to database failed")
        elif self.error_mode == "checkpoint_error":
            raise CheckpointError("Failed to retrieve checkpoint")
        
        return self.checkpoints.get(transform_id)
    
    async def update_checkpoint(
        self,
        transform_id: str,
        last_index: int,
        stage: StorageStage
    ) -> None:
        """Update storage checkpoint with simulated errors"""
        if self.error_mode == "connection_error":
            raise DatabaseError("Connection to database failed")
        elif self.error_mode == "checkpoint_error":
            raise CheckpointError("Failed to update checkpoint")
        
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
        """Get transformation data with simulated errors"""
        if self.error_mode == "connection_error":
            raise DatabaseError("Connection to database failed")
        
        nodes = [
            node.dict() for node in self.nodes 
            if node.properties.get("transform_id") == transform_id
        ]
        
        relationships = [
            rel.dict() for rel in self.relationships 
            if rel.properties.get("transform_id") == transform_id
        ]
        
        return TransformationResult(
            transform_id=transform_id,
            nodes=nodes,
            relationships=relationships,
            timestamp=datetime.now()
        )
    
    async def get_nodes_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Node]:
        """Get nodes by property with simulated errors"""
        if self.error_mode == "connection_error":
            raise DatabaseError("Connection to database failed")
        
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
        """Get relationships between nodes with simulated errors"""
        if self.error_mode == "connection_error":
            raise DatabaseError("Connection to database failed")
        
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
        """Get relationships between a set of nodes with simulated errors"""
        if self.error_mode == "connection_error":
            raise DatabaseError("Connection to database failed")
        
        return [
            rel for rel in self.relationships 
            if rel.source in node_ids and rel.target in node_ids
        ]


class TestStorageEdgeCases:
    """Tests for edge cases and error handling in the storage layer"""
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of database connection errors"""
        # Arrange
        storage = ErrorProneStorage(error_mode="connection_error")
        
        # Act & Assert - store_nodes
        with pytest.raises(DatabaseError):
            await storage.store_nodes(
                [{"id": "test", "properties": {}}], 
                0, 
                "test_transform"
            )
        
        # Act & Assert - store_relationships
        with pytest.raises(DatabaseError):
            await storage.store_relationships(
                [{"source": "1", "target": "2", "type": "TEST"}], 
                0, 
                "test_transform"
            )
        
        # Act & Assert - get_storage_status
        with pytest.raises(DatabaseError):
            await storage.get_storage_status("test_transform")
    
    @pytest.mark.asyncio
    async def test_partial_batch_failure(self):
        """Test handling of partial batch failures"""
        # Arrange
        storage = ErrorProneStorage(error_mode="partial_failure")
        
        # Act - First batch succeeds
        result1 = await storage.store_nodes(
            [
                {"id": "1", "properties": {"name": "Node1"}},
                {"id": "2", "properties": {"name": "Node2"}}
            ], 
            0, 
            "test_transform"
        )
        
        # Act - Second batch partially fails
        result2 = await storage.store_nodes(
            [
                {"id": "3", "properties": {"name": "Node3"}},
                {"id": "4", "properties": {"name": "Node4"}}
            ], 
            1, 
            "test_transform"
        )
        
        # Assert
        assert result1.success is True
        assert result1.items_processed == 2
        
        assert result2.success is False
        assert result2.items_processed == 1
        assert "Partial batch failure" in result2.error
        assert len(result2.warnings) > 0
        
        # Verify the state
        nodes = await storage.get_nodes_by_property("transform_id", "test_transform")
        assert len(nodes) == 3  # 2 from first batch + 1 from second batch
    
    @pytest.mark.asyncio
    async def test_missing_nodes_for_relationships(self):
        """Test handling of relationships referencing non-existent nodes"""
        # Arrange
        storage = ErrorProneStorage(error_mode="missing_nodes")
        
        # Act
        result = await storage.store_relationships(
            [
                {"source": "nonexistent1", "target": "nonexistent2", "type": "TEST"}
            ], 
            0, 
            "test_transform"
        )
        
        # Assert
        assert result.success is False
        assert "Referenced nodes not found" in result.error
        assert len(result.warnings) > 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_error_handling(self):
        """Test handling of checkpoint errors"""
        # Arrange
        storage = ErrorProneStorage(error_mode="checkpoint_error")
        
        # Act & Assert - get_storage_status
        with pytest.raises(CheckpointError):
            await storage.get_storage_status("test_transform")
        
        # Act & Assert - update_checkpoint
        with pytest.raises(CheckpointError):
            await storage.update_checkpoint(
                "test_transform", 
                10, 
                StorageStage.NODES
            )


@patch('app.services.storage.neo4j.GraphDatabase')
class TestNeo4jStorageEdgeCases:
    """Tests for Neo4j-specific edge cases and error handling"""
    
    def test_empty_database_handling(self, mock_graph_db):
        """Test handling of operations on an empty database"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        
        # Empty results
        mock_result.__iter__.return_value = []
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act & Assert - get_transformation_data on empty database
        async def test_empty_transformation():
            result = await storage.get_transformation_data("nonexistent")
            assert result.node_count == 0
            assert result.relationship_count == 0
        
        # Run the async test
        import asyncio
        asyncio.run(test_empty_transformation())
    
    def test_large_property_values(self, mock_graph_db):
        """Test handling of large property values"""
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
        
        # Create a node with a very large property value
        large_text = "x" * 1000000  # 1MB string
        node = BaseNode(
            id="large_node",
            type="LargeNode",
            properties={
                "large_text": large_text
            }
        )
        
        # Act & Assert - Should handle large properties without error
        async def test_large_property():
            # This should not raise an exception
            query, params = storage._build_node_query(node, "test_transform")
            assert "large_text" in params["properties"]
            assert len(params["properties"]["large_text"]) == 1000000
        
        # Run the async test
        import asyncio
        asyncio.run(test_large_property())
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, mock_graph_db):
        """Test transaction rollback on error"""
        # Arrange
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Simulate an error during transaction
        mock_session.run.side_effect = Exception("Transaction error")
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687", 
            username="neo4j", 
            password="password"
        )
        
        # Act
        result = await storage.store_nodes(
            [BaseNode(id="test", type="Test", properties={})], 
            0, 
            "test_transform"
        )
        
        # Assert
        assert result.success is False
        assert "Transaction error" in result.error
    
    @pytest.mark.asyncio
    async def test_invalid_node_id(self, mock_graph_db):
        """Test handling of invalid node IDs"""
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
        
        # Node with invalid ID (empty string)
        invalid_node = BaseNode(
            id="",
            type="InvalidNode",
            properties={}
        )
        
        # Act
        result = await storage.store_nodes([invalid_node], 0, "test_transform")
        
        # Assert - Should fail with appropriate error
        assert result.success is False
        assert "id" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_special_characters_in_properties(self, mock_graph_db):
        """Test handling of special characters in property values"""
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
        
        # Node with special characters in properties
        special_node = BaseNode(
            id="special_node",
            type="SpecialNode",
            properties={
                "special_chars": "!@#$%^&*()_+{}[]|\\:;\"'<>,.?/",
                "cypher_injection": "MATCH (n) DETACH DELETE n",
                "nested_quotes": "He said \"Don't do that\""
            }
        )
        
        # Act - This should not raise an exception
        query, params = storage._build_node_query(special_node, "test_transform")
        
        # Assert - Parameters should be properly escaped
        assert "special_chars" in params["properties"]
        assert params["properties"]["special_chars"] == "!@#$%^&*()_+{}[]|\\:;\"'<>,.?/"
        assert params["properties"]["cypher_injection"] == "MATCH (n) DETACH DELETE n"
        assert params["properties"]["nested_quotes"] == "He said \"Don't do that\""
