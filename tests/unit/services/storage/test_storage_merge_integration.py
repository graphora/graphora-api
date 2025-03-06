"""
Unit tests for integration between storage layer and merge functionality.
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
from app.services.merge.service import MergeService
from tests.unit.services.storage.test_graph_storage import MockGraphStorage

# Test fixture for a populated mock storage
@pytest.fixture
def populated_mock_storage():
    """Create a mock storage with pre-populated data for testing"""
    storage = MockGraphStorage()
    
    # Add nodes to production graph
    storage.add_test_node(Node(
        id="prod1",
        label="Person",
        type="Person",
        properties={
            "name": "Alice",
            "age": 30,
            "transform_id": "prod_transform",
            "environment": "production"
        }
    ))
    storage.add_test_node(Node(
        id="prod2",
        label="Person",
        type="Person",
        properties={
            "name": "Bob",
            "age": 25,
            "transform_id": "prod_transform",
            "environment": "production"
        }
    ))
    storage.add_test_node(Node(
        id="prod3",
        label="Company",
        type="Company",
        properties={
            "name": "Acme Inc",
            "founded": 2010,
            "transform_id": "prod_transform",
            "environment": "production"
        }
    ))
    
    # Add relationships to production graph
    storage.add_test_relationship(Edge(
        id="prod_rel1",
        source="prod1",
        target="prod2",
        type="KNOWS",
        properties={
            "since": 2020,
            "transform_id": "prod_transform",
            "environment": "production"
        }
    ))
    storage.add_test_relationship(Edge(
        id="prod_rel2",
        source="prod1",
        target="prod3",
        type="WORKS_AT",
        properties={
            "position": "Engineer",
            "since": 2019,
            "transform_id": "prod_transform",
            "environment": "production"
        }
    ))
    
    # Add nodes to staging graph
    storage.add_test_node(Node(
        id="staging1",
        label="Person",
        type="Person",
        properties={
            "name": "Alice",  # Same name as in production
            "age": 31,  # Different age - conflict
            "transform_id": "staging_transform",
            "environment": "staging"
        }
    ))
    storage.add_test_node(Node(
        id="staging2",
        label="Person",
        type="Person",
        properties={
            "name": "Charlie",  # New person
            "age": 35,
            "transform_id": "staging_transform",
            "environment": "staging"
        }
    ))
    storage.add_test_node(Node(
        id="staging3",
        label="Company",
        type="Company",
        properties={
            "name": "Acme Inc",  # Same company
            "founded": 2010,
            "employees": 500,  # New property - no conflict
            "transform_id": "staging_transform",
            "environment": "staging"
        }
    ))
    
    # Add relationships to staging graph
    storage.add_test_relationship(Edge(
        id="staging_rel1",
        source="staging1",
        target="staging2",
        type="KNOWS",
        properties={
            "since": 2022,
            "transform_id": "staging_transform",
            "environment": "staging"
        }
    ))
    storage.add_test_relationship(Edge(
        id="staging_rel2",
        source="staging2",
        target="staging3",
        type="WORKS_AT",
        properties={
            "position": "Designer",
            "since": 2021,
            "transform_id": "staging_transform",
            "environment": "staging"
        }
    ))
    
    return storage


class TestStorageMergeIntegration:
    """Tests for integration between storage layer and merge functionality"""
    
    @pytest.mark.asyncio
    async def test_storage_provides_correct_data_for_merge(self, populated_mock_storage):
        """Test that storage provides the correct data for merge operations"""
        # Arrange
        storage = populated_mock_storage
        
        # Act - Get production data
        prod_data = await storage.get_transformation_data("prod_transform")
        
        # Act - Get staging data
        staging_data = await storage.get_transformation_data("staging_transform")
        
        # Assert
        assert prod_data.transform_id == "prod_transform"
        assert staging_data.transform_id == "staging_transform"
        
        # Check production data
        assert len(prod_data.nodes) == 3
        assert len(prod_data.relationships) == 2
        
        # Check staging data
        assert len(staging_data.nodes) == 3
        assert len(staging_data.relationships) == 2
        
        # Verify specific nodes exist
        prod_node_ids = [node["id"] for node in prod_data.nodes]
        staging_node_ids = [node["id"] for node in staging_data.nodes]
        
        assert "prod1" in prod_node_ids
        assert "prod2" in prod_node_ids
        assert "prod3" in prod_node_ids
        
        assert "staging1" in staging_node_ids
        assert "staging2" in staging_node_ids
        assert "staging3" in staging_node_ids
    
    @pytest.mark.asyncio
    async def test_merge_service_can_use_storage(self):
        """Test that merge service can use storage instances"""
        # Create mock storage instances
        staging_storage = MockGraphStorage()
        prod_storage = MockGraphStorage()
        
        # Create a mock progress tracker
        mock_progress_tracker = AsyncMock()
        mock_progress_tracker.initialize_merge = AsyncMock()
        mock_progress_tracker.start_merge_stage = AsyncMock()
        mock_progress_tracker.update_merge_progress = AsyncMock()
        mock_progress_tracker.complete_merge_stage = AsyncMock()
        
        # Create service
        service = MergeService(
            storage=staging_storage, 
            production_storage=prod_storage,
            progress_tracker=mock_progress_tracker
        )
        
        # Add test data
        staging_storage.add_test_node(
            Node(id="test1", label="Test", type="Test", properties={"name": "Test Node"})
        )
        
        # Verify service can access storage
        assert service.storage is not None
        assert service.production_storage is not None
        
        # Test storage operations
        nodes = await service.storage.get_nodes_by_property("name", "Test Node")
        assert len(nodes) == 1
        assert nodes[0].id == "test1"
    
    @pytest.mark.asyncio
    async def test_entity_matching_with_storage(self, populated_mock_storage):
        """Test entity matching functionality with storage"""
        # Arrange
        storage = populated_mock_storage
        
        # Act - Find nodes by property
        alice_nodes = await storage.get_nodes_by_property("name", "Alice")
        
        # Assert - Should find both Alice nodes (from prod and staging)
        assert len(alice_nodes) == 2
        
        # Check that we have one from each environment
        environments = [node.properties.get("environment") for node in alice_nodes]
        assert "production" in environments
        assert "staging" in environments
    
    @pytest.mark.asyncio
    async def test_relationship_retrieval_for_merge(self, populated_mock_storage):
        """Test retrieving relationships for merge operations"""
        # Arrange
        storage = populated_mock_storage
        
        # Get node IDs
        alice_prod = "prod1"
        bob_prod = "prod2"
        alice_staging = "staging1"
        charlie_staging = "staging2"
        
        # Act - Get relationships between nodes
        alice_bob_rels = await storage.get_relationships_between(alice_prod, bob_prod)
        alice_charlie_rels = await storage.get_relationships_between(alice_staging, charlie_staging)
        
        # Assert
        assert len(alice_bob_rels) == 1
        assert alice_bob_rels[0].type == "KNOWS"
        assert alice_bob_rels[0].properties.get("since") == 2020
        
        assert len(alice_charlie_rels) == 1
        assert alice_charlie_rels[0].type == "KNOWS"
        assert alice_charlie_rels[0].properties.get("since") == 2022
