"""Integration tests for transaction management"""
import pytest
import asyncio
import os
import uuid
from typing import Dict, Any, List

from app.services.storage.neo4j import Neo4jStorage
from app.services.storage.transaction import Neo4jTransactionManager
from app.services.merge.execution_service import MergeExecutionService
from app.services.merge.progress import ProgressTracker
from app.schemas.graph import GraphResponse, Node, Edge
from app.config import settings

pytestmark = pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Integration tests are skipped by default"
)

class TestTransactionIntegration:
    """Integration tests for transaction management"""
    
    @pytest.fixture(scope="function")
    async def storage(self):
        """Fixture for Neo4j storage"""
        storage = Neo4jStorage(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DB
        )
        async with storage:
            yield storage
    
    @pytest.fixture(scope="function")
    def transaction_manager(self, storage):
        """Fixture for transaction manager"""
        manager = Neo4jTransactionManager(storage.driver)
        storage.transaction_manager = manager
        return manager
    
    @pytest.fixture(scope="function")
    def progress_tracker(self):
        """Fixture for progress tracker"""
        return ProgressTracker()
    
    @pytest.fixture(scope="function")
    def execution_service(self, storage, progress_tracker, transaction_manager):
        """Fixture for execution service"""
        return MergeExecutionService(
            staging_storage=storage,
            prod_storage=storage,
            progress_tracker=progress_tracker,
            transaction_manager=transaction_manager
        )
    
    @pytest.mark.asyncio
    async def test_transaction_commit(self, transaction_manager, storage):
        """Test transaction commit with actual database"""
        # Create test data
        merge_id = f"test_merge_{uuid.uuid4().hex}"
        test_node = {
            "id": f"test_node_{uuid.uuid4().hex}",
            "label": "TestNode",
            "type": "TestType",
            "name": "Test Node",
            "description": "Test node for transaction testing"
        }
        
        # Begin transaction
        transaction_id = await transaction_manager.begin_transaction(merge_id)
        
        try:
            # Create node within transaction
            node = await storage.create_node(
                label=test_node["label"],
                properties={
                    "id": test_node["id"],
                    "type": test_node["type"],
                    "name": test_node["name"],
                    "description": test_node["description"]
                }
            )
            
            # Verify node exists in database
            retrieved_node = await storage.get_node_by_id(test_node["id"])
            assert retrieved_node is not None
            assert retrieved_node.id == test_node["id"]
            
            # Commit transaction
            commit_result = await transaction_manager.commit_transaction(transaction_id)
            assert commit_result is True
            
            # Verify node still exists after commit
            retrieved_node = await storage.get_node_by_id(test_node["id"])
            assert retrieved_node is not None
            assert retrieved_node.id == test_node["id"]
            
            # Clean up
            await storage._execute_query(f"MATCH (n {{id: '{test_node['id']}'}}) DELETE n")
            
        except Exception as e:
            # Rollback on error
            await transaction_manager.rollback_transaction(transaction_id)
            # Clean up
            await storage._execute_query(f"MATCH (n {{id: '{test_node['id']}'}}) DELETE n")
            raise e
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, transaction_manager, storage):
        """Test transaction rollback with actual database"""
        # Create test data
        merge_id = f"test_merge_{uuid.uuid4().hex}"
        test_node = {
            "id": f"test_node_{uuid.uuid4().hex}",
            "label": "TestNode",
            "type": "TestType",
            "name": "Test Node",
            "description": "Test node for transaction testing"
        }
        
        # Begin transaction
        transaction_id = await transaction_manager.begin_transaction(merge_id)
        
        try:
            # Create node within transaction
            node = await storage.create_node(
                label=test_node["label"],
                properties={
                    "id": test_node["id"],
                    "type": test_node["type"],
                    "name": test_node["name"],
                    "description": test_node["description"]
                }
            )
            
            # Verify node exists in database
            retrieved_node = await storage.get_node_by_id(test_node["id"])
            assert retrieved_node is not None
            assert retrieved_node.id == test_node["id"]
            
            # Rollback transaction
            rollback_result = await transaction_manager.rollback_transaction(transaction_id)
            assert rollback_result is True
            
            # Verify node no longer exists after rollback
            retrieved_node = await storage.get_node_by_id(test_node["id"])
            assert retrieved_node is None
            
        except Exception as e:
            # Rollback on error
            await transaction_manager.rollback_transaction(transaction_id)
            # Clean up
            await storage._execute_query(f"MATCH (n {{id: '{test_node['id']}'}}) DELETE n")
            raise e
    
    @pytest.mark.asyncio
    async def test_execute_in_transaction(self, transaction_manager, storage):
        """Test execute_in_transaction with actual database"""
        # Create test data
        merge_id = f"test_merge_{uuid.uuid4().hex}"
        test_node = {
            "id": f"test_node_{uuid.uuid4().hex}",
            "label": "TestNode",
            "type": "TestType",
            "name": "Test Node",
            "description": "Test node for transaction testing"
        }
        
        # Define callback function
        async def create_test_node(merge_id, transaction_id):
            # Create node
            node = await storage.create_node(
                label=test_node["label"],
                properties={
                    "id": test_node["id"],
                    "type": test_node["type"],
                    "name": test_node["name"],
                    "description": test_node["description"]
                }
            )
            
            return {
                "node_id": node.id,
                "transaction_id": transaction_id,
                "merge_id": merge_id
            }
        
        try:
            # Execute callback within transaction
            result = await transaction_manager.execute_in_transaction(
                create_test_node,
                merge_id=merge_id
            )
            
            # Verify result
            assert result["node_id"] == test_node["id"]
            assert result["merge_id"] == merge_id
            assert "transaction_id" in result
            
            # Verify node exists in database
            retrieved_node = await storage.get_node_by_id(test_node["id"])
            assert retrieved_node is not None
            assert retrieved_node.id == test_node["id"]
            
            # Clean up
            await storage._execute_query(f"MATCH (n {{id: '{test_node['id']}'}}) DELETE n")
            
        except Exception as e:
            # Clean up
            await storage._execute_query(f"MATCH (n {{id: '{test_node['id']}'}}) DELETE n")
            raise e
    
    @pytest.mark.asyncio
    async def test_execute_in_transaction_error(self, transaction_manager, storage):
        """Test execute_in_transaction with error"""
        # Create test data
        merge_id = f"test_merge_{uuid.uuid4().hex}"
        test_node = {
            "id": f"test_node_{uuid.uuid4().hex}",
            "label": "TestNode",
            "type": "TestType",
            "name": "Test Node",
            "description": "Test node for transaction testing"
        }
        
        # Define callback function that raises an error
        async def create_node_and_fail(merge_id, transaction_id):
            # Create node
            node = await storage.create_node(
                label=test_node["label"],
                properties={
                    "id": test_node["id"],
                    "type": test_node["type"],
                    "name": test_node["name"],
                    "description": test_node["description"]
                }
            )
            
            # Verify node exists in database
            retrieved_node = await storage.get_node_by_id(test_node["id"])
            assert retrieved_node is not None
            
            # Raise exception to trigger rollback
            raise ValueError("Test error to trigger rollback")
        
        try:
            # Execute callback within transaction (should fail)
            with pytest.raises(ValueError, match="Test error to trigger rollback"):
                await transaction_manager.execute_in_transaction(
                    create_node_and_fail,
                    merge_id=merge_id
                )
            
            # Verify node does not exist in database after rollback
            retrieved_node = await storage.get_node_by_id(test_node["id"])
            assert retrieved_node is None
            
        except Exception as e:
            # Clean up
            await storage._execute_query(f"MATCH (n {{id: '{test_node['id']}'}}) DELETE n")
            raise e
    
    @pytest.mark.asyncio
    async def test_multiple_operations_in_transaction(self, transaction_manager, storage):
        """Test multiple operations in a single transaction"""
        # Create test data
        merge_id = f"test_merge_{uuid.uuid4().hex}"
        test_nodes = [
            {
                "id": f"test_node_1_{uuid.uuid4().hex}",
                "label": "TestNode",
                "type": "TestType",
                "name": "Test Node 1"
            },
            {
                "id": f"test_node_2_{uuid.uuid4().hex}",
                "label": "TestNode",
                "type": "TestType",
                "name": "Test Node 2"
            }
        ]
        test_relationship = {
            "source_id": test_nodes[0]["id"],
            "target_id": test_nodes[1]["id"],
            "type": "TEST_RELATION"
        }
        
        # Define callback function
        async def create_graph(merge_id, transaction_id):
            # Create nodes
            nodes = []
            for node_data in test_nodes:
                node = await storage.create_node(
                    label=node_data["label"],
                    properties={
                        "id": node_data["id"],
                        "type": node_data["type"],
                        "name": node_data["name"]
                    }
                )
                nodes.append(node)
            
            # Create relationship
            edge = await storage.create_relationship(
                source_id=test_relationship["source_id"],
                target_id=test_relationship["target_id"],
                rel_type=test_relationship["type"]
            )
            
            return {
                "nodes": [node.id for node in nodes],
                "edge": edge.id,
                "transaction_id": transaction_id
            }
        
        try:
            # Execute callback within transaction
            result = await transaction_manager.execute_in_transaction(
                create_graph,
                merge_id=merge_id
            )
            
            # Verify result
            assert len(result["nodes"]) == 2
            assert result["nodes"][0] == test_nodes[0]["id"]
            assert result["nodes"][1] == test_nodes[1]["id"]
            assert "edge" in result
            
            # Verify nodes exist in database
            for node_data in test_nodes:
                retrieved_node = await storage.get_node_by_id(node_data["id"])
                assert retrieved_node is not None
                assert retrieved_node.id == node_data["id"]
            
            # Verify relationship exists
            edges = await storage.get_edges_between(
                test_nodes[0]["id"],
                test_nodes[1]["id"]
            )
            assert len(edges) == 1
            assert edges[0].source == test_nodes[0]["id"]
            assert edges[0].target == test_nodes[1]["id"]
            assert edges[0].type == test_relationship["type"]
            
            # Clean up
            for node_data in test_nodes:
                await storage._execute_query(f"MATCH (n {{id: '{node_data['id']}'}}) DETACH DELETE n")
            
        except Exception as e:
            # Clean up
            for node_data in test_nodes:
                await storage._execute_query(f"MATCH (n {{id: '{node_data['id']}'}}) DETACH DELETE n")
            raise e 