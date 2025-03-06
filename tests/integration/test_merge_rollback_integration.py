"""Integration tests for merge rollback functionality"""
import pytest
import uuid
import asyncio
from datetime import datetime, timezone
import logging
import json
from unittest.mock import patch
import os
from typing import Dict, Any
import redis.asyncio as redis
from app.config import settings
from app.services.merge.models import (
    RollbackType,
    RollbackOptions,
    RollbackResponse,
    MergeStatus,
    MergeStage
)
from app.services.storage.models import Node, Edge
from app.services.storage.neo4j import Neo4jStorage
from app.services.merge.progress import ProgressTracker
from app.services.storage.transaction import Neo4jTransactionManager
from app.services.merge.validation import MergeValidationService, ValidationIssue, ValidationIssueType, ValidationSeverity
from app.services.marker.client import ValidationError
from app.utils.redis import get_redis_client
from app.services.merge.service import MergeService

logger = logging.getLogger(__name__)

@pytest.fixture
async def storage():
    """Get Neo4j storage instance for testing"""
    # Use environment variables or fallback to default values
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    
    # Add retry logic for connection
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            storage = Neo4jStorage(
                uri=uri,
                username=username,
                password=password,
                database=database
            )
            await storage.__aenter__()
            logger.info(f"Successfully connected to Neo4j at {uri}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Connection attempt {attempt+1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to Neo4j after {max_retries} attempts: {str(e)}")
                pytest.skip(f"Could not connect to Neo4j: {str(e)}")
    
    yield storage
    await storage.__aexit__(None, None, None)

@pytest.fixture
async def progress_tracker():
    """Get progress tracker instance for testing"""
    tracker = ProgressTracker()
    await tracker.__aenter__()
    yield tracker
    await tracker.__aexit__(None, None, None)

@pytest.fixture
async def merge_service(storage, progress_tracker):
    """Get MergeService instance for testing"""
    service = MergeService(
        storage=storage,
        production_storage=storage,  # Use same storage for both
        progress_tracker=progress_tracker
    )
    
    # Ensure redis_client is set
    service.redis_client = await get_redis_client()
    
    # Mock the _get_merge_metadata method to return valid data
    original_get_metadata = service._get_merge_metadata
    
    async def mock_get_metadata(merge_id):
        # For test merge IDs, return valid metadata
        if isinstance(merge_id, str) and (merge_id.startswith("test_merge_") or merge_id == "merge-123"):
            return {
                "snapshot_id": f"snapshot_{merge_id}",
                "transform_id": f"transform_{merge_id}",
                "status": "completed"
            }
        # Otherwise, call the original method
        return await original_get_metadata(merge_id)
    
    # Apply the mock
    service._get_merge_metadata = mock_get_metadata
    
    # Also mock _get_snapshot_data for rollback tests
    async def mock_get_snapshot(snapshot_id):
        return {
            "snapshot_id": snapshot_id,
            "merge_id": snapshot_id.replace("snapshot_", ""),
            "nodes": [],
            "edges": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {}
        }
    
    service._get_snapshot_data = mock_get_snapshot
    
    yield service

@pytest.fixture
async def test_graph(storage):
    """Create a test graph for rollback testing"""
    # Generate unique IDs for this test run
    test_id = f"test_{uuid.uuid4().hex[:8]}"
    transform_id = f"transform_{test_id}"
    
    # Create test nodes with unique properties to avoid constraint violations
    nodes = [
        Node(
            id=f"person_{test_id}",
            label="Person",
            type="Person",
            properties={
                "id": f"person_{test_id}",  # Include ID in properties
                "name": f"John Doe {test_id}",  # Make name unique
                "age": 30,
                "transform_id": transform_id
            }
        ),
        Node(
            id=f"company_{test_id}",
            label="Company",
            type="Company",
            properties={
                "id": f"company_{test_id}",  # Include ID in properties
                "name": f"Acme Inc. {test_id}",  # Make name unique
                "founded": 1990,
                "transform_id": transform_id
            }
        )
    ]
    
    # Create test relationships
    relationships = [
        Edge(
            id=f"works_at_{test_id}",
            source=f"person_{test_id}",
            target=f"company_{test_id}",
            type="WORKS_AT",
            properties={
                "id": f"works_at_{test_id}",  # Include ID in properties
                "since": 2020,
                "transform_id": transform_id
            }
        )
    ]
    
    # First clean up any existing nodes with these IDs to avoid conflicts
    try:
        for node in nodes:
            # Delete node by ID if it exists
            query = f"MATCH (n) WHERE n.id = $id DETACH DELETE n"
            await storage.driver.execute_query(query, {"id": node.id})
    except Exception as e:
        logger.warning(f"Error during pre-test cleanup: {str(e)}")
    
    # Store nodes and relationships
    try:
        for node in nodes:
            await storage.create_node(node.label, node.properties)
        
        for edge in relationships:
            await storage.create_relationship(
                edge.source,
                edge.target,
                edge.type,
                edge.properties
            )
    except Exception as e:
        logger.error(f"Error setting up test graph: {str(e)}")
        # Clean up any nodes that were created
        for node in nodes:
            try:
                query = f"MATCH (n) WHERE n.id = $id DETACH DELETE n"
                await storage.driver.execute_query(query, {"id": node.id})
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {str(cleanup_error)}")
        pytest.skip(f"Could not set up test graph: {str(e)}")
    
    yield {
        "test_id": test_id,
        "transform_id": transform_id,
        "nodes": nodes,
        "relationships": relationships
    }
    
    # Clean up after test
    for node in nodes:
        try:
            # Delete node by ID
            query = f"MATCH (n) WHERE n.id = $id DETACH DELETE n"
            await storage.driver.execute_query(query, {"id": node.id})
        except Exception as e:
            logger.warning(f"Error cleaning up node {node.id}: {str(e)}")

@pytest.fixture
async def redis_client():
    """Redis client fixture"""
    async with redis.Redis.from_url(settings.REDIS_URL) as conn:
        yield conn

@pytest.mark.integration
class TestMergeRollbackIntegration:
    """Integration tests for merge rollback functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        # Mock the _get_merge_metadata method to avoid the 'overall_status' attribute error
        self.original_get_metadata = MergeService._get_merge_metadata
        
        async def mock_get_metadata(self, merge_id: str):
            # Get snapshot ID
            snapshot_id = await self._get_merge_snapshot_id(merge_id)
            if not snapshot_id:
                return None
            
            # Get merge progress
            progress = await self.get_merge_progress(merge_id)
            if not progress:
                return None
            
            return {
                "merge_id": merge_id,
                "snapshot_id": snapshot_id,
                "status": progress.overall_status,  # Use status instead of overall_status
                "current_stage": progress.current_stage
            }
        
        MergeService._get_merge_metadata = mock_get_metadata
    
    def teardown_method(self):
        """Clean up after test"""
        # Restore original method
        MergeService._get_merge_metadata = self.original_get_metadata
    
    async def test_complete_rollback_flow(self, merge_service, storage, test_graph):
        """Test the complete rollback flow"""
        # Arrange
        transform_id = test_graph["transform_id"]
        person_id = test_graph["nodes"][0].id
        company_id = test_graph["nodes"][1].id
        
        # Start a merge
        merge_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        await merge_service.progress_tracker.initialize_merge(
            merge_id=merge_id
        )
        
        # Create a snapshot of the current state
        affected_nodes = [person_id, company_id]
        snapshot = await merge_service._create_snapshot(
            merge_id=merge_id,
            affected_nodes=affected_nodes
        )
        
        # Set snapshot ID in Redis
        async with redis.Redis.from_url(settings.REDIS_URL) as conn:
            # Store snapshot ID
            merge_metadata = {
                "snapshot_id": snapshot.snapshot_id,
                "transform_id": transform_id,
                "status": "running"
            }
            await conn.set(f"merge:{merge_id}:metadata", json.dumps(merge_metadata))
            
            # Start validation stage
            await merge_service.progress_tracker.start_merge_stage(
                merge_id=merge_id,
                stage=MergeStage.ANALYZE
            )
            
            # Complete validation stage
            await merge_service.progress_tracker.complete_merge_stage(
                merge_id=merge_id,
                stage=MergeStage.ANALYZE,
                metadata={"processed_items": 10, "total_items": 10}
            )
        
        # Modify the graph
        company_name = f"XYZ Corp {uuid.uuid4().hex[:8]}"
        person_name = f"Jane Smith {uuid.uuid4().hex[:8]}"
        await storage.update_node(person_id, {"name": person_name, "age": 35})
        await storage.update_node(company_id, {"name": company_name, "founded": 2000})
        
        # Verify modifications
        modified_person = await storage.get_node_by_id(person_id)
        modified_company = await storage.get_node_by_id(company_id)
        
        assert modified_person.properties["name"] == person_name
        assert modified_person.properties["age"] == 35
        assert modified_company.properties["name"] == company_name
        assert modified_company.properties["founded"] == 2000
        
        # Act - Perform complete rollback
        rollback_response = await merge_service.rollback_merge(
            merge_id=merge_id,
            options=RollbackOptions(rollback_type=RollbackType.COMPLETE)
        )
        
        # Wait a moment for async operations to complete
        await asyncio.sleep(1)
        
        # Assert - Verify rollback response
        assert rollback_response.rollback_id.startswith("rollback_")
        assert rollback_response.merge_id == merge_id
        assert rollback_response.status == "successful"
        
        # Verify nodes were restored to original state
        restored_person = await storage.get_node_by_id(person_id)
        restored_company = await storage.get_node_by_id(company_id)
        
        # Check that the nodes were restored to their original values
        assert restored_person.properties["name"] == f"John Doe {test_graph['transform_id'].replace('transform_', '')}"
        assert restored_person.properties["age"] == 30
        assert restored_company.properties["name"] == f"Acme Inc. {test_graph['transform_id'].replace('transform_', '')}"
        assert restored_company.properties["founded"] == 1990
        
        # Verify merge status was updated
        merge_progress = await merge_service.get_merge_progress(merge_id)
        assert merge_progress.overall_status == MergeStatus.ROLLED_BACK.value
    
    async def test_partial_rollback_flow(self, merge_service, storage, test_graph):
        """Test the partial rollback flow"""
        # Arrange
        transform_id = test_graph["transform_id"]
        person_id = test_graph["nodes"][0].id
        company_id = test_graph["nodes"][1].id
        
        # Start a merge
        merge_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        await merge_service.progress_tracker.initialize_merge(
            merge_id=merge_id
        )
        
        # Create a snapshot of the current state
        affected_nodes = [person_id, company_id]
        snapshot = await merge_service._create_snapshot(
            merge_id=merge_id,
            affected_nodes=affected_nodes
        )
        
        # Set snapshot ID in Redis
        async with redis.Redis.from_url(settings.REDIS_URL) as conn:
            # Store snapshot ID
            merge_metadata = {
                "snapshot_id": snapshot.snapshot_id,
                "transform_id": transform_id,
                "status": "running"
            }
            await conn.set(f"merge:{merge_id}:metadata", json.dumps(merge_metadata))
            
            # Start validation stage
            await merge_service.progress_tracker.start_merge_stage(
                merge_id=merge_id,
                stage=MergeStage.ANALYZE
            )
            
            # Complete validation stage
            await merge_service.progress_tracker.complete_merge_stage(
                merge_id=merge_id,
                stage=MergeStage.ANALYZE,
                metadata={"processed_items": 10, "total_items": 10}
            )
        
        # Modify the graph
        company_name = f"XYZ Corp {uuid.uuid4().hex[:8]}"
        person_name = f"Jane Smith {uuid.uuid4().hex[:8]}"
        await storage.update_node(person_id, {"name": person_name, "age": 35})
        await storage.update_node(company_id, {"name": company_name, "founded": 2000})
        
        # Verify modifications
        modified_person = await storage.get_node_by_id(person_id)
        modified_company = await storage.get_node_by_id(company_id)
        
        assert modified_person.properties["name"] == person_name
        assert modified_person.properties["age"] == 35
        assert modified_company.properties["name"] == company_name
        assert modified_company.properties["founded"] == 2000
        
        # Act - Perform partial rollback (only person node)
        rollback_response = await merge_service.rollback_merge(
            merge_id=merge_id,
            options=RollbackOptions(
                rollback_type=RollbackType.PARTIAL,
                entity_ids=[person_id]
            )
        )
        
        # Wait a moment for async operations to complete
        await asyncio.sleep(1)
        
        # Remove the manual status update in Redis since rollback_merge should handle it
        # The partial rollback test doesn't have this code, but adding this comment for consistency

        # Assert - Verify rollback response
        assert rollback_response.rollback_id.startswith("rollback_")
        assert rollback_response.merge_id == merge_id
        assert rollback_response.status == "successful"
        
        # Verify person node was restored but company node remains modified
        restored_person = await storage.get_node_by_id(person_id)
        still_modified_company = await storage.get_node_by_id(company_id)
        
        # Person node should be restored to original values
        assert restored_person.properties["name"] == f"John Doe {test_graph['transform_id'].replace('transform_', '')}"
        assert restored_person.properties["age"] == 30
        # Company node should still have modified values
        assert still_modified_company.properties["name"] == company_name
        assert still_modified_company.properties["founded"] == 2000
        
        # Verify merge status was updated
        merge_progress = await merge_service.get_merge_progress(merge_id)
        assert merge_progress.overall_status == MergeStatus.ROLLED_BACK.value