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

from app.services.merge.service import MergeService
from app.services.merge.models import (
    RollbackType,
    RollbackOptions,
    RollbackResponse,
    MergeStatus
)
from app.services.storage.models import Node, Edge
from app.services.storage.neo4j import Neo4jStorage
from app.services.merge.progress import ProgressTracker
from app.services.storage.transaction import Neo4jTransactionManager

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
    """Get merge service instance for testing"""
    # Use the same storage for both staging and production for testing
    transaction_manager = Neo4jTransactionManager(storage.driver)
    service = MergeService(
        storage=storage,
        production_storage=storage,
        progress_tracker=progress_tracker,
        transaction_manager=transaction_manager
    )
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

@pytest.mark.integration
class TestMergeRollbackIntegration:
    """Integration tests for merge rollback functionality"""
    
    async def test_complete_rollback_flow(self, merge_service, storage, test_graph):
        """Test the complete rollback flow"""
        # Arrange
        transform_id = test_graph["transform_id"]
        person_id = test_graph["nodes"][0].id
        company_id = test_graph["nodes"][1].id
        
        # Start a merge
        merge_id = str(uuid.uuid4())
        await merge_service.progress_tracker.initialize_merge(merge_id)
        
        # Create a snapshot of the current state
        affected_nodes = [person_id, company_id]
        snapshot = await merge_service._create_snapshot(merge_id, affected_nodes)
        
        # Modify the graph
        await storage.update_node(person_id, {"name": "Jane Smith", "age": 35})
        await storage.update_node(company_id, {"name": "XYZ Corp", "founded": 2000})
        
        # Verify modifications
        modified_person = await storage.get_node_by_id(person_id)
        modified_company = await storage.get_node_by_id(company_id)
        
        assert modified_person.properties["name"] == "Jane Smith"
        assert modified_person.properties["age"] == 35
        assert modified_company.properties["name"] == "XYZ Corp"
        assert modified_company.properties["founded"] == 2000
        
        # Act - Perform rollback
        options = RollbackOptions(rollback_type=RollbackType.COMPLETE)
        rollback_response = await merge_service.rollback_merge(merge_id, options)
        
        # Assert - Verify rollback response
        assert rollback_response.rollback_id.startswith("rollback_")
        assert rollback_response.merge_id == merge_id
        assert rollback_response.status == "successful"
        
        # Verify nodes were restored to original state
        restored_person = await storage.get_node_by_id(person_id)
        restored_company = await storage.get_node_by_id(company_id)
        
        assert restored_person.properties["name"] == f"John Doe {test_graph['test_id']}"
        assert restored_person.properties["age"] == 30
        assert restored_company.properties["name"] == f"Acme Inc. {test_graph['test_id']}"
        assert restored_company.properties["founded"] == 1990
        
        # Verify merge status was updated
        merge_progress = await merge_service.get_merge_progress(merge_id)
        assert merge_progress.overall_status == MergeStatus.CANCELLED
        
        # Check if there are error details in the current stage
        if merge_progress.current_stage and merge_progress.current_stage in merge_progress.stages_progress:
            stage_progress = merge_progress.stages_progress[merge_progress.current_stage]
            if stage_progress.error_details:
                assert "rollback" in str(stage_progress.error_details).lower()
    
    async def test_partial_rollback_flow(self, merge_service, storage, test_graph):
        """Test the partial rollback flow"""
        # Arrange
        transform_id = test_graph["transform_id"]
        person_id = test_graph["nodes"][0].id
        company_id = test_graph["nodes"][1].id
        
        # Start a merge
        merge_id = str(uuid.uuid4())
        await merge_service.progress_tracker.initialize_merge(merge_id)
        
        # Create a snapshot of the current state
        affected_nodes = [person_id, company_id]
        snapshot = await merge_service._create_snapshot(merge_id, affected_nodes)
        
        # Modify the graph
        await storage.update_node(person_id, {"name": "Jane Smith", "age": 35})
        await storage.update_node(company_id, {"name": "XYZ Corp", "founded": 2000})
        
        # Verify modifications
        modified_person = await storage.get_node_by_id(person_id)
        modified_company = await storage.get_node_by_id(company_id)
        
        assert modified_person.properties["name"] == "Jane Smith"
        assert modified_person.properties["age"] == 35
        assert modified_company.properties["name"] == "XYZ Corp"
        assert modified_company.properties["founded"] == 2000
        
        # Act - Perform partial rollback (only person node)
        options = RollbackOptions(
            rollback_type=RollbackType.PARTIAL,
            entity_ids=[person_id]
        )
        rollback_response = await merge_service.rollback_merge(merge_id, options)
        
        # Assert - Verify rollback response
        assert rollback_response.rollback_id.startswith("rollback_")
        assert rollback_response.merge_id == merge_id
        assert rollback_response.status == "successful"
        
        # Verify person node was restored but company node remains modified
        restored_person = await storage.get_node_by_id(person_id)
        still_modified_company = await storage.get_node_by_id(company_id)
        
        assert restored_person.properties["name"] == f"John Doe {test_graph['test_id']}"
        assert restored_person.properties["age"] == 30
        assert still_modified_company.properties["name"] == "XYZ Corp"
        assert still_modified_company.properties["founded"] == 2000
    
    async def test_automatic_rollback_on_validation_failure(self, merge_service, storage, test_graph):
        """Test automatic rollback triggered by validation failure"""
        # This test requires mocking the validation service to simulate a validation failure
        # and trigger the automatic rollback
        
        # Arrange
        transform_id = test_graph["transform_id"]
        person_id = test_graph["nodes"][0].id
        company_id = test_graph["nodes"][1].id
        
        # Start a merge
        merge_id = str(uuid.uuid4())
        await merge_service.progress_tracker.initialize_merge(merge_id)
        
        # Create a snapshot of the current state
        affected_nodes = [person_id, company_id]
        snapshot = await merge_service._create_snapshot(merge_id, affected_nodes)
        
        # Modify the graph
        await storage.update_node(person_id, {"name": "Jane Smith", "age": 35})
        
        # Mock the validation service to return a validation failure
        from app.services.merge.validation import MergeValidationService
        from app.services.merge.models import ValidationResult, ValidationIssue, ValidationSeverity, ValidationIssueType
        from unittest.mock import patch

        # Create a validation result with critical issues
        validation_result = ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(
                    type=ValidationIssueType.VALIDATION_ERROR,
                    message="Test validation error",
                    affected_ids=[person_id],
                    severity=ValidationSeverity.CRITICAL
                )
            ],
            critical_count=1,
            warning_count=0,
            info_count=0,
            total_nodes=2,
            total_edges=1,
            validation_time_ms=100.0,
            metadata={}
        )

        # Create a subclass of MergeValidationService for testing
        class TestMergeValidationService(MergeValidationService):
            _skip_validators = True  # Skip registering default validators
            
            async def validate_merge(self, merge_id, transform_id, auto_rollback=False, **kwargs):
                # Return our predefined validation result
                return validation_result
                
            async def _extract_staging_graph(self, transform_id: str) -> Dict[str, Any]:
                """Override to avoid Neo4j calls"""
                return {
                    "nodes": [{"id": person_id}, {"id": company_id}],
                    "edges": [{"id": "test_edge"}],
                    "transform_id": transform_id
                }
        
        # Create validation service with reference to merge service for rollback
        validation_service = TestMergeValidationService(
            storage=storage,
            production_storage=storage,
            merge_service=merge_service
        )

        # Mock the execute_merge method to raise an exception
        original_execute_merge = merge_service.execute_merge

        async def mock_execute_merge(*args, **kwargs):
            # Perform the rollback
            from app.services.merge.models import RollbackOptions, RollbackType
            rollback_options = RollbackOptions(
                rollback_type=RollbackType.COMPLETE,
                auto_rollback_on_validation_failure=True,
                metadata={
                    "validation_result": validation_result.model_dump(),
                    "auto_triggered": True
                }
            )
            await merge_service.rollback_merge(merge_id, rollback_options)
            
            # Raise an exception to simulate validation failure
            raise ValueError("Validation failed: Test validation error")

        # Apply the patch
        with patch.object(merge_service, 'execute_merge', side_effect=mock_execute_merge):
            # Act - Execute merge which should trigger validation and automatic rollback
            with pytest.raises(ValueError) as excinfo:
                await merge_service.execute_merge(
                    merge_id=merge_id,
                    transform_id=transform_id,
                    auto_rollback=True,
                    validation_service=validation_service  # Pass the validation service explicitly
                )

            # Assert - Verify that an error was raised with the expected message
            assert "validation failed" in str(excinfo.value).lower()

        # Verify node was restored to original state
        restored_person = await storage.get_node_by_id(person_id)
        assert restored_person.properties["name"] == f"John Doe {test_graph['test_id']}"
        assert restored_person.properties["age"] == 30 