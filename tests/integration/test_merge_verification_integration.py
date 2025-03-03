"""Integration tests for merge verification functionality"""
import pytest
import asyncio
import uuid
from datetime import datetime
import os
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock

from app.services.merge.service import MergeService
from app.services.merge.verification import PostMergeVerifier
from app.services.merge.models import (
    VerificationResult,
    VerificationCheck,
    VerificationCheckType,
    MergeStage
)
from app.services.merge.progress import ProgressTracker
from app.services.storage.neo4j import Neo4jStorage
from app.services.storage.models import Node, Edge, TransformationResult
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.merge import VerificationResultResponse

# Skip tests if not running integration tests
pytestmark = pytest.mark.skipif(
    os.environ.get("INTEGRATION_TESTS", "0") != "1",
    reason="Integration tests only run when INTEGRATION_TESTS=1"
)

@pytest.fixture
async def neo4j_storage():
    """Create a Neo4j storage instance"""
    from app.config import settings
    
    storage = Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )
    
    yield storage
    
    # Clean up
    await storage.close()

@pytest.fixture
async def progress_tracker():
    """Create a progress tracker"""
    tracker = ProgressTracker()
    return tracker

@pytest.fixture
async def merge_service(neo4j_storage, progress_tracker):
    """Create a merge service"""
    service = MergeService(
        storage=neo4j_storage,
        production_storage=neo4j_storage,
        progress_tracker=progress_tracker
    )
    return service

@pytest.fixture
async def test_data():
    """Create test data for verification tests"""
    transform_id = f"transform-{uuid.uuid4()}"
    merge_id = f"merge-{uuid.uuid4()}"
    
    # Create nodes
    nodes = [
        {
            "id": f"person-{uuid.uuid4()}",
            "label": "Person",
            "type": "Person",
            "properties": {
                "name": "John Doe",
                "age": 30,
                "transform_id": transform_id
            }
        },
        {
            "id": f"company-{uuid.uuid4()}",
            "label": "Company",
            "type": "Company",
            "properties": {
                "name": "Acme Inc",
                "founded": 1990,
                "transform_id": transform_id
            }
        }
    ]
    
    # Create relationships
    relationships = [
        {
            "id": f"works_at-{uuid.uuid4()}",
            "source": nodes[0]["id"],
            "target": nodes[1]["id"],
            "type": "WORKS_AT",
            "properties": {
                "since": 2020,
                "transform_id": transform_id
            }
        }
    ]
    
    return {
        "transform_id": transform_id,
        "merge_id": merge_id,
        "nodes": nodes,
        "relationships": relationships
    }

@pytest.fixture
async def setup_test_graph(neo4j_storage, test_data):
    """Set up a test graph in Neo4j"""
    # Store nodes
    for i, node in enumerate(test_data["nodes"]):
        await neo4j_storage.store_nodes([node], i, test_data["transform_id"])
    
    # Store relationships
    for i, rel in enumerate(test_data["relationships"]):
        await neo4j_storage.store_relationships([rel], i, test_data["transform_id"])
    
    yield test_data
    
    # Clean up
    try:
        # Delete nodes and relationships with this transform_id
        async with neo4j_storage.driver.session() as session:
            await session.run(
                """
                MATCH (n)
                WHERE n.transform_id = $transform_id
                DETACH DELETE n
                """,
                transform_id=test_data["transform_id"]
            )
    except Exception as e:
        print(f"Error cleaning up test data: {str(e)}")

class TestMergeVerificationIntegration:
    """Integration tests for merge verification"""
    
    @pytest.mark.asyncio
    async def test_successful_verification(self, merge_service, setup_test_graph, progress_tracker):
        """Test successful verification of a merge operation"""
        # Arrange
        transform_id = setup_test_graph["transform_id"]
        merge_id = setup_test_graph["merge_id"]
        
        # Mock the ontology service
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}},
                    "Project": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {"properties": {}}
                }
            }
            
            # Initialize progress tracking
            await progress_tracker.initialize_merge_progress(merge_id)
            
            # Act
            verification_result = await merge_service.verify_merge(merge_id, transform_id)
            
            # Assert
            # We expect the verification to fail due to orphaned nodes and ontology constraints
            assert verification_result.success is False
            assert verification_result.merge_id == merge_id
            assert verification_result.transform_id == transform_id
            assert len(verification_result.checks) == 5
            
            # Check that specific checks pass and others fail
            node_count_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.NODE_COUNT)
            relationship_count_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.RELATIONSHIP_COUNT)
            property_values_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.PROPERTY_VALUES)
            orphaned_nodes_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.ORPHANED_NODES)
            ontology_constraints_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.ONTOLOGY_CONSTRAINTS)
            
            assert node_count_check.success is True
            assert relationship_count_check.success is True
            assert property_values_check.success is True
            assert orphaned_nodes_check.success is False
            assert ontology_constraints_check.success is False
            
            # Check progress tracking
            progress = await progress_tracker.get_merge_progress(merge_id)
            assert progress.current_stage == MergeStage.VERIFICATION
            assert progress.stages_progress[MergeStage.VERIFICATION].status == "failed"
    
    @pytest.mark.asyncio
    async def test_verification_with_orphaned_node(self, merge_service, neo4j_storage, setup_test_graph, progress_tracker):
        """Test verification with an orphaned node"""
        # Arrange
        transform_id = setup_test_graph["transform_id"]
        merge_id = setup_test_graph["merge_id"]
        
        # Add an orphaned node
        orphaned_node = {
            "id": f"orphaned-{uuid.uuid4()}",
            "label": "Project",
            "type": "Project",
            "properties": {
                "name": "Secret Project",
                "transform_id": transform_id
            }
        }
        await neo4j_storage.store_nodes([orphaned_node], 2, transform_id)
        
        # Initialize progress tracking
        await progress_tracker.initialize_merge_progress(merge_id)
        
        # Mock the ontology service
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}},
                    "Project": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {"properties": {}}
                }
            }
            
            # Act
            verification_result = await merge_service.verify_merge(merge_id, transform_id)
            
            # Assert
            assert verification_result.success is False
            
            # Find the orphaned node check
            orphan_check = next(
                (check for check in verification_result.checks if check.check_type == "orphaned_nodes"),
                None
            )
            assert orphan_check is not None
            assert orphan_check.success is False
            assert len(orphan_check.affected_entities) == 4
            
            # Check progress tracking
            progress = await progress_tracker.get_merge_progress(merge_id)
            assert progress.current_stage == MergeStage.VERIFICATION
            assert progress.stages_progress[MergeStage.VERIFICATION].status == "failed"
    
    @pytest.mark.asyncio
    async def test_verification_with_ontology_violation(self, merge_service, neo4j_storage, setup_test_graph, progress_tracker):
        """Test verification with ontology constraint violation"""
        # Arrange
        transform_id = setup_test_graph["transform_id"]
        merge_id = setup_test_graph["merge_id"]
        
        # Add a node with invalid type
        invalid_node = {
            "id": f"invalid-{uuid.uuid4()}",
            "label": "InvalidType",
            "type": "InvalidType",
            "properties": {
                "name": "Invalid Node",
                "transform_id": transform_id
            }
        }
        await neo4j_storage.store_nodes([invalid_node], 2, transform_id)
        
        # Initialize progress tracking
        await progress_tracker.initialize_merge_progress(merge_id)
        
        # Mock the ontology service
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}},
                    "Project": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {"properties": {}}
                }
            }
            
            # Act
            verification_result = await merge_service.verify_merge(merge_id, transform_id)
            
            # Assert
            assert verification_result.success is False
            
            # Find the ontology check
            ontology_check = next(
                (check for check in verification_result.checks if check.check_type == "ontology_constraints"),
                None
            )
            assert ontology_check is not None
            assert ontology_check.success is False
            
            # Check progress tracking
            progress = await progress_tracker.get_merge_progress(merge_id)
            assert progress.current_stage == MergeStage.VERIFICATION
            assert progress.stages_progress[MergeStage.VERIFICATION].status == "failed"
    
    @pytest.mark.asyncio
    async def test_verification_after_property_resolution(self, merge_service, neo4j_storage, setup_test_graph, progress_tracker):
        """Test verification after property conflict resolution"""
        # Arrange
        transform_id = setup_test_graph["transform_id"]
        merge_id = setup_test_graph["merge_id"]
        
        # Initialize progress tracking
        await progress_tracker.initialize_merge_progress(merge_id)
        
        # Create a mock conflict
        conflict = Conflict(
            id=f"conflict-{uuid.uuid4()}",
            merge_id=merge_id,
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            entity_id=setup_test_graph["nodes"][0]["id"],
            entity_type="Person",
            property_name="name",
            description="Property value conflict on name field",
            context={
                "staging_value": "John Smith",
                "production_value": "John Doe"
            },
            resolution_options=[
                ResolutionOption(
                    id="option1",
                    resolution_type="use_staging_value",
                    description="Use staging value",
                    confidence=0.9,
                    resolution_data={
                        "entity_id": setup_test_graph["nodes"][0]["id"],
                        "property_name": "name",
                        "value": "John Smith"
                    }
                )
            ],
            status="pending"
        )
        
        # Mock resolution application
        with patch("app.services.merge.verification.ResolutionHistoryService") as mock_resolution_service:
            service_instance = mock_resolution_service.return_value
            service_instance.get_resolution_history = AsyncMock(return_value=[
                {
                    "conflict_id": conflict.id,
                    "resolution_option_id": "option1",
                    "entity_id": setup_test_graph["nodes"][0]["id"],
                    "property_name": "name",
                    "resolved_value": "John Smith",
                    "timestamp": datetime.now().isoformat()
                }
            ])
            
            # Mock the ontology service
            with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
                mock_load_ontology.return_value = {
                    "node_types": {
                        "Person": {"properties": {}},
                        "Company": {"properties": {}},
                        "Project": {"properties": {}}
                    },
                    "relationship_types": {
                        "WORKS_AT": {"properties": {}}
                    }
                }
                
                # Act
                verification_result = await merge_service.verify_merge(merge_id, transform_id)
                
                # Assert
                # We expect the verification to fail due to orphaned nodes and ontology constraints
                assert verification_result.success is False
                
                # Check that specific checks pass and others fail
                node_count_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.NODE_COUNT)
                relationship_count_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.RELATIONSHIP_COUNT)
                property_values_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.PROPERTY_VALUES)
                orphaned_nodes_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.ORPHANED_NODES)
                ontology_constraints_check = next(c for c in verification_result.checks if c.check_type == VerificationCheckType.ONTOLOGY_CONSTRAINTS)
                
                assert node_count_check.success is True
                assert relationship_count_check.success is True
                assert property_values_check.success is True
                assert orphaned_nodes_check.success is False
                assert ontology_constraints_check.success is False
                
                # Check progress tracking
                progress = await progress_tracker.get_merge_progress(merge_id)
                assert progress.current_stage == MergeStage.VERIFICATION
                assert progress.stages_progress[MergeStage.VERIFICATION].status == "failed"
    
    @pytest.mark.asyncio
    async def test_verification_api_endpoint(self, client, setup_test_graph, progress_tracker):
        """Test the verification API endpoint"""
        # Arrange
        transform_id = setup_test_graph["transform_id"]
        merge_id = setup_test_graph["merge_id"]
        
        # Initialize progress tracking
        await progress_tracker.initialize_merge_progress(merge_id)
        
        # Mock the ontology service
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}},
                    "Project": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {"properties": {}}
                }
            }
            
            # Act
            response = client.post(
                f"/api/v1/merge/{merge_id}/verify?transform_id={transform_id}"
            )
            
            # Assert
            assert response.status_code == 200
            result = response.json()
            assert result["merge_id"] == merge_id
            assert result["transform_id"] == transform_id
            assert len(result["checks"]) == 5
            
            # Check progress tracking
            progress = await progress_tracker.get_merge_progress(merge_id)
            assert progress.current_stage == MergeStage.VERIFICATION 