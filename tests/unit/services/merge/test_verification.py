"""Unit tests for the PostMergeVerifier class"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.merge.verification import PostMergeVerifier
from app.services.merge.models import (
    VerificationResult,
    VerificationCheck,
    VerificationCheckType
)
from app.services.storage.models import TransformationResult

@pytest.fixture
def mock_storage_service():
    """Create a mock storage service"""
    storage = AsyncMock()
    
    # Mock get_transformation_data
    storage.get_transformation_data.return_value = TransformationResult(
        transform_id="transform-123",
        nodes=[
            {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "John"}},
            {"id": "node2", "type": "Company", "label": "Company", "properties": {"name": "Acme"}}
        ],
        relationships=[
            {"id": "rel1", "source": "node1", "target": "node2", "type": "WORKS_AT", "properties": {}}
        ],
        timestamp=datetime.now()
    )
    
    # Mock get_production_graph_for_transform
    storage.get_production_graph_for_transform.return_value = TransformationResult(
        transform_id="transform-123",
        nodes=[
            {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "John"}},
            {"id": "node2", "type": "Company", "label": "Company", "properties": {"name": "Acme"}}
        ],
        relationships=[
            {"id": "rel1", "source": "node1", "target": "node2", "type": "WORKS_AT", "properties": {}}
        ],
        timestamp=datetime.now()
    )
    
    return storage

@pytest.fixture
def mock_resolution_service():
    """Create a mock resolution history service"""
    with patch("app.services.merge.verification.ResolutionHistoryService") as mock:
        service = mock.return_value
        service.get_resolution_history = AsyncMock(return_value=[])
        yield service

@pytest.fixture
def mock_redis():
    """Create a mock Redis client"""
    with patch("redis.asyncio.Redis.from_url") as mock:
        client = mock.return_value
        client.set.return_value = True
        client.expire.return_value = True
        client.sadd.return_value = True
        yield client

@pytest.fixture
def verifier(mock_storage_service, mock_resolution_service):
    """Create a PostMergeVerifier instance"""
    return PostMergeVerifier(
        merge_id="merge-123",
        transform_id="transform-123",
        storage_service=mock_storage_service
    )

class TestPostMergeVerifier:
    """Tests for the PostMergeVerifier class"""
    
    @pytest.mark.asyncio
    async def test_successful_verification(self, verifier, mock_storage_service, mock_redis):
        """Test successful verification"""
        # Arrange
        # Mock ontology loading
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {
                        "valid_connections": [
                            {"source": "Person", "target": "Company"}
                        ]
                    }
                }
            }
            
            # Mock Redis
            with patch("redis.asyncio.Redis.from_url", return_value=mock_redis):
                # Act
                result = await verifier.verify_merge()
                
                # Assert
                assert result.success is True
                assert len(result.checks) == 5
                assert all(check.success for check in result.checks)
                assert result.merge_id == "merge-123"
                assert result.transform_id == "transform-123"
                assert result.completed_at is not None
                assert result.verification_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_node_count_mismatch(self, verifier, mock_storage_service, mock_redis):
        """Test node count mismatch detection"""
        # Arrange
        # Modify production graph to have fewer nodes
        production_graph = TransformationResult(
            transform_id="transform-123",
            nodes=[
                {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "John"}}
                # node2 is missing
            ],
            relationships=[],
            timestamp=datetime.now()
        )
        mock_storage_service.get_production_graph_for_transform.return_value = production_graph
        
        # Mock ontology loading
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}}
                },
                "relationship_types": {}
            }
            
            # Mock Redis
            with patch("redis.asyncio.Redis.from_url", return_value=mock_redis):
                # Act
                result = await verifier.verify_merge()
                
                # Assert
                assert result.success is False
                
                # Find the node count check
                node_count_check = next(
                    (check for check in result.checks if check.check_type == VerificationCheckType.NODE_COUNT),
                    None
                )
                assert node_count_check is not None
                assert node_count_check.success is False
                assert "mismatch" in node_count_check.message.lower()
                assert "Company" in node_count_check.details["missing_nodes"]
    
    @pytest.mark.asyncio
    async def test_relationship_count_mismatch(self, verifier, mock_storage_service, mock_redis):
        """Test relationship count mismatch detection"""
        # Arrange
        # Modify production graph to have no relationships
        production_graph = TransformationResult(
            transform_id="transform-123",
            nodes=[
                {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "John"}},
                {"id": "node2", "type": "Company", "label": "Company", "properties": {"name": "Acme"}}
            ],
            relationships=[],  # No relationships
            timestamp=datetime.now()
        )
        mock_storage_service.get_production_graph_for_transform.return_value = production_graph
        
        # Mock ontology loading
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {
                        "valid_connections": [
                            {"source": "Person", "target": "Company"}
                        ]
                    }
                }
            }
            
            # Mock Redis
            with patch("redis.asyncio.Redis.from_url", return_value=mock_redis):
                # Act
                result = await verifier.verify_merge()
                
                # Assert
                assert result.success is False
                
                # Find the relationship count check
                rel_count_check = next(
                    (check for check in result.checks if check.check_type == VerificationCheckType.RELATIONSHIP_COUNT),
                    None
                )
                assert rel_count_check is not None
                assert rel_count_check.success is False
                assert "mismatch" in rel_count_check.message.lower()
                assert "WORKS_AT" in rel_count_check.details["missing_relationships"]
    
    @pytest.mark.asyncio
    async def test_property_value_mismatch(self, verifier, mock_storage_service, mock_resolution_service, mock_redis):
        """Test property value mismatch detection"""
        # Arrange
        # Modify production graph to have different property value
        production_graph = TransformationResult(
            transform_id="transform-123",
            nodes=[
                {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "Jane"}},  # Changed name
                {"id": "node2", "type": "Company", "label": "Company", "properties": {"name": "Acme"}}
            ],
            relationships=[
                {"id": "rel1", "source": "node1", "target": "node2", "type": "WORKS_AT", "properties": {}}
            ],
            timestamp=datetime.now()
        )
        mock_storage_service.get_production_graph_for_transform.return_value = production_graph
        
        # Mock resolution history
        mock_resolution_service.get_resolution_history = AsyncMock(return_value=[
            MagicMock(
                conflict_id="conflict-1",
                resolution_id="resolution-1",
                resolution_type="use_staging_value",
                resolution_data={"entity_id": "node1"},
                entity_types=["Person"],
                property_names=["name"],
                relationship_types=[]
            )
        ])
        
        # Mock ontology loading
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {
                        "valid_connections": [
                            {"source": "Person", "target": "Company"}
                        ]
                    }
                }
            }
            
            # Mock Redis
            with patch("redis.asyncio.Redis.from_url", return_value=mock_redis):
                # Act
                result = await verifier.verify_merge()
                
                # Assert
                assert result.success is False
                
                # Find the property value check
                property_check = next(
                    (check for check in result.checks if check.check_type == VerificationCheckType.PROPERTY_VALUES),
                    None
                )
                assert property_check is not None
                assert property_check.success is False
                assert "mismatch" in property_check.message.lower()
                assert len(property_check.details["mismatches"]) == 1
                assert property_check.details["mismatches"][0]["entity_id"] == "node1"
                assert property_check.details["mismatches"][0]["property_name"] == "name"
                assert property_check.details["mismatches"][0]["expected_value"] == "John"
                assert property_check.details["mismatches"][0]["actual_value"] == "Jane"
    
    @pytest.mark.asyncio
    async def test_orphaned_node_detection(self, verifier, mock_storage_service, mock_redis):
        """Test orphaned node detection"""
        # Arrange
        # Add an orphaned node to production graph
        production_graph = TransformationResult(
            transform_id="transform-123",
            nodes=[
                {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "John"}},
                {"id": "node2", "type": "Company", "label": "Company", "properties": {"name": "Acme"}},
                {"id": "node3", "type": "Project", "label": "Project", "properties": {"name": "Secret"}}  # Orphaned
            ],
            relationships=[
                {"id": "rel1", "source": "node1", "target": "node2", "type": "WORKS_AT", "properties": {}}
            ],
            timestamp=datetime.now()
        )
        mock_storage_service.get_production_graph_for_transform.return_value = production_graph
        
        # Mock ontology loading
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}},
                    "Project": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {
                        "valid_connections": [
                            {"source": "Person", "target": "Company"}
                        ]
                    }
                }
            }
            
            # Mock Redis
            with patch("redis.asyncio.Redis.from_url", return_value=mock_redis):
                # Act
                result = await verifier.verify_merge()
                
                # Assert
                assert result.success is False
                
                # Find the orphaned node check
                orphan_check = next(
                    (check for check in result.checks if check.check_type == VerificationCheckType.ORPHANED_NODES),
                    None
                )
                assert orphan_check is not None
                assert orphan_check.success is False
                assert "orphaned" in orphan_check.message.lower()
                assert len(orphan_check.details["orphaned_nodes"]) == 1
                assert orphan_check.details["orphaned_nodes"][0]["id"] == "node3"
    
    @pytest.mark.asyncio
    async def test_ontology_constraint_violation(self, verifier, mock_storage_service, mock_redis):
        """Test ontology constraint violation detection"""
        # Arrange
        # Add a relationship that violates ontology constraints
        production_graph = TransformationResult(
            transform_id="transform-123",
            nodes=[
                {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "John"}},
                {"id": "node2", "type": "Company", "label": "Company", "properties": {"name": "Acme"}}
            ],
            relationships=[
                # Reversed direction - Company to Person
                {"id": "rel1", "source": "node2", "target": "node1", "type": "WORKS_AT", "properties": {}}
            ],
            timestamp=datetime.now()
        )
        mock_storage_service.get_production_graph_for_transform.return_value = production_graph
        
        # Mock ontology loading
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {
                        "valid_connections": [
                            # Only Person->Company is valid, not Company->Person
                            {"source": "Person", "target": "Company"}
                        ]
                    }
                }
            }
            
            # Mock Redis
            with patch("redis.asyncio.Redis.from_url", return_value=mock_redis):
                # Act
                result = await verifier.verify_merge()
                
                # Assert
                assert result.success is False
                
                # Find the ontology constraint check
                ontology_check = next(
                    (check for check in result.checks if check.check_type == VerificationCheckType.ONTOLOGY_CONSTRAINTS),
                    None
                )
                assert ontology_check is not None
                assert ontology_check.success is False
                assert "violation" in ontology_check.message.lower()
                assert len(ontology_check.details["violations"]) == 1
                assert ontology_check.details["violations"][0]["type"] == "invalid_relationships"
                assert len(ontology_check.details["violations"][0]["details"]) == 1
                assert ontology_check.details["violations"][0]["details"][0]["error"] == "Invalid relationship direction"
    
    @pytest.mark.asyncio
    async def test_verification_with_deleted_nodes(self, verifier, mock_storage_service, mock_resolution_service, mock_redis):
        """Test verification with deleted nodes"""
        # Arrange
        # Staging has a node that was deleted during merge
        staging_graph = TransformationResult(
            transform_id="transform-123",
            nodes=[
                {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "John"}},
                {"id": "node2", "type": "Company", "label": "Company", "properties": {"name": "Acme"}},
                {"id": "node3", "type": "Project", "label": "Project", "properties": {"name": "Secret"}}  # To be deleted
            ],
            relationships=[
                {"id": "rel1", "source": "node1", "target": "node2", "type": "WORKS_AT", "properties": {}}
            ],
            timestamp=datetime.now()
        )
        mock_storage_service.get_transformation_data.return_value = staging_graph
        
        # Production doesn't have the deleted node
        production_graph = TransformationResult(
            transform_id="transform-123",
            nodes=[
                {"id": "node1", "type": "Person", "label": "Person", "properties": {"name": "John"}},
                {"id": "node2", "type": "Company", "label": "Company", "properties": {"name": "Acme"}}
                # node3 is deleted
            ],
            relationships=[
                {"id": "rel1", "source": "node1", "target": "node2", "type": "WORKS_AT", "properties": {}}
            ],
            timestamp=datetime.now()
        )
        mock_storage_service.get_production_graph_for_transform.return_value = production_graph
        
        # Mock resolution history to indicate node3 was deleted
        mock_resolution_service.get_resolution_history = AsyncMock(return_value=[
            MagicMock(
                conflict_id="conflict-1",
                resolution_id="resolution-1",
                resolution_type="delete_node",
                resolution_data={"entity_id": "node3"},
                entity_types=["Project"],
                property_names=[],
                relationship_types=[]
            )
        ])
        
        # Mock ontology loading
        with patch("app.services.merge.verification.load_ontology") as mock_load_ontology:
            mock_load_ontology.return_value = {
                "node_types": {
                    "Person": {"properties": {}},
                    "Company": {"properties": {}},
                    "Project": {"properties": {}}
                },
                "relationship_types": {
                    "WORKS_AT": {
                        "valid_connections": [
                            {"source": "Person", "target": "Company"}
                        ]
                    }
                }
            }
            
            # Mock Redis
            with patch("redis.asyncio.Redis.from_url", return_value=mock_redis):
                # Act
                result = await verifier.verify_merge()
                
                # Assert
                assert result.success is True
                
                # Find the node count check
                node_count_check = next(
                    (check for check in result.checks if check.check_type == VerificationCheckType.NODE_COUNT),
                    None
                )
                assert node_count_check is not None
                assert node_count_check.success is True
                assert "Project" in node_count_check.details["deleted_counts"]
                assert node_count_check.details["deleted_counts"]["Project"] == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, verifier, mock_storage_service):
        """Test error handling during verification"""
        # Arrange
        # Make storage service throw an exception
        mock_storage_service.get_transformation_data.side_effect = Exception("Database error")
        
        # Act
        result = await verifier.verify_merge()
        
        # Assert
        assert result.success is False
        assert "error" in result.metadata
        assert "Database error" in result.metadata["error"]
        assert result.completed_at is not None 