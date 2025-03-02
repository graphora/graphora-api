"""Tests for the MergeService validation and execute_merge methods"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime
import pytz

from app.services.merge.service import MergeService
from app.services.merge.models import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationIssueType,
    MergeStage,
    MergeStatus,
    StageStatus
)
from app.schemas.graph import GraphResponse, Node, Edge

@pytest.fixture
def mock_storage_factory():
    """Mock storage factory fixture"""
    mock_staging_storage = AsyncMock()
    mock_production_storage = AsyncMock()
    mock_conflict_storage = AsyncMock()
    
    def factory(is_staging=False, is_conflict_storage=False):
        if is_conflict_storage:
            return mock_conflict_storage
        elif is_staging:
            return mock_staging_storage
        else:
            return mock_production_storage
    
    # Add references to the storages for assertions
    factory.staging_storage = mock_staging_storage
    factory.production_storage = mock_production_storage
    factory.conflict_storage = mock_conflict_storage
    
    return factory

@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker fixture"""
    tracker = AsyncMock()
    tracker.start_stage = AsyncMock()
    tracker.complete_stage = AsyncMock()
    tracker.fail_stage = AsyncMock()
    return tracker

@pytest.fixture
def valid_graph():
    """Valid graph fixture"""
    return GraphResponse(
        nodes=[
            Node(
                id="node1",
                label="Person",
                type="Person",
                properties={"name": "Alice", "age": 30}
            ),
            Node(
                id="node2",
                label="Person",
                type="Person",
                properties={"name": "Bob", "age": 25}
            ),
            Node(
                id="node3",
                label="Company",
                type="Company",
                properties={"name": "Acme Inc", "founded": 2010}
            )
        ],
        edges=[
            Edge(
                id="edge1",
                source="node1",
                target="node3",
                type="WORKS_AT",
                properties={"since": 2018}
            ),
            Edge(
                id="edge2",
                source="node2",
                target="node3",
                type="WORKS_AT",
                properties={"since": 2019}
            )
        ]
    )

@pytest.fixture
def merge_service(mock_storage_factory, mock_progress_tracker):
    """Merge service fixture"""
    service = MergeService(
        storage=mock_storage_factory(is_staging=True),
        production_storage=mock_storage_factory(is_staging=False),
        progress_tracker=mock_progress_tracker
    )
    # Add a reference to the storage factory for convenience
    service.get_storage = mock_storage_factory
    return service

class TestMergeServiceValidation:
    """Tests for the MergeService validation and execute_merge methods"""
    
    @pytest.mark.asyncio
    @patch("app.services.merge.validation.MergeValidationService.validate_merge")
    async def test_validate_merge(self, mock_validate_merge, merge_service, valid_graph):
        """Test validate_merge method"""
        # Arrange
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        ontology_id = "test_ontology"
        
        # Create mock validation result
        mock_result = ValidationResult(
            valid=True,
            issues=[],
            critical_count=0,
            warning_count=0,
            info_count=0,
            total_nodes=3,
            total_edges=2,
            validation_time_ms=100.0,
            metadata={
                "merge_id": merge_id,
                "transform_id": transform_id,
                "ontology_id": ontology_id
            }
        )
        mock_validate_merge.return_value = mock_result
        
        # Act
        result = await merge_service.validate_merge(merge_id, transform_id, ontology_id)
        
        # Assert
        assert result.valid is True
        assert result.critical_count == 0
        assert result.total_nodes == 3
        assert result.total_edges == 2
        
        # Verify mock calls
        mock_validate_merge.assert_called_once_with(merge_id, transform_id, ontology_id)
    
    @pytest.mark.asyncio
    @patch("app.services.merge.validation.MergeValidationService.validate_merge")
    @patch("app.services.merge.validation.MergeValidationService.validate_no_orphaned_nodes")
    async def test_validate_merge_with_allowed_orphans(
        self, mock_validate_orphans, mock_validate_merge, merge_service, valid_graph
    ):
        """Test validate_merge method with allowed orphan types"""
        # Arrange
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        ontology_id = "test_ontology"
        allowed_orphan_types = ["Orphan"]
        
        # Create mock validation results
        mock_result = ValidationResult(
            valid=True,
            issues=[
                ValidationIssue(
                    type=ValidationIssueType.ORPHANED_NODE,
                    message="Found 1 orphaned nodes of type 'Orphan'",
                    affected_ids=["node4"],
                    severity=ValidationSeverity.WARNING,
                    metadata={"node_type": "Orphan"}
                )
            ],
            critical_count=0,
            warning_count=1,
            info_count=0,
            total_nodes=4,
            total_edges=2,
            validation_time_ms=100.0,
            metadata={
                "merge_id": merge_id,
                "transform_id": transform_id,
                "ontology_id": ontology_id
            }
        )
        mock_validate_merge.return_value = mock_result
        
        # Mock orphan validation with allowed types
        mock_validate_orphans.return_value = []
        
        # Mock get_graph_by_transform_id
        merge_service.storage.get_graph_by_transform_id.return_value = valid_graph
        
        # Act
        result = await merge_service.validate_merge(
            merge_id, transform_id, ontology_id, allowed_orphan_types=allowed_orphan_types
        )
        
        # Assert
        assert result.valid is True
        assert result.warning_count == 0  # No warnings because orphans are allowed
        
        # Verify mock calls
        mock_validate_merge.assert_called_once_with(merge_id, transform_id, ontology_id)
        mock_validate_orphans.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("app.services.merge.service.MergeService.validate_merge")
    @patch("app.services.merge.service.extract_staging_graph")
    @patch("app.services.merge.service.map_production_entities")
    @patch("app.services.merge.service.start_stage")
    @patch("app.services.merge.service.complete_merge_stage")
    @patch("app.services.merge.service.complete_merge")
    @patch("app.services.merge.service.fail_merge")
    @patch("app.services.merge.service.detect_merge_conflicts")
    async def test_execute_merge_with_valid_validation(self, mock_detect_conflicts, mock_fail_merge, mock_complete_merge, mock_complete_merge_stage, mock_start_stage, mock_map_production_entities, mock_extract_staging_graph, mock_validate_merge, merge_service, valid_graph):
        """Test execute_merge method with valid validation"""
        # Arrange
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        
        # Mock validation result
        mock_validate_merge.return_value = ValidationResult(
            valid=True,
            issues=[],
            critical_count=0,
            warning_count=0,
            info_count=0,
            total_nodes=3,
            total_edges=2,
            validation_time_ms=100.0,
            metadata={}
        )
        
        # Mock extract_staging_graph to return an awaitable that returns valid_graph
        async def mock_extract(storage, transform_id):
            return valid_graph
        mock_extract_staging_graph.side_effect = mock_extract
        
        # Mock map_production_entities to return an awaitable that returns entity_mapping
        entity_mapping = MagicMock()
        entity_mapping.matches = {
            "node1": MagicMock(production_matches=["prod-node1"]),
            "node2": MagicMock(production_matches=["prod-node2"]),
            "node3": MagicMock(production_matches=["prod-node3"])
        }
        async def mock_map(storage, graph):
            return entity_mapping
        mock_map_production_entities.side_effect = mock_map
        
        # Mock detect_merge_conflicts to do nothing
        async def mock_detect(merge_id, graph, entity_mapping, storage, progress_tracker):
            return None
        mock_detect_conflicts.side_effect = mock_detect
        
        # Mock start_stage, complete_merge_stage, complete_merge, and fail_merge
        async def mock_start(merge_id, stage, progress_tracker):
            return None
        mock_start_stage.side_effect = mock_start
        
        async def mock_complete_stage(merge_id, stage, progress_tracker, metadata=None):
            return None
        mock_complete_merge_stage.side_effect = mock_complete_stage
        
        async def mock_complete(merge_id, progress_tracker):
            return None
        mock_complete_merge.side_effect = mock_complete
        
        async def mock_fail(merge_id, error_message, progress_tracker):
            return None
        mock_fail_merge.side_effect = mock_fail
        
        # Mock storage methods
        merge_service.production_storage.update_node = AsyncMock()
        merge_service.production_storage.create_node = AsyncMock()
        merge_service.production_storage.get_relationships_between = AsyncMock(return_value=[])
        merge_service.production_storage.create_relationship = AsyncMock()
        merge_service.production_storage.get_relationship = AsyncMock(return_value=None)
        merge_service.production_storage.update_relationship = AsyncMock()
        
        # Mock get_node_by_id to return a proper Node object
        async def mock_get_node_by_id(node_id):
            from app.schemas.graph import Node
            return Node(
                id=node_id,
                label="Person",
                type="Person",
                properties={"name": f"Person {node_id}", "age": 30}
            )
        merge_service.production_storage.get_node_by_id = AsyncMock(side_effect=mock_get_node_by_id)
        
        # Mock get_conflicts to return empty conflicts
        merge_service.get_conflicts = AsyncMock(return_value=([], 0))
        
        # Mock transaction manager
        class MockTransactionManager:
            async def __aenter__(self):
                return "transaction-123"
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        class MockTransaction:
            def __init__(self):
                self.transaction_id = "transaction-123"
            
            async def begin_transaction(self, merge_id: str) -> str:
                return self.transaction_id
                
            async def commit_transaction(self, transaction_id: str) -> bool:
                return True
                
            async def rollback_transaction(self, transaction_id: str) -> bool:
                return True
                
            def transaction(self):
                return MockTransactionManager()

        transaction_manager = MockTransaction()
        merge_service._get_transaction_manager = MagicMock(return_value=transaction_manager)
        
        # Mock snapshot creation
        snapshot = MagicMock()
        snapshot.snapshot_id = "snapshot-123"
        merge_service._create_snapshot = AsyncMock(return_value=snapshot)
        
        # Act
        result = await merge_service.execute_merge(merge_id, transform_id)
        
        # Assert
        assert result["status"] == "success"
        assert "metrics" in result
        assert result["metrics"]["nodes_merged"] == 3
        assert result["metrics"]["relationships_merged"] == 2
        assert "snapshot_id" in result["metrics"]
        assert result["metrics"]["snapshot_id"] == "snapshot-123"
        
        # Verify mock calls
        mock_validate_merge.assert_called_once_with(merge_id, transform_id, None, None)
        mock_extract_staging_graph.assert_called_once_with(merge_service.storage, transform_id)
        mock_map_production_entities.assert_called_once_with(merge_service.production_storage, valid_graph)
        mock_start_stage.assert_called_once_with(merge_id, MergeStage.MERGE, merge_service.progress_tracker)
        mock_complete_merge_stage.assert_called_once()
        mock_complete_merge.assert_called_once_with(merge_id, merge_service.progress_tracker)
        mock_fail_merge.assert_not_called()
    
    @pytest.mark.asyncio
    @patch("app.services.merge.service.MergeService.validate_merge")
    async def test_execute_merge_with_invalid_validation(self, mock_validate_merge, merge_service):
        """Test execute_merge method with invalid validation"""
        # Arrange
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        
        # Mock validation result with critical issues
        mock_validate_merge.return_value = ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(
                    type=ValidationIssueType.UNRESOLVED_CONFLICTS,
                    message="Found 2 unresolved conflicts",
                    affected_ids=["conflict1", "conflict2"],
                    severity=ValidationSeverity.CRITICAL,
                    metadata={"total_unresolved": 2}
                )
            ],
            critical_count=1,
            warning_count=0,
            info_count=0,
            total_nodes=3,
            total_edges=2,
            validation_time_ms=100.0,
            metadata={}
        )
        
        # Act
        result = await merge_service.execute_merge(merge_id, transform_id)
        
        # Assert
        assert result["status"] == "failed"
        assert result["reason"] == "validation_failed"
        assert "validation_result" in result
        
        # Verify mock calls - updated to match the actual call without validation_service
        mock_validate_merge.assert_called_once_with(merge_id, transform_id, None, None)
        # Note: start_merge_stage is not called for invalid validation
    
    @pytest.mark.asyncio
    @patch("app.services.merge.service.extract_staging_graph")
    @patch("app.services.merge.service.map_production_entities")
    @patch("app.services.merge.service.start_stage")
    @patch("app.services.merge.service.complete_merge_stage")
    @patch("app.services.merge.service.complete_merge")
    @patch("app.services.merge.service.fail_merge")
    @patch("app.services.merge.service.detect_merge_conflicts")
    async def test_execute_merge_with_skip_validation(self, mock_detect_conflicts, mock_fail_merge, mock_complete_merge, mock_complete_merge_stage, mock_start_stage, mock_map_production_entities, mock_extract_staging_graph, merge_service, valid_graph):
        """Test execute_merge method with skip_validation=True"""
        # Arrange
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        
        # Mock extract_staging_graph to return an awaitable that returns valid_graph
        async def mock_extract(storage, transform_id):
            return valid_graph
        mock_extract_staging_graph.side_effect = mock_extract
        
        # Mock map_production_entities to return an awaitable that returns entity_mapping
        entity_mapping = MagicMock()
        entity_mapping.matches = {
            "node1": MagicMock(production_matches=["prod-node1"]),
            "node2": MagicMock(production_matches=["prod-node2"]),
            "node3": MagicMock(production_matches=["prod-node3"])
        }
        async def mock_map(storage, graph):
            return entity_mapping
        mock_map_production_entities.side_effect = mock_map
        
        # Mock detect_merge_conflicts to do nothing
        async def mock_detect(merge_id, graph, entity_mapping, storage, progress_tracker):
            return None
        mock_detect_conflicts.side_effect = mock_detect
        
        # Mock start_stage, complete_merge_stage, complete_merge, and fail_merge
        async def mock_start(merge_id, stage, progress_tracker):
            return None
        mock_start_stage.side_effect = mock_start
        
        async def mock_complete_stage(merge_id, stage, progress_tracker, metadata=None):
            return None
        mock_complete_merge_stage.side_effect = mock_complete_stage
        
        async def mock_complete(merge_id, progress_tracker):
            return None
        mock_complete_merge.side_effect = mock_complete
        
        async def mock_fail(merge_id, error_message, progress_tracker):
            return None
        mock_fail_merge.side_effect = mock_fail
        
        # Mock storage methods
        merge_service.production_storage.update_node = AsyncMock()
        merge_service.production_storage.create_node = AsyncMock()
        merge_service.production_storage.get_relationships_between = AsyncMock(return_value=[])
        merge_service.production_storage.create_relationship = AsyncMock()
        merge_service.production_storage.get_relationship = AsyncMock(return_value=None)
        merge_service.production_storage.update_relationship = AsyncMock()
        
        # Mock get_node_by_id to return a proper Node object
        async def mock_get_node_by_id(node_id):
            from app.schemas.graph import Node
            return Node(
                id=node_id,
                label="Person",
                type="Person",
                properties={"name": f"Person {node_id}", "age": 30}
            )
        merge_service.production_storage.get_node_by_id = AsyncMock(side_effect=mock_get_node_by_id)
        
        # Mock get_conflicts to return empty conflicts
        merge_service.get_conflicts = AsyncMock(return_value=([], 0))
        
        # Mock transaction manager
        class MockTransactionManager:
            async def __aenter__(self):
                return "transaction-123"
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        class MockTransaction:
            def __init__(self):
                self.transaction_id = "transaction-123"
            
            async def begin_transaction(self, merge_id: str) -> str:
                return self.transaction_id
                
            async def commit_transaction(self, transaction_id: str) -> bool:
                return True
                
            async def rollback_transaction(self, transaction_id: str) -> bool:
                return True
                
            def transaction(self):
                return MockTransactionManager()

        transaction_manager = MockTransaction()
        merge_service._get_transaction_manager = MagicMock(return_value=transaction_manager)
        
        # Mock snapshot creation
        snapshot = MagicMock()
        snapshot.snapshot_id = "snapshot-123"
        merge_service._create_snapshot = AsyncMock(return_value=snapshot)
        
        # Act
        result = await merge_service.execute_merge(merge_id, transform_id, skip_validation=True)
        
        # Assert
        assert result["status"] == "success"
        assert "metrics" in result
        assert result["metrics"]["nodes_merged"] == 3
        assert result["metrics"]["relationships_merged"] == 2
        assert "snapshot_id" in result["metrics"]
        assert result["metrics"]["snapshot_id"] == "snapshot-123"
        
        # Verify mock calls
        mock_extract_staging_graph.assert_called_once_with(merge_service.storage, transform_id)
        mock_map_production_entities.assert_called_once_with(merge_service.production_storage, valid_graph)
        mock_start_stage.assert_called_once_with(merge_id, MergeStage.MERGE, merge_service.progress_tracker)
        mock_complete_merge_stage.assert_called_once()
        mock_complete_merge.assert_called_once_with(merge_id, merge_service.progress_tracker)
        mock_fail_merge.assert_not_called() 