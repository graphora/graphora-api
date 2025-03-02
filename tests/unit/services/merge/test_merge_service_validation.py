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
    async def test_execute_merge_with_valid_validation(self, mock_validate_merge, merge_service, valid_graph):
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
        
        # Mock get_graph_by_transform_id
        merge_service.storage.get_graph_by_transform_id.return_value = valid_graph
        
        # Act
        result = await merge_service.execute_merge(merge_id, transform_id)
        
        # Assert
        assert result["merge_id"] == merge_id
        assert result["status"] == "completed"
        assert result["nodes_merged"] == 3
        assert result["edges_merged"] == 2
        
        # Verify mock calls
        mock_validate_merge.assert_called_once_with(merge_id, transform_id, None, None)
        merge_service.progress_tracker.start_stage.assert_called_once_with(merge_id, MergeStage.MERGE)
        merge_service.progress_tracker.complete_stage.assert_called_once_with(merge_id, MergeStage.MERGE)
        merge_service.progress_tracker.fail_stage.assert_not_called()
    
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
        
        # Act & Assert
        with pytest.raises(ValueError, match="Merge validation failed with 1 critical issues"):
            await merge_service.execute_merge(merge_id, transform_id)
        
        # Verify mock calls
        mock_validate_merge.assert_called_once_with(merge_id, transform_id, None, None)
        merge_service.progress_tracker.start_stage.assert_called_once_with(merge_id, MergeStage.MERGE)
        
        # The fail_stage is called twice in the implementation:
        # 1. When validation fails
        # 2. When the exception is caught and logged
        assert merge_service.progress_tracker.fail_stage.call_count == 2
        merge_service.progress_tracker.complete_stage.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_execute_merge_with_skip_validation(self, merge_service, valid_graph):
        """Test execute_merge method with skip_validation=True"""
        # Arrange
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        
        # Mock get_graph_by_transform_id
        merge_service.storage.get_graph_by_transform_id.return_value = valid_graph
        
        # Act
        result = await merge_service.execute_merge(merge_id, transform_id, skip_validation=True)
        
        # Assert
        assert result["merge_id"] == merge_id
        assert result["status"] == "completed"
        assert result["nodes_merged"] == 3
        assert result["edges_merged"] == 2
        
        # Verify mock calls
        merge_service.progress_tracker.start_stage.assert_called_once_with(merge_id, MergeStage.MERGE)
        merge_service.progress_tracker.complete_stage.assert_called_once_with(merge_id, MergeStage.MERGE)
        merge_service.progress_tracker.fail_stage.assert_not_called() 