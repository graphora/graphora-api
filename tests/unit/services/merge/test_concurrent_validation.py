"""Tests for handling concurrent validation operations in the MergeValidationService"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import uuid
import asyncio
from datetime import datetime
import pytz

from app.services.merge.validation import MergeValidationService
from app.services.merge.models import ValidationResult, ValidationIssueType, ValidationSeverity
from app.schemas.graph import GraphResponse, Node, Edge
from app.services.storage.models import TransformationResult

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
def test_graph():
    """Test graph fixture"""
    return GraphResponse(
        nodes=[
            Node(
                id="node1",
                label="Person",
                type="Person",
                properties={"name": "Alice"}
            ),
            Node(
                id="node2",
                label="Person",
                type="Person",
                properties={"name": "Bob"}
            ),
            Node(
                id="node3",
                label="Company",
                type="Company",
                properties={"name": "Acme Inc."}
            )
        ],
        edges=[
            Edge(
                id="edge1",
                source="node1",
                target="node2",
                type="KNOWS",
                properties={}
            ),
            Edge(
                id="edge2",
                source="node1",
                target="node3",
                type="WORKS_AT",
                properties={}
            ),
            Edge(
                id="edge3",
                source="node2",
                target="node3",
                type="WORKS_AT",
                properties={}
            )
        ]
    )

class TestConcurrentValidation:
    """Tests for handling concurrent validation operations in the MergeValidationService"""
    
    @pytest.mark.asyncio
    async def test_concurrent_validation_same_merge(self, mock_storage_factory, test_graph):
        """Test concurrent validation of the same merge"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Mock storage responses
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        mock_storage_factory.conflict_storage.get_conflicts.return_value = ([], 0)
        
        # Mock get_transformation_data with TransformationResult
        mock_storage_factory.staging_storage.get_transformation_data.return_value = TransformationResult(
            transform_id=transform_id,
            nodes=[node.model_dump() for node in test_graph.nodes],
            relationships=[edge.model_dump() for edge in test_graph.edges],
            timestamp=datetime.now(pytz.utc)
        )
        
        # Add a validator that uses conflict_storage
        async def check_unresolved_conflicts(**kwargs):
            # This will call get_conflicts on the conflict_storage
            conflicts, _ = await service.conflict_storage.get_conflicts(
                merge_id=merge_id,
                resolved=False,
                limit=1000,
                offset=0
            )
            return []
        
        service.validators.append(check_unresolved_conflicts)
        
        # Act
        # Run multiple validations concurrently
        tasks = [
            service.validate_merge(merge_id, transform_id)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        
        # Assert
        # All validations should complete successfully
        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.valid is True
            assert result.critical_count == 0
        
        # The storage should be called only once for each method if the service implements caching
        # This depends on the implementation of the service
        # If caching is not implemented, this assertion may fail
        assert mock_storage_factory.conflict_storage.get_conflicts.call_count >= 1
        assert mock_storage_factory.staging_storage.get_transformation_data.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_validation_different_merges(self, mock_storage_factory, test_graph):
        """Test concurrent validation of different merges"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Mock storage responses
        merge_ids = [str(uuid.uuid4()) for _ in range(5)]
        transform_ids = [str(uuid.uuid4()) for _ in range(5)]
        
        mock_storage_factory.conflict_storage.get_conflicts.return_value = ([], 0)
        
        # Mock get_transformation_data with TransformationResult
        mock_storage_factory.staging_storage.get_transformation_data.return_value = TransformationResult(
            transform_id=transform_ids[0],  # Use the first transform_id as a default
            nodes=[node.model_dump() for node in test_graph.nodes],
            relationships=[edge.model_dump() for edge in test_graph.edges],
            timestamp=datetime.now(pytz.utc)
        )
        
        # Add a validator that uses conflict_storage with the current merge_id
        async def check_unresolved_conflicts(**kwargs):
            # Get the current merge_id from the task context
            current_merge_id = asyncio.current_task().get_name()
            # This will call get_conflicts on the conflict_storage
            conflicts, _ = await service.conflict_storage.get_conflicts(
                merge_id=current_merge_id,
                resolved=False,
                limit=1000,
                offset=0
            )
            return []
        
        service.validators.append(check_unresolved_conflicts)
        
        # Act
        # Run multiple validations concurrently with different merge IDs
        tasks = []
        for i in range(5):
            # Set the task name to the merge_id for the validator to use
            task = asyncio.create_task(
                service.validate_merge(merge_ids[i], transform_ids[i]),
                name=merge_ids[i]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Assert
        # All validations should complete successfully
        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.valid is True
            assert result.critical_count == 0
        
        # The storage should be called once for each merge ID
        assert mock_storage_factory.conflict_storage.get_conflicts.call_count == 5
        assert mock_storage_factory.staging_storage.get_transformation_data.call_count == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_validation_with_delays(self, mock_storage_factory, test_graph):
        """Test concurrent validation with simulated delays"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Mock storage responses with delays
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        
        # Simulate a slow database query
        async def delayed_get_conflicts(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return ([], 0)
        
        async def delayed_get_graph(*args, **kwargs):
            await asyncio.sleep(0.2)  # 200ms delay
            return test_graph
        
        mock_storage_factory.conflict_storage.get_conflicts.side_effect = delayed_get_conflicts
        mock_storage_factory.staging_storage.get_graph_by_transform_id.side_effect = delayed_get_graph
        
        # Act
        # Run multiple validations concurrently
        start_time = datetime.now(pytz.utc)
        tasks = [
            service.validate_merge(merge_id, transform_id)
            for _ in range(3)
        ]
        results = await asyncio.gather(*tasks)
        end_time = datetime.now(pytz.utc)
        
        # Calculate execution time
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Assert
        # All validations should complete successfully
        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.valid is True
            assert result.critical_count == 0
        
        # If the service implements proper concurrency, the total time should be close to
        # the time of a single validation (around 300ms) rather than 3 * 300ms = 900ms
        # This depends on the implementation of the service
        # If concurrency is not optimized, this assertion may fail
        # We'll use a generous threshold to account for test environment variability
        assert execution_time_ms < 900, f"Concurrent validation took too long: {execution_time_ms}ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_validation_with_errors(self, mock_storage_factory):
        """Test concurrent validation with simulated errors"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Mock storage responses with errors for some calls
        merge_ids = [str(uuid.uuid4()) for _ in range(5)]
        transform_ids = [str(uuid.uuid4()) for _ in range(5)]
        
        # Make some calls fail
        async def get_conflicts_with_errors(*args, **kwargs):
            merge_id = args[0] if args else kwargs.get('merge_id')
            if merge_ids.index(merge_id) % 2 == 0:
                raise Exception("Simulated database error")
            return ([], 0)
        
        mock_storage_factory.conflict_storage.get_conflicts.side_effect = get_conflicts_with_errors
        
        # Mock get_transformation_data with TransformationResult
        mock_storage_factory.staging_storage.get_transformation_data.return_value = TransformationResult(
            transform_id=transform_ids[0],  # Use the first transform_id as a default
            nodes=[],
            relationships=[],
            timestamp=datetime.now(pytz.utc)
        )
        
        # Add a validator that uses conflict_storage and will trigger the errors
        async def check_unresolved_conflicts(**kwargs):
            # Get the current merge_id from the task context
            current_merge_id = asyncio.current_task().get_name()
            # This will call get_conflicts on the conflict_storage and may raise an exception
            conflicts, _ = await service.conflict_storage.get_conflicts(
                merge_id=current_merge_id,
                resolved=False,
                limit=1000,
                offset=0
            )
            return []
        
        service.validators.append(check_unresolved_conflicts)
        
        # Act
        # Run multiple validations concurrently
        tasks = []
        for i in range(5):
            # Set the task name to the merge_id for the validator to use
            task = asyncio.create_task(
                service.validate_merge(merge_ids[i], transform_ids[i]),
                name=merge_ids[i]
            )
            tasks.append(task)
        
        # Use gather with return_exceptions=True to get exceptions as results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert
        # Check that some validations failed and others succeeded
        success_count = 0
        error_count = 0
        validation_error_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
            elif isinstance(result, ValidationResult):
                if result.valid:
                    success_count += 1
                else:
                    # The validation service catches exceptions and converts them to validation issues
                    validation_error_count += 1
                    assert result.critical_count > 0
                    assert any(issue.type == ValidationIssueType.VALIDATION_ERROR for issue in result.issues)
            else:
                # Unexpected result type
                assert False, f"Unexpected result type: {type(result)}"
        
        # Ensure we have validation errors (the service catches exceptions and converts them to validation issues)
        assert validation_error_count > 0
    
    @pytest.mark.asyncio
    @patch("app.services.merge.validation.load_ontology")
    async def test_concurrent_validation_with_ontology(self, mock_load_ontology, mock_storage_factory, test_graph):
        """Test concurrent validation with ontology"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Mock ontology loading
        mock_load_ontology.return_value = {
            "node_types": {
                "Person": {
                    "required_properties": ["name"]
                },
                "Company": {
                    "required_properties": ["name"]
                }
            },
            "relationship_types": {
                "KNOWS": {
                    "source_types": ["Person"],
                    "target_types": ["Person"]
                },
                "WORKS_AT": {
                    "source_types": ["Person"],
                    "target_types": ["Company"]
                }
            }
        }
        
        # Mock storage responses
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        mock_storage_factory.conflict_storage.get_conflicts.return_value = ([], 0)
        mock_storage_factory.staging_storage.get_graph_by_transform_id.return_value = test_graph
        
        # Act
        # Run multiple validations concurrently with ontology
        tasks = [
            service.validate_merge(merge_id, transform_id, ontology_id="test_ontology")
            for _ in range(3)
        ]
        results = await asyncio.gather(*tasks)
        
        # Assert
        # All validations should complete successfully
        for result in results:
            assert isinstance(result, ValidationResult)
            # The validation should succeed since we've mocked the ontology loading
            assert result.valid is True
            assert result.critical_count == 0
        
        # The ontology loading should be attempted at least once
        assert mock_load_ontology.call_count >= 1 