"""Integration tests for the MergeValidationService"""
import pytest
import uuid
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import os

from app.services.merge.validation import MergeValidationService
from app.services.merge.models import ValidationResult, ValidationIssue, ValidationIssueType, ValidationSeverity
from app.schemas.conflicts import ConflictStatus, ConflictType, ConflictSeverity, Conflict
from app.schemas.graph import GraphResponse, Node, Edge
from app.services.storage.neo4j import Neo4jStorage
from app.services.storage.conflicts import ConflictStorage

# Skip these tests by default
pytestmark = pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Integration tests are skipped by default"
)

@pytest.fixture
async def storage_factory():
    """Mock storage factory fixture"""
    # Create mock storages
    staging_storage = AsyncMock()
    prod_storage = AsyncMock()
    conflict_storage = AsyncMock()
    
    def factory(is_staging=False, is_conflict_storage=False):
        if is_conflict_storage:
            return conflict_storage
        elif is_staging:
            return staging_storage
        else:
            return prod_storage
    
    # Add references to the storages for assertions
    factory.staging_storage = staging_storage
    factory.production_storage = prod_storage
    factory.conflict_storage = conflict_storage
    
    return factory

@pytest.fixture
async def test_graph():
    """Test graph fixture"""
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
            ),
            Node(
                id="node4",
                label="Orphan",
                type="Orphan",
                properties={"name": "Orphaned Node"}
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
async def test_ontology():
    """Test ontology fixture"""
    return {
        "node_types": {
            "Person": {
                "required_properties": ["name", "age"]
            },
            "Company": {
                "required_properties": ["name", "founded"]
            }
        },
        "relationship_types": {
            "WORKS_AT": {
                "source_types": ["Person"],
                "target_types": ["Company"],
                "required_properties": ["since"]
            }
        }
    }

@pytest.fixture
async def setup_test_data(storage_factory, test_graph):
    """Setup test data with mocked storage"""
    merge_id = str(uuid.uuid4())
    transform_id = str(uuid.uuid4())
    
    # Configure mock storage responses
    staging_storage = storage_factory.staging_storage
    conflict_storage = storage_factory.conflict_storage
    
    # Mock get_graph_by_transform_id to return our test graph
    staging_storage.get_graph_by_transform_id.return_value = test_graph
    
    # Create mock conflicts
    conflicts = [
        Conflict(
            id=str(uuid.uuid4()),
            merge_id=merge_id,
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            entity_id="node1",
            entity_type="Person",
            property_name="age",
            staging_value=30,
            production_value=31,
            description="Age property conflict",
            resolved=False
        ),
        Conflict(
            id=str(uuid.uuid4()),
            merge_id=merge_id,
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.CRITICAL,
            entity_id="edge1",
            entity_type="WORKS_AT",
            description="Relationship type conflict",
            resolved=False
        )
    ]
    
    # Mock get_conflicts to return our conflicts
    conflict_storage.get_conflicts.return_value = (conflicts, len(conflicts))
    
    # Mock update_conflict to simulate conflict resolution
    async def mock_update_conflict(conflict):
        # Find the conflict in our list and update it
        for c in conflicts:
            if c.id == conflict.id:
                c.resolved = conflict.resolved
        
        # After updating, if we're checking for unresolved conflicts, return empty list
        if all(c.resolved for c in conflicts):
            conflict_storage.get_conflicts.return_value = ([], 0)
    
    conflict_storage.update_conflict.side_effect = mock_update_conflict
    
    return {
        "merge_id": merge_id,
        "transform_id": transform_id,
        "conflicts": conflicts
    }

@pytest.mark.asyncio
async def test_validate_conflict_resolution(storage_factory, setup_test_data):
    """Test validation of conflict resolution"""
    # Arrange
    merge_id = setup_test_data["merge_id"]
    validation_service = MergeValidationService(storage_factory=storage_factory)
    
    # Act
    issues = await validation_service.validate_conflict_resolution(merge_id)
    
    # Assert
    assert len(issues) == 1
    assert issues[0].type == ValidationIssueType.UNRESOLVED_CONFLICTS
    assert issues[0].severity == ValidationSeverity.CRITICAL
    assert len(issues[0].affected_ids) == 2
    
    # Resolve conflicts
    conflict_storage = storage_factory.conflict_storage
    conflicts, _ = await conflict_storage.get_conflicts(merge_id=merge_id)
    
    for conflict in conflicts:
        # Update conflict using model_copy instead of direct attribute assignment
        updated_conflict = conflict.model_copy(update={"resolved": True})
        # Store the updated conflict
        await conflict_storage.update_conflict(updated_conflict)
    
    # Act again
    issues = await validation_service.validate_conflict_resolution(merge_id)
    
    # Assert
    assert len(issues) == 0

@pytest.mark.asyncio
async def test_validate_graph_structure(storage_factory, setup_test_data, test_graph):
    """Test validation of graph structure"""
    # Arrange
    transform_id = setup_test_data["transform_id"]
    validation_service = MergeValidationService(storage_factory=storage_factory)
    
    # Get graph from storage
    staging_storage = storage_factory.staging_storage
    graph = await staging_storage.get_graph_by_transform_id(transform_id)
    
    # Act
    issues = await validation_service.validate_graph_structure(graph)
    
    # Assert
    assert len(issues) == 0
    
    # Create a new graph with an invalid relationship
    modified_graph = GraphResponse(
        nodes=graph.nodes.copy(),
        edges=graph.edges.copy()
    )
    
    # Add invalid relationship
    invalid_edge = Edge(
        id="invalid_edge",
        source="node1",
        target="nonexistent_node",
        type="INVALID_REL",
        properties={}
    )
    
    # Add to graph
    modified_graph.edges.append(invalid_edge)
    
    # Update the mock to return the modified graph for subsequent calls
    staging_storage.get_graph_by_transform_id.return_value = modified_graph
    
    # Act again with the modified graph
    issues = await validation_service.validate_graph_structure(modified_graph)
    
    # Assert
    assert len(issues) == 1
    assert issues[0].type == ValidationIssueType.INVALID_RELATIONSHIP_REFERENCE
    assert issues[0].severity == ValidationSeverity.CRITICAL
    assert "invalid_edge" in issues[0].affected_ids

@pytest.mark.asyncio
@pytest.mark.parametrize("with_ontology", [True, False])
@patch("app.services.merge.validation.load_ontology")
async def test_validate_merge_complete(mock_load_ontology, storage_factory, setup_test_data, with_ontology, test_ontology):
    """Test complete merge validation with and without ontology"""
    # Arrange
    merge_id = setup_test_data["merge_id"]
    transform_id = setup_test_data["transform_id"]
    validation_service = MergeValidationService(storage_factory=storage_factory)
    
    # Mock the load_ontology function to return our test ontology
    mock_load_ontology.return_value = test_ontology
    
    # Mock the validate_ontology_compliance and validate_required_properties methods
    # to avoid actual ontology loading
    original_validate_ontology = validation_service.validate_ontology_compliance
    original_validate_properties = validation_service.validate_required_properties
    original_validate_conflict_resolution = validation_service.validate_conflict_resolution
    
    async def mock_validate_ontology_compliance(graph, ontology_id):
        if with_ontology:
            # Return a warning for the orphaned node
            return [
                ValidationIssue(
                    type=ValidationIssueType.ORPHANED_NODE,
                    message="Node 'node4' is orphaned",
                    affected_ids=["node4"],
                    severity=ValidationSeverity.WARNING,
                    metadata={"node_type": "Orphan"}
                )
            ]
        return []
        
    async def mock_validate_required_properties(graph, ontology_id):
        return []
    
    async def mock_validate_conflict_resolution(merge_id):
        # Initially return unresolved conflicts
        conflicts, _ = await storage_factory.conflict_storage.get_conflicts(merge_id=merge_id)
        if conflicts and any(not c.resolved for c in conflicts):
            return [
                ValidationIssue(
                    type=ValidationIssueType.UNRESOLVED_CONFLICTS,
                    message=f"Found {len(conflicts)} unresolved conflicts",
                    affected_ids=[c.id for c in conflicts],
                    severity=ValidationSeverity.CRITICAL,
                    metadata={"total_unresolved": len(conflicts)}
                )
            ]
        return []
    
    # Create a custom validator that calls validate_conflict_resolution
    async def custom_validator(staging_graph, prod_storage, ontology, allowed_orphan_types):
        return await mock_validate_conflict_resolution(merge_id)
    
    # Create a custom validator for orphaned nodes
    async def orphaned_nodes_validator(staging_graph, prod_storage, ontology, allowed_orphan_types):
        return await mock_validate_ontology_compliance(staging_graph, ontology)
    
    # Apply the patches
    validation_service.validate_ontology_compliance = mock_validate_ontology_compliance
    validation_service.validate_required_properties = mock_validate_required_properties
    validation_service.validate_conflict_resolution = mock_validate_conflict_resolution
    
    # Add our custom validators to the validators list
    validation_service.validators.append(custom_validator)
    validation_service.validators.append(orphaned_nodes_validator)
    
    try:
        # Act
        result = await validation_service.validate_merge(
            merge_id, transform_id, ontology_id="test_ontology" if with_ontology else None
        )
        
        # Assert
        assert isinstance(result, ValidationResult)
        assert result.valid is False  # Should be invalid due to unresolved conflicts
        assert result.critical_count > 0
        
        # Resolve conflicts
        conflict_storage = storage_factory.conflict_storage
        conflicts, _ = await conflict_storage.get_conflicts(merge_id=merge_id)
        
        for conflict in conflicts:
            # Update conflict using model_copy instead of direct attribute assignment
            updated_conflict = conflict.model_copy(update={"resolved": True})
            # Store the updated conflict
            await conflict_storage.update_conflict(updated_conflict)
        
        # Act again
        result = await validation_service.validate_merge(
            merge_id, transform_id, ontology_id="test_ontology" if with_ontology else None
        )
        
        # Assert
        assert result.valid is True
        assert result.critical_count == 0
        
        # Check for orphaned node warning
        if with_ontology:
            # Should have warning for orphaned node
            assert result.warning_count > 0
            orphan_issues = [i for i in result.issues if i.type == ValidationIssueType.ORPHANED_NODE]
            assert len(orphan_issues) > 0
            assert "node4" in orphan_issues[0].affected_ids
    finally:
        # Restore original methods
        validation_service.validate_ontology_compliance = original_validate_ontology
        validation_service.validate_required_properties = original_validate_properties
        validation_service.validate_conflict_resolution = original_validate_conflict_resolution
        # Remove our custom validators
        if custom_validator in validation_service.validators:
            validation_service.validators.remove(custom_validator)
        if orphaned_nodes_validator in validation_service.validators:
            validation_service.validators.remove(orphaned_nodes_validator) 