"""Tests for custom validation rules in MergeValidationService"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional

from app.services.merge.validation import MergeValidationService
from app.services.merge.models import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationIssueType
)
from app.schemas.graph import GraphResponse, Node, Edge

class CustomValidator:
    """Custom validator class for testing custom validation rules"""
    
    @staticmethod
    async def validate_node_naming_convention(graph: GraphResponse) -> List[ValidationIssue]:
        """
        Custom validation rule to check node naming conventions
        
        Args:
            graph: The graph to validate
            
        Returns:
            List[ValidationIssue]: List of validation issues
        """
        issues: List[ValidationIssue] = []
        
        # Check that all node IDs follow the convention: type_uniqueId
        for node in graph.nodes:
            if not node.id.startswith(f"{node.type.lower()}_"):
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.VALIDATION_ERROR,
                        message=f"Node '{node.id}' does not follow naming convention: {node.type.lower()}_uniqueId",
                        affected_ids=[node.id],
                        severity=ValidationSeverity.WARNING,
                        metadata={
                            "node_type": node.type,
                            "expected_prefix": f"{node.type.lower()}_"
                        }
                    )
                )
        
        return issues
    
    @staticmethod
    async def validate_relationship_properties_consistency(graph: GraphResponse) -> List[ValidationIssue]:
        """
        Custom validation rule to check relationship properties consistency
        
        Args:
            graph: The graph to validate
            
        Returns:
            List[ValidationIssue]: List of validation issues
        """
        issues: List[ValidationIssue] = []
        
        # Group relationships by type
        relationships_by_type: Dict[str, List[Edge]] = {}
        for edge in graph.edges:
            if edge.type not in relationships_by_type:
                relationships_by_type[edge.type] = []
            relationships_by_type[edge.type].append(edge)
        
        # Check that all relationships of the same type have consistent property keys
        for rel_type, edges in relationships_by_type.items():
            if len(edges) <= 1:
                continue
                
            # Get all property keys used across relationships of this type
            all_keys = set()
            for edge in edges:
                all_keys.update(edge.properties.keys())
                
            # Check each relationship for missing properties
            for edge in edges:
                missing_keys = all_keys - set(edge.properties.keys())
                if missing_keys:
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.VALIDATION_ERROR,
                            message=f"Relationship '{edge.id}' of type '{rel_type}' is missing properties: {', '.join(missing_keys)}",
                            affected_ids=[edge.id],
                            severity=ValidationSeverity.INFO,
                            metadata={
                                "relationship_type": rel_type,
                                "missing_properties": list(missing_keys)
                            }
                        )
                    )
        
        return issues

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
    """Test graph fixture with naming convention issues"""
    return GraphResponse(
        nodes=[
            Node(
                id="person_1",  # Follows convention
                label="Person",
                type="Person",
                properties={"name": "Alice", "age": 30}
            ),
            Node(
                id="bob",  # Doesn't follow convention
                label="Person",
                type="Person",
                properties={"name": "Bob", "age": 25}
            ),
            Node(
                id="company_1",  # Follows convention
                label="Company",
                type="Company",
                properties={"name": "Acme Inc", "founded": 2010}
            )
        ],
        edges=[
            Edge(
                id="rel1",
                source="person_1",
                target="company_1",
                type="WORKS_AT",
                properties={"since": 2018, "role": "Developer"}
            ),
            Edge(
                id="rel2",
                source="bob",
                target="company_1",
                type="WORKS_AT",
                properties={"since": 2019}  # Missing 'role' property
            )
        ]
    )

class TestCustomValidation:
    """Tests for custom validation rules"""
    
    @pytest.mark.asyncio
    async def test_custom_node_naming_validation(self, mock_storage_factory, test_graph):
        """Test custom node naming convention validation"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Add custom validation method to service
        service.validate_node_naming_convention = CustomValidator.validate_node_naming_convention
        
        # Act
        issues = await service.validate_node_naming_convention(test_graph)
        
        # Assert
        assert len(issues) == 1
        assert issues[0].type == ValidationIssueType.VALIDATION_ERROR
        assert "bob" in issues[0].affected_ids
        assert "person_" in issues[0].metadata["expected_prefix"]
    
    @pytest.mark.asyncio
    async def test_custom_relationship_properties_validation(self, mock_storage_factory, test_graph):
        """Test custom relationship properties consistency validation"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Add custom validation method to service
        service.validate_relationship_properties_consistency = CustomValidator.validate_relationship_properties_consistency
        
        # Act
        issues = await service.validate_relationship_properties_consistency(test_graph)
        
        # Assert
        assert len(issues) == 1
        assert issues[0].type == ValidationIssueType.VALIDATION_ERROR
        assert "rel2" in issues[0].affected_ids
        assert "role" in issues[0].metadata["missing_properties"]
    
    @pytest.mark.asyncio
    async def test_validate_merge_with_custom_validations(self, mock_storage_factory, test_graph):
        """Test validate_merge with custom validations"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Add custom validation methods
        service.validate_node_naming_convention = CustomValidator.validate_node_naming_convention
        service.validate_relationship_properties_consistency = CustomValidator.validate_relationship_properties_consistency
        
        # Mock storage responses
        mock_storage_factory.staging_storage.get_graph_by_transform_id.return_value = test_graph
        mock_storage_factory.conflict_storage.get_conflicts.return_value = ([], 0)
        
        # Patch the original validation methods to also call custom validations
        original_validate_merge = service.validate_merge
        
        async def extended_validate_merge(merge_id, transform_id, ontology_id=None):
            # Get the original validation result
            result = await original_validate_merge(merge_id, transform_id, ontology_id)
            
            # Get the graph
            graph = await mock_storage_factory.staging_storage.get_graph_by_transform_id(transform_id)
            
            # Run custom validations
            naming_issues = await service.validate_node_naming_convention(graph)
            property_issues = await service.validate_relationship_properties_consistency(graph)
            
            # Add custom validation issues to the result
            result.issues.extend(naming_issues)
            result.issues.extend(property_issues)
            
            # Update counts
            result.warning_count += sum(1 for i in naming_issues if i.severity == ValidationSeverity.WARNING)
            result.info_count += sum(1 for i in property_issues if i.severity == ValidationSeverity.INFO)
            
            return result
        
        # Replace the validate_merge method
        service.validate_merge = extended_validate_merge
        
        # Act
        result = await service.validate_merge("merge1", "transform1")
        
        # Assert
        assert isinstance(result, ValidationResult)
        
        # Check that custom validation issues are included
        naming_issues = [i for i in result.issues if i.type == ValidationIssueType.VALIDATION_ERROR and "expected_prefix" in i.metadata]
        property_issues = [i for i in result.issues if i.type == ValidationIssueType.VALIDATION_ERROR and "missing_properties" in i.metadata]
        
        assert len(naming_issues) == 1
        assert len(property_issues) == 1
        
        # Verify that counts are updated
        assert result.warning_count >= 1  # At least one warning from naming convention
        assert result.info_count >= 1  # At least one info from property consistency 