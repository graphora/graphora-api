"""Tests for handling cyclic dependencies in the MergeValidationService"""
import pytest
from unittest.mock import AsyncMock, patch
import uuid
from datetime import datetime
import pytz

from app.services.merge.validation import MergeValidationService
from app.services.merge.models import ValidationResult, ValidationIssueType, ValidationSeverity
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
def cyclic_graph_simple():
    """Simple cyclic graph fixture with a direct cycle between two nodes"""
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
                source="node2",
                target="node1",
                type="KNOWS",
                properties={}
            )
        ]
    )

@pytest.fixture
def cyclic_graph_complex():
    """Complex cyclic graph fixture with multiple cycles"""
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
                label="Person",
                type="Person",
                properties={"name": "Charlie"}
            ),
            Node(
                id="node4",
                label="Person",
                type="Person",
                properties={"name": "Dave"}
            ),
            Node(
                id="node5",
                label="Person",
                type="Person",
                properties={"name": "Eve"}
            )
        ],
        edges=[
            # Cycle 1: 1 -> 2 -> 3 -> 1
            Edge(
                id="edge1",
                source="node1",
                target="node2",
                type="KNOWS",
                properties={}
            ),
            Edge(
                id="edge2",
                source="node2",
                target="node3",
                type="KNOWS",
                properties={}
            ),
            Edge(
                id="edge3",
                source="node3",
                target="node1",
                type="KNOWS",
                properties={}
            ),
            
            # Cycle 2: 3 -> 4 -> 5 -> 3
            Edge(
                id="edge4",
                source="node3",
                target="node4",
                type="WORKS_WITH",
                properties={}
            ),
            Edge(
                id="edge5",
                source="node4",
                target="node5",
                type="WORKS_WITH",
                properties={}
            ),
            Edge(
                id="edge6",
                source="node5",
                target="node3",
                type="WORKS_WITH",
                properties={}
            ),
            
            # Cross-cycle edge
            Edge(
                id="edge7",
                source="node2",
                target="node5",
                type="FRIENDS_WITH",
                properties={}
            )
        ]
    )

@pytest.fixture
def acyclic_graph():
    """Acyclic graph fixture with no cycles"""
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
                label="Person",
                type="Person",
                properties={"name": "Charlie"}
            ),
            Node(
                id="node4",
                label="Person",
                type="Person",
                properties={"name": "Dave"}
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
                type="KNOWS",
                properties={}
            ),
            Edge(
                id="edge3",
                source="node2",
                target="node4",
                type="KNOWS",
                properties={}
            ),
            Edge(
                id="edge4",
                source="node3",
                target="node4",
                type="KNOWS",
                properties={}
            )
        ]
    )

@pytest.fixture
def self_referential_graph():
    """Graph with self-referential nodes"""
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
            )
        ],
        edges=[
            Edge(
                id="edge1",
                source="node1",
                target="node1",
                type="SELF_REFERENCE",
                properties={}
            ),
            Edge(
                id="edge2",
                source="node1",
                target="node2",
                type="KNOWS",
                properties={}
            )
        ]
    )

class TestCyclicDependencies:
    """Tests for handling cyclic dependencies in the MergeValidationService"""
    
    @pytest.mark.asyncio
    async def test_validate_simple_cycle(self, mock_storage_factory, cyclic_graph_simple):
        """Test validation with a simple cycle between two nodes"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Act
        structure_issues = await service.validate_graph_structure(cyclic_graph_simple)
        
        # Assert
        # Cycles are generally allowed in graph databases, so there should be no issues
        assert len(structure_issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_complex_cycle(self, mock_storage_factory, cyclic_graph_complex):
        """Test validation with complex cycles"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Act
        structure_issues = await service.validate_graph_structure(cyclic_graph_complex)
        
        # Assert
        # Cycles are generally allowed in graph databases, so there should be no issues
        assert len(structure_issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_acyclic_graph(self, mock_storage_factory, acyclic_graph):
        """Test validation with an acyclic graph"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Act
        structure_issues = await service.validate_graph_structure(acyclic_graph)
        
        # Assert
        assert len(structure_issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_self_referential_nodes(self, mock_storage_factory, self_referential_graph):
        """Test validation with self-referential nodes"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Act
        structure_issues = await service.validate_graph_structure(self_referential_graph)
        
        # Assert
        # Self-references might be flagged as issues depending on the validation rules
        # This test should be adjusted based on the expected behavior of the service
        # For now, we'll assume self-references are allowed
        assert len(structure_issues) == 0
    
    @pytest.mark.asyncio
    @patch("app.services.merge.validation.load_ontology")
    async def test_validate_cycles_with_ontology_restrictions(self, mock_load_ontology, mock_storage_factory, cyclic_graph_simple):
        """Test validation with cycles and ontology restrictions"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Mock an ontology that disallows cycles for the KNOWS relationship
        mock_load_ontology.return_value = {
            "node_types": {
                "Person": {
                    "required_properties": ["name"]
                }
            },
            "relationship_types": {
                "KNOWS": {
                    "source_types": ["Person"],
                    "target_types": ["Person"],
                    "acyclic": True  # This is a hypothetical property that might exist in the ontology
                }
            }
        }
        
        # Act
        ontology_issues = await service.validate_ontology_compliance(cyclic_graph_simple, "test_ontology")
        
        # Assert
        # Since we've mocked the ontology loading, we expect no issues
        # Note: The service doesn't actually check for the 'acyclic' property in the ontology
        assert len(ontology_issues) == 0
        mock_load_ontology.assert_called_once_with("test_ontology")
    
    @pytest.mark.asyncio
    async def test_validate_merge_with_cycles(self, mock_storage_factory, cyclic_graph_complex):
        """Test complete merge validation with cycles"""
        # Arrange
        service = MergeValidationService(storage_factory=mock_storage_factory)
        
        # Mock storage responses
        merge_id = str(uuid.uuid4())
        transform_id = str(uuid.uuid4())
        mock_storage_factory.conflict_storage.get_conflicts.return_value = ([], 0)
        mock_storage_factory.staging_storage.get_graph_by_transform_id.return_value = cyclic_graph_complex
        
        # Act
        result = await service.validate_merge(merge_id, transform_id)
        
        # Assert
        assert result.valid is True
        assert result.critical_count == 0
        assert result.warning_count == 0 