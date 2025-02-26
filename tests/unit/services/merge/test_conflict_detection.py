"""
Unit tests for conflict detection logic.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.merge.service import MergeService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionStrategy
from app.services.storage.models import Node as StorageNode, Edge as StorageEdge
from app.schemas.graph import GraphResponse, Node, Edge
from tests.unit.services.storage.test_graph_storage import MockGraphStorage

# Mock BAML responses
class MockPropertyConflictAnalysis:
    def __init__(self, staging_value, production_value):
        self.recommended_strategy = ResolutionStrategy.KEEP_STAGING
        self.confidence = 0.8
        self.explanation = "Test explanation"
        self.can_auto_resolve = True
        self.potential_risks = []

class MockEntitySimilarityAnalysis:
    def __init__(self):
        self.similarity_score = 0.9
        self.matching_properties = ["name"]
        self.mismatched_properties = []
        self.semantic_similarity = 0.9
        self.potential_merge_impact = "Low"
        self.reasoning = "Test reasoning"

@pytest.fixture
def mock_baml():
    """Mock BAML client responses"""
    with patch("app.services.merge.conflict.b") as mock_b:
        # Mock property conflict analysis
        def analyze_property_conflict(**kwargs):
            return MockPropertyConflictAnalysis(
                kwargs.get("staging_value"),
                kwargs.get("production_value")
            )
        mock_b.AnalyzePropertyConflict = analyze_property_conflict
        
        # Mock entity similarity analysis
        def analyze_entity_similarity(**kwargs):
            return MockEntitySimilarityAnalysis()
        mock_b.AnalyzeEntitySimilarity = analyze_entity_similarity
        
        yield mock_b

@pytest.fixture
def merge_service_with_conflicts(mock_baml):
    """Create service with mock dependencies and test data"""
    # Create mock storage
    staging_storage = MockGraphStorage()
    prod_storage = MockGraphStorage()
    
    # Create service with mock storage
    service = MergeService(staging_storage=staging_storage, prod_storage=prod_storage)
    
    # Add test data with known conflicts
    
    # Property value conflict
    staging_storage.add_test_node(
        StorageNode(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30, "transform_id": "test_id"})
    )
    prod_storage.add_test_node(
        StorageNode(id="p1", label="Person", type="Person", properties={"name": "Alice", "age": 32})
    )
    
    # Missing property conflict 
    staging_storage.add_test_node(
        StorageNode(id="s2", label="Person", type="Person", properties={"name": "Bob", "department": "Engineering", "transform_id": "test_id"})
    )
    prod_storage.add_test_node(
        StorageNode(id="p2", label="Person", type="Person", properties={"name": "Bob", "title": "Developer"})
    )
    
    # Relationship conflict
    staging_storage.add_test_node(
        StorageNode(id="s3", label="Department", type="Department", properties={"name": "Engineering", "transform_id": "test_id"})
    )
    prod_storage.add_test_node(
        StorageNode(id="p3", label="Department", type="Department", properties={"name": "Engineering"})
    )
    
    staging_storage.add_test_relationship(
        StorageEdge(id="sr1", source="s2", target="s3", type="WORKS_IN", properties={})
    )
    prod_storage.add_test_relationship(
        StorageEdge(id="pr1", source="p2", target="p3", type="BELONGS_TO", properties={})
    )
    
    # Entity matching conflict (multiple matches)
    staging_storage.add_test_node(
        StorageNode(id="s4", label="Project", type="Project", properties={"name": "API", "transform_id": "test_id"})
    )
    prod_storage.add_test_node(
        StorageNode(id="p4a", label="Project", type="Project", properties={"name": "API", "version": "1.0"})
    )
    prod_storage.add_test_node(
        StorageNode(id="p4b", label="Project", type="Project", properties={"name": "API", "version": "2.0"})
    )
    
    return service

class TestConflictDetection:
    """Test suite for conflict detection functionality"""
    
    @pytest.mark.asyncio
    async def test_property_value_conflict_detection(self, merge_service_with_conflicts):
        """Test detection of property value conflicts"""
        # Arrange
        staging_graph = GraphResponse(
            nodes=[{"id": "s1", "label": "Person", "type": "Person", "properties": {"name": "Alice", "age": 30, "transform_id": "test_id"}}],
            edges=[],
            total_nodes=1,
            total_edges=0
        )
        
        production_entity_mapping = {"s1": ["p1"]}
        
        # Act
        conflicts = await merge_service_with_conflicts.detect_property_conflicts_for_graph(
            staging_graph, production_entity_mapping
        )
        
        # Assert
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.PROPERTY
        assert conflicts[0].severity == ConflictSeverity.MINOR
        assert "age" in conflicts[0].description
        assert conflicts[0].staging_value == 30
        assert conflicts[0].production_value == 32
    
    @pytest.mark.asyncio
    async def test_missing_property_conflict_detection(self, merge_service_with_conflicts):
        """Test detection of missing property conflicts"""
        # Arrange
        staging_graph = GraphResponse(
            nodes=[{"id": "s2", "label": "Person", "type": "Person", "properties": {"name": "Bob", "department": "Engineering", "transform_id": "test_id"}}],
            edges=[],
            total_nodes=1,
            total_edges=0
        )
        
        production_entity_mapping = {"s2": ["p2"]}
        
        # Act
        conflicts = await merge_service_with_conflicts.detect_property_conflicts_for_graph(
            staging_graph, production_entity_mapping
        )
        
        # Assert
        assert len(conflicts) == 2  # One missing in each direction
        missing_props = [
            conflict for conflict in conflicts 
            if conflict.conflict_type == ConflictType.PROPERTY
        ]
        assert len(missing_props) == 2
        assert any("department" in c.description for c in missing_props)
        assert any("title" in c.description for c in missing_props)
    
    @pytest.mark.asyncio
    async def test_relationship_conflict_detection(self, merge_service_with_conflicts):
        """Test detection of relationship conflicts"""
        # Arrange
        staging_edge = Edge(
            id="sr1",
            source="s2",
            target="s3",
            type="WORKS_IN",
            properties={}
        )
        production_edge = Edge(
            id="pr1",
            source="p2",
            target="p3",
            type="BELONGS_TO",
            properties={}
        )
        
        # Act
        conflicts = await merge_service_with_conflicts.conflict_detection_service.detect_relationship_conflicts(
            staging_edge,
            production_edge,
            "test-merge-id"
        )
        
        # Assert
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.RELATIONSHIP_TYPE
        assert conflicts[0].severity == ConflictSeverity.MAJOR
        assert conflicts[0].staging_value == "WORKS_IN"
        assert conflicts[0].production_value == "BELONGS_TO"
    
    @pytest.mark.asyncio
    async def test_entity_matching_conflict_detection(self, merge_service_with_conflicts):
        """Test detection of entity matching conflicts"""
        # Arrange
        staging_graph = GraphResponse(
            nodes=[
                {"id": "s4", "label": "Project", "type": "Project", "properties": {"name": "API", "transform_id": "test_id"}}
            ],
            edges=[],
            total_nodes=1,
            total_edges=0
        )
        
        # Multiple matches for s4
        production_entity_mapping = {"s4": ["p4a", "p4b"]}
        
        # Act
        conflicts = await merge_service_with_conflicts.detect_entity_matching_conflicts(
            staging_graph, production_entity_mapping
        )
        
        # Assert
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.DUPLICATE_ENTITY
        assert conflicts[0].severity == ConflictSeverity.MAJOR
    
    @pytest.mark.asyncio
    async def test_edge_case_empty_graph(self, merge_service_with_conflicts):
        """Test conflict detection with empty graph"""
        # Arrange
        empty_graph = GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)
        empty_mapping = {}
        
        # Act & Assert
        conflicts = await merge_service_with_conflicts.detect_property_conflicts_for_graph(
            empty_graph, empty_mapping
        )
        assert len(conflicts) == 0
        
        conflicts = await merge_service_with_conflicts.detect_relationship_conflicts(
            empty_graph, empty_mapping
        )
        assert len(conflicts) == 0
        
        conflicts = await merge_service_with_conflicts.detect_entity_matching_conflicts(
            empty_graph, empty_mapping
        )
        assert len(conflicts) == 0
    
    @pytest.mark.asyncio
    async def test_conflict_storage_and_retrieval(self, merge_service_with_conflicts):
        """Test storing and retrieving conflicts"""
        # Arrange
        merge_id = "test-merge-unique-123"  # Use a unique merge ID to avoid conflicts
        conflict = Conflict(
            id="test-conflict-1",
            merge_id=merge_id,
            conflict_type=ConflictType.PROPERTY,
            severity=ConflictSeverity.MINOR,
            description="Test conflict",
            staging_value=30,
            production_value=32,
            staging_ids=["s1"],
            production_ids=["p1"]
        )

        # Act
        # Store and retrieve
        await merge_service_with_conflicts._store_conflicts(merge_id, [conflict])
        retrieved_conflicts, total_count = await merge_service_with_conflicts.get_conflicts(merge_id)

        # Assert
        assert len(retrieved_conflicts) == 1
        assert total_count == 1
        retrieved = retrieved_conflicts[0]
        assert retrieved.id == conflict.id
        assert retrieved.merge_id == merge_id
        assert retrieved.conflict_type == conflict.conflict_type
        assert retrieved.severity == conflict.severity
