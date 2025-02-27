"""Tests for mock data generator"""
import pytest
from tests.utils.mock_data_generator import MockDataGenerator
from app.schemas.conflicts import ConflictType

class TestMockDataGenerator:
    """Test suite for mock data generator"""
    
    @pytest.fixture
    def generator(self):
        """Create a mock data generator instance"""
        return MockDataGenerator()
    
    def test_generate_node(self, generator):
        """Test node generation"""
        # Test with specific entity type
        node = generator.generate_node(entity_type="Person")
        assert node.label == "Person"
        assert node.type == "Person"
        assert "name" in node.properties  # Required property
        assert "transform_id" in node.properties
        
        # Test without transform_id
        node = generator.generate_node(with_transform_id=False)
        assert "transform_id" not in node.properties
        
        # Test with custom transform_id
        node = generator.generate_node(transform_id="custom_id")
        assert node.properties["transform_id"] == "custom_id"
    
    def test_generate_edge(self, generator):
        """Test edge generation"""
        # Create nodes that can have a relationship
        person = generator.generate_node(entity_type="Person")
        dept = generator.generate_node(entity_type="Department")
        
        # Test with specific relationship type
        edge = generator.generate_edge(person, dept, relationship_type="WORKS_IN")
        assert edge.type == "WORKS_IN"
        assert edge.source == person.id
        assert edge.target == dept.id
        
        # Test automatic relationship type selection
        edge = generator.generate_edge(person, dept)
        assert edge.type in ["WORKS_IN", "MANAGES"]  # Valid relationships
        
        # Test with properties
        edge = generator.generate_edge(person, dept, with_properties=True)
        assert "created_at" in edge.properties
        assert "weight" in edge.properties
        
        # Test invalid relationship
        with pytest.raises(ValueError):
            generator.generate_edge(person, person)  # No valid relationship type
    
    def test_generate_graph(self, generator):
        """Test graph generation"""
        # Test with default parameters
        graph = generator.generate_graph()
        assert len(graph.nodes) == 10  # Default node count
        assert graph.total_nodes == len(graph.nodes)
        assert graph.total_edges == len(graph.edges)
        
        # Test with custom parameters
        graph = generator.generate_graph(node_count=5, edge_density=0.5)
        assert len(graph.nodes) == 5
        
        # Verify transform_id
        assert all("transform_id" in node.properties for node in graph.nodes)
        
        # Verify valid relationships
        for edge in graph.edges:
            source_node = next(n for n in graph.nodes if n.id == edge.source)
            target_node = next(n for n in graph.nodes if n.id == edge.target)
            rel_def = generator.ontology["relationships"].get(edge.type)
            assert rel_def is not None
            assert rel_def["source"] == source_node.label
            assert rel_def["target"] == target_node.label
    
    def test_generate_conflicting_graphs(self, generator):
        """Test generation of graphs with conflicts"""
        # Test with all conflict types
        staging, prod, conflicts = generator.generate_conflicting_graphs()

        # Verify basic structure
        assert staging.total_nodes == 3  # Person1, Person2, Department
        assert staging.total_edges == 2  # Two WORKS_IN relationships
        assert all("transform_id" in node.properties for node in staging.nodes)
        
        # Verify production graph structure
        assert prod.total_nodes == 4  # Person1, Person2_a, Person2_b, Department
        assert prod.total_edges == 2  # BELONGS_TO and WORKS_IN relationships
        
        # Verify conflicts
        conflict_types = {c.conflict_type for c in conflicts}
        assert ConflictType.PROPERTY in conflict_types
        assert ConflictType.RELATIONSHIP_TYPE in conflict_types
        assert ConflictType.DUPLICATE_ENTITY in conflict_types
        
        # Test with specific conflict types
        staging, prod, conflicts = generator.generate_conflicting_graphs(
            conflict_types=[ConflictType.PROPERTY]
        )
        assert all(c.conflict_type == ConflictType.PROPERTY for c in conflicts)
        
        # Verify conflict details
        for conflict in conflicts:
            assert conflict.merge_id is not None
            assert conflict.entity_id is not None
            assert conflict.entity_type is not None
            assert len(conflict.resolution_options) > 0
            
            # Verify resolution options
            for option in conflict.resolution_options:
                assert option.id is not None
                assert option.description is not None
                assert option.resolution_type is not None
                assert option.confidence is not None
                assert option.reasoning is not None
                assert isinstance(option.auto_resolvable, bool)
    
    def test_custom_ontology(self):
        """Test generator with custom ontology"""
        custom_ontology = {
            "entities": {
                "CustomType": {
                    "properties": {
                        "custom_prop": {"type": "string", "required": True}
                    }
                }
            },
            "relationships": {
                "CUSTOM_REL": {
                    "source": "CustomType",
                    "target": "CustomType"
                }
            }
        }
        
        generator = MockDataGenerator(ontology=custom_ontology)
        
        # Test node generation
        node = generator.generate_node(entity_type="CustomType")
        assert node.label == "CustomType"
        assert "custom_prop" in node.properties
        
        # Test graph generation
        graph = generator.generate_graph(node_count=3)
        assert all(node.label == "CustomType" for node in graph.nodes)
        assert all(edge.type == "CUSTOM_REL" for edge in graph.edges) 