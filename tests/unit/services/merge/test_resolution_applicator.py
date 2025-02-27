import pytest
from unittest.mock import MagicMock, AsyncMock
from app.services.merge.resolution_applicator import ResolutionApplicator
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.graph import Node, Edge
from datetime import datetime
import pytz

class MockGraphStorage:
    """Mock implementation of GraphStorageInterface for testing"""
    
    def __init__(self):
        self.nodes = {}
        self.relationships = {}
        
    def add_test_node(self, node):
        """Add a test node to the mock storage"""
        self.nodes[node.id] = node
        
    def add_test_relationship(self, relationship):
        """Add a test relationship to the mock storage"""
        self.relationships[relationship.id] = relationship
        
    async def get_node_by_id(self, node_id):
        """Get a node by ID"""
        return self.nodes.get(node_id)
        
    async def get_relationship_by_id(self, rel_id):
        """Get a relationship by ID"""
        return self.relationships.get(rel_id)
        
    async def update_node_property(self, node_id, prop_name, value):
        """Update a node property"""
        if node_id in self.nodes:
            self.nodes[node_id].properties[prop_name] = value
            return True
        return False
        
    async def remove_node_property(self, node_id, prop_name):
        """Remove a node property"""
        if node_id in self.nodes and prop_name in self.nodes[node_id].properties:
            del self.nodes[node_id].properties[prop_name]
            return True
        return False
        
    async def update_node(self, node_id, properties):
        """Update a node with new properties"""
        if node_id in self.nodes:
            self.nodes[node_id].properties.update(properties)
            return self.nodes[node_id]
        return None
        
    async def update_relationship_type(self, rel_id, new_type):
        """Update a relationship type"""
        if rel_id in self.relationships:
            self.relationships[rel_id].type = new_type
            return True
        return False
        
    async def create_relationship(self, source_id, target_id, rel_type, properties=None):
        """Create a new relationship"""
        if properties is None:
            properties = {}
        rel_id = f"{source_id}-{rel_type}-{target_id}"
        rel = Edge(
            id=rel_id,
            source=source_id,
            target=target_id,
            type=rel_type,
            properties=properties
        )
        self.relationships[rel_id] = rel
        return rel
        
    async def delete_relationship(self, rel_id):
        """Delete a relationship"""
        if rel_id in self.relationships:
            del self.relationships[rel_id]
            return True
        return False
        
    async def delete_node(self, node_id):
        """Delete a node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False
        
    async def get_relationships_between(self, source_id, target_id):
        """Get relationships between two nodes"""
        return [
            rel for rel in self.relationships.values()
            if rel.source == source_id and rel.target == target_id
        ]
        
    async def get_incoming_relationships(self, node_id):
        """Get incoming relationships for a node"""
        return [
            rel for rel in self.relationships.values()
            if rel.target == node_id
        ]
        
    async def get_outgoing_relationships(self, node_id):
        """Get outgoing relationships for a node"""
        return [
            rel for rel in self.relationships.values()
            if rel.source == node_id
        ]

@pytest.fixture
def applicator_with_mocks():
    """Set up applicator with mock storages"""
    staging_storage = MockGraphStorage()
    prod_storage = MockGraphStorage()
    
    # Add test nodes
    staging_storage.add_test_node(
        Node(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30, "department": "Engineering"})
    )
    
    prod_storage.add_test_node(
        Node(id="p1", label="Person", type="Person", properties={"name": "Alice", "age": 32})
    )
    
    # Add test relationships
    staging_storage.add_test_relationship(
        Edge(id="sr1", source="s1", target="s2", type="WORKS_IN", properties={})
    )
    
    prod_storage.add_test_relationship(
        Edge(id="pr1", source="p1", target="p2", type="BELONGS_TO", properties={})
    )
    
    return ResolutionApplicator(staging_storage, prod_storage)

class TestResolutionApplicator:
    @pytest.mark.asyncio
    async def test_apply_property_value_resolution_keep_staging(self, applicator_with_mocks):
        """Test applying a property value resolution with 'keep_staging' strategy"""
        # Arrange
        conflict = Conflict(
            id="prop_val_s1_p1_age",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'age' has different values",
            context={
                "property_name": "age",
                "staging_value": 30,
                "production_value": 32,
                "entity_type": "Person"
            }
        )
        
        resolution = ResolutionOption(
            id="opt1",
            description="Keep staging value: 30",
            resolution_type="keep_staging",
            resolution_data={"property_name": "age"},
            confidence=0.8
        )
        
        # Act
        result = await applicator_with_mocks.apply_resolution(conflict, resolution)
        
        # Assert
        assert result["applied"] == True
        assert result["conflict_id"] == conflict.id
        assert result["resolution_id"] == resolution.id
        assert "verification" in result
        assert result["verification"]["verified"] == True
        assert result["changes"]["property"] == "age"
        assert result["changes"]["old_value"] == 32
        assert result["changes"]["new_value"] == 30
        assert result["changes"]["action"] == "updated_production"
        
        # Verify the production node was updated
        prod_node = await applicator_with_mocks.prod_storage.get_node_by_id("p1")
        assert prod_node.properties["age"] == 30
    
    @pytest.mark.asyncio
    async def test_apply_property_value_resolution_keep_production(self, applicator_with_mocks):
        """Test applying a property value resolution with 'keep_production' strategy"""
        # Arrange
        conflict = Conflict(
            id="prop_val_s1_p1_age",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'age' has different values",
            context={
                "property_name": "age",
                "staging_value": 30,
                "production_value": 32,
                "entity_type": "Person"
            }
        )
        
        resolution = ResolutionOption(
            id="opt2",
            description="Keep production value: 32",
            resolution_type="keep_production",
            resolution_data={"property_name": "age"},
            confidence=0.5
        )
        
        # Act
        result = await applicator_with_mocks.apply_resolution(conflict, resolution)
        
        # Assert
        assert result["applied"] == True
        assert result["conflict_id"] == conflict.id
        assert result["resolution_id"] == resolution.id
        assert "verification" in result
        assert result["verification"]["verified"] == True
        assert result["changes"]["property"] == "age"
        assert result["changes"]["value"] == 32
        assert result["changes"]["action"] == "no_change"
        
        # Verify the production node was not changed
        prod_node = await applicator_with_mocks.prod_storage.get_node_by_id("p1")
        assert prod_node.properties["age"] == 32
    
    @pytest.mark.asyncio
    async def test_apply_property_value_resolution_merge_values(self, applicator_with_mocks):
        """Test applying a property value resolution with 'merge_values' strategy"""
        # Arrange
        conflict = Conflict(
            id="prop_val_s1_p1_name",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'name' has different values",
            context={
                "property_name": "name",
                "staging_value": "Alice Smith",
                "production_value": "Alice Jones",
                "entity_type": "Person"
            }
        )
        
        resolution = ResolutionOption(
            id="opt3",
            description="Merge values",
            resolution_type="merge_values",
            resolution_data={"strategy": "concat"},
            confidence=0.7
        )
        
        # Act
        result = await applicator_with_mocks.apply_resolution(conflict, resolution)
        
        # Assert
        assert result["applied"] == True
        assert result["conflict_id"] == conflict.id
        assert result["resolution_id"] == resolution.id
        assert "verification" in result
        assert result["verification"]["verified"] == True
        assert result["changes"]["property"] == "name"
        assert result["changes"]["old_value"] == "Alice Jones"
        assert result["changes"]["new_value"] == "Alice Jones | Alice Smith"
        assert result["changes"]["action"] == "merged_values"
        
        # Verify the production node was updated with merged value
        prod_node = await applicator_with_mocks.prod_storage.get_node_by_id("p1")
        assert prod_node.properties["name"] == "Alice Jones | Alice Smith"
    
    @pytest.mark.asyncio
    async def test_apply_property_missing_resolution_add_to_production(self, applicator_with_mocks):
        """Test applying a missing property resolution with 'add_to_production' strategy"""
        # Arrange
        conflict = Conflict(
            id="prop_missing_s1_p1_department",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_MISSING,
            severity=ConflictSeverity.MINOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property exists in staging but not in production",
            context={
                "property_name": "department",
                "missing_in": "production",
                "entity_type": "Person"
            }
        )
        
        resolution = ResolutionOption(
            id="opt1",
            description="Add property to production: department",
            resolution_type="add_to_production",
            resolution_data={},
            confidence=0.9
        )
        
        # Act
        result = await applicator_with_mocks.apply_resolution(conflict, resolution)
        
        # Assert
        assert result["applied"] == True
        assert result["conflict_id"] == conflict.id
        assert result["resolution_id"] == resolution.id
        assert result["changes"]["property"] == "department"
        assert result["changes"]["value"] == "Engineering"
        assert result["changes"]["action"] == "added_to_production"
        
        # Verify the production node was updated with the new property
        prod_node = await applicator_with_mocks.prod_storage.get_node_by_id("p1")
        assert "department" in prod_node.properties
        assert prod_node.properties["department"] == "Engineering"
    
    @pytest.mark.asyncio
    async def test_apply_property_missing_resolution_remove_from_production(self, applicator_with_mocks):
        """Test applying a missing property resolution with 'remove_from_production' strategy"""
        # Arrange
        # First add a property to production that doesn't exist in staging
        prod_node = await applicator_with_mocks.prod_storage.get_node_by_id("p1")
        await applicator_with_mocks.prod_storage.update_node_property("p1", "title", "Manager")
        
        conflict = Conflict(
            id="prop_missing_s1_p1_title",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_MISSING,
            severity=ConflictSeverity.MINOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property exists in production but not in staging",
            context={
                "property_name": "title",
                "missing_in": "staging",
                "entity_type": "Person"
            }
        )
        
        resolution = ResolutionOption(
            id="opt1",
            description="Remove property from production: title",
            resolution_type="remove_from_production",
            resolution_data={},
            confidence=0.9
        )
        
        # Act
        result = await applicator_with_mocks.apply_resolution(conflict, resolution)
        
        # Assert
        assert result["applied"] == True
        assert result["conflict_id"] == conflict.id
        assert result["resolution_id"] == resolution.id
        assert result["changes"]["property"] == "title"
        assert result["changes"]["action"] == "removed_from_production"
        
        # Verify the property was removed from production
        prod_node = await applicator_with_mocks.prod_storage.get_node_by_id("p1")
        assert "title" not in prod_node.properties
    
    @pytest.mark.asyncio
    async def test_apply_relationship_type_resolution(self, applicator_with_mocks):
        """Test applying a relationship type resolution"""
        # Arrange
        conflict = Conflict(
            id="rel_type_sr1_pr1",
            merge_id="test_merge",
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["sr1"],
            production_ids=["pr1"],
            description="Relationship has different types",
            context={
                "staging_type": "WORKS_IN",
                "production_type": "BELONGS_TO"
            }
        )
        
        resolution = ResolutionOption(
            id="opt1",
            description="Keep staging relationship type: WORKS_IN",
            resolution_type="keep_staging_rel_type",
            resolution_data={},
            confidence=0.8
        )
        
        # Act
        result = await applicator_with_mocks.apply_resolution(conflict, resolution)
        
        # Assert
        assert result["applied"] == True
        assert result["conflict_id"] == conflict.id
        assert result["resolution_id"] == resolution.id
        assert result["changes"]["old_type"] == "BELONGS_TO"
        assert result["changes"]["new_type"] == "WORKS_IN"
        assert result["changes"]["action"] == "updated_relationship_type"
        
        # Verify the relationship type was updated
        prod_rel = await applicator_with_mocks.prod_storage.get_relationship_by_id("pr1")
        assert prod_rel.type == "WORKS_IN"
    
    @pytest.mark.asyncio
    async def test_verification_failure(self, applicator_with_mocks):
        """Test verification failing when the change wasn't applied correctly"""
        # Arrange
        conflict = Conflict(
            id="prop_val_s1_p1_age",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'age' has different values",
            context={
                "property_name": "age",
                "staging_value": 30,
                "production_value": 32,
                "entity_type": "Person"
            }
        )
        
        resolution = ResolutionOption(
            id="opt1",
            description="Keep staging value: 30",
            resolution_type="keep_staging",
            resolution_data={"property_name": "age"},
            confidence=0.8
        )
        
        # Mock update_node_property to not actually update the value
        original_update = applicator_with_mocks.prod_storage.update_node_property
        applicator_with_mocks.prod_storage.update_node_property = AsyncMock(return_value=True)
        
        # Act
        result = await applicator_with_mocks.apply_resolution(conflict, resolution)
        
        # Restore original method
        applicator_with_mocks.prod_storage.update_node_property = original_update
        
        # Assert
        assert result["applied"] == True  # Application was attempted
        assert result["conflict_id"] == conflict.id
        assert result["resolution_id"] == resolution.id
        assert "verification" in result
        assert result["verification"]["verified"] == False  # But verification failed
        assert result["verification"]["current_value"] == 32
        assert result["verification"]["expected_value"] == 30
    
    @pytest.mark.asyncio
    async def test_apply_with_exception(self, applicator_with_mocks):
        """Test handling exceptions during resolution application"""
        # Arrange
        conflict = Conflict(
            id="prop_val_s1_p1_age",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'age' has different values",
            context={
                "property_name": "age",
                "staging_value": 30,
                "production_value": 32,
                "entity_type": "Person"
            }
        )
        
        resolution = ResolutionOption(
            id="opt1",
            description="Keep staging value: 30",
            resolution_type="keep_staging",
            resolution_data={"property_name": "age"},
            confidence=0.8
        )
        
        # Mock update_node_property to raise an exception
        applicator_with_mocks.prod_storage.update_node_property = AsyncMock(side_effect=Exception("Update failed"))
        
        # Act
        result = await applicator_with_mocks.apply_resolution(conflict, resolution)
        
        # Assert
        assert result["applied"] == False
        assert result["conflict_id"] == conflict.id
        assert result["resolution_id"] == resolution.id
        assert "error" in result
        assert "Update failed" in result["error"] 