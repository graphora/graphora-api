"""
Debug script for conflict detection logic.
"""
import asyncio
import sys
import os
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from app.services.merge.service import MergeService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity
from app.services.storage.models import Node as StorageNode, Edge as StorageEdge
from app.schemas.graph import GraphResponse, Node, Edge
from tests.unit.services.storage.test_graph_storage import MockGraphStorage

async def debug_relationship_conflict():
    """Debug relationship conflict detection"""
    # Create mock storage
    storage = MockGraphStorage()
    
    # Create service with mock storage
    service = MergeService(storage=storage)
    
    # Replace staging/prod storage with separate mocks for testing
    service.staging_storage = MockGraphStorage()
    service.prod_storage = MockGraphStorage()
    
    # Add test data with known conflicts
    service.staging_storage.add_test_node(
        StorageNode(id="s2", label="Person", type="Person", properties={"name": "Bob", "department": "Engineering", "transform_id": "test_id"})
    )
    service.staging_storage.add_test_node(
        StorageNode(id="s3", label="Department", type="Department", properties={"name": "Engineering", "transform_id": "test_id"})
    )
    service.prod_storage.add_test_node(
        StorageNode(id="p2", label="Person", type="Person", properties={"name": "Bob", "title": "Developer"})
    )
    service.prod_storage.add_test_node(
        StorageNode(id="p3", label="Department", type="Department", properties={"name": "Engineering"})
    )
    
    service.staging_storage.add_test_relationship(
        StorageEdge(id="sr1", source="s2", target="s3", type="WORKS_IN", properties={})
    )
    service.prod_storage.add_test_relationship(
        StorageEdge(id="pr1", source="p2", target="p3", type="BELONGS_TO", properties={})
    )
    
    # Create test graph
    staging_graph = GraphResponse(
        nodes=[
            {"id": "s2", "label": "Person", "type": "Person", "properties": {"name": "Bob", "department": "Engineering", "transform_id": "test_id"}},
            {"id": "s3", "label": "Department", "type": "Department", "properties": {"name": "Engineering", "transform_id": "test_id"}}
        ],
        edges=[
            {"id": "sr1", "source": "s2", "target": "s3", "type": "WORKS_IN", "properties": {}}
        ],
        total_nodes=2,
        total_edges=1
    )
    
    production_entity_mapping = {"s2": ["p2"], "s3": ["p3"]}
    
    # Debug the Edge class
    edge = staging_graph.edges[0]
    print(f"Edge attributes: {dir(edge)}")
    print(f"Edge source: {edge.source}")
    print(f"Edge target: {edge.target}")
    
    # Fix the relationship conflict detection
    # Monkey patch the detect_relationship_conflicts method
    original_method = service.detect_relationship_conflicts
    
    async def patched_detect_relationship_conflicts(
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Patched method to detect conflicts in relationships"""
        conflicts = []
        
        # Process each edge in staging graph
        for staging_edge in staging_graph.edges:
            # Skip if either endpoint has no production matches
            if (staging_edge.source not in production_entity_mapping or
                staging_edge.target not in production_entity_mapping):
                continue
                
            # Get production matches for both endpoints
            source_matches = production_entity_mapping[staging_edge.source] 
            target_matches = production_entity_mapping[staging_edge.target]
            
            # Check relationships between all possible endpoint combinations
            for source_id in source_matches:
                for target_id in target_matches:
                    # Get production relationships
                    prod_edges = await service.storage.get_relationships_between(
                        source_id,
                        target_id,
                        staging_edge.type
                    )
                    
                    # Create a simple conflict if relationship types differ
                    for prod_edge in prod_edges:
                        if staging_edge.type != prod_edge.type:
                            conflict = Conflict(
                                id=f"rel_type_{staging_edge.id}_{prod_edge.id}",
                                conflict_type=ConflictType.RELATIONSHIP_TYPE,
                                severity=ConflictSeverity.MAJOR,
                                description=f"Relationship type differs: {staging_edge.type} vs {prod_edge.type}",
                                staging_ids=[staging_edge.id],
                                production_ids=[prod_edge.id],
                                staging_value=staging_edge.type,
                                production_value=prod_edge.type
                            )
                            conflicts.append(conflict)
        
        return conflicts
    
    # Apply the monkey patch
    service.detect_relationship_conflicts = patched_detect_relationship_conflicts
    
    try:
        # Test the patched method
        conflicts = await service.detect_relationship_conflicts(
            staging_graph, production_entity_mapping
        )
        
        print(f"Relationship conflicts: {len(conflicts)}")
        for conflict in conflicts:
            print(f"  - {conflict.conflict_type}: {conflict.description}")
    except Exception as e:
        print(f"Error in relationship conflict detection: {e}")
    
    # Restore the original method
    service.detect_relationship_conflicts = original_method

async def debug_property_conflicts():
    """Debug property conflict detection"""
    # Create mock storage
    storage = MockGraphStorage()
    
    # Create service with mock storage
    service = MergeService(storage=storage)
    
    # Replace staging/prod storage with separate mocks for testing
    service.staging_storage = MockGraphStorage()
    service.prod_storage = MockGraphStorage()
    
    # Add test data with known conflicts
    service.staging_storage.add_test_node(
        StorageNode(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30, "transform_id": "test_id"})
    )
    service.prod_storage.add_test_node(
        StorageNode(id="p1", label="Person", type="Person", properties={"name": "Alice", "age": 32})
    )
    
    # Create test graph
    staging_graph = GraphResponse(
        nodes=[{"id": "s1", "label": "Person", "type": "Person", "properties": {"name": "Alice", "age": 30, "transform_id": "test_id"}}],
        edges=[],
        total_nodes=1,
        total_edges=0
    )
    
    production_entity_mapping = {"s1": ["p1"]}
    
    # Monkey patch the detect_property_conflicts method
    original_method = service.detect_property_conflicts
    
    async def patched_detect_property_conflicts(
        staging_node: Node,
        production_node: Node
    ) -> List[Conflict]:
        """Patched method to detect property conflicts"""
        conflicts = []
        
        # Get all property names
        all_props = set(staging_node.properties.keys()) | set(production_node.properties.keys())
        
        for prop_name in all_props:
            staging_value = staging_node.properties.get(prop_name)
            prod_value = production_node.properties.get(prop_name)
            
            # Skip if values are identical
            if staging_value == prod_value:
                continue
                
            # Create property conflict
            conflict_id = f"prop_{staging_node.id}_{production_node.id}_{prop_name}"
            
            # Determine conflict type and description
            if staging_value is None:
                conflict_type = ConflictType.PROPERTY_VALUE 
                description = f"Property '{prop_name}' exists in production but is missing in staging"
            elif prod_value is None:
                conflict_type = ConflictType.PROPERTY_VALUE
                description = f"Property '{prop_name}' exists in staging but is missing in production"
            else:
                conflict_type = ConflictType.PROPERTY_VALUE
                description = f"Property '{prop_name}' has different values in staging and production"
                
            # Create conflict
            conflict = Conflict(
                id=conflict_id,
                conflict_type=conflict_type,
                severity=ConflictSeverity.MINOR,
                description=description,
                staging_ids=[staging_node.id],
                production_ids=[production_node.id],
                staging_value=staging_value,
                production_value=prod_value
            )
            conflicts.append(conflict)
        
        return conflicts
    
    # Apply the monkey patch
    service.detect_property_conflicts = patched_detect_property_conflicts
    
    try:
        # Test the patched method
        conflicts = await service.detect_property_conflicts_for_graph(
            staging_graph, production_entity_mapping
        )
        
        print(f"Property conflicts: {len(conflicts)}")
        for conflict in conflicts:
            print(f"  - {conflict.conflict_type}: {conflict.description}")
    except Exception as e:
        print(f"Error in property conflict detection: {e}")
    
    # Restore the original method
    service.detect_property_conflicts = original_method

async def debug_entity_matching_conflicts():
    """Debug entity matching conflict detection"""
    # Create mock storage
    storage = MockGraphStorage()
    
    # Create service with mock storage
    service = MergeService(storage=storage)
    
    # Replace staging/prod storage with separate mocks for testing
    service.staging_storage = MockGraphStorage()
    service.prod_storage = MockGraphStorage()
    
    # Add test data with known conflicts
    service.staging_storage.add_test_node(
        StorageNode(id="s4", label="Project", type="Project", properties={"name": "API", "transform_id": "test_id"})
    )
    service.prod_storage.add_test_node(
        StorageNode(id="p4a", label="Project", type="Project", properties={"name": "API", "version": "1.0"})
    )
    service.prod_storage.add_test_node(
        StorageNode(id="p4b", label="Project", type="Project", properties={"name": "API", "version": "2.0"})
    )
    
    # Create test graph
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
    
    # Monkey patch the detect_entity_matching_conflicts method
    original_method = service.detect_entity_matching_conflicts
    
    async def patched_detect_entity_matching_conflicts(
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Patched method to detect entity matching conflicts"""
        conflicts = []
        
        # Process each node with multiple matches
        for staging_id, prod_matches in production_entity_mapping.items():
            if len(prod_matches) <= 1:
                continue
                
            # Get staging node
            staging_node = next(
                (n for n in staging_graph.nodes if n.id == staging_id),
                None
            )
            if not staging_node:
                continue
                
            # Create a simple duplicate entity conflict
            conflict = Conflict(
                id=f"dup_{staging_id}",
                conflict_type=ConflictType.DUPLICATE_ENTITY,
                severity=ConflictSeverity.MAJOR,
                description=f"Entity {staging_id} matches multiple production entities: {', '.join(prod_matches)}",
                staging_ids=[staging_id],
                production_ids=prod_matches
            )
            conflicts.append(conflict)
        
        return conflicts
    
    # Apply the monkey patch
    service.detect_entity_matching_conflicts = patched_detect_entity_matching_conflicts
    
    try:
        # Test the patched method
        conflicts = await service.detect_entity_matching_conflicts(
            staging_graph, production_entity_mapping
        )
        
        print(f"Entity matching conflicts: {len(conflicts)}")
        for conflict in conflicts:
            print(f"  - {conflict.conflict_type}: {conflict.description}")
    except Exception as e:
        print(f"Error in entity matching conflict detection: {e}")
    
    # Restore the original method
    service.detect_entity_matching_conflicts = original_method

async def main():
    """Run all debug functions"""
    print("=== Debugging Relationship Conflict Detection ===")
    await debug_relationship_conflict()
    
    print("\n=== Debugging Property Conflict Detection ===")
    await debug_property_conflicts()
    
    print("\n=== Debugging Entity Matching Conflict Detection ===")
    await debug_entity_matching_conflicts()

if __name__ == "__main__":
    asyncio.run(main())
