"""Utility to generate mock data for testing"""
import random
from typing import Dict, List, Any, Optional, Tuple
import uuid
from datetime import datetime

from app.schemas.graph import Node, Edge, GraphResponse
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption, ResolutionStrategy

class MockDataGenerator:
    """Utility to generate mock data for testing"""
    
    def __init__(self, ontology: Optional[Dict[str, Any]] = None):
        """Initialize with optional ontology for realistic data"""
        self.ontology = ontology or self._default_ontology()
        self.entity_types = list(self.ontology.get("entities", {}).keys())
        
    def _default_ontology(self) -> Dict[str, Any]:
        """Default simple ontology for testing"""
        return {
            "entities": {
                "Person": {
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "age": {"type": "number"},
                        "email": {"type": "string"},
                        "department": {"type": "string"}
                    }
                },
                "Department": {
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "location": {"type": "string"},
                        "budget": {"type": "number"}
                    }
                },
                "Project": {
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "status": {"type": "string"},
                        "priority": {"type": "number"}
                    }
                }
            },
            "relationships": {
                "WORKS_IN": {
                    "source": "Person",
                    "target": "Department"
                },
                "MANAGES": {
                    "source": "Person",
                    "target": "Department"
                },
                "WORKS_ON": {
                    "source": "Person",
                    "target": "Project"
                },
                "OWNS": {
                    "source": "Department",
                    "target": "Project"
                }
            }
        }

    def generate_node(
        self, 
        entity_type: Optional[str] = None,
        with_transform_id: bool = True,
        transform_id: str = "test_id"
    ) -> Node:
        """Generate a random node of given type"""
        if not entity_type:
            entity_type = random.choice(self.entity_types)
            
        node_id = str(uuid.uuid4())
        
        # Generate properties based on ontology
        entity_def = self.ontology["entities"].get(entity_type, {})
        property_defs = entity_def.get("properties", {})
        
        properties = {}
        for prop_name, prop_def in property_defs.items():
            if prop_def.get("required", False) or random.random() > 0.3:
                prop_type = prop_def.get("type", "string")
                if prop_type == "string":
                    properties[prop_name] = f"{prop_name}_{node_id[:5]}"
                elif prop_type == "number":
                    properties[prop_name] = random.randint(1, 100)
                # Add other types as needed
        
        if with_transform_id:
            properties["transform_id"] = transform_id
            
        return Node(
            id=node_id,
            label=entity_type,
            type=entity_type,
            properties=properties
        )
        
    def generate_edge(
        self,
        source_node: Node,
        target_node: Node,
        relationship_type: Optional[str] = None,
        with_properties: bool = True
    ) -> Edge:
        """Generate a relationship between two nodes"""
        # If no relationship type specified, find valid ones from ontology
        if not relationship_type:
            valid_relationships = []
            for rel_name, rel_def in self.ontology.get("relationships", {}).items():
                if rel_def.get("source") == source_node.label and rel_def.get("target") == target_node.label:
                    valid_relationships.append(rel_name)
            
            if not valid_relationships:
                raise ValueError(f"No valid relationship types between {source_node.label} and {target_node.label}")
                
            relationship_type = random.choice(valid_relationships)
        
        edge_id = str(uuid.uuid4())
        
        properties = {}
        if with_properties:
            # Add some default relationship properties
            properties["created_at"] = datetime.now().isoformat()
            properties["weight"] = random.randint(1, 10)
            
        return Edge(
            id=edge_id,
            source=source_node.id,
            target=target_node.id,
            type=relationship_type,
            properties=properties
        )
    
    def generate_graph(
        self,
        node_count: int = 10,
        edge_density: float = 0.3,
        transform_id: str = "test_id"
    ) -> GraphResponse:
        """Generate a complete graph with nodes and relationships"""
        nodes = []
        for _ in range(node_count):
            nodes.append(self.generate_node(transform_id=transform_id))
        
        edges = []
        # Create relationships based on density
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i != j and random.random() < edge_density:
                    try:
                        edge = self.generate_edge(source, target)
                        edges.append(edge)
                    except ValueError:
                        # No valid relationship type, skip
                        pass
        
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges)
        )
    
    def generate_conflicting_graphs(
        self,
        transform_id: str = "test_id",
        conflict_types: Optional[List[ConflictType]] = None
    ) -> Tuple[GraphResponse, GraphResponse, List[Conflict]]:
        """Generate staging and production graphs with known conflicts"""
        # Default conflict types if none specified
        if conflict_types is None:
            conflict_types = [
                ConflictType.PROPERTY_VALUE,
                ConflictType.RELATIONSHIP_TYPE,
                ConflictType.DUPLICATE_ENTITY
            ]
        
        # Generate base nodes
        person1 = self.generate_node("Person", transform_id=transform_id)
        person2 = self.generate_node("Person", transform_id=transform_id)
        dept = self.generate_node("Department", transform_id=transform_id)
        
        # Create staging graph
        staging_nodes = [person1, person2, dept]
        staging_edges = [
            self.generate_edge(person1, dept, relationship_type="WORKS_IN"),
            self.generate_edge(person2, dept, relationship_type="WORKS_IN")
        ]
        
        # Create production nodes with conflicts
        prod_person1 = Node(
            id=person1.id,
            label=person1.label,
            type=person1.type,
            properties={
                **person1.properties,
                "age": person1.properties.get("age", 30) + 1  # Property conflict
            }
        )
        
        prod_person2_a = Node(  # Duplicate entity conflict
            id=f"{person2.id}_a",
            label=person2.label,
            type=person2.type,
            properties=person2.properties
        )
        
        prod_person2_b = Node(
            id=f"{person2.id}_b",
            label=person2.label,
            type=person2.type,
            properties={**person2.properties, "title": "Manager"}
        )
        
        prod_dept = Node(
            id=dept.id,
            label=dept.label,
            type=dept.type,
            properties=dept.properties
        )
        
        # Create production edges with conflicts
        prod_edges = [
            Edge(  # Relationship type conflict
                id=staging_edges[0].id,
                source=prod_person1.id,
                target=prod_dept.id,
                type="BELONGS_TO",  # Different relationship type
                properties=staging_edges[0].properties
            ),
            Edge(
                id=staging_edges[1].id,
                source=prod_person2_a.id,
                target=prod_dept.id,
                type="WORKS_IN",
                properties=staging_edges[1].properties
            )
        ]
        
        # Create expected conflicts
        conflicts = []
        
        if ConflictType.PROPERTY_VALUE in conflict_types:
            conflicts.append(Conflict(
                id=f"conflict_prop_{person1.id}",
                merge_id="test_merge",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MINOR,
                description=f"Property 'age' has different values",
                staging_value=person1.properties.get("age"),
                production_value=prod_person1.properties.get("age"),
                entity_id=person1.id,
                entity_type=person1.type,
                source_data={"age": person1.properties.get("age")},
                target_data={"age": prod_person1.properties.get("age")},
                resolution_options=[
                    ResolutionOption(
                        id="keep_staging",
                        description="Keep staging value",
                        resolution_type=ResolutionStrategy.KEEP_STAGING,
                        confidence=0.8,
                        reasoning="Staging data is more recent",
                        auto_resolvable=True
                    ),
                    ResolutionOption(
                        id="keep_prod",
                        description="Keep production value",
                        resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                        confidence=0.6,
                        reasoning="Production data might be verified",
                        auto_resolvable=True
                    )
                ]
            ))
            
        if ConflictType.RELATIONSHIP_TYPE in conflict_types:
            conflicts.append(Conflict(
                id=f"conflict_rel_{staging_edges[0].id}",
                merge_id="test_merge",
                conflict_type=ConflictType.RELATIONSHIP_TYPE,
                severity=ConflictSeverity.MAJOR,
                description=f"Relationship type mismatch: WORKS_IN vs BELONGS_TO",
                entity_id=staging_edges[0].id,
                entity_type="relationship",
                source_data={"type": staging_edges[0].type},
                target_data={"type": prod_edges[0].type},
                resolution_options=[
                    ResolutionOption(
                        id="keep_staging_rel",
                        description="Keep staging relationship type",
                        resolution_type=ResolutionStrategy.KEEP_STAGING_REL,
                        confidence=0.7,
                        reasoning="Staging data is more recent",
                        auto_resolvable=False
                    ),
                    ResolutionOption(
                        id="keep_prod_rel",
                        description="Keep production relationship type",
                        resolution_type=ResolutionStrategy.KEEP_PRODUCTION_REL,
                        confidence=0.5,
                        reasoning="Production data might be verified",
                        auto_resolvable=False
                    ),
                    ResolutionOption(
                        id="keep_both_rels",
                        description="Keep both relationships",
                        resolution_type=ResolutionStrategy.KEEP_BOTH_RELS,
                        confidence=0.8,
                        reasoning="Both relationships might be valid",
                        auto_resolvable=False
                    )
                ]
            ))
            
        if ConflictType.DUPLICATE_ENTITY in conflict_types:
            conflicts.append(Conflict(
                id=f"conflict_dup_{person2.id}",
                merge_id="test_merge",
                conflict_type=ConflictType.DUPLICATE_ENTITY,
                severity=ConflictSeverity.CRITICAL,
                description=f"Multiple potential matches for entity",
                entity_id=person2.id,
                entity_type=person2.type,
                source_data={"id": person2.id, "type": person2.type},
                target_data={"matches": [{"id": prod_person2_a.id}, {"id": prod_person2_b.id}]},
                resolution_options=[
                    ResolutionOption(
                        id="match_a",
                        description=f"Match with entity {prod_person2_a.id}",
                        resolution_type=ResolutionStrategy.MATCH_ENTITY,
                        confidence=0.6,
                        reasoning="Similar properties",
                        auto_resolvable=False,
                        resolution_data={"target_entity_id": prod_person2_a.id}
                    ),
                    ResolutionOption(
                        id="match_b",
                        description=f"Match with entity {prod_person2_b.id}",
                        resolution_type=ResolutionStrategy.MATCH_ENTITY,
                        confidence=0.4,
                        reasoning="Similar but has additional properties",
                        auto_resolvable=False,
                        resolution_data={"target_entity_id": prod_person2_b.id}
                    ),
                    ResolutionOption(
                        id="create_new",
                        description="Create as new entity",
                        resolution_type=ResolutionStrategy.CREATE_NEW,
                        confidence=0.3,
                        reasoning="Might be a new entity",
                        auto_resolvable=False
                    )
                ]
            ))
        
        # Create graph responses
        staging_graph = GraphResponse(
            nodes=staging_nodes,
            edges=staging_edges,
            total_nodes=len(staging_nodes),
            total_edges=len(staging_edges)
        )
        
        prod_graph = GraphResponse(
            nodes=[prod_person1, prod_person2_a, prod_person2_b, prod_dept],
            edges=prod_edges,
            total_nodes=4,
            total_edges=len(prod_edges)
        )
        
        return staging_graph, prod_graph, conflicts 