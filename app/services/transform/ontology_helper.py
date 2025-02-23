from typing import Dict, List, Any, Type, Optional, Union, Set
import yaml
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, create_model, Field

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class OntologyParser:
    """Parser for YAML ontology definitions"""
    
    def __init__(self, yaml_path: Union[str, Path]):
        """Initialize parser with YAML ontology"""
        # Load YAML content
        if isinstance(yaml_path, Path):
            with open(yaml_path) as f:
                yaml_content = f.read()
        else:
            yaml_content = yaml_path
            
        self.parsed_ontology = yaml.safe_load(yaml_content)
        self.validate_ontology_structure()
        
    def validate_ontology_structure(self) -> None:
        """Validate ontology has required structure"""
        required_keys = ['version', 'entities']
        if not all(key in self.parsed_ontology for key in required_keys):
            raise ValueError(f"Ontology missing required keys: {required_keys}")
            
        # Validate each entity has properties
        # for entity, definition in self.parsed_ontology['entities'].items():
        #     if 'properties' not in definition:
        #         raise ValueError(f"Entity {entity} missing 'properties' definition")
    
    def build_graph_model(self) -> Type[BaseModel]:
        """
        Build a complete Pydantic model structure from a YAML ontology definition.
        Returns a KnowledgeGraph class that can be used as a response_model.
        """
        # Parse YAML
        ontology = self.parsed_ontology
        entities = ontology.get('entities', {})
        
        # Dictionary to store all dynamically created models
        entity_models = {}
        relationship_models = {}
        
        # First create all entity models
        for entity_name, entity_def in entities.items():
            # Create the base properties
            props = entity_def.get('properties', {})
            field_definitions = {}
            
            for prop_name, prop_def in props.items():
                field_type = self._get_field_type(prop_def.get('type', 'str'))
                
                is_required = prop_def.get('required', False)
                is_unique = prop_def.get('unique', False)
                is_indexed = prop_def.get('index', False)
                
                # Set default only if not required
                default_value = None if not is_required else ...
                
                field_definitions[prop_name] = (
                    Optional[field_type] if not is_required else field_type,
                    Field(
                        default=default_value,
                        description=prop_def.get('description', ''),
                        title=prop_name
                    )
                )
            
            # Create entity model
            entity_model = create_model(
                entity_name,
                __base__=BaseModel,
                __domain__="graphit",
                **field_definitions
            )
            
            # Store model
            entity_models[entity_name] = entity_model
            globals()[entity_name] = entity_model
        
        # Then create relationship models
        for entity_name, entity_def in entities.items():
            relationships = entity_def.get('relationships', {})
            entity_relationship_models = {}
            
            for rel_name, rel_def in relationships.items():
                target_name = rel_def.get('target')
                target_model = entity_models.get(target_name)
                
                if not target_model:
                    continue
                    
                # Handle relationship properties
                rel_props = rel_def.get('properties', {})
                rel_field_defs = {}
                
                for prop_name, prop_def in rel_props.items():
                    field_type = self._get_field_type(prop_def.get('type', 'str'))
                    is_required = prop_def.get('required', False)
                    is_unique = prop_def.get('unique', False)
                    
                    default_value = None if not is_required else ...
                    
                    rel_field_defs[prop_name] = (
                        Optional[field_type] if not is_required else field_type,
                        Field(
                            default=default_value,
                            description=prop_def.get('description', ''),
                            title=prop_name
                        )
                    )
                
                # Create relationship property model if needed
                if rel_field_defs:
                    rel_property_model_name = f"{entity_name}_{rel_name}_Properties"
                    rel_property_model = create_model(
                        rel_property_model_name,
                        __base__=BaseModel,
                        **rel_field_defs
                    )
                    globals()[rel_property_model_name] = rel_property_model
                else:
                    rel_property_model = None
                
                # Create relationship model
                rel_model_name = f"{entity_name}_{rel_name}_Relationship"
                
                if rel_property_model:
                    # With properties
                    rel_model = create_model(
                        rel_model_name,
                        __base__=BaseModel,
                        __domain__="graphit",
                        source=(entity_models[entity_name], ...),
                        target=(target_model, ...),
                        properties=(Optional[rel_property_model], None)
                    )
                else:
                    # Without properties
                    rel_model = create_model(
                        rel_model_name,
                        __base__=BaseModel,
                        __domain__="graphit",
                        source=(entity_models[entity_name], ...),
                        target=(target_model, ...)
                    )
                    
                globals()[rel_model_name] = rel_model
                entity_relationship_models[rel_name] = rel_model
            
            if entity_relationship_models:
                relationship_models[entity_name] = entity_relationship_models
        
        # Now create the KnowledgeGraph model that includes all entities and relationships
        kg_fields = {
            "extraction_timestamp": (str, Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())),
            "tokens_used": (Optional[int], None),
            "confidence_score": (Optional[float], None)
        }
        
        # Add fields for each entity type (list of that entity)
        for entity_name, entity_model in entity_models.items():
            kg_fields[entity_name+"_list"] = (
                Optional[List[entity_model]],
                Field(default_factory=list, description=f"List of {entity_name} entities")
            )
        
        # Add fields for relationships
        for source_name, rels in relationship_models.items():
            for rel_name, rel_model in rels.items():
                field_name = f"{source_name}_{rel_name}"
                kg_fields[field_name] = (
                    Optional[List[rel_model]],
                    Field(default_factory=list, description=f"Relationships of type {rel_name} from {source_name}")
                )
        
        # Create the KnowledgeGraph model
        KnowledgeGraph = create_model(
            "KnowledgeGraph",
            __base__=BaseModel,
            __domain__="graphit",
            **kg_fields
        )
        
        # Attach metadata to help with serialization/deserialization
        KnowledgeGraph.__entity_models__ = entity_models
        KnowledgeGraph.__relationship_models__ = relationship_models
        
        return KnowledgeGraph

    def _get_field_type(self, type_str: str) -> Type:
        """Convert string type to actual Python type."""
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': List[Any],
            'dict': Dict[str, Any],
        }
        return type_mapping.get(type_str, str)

class DependencyTreeBuilder:
    """Builds a dependency tree from ontology for ordered graph construction"""
    
    def __init__(self, ontology_parser: OntologyParser):
        self.ontology = ontology_parser.parsed_ontology
        self.entities = self.ontology.get('entities', {})
        self.dependency_graph = {}
        self.build_dependency_graph()
        
    def build_dependency_graph(self):
        """Build dependency graph from ontology relationships"""
        # Initialize dependency graph for all entities
        for entity_type in self.entities:
            self.dependency_graph[entity_type] = {
                'depends_on': set(),  # Entities this one depends on
                'depended_by': set(), # Entities that depend on this one
                'relationships': {}   # Detailed relationship info
            }
        
        # Build dependencies based on relationships
        for entity_type, entity_def in self.entities.items():
            relationships = entity_def.get('relationships', {})
            
            for rel_name, rel_def in relationships.items():
                target_type = rel_def.get('target')
                if not target_type:
                    continue
                
                # Add dependency
                self.dependency_graph[entity_type]['depends_on'].add(target_type)
                self.dependency_graph[target_type]['depended_by'].add(entity_type)
                
                # Store relationship details
                self.dependency_graph[entity_type]['relationships'][rel_name] = {
                    'target': target_type,
                    'properties': rel_def.get('properties', {}),
                    'required': rel_def.get('required', False)
                }
    
    def get_processing_levels(self) -> List[List[str]]:
        """
        Get entity types grouped by processing level.
        Returns list of lists, where each inner list contains entity types
        that can be processed in parallel.
        """
        levels = []
        remaining = set(self.entities.keys())
        processed = set()
        
        while remaining:
            # Find all entities that only depend on already processed ones
            current_level = set()
            
            for entity_type in remaining:
                dependencies = self.dependency_graph[entity_type]['depends_on']
                if dependencies.issubset(processed):
                    current_level.add(entity_type)
            
            if not current_level:
                # Handle cycles by taking entities with minimal unprocessed dependencies
                min_deps = float('inf')
                for entity_type in remaining:
                    unprocessed_deps = len(self.dependency_graph[entity_type]['depends_on'] - processed)
                    min_deps = min(min_deps, unprocessed_deps)
                
                current_level = {
                    entity_type for entity_type in remaining
                    if len(self.dependency_graph[entity_type]['depends_on'] - processed) == min_deps
                }
            
            levels.append(list(current_level))
            processed.update(current_level)
            remaining -= current_level
        
        return levels
      
class OntologyHierarchyBuilder:
    """Builds hierarchical structure from ontology definition"""
    
    def __init__(self, ontology_parser: OntologyParser):
        self.ontology = ontology_parser.parsed_ontology
        self.entities = self.ontology.get('entities', {})
        self.hierarchy = {}
        self.dependency_graph = {}
        self.root_entities = set()
        self.build_hierarchy()
        
    def build_hierarchy(self):
        """
        Build hierarchy from ontology by analyzing relationships.
        A parent-child relationship is inferred when one entity type
        has outgoing relationships to another entity type.
        """
        # First build dependency graph
        for entity_type, entity_def in self.entities.items():
            self.dependency_graph[entity_type] = {
                'outgoing': set(),  # Entity types this one points to
                'incoming': set(),  # Entity types that point to this one
                'relationships': {} # Map of target types to relationship types
            }
            
            # Analyze relationships
            relationships = entity_def.get('relationships', {})
            for rel_name, rel_def in relationships.items():
                target_type = rel_def.get('target')
                if target_type:
                    self.dependency_graph[entity_type]['outgoing'].add(target_type)
                    if target_type not in self.dependency_graph:
                        self.dependency_graph[target_type] = {
                            'outgoing': set(),
                            'incoming': set(),
                            'relationships': {}
                        }
                    self.dependency_graph[target_type]['incoming'].add(entity_type)
                    
                    # Store relationship type information
                    if target_type not in self.dependency_graph[entity_type]['relationships']:
                        self.dependency_graph[entity_type]['relationships'][target_type] = []
                    self.dependency_graph[entity_type]['relationships'][target_type].append(rel_name)
        
        # Find root entities (those with no incoming relationships)
        self.root_entities = {
            entity_type for entity_type, deps in self.dependency_graph.items()
            if not deps['incoming']
        }
        
        # Build hierarchy tree for each root
        for root in self.root_entities:
            self.hierarchy[root] = self._build_subtree(root)
    
    def _build_subtree(self, entity_type: str, visited: Optional[Set[str]] = None) -> Dict:
        """Recursively build hierarchy subtree"""
        if visited is None:
            visited = set()
            
        if entity_type in visited:
            return {}  # Prevent cycles
            
        visited.add(entity_type)
        children = {}
        
        # Get all target types from relationships
        target_types = self.dependency_graph[entity_type]['outgoing']
        
        for target in target_types:
            rel_types = self.dependency_graph[entity_type]['relationships'].get(target, [])
            children[target] = {
                'relationship_types': rel_types,
                'children': self._build_subtree(target, visited.copy())
            }
            
        return children
    
    def get_relationship_types(self, source_type: str, target_type: str) -> List[str]:
        """Get valid relationship types between two entity types"""
        return self.dependency_graph.get(source_type, {}).get('relationships', {}).get(target_type, [])
    
    def get_parent_types(self, entity_type: str) -> Set[str]:
        """Get all possible parent entity types"""
        return self.dependency_graph.get(entity_type, {}).get('incoming', set())
    
    def get_child_types(self, entity_type: str) -> Set[str]:
        """Get all possible child entity types"""
        return self.dependency_graph.get(entity_type, {}).get('outgoing', set())