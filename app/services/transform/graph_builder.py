from typing import Dict, List, Any, Type, Tuple, Optional, Callable, Union
import yaml
import copy
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, create_model, Field
import traceback
from app.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    KnowledgeGraph,
    ExtractionMetrics
)
from app.services.llm.client import LLMClient
from app.utils.logger import logger

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
                List[entity_model],
                Field(default_factory=list, description=f"List of {entity_name} entities")
            )
        
        # Add fields for relationships
        for source_name, rels in relationship_models.items():
            for rel_name, rel_model in rels.items():
                field_name = f"{source_name}_{rel_name}"
                kg_fields[field_name] = (
                    List[rel_model],
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

class KnowledgeGraphBuilder:
    """Builds unified knowledge graph from document chunks"""
    
    def __init__(
        self,
        ontology_parser: OntologyParser
    ):
        self.ontology_parser = ontology_parser
        self.pydantic_cls = self.ontology_parser.build_graph_model()
        self.graph = self.pydantic_cls()
        self.metrics = ExtractionMetrics(start_time=datetime.now())
        self.llm_client = LLMClient()
        
        # Store entity and relationship models for easier access
        self.entity_models = self.pydantic_cls.__entity_models__
        self.relationship_models = self.pydantic_cls.__relationship_models__
        
        # Node registry for deduplication (entity_type -> {node_key -> node_id})
        # node_key is a deterministic key based on unique properties
        self.node_registry = {}
        
        # Relationship registry for deduplication (source_id -> target_id -> rel_type -> rel_id)
        self.relationship_registry = {}
    
    def _generate_node_key(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Generate a deterministic key for node based on unique properties"""
        # Get entity definition
        entity_def = self.ontology_parser.parsed_ontology.get('entities', {}).get(entity_type, {})
        
        # Find unique properties 
        unique_props = []
        for prop_name, prop_def in entity_def.get('properties', {}).items():
            if prop_def.get('unique', False) and prop_name in properties and properties[prop_name] is not None:
                unique_props.append((prop_name, properties[prop_name]))
        
        if unique_props:
            # Sort by property name for deterministic key
            sorted_props = sorted(unique_props)
            return f"{entity_type}:" + ":".join(f"{k}={v}" for k, v in sorted_props)
        else:
            # If no unique properties, use all non-None properties
            non_empty_props = sorted([(k, v) for k, v in properties.items() if v is not None])
            if non_empty_props:
                return f"{entity_type}:" + ":".join(f"{k}={v}" for k, v in non_empty_props)
            else:
                # Last resort: use a random UUID
                return f"{entity_type}:uuid={uuid.uuid4()}"
  
    async def process_chunk(
        self,
        chunk: str,
        chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """Process single chunk with LLM extraction"""
        start_time = datetime.now()
        
        try:
            # Call LLM for extraction of whole ontology
            extraction_result = await self.llm_client.extract_from_chunk(
                chunk=chunk,
                response_model=self.pydantic_cls
            )
            
            # Initialize token usage and confidence if not set
            if not hasattr(extraction_result, 'tokens_used') or extraction_result.tokens_used is None:
                extraction_result.tokens_used = 0
                    
            if not hasattr(extraction_result, 'confidence_score') or extraction_result.confidence_score is None:
                extraction_result.confidence_score = 0.0
            
            # Track metrics
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Process extracted entities
            nodes = []
            relationships = []
            
            # Helper function to extract properties
            def extract_properties(item):
                """Extract meaningful properties while filtering out Pydantic metadata."""
                if item is None:
                    return {}
                    
                # Known Pydantic metadata fields to exclude
                metadata_fields = {
                    'model_computed_fields', 'model_config', 'model_fields', 
                    'model_fields_set', '__fields__', '__annotations__', 
                    '__field_defaults__', '__private_attributes__'
                }
                
                # Try multiple serialization methods
                try:
                    if hasattr(item, 'dict'):  # Pydantic v1
                        item_dict = item.dict()
                    elif hasattr(item, 'model_dump'):  # Pydantic v2
                        item_dict = item.model_dump()
                    else:
                        # Fallback to dict attrs
                        item_dict = {k: v for k, v in vars(item).items() if not k.startswith('_')}
                        
                    # Clean up metadata and None values
                    for field in metadata_fields:
                        item_dict.pop(field, None)
                    item_dict.pop('type', None)  # Remove type to avoid conflicts
                    return {k: v for k, v in item_dict.items() if v is not None}
                    
                except Exception:
                    # Manual property extraction
                    properties = {}
                    for attr_name in dir(item):
                        # Skip private attributes, methods, and metadata
                        if (attr_name.startswith('_') or 
                            callable(getattr(item, attr_name)) or 
                            attr_name in metadata_fields or 
                            attr_name == 'type'):
                            continue
                            
                        try:
                            value = getattr(item, attr_name)
                            if value is not None:
                                properties[attr_name] = value
                        except Exception:
                            pass
                    
                    return properties
            
            # Temporary node registry for this chunk (entity_type -> node_key -> node_id)
            # Used to link relationships to nodes within this extraction
            chunk_node_registry = {}
            
            # Phase 1: Process all entity lists
            for field_name in dir(extraction_result):
                if not field_name.endswith('_list') or field_name.startswith('_'):
                    continue
                    
                entity_list = getattr(extraction_result, field_name)
                if not isinstance(entity_list, list):
                    continue
                    
                entity_type = field_name[:-5]  # Remove '_list'
                if entity_type not in chunk_node_registry:
                    chunk_node_registry[entity_type] = {}
                
                for item in entity_list:
                    if not item:
                        continue
                        
                    # Extract properties
                    raw_properties = extract_properties(item)
                    properties = self._filter_properties_by_ontology(entity_type, raw_properties)
                    
                    # Skip if no valuable properties according to ontology
                    if not self._is_node_valuable(entity_type, properties):
                        continue
                    
                    # Generate node key
                    node_key = self._generate_node_key(entity_type, properties)
                    
                    # Create node with stable ID based on key
                    node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, node_key))
                    node = BaseNode(
                        id=node_id,
                        type=entity_type,
                        properties=properties,
                        provenance=NodeProvenance(
                            chunk_ids=[chunk_id],
                            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                            confidence_score=extraction_result.confidence_score
                        )
                    )
                    
                    # Register node in chunk registry
                    chunk_node_registry[entity_type][node_key] = node_id
                    nodes.append(node)
            
            # Phase 2: Process relationships
            for field_name in dir(extraction_result):
                if field_name.endswith('_list') or field_name.startswith('_') or '_' not in field_name:
                    continue
                    
                rel_list = getattr(extraction_result, field_name)
                if not isinstance(rel_list, list):
                    continue
                    
                parts = field_name.split('_', 1)
                if len(parts) != 2:
                    continue
                    
                source_type, rel_type = parts
                
                for rel_item in rel_list:
                    if not hasattr(rel_item, 'source') or not hasattr(rel_item, 'target'):
                        continue
                    
                    # Get source and target properties
                    source_raw_props = extract_properties(rel_item.source)

                    # Determine target type first before accessing properties
                    target_type = None
                    # From ontology definition
                    if source_type in self.ontology_parser.parsed_ontology.get('entities', {}):
                        relationships_def = self.ontology_parser.parsed_ontology['entities'][source_type].get('relationships', {})
                        if rel_type in relationships_def:
                            target_type = relationships_def[rel_type].get('target')

                    # From relationship model
                    if not target_type and source_type in self.relationship_models:
                        rel_model = self.relationship_models[source_type].get(rel_type)
                        if rel_model and hasattr(rel_model, '_target_type'):
                            target_type = rel_model._target_type

                    # From target class
                    if not target_type and hasattr(rel_item.target, '__class__'):
                        target_type = rel_item.target.__class__.__name__

                    # If we still can't determine target type, skip this relationship
                    if not target_type:
                        logger.warning(f"Skipping relationship {rel_type} - cannot determine target type")
                        continue
                    target_raw_props = extract_properties(rel_item.target)
                    
                    # Get source and target properties
                    source_props = self._filter_properties_by_ontology(source_type, source_raw_props)
                    target_props = self._filter_properties_by_ontology(target_type, target_raw_props)
                    
                    if not (self._is_node_valuable(source_type, source_props) or 
                            self._is_node_valuable(target_type, target_props)):
                        continue
                    
                    # Get target type
                    target_type = None
                    # From ontology definition
                    if source_type in self.ontology_parser.parsed_ontology.get('entities', {}):
                        relationships_def = self.ontology_parser.parsed_ontology['entities'][source_type].get('relationships', {})
                        if rel_type in relationships_def:
                            target_type = relationships_def[rel_type].get('target')
                    
                    # From relationship model
                    if not target_type and source_type in self.relationship_models:
                        rel_model = self.relationship_models[source_type].get(rel_type)
                        if rel_model and hasattr(rel_model, '_target_type'):
                            target_type = rel_model._target_type
                    
                    # From target class
                    if not target_type and hasattr(rel_item.target, '__class__'):
                        target_type = rel_item.target.__class__.__name__
                    
                    # Skip if we can't determine target type
                    if not target_type:
                        continue
                    
                    # Generate node keys
                    source_key = self._generate_node_key(source_type, source_props)
                    target_key = self._generate_node_key(target_type, target_props)
                    
                    # Get node IDs or generate new ones
                    if source_type in chunk_node_registry and source_key in chunk_node_registry[source_type]:
                        source_id = chunk_node_registry[source_type][source_key]
                    else:
                        # Create source node if it doesn't exist yet
                        source_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, source_key))
                        node = BaseNode(
                            id=source_id,
                            type=source_type,
                            properties=source_props,
                            provenance=NodeProvenance(
                                chunk_ids=[chunk_id],
                                extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                                confidence_score=extraction_result.confidence_score
                            )
                        )
                        nodes.append(node)
                        # Register the new node
                        if source_type not in chunk_node_registry:
                            chunk_node_registry[source_type] = {}
                        chunk_node_registry[source_type][source_key] = source_id
                    
                    if target_type in chunk_node_registry and target_key in chunk_node_registry[target_type]:
                        target_id = chunk_node_registry[target_type][target_key]
                    else:
                        # Create target node if it doesn't exist yet
                        target_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, target_key))
                        node = BaseNode(
                            id=target_id,
                            type=target_type,
                            properties=target_props,
                            provenance=NodeProvenance(
                                chunk_ids=[chunk_id],
                                extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                                confidence_score=extraction_result.confidence_score
                            )
                        )
                        nodes.append(node)
                        # Register the new node
                        if target_type not in chunk_node_registry:
                            chunk_node_registry[target_type] = {}
                        chunk_node_registry[target_type][target_key] = target_id
                    
                    # Get relationship properties
                    rel_properties = {}
                    if hasattr(rel_item, 'properties') and rel_item.properties:
                        rel_properties = extract_properties(rel_item.properties)
                    
                    # Create deterministic relationship ID
                    rel_key = f"{source_id}:{target_id}:{rel_type}"
                    rel_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rel_key))
                    
                    # Create relationship
                    rel = RelationshipInstance(
                        id=rel_id,
                        type=rel_type,
                        source_id=source_id,
                        target_id=target_id,
                        source_type=source_type,
                        target_type=target_type,
                        properties=rel_properties,
                        provenance=NodeProvenance(
                            chunk_ids=[chunk_id],
                            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                            confidence_score=extraction_result.confidence_score
                        )
                    )
                    relationships.append(rel)
            
            # Track metrics
            self.metrics.track_extraction(
                chunk_id=chunk_id,
                duration_ms=duration_ms,
                llm_token_usage={'total': extraction_result.tokens_used} if extraction_result.tokens_used else {},
                entity_count=len(nodes)
            )
            
            return {
                'chunk_id': chunk_id,
                'nodes': nodes,
                'relationships': relationships,
                'metrics': {
                    'tokens_used': extraction_result.tokens_used,
                    'confidence_score': extraction_result.confidence_score,
                    'extraction_timestamp': getattr(extraction_result, 'extraction_timestamp', 
                                                  datetime.now(timezone.utc).isoformat())
                }
            }
                
        except Exception as e:
            self.metrics.record_failure(chunk_id, str(e))
            logger.error(f"Extraction failed for chunk {chunk_id}: {str(e)}")
            traceback.print_exc()
            return None
    
    
    async def process_chunks(
        self,
        chunks: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[BaseModel, ExtractionMetrics]:
        """Process multiple chunks and build knowledge graph"""
        
        # Initialize metrics
        self.metrics.total_chunks = len(chunks)
        self.metrics.new_nodes = 0
        self.metrics.merged_nodes = 0
        self.metrics.total_relationships = 0
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            result = await self.process_chunk(chunk, chunk_id)
            
            if result:
                # Add extracted information to the graph
                self.add_extraction_result(result)
                    
            # Update progress
            if progress_callback:
                try:
                    await progress_callback(i + 1, len(chunks))
                except:
                    # Handle synchronous callback
                    progress_callback(i + 1, len(chunks))
        
        # Finalize the graph
        final_graph = self.finalize_graph()
        
        return final_graph, self.metrics
    
    def _find_node_by_id(self, entity_type: str, node_id: str) -> Optional[BaseNode]:
        """Find a node by ID in the graph"""
        entity_list_field = f"{entity_type}_list"
        if not hasattr(self.graph, entity_list_field):
            return None
            
        entity_list = getattr(self.graph, entity_list_field)
        for node in entity_list:
            if node.id == node_id:
                return node
                
        return None
    
    def _find_relationship_by_id(self, source_type: str, rel_type: str, rel_id: str) -> Optional[RelationshipInstance]:
        """Find a relationship by ID in the graph"""
        rel_field = f"{source_type}_{rel_type}"
        if not hasattr(self.graph, rel_field):
            return None
            
        rel_list = getattr(self.graph, rel_field)
        for rel in rel_list:
            if rel.id == rel_id:
                return rel
                
        return None
    
    def _find_matching_node(self, node: BaseNode) -> Optional[BaseNode]:
        """Find matching node in graph based on identity properties"""
        entity_type = node.type
        
        # Get entity definition from ontology
        entity_def = self.ontology_parser.parsed_ontology.get('entities', {}).get(entity_type)
        if not entity_def:
            return None
            
        # Find unique properties
        unique_props = []
        for prop_name, prop_def in entity_def.get('properties', {}).items():
            if prop_def.get('unique', False) and hasattr(node, prop_name) and getattr(node, prop_name) is not None:
                unique_props.append(prop_name)
        
        if not unique_props:
            return None
            
        # Search for matching node in the graph
        existing_nodes = []
        entity_list_field = f"{entity_type}_list"
        if hasattr(self.graph, entity_list_field):
            existing_nodes = getattr(self.graph, entity_list_field)
        
        for existing_node in existing_nodes:
            for prop in unique_props:
                if (
                    hasattr(existing_node, prop) and
                    getattr(existing_node, prop) == getattr(node, prop)
                ):
                    return existing_node
                    
        return None
    
    def _merge_nodes(self, existing_node: BaseNode, new_node: BaseNode) -> BaseNode:
        """Merge properties and provenance from two nodes"""
        # Create a copy of the existing node
        merged_node = copy.deepcopy(existing_node)
        
        # Merge provenance
        if hasattr(new_node, 'provenance') and new_node.provenance:
            if not hasattr(merged_node, 'provenance') or not merged_node.provenance:
                merged_node.provenance = new_node.provenance
            else:
                # Combine chunk IDs
                merged_node.provenance.chunk_ids.extend(new_node.provenance.chunk_ids)
                merged_node.provenance.chunk_ids = list(set(merged_node.provenance.chunk_ids))
        
        # Merge properties
        if hasattr(new_node, 'properties') and new_node.properties:
            if not hasattr(merged_node, 'properties') or not merged_node.properties:
                merged_node.properties = new_node.properties
            else:
                for key, value in new_node.properties.items():
                    if value is not None:
                        # Take new value if not in existing or existing is None
                        if key not in merged_node.properties or merged_node.properties[key] is None:
                            merged_node.properties[key] = value
                        # For overlapping values, take the one with higher confidence or longer value
                        elif (hasattr(new_node, 'confidence_score') and 
                              hasattr(existing_node, 'confidence_score') and
                              new_node.confidence_score and 
                              existing_node.confidence_score and
                              new_node.confidence_score > existing_node.confidence_score):
                            merged_node.properties[key] = value
                        elif (isinstance(value, str) and
                              isinstance(merged_node.properties[key], str) and
                              len(value) > len(merged_node.properties[key])):
                            merged_node.properties[key] = value
        
        # Update confidence score to max of both
        if (hasattr(new_node, 'confidence_score') and 
            hasattr(existing_node, 'confidence_score') and
            new_node.confidence_score and 
            existing_node.confidence_score):
            merged_node.confidence_score = max(
                new_node.confidence_score,
                existing_node.confidence_score
            )
            
        return merged_node
    
    def _merge_relationships(self, existing_rel: RelationshipInstance, new_rel: RelationshipInstance) -> RelationshipInstance:
        """Merge properties and provenance from two relationships"""
        # Create a copy of the existing relationship
        merged_rel = copy.deepcopy(existing_rel)
        
        # Merge provenance
        if hasattr(new_rel, 'provenance') and new_rel.provenance:
            if not hasattr(merged_rel, 'provenance') or not merged_rel.provenance:
                merged_rel.provenance = new_rel.provenance
            else:
                # Combine chunk IDs
                merged_rel.provenance.chunk_ids.extend(new_rel.provenance.chunk_ids)
                merged_rel.provenance.chunk_ids = list(set(merged_rel.provenance.chunk_ids))
        
        # Merge properties
        if hasattr(new_rel, 'properties') and new_rel.properties:
            if not hasattr(merged_rel, 'properties') or not merged_rel.properties:
                merged_rel.properties = new_rel.properties
            else:
                for key, value in new_rel.properties.items():
                    if value is not None:
                        if key not in merged_rel.properties or merged_rel.properties[key] is None:
                            merged_rel.properties[key] = value
                        # For overlapping values, take the one with higher confidence or longer value
                        elif (hasattr(new_rel, 'confidence_score') and 
                              hasattr(existing_rel, 'confidence_score') and
                              new_rel.confidence_score and 
                              existing_rel.confidence_score and
                              new_rel.confidence_score > existing_rel.confidence_score):
                            merged_rel.properties[key] = value
                        elif (isinstance(value, str) and
                              isinstance(merged_rel.properties[key], str) and
                              len(value) > len(merged_rel.properties[key])):
                            merged_rel.properties[key] = value
        
        # Update confidence score
        if (hasattr(new_rel, 'confidence_score') and 
            hasattr(existing_rel, 'confidence_score') and
            new_rel.confidence_score and 
            existing_rel.confidence_score):
            merged_rel.confidence_score = max(
                new_rel.confidence_score,
                existing_rel.confidence_score
            )
            
        return merged_rel
    
    async def build_graph_from_chunks(
        self,
        chunks: List[str],
        transform_id: str,
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BaseModel:
        """Process all chunks and build unified graph"""
        chunk_results = []
        
        # Process chunks with controlled concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(chunk: str, idx: int):
            async with semaphore:
                result = await self.process_chunk(chunk, f"{transform_id}_chunk_{idx}")
                print("***** Processed Chunk Result: *****")
                print(result)
                print("***** Processed Chunk Result ENDS *****")
                if progress_callback:
                    try:
                        await progress_callback(idx + 1, len(chunks))
                    except:
                        # Handle synchronous callback
                        progress_callback(idx + 1, len(chunks))
                return result
        
        # Process all chunks concurrently with controlled parallelism
        tasks = [
            process_with_semaphore(chunk, idx)
            for idx, chunk in enumerate(chunks)
        ]
        chunk_results = await asyncio.gather(*tasks)
        chunk_results = [r for r in chunk_results if r is not None]
        
        # Merge all extraction results
        for result in chunk_results:
            self.add_extraction_result(result)
            
        # Finalize the graph
        return self.finalize_graph()
    
    def add_extraction_result(self, result: Dict[str, Any]) -> None:
        """Add extraction result to graph with intelligent node merging and relationship deduplication"""
        # Initialize node and relationship registries if not exists
        for entity_type in self.entity_models:
            if entity_type not in self.node_registry:
                self.node_registry[entity_type] = {}
        
        # Track nodes used in relationships
        nodes_in_relationships = set()
        for rel in result['relationships']:
            nodes_in_relationships.add((rel.source_type, rel.source_id))
            nodes_in_relationships.add((rel.target_type, rel.target_id))
        
        # Process nodes
        for node in result['nodes']:
            # Filter properties by ontology
            if hasattr(node, 'properties') and node.properties:
                node.properties = self._filter_properties_by_ontology(node.type, node.properties)
                
            # Skip empty nodes not referenced in relationships
            if not self._is_node_valuable(node.type, node.properties) and (node.type, node.id) not in nodes_in_relationships:
                continue
                
            # Generate node key
            node_key = self._generate_node_key(node.type, node.properties)
            
            # Check if node already exists
            if node.type in self.node_registry and node_key in self.node_registry[node.type]:
                existing_id = self.node_registry[node.type][node_key]
                # Find the existing node
                existing_node = self._find_node_by_id(node.type, existing_id)
                if existing_node:
                    # Merge with existing node
                    merged_node = self._merge_nodes(existing_node, node)
                    self._update_node_in_graph(merged_node)
                    self.metrics.merged_nodes += 1
                else:
                    # Registry contains ID but node not found, add as new
                    self._add_node_to_graph(node)
                    self.node_registry[node.type][node_key] = node.id
                    self.metrics.new_nodes += 1
            else:
                # New node
                self._add_node_to_graph(node)
                if node.type not in self.node_registry:
                    self.node_registry[node.type] = {}
                self.node_registry[node.type][node_key] = node.id
                self.metrics.new_nodes += 1
        
        # Process relationships
        for rel in result['relationships']:
            # Skip if source or target doesn't exist
            if not self._node_exists(rel.source_type, rel.source_id) or not self._node_exists(rel.target_type, rel.target_id):
                continue
            
            # Check for duplicate relationship
            rel_key = (rel.source_id, rel.target_id, rel.type)
            if rel_key not in self.relationship_registry:
                self._add_relationship_to_graph(rel)
                self.relationship_registry[rel_key] = rel.id
                self.metrics.total_relationships += 1
            else:
                # Merge with existing relationship
                existing_rel = self._find_relationship_by_id(rel.source_type, rel.type, self.relationship_registry[rel_key])
                if existing_rel:
                    merged_rel = self._merge_relationships(existing_rel, rel)
                    self._update_relationship_in_graph(merged_rel)
    
    def _add_node_to_graph(self, node: BaseNode) -> None:
        """Add a node to the appropriate list in the graph"""
        entity_type = node.type
        entity_list_field = f"{entity_type}_list"
        
        if not hasattr(self.graph, entity_list_field):
            setattr(self.graph, entity_list_field, [])
        
        entity_list = getattr(self.graph, entity_list_field)
        entity_list.append(node)
    
    def _update_node_in_graph(self, node: BaseNode) -> None:
        """Update a node in the graph"""
        entity_type = node.type
        entity_list_field = f"{entity_type}_list"
        
        if not hasattr(self.graph, entity_list_field):
            setattr(self.graph, entity_list_field, [node])
            return
        
        entity_list = getattr(self.graph, entity_list_field)
        
        # Find the node to update
        for i, existing_node in enumerate(entity_list):
            if existing_node.id == node.id:
                entity_list[i] = node
                return
        
        # If node not found, append it
        entity_list.append(node)
    
    def _add_relationship_to_graph(self, relationship: RelationshipInstance) -> None:
        """Add a relationship to the graph"""
        source_type = relationship.source_type
        rel_type = relationship.type
        rel_field = f"{source_type}_{rel_type}"
        
        if not hasattr(self.graph, rel_field):
            setattr(self.graph, rel_field, [])
        
        rel_list = getattr(self.graph, rel_field)
        rel_list.append(relationship)
    
    def _update_relationship_in_graph(self, relationship: RelationshipInstance) -> None:
        """Update a relationship in the graph"""
        source_type = relationship.source_type
        rel_type = relationship.type
        rel_field = f"{source_type}_{rel_type}"
        
        if not hasattr(self.graph, rel_field):
            setattr(self.graph, rel_field, [relationship])
            return
        
        rel_list = getattr(self.graph, rel_field)
        
        # Find the relationship to update
        for i, existing_rel in enumerate(rel_list):
            if existing_rel.id == relationship.id:
                rel_list[i] = relationship
                return
        
        # If relationship not found, append it
        rel_list.append(relationship)
    
    def finalize_graph(self) -> KnowledgeGraph:
        """Validate and finalize the graph"""
        # Prune orphaned nodes (empty nodes not in relationships)
        self._prune_orphaned_nodes()
        
        # Validate relationships
        self._validate_relationships()
        
        # Set final metrics
        self.graph.tokens_used = self.metrics.total_tokens
        self.graph.confidence_score = self._calculate_average_confidence()
        self.graph.extraction_timestamp = datetime.now(timezone.utc).isoformat()
        
        return self.graph
    
    def _prune_orphaned_nodes(self) -> None:
        """Remove nodes with no ontology-defined properties that aren't referenced in any relationship"""
        # Build set of node IDs used in relationships
        nodes_in_relationships = set()
        
        # Check all relationship fields
        for field_name in dir(self.graph):
            if field_name.endswith('_list') or not '_' in field_name or field_name.startswith('_'):
                continue
                
            rel_list = getattr(self.graph, field_name)
            if not isinstance(rel_list, list):
                continue
                
            for rel in rel_list:
                nodes_in_relationships.add((rel.source_type, rel.source_id))
                nodes_in_relationships.add((rel.target_type, rel.target_id))
        
        # Prune nodes
        for entity_type in self.entity_models:
            entity_list_field = f"{entity_type}_list"
            if not hasattr(self.graph, entity_list_field):
                continue
                
            entity_list = getattr(self.graph, entity_list_field)
            pruned_list = []
            
            for node in entity_list:
                # Re-filter properties by ontology (in case they weren't filtered before)
                if hasattr(node, 'properties') and node.properties:
                    node.properties = self._filter_properties_by_ontology(entity_type, node.properties)
                
                # Keep if has ontology-defined properties or is referenced in relationships
                if self._is_node_valuable(entity_type, node.properties) or (entity_type, node.id) in nodes_in_relationships:
                    pruned_list.append(node)
                else:
                    logger.debug(f"Pruning node of type {entity_type} with ID {node.id} - no ontology-defined properties")
            
            # Update entity list
            setattr(self.graph, entity_list_field, pruned_list)
        
    def _validate_relationships(self) -> None:
        """Ensure all relationships point to existing nodes"""
        # For each relationship type
        for source_type, rel_dict in self.relationship_models.items():
            for rel_name, _ in rel_dict.items():
                rel_field = f"{source_type}_{rel_name}"
                
                if not hasattr(self.graph, rel_field):
                    continue
                    
                rel_list = getattr(self.graph, rel_field)
                valid_relationships = []
                
                for rel in rel_list:
                    # Check if source and target nodes exist
                    if self._node_exists(rel.source_type, rel.source_id) and self._node_exists(rel.target_type, rel.target_id):
                        valid_relationships.append(rel)
                    else:
                        # Count invalid relationship
                        if hasattr(self.metrics, 'invalid_relationships'):
                            self.metrics.invalid_relationships += 1
                
                # Update relationship list
                setattr(self.graph, rel_field, valid_relationships)
    
    def _node_exists(self, entity_type: str, node_id: str) -> bool:
        """Check if a node exists in the graph"""
        entity_list_field = f"{entity_type}_list"
        
        if not hasattr(self.graph, entity_list_field):
            return False
            
        entity_list = getattr(self.graph, entity_list_field)
        
        for node in entity_list:
            if node.id == node_id:
                return True
                
        return False
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence score across all nodes"""
        total_confidence = 0.0
        node_count = 0
        
        # Count nodes across all entity types
        for entity_type in self.entity_models.keys():
            entity_list_field = f"{entity_type}_list"
            
            if hasattr(self.graph, entity_list_field):
                entity_list = getattr(self.graph, entity_list_field)
                
                for node in entity_list:
                    if hasattr(node, 'confidence_score') and node.confidence_score is not None:
                        total_confidence += node.confidence_score
                        node_count += 1
        
        if node_count == 0:
            return 0.0
            
        return total_confidence / node_count
    
    def _filter_properties_by_ontology(self, entity_type: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter properties based on ontology definition.
        Only keeps properties that are defined in the ontology for this entity type.
        """
        # Get entity definition from ontology
        entity_def = self.ontology_parser.parsed_ontology.get('entities', {}).get(entity_type, {})
        if not entity_def:
            return {}
            
        # Get defined properties from ontology
        defined_properties = entity_def.get('properties', {})
        if not defined_properties:
            return {}
            
        # Filter properties to only include those defined in ontology
        filtered_props = {}
        for prop_name, prop_value in properties.items():
            if prop_name in defined_properties and prop_value is not None:
                filtered_props[prop_name] = prop_value
                
        return filtered_props
    
    def _is_node_valuable(self, entity_type: str, properties: Dict[str, Any]) -> bool:
        """
        Determine if a node has valuable information based on ontology-defined properties.
        A node is valuable if it has at least one non-null property defined in the ontology.
        """
        # Filter properties by ontology
        filtered_props = self._filter_properties_by_ontology(entity_type, properties)
        
        # Node is valuable if it has any ontology-defined properties with non-null values
        return len(filtered_props) > 0