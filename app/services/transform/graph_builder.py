from typing import Dict, List, Any, Type, Tuple, Optional, Callable, Union
import yaml
import copy
import asyncio
import uuid
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, create_model, Field
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
        self.extracted_triples = []
        self.entity_registry = {}
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
        """
        Generate a deterministic key for node based on unique properties.
        Improved to handle case insensitivity and normalize text values.
        """
        # Get entity definition
        entity_def = self.ontology_parser.parsed_ontology.get('entities', {}).get(entity_type, {})
        
        # Find unique properties 
        unique_props = []
        for prop_name, prop_def in entity_def.get('properties', {}).items():
            if prop_def.get('unique', False) and prop_name in properties and properties[prop_name] is not None:
                # Normalize string values: lowercase and strip whitespace
                value = properties[prop_name]
                if isinstance(value, str):
                    value = value.lower().strip()
                unique_props.append((prop_name, value))
        
        if unique_props:
            # Sort by property name for deterministic key
            sorted_props = sorted(unique_props)
            return f"{entity_type}:" + ":".join(f"{k}={v}" for k, v in sorted_props)
        else:
            # If no unique properties, use all non-None properties with normalization
            non_empty_props = []
            for k, v in properties.items():
                if v is not None:
                    # Normalize string values
                    if isinstance(v, str):
                        v = v.lower().strip()
                    non_empty_props.append((k, v))
            
            if non_empty_props:
                sorted_props = sorted(non_empty_props)
                return f"{entity_type}:" + ":".join(f"{k}={v}" for k, v in sorted_props)
            else:
                # Last resort: use a random UUID
                return f"{entity_type}:uuid={uuid.uuid4()}"
  
    async def process_chunk(
        self,
        chunk: str,
        chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """Process single chunk with LLM extraction using RDF context"""
        start_time = datetime.now()
        
        try:
            # Generate RDF context from previously extracted entities and relationships
            context = self._generate_rdf_context()
            
            # Call LLM for extraction of whole ontology with context
            extraction_result = await self.llm_client.extract_from_chunk(
                chunk=chunk,
                response_model=self.pydantic_cls,
                context=context
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
            
            # Prepare result
            result = {
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
            
            # Update RDF triples for future context
            self._update_extracted_triples(result)
            
            return result
                
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
        """Process multiple chunks and build knowledge graph with unified approach"""
        
        # Initialize metrics
        self.metrics.total_chunks = len(chunks)
        self.metrics.new_nodes = 0
        self.metrics.merged_nodes = 0
        self.metrics.total_relationships = 0
        
        # Store all chunk results for unified processing
        all_chunk_results = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            result = await self.process_chunk(chunk, chunk_id)
            
            if result:
                all_chunk_results.append(result)
                
                # Update RDF context for next chunk
                self._update_extracted_triples(result)
                    
            # Update progress
            if progress_callback:
                try:
                    await progress_callback(i + 1, len(chunks))
                except:
                    # Handle synchronous callback
                    progress_callback(i + 1, len(chunks))
        
        # Instead of adding results one by one, build unified graph
        self.graph = self.build_unified_graph(all_chunk_results)
        
        # Set merged node metrics
        total_nodes = sum(
            len(getattr(self.graph, f"{entity_type}_list", []))
            for entity_type in self.entity_models.keys()
        )
        total_raw_nodes = sum(len(result['nodes']) for result in all_chunk_results)
        self.metrics.merged_nodes = total_raw_nodes - total_nodes
        self.metrics.new_nodes = total_nodes
        
        # Count relationships
        total_relationships = sum(
            len(getattr(self.graph, field_name, []))
            for field_name in dir(self.graph)
            if not field_name.endswith('_list') and not field_name.startswith('_') and '_' in field_name
        )
        self.metrics.total_relationships = total_relationships
        
        # Finalize graph (validate relationships, etc.)
        final_graph = await self.finalize_graph()
        
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
        return await self.finalize_graph()
    
    def add_extraction_result(self, result: Dict[str, Any]) -> None:
        """Add extraction result to graph with node merging and RDF context update"""
        # Add all nodes with merging
        for node in result['nodes']:
            matching_node = self._find_matching_node(node)
            
            if matching_node:
                # Merge with existing node
                merged_node = self.merge_nodes(matching_node, node)
                self._update_node_in_graph(merged_node)
                self.metrics.merged_nodes += 1
            else:
                # Add as new node
                self._add_node_to_graph(node)
                self.metrics.new_nodes += 1
        
        # Add all relationships
        for rel in result['relationships']:
            self._add_relationship_to_graph(rel)
            self.metrics.total_relationships += 1
            
        # Update RDF triples with merged results
        # This ensures that future extractions will have context
        # about our current state of the graph
        self._update_final_triples()
    
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
    
    async def finalize_graph(self) -> KnowledgeGraph:
        """Validate and finalize the graph"""
        # Prune orphaned nodes (empty nodes not in relationships)
        self._prune_orphaned_nodes()
        
        # Validate relationships
        self._validate_relationships()
        
        # Set final metrics
        self.graph.tokens_used = self.metrics.total_tokens
        self.graph.confidence_score = self._calculate_average_confidence()
        self.graph.extraction_timestamp = datetime.now(timezone.utc).isoformat()
        self.graph = await self.post_process_graph(self.graph)
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
    
    def _generate_rdf_context(self, max_triples: int = 100) -> str:
        """
        Generate RDF-style context from previously extracted entities and relationships
        Limited to most recent/relevant triples to avoid context overload
        """
        if not hasattr(self, 'extracted_triples'):
            self.extracted_triples = []
            
        if not self.extracted_triples:
            return ""
        
        # If we have too many triples, prioritize most recent extractions
        context_triples = self.extracted_triples
        if len(context_triples) > max_triples:
            context_triples = context_triples[-max_triples:]
        
        # Format as RDF-style triples
        context = "# Previously extracted entities and relationships:\n"
        context += "\n".join(context_triples)
        return context

    def _create_safe_property_value(self, value) -> str:
        """Create a safely quoted string value for RDF triples"""
        # Convert to string and escape quotes
        safe_value = str(value)
        # Replace double quotes with single quotes
        safe_value = safe_value.replace('"', "'")
        return f"\"{safe_value}\""

    def _update_extracted_triples(self, result: Dict[str, Any]) -> None:
        """Update RDF triple context from extraction result"""
        if not hasattr(self, 'extracted_triples'):
            self.extracted_triples = []
            
        # Process nodes first
        for node in result.get('nodes', []):
            entity_type = node.type
            entity_id = node.id
            
            # Generate triples for entity properties
            for prop_name, prop_value in node.properties.items():
                if prop_value is not None:
                    # Format as: <entity_type>(<entity_id>) <property> <value>
                    safe_value = self._create_safe_property_value(prop_value)
                    triple = f"{entity_type}({entity_id}) hasProperty:{prop_name} {safe_value}"
                    self.extracted_triples.append(triple)
        
        # Process relationships
        for rel in result.get('relationships', []):
            source_type = rel.source_type
            source_id = rel.source_id
            target_type = rel.target_type
            target_id = rel.target_id
            rel_type = rel.type
            
            # Format as: <source_type>(<source_id>) <relationship> <target_type>(<target_id>)
            triple = f"{source_type}({source_id}) {rel_type} {target_type}({target_id})"
            self.extracted_triples.append(triple)
            
            # Add relationship properties if any
            if hasattr(rel, 'properties') and rel.properties:
                for prop_name, prop_value in rel.properties.items():
                    if prop_value is not None:
                        # Format as: Relationship(<rel_id>) <property> <value>
                        safe_value = self._create_safe_property_value(prop_value)
                        rel_triple = f"Relationship({rel.id}) hasProperty:{prop_name} {safe_value}"
                        self.extracted_triples.append(rel_triple)
                        
    def _update_final_triples(self):
        """Update RDF triples based on the final state of the graph"""
        # Clear existing triples to rebuild from current graph state
        self.extracted_triples = []
        
        # Process all entity types in the graph
        for entity_type in self.entity_models.keys():
            entity_list_field = f"{entity_type}_list"
            if not hasattr(self.graph, entity_list_field):
                continue
                
            entity_list = getattr(self.graph, entity_list_field)
            if not isinstance(entity_list, list):
                continue
                
            # Add triples for each entity
            for node in entity_list:
                if not hasattr(node, 'id') or not hasattr(node, 'properties'):
                    continue
                    
                entity_id = node.id
                
                # Generate triples for entity properties
                for prop_name, prop_value in node.properties.items():
                    if prop_value is not None:
                        # Format as: <entity_type>(<entity_id>) <property> <value>
                        # Use safe string replacement
                        value_str = str(prop_value).replace('"', r'\"')
                        triple = f"{entity_type}({entity_id}) hasProperty:{prop_name} \"{value_str}\""
                        self.extracted_triples.append(triple)
        
        # Process all relationship types
        for field_name in dir(self.graph):
            if field_name.endswith('_list') or field_name.startswith('_') or '_' not in field_name:
                continue
                
            rel_list = getattr(self.graph, field_name)
            if not isinstance(rel_list, list):
                continue
                
            # Parse relationship field
            parts = field_name.split('_', 1)
            if len(parts) != 2:
                continue
                
            source_type, rel_type = parts
            
            # Add triples for each relationship
            for rel in rel_list:
                if not hasattr(rel, 'source_id') or not hasattr(rel, 'target_id'):
                    continue
                    
                source_id = rel.source_id
                target_id = rel.target_id
                target_type = rel.target_type
                
                # Format as: <source_type>(<source_id>) <relationship> <target_type>(<target_id>)
                triple = f"{source_type}({source_id}) {rel_type} {target_type}({target_id})"
                self.extracted_triples.append(triple)
                
                # Add relationship properties if any
                if hasattr(rel, 'properties') and rel.properties:
                    for prop_name, prop_value in rel.properties.items():
                        if prop_value is not None:
                            # Format as: Relationship(<rel_id>) <property> <value>
                            value_str = str(prop_value).replace('"', r'\"')
                            rel_triple = f"Relationship({rel.id}) hasProperty:{prop_name} \"{value_str}\""
                            self.extracted_triples.append(rel_triple)
        
        # Limit total number of triples to avoid context explosion
        max_triples = 500  # Adjust based on your model's context window
        if len(self.extracted_triples) > max_triples:
            # Prioritize:
            # 1. Entity identity triples (name properties)
            # 2. Relationship triples
            # 3. Other property triples
            
            # First, collect identity triples (nodes with name properties)
            identity_triples = [t for t in self.extracted_triples if "hasProperty:name" in t]
            
            # Then, relationship triples (not property triples)
            relationship_triples = [t for t in self.extracted_triples 
                                if not t.startswith("Relationship") and 
                                    not "hasProperty:" in t]
            
            # Finally, other property triples
            other_triples = [t for t in self.extracted_triples
                        if t not in identity_triples and 
                            t not in relationship_triples]
            
            # Combine with priorities
            self.extracted_triples = (
                identity_triples + 
                relationship_triples + 
                other_triples
            )[:max_triples]
            
    def build_unified_graph(self, chunk_results: List[Dict[str, Any]]) -> KnowledgeGraph:
        """
        Build a unified knowledge graph from all chunk results
        using ontology definitions for better coherence and quality
        """
        # Create fresh graph instance
        unified_graph = self.pydantic_cls()
        
        # 1. First pass: collect and merge all nodes by entity type
        for entity_type in self.entity_models.keys():
            # Get all nodes of this type across all chunks
            all_nodes_of_type = []
            for result in chunk_results:
                for node in result.get('nodes', []):
                    if node.type == entity_type:
                        # Filter out empty nodes first
                        if self._is_node_valuable(entity_type, node.properties):
                            all_nodes_of_type.append(node)
            
            if not all_nodes_of_type:
                continue
                
            # Apply improved node merging with normalization
            merged_nodes = self._merge_nodes_by_ontology(entity_type, all_nodes_of_type)
            
            # Add to unified graph
            setattr(unified_graph, f"{entity_type}_list", merged_nodes)
        
        # Store temporary reference to graph for relationship processing
        self.graph = unified_graph
        
        # 2. Process relationships with inference capabilities
        processed_relationships = self._process_relationships_with_inference(chunk_results)
        
        # 3. Group relationships by type and add to graph
        relationship_groups = {}
        for rel in processed_relationships:
            group_key = f"{rel.source_type}_{rel.type}"
            if group_key not in relationship_groups:
                relationship_groups[group_key] = []
            relationship_groups[group_key].append(rel)
        
        # Add processed relationships to graph
        for group_key, rels in relationship_groups.items():
            setattr(unified_graph, group_key, rels)
        
        # 4. Calculate and set graph-level metrics
        tokens_used = sum(
            result.get('metrics', {}).get('tokens_used', 0) 
            for result in chunk_results if result
        )
        
        # Calculate average confidence from all entities
        confidence_values = []
        for entity_type in self.entity_models.keys():
            entity_list_field = f"{entity_type}_list"
            if hasattr(unified_graph, entity_list_field):
                entity_list = getattr(unified_graph, entity_list_field)
                for entity in entity_list:
                    if hasattr(entity, 'provenance') and entity.provenance and entity.provenance.confidence_score:
                        confidence_values.append(entity.provenance.confidence_score)
        
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        
        # Set graph metrics
        unified_graph.tokens_used = tokens_used
        unified_graph.confidence_score = avg_confidence
        unified_graph.extraction_timestamp = datetime.now(timezone.utc).isoformat()
        
        return unified_graph
        
        
    def _merge_nodes_by_ontology(self, entity_type: str, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        Merge nodes based on ontology-defined properties.
        Improved to handle case insensitivity and perform smarter merging.
        """
        if not nodes:
            return []
            
        # Get unique properties from ontology
        entity_def = self.ontology_parser.parsed_ontology.get('entities', {}).get(entity_type, {})
        unique_props = []
        
        for prop_name, prop_def in entity_def.get('properties', {}).items():
            if prop_def.get('unique', False):
                unique_props.append(prop_name)
        
        # Also consider name properties as potentially unique for deduplication
        name_props = []
        for prop_name in entity_def.get('properties', {}):
            if prop_name.lower() in ['name', 'title', 'label']:
                name_props.append(prop_name)
        
        # Combine unique and name properties for better deduplication
        dedup_props = list(set(unique_props + name_props))
        
        # Group nodes by their identifying properties
        node_groups = {}
        
        for node in nodes:
            # Generate key based on unique properties with normalization
            key_parts = []
            
            # Try using ontology-defined unique properties first
            for prop in dedup_props:
                if prop in node.properties and node.properties[prop] is not None:
                    # Normalize for comparison (lowercase and strip for strings)
                    value = node.properties[prop]
                    if isinstance(value, str):
                        value = value.lower().strip()
                    key_parts.append(f"{prop}:{value}")
            
            if not key_parts and node.properties:
                # Fallback: use all non-empty properties with normalization
                for k, v in node.properties.items():
                    if v is not None:
                        if isinstance(v, str):
                            v = v.lower().strip()
                        key_parts.append(f"{k}:{v}")
            
            if key_parts:
                group_key = "|".join(sorted(key_parts))
            else:
                group_key = f"id:{node.id}"
            
            if group_key not in node_groups:
                node_groups[group_key] = []
            node_groups[group_key].append(node)
        
        # Merge nodes in each group
        merged_nodes = []
        
        for group_key, group_nodes in node_groups.items():
            if not group_nodes:
                continue
                
            if len(group_nodes) == 1:
                merged_nodes.append(group_nodes[0])
                continue
                
            # Sort by confidence, property completeness, and chunk coverage
            sorted_nodes = sorted(
                group_nodes,
                key=lambda n: (
                    n.provenance.confidence_score if n.provenance else 0,
                    sum(1 for p in n.properties.values() if p is not None),
                    len(n.provenance.chunk_ids if n.provenance else [])
                ),
                reverse=True
            )
            
            # Use highest quality node as base
            best_node = copy.deepcopy(sorted_nodes[0])
            
            # Merge in properties from other nodes
            for other in sorted_nodes[1:]:
                # Merge provenance with confidence score handling
                if best_node.provenance and other.provenance:
                    # Combine chunk IDs
                    best_node.provenance.chunk_ids.extend(other.provenance.chunk_ids)
                    best_node.provenance.chunk_ids = list(set(best_node.provenance.chunk_ids))
                    
                    # Take higher confidence score
                    if (other.provenance.confidence_score and 
                        (not best_node.provenance.confidence_score or 
                        other.provenance.confidence_score > best_node.provenance.confidence_score)):
                        best_node.provenance.confidence_score = other.provenance.confidence_score
                
                # Fill in missing properties and prefer non-null values
                for prop, value in other.properties.items():
                    # Skip null values
                    if value is None:
                        continue
                        
                    # For string properties, prefer longer/more detailed values
                    if isinstance(value, str) and prop in best_node.properties:
                        # Use proper capitalization from the highest confidence node
                        best_value = best_node.properties[prop]
                        if best_value is None or (isinstance(best_value, str) and len(value) > len(best_value)):
                            best_node.properties[prop] = value
                    # Otherwise just fill in missing properties
                    elif prop not in best_node.properties or best_node.properties[prop] is None:
                        best_node.properties[prop] = value
            
            # Ensure consistent capitalization for name-like properties
            for prop in name_props:
                if prop in best_node.properties and isinstance(best_node.properties[prop], str):
                    # Use title case for names and titles
                    best_node.properties[prop] = self._normalize_entity_name(best_node.properties[prop])
            
            merged_nodes.append(best_node)
        
        return merged_nodes
    
    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity names for consistent capitalization.
        Handles business terms, acronyms, and common patterns.
        """
        if not name:
            return name
            
        # Preserve common acronyms
        acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
        acronyms = acronym_pattern.findall(name)
        
        # Convert to title case first
        name = name.title()
        
        # Restore acronyms
        for acronym in acronyms:
            name = re.sub(r'\b' + acronym.title() + r'\b', acronym, name)
        
        # Handle common lowercase words (like articles and prepositions)
        for word in ['And', 'Of', 'The', 'In', 'On', 'For', 'With', 'To', 'By']:
            name = re.sub(r'\b' + word + r'\b', word.lower(), name)
        
        return name
    
    async def post_process_graph(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """
        Apply post-processing to improve graph quality using LLM validation
        This helps ensure consistency in naming, capitalization, and property values
        """
        # 1. Create consistency groups for similar entities
        # Group entities of the same type with similar names
        consistency_groups = {}
        
        for entity_type in self.entity_models.keys():
            entity_list_field = f"{entity_type}_list"
            if not hasattr(graph, entity_list_field):
                continue
                
            entity_list = getattr(graph, entity_list_field)
            
            # Skip if no entities
            if not entity_list:
                continue
                
            # Group by normalized names
            groups_by_name = {}
            for entity in entity_list:
                # Get entity name if available
                entity_name = None
                for name_field in ['name', 'title', 'label']:
                    if name_field in entity.properties and entity.properties[name_field]:
                        entity_name = str(entity.properties[name_field])
                        break
                        
                if not entity_name:
                    continue
                    
                # Normalize for grouping
                norm_name = entity_name.lower().strip()
                
                # Handle fuzzy matching by checking similarity
                matched = False
                for existing_name in list(groups_by_name.keys()):
                    # Simple similarity: check if one is contained in the other
                    # or if they share significant words
                    if (norm_name in existing_name or 
                        existing_name in norm_name or
                        self._calculate_word_overlap(norm_name, existing_name) > 0.7):
                        groups_by_name[existing_name].append(entity)
                        matched = True
                        break
                        
                if not matched:
                    groups_by_name[norm_name] = [entity]
                    
            # Store groups with multiple entities for consistency check
            for name, entities in groups_by_name.items():
                if len(entities) > 1:
                    group_key = f"{entity_type}:{name}"
                    consistency_groups[group_key] = entities
        
        # 2. Apply LLM-based standardization to groups that need it
        for group_key, entities in consistency_groups.items():
            entity_type, _ = group_key.split(':', 1)
            
            # Skip small groups
            if len(entities) < 2:
                continue
                
            # Prepare entity data for LLM
            entity_data = []
            for idx, entity in enumerate(entities):
                entity_data.append({
                    'id': entity.id,
                    'index': idx,
                    'properties': entity.properties,
                    'confidence': entity.provenance.confidence_score if hasattr(entity, 'provenance') else 0.0
                })
                
            # Call LLM to standardize properties
            try:
                standardized = await self._standardize_entity_group(entity_type, entity_data)
                if standardized and 'standardized_properties' in standardized:
                    # Apply standardized properties back to entities
                    standard_props = standardized['standardized_properties']
                    for entity in entities:
                        # Update properties while preserving entity-specific ones
                        for prop, value in standard_props.items():
                            if value is not None:
                                entity.properties[prop] = value
            except Exception as e:
                logger.warning(f"Error in LLM standardization for {group_key}: {str(e)}")
        
        # 3. Enhance property consistency
        await self._enhance_property_consistency(graph)
        
        # 4. Enhance relationship confidence scores 
        self._enhance_relationship_confidence(graph)
        
        # 5. Validate and ensure relationship consistency
        self._validate_relationship_consistency(graph)
        
        return graph

    async def _standardize_entity_group(self, entity_type: str, entity_data: List[Dict]) -> Optional[Dict]:
        """
        Use LLM to standardize property values across similar entities
        """
        if not entity_data:
            return None
            
        # Sort by confidence score to prioritize high-confidence entities
        sorted_entities = sorted(entity_data, key=lambda e: e.get('confidence', 0), reverse=True)
        
        # Create prompt for LLM standardization
        # This doesn't use any hardcoded schema - works with any entity type
        prompt = f"""
        You are processing a group of similar {entity_type} entities that should be standardized.
        Below are the properties of these entities:
        
        {json.dumps(sorted_entities, indent=2)}
        
        Please standardize the properties by:
        1. Using the most accurate/complete value for each property
        2. Fixing capitalization and formatting inconsistencies 
        3. Using proper business terminology
        
        Return only a JSON object with the standardized properties:
        {{
        "standardized_properties": {{
            "property1": "standardized value",
            "property2": "standardized value"
        }}
        }}
        """
        
        # Call LLM for standardization
        try:
            result = await self.llm_client.generate_text(prompt=prompt, json_response=True)
            return result
        except Exception as e:
            logger.warning(f"Error in LLM standardization: {str(e)}")
            traceback.print_exc()
            return None

    async def _enhance_property_consistency(self, graph: KnowledgeGraph) -> None:
        """
        Enhance property naming consistency across entity types.
        This uses the ontology definitions to ensure property names match expectations.
        """
        # Get expected property names from ontology
        property_standards = {}
        ontology = self.ontology_parser.parsed_ontology
        
        for entity_type, entity_def in ontology.get('entities', {}).items():
            property_standards[entity_type] = {}
            for prop_name, prop_def in entity_def.get('properties', {}).items():
                # Store canonical property name
                property_standards[entity_type][prop_name.lower()] = prop_name
        
        # For each entity type
        for entity_type in self.entity_models.keys():
            entity_list_field = f"{entity_type}_list"
            if not hasattr(graph, entity_list_field):
                continue
                
            # Skip if no property standards for this type
            if entity_type not in property_standards:
                continue
                
            entity_list = getattr(graph, entity_list_field)
            type_standards = property_standards[entity_type]
            
            # For each entity, standardize property names
            for entity in entity_list:
                if not hasattr(entity, 'properties') or not entity.properties:
                    continue
                    
                standardized_props = {}
                for prop, value in entity.properties.items():
                    # Skip null values
                    if value is None:
                        continue
                        
                    # Check if we have a canonical name for this property
                    canonical_name = type_standards.get(prop.lower())
                    if canonical_name:
                        standardized_props[canonical_name] = value
                    else:
                        # Keep original if no standard defined
                        standardized_props[prop] = value
                        
                # Update properties
                entity.properties = standardized_props

    def _validate_relationship_consistency(self, graph: KnowledgeGraph) -> None:
        """
        Ensure relationships are consistent with ontology definitions.
        """
        ontology = self.ontology_parser.parsed_ontology
        
        # For each relationship field in graph
        for field_name in dir(graph):
            if field_name.endswith('_list') or not '_' in field_name or field_name.startswith('_'):
                continue
                
            # Try to parse relationship field
            try:
                # Extract source type and relationship type from field name
                parts = field_name.split('_', 1)
                if len(parts) != 2:
                    continue
                    
                source_type, rel_type = parts
                
                # Check if this is a valid relationship in ontology
                entity_def = ontology.get('entities', {}).get(source_type, {})
                rel_def = entity_def.get('relationships', {}).get(rel_type)
                
                if not rel_def:
                    continue  # Not defined in ontology
                    
                target_type = rel_def.get('target')
                if not target_type:
                    continue
                    
                # Get relationships
                rel_list = getattr(graph, field_name)
                if not isinstance(rel_list, list):
                    continue
                    
                valid_rels = []
                for rel in rel_list:
                    # Validate relationship with ontology
                    if (hasattr(rel, 'source_type') and hasattr(rel, 'target_type') and
                        rel.source_type == source_type and rel.target_type == target_type):
                        valid_rels.append(rel)
                        
                # Update with valid relationships
                setattr(graph, field_name, valid_rels)
                
            except Exception as e:
                logger.warning(f"Error validating relationship {field_name}: {str(e)}")
                continue

    def _enhance_relationship_confidence(self, graph: KnowledgeGraph) -> None:
        """
        Enhance relationship confidence scores based on connected entities.
        """
        for field_name in dir(graph):
            if field_name.endswith('_list') or not '_' in field_name or field_name.startswith('_'):
                continue
                
            rel_list = getattr(graph, field_name)
            if not isinstance(rel_list, list):
                continue
                
            for rel in rel_list:
                if not hasattr(rel, 'provenance') or not rel.provenance:
                    continue
                    
                # Skip if already has good confidence
                if rel.provenance.confidence_score and rel.provenance.confidence_score > 0.5:
                    continue
                    
                # Find source and target entities
                source_entity = self._find_node_by_id(rel.source_type, rel.source_id)
                target_entity = self._find_node_by_id(rel.target_type, rel.target_id)
                
                if not source_entity or not target_entity:
                    continue
                    
                # Calculate new confidence from entities
                source_conf = (source_entity.provenance.confidence_score 
                            if hasattr(source_entity, 'provenance') else 0.0)
                target_conf = (target_entity.provenance.confidence_score 
                            if hasattr(target_entity, 'provenance') else 0.0)
                            
                # Use average of entity confidence scores
                if source_conf > 0 or target_conf > 0:
                    entity_conf = (source_conf + target_conf) / 2
                    # Update relationship confidence (weighted average with original)
                    original_conf = rel.provenance.confidence_score or 0.0
                    new_conf = (original_conf + entity_conf * 2) / 3  # Weight entity confidence higher
                    rel.provenance.confidence_score = new_conf

    def _resolve_node_reference(self, node_id: str, entity_type: str, node_registry: Dict) -> Optional[str]:
        """Resolve node ID to canonical ID if found in registry"""
        # Direct match in registry values
        for registry_values in node_registry.get(entity_type, {}).values():
            if registry_values == node_id:
                return node_id
                
        # Try to find node and generate key
        entity_list_field = f"{entity_type}_list"
        if not hasattr(self.graph, entity_list_field):
            return node_id
            
        node_list = getattr(self.graph, entity_list_field)
        for node in node_list:
            if node.id == node_id:
                key = self._generate_node_key(entity_type, node.properties)
                if key in node_registry.get(entity_type, {}):
                    return node_registry[entity_type][key]
        
        return node_id
    
    def _process_relationships_with_inference(self, chunk_results: List[Dict[str, Any]]) -> List[RelationshipInstance]:
        """
        Process relationships with inference capabilities to improve completeness.
        This function analyzes extracted relationships and infers missing connections.
        """
        # Collect all relationships from chunks
        all_relationships = []
        for result in chunk_results:
            if 'relationships' in result:
                all_relationships.extend(result['relationships'])
        
        # Build entity lookup tables for faster access
        entity_by_id = {}
        entity_by_type_name = {}
        
        for entity_type in self.entity_models.keys():
            entity_list_field = f"{entity_type}_list"
            if not hasattr(self.graph, entity_list_field):
                continue
                
            for node in getattr(self.graph, entity_list_field):
                # Store by ID
                entity_by_id[node.id] = node
                
                # Store by type and name (if available)
                name_props = ['name', 'title', 'label']
                for prop in name_props:
                    if prop in node.properties and node.properties[prop]:
                        normalized_name = str(node.properties[prop]).lower().strip()
                        key = f"{node.type}:{normalized_name}"
                        entity_by_type_name[key] = node
        
        # Process and validate relationships
        processed_relationships = []
        seen_relationship_keys = set()
        
        for relationship in all_relationships:
            # Skip if missing essential fields
            if not (hasattr(relationship, 'source_id') and 
                    hasattr(relationship, 'target_id') and
                    hasattr(relationship, 'source_type') and
                    hasattr(relationship, 'target_type') and
                    hasattr(relationship, 'type')):
                continue
                
            source_id = relationship.source_id
            target_id = relationship.target_id
            source_type = relationship.source_type
            target_type = relationship.target_type
            rel_type = relationship.type
            
            # Resolve source and target entities
            source_entity = entity_by_id.get(source_id)
            target_entity = entity_by_id.get(target_id)
            
            # If entities not found, try to resolve through inference
            if not source_entity and hasattr(relationship, 'source') and hasattr(relationship.source, 'properties'):
                source_entity = self._infer_entity_from_properties(relationship.source, source_type, entity_by_type_name)
                if source_entity:
                    source_id = source_entity.id
                    
            if not target_entity and hasattr(relationship, 'target') and hasattr(relationship.target, 'properties'):
                target_entity = self._infer_entity_from_properties(relationship.target, target_type, entity_by_type_name)
                if target_entity:
                    target_id = target_entity.id
            
            # Skip if we couldn't resolve both entities
            if not source_entity or not target_entity:
                continue
                
            # Create a unique key for deduplication
            rel_key = f"{source_id}:{target_id}:{rel_type}"
            if rel_key in seen_relationship_keys:
                continue
                
            seen_relationship_keys.add(rel_key)
            
            # Create relationship instance
            rel_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rel_key))
            rel_properties = {}
            if hasattr(relationship, 'properties') and relationship.properties:
                rel_properties = relationship.properties
                
            rel_confidence = 0.0
            rel_chunks = []
            
            # Determine confidence from source entities or relationship
            if hasattr(relationship, 'provenance') and relationship.provenance:
                rel_confidence = relationship.provenance.confidence_score or 0.0
                if hasattr(relationship.provenance, 'chunk_ids'):
                    rel_chunks = relationship.provenance.chunk_ids
            
            # If relationship has no confidence, derive from connected entities
            if rel_confidence == 0.0:
                source_confidence = (source_entity.provenance.confidence_score 
                                    if hasattr(source_entity, 'provenance') and source_entity.provenance else 0.0)
                target_confidence = (target_entity.provenance.confidence_score 
                                    if hasattr(target_entity, 'provenance') and target_entity.provenance else 0.0)
                
                # Average of source and target confidences
                if source_confidence > 0 or target_confidence > 0:
                    total = source_confidence + target_confidence
                    divisor = (1 if source_confidence > 0 else 0) + (1 if target_confidence > 0 else 0)
                    rel_confidence = total / divisor if divisor > 0 else 0.0
            
            # Combine chunk ids from entities if relationship doesn't have any
            if not rel_chunks:
                source_chunks = (source_entity.provenance.chunk_ids 
                                if hasattr(source_entity, 'provenance') and source_entity.provenance else [])
                target_chunks = (target_entity.provenance.chunk_ids 
                                if hasattr(target_entity, 'provenance') and target_entity.provenance else [])
                rel_chunks = list(set(source_chunks + target_chunks))
            
            # Create final relationship
            processed_rel = RelationshipInstance(
                id=rel_id,
                type=rel_type,
                source_id=source_id,
                target_id=target_id,
                source_type=source_type,
                target_type=target_type,
                properties=rel_properties,
                provenance=NodeProvenance(
                    chunk_ids=rel_chunks,
                    extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                    confidence_score=rel_confidence
                )
            )
            
            processed_relationships.append(processed_rel)
        
        # Dynamically infer relationships based on ontology definitions
        inferred_relationships = self._infer_missing_relationships(entity_by_id)
        for inferred_rel in inferred_relationships:
            rel_key = f"{inferred_rel.source_id}:{inferred_rel.target_id}:{inferred_rel.type}"
            if rel_key not in seen_relationship_keys:
                seen_relationship_keys.add(rel_key)
                processed_relationships.append(inferred_rel)
        
        return processed_relationships

    def _infer_entity_from_properties(self, entity, entity_type, entity_lookup):
        """
        Infer entity from properties when direct ID lookup fails.
        Uses name matching with normalization for better accuracy.
        """
        if not hasattr(entity, 'properties') or not entity.properties:
            return None
        
        # Try to find by name properties
        name_props = ['name', 'title', 'label']
        for prop in name_props:
            if prop in entity.properties and entity.properties[prop]:
                normalized_name = str(entity.properties[prop]).lower().strip()
                lookup_key = f"{entity_type}:{normalized_name}"
                if lookup_key in entity_lookup:
                    return entity_lookup[lookup_key]
        
        # If no match by name, try other significant properties
        # This is a simplified approach - could be extended with more sophisticated matching
        for prop, value in entity.properties.items():
            if value and isinstance(value, str) and len(value) > 3:  # Only consider meaningful string values
                normalized_value = value.lower().strip()
                for key, candidate in entity_lookup.items():
                    if key.startswith(f"{entity_type}:"):
                        # Check if this value appears in any of the stored entity's properties
                        for _, candidate_value in candidate.properties.items():
                            if (isinstance(candidate_value, str) and 
                                normalized_value in candidate_value.lower()):
                                return candidate
        
        return None

    def _infer_missing_relationships(self, entity_by_id):
        """
        Dynamically infer missing relationships based on ontology definitions and content analysis.
        Uses the ontology to determine valid relationship types without hardcoding.
        """
        inferred_relationships = []
        
        # Group entities by type for easier access
        entities_by_type = {}
        for entity_id, entity in entity_by_id.items():
            entity_type = entity.type
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Get relationship types from ontology
        ontology = self.ontology_parser.parsed_ontology
        
        # For each entity type in the ontology, check its possible relationships
        for source_type, entity_def in ontology.get('entities', {}).items():
            # Skip if we don't have entities of this type
            if source_type not in entities_by_type:
                continue
                
            relationships = entity_def.get('relationships', {})
            for rel_type, rel_def in relationships.items():
                target_type = rel_def.get('target')
                
                # Skip if we don't have target entities
                if not target_type or target_type not in entities_by_type:
                    continue
                    
                # For each source entity, analyze which target entities it might relate to
                for source_entity in entities_by_type[source_type]:
                    # Skip entities without properties
                    if not hasattr(source_entity, 'properties') or not source_entity.properties:
                        continue
                        
                    source_id = source_entity.id
                    source_text = self._get_entity_text(source_entity)
                    
                    # Check which target entities might be related based on textual similarity
                    for target_entity in entities_by_type[target_type]:
                        target_id = target_entity.id
                        
                        # Skip if no properties
                        if not hasattr(target_entity, 'properties') or not target_entity.properties:
                            continue
                            
                        # Skip if relationship already exists
                        rel_key = f"{source_id}:{target_id}:{rel_type}"
                        rel_field = f"{source_type}_{rel_type}"
                        
                        existing_connection = False
                        if hasattr(self.graph, rel_field):
                            rel_list = getattr(self.graph, rel_field)
                            for rel in rel_list:
                                if rel.source_id == source_id and rel.target_id == target_id:
                                    existing_connection = True
                                    break
                                    
                        if existing_connection:
                            continue
                            
                        # Determine if relationship should be inferred
                        should_infer = False
                        inference_confidence = 0.0
                        
                        # Method 1: Textual similarity
                        target_text = self._get_entity_text(target_entity)
                        text_overlap = self._calculate_text_overlap(source_text, target_text)
                        
                        # If significant overlap, infer relationship
                        if text_overlap > 0.3:  # Threshold can be tuned
                            should_infer = True
                            inference_confidence = min(0.7, text_overlap)  # Cap confidence
                        
                        # Method 2: Property matching (for relationships that connect related entities)
                        property_match = self._check_property_matches(source_entity, target_entity)
                        if property_match > 0:
                            should_infer = True
                            inference_confidence = max(inference_confidence, min(0.8, property_match))
                        
                        # Create relationship if inference criteria met
                        if should_infer:
                            rel_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rel_key))
                            
                            # Combine confidence and chunk information
                            source_confidence = source_entity.provenance.confidence_score if hasattr(source_entity, 'provenance') else 0.0
                            target_confidence = target_entity.provenance.confidence_score if hasattr(target_entity, 'provenance') else 0.0
                            
                            # Base confidence on entity confidence and inference confidence
                            entity_confidence = (source_confidence + target_confidence) / 2 if (source_confidence or target_confidence) else 0.0
                            combined_confidence = (entity_confidence + inference_confidence) / 2
                            
                            # Combine chunks
                            source_chunks = source_entity.provenance.chunk_ids if hasattr(source_entity, 'provenance') else []
                            target_chunks = target_entity.provenance.chunk_ids if hasattr(target_entity, 'provenance') else []
                            combined_chunks = list(set(source_chunks + target_chunks))
                            
                            new_rel = RelationshipInstance(
                                id=rel_id,
                                type=rel_type,
                                source_id=source_id,
                                target_id=target_id,
                                source_type=source_type,
                                target_type=target_type,
                                properties={},
                                provenance=NodeProvenance(
                                    chunk_ids=combined_chunks,
                                    extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                                    confidence_score=combined_confidence
                                )
                            )
                            
                            inferred_relationships.append(new_rel)
        
        return inferred_relationships

    def _get_entity_text(self, entity):
        """
        Get all textual content from an entity for analysis.
        Combines name, description, and other text fields.
        """
        if not hasattr(entity, 'properties') or not entity.properties:
            return ""
            
        text_parts = []
        
        # Try different property names that might contain useful text
        text_field_patterns = [
            # Name-like fields
            'name', 'title', 'label', 'id', 'identifier',
            # Description-like fields
            'description', 'details', 'info', 'text', 'summary', 'overview',
            'content', 'notes', 'remarks', 'comment', 'about',
            # Type-specific fields
            'category', 'type', 'classification', 'group', 'class',
            'field', 'industry', 'sector', 'domain', 'area'
        ]
        
        for field in entity.properties:
            # Check if this property might contain text
            if any(pattern in field.lower() for pattern in text_field_patterns):
                value = entity.properties[field]
                if value and isinstance(value, str):
                    text_parts.append(value)
                    
        # If no matching fields found, use all string properties
        if not text_parts:
            for field, value in entity.properties.items():
                if value and isinstance(value, str):
                    text_parts.append(value)
                    
        return " ".join(text_parts)

    def _calculate_text_overlap(self, text1, text2):
        """
        Calculate text similarity based on word overlap.
        Returns a score between 0 and 1.
        """
        if not text1 or not text2:
            return 0.0
            
        # Normalize texts
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Tokenize into words (simple approach)
        words1 = set(w for w in text1.split() if len(w) > 3)  # Skip short words
        words2 = set(w for w in text2.split() if len(w) > 3)
        
        # Skip if either set is empty
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _calculate_word_overlap(self, text1, text2):
        """
        Calculate word overlap between two texts.
        Used for fuzzy entity matching.
        """
        if not text1 or not text2:
            return 0.0
            
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        # For short texts, require higher overlap
        min_word_count = min(len(words1), len(words2))
        if min_word_count <= 2:
            # For very short texts (1-2 words), require exact match
            return 1.0 if words1 == words2 else 0.0
            
        intersection = len(words1.intersection(words2))
        # Use smaller set as denominator for better matching of substrings
        return intersection / min_word_count

    def _check_property_matches(self, entity1, entity2):
        """
        Check how many properties match between two entities.
        Returns a score between 0 and 1 representing match quality.
        """
        if (not hasattr(entity1, 'properties') or not entity1.properties or
            not hasattr(entity2, 'properties') or not entity2.properties):
            return 0.0
            
        # Count matching properties
        matches = 0
        total_comparable = 0
        
        for prop1, value1 in entity1.properties.items():
            if value1 is None:
                continue
                
            # Only compare string and numeric properties
            if not isinstance(value1, (str, int, float)):
                continue
                
            total_comparable += 1
            
            # Check if property exists in entity2
            if prop1 in entity2.properties:
                value2 = entity2.properties[prop1]
                
                # Skip None values
                if value2 is None:
                    continue
                    
                # For strings, check normalized equality
                if isinstance(value1, str) and isinstance(value2, str):
                    if value1.lower().strip() == value2.lower().strip():
                        matches += 1
                    # Partial match for longer strings
                    elif len(value1) > 10 and len(value2) > 10:
                        # Check if one is substring of other
                        if value1.lower() in value2.lower() or value2.lower() in value1.lower():
                            matches += 0.5
                # For numbers, check equality
                elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    if value1 == value2:
                        matches += 1
        
        # Calculate matching score
        return matches / total_comparable if total_comparable > 0 else 0.0