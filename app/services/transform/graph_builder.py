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
    DocumentKnowledgeGraph,
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
        
    async def build_graph_from_chunks(
        self,
        chunks: List[str],
        transform_id: str,
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DocumentKnowledgeGraph:
        """Process all chunks and build unified graph"""
        chunk_results = []
        
        # Create event loop if not already running
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Process chunks with controlled concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(chunk: str, idx: int):
            async with semaphore:
                try:
                    print(f"Processing chunk {idx + 1} / {len(chunks)}")
                    result = await self.process_chunk(chunk, f"{transform_id}_chunk_{idx}")
                    if progress_callback:
                        try:
                            if asyncio.iscoroutinefunction(progress_callback):
                                await progress_callback(idx + 1, len(chunks))
                            else:
                                # Handle synchronous callback
                                progress_callback(idx + 1, len(chunks))
                        except Exception as e:
                            logger.warning(f"Progress callback failed: {str(e)}")
                    return result
                except Exception as e:
                    logger.error(f"Chunk processing failed for idx {idx}: {str(e)}")
                    return None
        
        # Create processing tasks
        tasks = []
        for idx, chunk in enumerate(chunks):
            task = asyncio.create_task(process_with_semaphore(chunk, idx))
            tasks.append(task)
        
        # Process all chunks concurrently with controlled parallelism
        chunk_results = await asyncio.gather(*tasks, return_exceptions=False)
        chunk_results = [r for r in chunk_results if r is not None]
        
        # Merge all extraction results
        for result in chunk_results:
            self.add_extraction_result(result)
            
        # Finalize the graph
        return await self.finalize_graph()
    
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
                    if hasattr(item, 'model_dump'):  # Pydantic v2
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
    
    async def finalize_graph(self) -> DocumentKnowledgeGraph:
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
        nodes_in_relationships = []
        
        # Check all relationship fields
        for field_name in dir(self.graph):
            if field_name.endswith('_list') or not '_' in field_name or field_name.startswith('_'):
                continue
                
            rel_list = getattr(self.graph, field_name)
            if not isinstance(rel_list, list):
                continue
                
            for rel in rel_list:
                nodes_in_relationships.append((rel.source_type, rel.source_id))
                nodes_in_relationships.append((rel.target_type, rel.target_id))
        
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
        entity_def = self.ontology_parser.parsed_ontology.get('entities', {}).get(entity_type)
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
            if '_' in field_name and not field_name.endswith('_list') and not field_name.startswith('_'):
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
        
    async def post_process_graph(self, graph: KnowledgeGraph) -> DocumentKnowledgeGraph:
        """
        Apply post-processing to improve graph quality and convert to DocumentKnowledgeGraph format.
        This helps ensure consistency in naming, capitalization, and property values.
        Includes entity resolution using LLM.
        """
        # Create DocumentKnowledgeGraph
        doc_graph = DocumentKnowledgeGraph(
            extraction_timestamp=graph.extraction_timestamp,
            tokens_used=graph.tokens_used,
            confidence_score=graph.confidence_score,
            metrics=self.metrics
        )
        
        # Collect all nodes from entity lists
        for field_name in dir(graph):
            if field_name.endswith('_list') and not field_name.startswith('_'):
                entity_list = getattr(graph, field_name)
                if isinstance(entity_list, list):
                    doc_graph.nodes.extend(entity_list)
        
        # Collect all relationships
        for field_name in dir(graph):
            if '_' in field_name and not field_name.endswith('_list') and not field_name.startswith('_'):
                rel_list = getattr(graph, field_name)
                if isinstance(rel_list, list):
                    doc_graph.relationships.extend(rel_list)

        # Group nodes by entity type for resolution
        entity_groups = {}
        for node in doc_graph.nodes:
            entity_type = node.type
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(node)

        # Process each entity type group for resolution
        merged_nodes = {}  # old_id -> new_id mapping
        final_nodes = []
        
        for entity_type, nodes in entity_groups.items():
            if len(nodes) <= 1:
                final_nodes.extend(nodes)
                continue
                
            # Perform entity resolution for this group
            resolved_groups = await self._resolve_entity_group(entity_type, nodes)
            
            # Process resolved groups
            for group in resolved_groups:
                if len(group) == 1:
                    # Single node, no merging needed
                    final_nodes.append(group[0])
                    continue
                    
                # Sort by confidence score to use highest confidence node as base
                sorted_nodes = sorted(group, 
                                    key=lambda x: x.provenance.confidence_score if hasattr(x, 'provenance') and x.provenance else 0, 
                                    reverse=True)
                
                # Use highest confidence node as base and merge others into it
                base_node = sorted_nodes[0]
                for other_node in sorted_nodes[1:]:
                    merged_nodes[other_node.id] = base_node.id
                    # Merge properties and provenance
                    base_node = self._merge_nodes(base_node, other_node)
                
                final_nodes.append(base_node)

        # Update relationships with merged node IDs
        final_relationships = []
        for rel in doc_graph.relationships:
            # Check if source or target nodes were merged
            source_id = merged_nodes.get(rel.source_id, rel.source_id)
            target_id = merged_nodes.get(rel.target_id, rel.target_id)
            
            # Skip if either node was dropped
            if source_id and target_id:
                rel.source_id = source_id
                rel.target_id = target_id
                final_relationships.append(rel)

        # Update graph with resolved entities
        doc_graph.nodes = final_nodes
        doc_graph.relationships = final_relationships
            
        # Continue with existing post-processing steps
        # Standardize entity values
        entity_groups = {}
        for node in doc_graph.nodes:
            entity_type = node.__class__.__name__
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(node.model_dump())
            
        # Process each entity group
        for entity_type, entities in entity_groups.items():
            await self._standardize_entity_group(entity_type, entities)
            
        # Enhance property consistency
        self._enhance_property_consistency(doc_graph)
        
        # Validate relationship consistency
        self._validate_relationship_consistency(doc_graph)
        
        # Enhance relationship confidence
        self._enhance_relationship_confidence(doc_graph)
        
        self.metrics.total_nodes = len(doc_graph.nodes)
        self.metrics.total_relationships = len(doc_graph.relationships)
        
        return doc_graph

    async def _resolve_entity_group(self, entity_type: str, nodes: List[BaseNode]) -> List[List[BaseNode]]:
        """
        Use LLM to identify and group matching entities that should be merged.
        Returns list of groups, where each group contains matching nodes.
        """
        if len(nodes) <= 1:
            return [[nodes[0]]] if nodes else []

        # Convert nodes to simple dict representation for LLM
        node_dicts = []
        for idx, node in enumerate(nodes):
            node_dict = {
                "id": node.id,
                "index": idx,  # Keep track of original position
                "properties": node.properties,
                "confidence": node.provenance.confidence_score if hasattr(node, 'provenance') and node.provenance else None
            }
            node_dicts.append(node_dict)
            
        node_dicts_str = json.dumps(node_dicts, indent=2)

        # Create prompt for LLM entity resolution
        prompt = f"""
        You are performing entity resolution on a group of {entity_type} entities.
        Your task is to identify which entities refer to the same real-world entity and should be merged.
        
        Entities:
        {node_dicts_str}
        
        Use these guidelines for matching:
        1. Compare all properties to determine if entities match
        2. Handle variations in naming, formatting, and completeness
        3. Consider similarity in key identifying properties
        4. Be conservative - only match if reasonably confident (>80% sure)
        5. Consider property values that are similar but not exactly matching
        6. Look for complementary information across entities
        
        Return a JSON array of arrays, where each inner array contains indices of matching entities:
        {{
            "matching_groups": [
                [0, 2, 5],  # Example: entities 0, 2, and 5 match
                [1, 4],     # Example: entities 1 and 4 match
                [3],        # Example: entity 3 has no matches
                [6]         # Example: entity 6 has no matches
            ],
            "confidence_scores": [
                0.95,  # Confidence for first group
                0.85,  # Confidence for second group
                1.0,   # Single entity groups always have 1.0 confidence
                1.0
            ]
        }}

        Also provide brief explanations for why you grouped entities together:
        {{
            "explanations": {{
                "group_0": "These entities share the same name with minor variations and have overlapping properties",
                "group_1": "These entities have matching identifiers and complementary information"
            }}
        }}
        """

        try:
            # Call LLM for entity resolution
            print(f"Resolving `{entity_type}` entities")
            result = await self.llm_client.generate_text(prompt=prompt, json_response=True)
            
            if not result or 'matching_groups' not in result:
                # Fallback: treat each node as separate group
                return [[node] for node in nodes]
            
            # Convert index groups back to node groups
            resolved_groups = []
            for idx, index_group in enumerate(result['matching_groups']):
                node_group = []
                for idx in index_group:
                    if 0 <= idx < len(nodes):  # Validate index
                        node_group.append(nodes[idx])
                if node_group:  # Only add non-empty groups
                    resolved_groups.append(node_group)
                    
                    # Log explanation if available
                    if 'explanations' in result and f'group_{idx}' in result['explanations']:
                        logger.info(f"Entity resolution group {idx}: {result['explanations'][f'group_{idx}']}")
            
            # Check if any nodes were missed (safeguard)
            included_nodes = [node for group in resolved_groups for node in group]
            missed_nodes = [node for node in nodes if node not in included_nodes]
            
            # Add missed nodes as single-node groups
            for node in missed_nodes:
                resolved_groups.append([node])
            
            return resolved_groups
            
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Error in LLM entity resolution: {str(e)}")
            # Fallback: treat each node as separate group
            return [[node] for node in nodes]

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

    def _enhance_property_consistency(self, graph: DocumentKnowledgeGraph) -> None:
        """
        Enhance property naming consistency across entity types.
        This uses the ontology definitions to ensure property names match expectations.
        """
        ontology = self.ontology_parser.parsed_ontology
        
        # Process all nodes
        for node in graph.nodes:
            entity_type = node.__class__.__name__
            entity_def = ontology.get('entities', {}).get(entity_type, {})
            property_defs = entity_def.get('properties', {})
            
            if not hasattr(node, 'properties'):
                continue
                
            # Normalize property names based on ontology
            normalized_props = {}
            for prop_name, value in node.properties.items():
                # Find matching ontology property
                matched_prop = None
                for onto_prop, prop_def in property_defs.items():
                    if onto_prop.lower() == prop_name.lower():
                        matched_prop = onto_prop
                        break
                
                # Use ontology name if found, otherwise keep original
                final_name = matched_prop if matched_prop else prop_name
                normalized_props[final_name] = value
                
            node.properties = normalized_props
    
    def _validate_relationship_consistency(self, graph: DocumentKnowledgeGraph) -> None:
        """
        Ensure relationships are consistent with ontology definitions.
        """
        ontology = self.ontology_parser.parsed_ontology
        valid_edges = []
        
        for edge in graph.relationships:
            if not hasattr(edge, 'source_type') or not hasattr(edge, 'type'):
                continue
                
            # Check if relationship type is valid in ontology
            source_type = edge.source_type
            rel_type = edge.type
            
            entity_def = ontology.get('entities', {}).get(source_type, {})
            rel_def = entity_def.get('relationships', {}).get(rel_type)
            
            if rel_def and rel_def.get('target') == edge.target_type:
                valid_edges.append(edge)
        
        # Update edges with only valid relationships
        graph.relationships = valid_edges
    
    def _enhance_relationship_confidence(self, graph: DocumentKnowledgeGraph) -> None:
        """
        Enhance relationship confidence scores based on connected entities.
        """
        for edge in graph.relationships:
            if not hasattr(edge, 'provenance') or not edge.provenance:
                continue
                
            # Skip if already has good confidence
            if edge.provenance.confidence_score and edge.provenance.confidence_score > 0.5:
                continue
                
            # Find source and target entities
            source_entity = None
            target_entity = None
            
            for node in graph.nodes:
                if node.id == edge.source_id:
                    source_entity = node
                elif node.id == edge.target_id:
                    target_entity = node
                    
                if source_entity and target_entity:
                    break
            
            if not source_entity or not target_entity:
                continue
                
            # Calculate confidence based on entity confidence
            source_conf = source_entity.provenance.confidence_score if hasattr(source_entity, 'provenance') else 0.0
            target_conf = target_entity.provenance.confidence_score if hasattr(target_entity, 'provenance') else 0.0
            
            # Average of entity confidences
            edge.provenance.confidence_score = (source_conf + target_conf) / 2

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