from typing import Dict, List, Any, Optional, Callable, Tuple
import copy
import asyncio
import uuid
import json
import traceback
from datetime import datetime, timezone
from app.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    KnowledgeGraph,
    DocumentKnowledgeGraph,
    ExtractionMetrics
)
from app.services.transform.ontology_helper import (
    OntologyParser,
    OntologyHierarchyBuilder
)
from app.services.llm.client import LLMClient
from app.utils.logger import logger

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
        self.extracted_triples = set()
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
        window_size: int = 3,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DocumentKnowledgeGraph:
        """Process all chunks and build unified graph with sliding window context"""
        chunk_results = []
        
        # Process chunks sequentially with sliding window context
        for idx, chunk in enumerate(chunks):
            # Get window of surrounding chunks for context
            window_start = max(0, idx - window_size)
            window_end = min(len(chunks), idx + window_size + 1)
            context_chunks = chunks[window_start:idx] + chunks[idx+1:window_end]
            
            try:
                print(f"Processing chunk {idx + 1} / {len(chunks)}")
                result = await self.process_chunk(
                    chunk=chunk,
                    chunk_id=f"{transform_id}_chunk_{idx}",
                    context_chunks=context_chunks
                )
                
                if result:
                    chunk_results.append(result)
                    
                if progress_callback:
                    try:
                        if asyncio.iscoroutinefunction(progress_callback):
                            await progress_callback(idx + 1, len(chunks))
                        else:
                            # Handle synchronous callback
                            progress_callback(idx + 1, len(chunks))
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Chunk processing failed for idx {idx}: {str(e)}")
                continue
        
        # Merge all extraction results
        for result in chunk_results:
            self.add_extraction_result(result)
            
        # Finalize the graph
        return await self.finalize_graph()
    
    async def process_chunk(
        self,
        chunk: str,
        chunk_id: str,
        context_chunks: List[str] = []
    ) -> Optional[Dict[str, Any]]:
        """Process single chunk with LLM extraction using RDF context"""
        start_time = datetime.now()
        
        try:
            # Generate RDF context from previously extracted entities and relationships
            context = self._generate_rdf_context()
            
            # Add context from surrounding chunks
            if context_chunks:
                context += "\n# Context from surrounding chunks:\n"
                for ctx_chunk in context_chunks:
                    context += ctx_chunk + "\n"
            
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
                    # node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, node_key))
                    node_id = str(uuid.uuid4())
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
                    if not hasattr(rel_item, 'source_id') or not hasattr(rel_item, 'target_id'):
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
                merged_node = self._merge_nodes(matching_node, node)
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
        
        if hasattr(self.graph, 'nodes'):
            self.graph.nodes.append(node)
        else:
            entity_list_field = f"{entity_type}_list"
            if not hasattr(self.graph, entity_list_field):
                setattr(self.graph, entity_list_field, [])
            
            entity_list = getattr(self.graph, entity_list_field)
            entity_list.append(node)
    
    def _update_node_in_graph(self, node: BaseNode) -> None:
        """Update a node in the graph"""
        entity_type = node.type
        
        if hasattr(self.graph, 'nodes'):
            for i, existing_node in enumerate(self.graph.nodes):
                if existing_node.id == node.id:
                    self.graph.nodes[i] = node
                    return
            self.graph.nodes.append(node)
        else:
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
        
        if hasattr(self.graph, 'relationships'):
            self.graph.relationships.append(relationship)
        else:
            rel_field = f"{source_type}_{rel_type}"
            if not hasattr(self.graph, rel_field):
                setattr(self.graph, rel_field, [])
            
            rel_list = getattr(self.graph, rel_field)
            rel_list.append(relationship)
    
    def _update_relationship_in_graph(self, relationship: RelationshipInstance) -> None:
        """Update a relationship in the graph"""
        source_type = relationship.source_type
        rel_type = relationship.type
        
        if hasattr(self.graph, 'relationships'):
            for i, existing_rel in enumerate(self.graph.relationships):
                if existing_rel.id == relationship.id:
                    self.graph.relationships[i] = relationship
                    return
            self.graph.relationships.append(relationship)
        else:
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
        
        # First validate relationships to ensure we have valid connections
        self._validate_relationships()
        
        # Set final metrics
        self.graph.tokens_used = self.metrics.total_tokens
        self.graph.confidence_score = self._calculate_average_confidence()
        self.graph.extraction_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Post process the graph (includes entity resolution and relationship enhancement)
        processed_graph = await self.post_process_graph(self.graph)
        
        # Update self.graph to match processed_graph structure
        self.graph = processed_graph
        
        # Finally prune any orphaned nodes after all processing is complete
        self._prune_orphaned_nodes()
        
        return self.graph
    
    def _prune_orphaned_nodes(self) -> None:
        """Remove nodes with no ontology-defined properties that aren't referenced in any relationship"""
        # Build set of node IDs used in relationships (both source and target)
        nodes_in_relationships = set()
        
        # Check relationships list for DocumentKnowledgeGraph
        if hasattr(self.graph, 'relationships'):
            for rel in self.graph.relationships:
                nodes_in_relationships.add(rel.source_id)
                nodes_in_relationships.add(rel.target_id)
        else:
            # Check all relationship fields for KnowledgeGraph
            for field_name in dir(self.graph):
                if field_name.endswith('_list') or not '_' in field_name or field_name.startswith('_'):
                    continue
                rel_list = getattr(self.graph, field_name)
                if isinstance(rel_list, list):
                    for rel in rel_list:
                        nodes_in_relationships.add(rel.source_id)
                        nodes_in_relationships.add(rel.target_id)

        # Get all nodes
        all_nodes = self.graph.nodes if hasattr(self.graph, 'nodes') else []
        if not all_nodes:
            for field_name in dir(self.graph):
                if not field_name.endswith('_list') or field_name.startswith('_'):
                    continue
                node_list = getattr(self.graph, field_name)
                if isinstance(node_list, list):
                    all_nodes.extend(node_list)

        # Find orphaned nodes and try to link them
        orphaned_nodes = []
        for node in all_nodes:
            if node.id not in nodes_in_relationships:
                # Check if this node type has only one possible relationship type in ontology
                node_type = node.type
                possible_rels = self._get_possible_relationships_for_type(node_type)
                
                if len(possible_rels) == 1:
                    # Get the single relationship type and direction
                    rel_type = list(possible_rels.keys())[0]
                    rel_info = possible_rels[rel_type]
                    
                    # Find potential nodes to link with
                    potential_nodes = []
                    for other_node in all_nodes:
                        if other_node.id != node.id:
                            # Check both source->target and target->source possibilities
                            if (rel_info['source'] == node_type and rel_info['target'] == other_node.type) or \
                               (rel_info['target'] == node_type and rel_info['source'] == other_node.type):
                                potential_nodes.append(other_node)
                    
                    # If we found exactly one potential node, create the relationship
                    if len(potential_nodes) == 1:
                        other_node = potential_nodes[0]
                        # Determine direction based on ontology
                        if rel_info['source'] == node_type:
                            source_id, target_id = node.id, other_node.id
                            source_type, target_type = node_type, other_node.type
                        else:
                            source_id, target_id = other_node.id, node.id
                            source_type, target_type = other_node.type, node_type
                            
                        # Create relationship
                        rel_id = f"{source_id}_{rel_type}_{target_id}"
                        new_rel = RelationshipInstance(
                            id=rel_id,
                            type=rel_type,
                            source_id=source_id,
                            target_id=target_id,
                            source_type=source_type,
                            target_type=target_type,
                            properties={},
                            provenance=NodeProvenance(
                                chunk_ids=[],
                                extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                                confidence_score=0.9
                            )
                        )
                        
                        # Add to relationships
                        if hasattr(self.graph, 'relationships'):
                            self.graph.relationships.append(new_rel)
                        else:
                            rel_field = f"{source_type.lower()}_{rel_type.lower()}_{target_type.lower()}"
                            if not hasattr(self.graph, rel_field):
                                setattr(self.graph, rel_field, [])
                            getattr(self.graph, rel_field).append(new_rel)
                            
                        # Node is no longer orphaned since it's now in a relationship
                        continue
                
                # If we couldn't auto-link, add to orphaned list
                orphaned_nodes.append(node)

        # Remove orphaned nodes that couldn't be linked
        if hasattr(self.graph, 'nodes'):
            self.graph.nodes = [n for n in self.graph.nodes if n not in orphaned_nodes]
        else:
            for field_name in dir(self.graph):
                if not field_name.endswith('_list') or field_name.startswith('_'):
                    continue
                node_list = getattr(self.graph, field_name)
                if isinstance(node_list, list):
                    setattr(self.graph, field_name, [n for n in node_list if n not in orphaned_nodes])
    
    def _get_possible_relationships_for_type(self, node_type: str) -> Dict[str, Dict[str, str]]:
        """Get all possible relationship types for a given node type from ontology"""
        possible_rels = {}
        
        # Use the ontology passed to the builder
        ontology = self.ontology_parser.parsed_ontology
            
        # Get relationships where this type is either source or target
        entity_def = ontology.get('entities', {}).get(node_type, {})
        if not entity_def:
            return {}
        
        relationships = entity_def.get('relationships', {})
        for rel_type, rel_info in relationships.items():
            source_type = node_type
            target_type = rel_info.get('target')
            
            if source_type == node_type or target_type == node_type:
                possible_rels[rel_type] = {
                    'source': source_type,
                    'target': target_type
                }
        
        return possible_rels

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
        Generate RDF-style context from previously extracted entities and relationships.
        Limited to most recent/relevant triples to avoid context overload.
        Includes entity summaries and relationship patterns.
        """
        if not hasattr(self, 'extracted_triples'):
            self.extracted_triples = set()
            
        if not self.extracted_triples:
            return ""
        
        # Build context sections
        context_parts = []
        
        # 1. Entity summaries
        entity_summaries = {}
        for triple in self.extracted_triples:
            if triple.startswith("<"):  # Entity triple
                entity_type = triple.split()[0].strip("<>")
                if entity_type not in entity_summaries:
                    entity_summaries[entity_type] = []
                entity_summaries[entity_type].append(triple)
        
        if entity_summaries:
            context_parts.append("# Entity summaries by type:")
            for entity_type, triples in entity_summaries.items():
                context_parts.append(f"\n## {entity_type}:")
                # Take most recent entities of each type
                recent_triples = triples[-5:]  # Keep last 5 entities of each type
                context_parts.extend(recent_triples)
        
        # 2. Relationship patterns
        relationship_patterns = {}
        for triple in self.extracted_triples:
            if " -> " in triple:  # Relationship triple
                rel_type = triple.split("->")[1].strip().split()[0]
                if rel_type not in relationship_patterns:
                    relationship_patterns[rel_type] = []
                relationship_patterns[rel_type].append(triple)
        
        if relationship_patterns:
            context_parts.append("\n# Recent relationship patterns:")
            for rel_type, triples in relationship_patterns.items():
                context_parts.append(f"\n## {rel_type}:")
                # Take most recent relationships of each type
                recent_triples = triples[-3:]  # Keep last 3 relationships of each type
                context_parts.extend(recent_triples)
        
        # 3. Most recent extractions
        context_parts.append("\n# Most recent extractions:")
        recent_triples = list(self.extracted_triples)[-max_triples:]
        context_parts.extend(recent_triples)
        
        return "\n".join(context_parts)

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
            self.extracted_triples = set()
            
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
                    self.extracted_triples.add(triple)
        
        # Process relationships
        for rel in result.get('relationships', []):
            source_type = rel.source_type
            source_id = rel.source_id
            target_type = rel.target_type
            target_id = rel.target_id
            rel_type = rel.type
            
            # Format as: <source_type>(<source_id>) <relationship> <target_type>(<target_id>)
            triple = f"{source_type}({source_id}) {rel_type} {target_type}({target_id})"
            self.extracted_triples.add(triple)
            
            # Add relationship properties if any
            if hasattr(rel, 'properties') and rel.properties:
                for prop_name, prop_value in rel.properties.items():
                    if prop_value is not None:
                        # Format as: Relationship(<rel_id>) <property> <value>
                        safe_value = self._create_safe_property_value(prop_value)
                        rel_triple = f"Relationship({rel.id}) hasProperty:{prop_name} {safe_value}"
                        self.extracted_triples.add(rel_triple)
                        
    def _update_final_triples(self):
        """Update RDF triples based on the final state of the graph"""
        # Clear existing triples to rebuild from current graph state
        self.extracted_triples = set()
        
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
                        self.extracted_triples.add(triple)
        
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
                    self.extracted_triples.add(triple)
                    
                    # Add relationship properties if any
                    if hasattr(rel, 'properties') and rel.properties:
                        for prop_name, prop_value in rel.properties.items():
                            if prop_value is not None:
                                # Format as: Relationship(<rel_id>) <property> <value>
                                value_str = str(prop_value).replace('"', r'\"')
                                rel_triple = f"Relationship({rel.id}) hasProperty:{prop_name} \"{value_str}\""
                                self.extracted_triples.add(rel_triple)
        
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
            if node.id not in [n.id for n in entity_groups[entity_type]]:
                entity_groups[entity_type].append(node)
            else:
                #merge nodes with same id
                base_node = [n for n in entity_groups[entity_type] if n.id == node.id][0]
                entity_groups[entity_type].remove(base_node)
                base_node = self._merge_nodes(base_node, node)
                entity_groups[entity_type].append(base_node)
        
        # Enhance property consistency
        doc_graph = self._enhance_property_consistency(doc_graph)
            
        # Process each entity group
        standardised_entity_groups = await asyncio.gather(
            *[
                self._standardize_entity_group(entity_type, entities)
                for entity_type, entities in entity_groups.items()
            ]
        )
        for (entity_type, standardized_entities) in standardised_entity_groups:
            entity_groups[entity_type] = standardized_entities

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
                                    key=lambda x: x.confidence_score if x.confidence_score else 0, 
                                    reverse=True)
                
                # Use highest confidence node as base and merge others into it
                base_node = sorted_nodes[0]
                for other_node in sorted_nodes[1:]:
                    merged_nodes[other_node.id] = base_node.id
                    # Merge properties and provenance
                    base_node = self._merge_nodes(base_node, other_node)
                
                final_nodes.append(base_node)

        # Update relationships with merged node IDs and remove invalid ones
        final_relationships = []
        for rel in doc_graph.relationships:
            # Get final node IDs after merging
            source_id = merged_nodes.get(rel.source_id, rel.source_id)
            target_id = merged_nodes.get(rel.target_id, rel.target_id)
            
            # Check if both nodes still exist in final_nodes
            source_exists = any(n.id == source_id for n in final_nodes)
            target_exists = any(n.id == target_id for n in final_nodes)
            
            # Only keep relationships where both nodes exist
            if source_exists and target_exists:
                rel.source_id = source_id
                rel.target_id = target_id
                final_relationships.append(rel)

        # Update graph with resolved entities
        doc_graph.nodes = final_nodes
        doc_graph.relationships = final_relationships

        
        # Validate relationship consistency
        doc_graph = self._validate_relationship_consistency(doc_graph)
        
        # Enhance relationship confidence
        self._enhance_relationship_confidence(doc_graph)
        
        # Infer missing relationships
        await self._infer_missing_relationships(doc_graph)
        
        # Validate final graph
        self._validate_graph(doc_graph)
        
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
        for node in nodes:
            node_dict = {
                "id": node.id,
                "properties": node.properties,
                "confidence": node.confidence_score if node.confidence_score else 0.3
            }
            node_dicts.append(node_dict)
            
        node_dicts_str = json.dumps(node_dicts, indent=2)

        try:
            # Call LLM for entity resolution
            print(f"Resolving `{entity_type}` entities")
            results = await self.llm_client.resolve_entities(
                entity_type=entity_type, node_dicts_str=node_dicts_str)
            
            if not results or len(results) == 0:
                # Fallback: treat each node as separate group
                return [[node] for node in nodes]
            
            # Convert index groups back to node groups
            resolved_groups = []
            node_map = {node.id: node for node in nodes}
            for result in results:
                if result.matching_ids is None or len(result.matching_ids) == 0:
                    continue
                node_group = []
                for matching_node_id in result.matching_ids:
                    if matching_node_id in node_map:
                        _node = node_map[matching_node_id]
                        _node.confidence_score = result.confidence_score
                        node_group.append(_node)
                if node_group:  # Only add non-empty groups
                    resolved_groups.append(node_group)
                    
                    # Log explanation if available
                    if result.explanation:
                        logger.info(f"Entity resolution group {node_group[0].type}: {result.explanation}")
            
            # Check if any nodes were missed (safeguard)
            included_nodes = [node for group in resolved_groups for node in group]
            
            # Add missed nodes as single-node groups
            for node in nodes:
                if node not in included_nodes:
                    resolved_groups.append([node])
            
            return resolved_groups
            
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Error in LLM entity resolution: {str(e)}")
            # Fallback: treat each node as separate group
            return [[node] for node in nodes]

    async def _standardize_entity_group(self, entity_type: str, entity_data: List[BaseNode]) -> Tuple[str, List[BaseNode]]:
        """
        Use LLM to standardize property values across similar entities
        """
        if not entity_data:
            return (entity_type, [])
            
        # Sort by confidence score to prioritize high-confidence entities
        sorted_entities = sorted(entity_data, 
                                 key=lambda e: e.confidence_score if e.confidence_score is not None else 0.0, 
                                 reverse=True)
        
        # Call LLM for standardization
        try:
            results = await self.llm_client.standardise_properties(
                entity_group_type=entity_type, 
                entities_json=json.dumps([entity.model_dump() for entity in sorted_entities], indent=2))
            if results:
                entity_map = { entity.id: entity for entity in entity_data }
                for result in results:
                    std_props = result.properties
                    entity_map[result.entity_id].properties.update(std_props)
                return (entity_type, list(entity_map.values()))
                    
            return (entity_type, entity_data)
            
        except Exception as e:
            logger.warning(f"Error in LLM standardization: {str(e)}")
            traceback.print_exc()
            return (entity_type, entity_data)
    
    def _enhance_property_consistency(self, graph: DocumentKnowledgeGraph) -> DocumentKnowledgeGraph:
        """
        Enhance property naming consistency across entity types.
        This uses the ontology definitions to ensure property names match expectations.
        """
        ontology = self.ontology_parser.parsed_ontology
        
        # Process all nodes
        for node in graph.nodes:
            entity_type = node.type
            entity_def = ontology.get('entities', {}).get(entity_type, {})
            property_defs = entity_def.get('properties', {})
            if not property_defs:
                continue
                
            # Normalize property names based on ontology
            normalized_props = {}
            for prop_name, prop_value in node.properties.items():
                # Find matching ontology property
                matched_prop = None
                for onto_prop, prop_def in property_defs.items():
                    if onto_prop.lower() == prop_name.lower():
                        matched_prop = onto_prop
                        break
                
                # Use ontology name if found, otherwise keep original
                final_name = matched_prop if matched_prop else prop_name
                normalized_props[final_name] = prop_value
                
            node.properties = normalized_props
        
        return graph
    
    def _validate_relationship_consistency(self, graph: DocumentKnowledgeGraph) -> DocumentKnowledgeGraph:
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

        return graph
    
    def _enhance_relationship_confidence(self, graph: DocumentKnowledgeGraph) -> DocumentKnowledgeGraph:
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

        return graph

    async def _infer_missing_relationships(self, graph: DocumentKnowledgeGraph) -> None:
        """
        Infer potential missing relationships for dangling subgraphs.
        Only attempts to connect subgraphs that should be connected according to ontology.
        """
        # Find dangling subgraphs
        dangling_nodes = self._find_dangling_subgraphs(graph)
        if not dangling_nodes:
            return
            
        print(f"Found dangling nodes of types: {list(dangling_nodes.keys())}")
        
        hierarchy_builder = OntologyHierarchyBuilder(self.ontology_parser)
        entity_types = set(dangling_nodes.keys())
        
        # For each pair of entity types with dangling nodes
        for source_type in entity_types:
            for target_type in entity_types:
                if source_type == target_type:
                    continue
                    
                # Get valid relationship types between these entity types
                rel_types = hierarchy_builder.get_relationship_types(source_type, target_type)
                if not rel_types:
                    continue
                    
                # For each valid relationship type
                for rel_type in rel_types:
                    # Get existing relationships of this type
                    existing_rels = [
                        edge for edge in graph.relationships 
                        if edge.type == rel_type and 
                        edge.source_type == source_type and 
                        edge.target_type == target_type
                    ]
                    existing_pairs = {(edge.source_id, edge.target_id) for edge in existing_rels}
                    
                    # Look for potential new relationships between dangling nodes
                    source_entities = dangling_nodes[source_type]
                    target_entities = dangling_nodes[target_type]
                    
                    print(f"Attempting to infer {rel_type} relationships between {len(source_entities)} {source_type} nodes and {len(target_entities)} {target_type} nodes")
                    
                    # Call LLM to infer relationships
                    try:
                        print(f"Inferring {rel_type} relationships between {source_type} and {target_type}")
                        inferred = await self.llm_client.infer_relationship(
                            rel_type=rel_type,
                            source_type=source_type,
                            source_entities=self._format_entities(source_entities),
                            target_type=target_type,
                            target_entities=self._format_entities(target_entities),
                            existing_rels=self._format_relationships(existing_rels)
                        )
                        
                        if inferred:
                            print(f"Inferred {len(inferred)} new relationships")
                            for rel in inferred:
                                if (rel.source_id, rel.target_id) not in existing_pairs:
                                    # Find source and target nodes
                                    source_node = next((n for n in source_entities if n.id == rel.source_id), None)
                                    target_node = next((n for n in target_entities if n.id == rel.target_id), None)
                                    
                                    if source_node and target_node:
                                        # Create new edge with provenance
                                        provenance = NodeProvenance(
                                            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                                            confidence_score=rel.confidence_score if hasattr(rel, 'confidence_score') else 0.5
                                        )
                                        
                                        edge = RelationshipInstance(
                                            id=rel.id or f"{rel.source_id}_{rel.type}_{rel.target_id}",
                                            source_id=source_node.id,
                                            target_id=target_node.id,
                                            source_type=source_node.type,
                                            target_type=target_node.type,
                                            type=rel.type,
                                            properties=rel.properties or {},
                                            provenance=provenance
                                        )
                                        graph.relationships.append(edge)
                                        existing_pairs.add((edge.source_id, edge.target_id))
                            
                    except Exception as e:
                        traceback.print_exc()
                        logger.warning(f"Failed to infer relationships for {source_type}->{target_type}: {str(e)}")

    def _find_dangling_subgraphs(self, graph: DocumentKnowledgeGraph) -> Dict[str, List[BaseNode]]:
        """
        Find nodes that are part of disconnected subgraphs.
        Returns a dict mapping entity type to list of nodes that are disconnected
        from the main graph but should be connected according to ontology.
        """
        # Build adjacency map
        adj_map = {}
        for rel in graph.relationships:
            if rel.source_id not in adj_map:
                adj_map[rel.source_id] = set()
            if rel.target_id not in adj_map:
                adj_map[rel.target_id] = set()
            adj_map[rel.source_id].add(rel.target_id)
            adj_map[rel.target_id].add(rel.source_id)
            
        # Find connected components using DFS
        visited = set()
        components = []
        
        def dfs(node_id: str, component: set):
            visited.add(node_id)
            component.add(node_id)
            for neighbor in adj_map.get(node_id, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        # Run DFS from each unvisited node
        for node in graph.nodes:
            if node.id not in visited:
                component = set()
                dfs(node.id, component)
                components.append(component)
                
        # If we have only one component, no dangling subgraphs
        if len(components) <= 1:
            return {}
            
        # Group nodes by type within each component
        component_nodes = {}
        for i, component in enumerate(components):
            component_nodes[i] = {}
            for node_id in component:
                node = next(n for n in graph.nodes if n.id == node_id)
                if node.type not in component_nodes[i]:
                    component_nodes[i][node.type] = []
                component_nodes[i][node.type].append(node)
                
        # Find entity types that should be connected according to ontology
        hierarchy_builder = OntologyHierarchyBuilder(self.ontology_parser)
        dangling_nodes = {}
        
        # For each pair of components
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                # For each entity type in component i
                for type_i, nodes_i in component_nodes[i].items():
                    # For each entity type in component j
                    for type_j, nodes_j in component_nodes[j].items():
                        # Check if these types should have relationships
                        rel_types = hierarchy_builder.get_relationship_types(type_i, type_j)
                        if rel_types:
                            # These components should be connected
                            if type_i not in dangling_nodes:
                                dangling_nodes[type_i] = []
                            if type_j not in dangling_nodes:
                                dangling_nodes[type_j] = []
                            dangling_nodes[type_i].extend(nodes_i)
                            dangling_nodes[type_j].extend(nodes_j)
                            
        return dangling_nodes

    def _format_entities(self, entities: List[BaseNode]) -> str:
        """Format entities for LLM prompt"""
        formatted = []
        for e in entities:
            props = {k:v for k,v in e.properties.items() if v is not None}
            formatted.append(f"ID: {e.id}\nType: {e.type}\nProperties: {props}")
        return "\n".join(formatted)
        
    def _format_relationships(self, relationships: List[RelationshipInstance]) -> str:
        """Format relationships for LLM prompt"""
        formatted = []
        for r in relationships:
            formatted.append(f"{r.source_id} -> {r.type} -> {r.target_id}")
        return "\n".join(formatted)

    def _validate_graph(self, graph: DocumentKnowledgeGraph) -> None:
        """
        Validate the generated knowledge graph against ontology rules.
        Checks:
        1. Required properties are present
        2. Property types match ontology
        3. Relationship constraints are satisfied
        4. No duplicate relationships
        """
        ontology = self.ontology_parser.parsed_ontology
        
        # Validate entities
        for entity_type, entity_def in ontology['entities'].items():
            entity_list = getattr(graph, f"{entity_type}_list", [])
            
            # Check required properties
            required_props = {
                name for name, prop in entity_def.get('properties', {}).items()
                if prop.get('required', False)
            }
            
            for entity in entity_list:
                missing = required_props - set(entity.model_fields().keys())
                if missing:
                    logger.warning(f"Entity {entity.id} missing required properties: {missing}")
                    
        # Validate relationships
        for source_type, source_def in ontology['entities'].items():
            for rel_name, rel_def in source_def.get('relationships', {}).items():
                rel_field = f"{source_type}_{rel_name}"
                
                if not hasattr(graph, rel_field):
                    continue
                    
                rel_list = getattr(graph, rel_field)
                valid_relationships = []
                
                for rel in rel_list:
                    # Check for duplicates
                    seen = set()
                    unique_rels = []
                    for rel in rel_list:
                        key = (rel.source.id, rel.target.id)
                        if key not in seen:
                            seen.add(key)
                            unique_rels.append(rel)
                        else:
                            logger.warning(f"Duplicate relationship found: {rel.source.id} -> {rel.target.id}")
                    
                    if len(unique_rels) != len(rel_list):
                        setattr(graph, rel_field, unique_rels)
                        
                    # Validate relationship properties
                    required_props = {
                        name for name, prop in rel_def.get('properties', {}).items()
                        if prop.get('required', False)
                    }
                    
                    for rel in unique_rels:
                        if hasattr(rel, 'properties'):
                            missing = required_props - set(rel.properties.keys())
                            if missing:
                                logger.warning(f"Relationship {rel.source.id}->{rel.target.id} missing required properties: {missing}")