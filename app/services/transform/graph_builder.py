from typing import Dict, List, Any, Type, Tuple, Optional, Callable, Union
import yaml
import copy
import asyncio
from datetime import datetime
from pathlib import Path
from pydantic import create_model, Field

from app.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    KnowledgeGraph,
    ExtractionMetrics
)
from app.services.llm.client import LLMClient
from app.utils.logger import logger

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
    
    def get_entity_definitions(self) -> Dict[str, Dict]:
        """Get all entity definitions"""
        return self.parsed_ontology['entities']
    
    def get_relationship_definitions(self) -> Dict[str, Dict[str, Dict]]:
        """Get all relationship definitions"""
        relationships = {}
        for entity, definition in self.parsed_ontology['entities'].items():
            if 'relationships' in definition:
                relationships[entity] = definition['relationships']
        return relationships
                
    def generate_pydantic_models(self) -> Dict[str, Type[BaseNode]]:
        """Generate Pydantic models for all entities"""
        models = {}
        
        for entity_name, definition in self.get_entity_definitions().items():
            # Prepare field definitions
            fields = {
                # Base fields all nodes must have
                'id': (str, ...),
                'type': (str, ...),
                'confidence_score': (Optional[float], None),
                'provenance': (NodeProvenance, Field(default_factory=NodeProvenance)),
            }
            
            # Add entity-specific fields
            if 'properties' in definition:
                for prop_name, prop_def in definition['properties'].items():
                    prop_type = self._map_yaml_type_to_python(prop_def['type'])
                    required = prop_def.get('required', False)
                    
                    if required:
                        fields[prop_name] = (prop_type, ...)
                    else:
                        fields[prop_name] = (Optional[prop_type], None)
            
            # Create the model dynamically
            models[entity_name] = create_model(
                entity_name,
                __base__=BaseNode,
                **fields
            )
            
        return models
    
    
    def create_ontology_spec(
        self
    ) -> str:
        """Format ontology for LLM prompt"""
        spec = []
        models = self.generate_pydantic_models()
        relationships = self.get_relationship_definitions()
        
        # Add entities
        spec.append("Entities:")
        for entity_name, model in models.items():
            properties = []
            for field_name, field in model.model_fields.items():
                if field_name not in ['id', 'type', 'confidence_score', 'provenance']:
                    # Get field type from annotation
                    field_type = field.annotation
                    if hasattr(field_type, "__origin__"):
                        # Handle Optional types
                        if field_type.__origin__ is Union:
                            field_type = field_type.__args__[0]
                    
                    # Get the type name
                    type_name = getattr(field_type, "__name__", str(field_type))
                    required = field.is_required()
                    properties.append(f"  - {field_name}: {type_name} {'(required)' if required else '(optional)'}")
            
            spec.append(f"- {entity_name}:")
            spec.append("  Properties:")
            spec.extend(properties)
        
        # Add relationships
        spec.append("\nRelationships:")
        for source_entity, rels in relationships.items():
            for rel_type, rel_def in rels.items():
                target = rel_def['target']
                spec.append(f"- {source_entity} -{rel_type}-> {target}")
                
                if 'properties' in rel_def:
                    spec.append("  Properties:")
                    for prop_name, prop_def in rel_def['properties'].items():
                        prop_type = prop_def['type']
                        required = prop_def.get('required', False)
                        spec.append(f"  - {prop_name}: {prop_type} {'(required)' if required else '(optional)'}")
        
        return "\n".join(spec)
    
    def _map_yaml_type_to_python(self, yaml_type: str) -> Any:
        """Map YAML type to Python type"""
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list[str]': List[str],
            'list[int]': List[int],
            'list[float]': List[float],
            'date': datetime.date,
            'datetime': datetime,
        }
        return type_mapping.get(yaml_type, str)

class KnowledgeGraphBuilder:
    """Builds unified knowledge graph from document chunks"""
    
    def __init__(
        self,
        ontology_parser: OntologyParser
    ):
        self.ontology_parser = ontology_parser
        self.models = ontology_parser.generate_pydantic_models()
        self.relationships = ontology_parser.get_relationship_definitions()
        self.ontology_spec = ontology_parser.create_ontology_spec()
        self.graph = KnowledgeGraph()
        self.metrics = ExtractionMetrics(start_time=datetime.now())
        self.llm_client = LLMClient()
        
        # Create dynamic entity list models
        entity_list_models = {}
        for entity_name, model in self.models.items():
            entity_list_models[f"{entity_name}_list"] = (List[model], Field(default_factory=list))
        
        # Create dynamic relationship models
        relationship_models = {}
        for source, rels in self.relationships.items():
            for rel_type, rel_def in rels.items():
                rel_name = f"{source}_{rel_type}_{rel_def['target']}"
                relationship_models[rel_name] = (
                    List[RelationshipInstance],
                    Field(default_factory=list)
                )
        
        # Create the ontology model
        self.ontology_model = create_model(
            "OntologyExtraction",
            **{
                **entity_list_models,
                **relationship_models,
                "extraction_timestamp": (datetime, Field(default_factory=datetime.utcnow)),
                "tokens_used": (Optional[int], None),
                "confidence_score": (Optional[float], None)
            }
        )
        
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
                ontology_spec=self.ontology_spec,
                response_model=self.ontology_model
            )
            
            # Track metrics
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Extract nodes from all entity lists
            nodes = []
            for field_name, field_value in extraction_result.__dict__.items():
                if field_name.endswith('_list') and field_value:
                    entity_type = field_name[:-5]  # Remove '_list' suffix
                    for node in field_value:
                        node.type = entity_type
                        node.provenance = NodeProvenance(
                            chunk_ids=[chunk_id],
                            extraction_timestamp=datetime.now(),
                            confidence_score=extraction_result.confidence_score
                        )
                        nodes.append(node)
            
            # Extract relationships
            relationships = []
            for field_name, field_value in extraction_result.__dict__.items():
                if not field_name.endswith('_list') and field_value:
                    # Format: source_reltype_target
                    parts = field_name.split('_')
                    if len(parts) >= 3 and isinstance(field_value, list):
                        for rel in field_value:
                            rel.provenance = NodeProvenance(
                                chunk_ids=[chunk_id],
                                extraction_timestamp=datetime.now(),
                                confidence_score=extraction_result.confidence_score
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
                    'extraction_timestamp': extraction_result.extraction_timestamp
                }
            }
            
        except Exception as e:
            self.metrics.record_failure(chunk_id, str(e))
            logger.error(f"Extraction failed for chunk {chunk_id}: {str(e)}")
            return None
    
    async def process_chunks(
        self,
        chunks: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[KnowledgeGraph, ExtractionMetrics]:
        """Process multiple chunks and build knowledge graph"""
        
        # Initialize metrics
        self.metrics.total_chunks = len(chunks)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            result = await self.process_chunk(chunk, chunk_id)
            
            if result:
                # Add nodes and relationships to graph
                for node in result['nodes']:
                    self.graph.add_node(node)
                for rel in result['relationships']:
                    self.graph.add_relationship(rel)
                    
                # Update relationship count
                self.metrics.record_node_stats(
                    self.metrics.new_nodes,
                    self.metrics.merged_nodes,
                    self.metrics.total_relationships + len(result['relationships'])
                )
            
            # Update progress
            if progress_callback:
                progress_callback(i + 1, len(chunks))
        
        return self.graph, self.metrics
    
    def _find_matching_node(self, node: BaseNode) -> Optional[BaseNode]:
        """Find matching node in graph based on identity properties"""
        entity_type = node.type
        
        # Get entity definition
        entity_def = self.ontology_parser.get_entity_definitions().get(entity_type)
        if not entity_def:
            return None
            
        # Find unique properties
        unique_props = [
            prop_name for prop_name, prop_def in entity_def['properties'].items()
            if prop_def.get('unique', False) and hasattr(node, prop_name) and 
            getattr(node, prop_name) is not None
        ]
        
        if not unique_props:
            return None
            
        # Search for matching node
        for existing_node in self.graph.get_nodes_by_type(entity_type):
            for prop in unique_props:
                if (
                    hasattr(existing_node, prop) and
                    getattr(existing_node, prop) == getattr(node, prop)
                ):
                    return existing_node
                    
        return None
    
    def merge_nodes(self, existing_node: BaseNode, new_node: BaseNode) -> BaseNode:
        """Merge two nodes, preserving the best information"""
        merged_node = copy.deepcopy(existing_node)
        
        # Merge all properties except ID, type and provenance
        for field_name in new_node.model_fields.keys():
            if field_name in ['id', 'type']:
                continue
                
            if field_name == 'provenance':
                # Merge provenance
                merged_node.provenance.chunk_ids.extend(new_node.provenance.chunk_ids)
                merged_node.provenance.chunk_ids = list(set(merged_node.provenance.chunk_ids))
                continue
                
            # Get new value
            new_value = getattr(new_node, field_name)
            if new_value is None:
                continue
                
            # Get existing value
            existing_value = getattr(merged_node, field_name)
            
            if existing_value is None:
                # Take new value if existing is None
                setattr(merged_node, field_name, new_value)
            elif isinstance(existing_value, list) and isinstance(new_value, list):
                # Merge lists without duplicates
                merged_list = list(set(existing_value + new_value))
                setattr(merged_node, field_name, merged_list)
            elif new_node.confidence_score and existing_node.confidence_score:
                # Take higher confidence value
                if new_node.confidence_score > existing_node.confidence_score:
                    setattr(merged_node, field_name, new_value)
            elif len(str(new_value)) > len(str(existing_value)):
                # Take more complete value (heuristic)
                setattr(merged_node, field_name, new_value)
        
        # Update confidence score to max of both
        if new_node.confidence_score and existing_node.confidence_score:
            merged_node.confidence_score = max(
                new_node.confidence_score,
                existing_node.confidence_score
            )
            
        return merged_node
    
    async def build_graph_from_chunks(
        self,
        chunks: List[str],
        transform_id: str,
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> KnowledgeGraph:
        """Process all chunks and build unified graph"""
        chunk_results = []
        
        # Process chunks with controlled concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(chunk: str, idx: int):
            async with semaphore:
                result = await self.process_chunk(chunk, f"{transform_id}_chunk_{idx}")
                if progress_callback:
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
        """Add extraction result to graph with node merging"""
        # Add all nodes with merging
        for node in result['nodes']:
            matching_node = self._find_matching_node(node)
            
            if matching_node:
                # Merge with existing node
                merged_node = self.merge_nodes(matching_node, node)
                self.graph.update_node(merged_node)
                self.metrics.merged_nodes += 1
            else:
                # Add as new node
                self.graph.add_node(node)
                self.metrics.new_nodes += 1
        
        # Add all relationships
        for rel in result['relationships']:
            self.graph.add_relationship(rel)
            self.metrics.relationships_added += 1
    
    def finalize_graph(self) -> KnowledgeGraph:
        """Validate and finalize the graph"""
        # Validate all relationships point to existing nodes
        self._validate_relationships()
        
        # Finalize metrics
        self.graph.metrics = self.metrics
        
        return self.graph
        
    def _validate_relationships(self) -> None:
        """Ensure all relationships point to existing nodes"""
        valid_relationships = []
        
        for rel in self.graph.relationships:
            if (
                self.graph.has_node(rel.source_id) and
                self.graph.has_node(rel.target_id)
            ):
                valid_relationships.append(rel)
            else:
                self.metrics.invalid_relationships += 1
                
        self.graph.relationships = valid_relationships
