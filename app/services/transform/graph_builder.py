import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Type
from pydantic import create_model, ValidationError
import uuid
from functools import lru_cache

from app.services.transform.models import (
    PropertyType,
    PropertyDefinition,
    OntologyDefinition,
    BaseNode,
    NodeProvenance,
    KnowledgeGraph,
    RelationshipInstance
)
from app.services.llm.client import call_llm_gemini

class ModelGenerator:
    """Generates Pydantic models from YAML ontology"""
    
    def __init__(self, ontology_yaml: str):
        self.ontology = yaml.safe_load(ontology_yaml)
        self.validated_ontology = OntologyDefinition(**self.ontology)
        self.models: Dict[str, Type[BaseNode]] = {}
    
    def _get_python_type(self, prop: PropertyDefinition) -> Tuple[Type, Any]:
        """Convert ontology types to Python types"""
        type_mapping = {
            PropertyType.STRING: (str, None),
            PropertyType.INTEGER: (int, None),
            PropertyType.FLOAT: (float, None),
            PropertyType.DATETIME: (datetime, None),
            PropertyType.BOOLEAN: (bool, None),
            PropertyType.LIST: (List[self._get_list_item_type(prop)], []),
            PropertyType.OBJECT: (Dict[str, Any], {}),
            PropertyType.REFERENCE: (str, None)  # References stored as IDs
        }
        return type_mapping.get(prop.type, (str, None))
    
    def _get_list_item_type(self, prop: PropertyDefinition) -> Type:
        """Get the type for list items"""
        if not prop.items_type:
            return str
        
        type_mapping = {
            PropertyType.STRING: str,
            PropertyType.INTEGER: int,
            PropertyType.FLOAT: float,
            PropertyType.DATETIME: datetime,
            PropertyType.BOOLEAN: bool,
            PropertyType.REFERENCE: str
        }
        return type_mapping.get(prop.items_type, str)
    
    def _create_property_validators(
        self,
        entity: str,
        props: Dict[str, PropertyDefinition]
    ) -> Dict[str, classmethod]:
        """Create validators for properties"""
        validators = {}
        
        for prop_name, prop in props.items():
            if prop.type == PropertyType.REFERENCE:
                # Create validator for reference properties
                def make_validator(target_entity):
                    @classmethod
                    def validate_reference(cls, v):
                        if not v.startswith(f"{target_entity}_"):
                            raise ValueError(
                                f"Invalid reference format for {target_entity}"
                            )
                        return v
                    return validate_reference
                
                validator_name = f"validate_{prop_name}"
                validators[validator_name] = make_validator(prop.reference_to)
        
        return validators
    
    def generate_models(self) -> Dict[str, Type[BaseNode]]:
        """Generate Pydantic models from ontology"""
        # First pass: Create basic models
        for entity_name, entity in self.validated_ontology.entities.items():
            # Create properties dict
            properties = {
                'id': (str, ...),  # Required unique ID
                'type': (str, entity_name),  # Entity type
                'provenance': (NodeProvenance, ...),  # Required provenance
            }
            
            # Add entity properties
            for prop_name, prop in entity.properties.items():
                python_type, default = self._get_python_type(prop)
                properties[prop_name] = (
                    python_type,
                    ... if prop.required else default
                )
            
            # Create model
            validators = self._create_property_validators(
                entity_name,
                entity.properties
            )
            
            model = create_model(
                entity_name,
                __base__=BaseNode,
                **properties,
                __validators__=validators
            )
            
            self.models[entity_name] = model
        
        return self.models

class KnowledgeGraphBuilder:
    """Builds knowledge graph from document chunks"""
    
    def __init__(
        self,
        models: Dict[str, Type[BaseNode]],
        llm_config: Optional[Dict[str, Any]] = None
    ):
        self.models = models
        self.graph = KnowledgeGraph()
        self.llm_config = llm_config or {}
    
    def _generate_node_id(self, entity_type: str) -> str:
        """Generate unique ID for a node"""
        return f"{entity_type}_{uuid.uuid4().hex[:12]}"
    
    def _validate_extraction(
        self,
        extraction: Dict[str, Any]
    ) -> List[BaseNode]:
        """Validate extracted data against models"""
        validated_nodes = []
        
        for entity_type, instances in extraction.items():
            if entity_type not in self.models:
                continue
                
            model = self.models[entity_type]
            
            for instance in instances:
                try:
                    # Add required fields
                    instance['id'] = self._generate_node_id(entity_type)
                    instance['type'] = entity_type
                    
                    # Create node
                    node = model(**instance)
                    validated_nodes.append(node)
                    
                except ValidationError as e:
                    self.graph.metrics.add_failed_chunk(
                        instance.get('id', 'unknown'),
                        f"Validation error: {str(e)}"
                    )
        
        return validated_nodes
    
    @lru_cache(maxsize=1000)
    def _get_extraction_prompt(self, entity_type: str) -> str:
        """Get cached extraction prompt for entity type"""
        model = self.models[entity_type]
        fields = model.model_fields
        
        prompt = f"""
        Extract information about {entity_type} from the following text.
        Required fields: {[f for f, field in fields.items() if field.required]}
        Optional fields: {[f for f, field in fields.items() if not field.required]}
        
        Format the output as a JSON object with these fields.
        If a field's information is not present, omit it from the output.
        
        Text: {{text}}
        """
        return prompt.strip()
    
    async def process_chunk(
        self,
        chunk: str,
        chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        """Process single chunk with LLM"""
        start_time = datetime.now()
        
        try:
            # Call LLM for extraction
            extraction = await call_llm_gemini(
                chunk,
                self.models,
                track_metrics=True
            )
            
            # Track metrics
            duration_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            self.graph.metrics.track_extraction_time(duration_ms)
            
            # Validate extraction
            validated_nodes = self._validate_extraction(extraction)
            
            # Add chunk provenance
            for node in validated_nodes:
                node.provenance.chunk_ids.append(chunk_id)
            
            # Update success metrics
            self.graph.metrics.successful_chunks += 1
            
            return validated_nodes
            
        except Exception as e:
            self.graph.metrics.add_failed_chunk(chunk_id, str(e))
            return None
    
    def add_nodes_to_graph(
        self,
        nodes: List[BaseNode],
        relationships: Optional[List[RelationshipInstance]] = None
    ) -> None:
        """Add nodes and relationships to graph"""
        # Add nodes
        for node in nodes:
            self.graph.add_node(node)
        
        # Add relationships
        if relationships:
            for rel in relationships:
                self.graph.add_relationship(rel)
    
    def finalize_graph(self) -> KnowledgeGraph:
        """Finalize and return the graph"""
        self.graph.finalize()
        return self.graph
