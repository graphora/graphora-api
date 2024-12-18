from typing import Dict, List, Optional, Any
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import instructor
from pydantic import BaseModel, create_model, Field
from app.config import settings
from app.utils.logger import logger
from enum import Enum

class PropertyType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    BOOLEAN = "boolean"

class OntologyProperty(BaseModel):
    """Definition of a property in the ontology"""
    name: str
    type: PropertyType
    required: bool = False
    description: str
    examples: List[str] = Field(default_factory=list)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)

class OntologyNode(BaseModel):
    """Definition of a node type in the ontology"""
    type: str
    description: str
    properties: List[OntologyProperty]
    examples: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)

class OntologyRelationship(BaseModel):
    """Definition of a relationship type in the ontology"""
    type: str
    source: str
    target: str
    description: str
    properties: Optional[List[OntologyProperty]] = None
    examples: List[str] = Field(default_factory=list)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)

class OntologyResponse(BaseModel):
    """Response structure for LLM"""
    nodes: List[OntologyNode]
    relationships: List[OntologyRelationship]
    assumptions: List[str] = Field(default_factory=list)

class OntologyGeneratorService:
    def __init__(self):
        self.client = None
        self._init_client()
        
    def _init_client(self) -> None:
        """Initialize OpenAI client with instructor patch"""
        if not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY and not settings.GOOGLE_GEMINI_API_KEY:
            logger.warning("LLM API key not set. Schema generation will be unavailable.")
            return
        
        try:
            if settings.OPENAI_API_KEY.strip():
                base_client = OpenAI(api_key=settings.OPENAI_API_KEY.strip())
                self.client = instructor.from_openai(base_client)
                self.model = 'gpt-4'
            elif settings.ANTHROPIC_API_KEY.strip():
                base_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY.strip())
                self.client = instructor.from_anthropic(base_client)
                self.model = 'claude-3-5-haiku-20241022'
            else:
                genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY.strip())
                self.client = instructor.from_gemini(
                                    client=genai.GenerativeModel(
                                        model_name="models/gemini-1.5-flash-latest",
                                    ),
                                    mode=instructor.Mode.GEMINI_JSON,
                                )
                self.model = 'models/gemini-1.5-flash-latest'
            logger.info(f"LLM client [{str(self.client.provider.name)}] initialized successfully for schema generation")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.client = None

    def generate_ontology(self, text: str) -> OntologyResponse:
        """Generate ontology from text description using LLM"""
        if not self.client:
            raise ValueError("LLM client not initialized")
            
        logger.info(f"Generating ontology from text of length {len(text)}")
        
        try:
            system_prompt = """You are an expert in knowledge graphs and ontology design.
            Your task is to analyze the provided text and:
            1. Identify entity types (nodes) and their properties
            2. Identify relationships between entities
            3. Determine appropriate property types and validations
            4. Handle ambiguous cases by making reasonable assumptions
            5. Provide examples for each node and relationship type

            When defining properties:
            - Use string for text and identifiers
            - Use integer for counts and whole numbers
            - Use float for measurements and decimals
            - Use date for timestamps
            - Use boolean for flags

            For relationships:
            - Clearly specify source and target node types
            - Include properties relevant to the relationship
            - Consider relationship direction
            - Include examples showing usage

            For validation rules consider:
            - Required vs optional properties
            - Value ranges and constraints
            - Pattern matching for identifiers
            - Cardinality of relationships

            When handling ambiguity:
            - Make reasonable assumptions based on domain knowledge
            - Document all assumptions made
            - Use optional properties when uncertain
            - Consider common patterns in similar domains

            Ensure all node and relationship types have:
            - Clear descriptions
            - Example instances
            - Proper property definitions
            - Validation rules where appropriate"""

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a complete ontology from this description:\n\n{text}"}
                ],
                response_model=OntologyResponse
            )
            
            logger.info(f"Generated ontology with {len(response.nodes)} nodes and {len(response.relationships)} relationships")
            if response.assumptions:
                logger.info(f"Made {len(response.assumptions)} assumptions: {response.assumptions}")
                
            return response

        except Exception as e:
            logger.error(f"Error generating ontology: {str(e)}")
            raise

    def validate_ontology(self, ontology: OntologyResponse) -> List[str]:
        """Validate the generated ontology"""
        errors = []
        
        # Track defined node types
        node_types = {node.type for node in ontology.nodes}
        
        # Validate nodes
        for node in ontology.nodes:
            # Check for required identifier property
            has_identifier = any(
                prop.required and prop.type in [PropertyType.STRING, PropertyType.INTEGER]
                for prop in node.properties
            )
            if not has_identifier:
                errors.append(f"Node type {node.type} lacks required identifier property")
            
            # Check for examples
            if not node.examples:
                errors.append(f"Node type {node.type} missing examples")
        
        # Validate relationships
        for rel in ontology.relationships:
            # Check node type references
            if rel.source not in node_types:
                errors.append(f"Relationship {rel.type} references unknown source type {rel.source}")
            if rel.target not in node_types:
                errors.append(f"Relationship {rel.type} references unknown target type {rel.target}")
            
            # Check for examples
            if not rel.examples:
                errors.append(f"Relationship {rel.type} missing examples")
        
        return errors