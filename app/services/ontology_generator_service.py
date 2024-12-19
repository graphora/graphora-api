from typing import List, Optional, Any
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import instructor
from pydantic import BaseModel, field_validator, Field
from app.config import settings
from app.utils.logger import logger
from enum import Enum
from app.utils.llm_client_service import LLMClientService

class PropertyType(str, Enum):
    """Neo4j-compatible property types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    LIST = "list"

class PropertyDefinition(BaseModel):
    """Definition of a Neo4j property"""
    name: str
    type: PropertyType
    required: bool = False
    description: str
    default_value: Optional[Any] = None
    array_type: Optional[PropertyType] = None
    indexed: bool = False
    unique: bool = False
    examples: List[str] = Field(
        default_factory=list,
        description="Example values as strings, e.g., ['John Doe', 'Jane Smith']"
    )

    @field_validator('examples')
    def validate_examples(cls, v):
        """Ensure examples are strings"""
        return [str(example) for example in v]

class NodeLabel(BaseModel):
    """Neo4j node label definition"""
    name: str
    description: str
    
    @field_validator('name')
    def validate_label_name(cls, v):
        if not v[0].isupper():
            raise ValueError("Node label must start with uppercase letter")
        if ' ' in v:
            raise ValueError("Node label cannot contain spaces")
        return v

class NodeDefinition(BaseModel):
    """Complete Neo4j node definition"""
    label: NodeLabel
    properties: List[PropertyDefinition]
    indexes: List[List[str]] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    examples: List[str] = Field(
        default_factory=list,
        description="Example nodes as Cypher-like strings"
    )

    @field_validator('examples')
    def validate_examples(cls, v, values):
        """Convert dictionary examples to strings if needed"""
        formatted_examples = []
        for example in v:
            if isinstance(example, dict):
                # Convert dict to Cypher-like string
                props = [f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}" 
                        for k, v in example.items()]
                formatted = f"({values['label'].name} {{{', '.join(props)}}})"
                formatted_examples.append(formatted)
            else:
                formatted_examples.append(str(example))
        return formatted_examples

class RelationshipLabel(BaseModel):
    """Neo4j relationship type definition"""
    name: str
    description: str
    
    @field_validator('name')
    def validate_relationship_name(cls, v):
        if not v.isupper() or ' ' in v:
            raise ValueError("Relationship name must be uppercase with underscores")
        return v

class RelationshipDirection(str, Enum):
    """Neo4j relationship direction"""
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    BOTH = "BOTH"

class RelationshipDefinition(BaseModel):
    """Complete Neo4j relationship definition"""
    label: RelationshipLabel
    source_label: str
    target_label: str
    direction: RelationshipDirection = RelationshipDirection.RIGHT
    properties: Optional[List[PropertyDefinition]] = None
    constraints: List[str] = Field(default_factory=list)
    examples: List[str] = Field(
        default_factory=list,
        description="Example relationships as Cypher patterns"
    )
    cypher_patterns: List[str] = Field(default_factory=list)

    def get_cypher_pattern(self, include_props: bool = False) -> str:
        """Generate a Cypher pattern based on the relationship definition"""
        source_node = f"({self.source_label.lower()[0]}:{self.source_label})"
        target_node = f"({self.target_label.lower()[0]}:{self.target_label})"
        rel_label = f"[r:{self.label.name}]"
        
        if include_props and self.properties:
            # Add required properties to pattern
            props = {
                p.name: "value" 
                for p in self.properties 
                if p.required
            }
            if props:
                props_str = ", ".join(f"{k}: '{v}'" for k, v in props.items())
                rel_label = f"[r:{self.label.name} {{{props_str}}}]"
        
        match self.direction:
            case RelationshipDirection.RIGHT:
                return f"{source_node}-{rel_label}->{target_node}"
            case RelationshipDirection.LEFT:
                return f"{source_node}<-{rel_label}-{target_node}"
            case RelationshipDirection.BOTH:
                return f"{source_node}<-{rel_label}->{target_node}"

    @field_validator('cypher_patterns')
    def validate_cypher_patterns(cls, v, values) -> List[str]:
        """Ensure cypher patterns exist and are valid"""
        if not v:
            # If no patterns provided, create a default one
            try:
                # Create temporary instance to generate pattern
                # This is needed because the full instance isn't created yet
                temp_rel = RelationshipDefinition(
                    label=values['label'],
                    source_label=values['source_label'],
                    target_label=values['target_label'],
                    direction=values.get('direction', RelationshipDirection.RIGHT)
                )
                return [temp_rel.get_cypher_pattern()]
            except Exception as e:
                logger.warning(f"Failed to generate default Cypher pattern: {str(e)}")
                return [""]
        return v

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "label": {
                    "name": "CONTAINS",
                    "description": "Indicates a document contains a section"
                },
                "source_label": "Form10K",
                "target_label": "PartI",
                "direction": "RIGHT",
                "properties": [
                    {
                        "name": "order",
                        "type": "integer",
                        "required": True,
                        "description": "Order of the section in document",
                        "examples": ["1"]
                    }
                ],
                "cypher_patterns": [
                    "(f:Form10K)-[r:CONTAINS]->(p:PartI)",
                    "(f:Form10K)-[r:CONTAINS {order: 1}]->(p:PartI)"
                ]
            }
        }

class Neo4jOntology(BaseModel):
    """Complete Neo4j database structure"""
    nodes: List[NodeDefinition]
    relationships: List[RelationshipDefinition]
    indexes: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)

class OntologyGeneratorService:
    def __init__(self):
        self.llm_service = LLMClientService()
        
    def generate_ontology(self, text: str, max_retries: int = 5) -> Neo4jOntology:
        """Generate Neo4j ontology from text description with validation feedback loop"""
        if not self.llm_service.is_available:
            raise ValueError("LLM client not initialized")
            
        logger.info(f"Generating Neo4j ontology from text of length {len(text)}")
        
        attempt = 0
        previous_errors = []
        
        while attempt < max_retries:
            try:
                # Create refinement prompt based on previous errors
                if attempt == 0:
                    system_prompt = """You are an expert in Neo4j graph database design.
                    Create a complete Neo4j graph structure following these specific rules:

                    1. Node Labels:
                    - Use PascalCase (e.g., Form10K, BusinessSection)
                    - Include required and optional properties
                    - Provide string examples in this format:
                        "(Form10K {uniqueId: '10K-2024-001', fiscalYear: 2024})"

                    2. Property Examples:
                    - Always provide as strings, even for numbers and lists
                    - For lists: "['value1', 'value2']"
                    - For dates: "'2024-12-31'"
                    - For numbers: "'123'" or "'123.45'"

                    3. Relationship Types:
                    - Use UPPERCASE_WITH_UNDERSCORES
                    - Provide Cypher pattern examples:
                        "(Form10K)-[r:CONTAINS]->(PartI)"

                    4. Example Format:
                    - Node examples should be Cypher-like strings
                    - Include property values in examples
                    - Use single quotes for string values
                    - Format lists as string representations

                    Examples of correct formats:
                    - Node example: "(Form10K {uniqueId: '10K-2024-001', companyName: 'Acme Inc'})"
                    - Property example: ['Manufacturing', 'Sales']
                    - Relationship pattern: "(Business)-[r:HAS_SEGMENT]->(Segment)"

                    Avoid returning raw dictionaries or lists as examples.
                    Always convert them to string representations."""
                else:
                    system_prompt = f"""You are an expert in Neo4j graph database design.
                    Your previous ontology design had the following issues that need to be fixed:

                    {self._format_validation_errors(previous_errors)}

                    Please revise the ontology to address these specific issues while maintaining all other correct aspects.
                    Ensure you:
                    1. Fix each validation error listed above
                    2. Maintain the correct parts of the previous design
                    3. Verify the fixes don't introduce new issues
                    4. Include proper indexes and constraints
                    5. Provide updated Cypher patterns

                    Original description:
                    {text}

                    Generate a corrected version of the Neo4j structure addressing all issues."""

                # Generate or refine ontology
                response = self.llm_service.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate a Neo4j graph structure from this description:\n\n{text}"}
                    ],
                    response_model=Neo4jOntology
                )
                
                # Validate the generated ontology
                try:
                    self._validate_ontology(response)
                    logger.info(f"Successfully generated valid ontology after {attempt + 1} attempts")
                    return response
                except ValueError as ve:
                    validation_errors = str(ve).split('\n')
                    if validation_errors == previous_errors:
                        logger.warning("LLM unable to fix validation errors after retry")
                        raise ValueError(f"Unable to generate valid ontology after {attempt + 1} attempts. "
                                      f"Persistent errors: {validation_errors}")
                    previous_errors = validation_errors
                    attempt += 1
                    logger.info(f"Validation failed, attempting refinement (attempt {attempt + 1}/{max_retries})")
                    continue

            except Exception as e:
                logger.error(f"Error in ontology generation attempt {attempt + 1}: {str(e)}")
                raise

        raise ValueError(f"Failed to generate valid ontology after {max_retries} attempts. "
                        f"Last validation errors: {previous_errors}")

    def _format_validation_errors(self, errors: List[str]) -> str:
        """Format validation errors for LLM prompt"""
        formatted = "Current validation errors that need to be fixed:\n\n"
        for i, error in enumerate(errors, 1):
            formatted += f"{i}. {error}\n"
            formatted += self._get_error_guidance(error) + "\n"
        return formatted

    def _get_error_guidance(self, error: str) -> str:
        """Provide specific guidance for different types of validation errors"""
        if "needs at least one indexed identifier property" in error:
            return "   - Add a required, indexed property that uniquely identifies the node\n" \
                   "   - Consider using properties like 'id', 'uuid', or natural business keys\n" \
                   "   - Mark the property as both required and indexed"
        
        elif "references unknown" in error:
            return "   - Ensure the referenced node label is defined in the ontology\n" \
                   "   - Check for typos in node label names\n" \
                   "   - Verify the relationship connects existing node types"
        
        elif "Duplicate relationship label" in error:
            return "   - Choose a unique name for this relationship\n" \
                   "   - Consider adding context to the relationship name\n" \
                   "   - Ensure relationship names are descriptive and distinct"
        
        elif "missing examples" in error:
            return "   - Add realistic example instances\n" \
                   "   - Include Cypher patterns showing typical usage\n" \
                   "   - Provide examples with property values"
        
        elif "relationship label must be uppercase" in error:
            return "   - Convert relationship name to uppercase\n" \
                   "   - Use underscores between words\n" \
                   "   - Follow Neo4j relationship naming conventions"
        
        return "   - Review Neo4j best practices for this issue\n" \
               "   - Ensure the fix follows Neo4j conventions\n" \
               "   - Validate the correction against requirements"


    def _validate_ontology(self, ontology: Neo4jOntology) -> None:
        """Validate Neo4j-specific aspects of the ontology"""
        errors = []
        
        # Track defined node labels
        node_labels = {node.label.name for node in ontology.nodes}
        
        # Validate nodes
        for node in ontology.nodes:
            # Check for at least one unique identifier property
            has_identifier = any(
                prop.required and (prop.unique or prop.indexed)
                for prop in node.properties
            )
            if not has_identifier:
                errors.append(f"Node {node.label.name} needs at least one indexed identifier property")
            
            # Check for examples
            if not node.examples:
                errors.append(f"Node {node.label.name} missing examples")
        
        # Validate relationships
        used_labels = set()
        for rel in ontology.relationships:
            # Check for duplicate labels
            if rel.label.name in used_labels:
                errors.append(f"Duplicate relationship label: {rel.label.name}")
            used_labels.add(rel.label.name)
            
            # Validate node references
            if rel.source_label not in node_labels:
                errors.append(f"Relationship {rel.label.name} references unknown source label {rel.source_label}")
            if rel.target_label not in node_labels:
                errors.append(f"Relationship {rel.label.name} references unknown target label {rel.target_label}")
            
            # Ensure Cypher patterns exist
            if not rel.cypher_patterns:
                rel.cypher_patterns = [rel.get_cypher_pattern()]
            
            # Validate pattern format
            for pattern in rel.cypher_patterns:
                if not pattern or not isinstance(pattern, str):
                    errors.append(f"Invalid Cypher pattern in relationship {rel.label.name}")
        
        if errors:
            raise ValueError("Neo4j ontology validation failed:\n" + "\n".join(errors))

    def generate_cypher_schema(self, ontology: Neo4jOntology) -> List[str]:
        """Generate Cypher commands for creating the schema"""
        commands = []
        
        # Create indexes and constraints for nodes
        for node in ontology.nodes:
            for idx in node.indexes:
                if len(idx) == 1:
                    commands.append(f"CREATE INDEX FOR (n:{node.label.name}) ON (n.{idx[0]})")
                else:
                    props = ", ".join(f"n.{p}" for p in idx)
                    commands.append(f"CREATE INDEX FOR (n:{node.label.name}) ON ({props})")
            
            for constraint in node.constraints:
                commands.append(constraint)
        
        # Add any database-level constraints
        commands.extend(ontology.constraints)
        
        return commands
