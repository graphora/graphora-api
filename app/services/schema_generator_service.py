from typing import Dict, List, Optional, Any
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import instructor
from pydantic import BaseModel, create_model, Field
from app.config import settings
from app.utils.logger import logger
import datetime

class FieldDefinition(BaseModel):
    """Definition of a single field in a model"""
    field_type: str
    description: str
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    examples: Optional[List[str]] = Field(default_factory=list)

class ModelDefinition(BaseModel):
    """Definition of a complete model"""
    name: str
    description: str
    fields: Dict[str, FieldDefinition]
    dependencies: List[str] = Field(default_factory=list)

class SchemaDefinition(BaseModel):
    """Container for generated schema information"""
    model_definitions: Dict[str, Dict[str, Any]]
    assumptions: List[str] = Field(default_factory=list)

class OntologyResponse(BaseModel):
    """Response structure for LLM"""
    models: List[ModelDefinition]
    assumptions: List[str] = Field(default_factory=list)

class SchemaGeneratorService:
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

    def generate_schema(self, text: str) -> SchemaDefinition:
        """Generate Pydantic models from text description using LLM"""
        if not self.client:
            raise ValueError("LLM client not initialized")
            
        logger.info(f"Generating schema from text of length {len(text)}")
        
        try:
            # Prepare the prompt for the LLM
            system_prompt = """You are an expert in converting natural language descriptions into structured Pydantic models.
            Your task is to analyze the provided text and:
            1. Identify entities and their relationships
            2. Determine appropriate field types and validations
            3. Handle ambiguous or unclear descriptions by making reasonable assumptions
            4. Track any assumptions made during the process

            When inferring types:
            - Use string for text and identifiers
            - Use int for counts and whole numbers
            - Use float for measurements and decimals
            - Use datetime for timestamps
            - Use List[] for collections
            - Use proper type hints (str, int, float, datetime, List, Optional)
            
            For validation rules consider:
            - String patterns for identifiers
            - Min/max values for numbers
            - Required vs optional fields
            - Valid value ranges
            - Date ranges if applicable
            
            When fields are ambiguous:
            - Make reasonable assumptions based on context
            - Document your assumptions
            - Use optional fields when unsure
            - Consider common patterns in similar domains

            Return a complete schema with:
            - Clear model names and descriptions
            - Properly typed fields
            - Validation rules where appropriate
            - Documentation of assumptions made
            """

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a complete schema from this description:\n\n{text}"}
                ],
                response_model=OntologyResponse
            )
            
            logger.info(f"Generated schema with {len(response.models)} models")
            if response.assumptions:
                logger.info(f"Made {len(response.assumptions)} assumptions: {response.assumptions}")
                
            # Convert models to serializable format
            model_definitions = self._create_serializable_models(response.models)
            
            # Return serializable schema definition
            return SchemaDefinition(
                model_definitions=model_definitions,
                assumptions=response.assumptions
            )

        except Exception as e:
            logger.error(f"Error generating schema: {str(e)}")
            raise

    def _create_serializable_models(self, model_definitions: List[ModelDefinition]) -> Dict[str, Dict[str, Any]]:
        """Create serializable model definitions"""
        serializable_models = {}
        
        for model_def in model_definitions:
            fields_dict = {}
            for field_name, field_def in model_def.fields.items():
                try:
                    # Convert field type to string representation
                    field_type = self._get_type_str(field_def.field_type)
                    
                    # Create field definition
                    field_dict = {
                        "type": field_type,
                        "description": field_def.description,
                        "required": field_def.required,
                        "validation_rules": field_def.validation_rules
                    }
                    
                    fields_dict[field_name] = field_dict
                    
                except Exception as e:
                    logger.warning(f"Error processing field {field_name}: {str(e)}")
                    fields_dict[field_name] = {
                        "type": "any",
                        "description": field_def.description,
                        "required": field_def.required
                    }
            
            model_dict = {
                "name": model_def.name,
                "description": model_def.description,
                "fields": fields_dict,
                "dependencies": model_def.dependencies
            }
            
            serializable_models[model_def.name] = model_dict
        
        return serializable_models

    def _get_type_str(self, type_str: str) -> str:
        """Convert type to string representation"""
        type_str = type_str.lower().strip()
        
        # Handle List types
        if type_str.startswith(("list[", "list[")):
            inner_type = type_str[5:-1].strip()
            return f"List[{self._get_type_str(inner_type)}]"
        
        # Handle Optional types
        if type_str.startswith(("optional[", "optional[")):
            inner_type = type_str[9:-1].strip()
            return f"Optional[{self._get_type_str(inner_type)}]"
        
        # Map basic types
        type_mapping = {
            "str": "str",
            "string": "str",
            "int": "int",
            "integer": "int",
            "float": "float",
            "bool": "bool",
            "boolean": "bool",
            "date": "datetime.date",
            "datetime": "datetime.datetime",
            "dict": "Dict",
            "any": "Any"
        }
        
        return type_mapping.get(type_str, type_str)