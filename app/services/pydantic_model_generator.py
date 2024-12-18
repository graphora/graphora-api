from typing import Dict, List, Optional, Any, Type
from pydantic import BaseModel, create_model, Field
from app.services.schema_generator_service import SchemaDefinition
import datetime
from functools import lru_cache

class PydanticModelGenerator:
    """Helper class to generate Pydantic models from schema definitions"""
    
    def __init__(self):
        self.type_mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "bool": bool,
            "boolean": bool,
            "datetime.date": datetime.date,
            "datetime.datetime": datetime.datetime,
            "dict": Dict,
            "any": Any,
            "object": Dict[str, Any]  # Default mapping for object type
        }
        self.generated_models: Dict[str, Type[BaseModel]] = {}

    def generate_models(self, schema_definition: SchemaDefinition) -> Dict[str, Type[BaseModel]]:
        """Generate all Pydantic models from schema definition"""
        model_definitions = schema_definition.model_definitions
        
        # Clear any previously generated models
        self.generated_models = {}
        
        # First pass: Create placeholder models for circular dependencies
        for model_name in model_definitions:
            self._create_placeholder_model(model_name)
        
        # Second pass: Create actual models with all fields
        for model_name, model_def in model_definitions.items():
            self._create_full_model(model_name, model_def)
            
        return self.generated_models

    def _create_placeholder_model(self, model_name: str) -> None:
        """Create a placeholder model for handling circular dependencies"""
        if model_name not in self.generated_models:
            self.generated_models[model_name] = create_model(
                model_name,
                __base__=BaseModel,
            )

    def _create_full_model(self, model_name: str, model_def: Dict[str, Any]) -> None:
        """Create a full model with all fields"""
        fields = {}
        
        for field_name, field_def in model_def["fields"].items():
            try:
                field_type = self._resolve_type(field_def["type"])
                
                # Create field with validation rules
                field_params = {
                    "description": field_def["description"],
                    **field_def.get("validation_rules", {})
                }
                
                # Handle required/optional fields
                if not field_def.get("required", True):
                    field_type = Optional[field_type]
                
                fields[field_name] = (field_type, Field(**field_params))
                
            except Exception as e:
                print(f"Error processing field {field_name}: {str(e)}")
                # Fallback to Any type
                fields[field_name] = (Any, Field(description=field_def["description"]))
        
        # Update the model with full field definitions
        model = create_model(
            model_name,
            __base__=BaseModel,
            __doc__=model_def["description"],
            **fields
        )
        
        self.generated_models[model_name] = model

    def _resolve_type(self, type_str: str) -> Any:
        """Resolve type string to actual type"""
        # Handle List types
        if type_str.startswith(("List[", "list[")):
            inner_type = type_str[5:-1].strip()
            resolved_inner = self._resolve_type(inner_type)
            return List[resolved_inner]
        
        # Handle Optional types
        if type_str.startswith(("Optional[", "optional[")):
            inner_type = type_str[9:-1].strip()
            resolved_inner = self._resolve_type(inner_type)
            return Optional[resolved_inner]
        
        # Handle basic types
        if type_str.lower() in self.type_mapping:
            return self.type_mapping[type_str.lower()]
        
        # Handle references to other models
        if type_str in self.generated_models:
            return self.generated_models[type_str]
        
        # Default to Any for unknown types
        return Any

    @lru_cache(maxsize=None)
    def get_model(self, model_name: str) -> Optional[Type[BaseModel]]:
        """Get a generated model by name"""
        return self.generated_models.get(model_name)