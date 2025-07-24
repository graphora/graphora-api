"""Utilities for converting between snake_case and camelCase."""

import re
from typing import Dict, Any, List, Union


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case string to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(word.capitalize() for word in components[1:])


def camel_to_snake(camel_str: str) -> str:  
    """Convert camelCase string to snake_case."""
    return re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)


def convert_dict_to_camel(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert all keys in a dictionary from snake_case to camelCase."""
    if not isinstance(data, dict):
        return data
        
    converted = {}
    for key, value in data.items():
        # Convert the key to camelCase
        camel_key = snake_to_camel(key)
        
        # Recursively convert nested dictionaries
        if isinstance(value, dict):
            converted[camel_key] = convert_dict_to_camel(value)
        elif isinstance(value, list):
            converted[camel_key] = [
                convert_dict_to_camel(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            converted[camel_key] = value
            
    return converted


def convert_dict_to_snake(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert all keys in a dictionary from camelCase to snake_case."""
    if not isinstance(data, dict):
        return data
        
    converted = {}
    for key, value in data.items():
        # Convert the key to snake_case
        snake_key = camel_to_snake(key).lower()
        
        # Recursively convert nested dictionaries
        if isinstance(value, dict):
            converted[snake_key] = convert_dict_to_snake(value)
        elif isinstance(value, list):
            converted[snake_key] = [
                convert_dict_to_snake(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            converted[snake_key] = value
            
    return converted


def convert_quality_response_to_camel(quality_results: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a quality validation response to camelCase for API consumption."""
    return convert_dict_to_camel(quality_results)