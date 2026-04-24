import yaml
from typing import Dict, Any


class OntologyValidationError(Exception):
    """Custom exception for ontology validation errors"""

    pass


def validate_ontology_schema(ontology_dict: Dict[str, Any]) -> None:
    """
    Validate the ontology schema structure.

    Args:
        ontology_dict: Dictionary containing the parsed ontology YAML

    Raises:
        OntologyValidationError: If schema validation fails
    """
    # TODO: Implement detailed schema validation
    if not isinstance(ontology_dict, dict):
        raise OntologyValidationError("Ontology must be a valid YAML mapping")

    # Basic structure validation
    required_keys = []
    missing_keys = [key for key in required_keys if key not in ontology_dict]
    if missing_keys:
        raise OntologyValidationError(
            f"Missing required keys: {', '.join(missing_keys)}"
        )


def parse_and_validate_yaml(yaml_str: str) -> Dict[str, Any]:
    """
    Parse YAML string and validate its syntax and schema.

    Args:
        yaml_str: String containing YAML content

    Returns:
        Dict containing parsed YAML

    Raises:
        yaml.YAMLError: If YAML syntax is invalid
        OntologyValidationError: If schema validation fails
    """
    try:
        ontology_dict = yaml.safe_load(yaml_str)
        validate_ontology_schema(ontology_dict)
        return ontology_dict
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax: {str(e)}")
    except OntologyValidationError as e:
        raise OntologyValidationError(str(e))
