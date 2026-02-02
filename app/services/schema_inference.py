"""Automatic schema inference from document content.

This service analyzes document text and generates an appropriate ontology schema
using LLM-based inference. It enables zero-config document processing by
auto-detecting entity types, relationships, and properties.
"""

import re
import logging
from typing import Dict, Any, List
from uuid import uuid4

import yaml
from google.genai import types

from app.utils.llm_helper import get_user_llm_credentials, create_gemini_client
from app.services.ontology_storage_service import ontology_storage_service

logger = logging.getLogger(__name__)


SCHEMA_INFERENCE_PROMPT = """Analyze the following text and extract a knowledge graph schema.

<text>
{text_sample}
</text>

Generate a YAML ontology that captures:
1. Main entity types mentioned (people, organizations, concepts, events, locations, etc.)
2. Key relationships between entities
3. Important properties for each entity type

Rules:
- Only include entity types that are clearly present in the text
- Use PascalCase for entity names (e.g., Person, Organization, Event)
- Use SCREAMING_SNAKE_CASE for relationship types (e.g., WORKS_AT, KNOWS)
- Include 2-5 properties per entity type
- Mark key identifying properties as required
- Use appropriate data types: str, int, float, bool, datetime, list

Output ONLY valid YAML in this exact format (no markdown code blocks, no explanation):

version: "0.1.0"
entities:
  EntityName:
    description: "Brief description of the entity"
    properties:
      property_name:
        type: str
        description: "Property description"
        required: true
relationships:
  RELATIONSHIP_TYPE:
    description: "Brief description"
    source: SourceEntity
    target: TargetEntity
    properties: {{}}
"""


async def infer_schema_from_text(
    text_chunks: List[str],
    user_id: str,
    max_sample_chars: int = 15000,
) -> Dict[str, Any]:
    """Infer ontology schema from document text.

    Args:
        text_chunks: List of text chunks from documents
        user_id: User ID for LLM credentials
        max_sample_chars: Maximum characters to sample for inference

    Returns:
        Parsed ontology dictionary
    """
    # Sample text from chunks, taking from multiple chunks if available
    sample = ""
    for chunk in text_chunks:
        remaining = max_sample_chars - len(sample)
        if remaining <= 0:
            break
        sample += chunk[:remaining] + "\n\n"

    if not sample.strip():
        raise ValueError("No text content available for schema inference")

    logger.info(
        f"Inferring schema from {len(sample)} characters of text for user {user_id}"
    )

    # Get user's LLM credentials
    api_key, model_name = await get_user_llm_credentials(user_id)
    client = create_gemini_client(api_key)

    # Call LLM for schema inference
    prompt = SCHEMA_INFERENCE_PROMPT.format(text_sample=sample)

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,  # Lower temperature for consistent schema
            max_output_tokens=4096,
        ),
    )

    response_text = response.text.strip()
    logger.debug(f"Schema inference response: {response_text[:500]}...")

    # Parse YAML from response
    ontology = _parse_yaml_response(response_text)

    # Validate basic structure
    if "entities" not in ontology or not ontology["entities"]:
        raise ValueError("Inferred schema has no entities")

    # Ensure version is present
    if "version" not in ontology:
        ontology["version"] = "0.1.0"

    # Ensure relationships dict exists
    if "relationships" not in ontology:
        ontology["relationships"] = {}

    logger.info(
        f"Inferred schema with {len(ontology.get('entities', {}))} entities "
        f"and {len(ontology.get('relationships', {}))} relationships"
    )

    return ontology


def _parse_yaml_response(response_text: str) -> Dict[str, Any]:
    """Parse YAML from LLM response, handling various formats.

    Args:
        response_text: Raw LLM response text

    Returns:
        Parsed YAML dictionary
    """
    # Try to extract YAML from markdown code blocks
    yaml_match = re.search(r"```(?:yaml)?\n(.*?)```", response_text, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1)
    else:
        # Assume entire response is YAML
        yaml_content = response_text

    # Clean up common issues
    yaml_content = yaml_content.strip()

    try:
        parsed = yaml.safe_load(yaml_content)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict, got {type(parsed)}")
        return parsed
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML: {e}")
        logger.debug(f"YAML content: {yaml_content}")
        raise ValueError(f"Failed to parse inferred schema: {e}")


async def create_auto_schema_ontology(
    text_chunks: List[str],
    user_id: str,
    transform_id: str,
) -> str:
    """Create a temporary ontology from inferred schema.

    Args:
        text_chunks: List of text chunks from documents
        user_id: User ID
        transform_id: Transform ID for naming

    Returns:
        Ontology ID of the created schema
    """
    # Infer schema from text
    ontology_dict = await infer_schema_from_text(text_chunks, user_id)

    # Convert to YAML
    yaml_content = yaml.dump(ontology_dict, default_flow_style=False, sort_keys=False)

    # Generate ontology ID
    ontology_id = f"auto_{uuid4().hex[:12]}"

    # Store ontology
    success = await ontology_storage_service.store_ontology(
        user_id=user_id,
        ontology_id=ontology_id,
        yaml_content=yaml_content,
        name=f"Auto-generated ({transform_id[:8]})",
        description="Automatically inferred schema from document content",
    )

    if not success:
        raise ValueError("Failed to store auto-generated ontology")

    logger.info(f"Created auto-schema ontology {ontology_id} for transform {transform_id}")

    return ontology_id


def get_default_generic_schema() -> str:
    """Get a default generic schema for basic entity extraction.

    Returns:
        YAML string with generic entity types
    """
    return """version: "0.1.0"
entities:
  Entity:
    description: "A generic entity extracted from text"
    properties:
      name:
        type: str
        description: "Name or identifier of the entity"
        required: true
      type:
        type: str
        description: "Specific type or category"
      description:
        type: str
        description: "Brief description"
  Person:
    description: "A person or individual"
    properties:
      name:
        type: str
        description: "Full name"
        required: true
      title:
        type: str
        description: "Job title or role"
      organization:
        type: str
        description: "Associated organization"
  Organization:
    description: "A company, institution, or group"
    properties:
      name:
        type: str
        description: "Organization name"
        required: true
      type:
        type: str
        description: "Organization type (company, nonprofit, government, etc.)"
  Concept:
    description: "An abstract concept, topic, or idea"
    properties:
      name:
        type: str
        description: "Concept name"
        required: true
      category:
        type: str
        description: "Category or domain"
relationships:
  RELATED_TO:
    description: "Generic relationship between entities"
    source: Entity
    target: Entity
    properties: {}
  WORKS_AT:
    description: "Employment or affiliation relationship"
    source: Person
    target: Organization
    properties: {}
  KNOWS:
    description: "Personal or professional connection"
    source: Person
    target: Person
    properties: {}
"""
