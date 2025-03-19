import pathlib
from google.genai import types

from app.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    DocumentKnowledgeGraph
)
from google.genai import client
from pydantic import BaseModel, Type
from typing import Dict, Any
import json
import uuid
from datetime import datetime, timezone
from app.services.transform.agents.ontology import OntologyParser

def _filter_properties_by_ontology(parsed_ontology, entity_type: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    entity_def = parsed_ontology.get('entities', {}).get(entity_type)
    if not entity_def:
        return {}
    defined_properties = entity_def.get('properties', {})
    if not defined_properties:
        return {}
    filtered_props = {}
    for prop_name, prop_value in properties.items():
        if prop_name in defined_properties and prop_value is not None:
            filtered_props[prop_name] = prop_value
    return filtered_props

def _is_node_valuable(parsed_ontology, entity_type: str, properties: Dict[str, Any]) -> bool:
    filtered_props = _filter_properties_by_ontology(parsed_ontology, entity_type, properties)
    return len(filtered_props) > 0

def _generate_node_key(parsed_ontology, entity_type: str, properties: Dict[str, Any]) -> str:
    entity_def = parsed_ontology.get('entities', {}).get(entity_type, {})
    unique_props = []
    for prop_name, prop_def in entity_def.get('properties', {}).items():
        if prop_def.get('unique', False) and prop_name in properties and properties[prop_name] is not None:
            value = properties[prop_name]
            if isinstance(value, str):
                value = value.lower().strip()
            unique_props.append((prop_name, value))
    if unique_props:
        sorted_props = sorted(unique_props)
        return f"{entity_type}:" + ":".join(f"{k}={v}" for k, v in sorted_props)
    else:
        non_empty_props = []
        for k, v in properties.items():
            if v is not None:
                if isinstance(v, str):
                    v = v.lower().strip()
                non_empty_props.append((k, v))
        if non_empty_props:
            sorted_props = sorted(non_empty_props)
            return f"{entity_type}:" + ":".join(f"{k}={v}" for k, v in sorted_props)
        else:
            return f"{entity_type}:uuid={uuid.uuid4()}"

def extract_properties(item):
    if item is None:
        return {}
    metadata_fields = {'model_computed_fields', 'model_config', 'model_fields', 'model_fields_set', '__fields__', '__annotations__', '__field_defaults__', '__private_attributes__'}
    try:
        if hasattr(item, 'model_dump'):
            item_dict = item.model_dump()
        else:
            item_dict = {k: v for k, v in vars(item).items() if not k.startswith('_')}
        for field in metadata_fields:
            item_dict.pop(field, None)
        item_dict.pop('type', None)
        return {k: v for k, v in item_dict.items() if v is not None}
    except Exception:
        properties = {}
        for attr_name in dir(item):
            if (attr_name.startswith('_') or callable(getattr(item, attr_name)) or attr_name in metadata_fields or attr_name == 'type'):
                continue
            try:
                value = getattr(item, attr_name)
                if value is not None:
                    properties[attr_name] = value
            except Exception:
                pass
        return properties

def extract_structured_data(file_path: str, ontology: Dict[str, Any],
                            ontology_yaml: str,
                            entities_only_model: Type[BaseModel],
                            relationships_only_model: Type[BaseModel],
                            model_id='gemini-2.0-flash-lite-001') -> DocumentKnowledgeGraph:
    filepath = pathlib.Path(file_path)
    file = types.Part.from_bytes(
        data=filepath.read_bytes(),
        mime_type='application/pdf',
    )

    # Step 1: Entity Extraction Agent
    entity_prompt = f"""
    Extract all entities from the PDF according to the ontology below.
    <ontology>
    {ontology_yaml}
    </ontology>

    Output a JSON object:
    1. "<entity_type>_list" for each entity type (e.g., "Company_list"), with properties per ontology including an optional "id" field (can be null).
    2. Metadata: "extraction_timestamp" (ISO), "tokens_used", "confidence_score" (0.0-1.0).
    Example: {{"Company_list": [{{"id": null, "name": "Apple Inc.", "cik": "0000320193"}}], "Business_list": [{{"id": null, "description": "iPhone production"}}]}}
    """
    entity_response = client.models.generate_content(
        model=model_id,
        contents=[file, entity_prompt],
        config={'response_mime_type': 'application/json', 'response_schema': entities_only_model}
    )
    logger.debug(f"Entity extraction response: {entity_response.text}")
    entity_result = entity_response.parsed

    nodes = []
    chunk_node_registry = {}

    # Process entities
    for field_name in dir(entity_result):
        if not field_name.endswith('_list') or field_name.startswith('_'):
            continue
        entity_list = getattr(entity_result, field_name)
        if not isinstance(entity_list, list):
            continue
        entity_type = field_name[:-5]
        chunk_node_registry[entity_type] = {}
        for item in entity_list:
            if not item:
                continue
            raw_properties = extract_properties(item)
            properties = _filter_properties_by_ontology(ontology, entity_type, raw_properties)
            if not _is_node_valuable(ontology, entity_type, properties):
                continue
            node_key = _generate_node_key(ontology, entity_type, properties)
            node_id = str(uuid.uuid4())
            node = BaseNode(
                id=node_id,
                type=entity_type,
                properties=properties,
                provenance=NodeProvenance(
                    chunk_ids=[node_id],
                    extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                    confidence_score=entity_result.confidence_score
                )
            )
            chunk_node_registry[entity_type][node_key] = node_id
            nodes.append(node)

    # Step 2: Relationship Extraction Agent with Node IDs
    nodes_p = json.dumps([{**n.properties, "id": n.id, "type": n.type} for n in nodes], indent=2)
    relationship_prompt = f"""
    Given the extracted entities (with their IDs) and ontology below, extract all relationships from the PDF.
    Entities:
    {nodes_p}
    <ontology>
    {ontology_yaml}
    </ontology>

    Output a JSON object:
    1. "<source>_<relationship>_<target>" for each relationship (e.g., "Company_HAS_BUSINESS_Business").
    2. Each relationship MUST include:
       - "source_id": The ID of the source node from the provided entities.
       - "target_id": The ID of the target node from the provided entities.
       - "properties": Any additional relationship properties (optional).
    3. Metadata: "extraction_timestamp" (ISO), "tokens_used", "confidence_score" (0.0-1.0).
    Example: {{"Company_HAS_BUSINESS_Business": [{{"source_id": "uuid1", "target_id": "uuid2", "properties": {{}}}}]}}
    """
    relationship_response = client.models.generate_content(
        model=model_id,
        contents=[file, relationship_prompt],
        config={'response_mime_type': 'application/json', 'response_schema': relationships_only_model}
    )
    logger.debug(f"Relationship extraction response: {relationship_response.text}")
    relationship_result = relationship_response.parsed
    print('#'*20)
    print(relationship_response.parsed)
    print('#'*20)

    relationships = []

    # Process relationships with flexible key parsing
    for field_name in dir(relationship_result):
        if field_name.endswith('_list') or field_name.startswith('_') or '_' not in field_name:
            continue
        rel_list = getattr(relationship_result, field_name)
        if not isinstance(rel_list, list):
            continue

        # Flexible parsing of relationship keys
        parts = field_name.split('_')
        print("#" * 10, parts)
        if len(parts) < 2:
            continue

        # Try to extract source_type, rel_type, and target_type
        source_type = parts[0]
        if source_type not in ontology.get('entities', {}):
            continue
        
        # Handle cases where target_type might be missing or concatenated
        rel_type_parts = parts[1:-1] if len(parts) > 2 else parts[1:]
        rel_type = '_'.join(rel_type_parts) if len(rel_type_parts) > 1 else rel_type_parts[0]
        target_type = parts[-1] if len(parts) > 2 else None

        # Infer target_type from ontology if not in field_name
        relationships_def = ontology['entities'].get(source_type, {}).get('relationships', {})
        print("relationships_def: ", relationships_def)
        if rel_type in relationships_def:
            target_type = target_type or relationships_def[rel_type].get('target')
        else:
            logger.warning(f"Skipping unknown relationship type: {rel_type} for {source_type}")
            print(f"Skipping unknown relationship type: {rel_type} for {source_type}")
            continue

        if not target_type or target_type not in ontology.get('entities', {}):
            logger.warning(f"Skipping relationship {field_name}: Could not determine valid target_type")
            print(f"Skipping relationship {field_name}: Could not determine valid target_type")
            continue
        print('rel_list: ', rel_list)
        for rel_item in rel_list:
            if not rel_item:
                continue
            source_id = getattr(rel_item, 'source_id', None)
            target_id = getattr(rel_item, 'target_id', None)
            
            if not source_id or not target_id:
                logger.warning(f"Skipping relationship {rel_type}: Missing source_id or target_id")
                print(f"Skipping relationship {rel_type}: Missing source_id or target_id")
                continue
            
            source_node = next((n for n in nodes if n.id == source_id and n.type == source_type), None)
            target_node = next((n for n in nodes if n.id == target_id and n.type == target_type), None)
            if not source_node or not target_node:
                logger.warning(f"Skipping relationship {rel_type}: Invalid source_id {source_id} or target_id {target_id}")
                print(f"Skipping relationship {rel_type}: Invalid source_id {source_id} or target_id {target_id}")
                continue

            rel_properties = extract_properties(getattr(rel_item, 'properties', {}))
            rel_id = str(uuid.uuid4())
            rel = RelationshipInstance(
                id=rel_id,
                type=rel_type,
                source_id=source_id,
                target_id=target_id,
                source_type=source_type,
                target_type=target_type,
                properties=rel_properties,
                provenance=NodeProvenance(
                    chunk_ids=[rel_id],
                    extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                    confidence_score=relationship_result.confidence_score
                )
            )
            relationships.append(rel)

    logger.info(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
    print(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
    return DocumentKnowledgeGraph(nodes=nodes, relationships=relationships)