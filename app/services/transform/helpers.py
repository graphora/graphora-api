from app.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    DocumentKnowledgeGraph
)
from pydantic import BaseModel
from typing import Dict, Any, List
import json
import uuid
from datetime import datetime, timezone
from app.utils.logger import logger
from app.services.llm.client import LLMClient
import copy
import traceback

def transform_as_nodes(ontology: Dict[str, Any], entity_result: BaseModel) -> List[BaseNode]:
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
            raw_properties = _extract_properties(item)
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
    return nodes

def transform_as_relationships(ontology: Dict[str, Any], 
                               nodes: List[BaseNode], relationship_result: BaseModel) -> List[RelationshipInstance]:
    relationships = []
    for field_name in dir(relationship_result):
        if field_name.endswith('_list') or field_name.startswith('_') or '_' not in field_name:
            continue
        rel_list = getattr(relationship_result, field_name)
        if not isinstance(rel_list, list):
            continue
        # Flexible parsing of relationship keys
        parts = field_name.split('_')
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
        logger.info("relationships_def: ", relationships_def)
        if rel_type in relationships_def:
            target_type = target_type or relationships_def[rel_type].get('target')
        else:
            logger.warning(f"Skipping unknown relationship type: {rel_type} for {source_type}")
            continue

        if not target_type or target_type not in ontology.get('entities', {}):
            logger.warning(f"Skipping relationship {field_name}: Could not determine valid target_type")
            continue
        logger.info('rel_list: ', rel_list)
        for rel_item in rel_list:
            if not rel_item:
                continue
            source_id = getattr(rel_item, 'source_id', None)
            target_id = getattr(rel_item, 'target_id', None)
            
            if not source_id or not target_id:
                logger.warning(f"Skipping relationship {rel_type}: Missing source_id or target_id")
                continue
            
            source_node = next((n for n in nodes if n.id == source_id and n.type == source_type), None)
            target_node = next((n for n in nodes if n.id == target_id and n.type == target_type), None)
            if not source_node or not target_node:
                logger.warning(f"Skipping relationship {rel_type}: Invalid source_id {source_id} or target_id {target_id}")
                continue

            rel_properties = _extract_properties(getattr(rel_item, 'properties', {}))
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

    return relationships


async def resolve_entity_group(entity_type: str, nodes: List[BaseNode]) -> List[List[BaseNode]]:
    """
    Use LLM to identify and group matching entities that should be merged.
    Returns list of groups, where each group contains matching nodes.
    """
    if len(nodes) <= 1:
        return [[nodes[0]]] if nodes else []
    llm_client = LLMClient()

    # Convert nodes to simple dict representation for LLM
    node_dicts = []
    for node in nodes:
        node_dict = {
            "id": node.id,
            "properties": node.properties,
            "confidence": node.confidence_score if node.confidence_score else 0.3
        }
        node_dicts.append(node_dict)
        
    node_dicts_str = json.dumps(node_dicts, indent=2)

    try:
        # Call LLM for entity resolution
        print(f"Resolving `{entity_type}` entities")
        results = await llm_client.resolve_entities(
            entity_type=entity_type, node_dicts_str=node_dicts_str)
        
        if not results or len(results) == 0:
            # Fallback: treat each node as separate group
            return [[node] for node in nodes]
        
        # Convert index groups back to node groups
        resolved_groups = []
        node_map = {node.id: node for node in nodes}
        for result in results:
            if result.matching_ids is None or len(result.matching_ids) == 0:
                continue
            node_group = []
            for matching_node_id in result.matching_ids:
                if matching_node_id in node_map:
                    _node = node_map[matching_node_id]
                    _node.confidence_score = result.confidence_score
                    node_group.append(_node)
            if node_group:  # Only add non-empty groups
                resolved_groups.append(node_group)
                
                # Log explanation if available
                if result.explanation:
                    logger.info(f"Entity resolution group {node_group[0].type}: {result.explanation}")
        
        # Check if any nodes were missed (safeguard)
        included_nodes = [node for group in resolved_groups for node in group]
        
        # Add missed nodes as single-node groups
        for node in nodes:
            if node not in included_nodes:
                resolved_groups.append([node])
        
        return resolved_groups
        
    except Exception as e:
        traceback.print_exc()
        logger.warning(f"Error in LLM entity resolution: {str(e)}")
        # Fallback: treat each node as separate group
        return [[node] for node in nodes]
    

def merge_nodes(existing_node: BaseNode, new_node: BaseNode) -> BaseNode:
        """Merge properties and provenance from two nodes"""
        # Create a copy of the existing node
        merged_node = copy.deepcopy(existing_node)
        
        # Merge provenance
        if hasattr(new_node, 'provenance') and new_node.provenance:
            if not hasattr(merged_node, 'provenance') or not merged_node.provenance:
                merged_node.provenance = new_node.provenance
            else:
                # Combine chunk IDs
                merged_node.provenance.chunk_ids.extend(new_node.provenance.chunk_ids)
                merged_node.provenance.chunk_ids = list(set(merged_node.provenance.chunk_ids))
        
        # Merge properties
        if hasattr(new_node, 'properties') and new_node.properties:
            if not hasattr(merged_node, 'properties') or not merged_node.properties:
                merged_node.properties = new_node.properties
            else:
                for key, value in new_node.properties.items():
                    if value is not None:
                        # Take new value if not in existing or existing is None
                        if key not in merged_node.properties or merged_node.properties[key] is None:
                            merged_node.properties[key] = value
                        # For overlapping values, take the one with higher confidence or longer value
                        elif (hasattr(new_node, 'confidence_score') and 
                              hasattr(existing_node, 'confidence_score') and
                              new_node.confidence_score and 
                              existing_node.confidence_score and
                              new_node.confidence_score > existing_node.confidence_score):
                            merged_node.properties[key] = value
                        elif (isinstance(value, str) and
                              isinstance(merged_node.properties[key], str) and
                              len(value) > len(merged_node.properties[key])):
                            merged_node.properties[key] = value
        
        # Update confidence score to max of both
        if (hasattr(new_node, 'confidence_score') and 
            hasattr(existing_node, 'confidence_score') and
            new_node.confidence_score and 
            existing_node.confidence_score):
            merged_node.confidence_score = max(
                new_node.confidence_score,
                existing_node.confidence_score
            )
            
        return merged_node

def prune_orphaned_nodes(ontology: Dict[str, Any], graph: DocumentKnowledgeGraph) -> None:
    """Remove nodes with no ontology-defined properties that aren't referenced in any relationship"""
    # Build set of node IDs used in relationships (both source and target)
    nodes_in_relationships = set()
    
    # Check relationships list for DocumentKnowledgeGraph
    if graph.relationships:
        for rel in graph.relationships:
            nodes_in_relationships.add(rel.source_id)
            nodes_in_relationships.add(rel.target_id)
    else:
        # Check all relationship fields for KnowledgeGraph
        for field_name in dir(graph):
            if field_name.endswith('_list') or not '_' in field_name or field_name.startswith('_'):
                continue
            rel_list = getattr(graph, field_name)
            if isinstance(rel_list, list):
                for rel in rel_list:
                    nodes_in_relationships.add(rel.source_id)
                    nodes_in_relationships.add(rel.target_id)

    # Get all nodes
    all_nodes = graph.nodes if hasattr(graph, 'nodes') else []
    if not all_nodes:
        for field_name in dir(graph):
            if not field_name.endswith('_list') or field_name.startswith('_'):
                continue
            node_list = getattr(graph, field_name)
            if isinstance(node_list, list):
                all_nodes.extend(node_list)

    # Find orphaned nodes and try to link them
    orphaned_nodes = []
    for node in all_nodes:
        if node.id not in nodes_in_relationships:
            # Check if this node type has only one possible relationship type in ontology
            node_type = node.type
            possible_rels = _get_possible_relationships_for_type(ontology, node_type)
            
            if len(possible_rels) == 1:
                # Get the single relationship type and direction
                rel_type = list(possible_rels.keys())[0]
                rel_info = possible_rels[rel_type]
                
                # Find potential nodes to link with
                potential_nodes = []
                for other_node in all_nodes:
                    if other_node.id != node.id:
                        # Check both source->target and target->source possibilities
                        if (rel_info['source'] == node_type and rel_info['target'] == other_node.type) or \
                            (rel_info['target'] == node_type and rel_info['source'] == other_node.type):
                            potential_nodes.append(other_node)
                
                # If we found exactly one potential node, create the relationship
                if len(potential_nodes) == 1:
                    other_node = potential_nodes[0]
                    # Determine direction based on ontology
                    if rel_info['source'] == node_type:
                        source_id, target_id = node.id, other_node.id
                        source_type, target_type = node_type, other_node.type
                    else:
                        source_id, target_id = other_node.id, node.id
                        source_type, target_type = other_node.type, node_type
                        
                    # Create relationship
                    rel_id = f"{source_id}_{rel_type}_{target_id}"
                    new_rel = RelationshipInstance(
                        id=rel_id,
                        type=rel_type,
                        source_id=source_id,
                        target_id=target_id,
                        source_type=source_type,
                        target_type=target_type,
                        properties={},
                        provenance=NodeProvenance(
                            chunk_ids=[],
                            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                            confidence_score=0.9
                        )
                    )
                    
                    # Add to relationships
                    if graph.relationships:
                        graph.relationships.append(new_rel)
                    else:
                        rel_field = f"{source_type.lower()}_{rel_type.lower()}_{target_type.lower()}"
                        if not hasattr(graph, rel_field):
                            setattr(graph, rel_field, [])
                        getattr(graph, rel_field).append(new_rel)
                        
                    # Node is no longer orphaned since it's now in a relationship
                    continue
            
            # If we couldn't auto-link, add to orphaned list
            orphaned_nodes.append(node)

    # Remove orphaned nodes that couldn't be linked
    if graph.nodes:
        graph.nodes = [n for n in graph.nodes if n not in orphaned_nodes]
    else:
        for field_name in dir(graph):
            if not field_name.endswith('_list') or field_name.startswith('_'):
                continue
            node_list = getattr(graph, field_name)
            if isinstance(node_list, list):
                setattr(graph, field_name, [n for n in node_list if n not in orphaned_nodes])

def _get_possible_relationships_for_type(ontology: Dict[str, Any], node_type: str) -> Dict[str, Dict[str, str]]:
    """Get all possible relationship types for a given node type from ontology"""
    possible_rels = {}
        
    # Get relationships where this type is either source or target
    entity_def = ontology.get('entities', {}).get(node_type, {})
    if not entity_def:
        return {}
    
    relationships = entity_def.get('relationships', {})
    for rel_type, rel_info in relationships.items():
        source_type = node_type
        target_type = rel_info.get('target')
        
        if source_type == node_type or target_type == node_type:
            possible_rels[rel_type] = {
                'source': source_type,
                'target': target_type
            }
    
    return possible_rels

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

def _extract_properties(item: BaseModel) -> Dict[str, Any]:
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