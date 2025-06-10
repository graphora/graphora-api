from app.schemas.graph import Edge, Node
from app.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
    DocumentKnowledgeGraph
)
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Optional
import json
import uuid
from datetime import datetime, timezone
from app.utils.logger import logger
from app.services.llm.client import LLMClient
from app.utils.constants import SYSTEM_PROPERTIES
from splink import block_on
import splink.comparison_library as cl
import pandas as pd
from splink import DuckDBAPI, Linker, SettingsCreator
import logging
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
    print("#"*30)
    print("relationship_result", relationship_result)
    print("nodes", nodes)
    print("#"*30)
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


async def deduplicate_entities_with_splink(
    entities: List[BaseNode | Node], 
    relationships: List[RelationshipInstance | Edge] = None, 
    entity_type: str = None, threshold: float = 0.95, 
    parsed_ontology: Dict[str, Any] = None) -> Tuple[List[BaseNode | Node], List[RelationshipInstance | Edge]]:
    """
    Deduplicate entities using the Splink library.
    
    Args:
        entities (List[BaseNode]): List of entity nodes to deduplicate
        relationships (List[RelationshipInstance], optional): List of relationships to provide context
        entity_type (str, optional): Type of entity to filter by. If None, all entity types are processed separately.
        threshold (float, optional): Match probability threshold for clustering. Default is 0.95.
        parsed_ontology (dict, optional): Parsed ontology to get property types
        
    Returns:
        Tuple[List[BaseNode], List[RelationshipInstance]]: Deduplicated list of entities and updated relationships
    """
    try:
        logging.info(f"Starting entity deduplication with {len(entities)} entities")
        print(f"Starting entity deduplication with {len(entities)} entities")
        if relationships:
            logging.info(f"Using {len(relationships)} relationships for context")
        
        # If no entities to process, return original list
        if not entities:
            return entities, relationships
            
        # Group entities by type
        entities_by_type = {}
        for e in entities:
            if not hasattr(e, 'type'):
                continue
                
            if e.type not in entities_by_type:
                entities_by_type[e.type] = []
            entities_by_type[e.type].append(e)
        
        logging.info(f"Grouped entities into {len(entities_by_type)} types: {list(entities_by_type.keys())}")
        
        # If specific entity type is requested, only process that type
        if entity_type and entity_type in entities_by_type:
            types_to_process = [entity_type]
        else:
            types_to_process = list(entities_by_type.keys())
        
        # Track which nodes were deduplicated (original ID -> representative ID)
        all_node_mappings = {}
        
        # Process each entity type separately
        all_deduplicated_entities = []
        for current_type in types_to_process:
            type_entities = entities_by_type[current_type]
            
            # Skip if too few entities of this type
            if len(type_entities) < 3:
                logging.info(f"Skipping deduplication for type '{current_type}' - only {len(type_entities)} entities")
                all_deduplicated_entities.extend(type_entities)
                continue
                
            logging.info(f"Processing {len(type_entities)} entities of type '{current_type}'")
            
            # Prepare entities data for this type
            entities_data = _prepare_entities_for_deduplication(type_entities, relationships, parsed_ontology)
            
            # Create DataFrame and prepare for Splink processing
            df, comparison_columns = _create_splink_dataframe(entities_data, SYSTEM_PROPERTIES)
            
            if not comparison_columns:
                logging.warning(f"No comparison columns found for type '{current_type}'")
                all_deduplicated_entities.extend(type_entities)
                continue
            
            # Create comparisons and blocking rules
            comparisons = _create_splink_comparisons(comparison_columns, df, current_type, parsed_ontology)
            
            if not comparisons:
                logging.warning(f"No comparison rules could be created for type '{current_type}'")
                all_deduplicated_entities.extend(type_entities)
                continue
            
            # Create blocking rules
            blocking_rules = _create_blocking_rules(comparison_columns, df, current_type, parsed_ontology)
            
            if not blocking_rules:
                logging.warning(f"No blocking rules could be created for type '{current_type}'")
                all_deduplicated_entities.extend(type_entities)
                continue
            
            # Run Splink deduplication
            logging.info(f"Running Splink deduplication for type '{current_type}'...")
            print(f"Running Splink deduplication for type '{current_type}'...")
            id_to_representative = _run_splink_deduplication(df, comparisons, blocking_rules, threshold)
            
            if not id_to_representative:
                logging.info(f"No duplicates found for type '{current_type}'")
                all_deduplicated_entities.extend(type_entities)
                continue
                
            # Log the resolved nodes (duplicates found)
            resolved_nodes = {}
            for entity_id, rep_id in id_to_representative.items():
                if entity_id != rep_id:  # Only log actual duplicates
                    if rep_id not in resolved_nodes:
                        resolved_nodes[rep_id] = []
                    resolved_nodes[rep_id].append(entity_id)
                    
                    # Add to the global mapping
                    all_node_mappings[entity_id] = rep_id
            
            # Log the resolved nodes
            if resolved_nodes:
                logging.info(f"Resolved nodes for type '{current_type}':")
                for rep_id, duplicate_ids in resolved_nodes.items():
                    print(f"  Representative {rep_id} <- duplicates: {duplicate_ids}")
                    logging.info(f"  Representative {rep_id} <- duplicates: {duplicate_ids}")
            
            logging.info(f"Found {len(set(id_to_representative.values()))} clusters for type '{current_type}'")
            
            # Create deduplicated entity list for this type
            type_deduplicated_entities = _create_deduplicated_entities(type_entities, id_to_representative)
            all_deduplicated_entities.extend(type_deduplicated_entities)
            
            logging.info(f"Reduced {len(type_entities)} entities to {len(type_deduplicated_entities)} for type '{current_type}'")
            print(f"Reduced {len(type_entities)} entities to {len(type_deduplicated_entities)} for type '{current_type}'")
        
        # Update relationships to use the representative node IDs
        updated_relationships = []
        if relationships and all_node_mappings:
            logging.info(f"Updating {len(relationships)} relationships with {len(all_node_mappings)} node mappings")
            
            for rel in relationships:
                source_id = rel.source_id
                target_id = rel.target_id
                
                # Check if source or target nodes were deduplicated
                source_changed = source_id in all_node_mappings
                target_changed = target_id in all_node_mappings
                
                if source_changed or target_changed:
                    # Create a new relationship with updated IDs
                    new_source_id = all_node_mappings.get(source_id, source_id)
                    new_target_id = all_node_mappings.get(target_id, target_id)
                    
                    # Skip self-relationships that might be created by deduplication
                    if new_source_id == new_target_id:
                        logging.info(f"Skipping self-relationship: {rel.type} from {source_id} to {target_id}")
                        continue
                    
                    # Create a new relationship with the updated IDs
                    new_rel = rel.copy()
                    new_rel.source_id = new_source_id
                    new_rel.target_id = new_target_id
                    
                    logging.debug(f"Updated relationship: {rel.type} from {source_id}->{new_source_id} to {target_id}->{new_target_id}")
                    updated_relationships.append(new_rel)
                else:
                    # Keep the original relationship
                    updated_relationships.append(rel)
        else:
            updated_relationships = relationships
        
        # Log summary of deduplication
        if all_node_mappings:
            logging.info(f"Deduplication summary: {len(all_node_mappings)} nodes mapped to representatives")
            logging.info(f"Completed deduplication: {len(entities)} entities reduced to {len(all_deduplicated_entities)}")
            logging.info(f"Relationships: {len(relationships)} original, {len(updated_relationships)} after updating")
        else:
            logging.info("No duplicates found across any entity types")
            
        return all_deduplicated_entities, updated_relationships
        
    except Exception as e:
        logging.error(f"Error in entity deduplication: {str(e)}")
        logging.debug(f"Deduplication error details: {traceback.format_exc()}")
        traceback.print_exc()
        return entities, relationships

async def resolve_entity_group(
    entity_type: str, 
    nodes: List[BaseNode], 
    user_id: Optional[str] = None, 
    transform_id: Optional[str] = None, 
    document_usage_id: Optional[str] = None
) -> List[List[BaseNode]]:
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
            entity_type=entity_type, 
            node_dicts_str=node_dicts_str,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id
        )
        
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
        print("resolved_groups", resolved_groups)
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
    
def _prepare_entities_for_deduplication(entities, relationships=None, parsed_ontology=None):
    """
    Prepare entities for deduplication by converting to dictionaries and 
    enriching with 1-degree related entities where possible.
    
    Args:
        entities (List[BaseNode]): List of entity nodes
        relationships (List[RelationshipInstance], optional): List of relationships
        parsed_ontology (dict, optional): Parsed ontology to get property types
        
    Returns:
        List[Dict]: Prepared entity data for deduplication
    """
    # Convert entities to dictionaries
    entities_data = [entity.dict() for entity in entities]
    
    # Create entity ID to index mapping for quick lookup
    entity_id_to_index = {entity.id: i for i, entity in enumerate(entities)}
    
    # If relationships are provided, enrich entities with 1-degree connections
    if relationships:
        # Create a graph representation for quick neighbor lookup
        neighbors = {}
        for rel in relationships:
            source_id = rel.source_id
            target_id = rel.target_id
            rel_type = rel.type
            
            if source_id not in neighbors:
                neighbors[source_id] = []
            if target_id not in neighbors:
                neighbors[target_id] = []
                
            neighbors[source_id].append({
                'id': target_id,
                'type': rel_type,
                'direction': 'outgoing'
            })
            neighbors[target_id].append({
                'id': source_id,
                'type': rel_type,
                'direction': 'incoming'
            })
        
        # Enrich entities with neighbor information
        for i, entity_data in enumerate(entities_data):
            entity_id = entity_data['id']
            if entity_id in neighbors:
                # Instead of creating complex nested structures, create flattened properties
                # with neighbor information that's easier for Splink to process
                neighbor_count = 0
                
                for neighbor in neighbors[entity_id]:
                    neighbor_id = neighbor['id']
                    if neighbor_id in entity_id_to_index:
                        try:
                            # Convert complex values to strings
                            neighbor_entity = entities[entity_id_to_index[neighbor_id]]
                            neighbor_data = neighbor_entity.dict()
                            
                            # Add basic neighbor info with a prefix and index to keep them separate
                            prefix = f"neighbor_{neighbor_count}_"
                            
                            # Add basic neighbor metadata
                            if 'properties' in entity_data and isinstance(entity_data['properties'], dict):
                                entity_data['properties'][f"{prefix}type"] = neighbor_data.get('type', '')
                                entity_data['properties'][f"{prefix}rel_type"] = neighbor['type']
                                entity_data['properties'][f"{prefix}direction"] = neighbor['direction']
                                
                                # Add selected non-system properties from neighbor
                                if 'properties' in neighbor_data and isinstance(neighbor_data['properties'], dict):
                                    non_system_properties = {k: v for k, v in neighbor_data['properties'].items() if k not in SYSTEM_PROPERTIES}
                                    for prop_name in non_system_properties:
                                        prop_value = non_system_properties[prop_name]
                                        entity_data['properties'][f"{prefix}{prop_name}"] = prop_value
                            
                            neighbor_count += 1
                            # Limit the number of neighbors to avoid explosion of properties
                            if neighbor_count >= 3:  # Only include up to 3 neighbors
                                break
                        except Exception as e:
                            logging.warning(f"Error processing neighbor {neighbor_id} for entity {entity_id}: {str(e)}")
    
    # Add property types from ontology
    if parsed_ontology:
        for entity_data in entities_data:
            entity_type = entity_data.get('type')
            if entity_type and entity_type in parsed_ontology.get('entities', {}):
                properties = entity_data.get('properties', {})
                # Create a copy of the properties to avoid modifying during iteration
                property_items = list(properties.items())
                
                for prop_name, prop_value in property_items:
                    if prop_name in parsed_ontology.get('entities', {}).get(entity_type, {}).get('properties', {}):
                        prop_type = parsed_ontology.get('entities', {}).get(entity_type, {}).get('properties', {}).get(prop_name, {}).get('type')
                        if prop_type:
                            entity_data['properties'][f"{prop_name}_type"] = prop_type
    
    return entities_data

def _create_splink_dataframe(entities_data, system_properties):
    """
    Create a DataFrame for Splink processing from entity data.
    
    Args:
        entities_data (List[Dict]): List of entity dictionaries
        system_properties (List[str]): List of system properties to exclude
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame and list of comparison columns
    """
    
    # First create a basic DataFrame with just the IDs
    df = pd.DataFrame(entities_data)
    
    # Get property columns for comparison (focus on properties dictionary)
    properties_columns = []
    for entity in entities_data:
        if 'properties' in entity and isinstance(entity['properties'], dict):
            for prop in entity['properties'].keys():
                if prop not in system_properties:
                    properties_columns.append(prop)
    
    # Get unique property columns
    properties_columns = list(set(properties_columns))
    
    # Pre-initialize all property columns with None to avoid setting with iterables
    for col in properties_columns:
        if col not in df.columns:
            df[col] = None
    
    # Flatten properties for comparison, excluding system properties
    for i, entity in enumerate(entities_data):
        if 'properties' in entity and isinstance(entity['properties'], dict):
            for prop, value in entity['properties'].items():
                if prop not in system_properties and prop in properties_columns:
                    # Handle potentially problematic values (lists, dicts, etc.)
                    if isinstance(value, (list, dict, tuple)):
                        try:
                            # Convert complex values to strings
                            df.at[i, prop] = str(value)
                        except Exception as e:
                            logging.warning(f"Could not set property {prop} with value {value}: {str(e)}")
                    else:
                        # For simple values, set directly but ensure they're strings for comparison
                        df.at[i, prop] = str(value) if value is not None else None
    
    return df, properties_columns

def _create_splink_comparisons(properties_columns, df, entity_type, parsed_ontology=None):
    """
    Create appropriate Splink comparisons based on property columns.
    
    Args:
        properties_columns (List[str]): List of property column names
        df (pd.DataFrame): DataFrame with entity data
        entity_type (str): Type of entity
        parsed_ontology (dict, optional): Parsed ontology to get property types
        
    Returns:
        List: List of Splink comparison objects
    """
    
    # Focus on the most important properties for comparison
    # Prioritize name, type, and description properties
    key_properties = []
    neighbour_properties = []
    other_properties = []
    
    # Use ontology information to determine property types and importance
    property_types = {}
    if parsed_ontology:
        for col in properties_columns:
            if col not in df.columns:
                continue
                
            # Get property type from ontology
            prop_type = parsed_ontology.get('entities', {}).get(entity_type, {}).get('properties', {}).get(col, {}).get('type')
            if prop_type:
                property_types[col] = prop_type
                
                # Categorize by importance based on ontology type
                if _is_prop_type_string(prop_type) or any(name_part in col.lower() for name_part in ['name', 'title']):
                    key_properties.append(col)
                elif _is_prop_type_number(prop_type) and not col.startswith('neighbor_'):
                    other_properties.append(col)
                elif not col.startswith('neighbor_'):
                    neighbour_properties.append(col)
    else:
        # Fallback to the original categorization if ontology info not available
        for col in properties_columns:
            if col not in df.columns:
                continue
                
            # Categorize properties by importance
            if col in ['type', 'id']:
                key_properties.append(col)
            elif not col.startswith('neighbor_'):  # Exclude neighbor properties to simplify
                neighbour_properties.append(col)
            else:
                other_properties.append(col)
            
    # Log the property categorization
    logging.info(f"Key properties for comparison: {key_properties}")
    logging.info(f"Other properties for comparison: {other_properties}")
    logging.info(f"Neighbour properties for comparison: {neighbour_properties}")
    
    # Create comparisons, limiting the total number
    comparisons = []
    
    # Add key properties with appropriate comparisons based on ontology types
    for col in key_properties[:3]:  # Limit to 3 key properties
        if col == 'type':
            comparisons.append(cl.ExactMatch(col))
        elif col in property_types:
            prop_type = property_types[col]
            if _is_prop_type_string(prop_type):
                comparisons.append(cl.JaroWinklerAtThresholds(col, [0.9, 0.7]))
            elif _is_prop_type_number(prop_type):
                comparisons.append(cl.ExactMatch(col))
        else:
            # Fallback to type checking if ontology info not available
            if df[col].dtype == 'object' and all(isinstance(x, str) or x is None for x in df[col].dropna()):
                comparisons.append(cl.JaroWinklerAtThresholds(col, [0.9, 0.7]))
            else:
                comparisons.append(cl.ExactMatch(col))
    
    # Add other properties with appropriate comparisons
    for col in other_properties[:2]:  # Limit to 2 other properties
        if col in property_types:
            prop_type = property_types[col]
            if _is_prop_type_string(prop_type):
                comparisons.append(cl.LevenshteinAtThresholds(col, [0.9, 0.7]))
            elif _is_prop_type_number(prop_type):
                comparisons.append(cl.ExactMatch(col))
        else:
            # Fallback to type checking
            if df[col].dtype == 'object' and all(isinstance(x, str) or x is None for x in df[col].dropna()):
                comparisons.append(cl.LevenshteinAtThresholds(col, [0.9, 0.7]))
            else:
                comparisons.append(cl.ExactMatch(col))
    
    # Add a few neighbour properties if we don't have enough comparisons yet
    if len(comparisons) < 3 and neighbour_properties:
        for col in neighbour_properties[:2]:  # Limit to 2 neighbour properties
            if col in property_types:
                prop_type = property_types[col]
                if _is_prop_type_string(prop_type):
                    comparisons.append(cl.LevenshteinAtThresholds(col, [0.9, 0.7]))
                elif _is_prop_type_number(prop_type):
                    comparisons.append(cl.ExactMatch(col))
            else:
                # Fallback to type checking
                if df[col].dtype == 'object' and all(isinstance(x, str) or x is None for x in df[col].dropna()):
                    comparisons.append(cl.LevenshteinAtThresholds(col, [0.9, 0.7]))
                else:
                    comparisons.append(cl.ExactMatch(col))
    
    logging.info(f"Created {len(comparisons)} comparisons for Splink")
    return comparisons

def _is_prop_type_string(prop_type: str) -> bool:
    return prop_type.lower() in ['str', 'string']

def _is_prop_type_number(prop_type: str) -> bool:
    return prop_type.lower() in ['int', 'double', 'integer', 'float', 'number']

def _is_prop_type_datetime(prop_type: str) -> bool:
    return prop_type.lower() in ['date', 'datetime', 'timestamp']

def _create_blocking_rules(properties_columns, df, entity_type=None, parsed_ontology=None):
    """
    Create blocking rules for Splink based on property columns.
    
    Args:
        properties_columns (List[str]): List of property column names
        df (pd.DataFrame): DataFrame with entity data
        entity_type (str, optional): Type of entity
        parsed_ontology (dict, optional): Parsed ontology to get property types
        
    Returns:
        List: List of Splink blocking rules
    """
    
    # Prioritize certain columns for blocking
    name_columns = [
        col for col in properties_columns 
        if any(name_part in col.lower() for name_part in ['name', 'title'])
    ]
    
    type_columns = [
        col for col in properties_columns 
        if col == 'type'
    ]
    
    id_columns = [
        col for col in properties_columns 
        if col.endswith('_id') and col != 'id'  # Skip primary key
    ]
    
    # Use ontology information to determine property types for blocking
    key_columns = []
    if parsed_ontology and entity_type:
        for col in properties_columns:
            prop_type = parsed_ontology.get('entities', {}).get(entity_type, {}).get('properties', {}).get(col, {}).get('type')
            if prop_type:
                # Use string properties for blocking
                if prop_type == 'string' and any(name_part in col.lower() for name_part in ['name', 'title', 'identifier']):
                    key_columns.append(col)
                # Use exact-match integer properties for blocking
                elif prop_type in ['integer', 'number'] and any(id_part in col.lower() for id_part in ['id', 'code']):
                    key_columns.append(col)
    
    # Create blocking rules
    blocking_rules = []
    
    # Add type blocking rule if available
    if type_columns:
        blocking_rules.append(block_on(type_columns[0]))
    
    # Add name blocking rules
    for col in name_columns[:1]:  # Limit to 1 name column
        blocking_rules.append(block_on(col))
    
    # Add key columns from ontology
    for col in key_columns[:2]:  # Limit to 2 key columns
        if col not in [rule.blocking_rule_sql for rule in blocking_rules]:
            blocking_rules.append(block_on(col))
    
    # Add ID blocking rules if we don't have enough yet
    if len(blocking_rules) < 2 and id_columns:
        for col in id_columns[:1]:  # Limit to 1 ID column
            blocking_rules.append(block_on(col))
    
    # Add a fallback blocking rule if we don't have any yet
    if not blocking_rules and properties_columns:
        for col in properties_columns:
            if df[col].nunique() > 1:  # Only use columns with multiple values
                blocking_rules.append(block_on(col))
                break
    
    logging.info(f"Created {len(blocking_rules)} blocking rules for Splink")
    return blocking_rules

def _run_splink_deduplication(df, comparisons, blocking_rules, threshold):
    """
    Run Splink deduplication on the prepared DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with entity data
        comparisons (List): List of Splink comparison objects
        blocking_rules (List): List of Splink blocking rules
        threshold (float): Match probability threshold
        
    Returns:
        Dict: Mapping of entity IDs to representative entity IDs
    """
    
    # Ensure 'id' column exists
    if 'id' not in df.columns:
        logging.error("DataFrame must have an 'id' column for deduplication")
        return {}
    
    # If we have too few records or comparisons, skip deduplication
    if len(df) < 5 or not comparisons or not blocking_rules:
        logging.warning(f"Insufficient data for deduplication: {len(df)} records, {len(comparisons)} comparisons, {len(blocking_rules)} blocking rules")
        return {}
    
    # Initialize DuckDB API
    db_api = DuckDBAPI()
    
    try:
        # Create settings with the correct unique_id_column_name and simplified parameters
        settings = SettingsCreator(
            link_type="dedupe_only",
            comparisons=comparisons,
            blocking_rules_to_generate_predictions=blocking_rules,
            unique_id_column_name="id",  # Specify the ID column name
            # Use default values for parameters that might be hard to estimate
            probability_two_random_records_match=0.001,  # Higher than default for better recall
            em_convergence=0.01,  # More lenient convergence to avoid too many iterations
            max_iterations=5  # Limit EM iterations to avoid excessive processing
        )
        
        # Create linker
        linker = Linker(df, settings, db_api)
        
        # Simplified training approach
        logging.info("Estimating parameters for deduplication...")
        
        # Skip complex parameter estimation if we have limited data
        if len(df) < 20:
            logging.info("Limited data available, using default parameters")
        else:
            try:
                # Estimate u parameters with a smaller sample to speed things up
                linker.training.estimate_u_using_random_sampling(max_pairs=1e5)
                
                # Use the first blocking rule for EM estimation with fewer iterations
                if blocking_rules:
                    linker.training.estimate_parameters_using_expectation_maximisation(
                        blocking_rules[0], max_iterations=3
                    )
            except Exception as e:
                logging.warning(f"Parameter estimation failed, using defaults: {str(e)}")
        
        # Generate predictions with a more lenient threshold for better recall
        logging.info("Generating deduplication predictions...")
        pairwise_predictions = linker.inference.predict(threshold_match_weight=-10)
        
        # Cluster predictions
        logging.info(f"Clustering predictions with threshold {threshold}...")
        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            pairwise_predictions, threshold
        )
        
        # Get the cluster results as a dataframe
        df_clusters = clusters.as_pandas_dataframe()
        
        # Check if we got any clusters
        if df_clusters.empty:
            logging.info("No duplicate clusters found")
            return {}
            
        logging.info(f"Found {len(df_clusters['cluster_id'].unique())} clusters")
        
        # Create a mapping of entity IDs to their representative entity IDs
        id_to_representative = {}
        
        # Check if expected columns exist
        id_col = "id"  # This should match unique_id_column_name in settings
        if id_col not in df_clusters.columns or 'cluster_id' not in df_clusters.columns:
            logging.warning(f"Missing expected columns in cluster dataframe. Available columns: {df_clusters.columns.tolist()}")
            return {}
        
        # Create a mapping of cluster IDs to representative entity IDs
        cluster_to_representative = {}
        for cluster_id in df_clusters['cluster_id'].unique():
            cluster_members = df_clusters[df_clusters['cluster_id'] == cluster_id][id_col].tolist()
            if cluster_members:
                representative = cluster_members[0]  # Use first entity as representative
                cluster_to_representative[cluster_id] = representative
                
                # Log cluster information for debugging
                if len(cluster_members) > 1:
                    logging.info(f"Cluster {cluster_id}: {len(cluster_members)} members, representative: {representative}")
        
        # Map each entity to its representative entity
        for _, row in df_clusters.iterrows():
            entity_id = row[id_col]
            cluster_id = row['cluster_id']
            if cluster_id in cluster_to_representative:
                id_to_representative[entity_id] = cluster_to_representative[cluster_id]
        
        return id_to_representative
    
    except Exception as e:
        logging.error(f"Error in Splink deduplication: {str(e)}")
        logging.debug(f"Deduplication error details: {traceback.format_exc()}")
        return {}

def _create_deduplicated_entities(entities, id_to_representative):
    """
    Create a deduplicated list of entities based on the deduplication results.
    
    Args:
        entities (List[BaseNode]): Original list of entities
        id_to_representative (Dict): Mapping of entity IDs to representative entity IDs
        
    Returns:
        List[BaseNode]: Deduplicated list of entities
    """
    # Create a new deduplicated list of entities
    deduplicated_entities = []
    processed_ids = set()
    
    for entity in entities:
        entity_id = entity.id
        
        # If this entity is not in any cluster or is the representative of its cluster
        if entity_id not in id_to_representative or id_to_representative[entity_id] == entity_id:
            deduplicated_entities.append(entity)
            processed_ids.add(entity_id)
        # If this entity is in a cluster but we haven't processed the representative yet
        elif id_to_representative[entity_id] not in processed_ids:
            # Find the representative entity
            for e in entities:
                if e.id == id_to_representative[entity_id]:
                    deduplicated_entities.append(e)
                    processed_ids.add(e.id)
                    break
    
    return deduplicated_entities
