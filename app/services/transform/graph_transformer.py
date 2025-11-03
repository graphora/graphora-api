from typing import List, Callable, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from app.services.transform.models import DocumentKnowledgeGraph
from app.services.transform.ontology_helper import OntologyParser
from app.services.transform.helpers import (
    transform_as_nodes,
    resolve_entity_group,
    merge_nodes,
    transform_as_relationships,
    prune_orphaned_nodes,
    deduplicate_entities_with_splink,
)
from app.services.llm.client import LLMClient
from app.services.transform.models import BaseNode, RelationshipInstance
from app.utils.logger import logger
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "true"


async def build_graph_from_chunks(
    ontology_parser: OntologyParser,
    chunks: List[str],
    transform_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    llm_client = LLMClient()
    return await _build_graph_from(
        ontology_parser,
        chunks,
        transform_id,
        llm_client.extract_nodes_from_chunk,
        llm_client.extract_relationships_from_chunk,
        progress_callback,
        user_id,
        document_usage_id,
    )


async def build_graph_from_pdfs(
    ontology_parser: OntologyParser,
    pdf_paths: List[str],
    transform_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    llm_client = LLMClient()
    return await _build_graph_from(
        ontology_parser,
        pdf_paths,
        transform_id,
        llm_client.extract_nodes_from_pdf,
        llm_client.extract_relationships_from_pdf,
        progress_callback,
        user_id,
        document_usage_id,
    )


async def _build_graph_from(
    ontology_parser: OntologyParser,
    chunks_or_pdf_paths: List[str],
    transform_id: str,
    node_extractor: Callable[[str, BaseModel, str], BaseModel],
    relationship_extractor: Callable[[str, BaseModel, str], BaseModel],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    user_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> DocumentKnowledgeGraph:
    nodes_only_ontology = ontology_parser.build_entities_only_model()
    context = "None"
    nodes = []
    # Step 1: LLM based Entity Extraction for each chunk based on ontology. Pass previous nodes and current chunk to LLM.
    for _chunk in chunks_or_pdf_paths:
        nodes_only_kg = await node_extractor(
            _chunk,
            response_model=nodes_only_ontology,
            context=context,
            ontology_yaml=ontology_parser.ontology_yaml,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )
        base_nodes = transform_as_nodes(
            ontology_parser.parsed_ontology,
            nodes_only_kg,
            transform_id=transform_id,
        )
        for new_node in base_nodes:
            is_duplicate = any(
                _is_duplicate_node(existing_node, new_node) for existing_node in nodes
            )
            if not is_duplicate:
                nodes.append(new_node)
        context = await _build_nodes_context(nodes)

    # Step 2: Compare & Merge entities if they are the same.
    nodes = await _compare_and_merge_nodes(
        nodes,
        user_id=user_id,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
    )
    nodes, _ = await deduplicate_entities_with_splink(
        nodes, None, parsed_ontology=ontology_parser.parsed_ontology
    )
    logger.info(f"Nodes after comparison: {nodes}")

    # Step 3: LLM based Relationship Inference for each chunk. Pass all relevant nodes & relationships, current chunk to LLM.
    relationships_only_ontology = ontology_parser.build_relationships_only_model()
    relationships = []
    for _chunk in chunks_or_pdf_paths:
        relationships_only_kg = await relationship_extractor(
            _chunk,
            response_model=relationships_only_ontology,
            context=context,
            ontology_yaml=ontology_parser.ontology_yaml,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )
        base_relationships = transform_as_relationships(
            ontology_parser.parsed_ontology, nodes, relationships_only_kg
        )
        for new_relationship in base_relationships:
            is_duplicate = any(
                _is_duplicate_relationship(existing_relationship, new_relationship)
                for existing_relationship in relationships
            )
            if not is_duplicate:
                relationships.append(new_relationship)
        context = await _build_relationships_context(nodes, relationships)

    # Step 4: Compare & Merge relationships if they are the same.
    relationships = _compare_and_merge_relationships(relationships)

    # Step 5: Splink based Entity deduplication within an entity group using entities & 1 degree related entities
    nodes, relationships = await deduplicate_entities_with_splink(
        entities=nodes,
        relationships=relationships,
        parsed_ontology=ontology_parser.parsed_ontology,
    )

    # Step 6: Build graph from nodes and relationships.
    kg = DocumentKnowledgeGraph(nodes=nodes, relationships=relationships)
    prune_orphaned_nodes(ontology_parser.parsed_ontology, kg)
    return kg


async def _build_nodes_context(
    nodes: List[BaseNode],
) -> str:
    if not nodes:
        return ""

    sorted_nodes = sorted(nodes, key=_node_context_sort_key)
    lines = []
    for node in sorted_nodes:
        properties_repr = _format_properties(node.properties)
        lines.append(
            f"Node Type: {node.type}, Id: {node.id}, Properties: {properties_repr}"
        )
    return "\n".join(lines) + "\n"


async def _build_relationships_context(
    nodes: List[BaseNode],
    relationships: List[RelationshipInstance],
) -> str:
    if not relationships and not nodes:
        return ""

    node_map = {node.id: node for node in nodes}
    lines = []

    sorted_relationships = sorted(relationships, key=_relationship_context_sort_key)
    for relationship in sorted_relationships:
        source_node = node_map.get(relationship.source_id)
        target_node = node_map.get(relationship.target_id)
        if not source_node or not target_node:
            continue

        source_repr = _format_properties(source_node.properties)
        target_repr = _format_properties(target_node.properties)
        rel_props = _format_properties(relationship.properties)
        lines.append(
            f"({source_node.type}:{{'id': '{source_node.id}', 'properties': {source_repr}}})"
            f"-[:{relationship.type}{{'properties': {rel_props}}}]->"
            f"({target_node.type}:{{'id': '{target_node.id}', 'properties': {target_repr}}})"
        )

    nodes_in_relationships = {rel.source_id for rel in relationships} | {
        rel.target_id for rel in relationships
    }
    nodes_not_in_relationships = [
        node for node in nodes if node.id not in nodes_in_relationships
    ]

    if nodes_not_in_relationships:
        lines.append("These Nodes without any relationships:")
        for node in sorted(nodes_not_in_relationships, key=_node_context_sort_key):
            node_repr = _format_properties(node.properties)
            lines.append(
                f"({node.type}:{{'id': '{node.id}', 'properties': {node_repr}}})"
            )

    return "\n".join(lines) + ("\n" if lines else "")


def _format_properties(properties: Optional[Dict[str, Any]]) -> str:
    if not properties:
        return "{}"
    return json.dumps(properties, sort_keys=True, default=str)


def _node_context_sort_key(node: BaseNode) -> Tuple[str, str, str]:
    return (
        node.type or "",
        _format_properties(node.properties),
        node.id or "",
    )


def _relationship_context_sort_key(
    relationship: RelationshipInstance,
) -> Tuple[str, str, str, str, str, str]:
    return (
        relationship.source_type or "",
        relationship.type or "",
        relationship.target_type or "",
        relationship.source_id or "",
        relationship.target_id or "",
        _format_properties(relationship.properties),
    )


async def _compare_and_merge_nodes(
    nodes: List[BaseNode],
    user_id: Optional[str] = None,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
) -> List[BaseNode]:
    """Compare all nodes and resolve them using LLM."""
    if not nodes or len(nodes) <= 1:
        return nodes

    entity_groups = {}
    for node in nodes:
        entity_type = node.type
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        if node.id not in [n.id for n in entity_groups[entity_type]]:
            entity_groups[entity_type].append(node)
        else:
            # merge nodes with same id
            base_node = [n for n in entity_groups[entity_type] if n.id == node.id][0]
            entity_groups[entity_type].remove(base_node)
            base_node = merge_nodes(base_node, node)
            entity_groups[entity_type].append(base_node)

    final_nodes = []
    for entity_type, nodes in entity_groups.items():
        if len(nodes) <= 1:
            final_nodes.extend(nodes)
            continue

        # Perform entity resolution for this group
        resolved_groups = await resolve_entity_group(
            entity_type,
            nodes,
            user_id=user_id,
            transform_id=transform_id,
            document_usage_id=document_usage_id,
        )
        # Process resolved groups
        for group in resolved_groups:
            if len(group) == 1:
                # Single node, no merging needed
                final_nodes.append(group[0])
                continue

            # Sort by confidence score to use highest confidence node as base
            sorted_nodes = sorted(
                group,
                key=lambda x: x.confidence_score if x.confidence_score else 0,
                reverse=True,
            )

            # Use highest confidence node as base and merge others into it
            base_node = sorted_nodes[0]
            for other_node in sorted_nodes[1:]:
                base_node = merge_nodes(base_node, other_node)

            final_nodes.append(base_node)
    return final_nodes


def _compare_and_merge_relationships(
    relationships: List[RelationshipInstance],
) -> List[RelationshipInstance]:
    """Compare all relationships and resolve them using LLM."""
    if not relationships or len(relationships) <= 1:
        return relationships

    relationship_groups = {}
    for relationship in relationships:
        relationship_uid = (
            f"{relationship.source_id}-{relationship.type}-{relationship.target_id}"
        )
        if relationship_uid not in relationship_groups:
            relationship_groups[relationship_uid] = relationship
        else:
            rel_props = relationship_groups[relationship_uid].properties
            relationship_groups[relationship_uid].properties = {
                **rel_props,
                **relationship.properties,
            }

    final_relationships = []
    for relationship_uid, relationship in relationship_groups.items():
        final_relationships.append(relationship)

    return final_relationships


def _is_duplicate_node(existing_node: BaseNode, new_node: BaseNode) -> bool:
    """Check if two nodes have the same type and properties (excluding ID)"""
    if existing_node.type != new_node.type:
        return False

    # Compare properties excluding 'id'
    existing_props = {k: v for k, v in existing_node.properties.items() if k != "id"}
    new_props = {k: v for k, v in new_node.properties.items() if k != "id"}
    return existing_props == new_props


def _is_duplicate_relationship(
    existing_relationship: RelationshipInstance, new_relationship: RelationshipInstance
) -> bool:
    """Check if two relationships have the same source, type, and target"""
    return (
        existing_relationship.source_id == new_relationship.source_id
        and existing_relationship.type == new_relationship.type
        and existing_relationship.target_id == new_relationship.target_id
    )
