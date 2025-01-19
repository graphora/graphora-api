import yaml
from typing import List, Tuple, Dict, Type
from pydantic import BaseModel
from app.services.local_merge.schema_helpers import Neo4jStagingManager, Neo4jSchemaGenerator
from app.services.local_merge.ingestion_helpers import Neo4jIngestionGenerator
from app.utils.neo4j import run_cypher_batch_staging
from app.services.local_merge.resolvers import resolve_with_bert
from app.services.local_merge.normalizers import normalize_graph_data
from app.schemas.local import LocalNode, LocalEdge
from app.utils.logger import logger
import uuid
from app.services.local_merge.helpers import create_staging_from_extracted_data, create_staging_from_extracted_metadata


def sanitise(nodes: List[LocalNode], edges: List[LocalEdge]) -> Tuple[List[LocalNode], List[LocalEdge]]:
  normalized_nodes, normalized_edges = normalize_graph_data(nodes, edges)
  resolved_nodes, resolved_edges = resolve_with_bert(normalized_nodes, normalized_edges)
  return resolved_nodes, resolved_edges

def ingest(ontology_yaml: str,
           resolved_nodes: List[LocalNode], resolved_edges: List[LocalEdge],
           staging: Neo4jStagingManager):
    # Load ontology and initialize components
    ontology = yaml.safe_load(ontology_yaml)

    # Proceed with schema and data ingestion
    schema_gen = Neo4jSchemaGenerator(ontology, staging)
    ingestion_gen = Neo4jIngestionGenerator(ontology, staging)

    # Create schema
    staging_constraints = schema_gen.generate_constraints(staging.is_staging)
    staging_indexes = schema_gen.generate_indexes(staging.is_staging)

    # Generate and execute statements
    run_cypher_batch_staging(staging_constraints)
    run_cypher_batch_staging(staging_indexes)

    node_stmts = ingestion_gen.generate_node_creation(resolved_nodes)
    if node_stmts:
        nodes_with_ids = run_cypher_batch_staging(node_stmts)
        resolved_nodes = set_node_id(nodes_with_ids, resolved_nodes)

    rel_stmts = ingestion_gen.generate_relationship_creation(resolved_nodes, resolved_edges)
    if rel_stmts:
        run_cypher_batch_staging(rel_stmts)

    return resolved_nodes, resolved_edges

def create_document(ontology_yaml:str, metadata: LocalNode,
                    nodes: List[LocalNode], staging: Neo4jStagingManager):
  ontology = yaml.safe_load(ontology_yaml)
  ingestion_gen = Neo4jIngestionGenerator(ontology, staging)
  stmt = ingestion_gen.create_document_node(metadata, nodes)
  doc_ids = run_cypher_batch_staging([stmt])
  doc_id = doc_ids[0].iat[0, 0].split(': ')[1]
  stmts = ingestion_gen.create_document_rels(doc_id, metadata, nodes)
  run_cypher_batch_staging(stmts)


def group_data_by_section(data: List[List[Type[BaseModel]]]) -> Dict[str, List[Type[BaseModel]]]:
  grouped_data = {}
  for items in data:
    for item in items:
      section = item.metadata.section
      if section not in grouped_data:
        grouped_data[section] = []
      grouped_data[section].append(item)
  return grouped_data

def set_node_id(nodes_with_ids, resolved_nodes: List[LocalNode]):
    # Create mapping of node IDs to UIDs from the results
    id_to_uid = {}
    for result in nodes_with_ids:
        if result is not None and len(result) > 0:
            id_mapping = result.at[0, 'result']
            if id_mapping:
                node_id, uid = id_mapping.split(': ')
                id_to_uid[node_id] = uid

    # Update nodes with mapped UIDs
    for node in resolved_nodes:
        if node.id in id_to_uid:
            node.properties['_uid_'] = id_to_uid[node.id]
        else:
            node.properties['_uid_'] = str(uuid.uuid4())

        # Map merged IDs
        if '_merged_ids' in node.properties:
            merged = node.properties['_merged_ids']
            if isinstance(merged, str):
                merged = merged.split(',')
            for mid in merged:
                id_to_uid[mid.strip()] = node.properties['_uid_']

    return resolved_nodes

def sanitise_and_ingest(ontology: str, metadata: List[Type[BaseModel]],
                        data: List[List[Type[BaseModel]]], staging: Neo4jStagingManager) -> str:
    # Convert metadata to nodes and edges
    metadata_nodes, metadata_edges = create_staging_from_extracted_metadata(metadata)

    # Group data by section and process each section
    grouped_data = group_data_by_section(data)
    section_nodes = []
    section_edges = []

    # Process each section maintaining merge info
    for section, section_data in grouped_data.items():
        nodes, edges = create_staging_from_extracted_data(section_data)
        # Process section-level merges
        resolved_nodes, resolved_edges = sanitise(nodes, edges)
        section_nodes.extend(resolved_nodes)
        section_edges.extend(resolved_edges)

    # Combine metadata with section data
    all_nodes = metadata_nodes + section_nodes
    all_edges = metadata_edges + section_edges

    # Create node mapping including all merged IDs
    node_map = {}
    for node in all_nodes:
        node_map[node.id] = node
        for id_field in ['_merged_ids', 'merged_ids']:
            merged = node.properties.get(id_field)
            if merged:
                if isinstance(merged, str):
                    merged = [m.strip() for m in merged.split(',')]
                for merged_id in merged:
                    node_map[merged_id] = node

    # Deduplicate nodes preserving merge history
    final_nodes = []
    seen_ids = set()

    for node in all_nodes:
        base_node = node_map.get(node.id)
        if base_node and base_node.id not in seen_ids:
            final_nodes.append(base_node)
            seen_ids.add(base_node.id)
            # Add all merged IDs to seen set
            for id_field in ['_merged_ids', 'merged_ids']:
                merged = base_node.properties.get(id_field)
                if merged:
                    if isinstance(merged, str):
                        merged = [m.strip() for m in merged.split(',')]
                    seen_ids.update(merged)

    # Process edges using node mapping
    final_edges = []
    seen_edge_sigs = set()

    for edge in all_edges:
        source = node_map.get(edge.from_)
        target = node_map.get(edge.to)

        if source and target:
            edge_sig = f"{source.id}_{edge.relationship}_{target.id}"
            if edge_sig not in seen_edge_sigs:
                final_edges.append(LocalEdge(
                    from_=source.id,
                    to=target.id,
                    relationship=edge.relationship,
                    properties=edge.properties
                ))
                seen_edge_sigs.add(edge_sig)
                logger.info(f"Mapped edge: {source.id} -> {target.id} ({edge.relationship})")

    # Ingest final data
    ingest(ontology, final_nodes, final_edges, staging)

    # Create document relationships
    metadata_node = next((d for d in final_nodes if d.type_ == 'Metadata'), None)
    if metadata_node is not None:
        return create_document(ontology, metadata_node, final_nodes, staging)

    return None