from typing import List, Tuple, Type
from pydantic import BaseModel
from schemas.local import LocalNode, LocalEdge
from utils.yaml_helper import KnowledgeGraphYAMLExporter
import yaml


def create_staging_from_extracted_data(extracts: List[Type[BaseModel]]) -> Tuple[List[LocalNode], List[LocalEdge]]:
  staging_nodes = []
  staging_edges = []

  for ext in extracts:
    chunk_str = KnowledgeGraphYAMLExporter().to_yaml(ext)
    section = yaml.safe_load(chunk_str)
    metadata = section.get('metadata', {})

    # Parse nodes
    for node_data in section.get('nodes', []):
      extracted_id = node_data['id']
      node_type = node_data['type']
      properties = node_data.get('properties', {})

      # Create node using model_validate
      node_obj = LocalNode.model_validate({
        "id": extracted_id,
        "type": node_type,  # Using "type" instead of "type_"
        "properties": properties,
        "metadata": {
          'section': metadata.get('section'),
          'subsections': metadata.get('subsections')
        }
      })

      staging_nodes.append(node_obj)

    # Parse edges
    for edge_data in section.get('edges', []):
      from_id = edge_data['from']
      to_id = edge_data['to']
      relationship = edge_data['relationship']
      properties = edge_data.get('properties', {})

      # Create edge using model_validate
      edge_obj = LocalEdge.model_validate({
        "from": from_id,  # Using "from" instead of "from_"
        "to": to_id,
        "relationship": relationship,
        "properties": properties,
        "metadata": {
          'section': metadata.get('section'),
          'subsections': metadata.get('subsections')
        }
      })

      staging_edges.append(edge_obj)
  return staging_nodes, staging_edges


def create_staging_from_extracted_metadata(ext: Type[BaseModel]) -> Tuple[List[LocalNode], List[LocalEdge]]:
  staging_nodes = []
  staging_edges = []

  metadata_str = KnowledgeGraphYAMLExporter().to_yaml(ext)
  metadata = dict(yaml.safe_load(metadata_str))

  # Parse nodes
  for node_data in metadata.get('nodes', []):
    extracted_id = node_data['id']
    node_type = node_data.get('type', 'Metadata')
    properties = node_data.get('properties', {})

    # Create node using model_validate
    node_obj = LocalNode.model_validate({
      "id": extracted_id,
      "type": node_type,  # Using "type" instead of "type_"
      "properties": properties
    })

    staging_nodes.append(node_obj)

  # Parse edges
  for edge_data in metadata.get('edges', []):
    from_id = edge_data['from']
    to_id = edge_data['to']
    relationship = edge_data['relationship']
    properties = edge_data.get('properties', {})

    # Create edge using model_validate
    edge_obj = LocalEdge.model_validate({
      "from": from_id,  # Using "from" instead of "from_"
      "to": to_id,
      "relationship": relationship,
      "properties": properties
    })

    staging_edges.append(edge_obj)

  return staging_nodes, staging_edges