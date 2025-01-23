from collections import defaultdict
from typing import Dict, List, Any
import uuid
from app.services.local_merge.schema_helpers import Neo4jStagingManager
from app.schemas.local import LocalNode, LocalEdge
from app.utils.logger import logger

class Neo4jIngestionGenerator:
  def __init__(self, ontology: Dict[str, Any], staging_manager: Neo4jStagingManager):
    self.staging_manager = staging_manager
    self.ontology = ontology
    self.UID_FIELD = '_uid_'

  def generate_node_creation(self, nodes: List[LocalNode]) -> List[str]:
    statements = []
    nodes_by_type = defaultdict(list)
    staging_label = self.staging_manager.get_staging_label()

    for node in nodes:
        nodes_by_type[node.type_].append(node)

    for node_type, type_nodes in nodes_by_type.items():
        entity_def = self.ontology.get('entities', {}).get(node_type, {})
        unique_props = self._get_unique_properties(node_type)

        for node in type_nodes:
            props = node.properties.copy()
            if '_uid_' not in props:
                props['_uid_'] = str(uuid.uuid4())

            props_str = ", ".join(f"{k}: {self._sanitize_value(v)}" for k, v in props.items())

            if unique_props and any(prop in props for prop in unique_props):
                unique_prop = next(prop for prop in unique_props if prop in props)
                stmt = f"""
                MERGE (n:{node_type}:{staging_label} {{{unique_prop}: {self._sanitize_value(props[unique_prop])}}})
                ON CREATE SET n = {{{props_str}}}
                ON MATCH SET n = {{{props_str}}}"""
            else:
                stmt = f"CREATE (n:{node_type}:{staging_label} {{{props_str}}})"

            stmt += f"\nRETURN {self._sanitize_value(node.id + ': ')} + n._uid_ as result;"
            statements.append(stmt)

    return statements

  def generate_relationship_creation(self, nodes: List[LocalNode], edges: List[LocalEdge]) -> List[str]:
    statements = []
    node_uid_map = {}
    staging_label = self.staging_manager.get_staging_label()

    for node in nodes:
        if '_uid_' in node.properties:
            node_uid_map[node.id] = (node.properties['_uid_'], node.type_)
            if '_merged_ids' in node.properties:
                merged = node.properties['_merged_ids']
                if isinstance(merged, str):
                    merged = merged.split(',')
                for mid in merged:
                    node_uid_map[mid.strip()] = (node.properties['_uid_'], node.type_)

    for edge in edges:
        source_info = node_uid_map.get(edge.from_)
        target_info = node_uid_map.get(edge.to)

        if source_info and target_info:
            entity_def = self.ontology.get('entities', {}).get(source_info[1], {})
            relationships = entity_def.get('relationships', {})

            if edge.relationship in relationships:
                target_type = relationships[edge.relationship].get('target')
                if target_type == target_info[1]:
                    stmt = f"""
                    MATCH (source:{source_info[1]}:{staging_label}), (target:{target_info[1]}:{staging_label})
                    WHERE source._uid_ = '{source_info[0]}'
                    AND target._uid_ = '{target_info[0]}'
                    MERGE (source)-[r:{edge.relationship}]->(target)
                    ON CREATE SET r.{self.UID_FIELD} = '{uuid.uuid4()}';"""
                    statements.append(stmt)

    return statements

  def _generate_section_node(self, node: LocalNode) -> str:
      """Generate creation statement for section nodes with mandatory properties"""
      props = node.properties.copy()
      entity_def = self.ontology.get('entities', {}).get(node.type_, {})
      required_props = {
          name: prop_def for name, prop_def in entity_def.get('properties', {}).items()
          if isinstance(prop_def, dict) and prop_def.get('required', False)
      }

      # Validate required properties
      for prop_name in required_props:
          if prop_name not in props:
              raise ValueError(f"Missing required property {prop_name} for {node.type_}")

      if '_uid_' not in props:
          props['_uid_'] = str(uuid.uuid4())

      props_str = ", ".join(f"{k}: {self._sanitize_value(v)}" for k, v in props.items())
      return f"CREATE (n:{node.type_} {{{props_str}}}) RETURN {self._sanitize_value(node.id + ': ')} + n._uid_ as result;"

  def _generate_subsection_node(self, node: LocalNode) -> str:
      """Generate creation statement for subsection nodes with parent validation"""
      props = node.properties.copy()
      entity_def = self.ontology.get('entities', {}).get(node.type_, {})

      # Add unique constraint handling
      unique_props = {
          name: prop_def for name, prop_def in entity_def.get('properties', {}).items()
          if isinstance(prop_def, dict) and prop_def.get('unique', False)
      }

      if '_uid_' not in props:
          props['_uid_'] = str(uuid.uuid4())

      if unique_props and any(prop in props for prop in unique_props):
          unique_prop = next(prop for prop in unique_props if prop in props)
          props_str = ", ".join(f"{k}: {self._sanitize_value(v)}" for k, v in props.items())
          return f"""
          MERGE (n:{node.type_} {{{unique_prop}: {self._sanitize_value(props[unique_prop])}}})
          ON CREATE SET n = {{{props_str}}}
          ON MATCH SET n = {{{props_str}}}
          RETURN {self._sanitize_value(node.id + ': ')} + n._uid_ as result;
          """

      props_str = ", ".join(f"{k}: {self._sanitize_value(v)}" for k, v in props.items())
      return f"CREATE (n:{node.type_} {{{props_str}}}) RETURN {self._sanitize_value(node.id + ': ')} + n._uid_ as result;"

  def _belongs_to_section(self, node: LocalNode, section_node: LocalNode) -> bool:
      """Validate if node belongs to section based on ontology and metadata"""
      section_def = self.ontology.get('entities', {}).get(section_node.type_, {})
      relationships = section_def.get('relationships', {})

      # Check direct relationship in ontology
      for rel_def in relationships.values():
          if rel_def.get('target') == node.type_:
              return True

      # Check metadata-based relationship
      section_info = node.metadata.get('section') if hasattr(node, 'metadata') else None
      if section_info and section_info == section_node.type_:
          return True

      # Check nested relationships
      for rel_name, rel_def in relationships.items():
          target_type = rel_def.get('target')
          if isinstance(target_type, list) and node.type_ in target_type:
              return True

      return False

  def _create_relationship_stmt(
      self, source_node: LocalNode, target_node: LocalNode,
      relationship: str, properties: Dict, staging_label: str
  ) -> str:
      """Create a single relationship statement with UID checks"""
      # Check if source and target nodes have UIDs
      source_uid = source_node.properties.get(self.UID_FIELD)
      target_uid = target_node.properties.get(self.UID_FIELD)

      if not source_uid or not target_uid:
          logger.warning(f"Warning: Missing UID for relationship {source_node.id} -> {target_node.id}")
          return None

      create_stmt = f"MERGE (source)-[r:{relationship}]->(target)"

      set_stmts = [f"ON CREATE SET r.{self.UID_FIELD} = '{uuid.uuid4()}'"]
      if properties:
          for key, value in properties.items():
              if value is not None:
                  set_stmts.append(f"ON CREATE SET r.{key} = {self._sanitize_value(value)}")

      # Build match conditions for source and target nodes
      source_match = f"source.{self.UID_FIELD} = {self._sanitize_value(source_uid)}"
      target_match = f"target.{self.UID_FIELD} = {self._sanitize_value(target_uid)}"

      match_stmt = f"""
      MATCH (source:{source_node.type_}{staging_label})
      WHERE {source_match}
      WITH source
      MATCH (target:{target_node.type_}{staging_label})
      WHERE {target_match}
      WITH source, target
      LIMIT 1"""

      full_stmt = "\n".join([match_stmt, create_stmt] + set_stmts) + ";"
      return full_stmt

  def _get_unique_properties(self, node_type: str) -> List[str]:
    """Get unique properties for a node type from ontology"""
    entity_def = self.ontology.get('entities', {}).get(node_type, {})
    properties = entity_def.get('properties', {})

    unique_props = []
    for prop_name, prop_def in properties.items():
        if isinstance(prop_def, dict) and (
            prop_def.get('unique', False) or
            prop_def.get('required', False)
        ):
            unique_props.append(prop_name)
    return unique_props

  def _build_node_uid_map(self, nodes: List[LocalNode]):
    node_uid_map = {}
    for node in nodes:
        uid = node.properties.get('_uid_')
        if uid:
            node_uid_map[node.id] = (uid, node.type_)
            merged = node.properties.get('_merged_ids')
            if merged:
                if isinstance(merged, str):
                    merged = merged.split(',')
                for mid in merged:
                    node_uid_map[mid.strip()] = (uid, node.type_)
    return node_uid_map

  def _sanitize_value(self, value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace('\\', '\\\\').replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, list) or (isinstance(value, str) and ',' in value):
        return self._sanitize_value(','.join(str(x) for x in (value.split(',') if isinstance(value, str) else value)))
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)

  def create_document_node(self, doc_metadata: LocalNode, nodes: List[LocalNode]) -> List[str]:
    """Creates document node and relationships to section nodes"""
    staging_label = f":{self.staging_manager.get_staging_label()}" if self.staging_manager.get_staging_label() else ""

    # Create document node
    doc_props = " ".join([f"SET d.{k} = {self._sanitize_value(v)}" for k, v in doc_metadata.properties.items()])
    id_prop = f"{{{self.UID_FIELD}: '{uuid.uuid4()}'}}"
    return f"CREATE (d:Document{staging_label}{id_prop}) {doc_props}" + f" RETURN 'doc_id: ' + d.{self.UID_FIELD};"

  def create_document_rels(self, doc_id: str, doc_metadata: LocalNode, nodes: List[LocalNode]) -> List[str]:
    """Creates document node and relationships to section nodes"""
    statements = []
    staging_label = f":{self.staging_manager.get_staging_label()}" if self.staging_manager.get_staging_label() else ""
    doc_metadata_props = doc_metadata.properties
    match_props = f"n.{self.UID_FIELD} = {self._sanitize_value(doc_metadata_props['_uid_'])}"
    rel_stmt = f"""
    MATCH (d:Document{staging_label})
    WHERE d.{self.UID_FIELD} = '{doc_id}'
    WITH d
    MATCH (n:{doc_metadata.type_}{staging_label})
    WHERE {match_props}
    MERGE (d)-[r:HAS]->(n)
    SET r.section = '{doc_metadata.type_}',
    r.{self.UID_FIELD} = '{uuid.uuid4()}';
    """
    statements.append(rel_stmt)
    # Create relationships to sections
    section_types = self.ontology.get('sections', {})
    for node in nodes:
        if node.type_ in section_types:
            props = node.properties
            match_props = f"n.{self.UID_FIELD} = {self._sanitize_value(props['_uid_'])}"
            rel_stmt = f"""
            MATCH (d:Document{staging_label})
            WHERE d.{self.UID_FIELD} = '{doc_id}'
            WITH d
            MATCH (n:{node.type_}{staging_label})
            WHERE {match_props}
            MERGE (d)-[r:HAS]->(n)
            SET r.section = '{node.type_}',
            r.{self.UID_FIELD} = '{uuid.uuid4()}';
            """
            statements.append(rel_stmt)

    return statements