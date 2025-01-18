from typing import Dict, List, Any

class Neo4jStagingManager:
    def __init__(self, staging_label: str = "Staging", is_staging=True):
        self.staging_label = staging_label
        self.is_staging = True

    def cleanup_staging(self) -> str:
        return f"MATCH (n:{self.staging_label}) DETACH DELETE n;"

    def get_staging_label(self) -> str:
        return self.staging_label
      
class Neo4jSchemaGenerator:
    def __init__(self, ontology: Dict[str, Any], staging_manager: Neo4jStagingManager):
        self.ontology = ontology
        self.staging_manager = staging_manager
        self.UID_FIELD = '_uid_'
        self._constraints_cache = None

    def generate_constraints(self, is_staging: bool = True) -> List[str]:
      constraints = []
      staging_label = self.staging_manager.get_staging_label() if is_staging else ""

      # For staging graphs, only create non-unique indexes
      if is_staging:
          for entity_name, entity_def in self.ontology.get('entities', {}).items():
              properties = entity_def.get('properties', {})
              for prop_name, prop_def in properties.items():
                  if self._should_create_index(entity_name, prop_name, prop_def):
                      constraints.append(
                          self._create_index_stmt(entity_name, prop_name, staging_label)
                      )
          return constraints

      # For production, create unique constraints
      for entity_name, entity_def in self.ontology.get('entities', {}).items():
          properties = entity_def.get('properties', {})
          for prop_name, prop_def in properties.items():
              if self._should_create_constraint(prop_name, prop_def):
                  constraints.append(
                      self._create_constraint_stmt(entity_name, prop_name, staging_label)
                  )
      constraints.append(
          self._create_constraint_stmt(entity_name, self.UID_FIELD, staging_label)
      )
      return constraints

    def _should_create_constraint(self, prop_name: str, prop_def: Dict) -> bool:
        if isinstance(prop_def, dict):
            return prop_def.get('unique', False) or prop_def.get('required', False)
        return False

    def _create_constraint_stmt(self, entity: str, prop: str, staging_label: str) -> str:
        prefix = f"staging_{entity.lower()}" if staging_label else entity.lower()
        labels = f"{entity}" if not staging_label else f"{entity}:{staging_label}"
        return f"""CREATE CONSTRAINT {prefix}_{prop}
                  IF NOT EXISTS FOR (n:{entity})
                  REQUIRE n.{prop} IS UNIQUE;"""

    def generate_indexes(self, is_staging: bool = True) -> List[str]:
        indexes = []
        staging_label = self.staging_manager.get_staging_label() if is_staging else ""

        for entity_name, entity_def in self.ontology.get('entities', {}).items():
            properties = entity_def.get('properties', {})
            for prop_name, prop_def in properties.items():
                if self._should_create_index(entity_name, prop_name, prop_def):
                    indexes.append(
                        self._create_index_stmt(entity_name, prop_name, staging_label)
                    )
        return indexes

    def _should_create_index(self, entity: str, prop: str, prop_def: Dict) -> bool:
        return prop_def.get('index', False) or (prop in ['type', 'name', 'id'])

    def _has_constraint(self, entity: str, prop: str) -> bool:
        return any(prop in c for c in self.generate_constraints())

    def _create_index_stmt(self, entity: str, prop: str, staging_label: str) -> str:
        prefix = f"staging_{entity.lower()}" if staging_label else entity.lower()
        labels = f"{entity}" if not staging_label else f"{entity}:{staging_label}"
        return f"""CREATE INDEX {prefix}_{prop}_idx
                  IF NOT EXISTS FOR (n:{entity})
                  ON (n.{prop});"""