from typing import Dict, List, Any, Type, Set
from pydantic import BaseModel, Field, create_model, field_validator, model_validator
import yaml
import threading
import contextlib

class GraphContext:
    """Thread-safe context manager for node validation"""
    _thread_local = threading.local()

    def __init__(self, context_id: str):
        self.context_id = context_id

    @classmethod
    def _get_context_stack(cls):
        if not hasattr(cls._thread_local, 'contexts'):
            cls._thread_local.contexts = {}
        return cls._thread_local.contexts

    @contextlib.contextmanager
    def register_nodes(self, nodes: List[Dict]):
        """Thread-safe context manager for registering nodes"""
        try:
            # Initialize context for this instance
            contexts = self._get_context_stack()
            contexts[self.context_id] = {node['id'] for node in nodes}
            contexts[f"{self.context_id}_types"] = {node['id']: node['type'] for node in nodes}
            yield
        finally:
            # Cleanup context
            contexts = self._get_context_stack()
            contexts.pop(self.context_id, None)

    def get_node_ids(self) -> Set[str]:
        """Get node IDs for current context"""
        contexts = self._get_context_stack()
        return contexts.get(self.context_id, set())

    def get_node_types(self) -> Dict[str, str]:
      contexts = self._get_context_stack()
      return contexts.get(f"{self.context_id}_types", {})

class OntologyParser:
   def __init__(self, yaml_content: str):
       self.ontology = yaml.safe_load(yaml_content)
       self.models = {}
       self.relationships = {}

   def create_pydantic_models(self):
       for entity_name, entity_spec in self.ontology['entities'].items():
           properties = {}
           relationships = {}

           if 'properties' in entity_spec:
               for prop_name, prop_spec in entity_spec['properties'].items():
                   field_type = self._get_field_type(prop_spec['type'])
                   field_desc = prop_spec.get('description', '')
                   required = prop_spec.get('required', False)

                   properties[prop_name] = (
                       field_type,
                       Field(None if not required else ..., description=field_desc)
                   )

           if 'relationships' in entity_spec:
               for rel_name, rel_spec in entity_spec['relationships'].items():
                   relationships[rel_name] = rel_spec['target']

           model = create_model(
               entity_name,
               **properties,
               __base__=BaseModel
           )

           self.models[entity_name] = model
           self.relationships[entity_name] = relationships

   def _get_field_type(self, type_str: str) -> Type:
       type_map = {
           'str': str,
           'int': int,
           'float': float,
           'bool': bool
       }
       return type_map.get(type_str, str)

   def get_common_elements(self) -> Dict[str, Set]:
       all_targets = set()
       all_rels = set()
       sections = set(self.ontology.get('sections', {}).keys())

       for rels in self.relationships.values():
           for rel_name, target in rels.items():
               if not any(rel_name.startswith(section) for section in sections):
                   all_rels.add(rel_name)
               if not any(target.startswith(section) for section in sections):
                   all_targets.add(target)

       return {
           'common_nodes': all_targets,
           'common_relationships': all_rels
       }

class DynamicSchemaGenerator:
  def __init__(self, yaml_content: str):
    self.parser = OntologyParser(yaml_content)
    self.parser.create_pydantic_models()
    self.common_entities = self._get_common_entities()

  def _get_common_entities(self) -> Set[str]:
    common = set()
    for entity_spec in self.parser.ontology['entities'].values():
        if 'relationships' in entity_spec:
            for rel in entity_spec['relationships'].values():
                common.add(rel['target'])
    return common

  def create_section_knowledge_graph(self, section: str, subsection: str) -> Type[BaseModel]:
    valid_types = set([section])
    context_id = f'{section}-{subsection}'
    graph_context = GraphContext(context_id)
    ontology_entities = self.parser.ontology['entities']

    if subsection is not None:
      valid_types.add(subsection)
      for key, entity_spec in self.parser.ontology['entities'][subsection].items():
          if key == 'relationships':
            for rel in entity_spec.values():
                valid_types.add(rel['target'])

    class SectionNode(BaseModel):
        id: str
        type_: str = Field(..., alias="type")
        properties: Dict[str, Any] = Field(default_factory=dict)

        @field_validator('type_')
        @classmethod
        def validate_type(cls, v):
            if v not in valid_types:
                raise ValueError(f"Type must be one of: {valid_types}")
            return v

        @field_validator('properties')
        @classmethod
        def validate_properties(cls, v, info):
            entity_type = info.data.get('type_')
            if entity_type in valid_types:
                entity_spec = self.parser.ontology['entities'][entity_type]
                allowed_props = entity_spec.get('properties', {})
                return {k: v for k, v in v.items() if k in allowed_props}
            return v

    class SectionEdge(BaseModel):
      from_: str = Field(..., alias="from")  # Changed from Node to str
      to: str
      relationship: str
      properties: Dict[str, Any] = Field(default_factory=dict)

      @field_validator('from_')
      @classmethod
      def validate_from(cls, v):
          node_ids = graph_context.get_node_ids()
          if not node_ids:
              return v
          if v not in node_ids:
              raise ValueError(f"From node '{v}' does not exist in nodes")
          return v

      @field_validator('to')
      @classmethod
      def validate_to(cls, v):
          node_ids = graph_context.get_node_ids()
          if not node_ids:
              return v
          if v not in node_ids:
              raise ValueError(f"To node '{v}' does not exist in nodes")
          return v

      @model_validator(mode='after')
      def validate_relationship(self):
          # Get node types from context
          node_types = graph_context.get_node_types()  # Need to add this to GraphContext
          if not node_types or len(node_types) == 0:
              return self
          if self.from_ is None or self.from_ not in node_types:
              raise ValueError(f"Invalid from_ node type '{self.from_}'")
          from_type = node_types[self.from_]

          if self.to is None or self.to not in node_types:
              raise ValueError(f"Invalid to node type '{self.to}'")
          to_type = node_types[self.to]
          valid_rels = ontology_entities[from_type].get('relationships', {})
          if self.relationship not in valid_rels:
              raise ValueError(f"Invalid relationship {self.relationship} for type {from_type}")
          if valid_rels[self.relationship]['target'] != to_type:
              raise ValueError(f"Invalid target type {to_type} for relationship {self.relationship}")

          return self

    def validate_graph(cls, values):
        nodes = values.get('nodes', [])
        with graph_context.register_nodes(nodes):
            for node in nodes:
                SectionNode.model_validate(node)
            edges = values.get('edges', [])
            for edge in edges:
                SectionEdge.model_validate(edge)
        return values

    return create_model(
        f"KnowledgeGraph_{section}_{subsection}",
        metadata=(Metadata, ...),
        nodes=(List[SectionNode], Field(default_factory=list, description="List of node definitions")),
        edges=(List[SectionEdge], Field(default_factory=list, description="List of edge definitions")),
        __base__=BaseModel,
        model_config=dict(ignored_types=(classmethod,)),
        __validators__={'validate_all': model_validator(mode='before')(validate_graph)}
    )

  def create_schema_map(self) -> Dict[str, Dict[str, Type[BaseModel]]]:
      schema_map = {}

      for section, subsections in self.parser.ontology['sections'].items():
          if section not in schema_map:
            schema_map[section] = {}
          if subsections is None:
            continue
          for subsection in subsections:
              schema_map[section][subsection] = self.create_section_knowledge_graph(section, subsection)

      return schema_map

  def get_section_schema(self, section: str, subsection: str):
    if section not in self.parser.ontology['entities'] or subsection not in self.parser.ontology['entities']:
      return None
    section_spec = self.parser.ontology['entities'][section]
    subsection_spec = self.parser.ontology['entities'][subsection]

    # Get valid entities and relationships
    rel_targets = {rel['target'] for spec in [section_spec, subsection_spec]
                  for rel in spec.get('relationships', {}).values()}
    valid_entities = {section, subsection} | rel_targets

    valid_rels = {rel_name: {'from': spec_name, 'to': spec['relationships'][rel_name]['target']}
              for spec_name, spec in [(section, section_spec), (subsection, subsection_spec)]
              for rel_name in spec.get('relationships', {})
              if spec_name == subsection or (spec['relationships'][rel_name]['target'] == subsection)}

    return {
        "entities": set(valid_entities),
        "relationships": valid_rels,
        "properties": {
            entity: self.parser.ontology['entities'][entity].get('properties', {})
            for entity in valid_entities
        }
    }

  def get_metadata_schema(self):
    section = 'Metadata'
    if section not in self.parser.ontology['entities']:
      return None
    section_spec = self.parser.ontology['entities'][section]

    # Get valid entities and relationships
    rel_targets = {rel['target'] for spec in [section_spec]
                  for rel in spec.get('relationships', {}).values()}
    valid_entities = {section} | rel_targets

    valid_rels = {rel_name: {'from': spec_name, 'to': spec['relationships'][rel_name]['target']}
              for spec_name, spec in [(section, section_spec)]
              for rel_name in spec.get('relationships', {})}

    return {
      "entities": set(valid_entities),
      "relationships": valid_rels,
      "properties": {
        entity: self.parser.ontology['entities'][entity]['properties']
        for entity in valid_entities
        if 'properties' in self.parser.ontology['entities'][entity]
      }
    }