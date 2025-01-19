from langchain_core.documents import Document
from app.schemas.transform import KnowledgeGraph
from app.utils.llm_client_service import call_llm, call_llm_anthropic
from typing import List, Dict, Type
from pydantic import BaseModel
from app.utils.yaml_helper import KnowledgeGraphYAMLExporter
from app.services.transform.schema_helpers import DynamicSchemaGenerator
import json

def extract_knowledge_graph(docs: List[Document], ontology: dict) -> List[KnowledgeGraph]:
  results = []
  section_map = {}
  for sec, subsections in ontology['sections'].items():
    section_map[sec] = {'subsections': subsections}

  for doc in docs:
    prompt = f"""
    Help me understand the following by describing as a detailed knowledge graph.
    For the section and subsection property inside metadata, please decide based on the below map of sections and subsections:
    {str(section_map)}

    Text:
    {doc.page_content}
    """
    result = call_llm([{"role": "user", "content":prompt}], KnowledgeGraph)
    results.append(result)

  return results

def extract_by_ontology(ontology: str,
                        merged: Dict[str, Dict[str, Type[BaseModel]]]) -> List[Type[BaseModel]]:
  schema_gen = DynamicSchemaGenerator(ontology)
  schema_map = schema_gen.create_schema_map()

  results = []
  for section, sub in merged.items():
    if sub is None:
      continue
    for subsection, kg in sub.items():
      kg_yaml = KnowledgeGraphYAMLExporter.to_yaml(kg)
      schema = schema_gen.get_section_schema(section, subsection)
      if schema is None:
          continue

      prompt = f"""Extract knowledge graph following schema:
      Valid entities and required properties:
      {json.dumps(schema['properties'], indent=2)}

      Valid relationships per entity:
      {json.dumps(schema['relationships'], indent=2)}

      Rules:
      - Every node must have ALL required properties for its type
      - Skip nodes with missing required data
      - Follow relationship schema exactly
      - Ensure that the from and to nodes mentioned in the relationship are extracted as well and rightly referenced by their ids
      - Keep full info from source for complete nodes

      Source YAML: {kg_yaml}
      """
      response_model = schema_map[section][subsection]
      result = call_llm_anthropic([{"role": "user", "content":prompt}], response_model)
      results.append(result)

    return results
  
def extract_metadata(ontology: str, text: str) -> KnowledgeGraph:
  schema_gen = DynamicSchemaGenerator(ontology)
  section = "Metadata"

  schema = schema_gen.get_metadata_schema()
  if schema is None:
      return None

  prompt = f"""Extract knowledge graph following schema:
  Valid entities and required properties:
  {json.dumps(schema['properties'], indent=2)}

  Valid relationships per entity:
  {json.dumps(schema['relationships'], indent=2)}

  Rules:
  - Every node must have ALL required properties for its type
  - Skip nodes with missing required data
  - Follow relationship schema exactly
  - Keep full info from source for complete nodes

  Document text: {text}
  """
  response_model = schema_gen.create_section_knowledge_graph(section, section) #KnowledgeGraph
  result = call_llm_anthropic([{"role": "user", "content":prompt}], response_model)

  return result