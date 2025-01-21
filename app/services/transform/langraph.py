from typing import TypedDict, List, Dict, Type, Any
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
from app.utils.logger import logger
from app.schemas.transform import KnowledgeGraph
from .chunk import do_semantic_chunking
from .extraction_helpers import extract_knowledge_graph, extract_by_ontology
from .kg_helpers import KnowledgeGraphManager
from .extraction_helpers import extract_metadata
from app.services.job_manager import get_job_manager

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class State(TypedDict):
  app: FastAPI
  transform_id: str
  text: str
  chunks: List[Document]
  graphs: List[KnowledgeGraph]
  ontology_str: str
  ontology_obj: dict
  merged_graphs: Dict[str, Dict[str, Type[BaseModel]]]
  domain_graphs: List[List[Type[BaseModel]]]
  metadata: Dict[str, Any]

async def chunk(state: State):
  logger.info('Chunking...')
  return {"chunks": do_semantic_chunking(state['text'])}

async def graphs_extraction(state: State):
  logger.info('Chunking DONE')
  logger.info('Extracting Generic Graphs in Parallel...')
  job_manager = get_job_manager(state['app'])
  await job_manager.update_progress(state['transform_id'], 15.0)
  with ThreadPoolExecutor(max_workers=10) as executor:
      graphs = [
          graph[0] for graph in executor.map(
              lambda chunk: extract_knowledge_graph([chunk], state['ontology_obj']),
              state['chunks']
          )
      ]
  return {"graphs": graphs}

async def merge_graphs(state: State):
  logger.info('Extracting Generic Graph DONE')
  logger.info('Merging Graphs...')
  job_manager = get_job_manager(state['app'])
  await job_manager.update_progress(state['transform_id'], 25.0)
  return {"merged_graphs": KnowledgeGraphManager().merge_graphs(state['graphs'])}

async def build_domain_graph(state: State):
  logger.info('Building Domain Graphs in Parallel...')
  job_manager = get_job_manager(state['app'])
  await job_manager.update_progress(state['transform_id'], 35.0)
  with ThreadPoolExecutor(max_workers=10) as executor:
      # Ensure ontology_str is a list for mapping
      ontology_list = [state['ontology_str']] if isinstance(state['ontology_str'], str) else state['ontology_str']
      domain_graphs = list(executor.map(
          lambda ont: extract_by_ontology(ont, state['merged_graphs']),
          ontology_list
      ))
  return {"domain_graphs": domain_graphs}

async def generate_metadata(state: State):
  logger.info('Generating Metadata')
  text = state['chunks'][0].page_content
  job_manager = get_job_manager(state['app'])
  await job_manager.update_progress(state['transform_id'], 50.0)
  return {'metadata': extract_metadata(state['ontology_str'], text)}

async def create_processing_chain():
  workflow = StateGraph(State)

  # Add nodes to the graph
  workflow.add_node("chunk", chunk)
  workflow.add_node("graphs_extraction", graphs_extraction)
  workflow.add_node("merge_graphs", merge_graphs)
  workflow.add_node("build_domain_graph", build_domain_graph)
  workflow.add_node("generate_metadata", generate_metadata)

  # Add edges to the graph
  workflow.set_entry_point("chunk") # Set the entry point of the graph
  workflow.add_edge("chunk", "graphs_extraction")
  workflow.add_edge("graphs_extraction", "merge_graphs")
  workflow.add_edge("merge_graphs", "build_domain_graph")
  workflow.add_edge("build_domain_graph", "generate_metadata")
  workflow.add_edge("generate_metadata", END)

  # Compile the graph
  app = workflow.compile()
  return app