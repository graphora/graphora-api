from typing import List, Dict, Any, Type, TypedDict
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from app.schemas.transform import KnowledgeGraph
from pydantic import BaseModel
from app.utils.logger import logger
from concurrent.futures import ThreadPoolExecutor
from .chunk import do_semantic_chunking
from .extraction_helpers import extract_knowledge_graph, extract_by_ontology
from .kg_helpers import KnowledgeGraphManager
from .extraction_helpers import extract_metadata
from app.services.job_manager import get_job_manager
from fastapi import FastAPI
from langchain.chains.sequential import SequentialChain
from langchain.chains.base import Chain
from typing import Dict, List
import asyncio
from instructor.exceptions import InstructorRetryException
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class State(TypedDict):
    transform_id: str
    text: str
    chunks: List[Document]
    graphs: List[KnowledgeGraph]
    ontology_str: str
    ontology_obj: dict
    merged_graphs: Dict[str, Dict[str, Type[BaseModel]]]
    domain_graphs: List[List[Type[BaseModel]]]  
    metadata: Dict[str, Any]
    app: FastAPI

class ChunkingChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["text"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["chunks"]
    
    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        logger.info('Chunking...')
        return {"chunks": do_semantic_chunking(inputs['text'])}

class GraphExtractionChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["chunks", "transform_id", "app", "ontology_obj"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["graphs"]
    
    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """Synchronous call - delegates to async version"""
        return asyncio.run(self._acall(inputs))
    
    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        logger.info('Chunking DONE')
        logger.info('Extracting Generic Graphs in Parallel...')
        job_manager = get_job_manager(inputs['app'])
        await job_manager.update_progress(inputs['transform_id'], 15.0)
        with ThreadPoolExecutor(max_workers=10) as executor:
            graphs = []
            for chunk in inputs['chunks']:
                try:
                    graph = extract_knowledge_graph([chunk], inputs['ontology_obj'])
                    if graph:
                        graphs.append(graph[0])
                except Exception as e:
                    logger.error(f"Error extracting graph from chunk: {str(e)}")
                    continue
            
        return {"graphs": graphs}

class GraphMergeChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["graphs", "transform_id", "app"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["merged_graphs"]
    
    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """Synchronous call - delegates to async version"""
        return asyncio.run(self._acall(inputs))
    
    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        logger.info('Extracting Generic Graph DONE')
        logger.info('Merging Graphs...')
        job_manager = get_job_manager(inputs['app'])
        await job_manager.update_progress(inputs['transform_id'], 25.0)
        return {"merged_graphs": KnowledgeGraphManager().merge_graphs(inputs['graphs'])}

class DomainGraphChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["merged_graphs", "ontology_str", "transform_id", "app"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["domain_graphs"]
    
    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """Synchronous call - delegates to async version"""
        return asyncio.run(self._acall(inputs))
    
    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        logger.info('Merging Graphs DONE')
        logger.info('Building Domain Graph...')
        job_manager = get_job_manager(inputs['app'])
        await job_manager.update_progress(inputs['transform_id'], 35.0)
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Ensure ontology_str is a list for mapping
            ontology_list = [inputs['ontology_str']] if isinstance(inputs['ontology_str'], str) else inputs['ontology_str']
            domain_graphs = []
            for x in ontology_list:
                try:
                    result = extract_by_ontology(x, inputs['merged_graphs'])
                    if result:
                        domain_graphs.append(result)
                except InstructorRetryException as e:
                    logger.error(f"Validation error in domain extraction: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error in domain extraction: {str(e)}")
                    continue
        
        return {"domain_graphs": domain_graphs}

class MetadataChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["chunks", "ontology_str", "transform_id", "app"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["metadata"]
    
    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """Synchronous call - delegates to async version"""
        return asyncio.run(self._acall(inputs))
    
    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        logger.info('Generating Metadata')
        text = inputs['chunks'][0].page_content
        job_manager = get_job_manager(inputs['app'])
        await job_manager.update_progress(inputs['transform_id'], 40.0)
        try:
            metadata = extract_metadata(inputs['ontology_str'], text)
            return {'metadata': metadata}
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {'metadata': []}

def app():
    return SequentialChain(
        chains=[
            ChunkingChain(),
            GraphExtractionChain(),
            GraphMergeChain(),
            DomainGraphChain(),
            MetadataChain()
        ],
        input_variables=["text", "ontology_str", "ontology_obj", "transform_id", "app"],
        output_variables=["metadata", "domain_graphs"]
    )