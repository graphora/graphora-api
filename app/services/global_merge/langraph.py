from langgraph.graph import StateGraph, START
from app.schemas.global_merge import ERState, DbNode
from app.services.global_merge.staging_extractor import get_subgraph
from app.services.global_merge.human_review import HumanReviewQueue
from app.services.global_merge.global_db_connector import DBConnector
from app.services.global_merge.agents.disambiguation import DisambiguationAgent
from app.services.global_merge.agents.property_conflict import PropertyConflictAgent
from app.services.global_merge.agents.human_review import HumanReviewAgent
from app.services.global_merge.agents.graph_persistence import GraphPersistenceAgent
from app.config import settings
from typing import List, Dict, Any, Callable, Awaitable, Union
import asyncio
import logging

logger = logging.getLogger(__name__)

class ERPipeline:
    """Main Entity Resolution Pipeline"""

    def __init__(self, prod_db: DBConnector, 
                 review_queue: HumanReviewQueue, ontology: dict):
        self.prod_db = prod_db
        self.review_queue = review_queue
        self.ontology = ontology

        # Initialize agents
        self.disambiguation_agent = DisambiguationAgent(prod_db, review_queue, ontology)
        self.property_agent = PropertyConflictAgent(prod_db, review_queue)
        self.review_agent = HumanReviewAgent(review_queue)
        self.persistence_agent = GraphPersistenceAgent(prod_db)

    async def process_nodes(self, nodes: List[DbNode]) -> ERState:
        """Process a batch of nodes through the pipeline"""
        try:
            # Initialize state
            state = ERState(staging_nodes=nodes)
            
            # Run disambiguation
            state = self.disambiguation_agent.run(state)
            
            # Run property resolution
            state = self.property_agent.run(state)
            
            # Run human review (async)
            state = await self.review_agent.run(state)
            
            # Run persistence (async)
            state = await self.persistence_agent.run(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing nodes: {str(e)}")
            raise

async def run_pipeline(staging_label:str, ontology_dict: dict) -> ERState:
  prod_db_conn = DBConnector(settings.NEO4J_URI, settings.NEO4J_PASSWORD)
  review_queue = HumanReviewQueue()
  pipeline = ERPipeline(prod_db_conn, review_queue, ontology_dict)

  nodes, _ = get_subgraph(staging_label)

  return await pipeline.process_nodes(nodes)