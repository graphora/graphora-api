from langgraph.graph import StateGraph, START
from app.schemas.global_merge import ERState, DbNode, DbEdge
from app.services.global_merge.staging_extractor import get_subgraph
from app.services.global_merge.human_review import HumanReviewQueue
from app.services.global_merge.global_db_connector import DBConnector
from app.services.global_merge.agents.disambiguation import DisambiguationAgent
from app.services.global_merge.agents.property_conflict import PropertyConflictAgent
from app.services.global_merge.agents.human_review import HumanReviewAgent
from app.services.global_merge.agents.graph_persistence import GraphPersistenceAgent
from app.services.websocket_manager import WebSocketManager
from app.config import settings
from typing import List, Dict, Any, Callable, Awaitable, Union
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)

class ERPipeline:
    """Main Entity Resolution Pipeline"""
    
    def __init__(self, prod_db: DBConnector, 
                 review_queue: HumanReviewQueue, 
                 ontology: dict,
                 ws_manager: WebSocketManager = None,
                 session_id: str = None):
        self.prod_db = prod_db
        self.review_queue = review_queue
        self.ontology = ontology
        self.ws_manager = ws_manager
        self.session_id = session_id
        
        # Initialize agents
        self.disambiguation_agent = DisambiguationAgent(prod_db, review_queue, ontology)
        self.property_agent = PropertyConflictAgent(prod_db, review_queue)
        self.review_agent = HumanReviewAgent(review_queue)
        self.persistence_agent = GraphPersistenceAgent(prod_db, ws_manager, session_id)

    def _format_node_details(self, node: Dict) -> str:
        """Format node details in a user-friendly way"""
        props = node.get('properties', {})
        details = []
        
        if props.get('name'):
            details.append(f"Name: {props['name']}")
        if props.get('description'):
            details.append(f"Description: {props['description']}")
            
        # Add other relevant properties
        for key, value in props.items():
            if key not in ['name', 'description', '_merged_ids'] and not key.startswith('_'):
                details.append(f"{key.replace('_', ' ').title()}: {value}")
                
        return '\n'.join(details)

    def _format_change_message(self, change: Dict) -> str:
        """Format a single change in a user-friendly way"""
        node = change.get('node', {})
        change_type = change.get('type', '').upper()
        node_type = node.get('labels', ['Unknown'])[0]
        
        icon = {
            'CREATE': '➕',
            'UPDATE': '✏️',
            'DELETE': '🗑️'
        }.get(change_type, '🔹')
        
        return f"{icon} {change_type} {node_type}:\n{self._format_node_details(node)}"

    async def process_nodes(self, nodes: List[DbNode], edges: List[DbEdge]) -> ERState:
        """Process a batch of nodes through the pipeline"""
        try:
            # Initialize state
            state = ERState(staging_nodes=nodes, staging_edges=edges)
            
            # Run disambiguation
            state = self.disambiguation_agent.run(state)
            
            # Run property resolution (async)
            state = await self.property_agent.run(state)
            
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

  nodes, edges = get_subgraph(staging_label)

  return await pipeline.process_nodes(nodes, edges)