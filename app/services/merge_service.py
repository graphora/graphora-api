from typing import Dict, Optional, Any
import asyncio
import logging
from uu import Error
from uuid import uuid4

from app.services.websocket_manager import WebSocketManager
from app.schemas.merge_events import MergeAnswer
from app.services.global_merge.langraph import ERPipeline 
from app.services.global_merge.global_db_connector import DBConnector
from app.services.global_merge.staging_extractor import get_subgraph
from app.services.ontology_validator import parse_and_validate_yaml
from app.services.global_merge.human_review import HumanReviewQueue
from app.schemas.global_merge import ReviewStatus
from app.config import settings

logger = logging.getLogger(__name__)

class MergeService:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize instance attributes
            cls._instance.active_merges = {}
            cls._instance.ws_managers = {}
            cls._instance.session_states = {}
        return cls._instance

    def __init__(self):
        # Skip initialization if already done
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        self.active_merges: Dict[str, asyncio.Task] = {}
        self.ws_managers: Dict[str, WebSocketManager] = {}
        self.session_states: Dict[str, Dict] = {}

    def set_websocket_manager(self, manager: WebSocketManager, session_id: str):
        """Set the WebSocket manager for this service"""
        self.ws_managers[session_id] = manager
        if session_id in self.session_states:
            self.session_states[session_id]['ws_ready'].set()

    async def start_merge(self, session_id: str, transform_id: str):
        """Start a new merge process"""
        # Cancel any existing merge for this session
        if session_id in self.active_merges:
            await self.cancel_merge(session_id)

        # Initialize session state
        self.session_states[session_id] = {
            'transform_id': transform_id,
            'ws_ready': asyncio.Event()
        }

        # Create and store the merge task
        merge_task = asyncio.create_task(self._run_merge(session_id, transform_id))
        self.active_merges[session_id] = merge_task

    async def cancel_merge(self, session_id: str):
        """Cancel an ongoing merge process"""
        if session_id in self.active_merges:
            self.active_merges[session_id].cancel()
            try:
                await self.active_merges[session_id]
            except asyncio.CancelledError:
                pass
            finally:
                del self.active_merges[session_id]
                if session_id in self.session_states:
                    del self.session_states[session_id]
                if session_id in self.ws_managers:
                    del self.ws_managers[session_id]

    async def handle_answer(self, answer: MergeAnswer):
        """Handle an answer to a merge question"""
        # Implementation needed
        pass

    async def _run_merge(self, session_id: str, transform_id: str):
        """Run the merge process"""
        try:
            # Wait for WebSocket connection to be ready
            try:
                await asyncio.wait_for(self.session_states[session_id]['ws_ready'].wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.error(f"WebSocket connection not ready for session {session_id}")
                return
            except KeyError:
                logger.error(f"Session state not found for session {session_id}")
                return

            ws_manager = self.ws_managers.get(session_id)
            if not ws_manager:
                raise RuntimeError("WebSocket manager not initialized")

            # Send initial progress
            await ws_manager.send_progress(
                session_id,
                progress=0,
                current_step="Starting merge process"
            )
            
            try:
                # Get ontology
                with open(f"{settings.ONTOLOGY_DIR}/{session_id}.yaml", 'r') as f:
                    ontology_yaml = f.read()
                ontology_dict = parse_and_validate_yaml(ontology_yaml)
            except Exception as e:
                raise ValueError(f"Failed to load ontology for graph {session_id}: {str(e)}")

            # Update progress - Ontology loaded
            await ws_manager.send_progress(
                session_id,
                progress=20,
                current_step="Ontology loaded, initializing pipeline"
            )
                
            # Initialize pipeline components
            prod_db_conn = DBConnector(settings.NEO4J_URI, settings.NEO4J_PASSWORD)
            review_queue = HumanReviewQueue()
            pipeline = ERPipeline(prod_db_conn, review_queue, ontology_dict)
            
            # Get nodes from source graph using existing implementation
            nodes, _ = get_subgraph(transform_id)
            if not nodes:
                raise ValueError(f"No nodes found in graph {transform_id}")
                
            # Update progress - Graph loaded
            await ws_manager.send_progress(
                session_id,
                progress=40,
                current_step="Graph loaded, starting entity resolution"
            )
                
            # Process nodes through pipeline
            state = await pipeline.process_nodes(nodes)
            
            # Update progress - Processing complete
            await ws_manager.send_progress(
                session_id,
                progress=70,
                current_step="Entity resolution complete, handling reviews"
            )
            
            # Handle any pending reviews
            pending_reviews = await review_queue.get_pending_reviews()
            for review in pending_reviews:
                # Check for cancellation
                if session_id not in self.active_merges:
                    return
                    
                # Assign the review to system first
                await review_queue.assign_review(review.id, "system")
                
                await ws_manager.send_question(
                    session_id,
                    question_id=review.id,
                    content=f"Review required for node {review.staging_node.id}",
                    options=[
                        {"id": "accept", "label": "Accept changes"},
                        {"id": "reject", "label": "Reject changes"},
                        {"id": "modify", "label": "Modify changes"}
                    ]
                )
                
                # For now, we'll auto-accept after a delay
                # This should be replaced with actual user interaction handling
                await asyncio.sleep(1)
                await review_queue.submit_review(
                    review.id,
                    "system",
                    ReviewStatus.APPROVED
                )
            
            # Final progress update
            await ws_manager.send_progress(
                session_id,
                progress=100,
                current_step="Merge completed successfully"
            )
            
        except asyncio.CancelledError:
            logger.info(f"Merge cancelled for session {session_id}")
            if session_id in self.ws_managers:
                await self.ws_managers[session_id].send_error(session_id, "Merge process cancelled")
            raise
            
        except Exception as e:
            logger.error(f"Error in merge process: {str(e)}")
            if session_id in self.ws_managers:
                await self.ws_managers[session_id].send_error(session_id, str(e))
            raise
            
        finally:
            if session_id in self.active_merges:
                del self.active_merges[session_id]
                if session_id in self.session_states:
                    del self.session_states[session_id]
