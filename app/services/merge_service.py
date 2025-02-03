from typing import Dict, Optional
import asyncio
import logging

from app.services.global_merge.langraph import ERPipeline
from app.services.global_merge.human_review import HumanReviewQueue
from app.services.global_merge.staging_extractor import get_subgraph
from app.services.global_merge.global_db_connector import DBConnector
from app.services.websocket_manager import WebSocketManager
from app.schemas.global_merge import ReviewStatus, ERState, ResolutionStatus
from app.schemas.merge_events import MergeAnswer
from app.config import settings
from app.services.ontology_validator import parse_and_validate_yaml
import asyncio
import logging
from fastapi import WebSocket
from typing import List

logger = logging.getLogger(__name__)

class MergeService:
    """Service for handling graph merges"""
    
    def __init__(self):
        logger.info("Initializing MergeService instance")
        self.ws_manager = WebSocketManager()
        self.active_merges: Dict[str, asyncio.Task] = {}
        self.session_states: Dict[str, Dict] = {}
        self.ws_managers: Dict[str, WebSocketManager] = {}
        self._active_sessions: Dict[str, ERState] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session"""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    async def get_session_state(self, session_id: str) -> Optional[ERState]:
        """Get the current state of a merge session"""
        # Check both active sessions and websocket readiness
        if session_id not in self._active_sessions or session_id not in self.ws_managers:
            return None
        return self._active_sessions.get(session_id)

    async def start_merge(self, session_id: str, transform_id: str):
        """Start a new merge process"""
        logger.info(f"Starting merge for session {session_id} with transform {transform_id}")
        
        # Initialize session state
        self.session_states[session_id] = {
            'ws_ready': asyncio.Event(),
            'status': 'initializing'
        }
        
        # Initialize empty state with required fields
        empty_state = ERState(
            staging_nodes=[],  # Required field, initialize as empty list
            staging_edges=[],
            processed_nodes=[],
            review_queue=[],
            errors=[],
            metadata={},
            new_nodes=[],
            updated_nodes=[],
            conflicts=[]
        )
        self._active_sessions[session_id] = empty_state
        
        # Create a new task for the merge process
        task = asyncio.create_task(
            self._run_merge(session_id, transform_id)
        )
        self.active_merges[session_id] = task
        
        return {"status": "started"}

    async def _run_merge(self, session_id: str, transform_id: str):
        """Run the merge process"""
        try:
            # Initialize services and pipeline
            prod_db_conn = DBConnector(settings.NEO4J_URI, settings.NEO4J_PASSWORD)
            review_queue = HumanReviewQueue()
            
            # Get ontology
            ontology_dict = await self._get_ontology(session_id)
            
            # Create pipeline with websocket manager
            pipeline = ERPipeline(
                prod_db=prod_db_conn,
                review_queue=review_queue,
                ontology=ontology_dict,
                ws_manager=self.ws_managers.get(session_id),  # Use session-specific manager
                session_id=session_id
            )
            
            # Get nodes and edges from staging
            nodes, edges = get_subgraph(transform_id)
            
            # Update state with staging nodes and edges
            state = self._active_sessions[session_id]
            state.staging_nodes = nodes
            state.staging_edges = edges
            
            # Run pipeline
            state = await pipeline.process_nodes(nodes, edges)
            
            # Update session state
            self._active_sessions[session_id] = state
            
            # Update status and handle review process
            self.session_states[session_id]['status'] = 'in_review'
            await self._handle_review_process(session_id, state)
            
        except Exception as e:
            logger.error(f"Error in merge process: {str(e)}")
            if session_id in self.session_states:
                self.session_states[session_id]['status'] = 'error'
                self.session_states[session_id]['error'] = str(e)
            if session_id in self.ws_managers:
                await self.ws_managers[session_id].send_error(session_id, str(e))
            raise
            
        finally:
            # Cleanup
            if session_id in self.active_merges:
                del self.active_merges[session_id]

    async def cancel_merge(self, session_id: str):
        """Cancel an ongoing merge process"""
        logger.info(f"Cancelling merge for session {session_id}")
        
        # Cancel the task if it exists
        if session_id in self.active_merges:
            task = self.active_merges[session_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clean up session resources
        await self._cleanup_session(session_id)
        
        # Close WebSocket if it exists
        if session_id in self.ws_managers:
            await self.ws_managers[session_id].close()
        
        return {"status": "cancelled"}

    async def _cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.active_merges:
            del self.active_merges[session_id]
        if session_id in self.session_states:
            del self.session_states[session_id]
        if session_id in self.ws_managers:
            del self.ws_managers[session_id]
        self._active_sessions.pop(session_id, None)
        self._session_locks.pop(session_id, None)

    async def handle_answer(self, answer: MergeAnswer):
        """Handle an answer to a merge question"""
        logger.info(f"Received answer for session {answer.session_id}: {answer.selected_option}")
        
        if answer.session_id not in self.ws_managers:
            logger.error(f"No WebSocket manager found for session {answer.session_id}")
            return
            
        ws_manager = self.ws_managers[answer.session_id]
        await ws_manager.handle_answer(answer)

    async def handle_websocket_connection(self, session_id: str, websocket: WebSocket):
        """Handle new WebSocket connection"""
        logger.info(f"New WebSocket connection for session {session_id}")
        
        # Create WebSocket manager if needed
        if session_id not in self.ws_managers:
            self.ws_managers[session_id] = WebSocketManager()
            
        # Add connection
        self.ws_managers[session_id].add_connection(websocket)
        
        # Initialize empty state for visualization if not exists
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = ERState(
                staging_nodes=[],
                staging_edges=[],
                processed_nodes=[],
                review_queue=[],
                errors=[],
                metadata={},
                new_nodes=[],
                updated_nodes=[],
                conflicts=[]
            )
        
        # Mark WebSocket as ready
        if session_id in self.session_states:
            self.session_states[session_id]['ws_ready'].set()
        else:
            self.session_states[session_id] = {
                'ws_ready': asyncio.Event(),
                'status': 'connected'
            }
            self.session_states[session_id]['ws_ready'].set()
            
        logger.info(f"WebSocket marked as ready for session {session_id}")

        try:
            while True:
                # Wait for messages from the client
                message = await websocket.receive_json()
                
                # Handle different message types
                if message.get("type") == "ANSWER":
                    answer = MergeAnswer(**message.get("payload", {}))
                    await self.handle_answer(answer)
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {str(e)}")
            if websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "ERROR",
                    "payload": {"message": str(e)}
                })
        finally:
            # Clean up if this was the last connection
            if session_id in self.ws_managers:
                self.ws_managers[session_id].remove_connection(websocket)
                if not self.ws_managers[session_id].active_connections:
                    await self.cancel_merge(session_id)

    async def handle_websocket_disconnect(self, session_id: str, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if session_id in self.ws_managers:
            self.ws_managers[session_id].remove_connection(websocket)
            
            # If no more connections, clean up
            if not self.ws_managers[session_id].has_active_connections:
                del self.ws_managers[session_id]
                
                # Don't remove state here to keep visualization available
                if session_id in self.session_states:
                    del self.session_states[session_id]

    def is_websocket_ready(self, session_id: str) -> bool:
        """Check if WebSocket is ready for a session"""
        in_states = session_id in self.session_states
        in_managers = session_id in self.ws_managers
        has_connections = in_managers and self.ws_managers[session_id].has_active_connections
        is_ready = in_states and self.session_states[session_id].get('ws_ready', False)
        
        logger.info(f"WebSocket ready check for {session_id}:")
        logger.info(f"- In session states: {in_states}")
        logger.info(f"- In WS managers: {in_managers}")
        logger.info(f"- Has active connections: {has_connections}")
        logger.info(f"- Is marked ready: {is_ready}")
        
        return in_states and in_managers and has_connections and is_ready

    async def _handle_review_process(self, session_id: str, state: ERState):
        """Handle the review process - keeping existing websocket flow"""
        try:
            # Send initial status through websocket manager
            ws_manager = self.ws_managers.get(session_id)
            if not ws_manager:
                logger.error(f"WebSocket manager not found for session {session_id}")
                return

            # Send initial status
            await ws_manager._broadcast({
                "type": "STATUS",
                "payload": {
                    "total_nodes": len(state.staging_nodes),
                    "processed": len(state.processed_nodes),
                    "needs_review": len([n for n in state.processed_nodes 
                                      if n.status == ResolutionStatus.NEEDS_REVIEW])
                }
            })
            
            # Send conflicts as questions
            for result in state.processed_nodes:
                if result.conflicts:
                    for conflict in result.conflicts:
                        question_id = f"conflict_{result.staging_node.id}"
                        node_type = result.staging_node.labels[0] if result.staging_node.labels else "Node"
                        
                        # Format conflict details
                        conflict_msg = f"🔄 Conflict detected for {node_type}:\n"
                        conflict_msg += f"Type: {conflict.conflict_type}\n"
                        conflict_msg += f"Description: {conflict.description}\n"
                        
                        if conflict.properties_affected:
                            conflict_msg += "\nAffected Properties:\n"
                            for prop, values in conflict.properties_affected.items():
                                conflict_msg += f"- {prop}: {values}\n"
                        
                        # Format suggestions
                        options = []
                        for i, suggestion in enumerate(conflict.suggestions):
                            option_id = f"suggestion_{i}"
                            option_label = f"✅ {suggestion.description}"
                            options.append({
                                "id": option_id,
                                "label": option_label,
                                "suggestion": suggestion.dict()
                            })
                        
                        # Add reject option
                        options.append({
                            "id": "reject",
                            "label": "❌ Reject All Changes"
                        })
                        
                        # Send question through websocket
                        await ws_manager._broadcast({
                            "type": "QUESTION",
                            "payload": {
                                "questionId": question_id,
                                "content": conflict_msg,
                                "options": options,
                                "nodeId": result.staging_node.id,
                                "conflictType": conflict.conflict_type
                            }
                        })
            
            # Continue with existing review flow...
            pending_reviews = await HumanReviewQueue().get_pending_reviews()
            
            for review in pending_reviews:
                # Check for cancellation
                if session_id not in self.active_merges:
                    break
                    
                try:
                    await self._handle_review(session_id, review, ws_manager, HumanReviewQueue())
                except Exception as e:
                    logger.error(f"Error handling review: {str(e)}")
                    continue
            
            # Send completion status
            await ws_manager._broadcast({
                "type": "PROGRESS",
                "payload": {
                    "progress": 100,
                    "current_step": "Merge completed successfully"
                }
            })
            
        except Exception as e:
            logger.error(f"Error in review process: {str(e)}")
            if session_id in self.ws_managers:
                await self.ws_managers[session_id].send_error(session_id, str(e))
            raise

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

    async def _handle_review(self, session_id: str, review, ws_manager, review_queue):
        """Handle a single review"""
        try:
            # Assign review
            await review_queue.assign_review(review.id, "system")
            
            # Format review message
            if hasattr(review, 'changes') and isinstance(review.changes, list):
                changes_text = "\n\n".join(self._format_change_message(change) for change in review.changes)
                review_msg = f"📋 Please review the following changes:\n\n{changes_text}"
            else:
                node = review.staging_node
                node_type = node.labels[0] if node.labels else "Unknown"
                props = node.properties if hasattr(node, 'properties') else {}
                review_msg = f"📋 Review changes for {node_type}:\n{self._format_node_details({'properties': props})}"
            
            # Send question
            await ws_manager._broadcast({
                "type": "QUESTION",
                "payload": {
                    "question_id": review.id,
                    "content": review_msg,
                    "options": [
                        {"id": "approved", "label": "✅ Approve Changes"},
                        {"id": "rejected", "label": "❌ Reject Changes"},
                        {"id": "modify", "label": "✏️ Modify Changes"}
                    ]
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing review {review.id}: {str(e)}")
            await ws_manager.send_error(session_id, f"Error processing review: {str(e)}")
            raise

    async def _get_ontology(self, session_id):
        """Get ontology"""
        try:
            # Get ontology
            with open(f"{settings.ONTOLOGY_DIR}/{session_id}.yaml", 'r') as f:
                ontology_yaml = f.read()
            ontology_dict = parse_and_validate_yaml(ontology_yaml)
        except Exception as e:
            raise ValueError(f"Failed to load ontology: {str(e)}")
        return ontology_dict
