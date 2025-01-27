from app.schemas.global_merge import ERState, DbNode, DbEdge
from app.services.global_merge.global_db_connector import DBConnector
from app.services.websocket_manager import WebSocketManager
import logging
from typing import List, Dict, Any
import uuid
import json

logger = logging.getLogger(__name__)

class GraphPersistenceAgent:
    """Agent for persisting merged nodes to the production graph"""

    def __init__(self, prod_db: DBConnector, ws_manager: WebSocketManager = None, session_id: str = None):
        self.prod_db = prod_db
        self.ws_manager = ws_manager
        self.session_id = session_id

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

    async def run(self, state: ERState) -> ERState:
        """Run the persistence workflow"""
        try:
            # Collect changes to be made
            changes = self._collect_changes(state)
            
            if not changes:
                logger.info("No changes to persist")
                return state
                
            # Log the changes for review
            logger.info("Changes to be persisted:")
            for change in changes:
                logger.info(f"- {change}")
            
            if self.ws_manager and self.session_id:
                # Format review message
                changes_text = "\n\n".join(self._format_change_message(change) for change in changes)
                review_msg = f"📋 Please review the following changes:\n\n{changes_text}"
                
                # Send changes to WebSocket for user review
                await self.ws_manager.send_question(
                    self.session_id,
                    question_id=str(uuid.uuid4()),
                    content=review_msg,
                    options=[
                        {"id": "approved", "label": "✅ Approve Changes"},
                        {"id": "rejected", "label": "❌ Reject Changes"}
                    ]
                )
                
                # Wait for user confirmation
                confirmation = await self._wait_for_user_confirmation()
                
                if confirmation != 'approved':
                    logger.info("User cancelled persistence")
                    return state
            
            # Apply changes to production graph
            self._apply_changes(changes)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in persistence agent: {str(e)}")
            state.errors.append(f"Persistence error: {str(e)}")
            return state

    def _collect_changes(self, state: ERState) -> List[Dict[str, Any]]:
        """Collect all changes that need to be made to the production graph"""
        changes = []
        
        for result in state.processed_nodes:
            if not result.prod_node_id:
                # New node to be created
                changes.append({
                    'type': 'create',
                    'node': result.staging_node.model_dump()
                })
            else:
                # Existing node to be updated
                changes.append({
                    'type': 'update',
                    'node': result.staging_node.model_dump(),
                    'prod_id': result.prod_node_id
                })
        
        return changes

    async def _send_changes_to_websocket(self, changes: List[Dict[str, Any]]):
        """Send changes to WebSocket for user review"""
        if not self.ws_manager or not self.session_id:
            logger.warning("WebSocket manager or session_id not available, skipping user confirmation")
            return

        await self.ws_manager.send_question(
            self.session_id,
            question_id=str(uuid.uuid4()),
            content=f"Review and confirm graph changes:\n{json.dumps(changes, indent=2)}",
            options=[
                {"id": "approved", "label": "Approve Changes"},
                {"id": "rejected", "label": "Reject Changes"}
            ]
        )

    async def _wait_for_user_confirmation(self) -> str:
        """Wait for user confirmation via WebSocket"""
        if not self.ws_manager or not self.session_id:
            logger.warning("WebSocket manager or session_id not available, skipping user confirmation")
            return 'approved'  # Auto-approve if no WebSocket

        response = await self.ws_manager.wait_for_response(self.session_id)
        return response.get('selected_option', 'rejected')

    def _apply_changes(self, changes: List[Dict[str, Any]]):
        """Apply the collected changes to the production graph"""
        for change in changes:
            try:
                if change['type'] == 'create':
                    self._create_node(DbNode(**change['node']))
                elif change['type'] == 'update':
                    self._update_node(DbNode(**change['node']), change['prod_id'])
            except Exception as e:
                logger.error(f"Error applying change {change}: {str(e)}")
                raise

    def _create_node(self, node: DbNode):
        """Create a new node in the production graph"""
        # Generate a new UUID for the node
        node_id = str(uuid.uuid4())
        
        # Prepare Cypher query
        labels = ':'.join(node.labels)
        props = {**node.properties, '_uid_': node_id}
        
        query = f"""
        CREATE (n:{labels})
        SET n = $props
        RETURN n
        """
        
        # Execute query
        self.prod_db.execute_query(query, {'props': props})

    def _update_node(self, node: DbNode, prod_id: str):
        """Update an existing node in the production graph"""
        # Prepare Cypher query
        props = {**node.properties, '_uid_': prod_id}
        
        query = """
        MATCH (n)
        WHERE n._uid_ = $prod_id
        SET n = $props
        RETURN n
        """
        
        # Execute query
        self.prod_db.execute_query(query, {
            'prod_id': prod_id,
            'props': props
        })