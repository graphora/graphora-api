from app.schemas.global_merge import ERState, DbNode, DbEdge
from app.services.global_merge.global_db_connector import DBConnector
import logging
from typing import List, Dict, Any
import uuid
import asyncio
import websockets

logger = logging.getLogger(__name__)

class GraphPersistenceAgent:
    """Agent for persisting merged nodes to the production graph"""

    def __init__(self, prod_db: DBConnector):
        self.prod_db = prod_db
        self.websocket = None

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
            
            # Send changes to WebSocket for user review
            await self._send_changes_to_websocket(changes)
            
            # Wait for user confirmation
            confirmation = await self._wait_for_user_confirmation()
            
            if confirmation != 'yes':
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
                    'node': result.staging_node
                })
            else:
                # Existing node to be updated
                changes.append({
                    'type': 'update',
                    'node': result.staging_node,
                    'prod_id': result.prod_node_id
                })
        
        return changes

    async def _send_changes_to_websocket(self, changes: List[Dict[str, Any]]):
        """Send changes to WebSocket for user review"""
        if not self.websocket:
            self.websocket = await websockets.connect("ws://localhost:8765")
        
        await self.websocket.send(changes)

    async def _wait_for_user_confirmation(self) -> str:
        """Wait for user confirmation via WebSocket"""
        if not self.websocket:
            self.websocket = await websockets.connect("ws://localhost:8765")
        
        confirmation = await self.websocket.recv()
        return confirmation

    def _apply_changes(self, changes: List[Dict[str, Any]]):
        """Apply the collected changes to the production graph"""
        for change in changes:
            try:
                if change['type'] == 'create':
                    self._create_node(change['node'])
                elif change['type'] == 'update':
                    self._update_node(change['node'], change['prod_id'])
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