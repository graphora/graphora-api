from app.schemas.global_merge import ERState, DbNode, DbEdge, ResolutionStatus
from app.services.global_merge.global_db_connector import DBConnector
from app.services.websocket_manager import WebSocketManager
import logging
from typing import List, Dict, Any
import uuid
import json

logger = logging.getLogger(__name__)

class GraphPersistenceAgent:
    """Agent for persisting graph changes"""

    def __init__(self, prod_db: DBConnector, ws_manager: WebSocketManager = None, session_id: str = None):
        self.prod_db = prod_db
        self.ws_manager = ws_manager
        self.session_id = session_id
        self.IGNORED_PROPERTIES = {'_merged_ids', '_uid_'}

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

    def _get_prod_node(self, node_id: str) -> Dict:
        """Get node from production by ID"""
        query = """
        MATCH (n) WHERE n._uid_ = $node_id
        RETURN n
        """
        with self.prod_db.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            if record:
                node = record["n"]
                return {
                    "id": node["_uid_"],
                    "labels": list(node.labels),
                    "properties": dict(node)
                }
        return None

    def _format_changes(self, staging_node: Dict, prod_node: Dict = None) -> str:
        """Format changes showing clear differences"""
        node_type = staging_node['labels'][0]
        staging_props = staging_node['properties']
        
        if not prod_node:  # New node
            props_text = []
            for key, value in staging_props.items():
                if key not in self.IGNORED_PROPERTIES:
                    props_text.append(f"  + {key}: {value}")
            
            return f"➕ NEW {node_type}:\n" + "\n".join(props_text)
        
        # Compare with existing node
        prod_props = prod_node['properties']
        changes = []
        
        # Find modified and new properties
        for key, new_val in staging_props.items():
            if key in self.IGNORED_PROPERTIES:
                continue
                
            if key not in prod_props:
                changes.append(f"  + Added: {key} = {new_val}")
            elif prod_props[key] != new_val:
                changes.append(f"  ~ Changed: {key}\n    From: {prod_props[key]}\n    To:   {new_val}")
        
        # Find removed properties
        for key in prod_props:
            if key not in staging_props and key not in self.IGNORED_PROPERTIES:
                changes.append(f"  - Removed: {key} (was: {prod_props[key]})")
        
        if not changes:
            return None
            
        return f"✏️ UPDATE {node_type}:\n" + "\n".join(changes)

    async def run(self, state: ERState) -> ERState:
        """Run persistence operations"""
        try:
            for result in state.processed_nodes:
                if result.status == ResolutionStatus.NEEDS_REVIEW:
                    node = result.staging_node
                    node_type = node.labels[0] if node.labels else "Unknown"
                    name = node.properties.get('name', 'Unnamed')
                    
                    # Get existing node if it exists
                    prod_node = None
                    if result.prod_node_id:
                        prod_node = self._get_prod_node(result.prod_node_id)
                    
                    # Build a clear explanation of what needs review
                    msg_parts = []
                    
                    # 1. Header explaining why we need review
                    if any(issue.startswith("Multiple potential matches:") for issue in result.issues):
                        msg_parts.extend([
                            f"⚠️ Multiple Matching Nodes Found for {node_type}: {name}\n",
                            "We found multiple existing nodes that could match this one. Please review the details:"
                        ])
                    elif any(issue.startswith("Low confidence match") for issue in result.issues):
                        msg_parts.extend([
                            f"⚠️ Uncertain Match for {node_type}: {name}\n",
                            f"We found a potential match but we're not very confident (confidence: {result.confidence:.2f}).",
                            "Please review if this is actually the same entity:"
                        ])
                    else:
                        msg_parts.extend([
                            f"⚠️ Property Conflicts in {node_type}: {name}\n",
                            "The following properties have conflicts that need resolution:"
                        ])
                    
                    # 2. Show the differences clearly
                    if prod_node:
                        msg_parts.append("\nExisting Node Properties:")
                        for key, value in prod_node['properties'].items():
                            if key not in self.IGNORED_PROPERTIES:
                                msg_parts.append(f"  {key}: {value}")
                    
                    msg_parts.append("\nNew Node Properties:")
                    for key, value in node.properties.items():
                        if key not in self.IGNORED_PROPERTIES:
                            msg_parts.append(f"  {key}: {value}")
                    
                    # 3. Highlight specific issues
                    if result.issues:
                        msg_parts.append("\nDetected Issues:")
                        for issue in result.issues:
                            if not issue.startswith("Multiple potential matches:"):  # Already shown in header
                                msg_parts.append(f"- {issue}")
                    
                    # 4. Add clear instructions
                    msg_parts.extend([
                        "\nPlease review and choose:",
                        "✅ Approve - Accept the new values",
                        "❌ Reject - Keep the existing values",
                        "✏️ Modify - Manually edit the values"
                    ])
                    
                    review_msg = "\n".join(msg_parts)
                    
                    if self.ws_manager and self.session_id:
                        await self.ws_manager.send_question(
                            self.session_id,
                            question_id=str(uuid.uuid4()),
                            content=review_msg,
                            options=[
                                {"id": "approved", "label": "✅ Approve Changes"},
                                {"id": "rejected", "label": "❌ Reject Changes"},
                                {"id": "modify", "label": "✏️ Modify Changes"}
                            ]
                        )
            
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