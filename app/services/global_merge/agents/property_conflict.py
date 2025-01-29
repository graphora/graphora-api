from app.schemas.global_merge import DbNode, ResolutionStatus, ERState
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class PropertyConflictAgent:
    """Agent for handling property conflicts between staging and production nodes"""
    
    def __init__(self, prod_db, review_queue):
        self.prod_db = prod_db
        self.review_queue = review_queue
        self.IGNORED_PROPERTIES = {'_merged_ids', '_uid_'}

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

    def _format_property_conflicts(self, staging_props: Dict, prod_props: Dict) -> List[Dict]:
        """Format property conflicts in a user-friendly way"""
        conflicts = []
        
        # Compare properties and find differences
        for key in set(staging_props) | set(prod_props):
            if key in self.IGNORED_PROPERTIES:
                continue
                
            staging_val = staging_props.get(key)
            prod_val = prod_props.get(key)
            
            if staging_val != prod_val:
                conflicts.append({
                    "property": key,
                    "staging_value": staging_val,
                    "prod_value": prod_val,
                    "change_type": "modified" if key in prod_props else "added"
                })
                
        return conflicts

    def _format_conflict_message(self, node: Dict, conflicts: List[Dict]) -> str:
        """Format conflicts in a user-friendly message"""
        node_type = node.get('labels', ['Unknown'])[0]
        msg_parts = [f" Conflicts in {node_type}:"]
        
        # Add node identifier
        if node.get('properties', {}).get('name'):
            msg_parts.append(f"Name: {node['properties']['name']}")
            
        # Format each conflict
        for conflict in conflicts:
            prop_name = conflict['property'].replace('_', ' ').title()
            if conflict['change_type'] == 'modified':
                msg_parts.append(f"\n {prop_name}:")
                msg_parts.append(f"  - Existing: {conflict['prod_value']}")
                msg_parts.append(f"  - New: {conflict['staging_value']}")
            else:  # added
                msg_parts.append(f"\n New Property - {prop_name}:")
                msg_parts.append(f"  - Value: {conflict['staging_value']}")
                
        return '\n'.join(msg_parts)

    async def run(self, state: ERState) -> ERState:
        """Run property conflict checks"""
        try:
            for result in state.processed_nodes:
                if result.status == ResolutionStatus.RESOLVED:
                    # Get production node
                    prod_node = self._get_prod_node(result.prod_node_id)
                    if not prod_node:
                        continue
                        
                    # Find and format conflicts
                    conflicts = self._format_property_conflicts(
                        result.staging_node.properties,
                        prod_node['properties']
                    )
                    
                    if conflicts:
                        # Create conflict message
                        conflict_msg = self._format_conflict_message(
                            {"labels": result.staging_node.labels, "properties": result.staging_node.properties},
                            conflicts
                        )
                        
                        # Add to review queue
                        result.status = ResolutionStatus.NEEDS_REVIEW
                        result.issues = [conflict_msg]
                        await self.review_queue.enqueue(result)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in property conflict agent: {str(e)}")
            raise