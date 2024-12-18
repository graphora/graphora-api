from typing import Dict
from app.services.graph_service import GraphService
from app.utils.logger import logger

class FeedbackAgent:
    def __init__(self):
        self.graph_service = GraphService()
    
    async def process_feedback(self, document_id: str, feedback: Dict) -> bool:
        """Process human feedback for entity and relationship corrections"""
        try:
            # Validate feedback structure and content
            validation_result = self.validate_feedback(feedback)
            if not validation_result:
                logger.warning(f"Invalid feedback format for document {document_id}")
                return False
            
            # Apply feedback to temporary subgraph
            success = await self.graph_service.incorporate_feedback(document_id, feedback)
            if not success:
                logger.error(f"Failed to incorporate feedback for document {document_id}")
                return False
            
            logger.info(f"Successfully processed feedback for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return False
    
    def validate_feedback(self, feedback: Dict) -> bool:
        """Validate feedback structure and content"""
        try:
            # Check for required sections
            if not isinstance(feedback, dict):
                return False
            
            # Validate entity updates if present
            if "entity_updates" in feedback:
                if not isinstance(feedback["entity_updates"], list):
                    return False
                for update in feedback["entity_updates"]:
                    if not all(key in update for key in ["id", "changes"]):
                        return False
                    if not isinstance(update["changes"], dict):
                        return False
            
            # Validate relationship updates if present
            if "relationship_updates" in feedback:
                if not isinstance(feedback["relationship_updates"], list):
                    return False
                for update in feedback["relationship_updates"]:
                    if not all(key in update for key in ["source_id", "target_id", "changes"]):
                        return False
                    if not isinstance(update["changes"], dict):
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating feedback: {str(e)}")
            return False
