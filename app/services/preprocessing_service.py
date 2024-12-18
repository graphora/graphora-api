from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from app.utils.logger import logger
from app.services.extraction_service import ExtractionService
from app.services.graph_service import GraphService

class PreprocessingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class PreprocessingStep(Enum):
    INIT = "initialization"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    GRAPH_CREATION = "graph_creation"
    COMPLETE = "complete"

class PreprocessingService:
    def __init__(self):
        self.extraction_service = ExtractionService()
        self.graph_service = GraphService()
        self._processing_status = {}
        
    def _update_status(self, document_id: str, step: PreprocessingStep, 
                      status: PreprocessingStatus, message: str = ""):
        """Update status for a document's preprocessing step"""
        if document_id not in self._processing_status:
            self._processing_status[document_id] = {
                "started_at": datetime.now().isoformat(),
                "current_step": step.value,
                "steps": {}
            }
        
        self._processing_status[document_id]["steps"][step.value] = {
            "status": status.value,
            "updated_at": datetime.now().isoformat(),
            "message": message
        }
        self._processing_status[document_id]["current_step"] = step.value
        
        logger.info(f"Document {document_id} - {step.value}: {status.value} - {message}")

    def get_status(self, document_id: str) -> Optional[Dict]:
        """Get current preprocessing status for a document"""
        return self._processing_status.get(document_id)

    async def preprocess_document(self, content: str, document_id: str) -> Tuple[bool, str]:
        """
        Preprocess a document through defined steps with status tracking
        Returns: (success: bool, message: str)
        """
        try:
            # Step 1: Initialization
            self._update_status(document_id, PreprocessingStep.INIT, 
                              PreprocessingStatus.IN_PROGRESS)
            
            if not content.strip():
                self._update_status(document_id, PreprocessingStep.INIT, 
                                  PreprocessingStatus.FAILED, "Empty document content")
                return False, "Empty document content"
            
            self._update_status(document_id, PreprocessingStep.INIT, 
                              PreprocessingStatus.COMPLETED, "Document initialized")

            # Step 2: Entity Extraction
            self._update_status(document_id, PreprocessingStep.ENTITY_EXTRACTION, 
                              PreprocessingStatus.IN_PROGRESS)
            
            entities, entity_status = await self.extraction_service.extract_entities(content)
            if not entities:
                self._update_status(document_id, PreprocessingStep.ENTITY_EXTRACTION, 
                                  PreprocessingStatus.FAILED, f"Entity extraction failed: {entity_status}")
                return False, f"Entity extraction failed: {entity_status}"
            
            self._update_status(document_id, PreprocessingStep.ENTITY_EXTRACTION, 
                              PreprocessingStatus.COMPLETED, 
                              f"Extracted {len(entities)} entities")

            # Step 3: Relationship Extraction
            self._update_status(document_id, PreprocessingStep.RELATIONSHIP_EXTRACTION, 
                              PreprocessingStatus.IN_PROGRESS)
            
            relationships, rel_status = await self.extraction_service.extract_relationships(entities)
            if not relationships and "Warning" not in rel_status:
                self._update_status(document_id, PreprocessingStep.RELATIONSHIP_EXTRACTION, 
                                  PreprocessingStatus.FAILED, 
                                  f"Relationship extraction failed: {rel_status}")
                return False, f"Relationship extraction failed: {rel_status}"
            
            self._update_status(document_id, PreprocessingStep.RELATIONSHIP_EXTRACTION, 
                              PreprocessingStatus.COMPLETED, 
                              f"Extracted {len(relationships)} relationships")

            # Step 4: Create Temporary Subgraph
            self._update_status(document_id, PreprocessingStep.GRAPH_CREATION, 
                              PreprocessingStatus.IN_PROGRESS)
            
            success = await self.extraction_service.create_temp_subgraph(
                document_id, content, entities, relationships)
            
            if not success:
                self._update_status(document_id, PreprocessingStep.GRAPH_CREATION, 
                                  PreprocessingStatus.FAILED, 
                                  "Failed to create temporary subgraph")
                return False, "Failed to create temporary subgraph"
            
            self._update_status(document_id, PreprocessingStep.GRAPH_CREATION, 
                              PreprocessingStatus.COMPLETED, 
                              "Temporary subgraph created successfully")

            # Step 5: Mark Complete
            self._update_status(document_id, PreprocessingStep.COMPLETE, 
                              PreprocessingStatus.COMPLETED, 
                              "Preprocessing completed successfully")
            
            return True, "Preprocessing completed successfully"

        except Exception as e:
            error_msg = f"Error during preprocessing: {str(e)}"
            current_step = self._processing_status.get(document_id, {}).get("current_step", 
                                                                          PreprocessingStep.INIT.value)
            self._update_status(document_id, PreprocessingStep(current_step), 
                              PreprocessingStatus.FAILED, error_msg)
            logger.error(error_msg)
            return False, error_msg
