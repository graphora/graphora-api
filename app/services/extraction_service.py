from typing import List, Dict, Optional, Tuple
from datetime import datetime
from instructor import patch
from openai import OpenAI
from app.utils.logger import logger
from app.config import settings
from app.schemas.extraction import (
    ExtractedEntity, ExtractedRelationship,
    EntityExtractionResponse, RelationshipExtractionResponse
)

class ExtractionService:
    def __init__(self):
        self.temp_graphs = {}  # Store temporary subgraphs for review
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client with instructor patch"""
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not set. Entity extraction will be unavailable.")
            self.client = None
            return
        
        try:
            base_client = OpenAI(
                api_key=settings.OPENAI_API_KEY
            )
            self.client = patch(base_client)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
    
    async def extract_entities(self, text: str) -> Tuple[List[Dict], str]:
        """Extract entities from text using OpenAI"""
        if not self.client:
            error_msg = "OpenAI client not initialized. Cannot perform extraction."
            logger.error(error_msg)
            return [], error_msg

        try:
            logger.info("Starting entity extraction")
            chunks = self._split_text(text, max_length=1000)
            all_entities = []
            status = "Processing entities..."
            
            for i, chunk in enumerate(chunks):
                try:
                    completion = await self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        response_model=EntityExtractionResponse,
                        max_retries=2,
                        messages=[
                            {
                                "role": "system",
                                "content": """You are an expert entity extraction system. Extract named entities from the given text.
                                Focus on: PERSON (individual names), ORGANIZATION (company/group names), 
                                LOCATION (places), DATE (temporal references), EVENT (significant occurrences)."""
                            },
                            {
                                "role": "user",
                                "content": f"Extract entities from this text:\n\n{chunk}"
                            }
                        ]
                    )
                    
                    if completion and hasattr(completion, 'entities'):
                        chunk_entities = [
                            {
                                'id': str(len(all_entities) + idx),
                                'type': entity.type,
                                'value': entity.value,
                                'confidence': entity.confidence
                            }
                            for idx, entity in enumerate(completion.entities)
                        ]
                        all_entities.extend(chunk_entities)
                        status = f"Processed {i+1}/{len(chunks)} chunks"
                        logger.info(status)
                    else:
                        logger.warning(f"No entities found in chunk {i+1}")
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {i}: {str(chunk_error)}")
                    continue
            
            if not all_entities:
                return [], "Warning: No entities were extracted"
            
            logger.info(f"Successfully extracted {len(all_entities)} entities")
            return all_entities, status

        except Exception as e:
            error_msg = f"Error in entity extraction: {str(e)}"
            logger.error(error_msg)
            return [], error_msg
    
    async def extract_relationships(self, entities: List[Dict]) -> Tuple[List[Dict], str]:
        """Extract relationships between entities using OpenAI"""
        if not self.client:
            error_msg = "OpenAI client not initialized. Cannot perform extraction."
            logger.error(error_msg)
            return [], error_msg

        if not entities:
            return [], "No entities provided for relationship extraction"

        try:
            logger.info("Starting relationship extraction")
            relationships = []
            status = "Analyzing relationships..."
            
            # Process entities in batches to avoid token limits
            batches = [entities[i:i+5] for i in range(0, len(entities), 5)]
            
            for i, batch in enumerate(batches):
                try:
                    completion = await self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        response_model=RelationshipExtractionResponse,
                        max_retries=2,
                        messages=[
                            {
                                "role": "system",
                                "content": """You are an expert relationship extraction system. 
                                Identify meaningful relationships between entities, focusing on:
                                - WORKS_FOR (employment/affiliation)
                                - LOCATED_IN (physical location)
                                - PART_OF (membership/composition)
                                - ASSOCIATED_WITH (general connection)
                                - INTERACTS_WITH (direct interaction)"""
                            },
                            {
                                "role": "user",
                                "content": f"Analyze these entities and identify any relationships between them:\n\n{batch}"
                            }
                        ]
                    )
                    
                    if completion and hasattr(completion, 'relationships'):
                        batch_relationships = [
                            {
                                'source_id': rel.source_id,
                                'target_id': rel.target_id,
                                'type': rel.type,
                                'confidence': rel.confidence
                            }
                            for rel in completion.relationships
                        ]
                        relationships.extend(batch_relationships)
                        status = f"Processed relationships: {i+1}/{len(batches)} batches"
                        logger.info(status)
                    else:
                        logger.warning(f"No relationships found in batch {i+1}")
                except Exception as batch_error:
                    logger.error(f"Error processing batch {i}: {str(batch_error)}")
                    continue
            
            if not relationships:
                return [], "Warning: No relationships were extracted"
            
            logger.info(f"Successfully extracted {len(relationships)} relationships")
            return relationships, status

        except Exception as e:
            error_msg = f"Error in relationship extraction: {str(e)}"
            logger.error(error_msg)
            return [], error_msg
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    async def create_temp_subgraph(self, document_id: str, content: str, entities: List[Dict], relationships: List[Dict]) -> str:
        """Create a temporary subgraph for user review"""
        try:
            self.temp_graphs[document_id] = {
                "content": content,
                "entities": entities,
                "relationships": relationships,
                "status": "pending_review",
                "created_at": datetime.now().isoformat()
            }
            logger.info(f"Created temporary subgraph for document {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Error creating temporary subgraph: {str(e)}")
            return ""  # Return empty string instead of None for type safety
    
    async def get_temp_subgraph(self, document_id: str) -> Optional[Dict]:
        """Retrieve a temporary subgraph for review"""
        return self.temp_graphs.get(document_id)
    
    async def process_feedback(self, document_id: str, feedback: Dict) -> bool:
        """Process feedback for a temporary subgraph"""
        try:
            if document_id not in self.temp_graphs:
                logger.error(f"No temporary graph found for document {document_id}")
                return False
            
            temp_graph = self.temp_graphs[document_id]
            
            # Apply feedback updates
            if "entity_updates" in feedback:
                self._apply_entity_updates(temp_graph, feedback["entity_updates"])
            
            if "relationship_updates" in feedback:
                self._apply_relationship_updates(temp_graph, feedback["relationship_updates"])
            
            temp_graph["status"] = "updated_from_feedback"
            logger.info(f"Applied feedback to temporary graph for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return False
    
    def _apply_entity_updates(self, temp_graph: Dict, updates: List[Dict]):
        """Apply user feedback updates to entities"""
        for update in updates:
            entity_id = update.get("id")
            for i, entity in enumerate(temp_graph["entities"]):
                if entity["id"] == entity_id:
                    temp_graph["entities"][i].update(update)
                    break
    
    def _apply_relationship_updates(self, temp_graph: Dict, updates: List[Dict]):
        """Apply user feedback updates to relationships"""
        for update in updates:
            rel_id = (update.get("source_id"), update.get("target_id"))
            for i, rel in enumerate(temp_graph["relationships"]):
                if (rel["source_id"], rel["target_id"]) == rel_id:
                    temp_graph["relationships"][i].update(update)
                    break
