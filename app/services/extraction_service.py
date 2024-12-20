from typing import List, Dict, Type, Any, Tuple
from pydantic import BaseModel
from app.utils.logger import logger
from dataclasses import dataclass
from app.schemas.extraction import (
    ExtractedRelationship
)
from app.schemas.document import (
    Entity, 
    Relationship
)
from app.utils.llm_client_service import LLMClientService
from app.services.ontology_generator_service import Neo4jOntology, create_extraction_models
import spacy
import time

@dataclass
class Chunk:
    id: str
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict

@dataclass
class ExtractedData:
    entities: List[Entity]
    relationships: List[Relationship]
    chunk_id: str

class ExtractionService:
    def __init__(self):
        self.llm_service = LLMClientService()
        self.nlp = spacy.load("en_core_web_sm")

    async def extract(self, content: str, ontology: Neo4jOntology) -> ExtractedData:
        chunks = self._create_chunks(content)
        logger.info(f"Created {len(chunks)} chunks from document")

        # Create dynamic models
        models = create_extraction_models(ontology)
        # Process each chunk
        all_extractions = []
        for chunk in chunks:
            time.sleep(5)
            extracted_data = await self._process_chunk(chunk, models)
            all_extractions.append(extracted_data)
        
        # Combine extractions
        return self._combine_extractions(all_extractions)
    
    def generate_extraction_prompt(self, text: str) -> str:
        """Generate a prompt for the LLM to extract entities and relationships."""

        prompt = f"""Extract entities and relationships from the following text:

        Text to analyze:
        {text}

        Extract all entities and relationships that match the response definitions provided. For each extraction:
        1. Identify entities that match the node types defined above
        2. Include all required properties and any optional properties found in the text
        3. Identify relationships between entities following the patterns defined
        4. Ensure high confidence scores (>0.8) only for clear matches
        5. Include example matches that follow the patterns shown in the examples

        Format entities and relationships exactly according to the types and properties defined."""

        return prompt
    
    def _create_chunks(self, text: str) -> List[Chunk]:
        """Create intelligent chunks from text content."""
        chunks = []
        chunk_id = 0
        
        # Process with spaCy for linguistic boundaries
        doc = self.nlp(text)
        
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        # Constants for chunking
        MAX_CHUNK_LENGTH = 4000  # Characters
        MIN_CHUNK_LENGTH = 100
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)
            
            # Check if adding this sentence would exceed max length
            if current_length + sent_length > MAX_CHUNK_LENGTH and current_length >= MIN_CHUNK_LENGTH:
                # Create new chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    id=f"chunk_{chunk_id}",
                    text=chunk_text,
                    start_pos=chunk_start,
                    end_pos=chunk_start + len(chunk_text),
                    metadata={
                        "sentences": len(current_chunk),
                        "length": current_length
                    }
                ))
                chunk_id += 1
                current_chunk = []
                current_length = 0
                chunk_start = sent.start_char
            
            current_chunk.append(sent_text)
            current_length += sent_length
        
        # Add final chunk if there's content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                start_pos=chunk_start,
                end_pos=chunk_start + len(chunk_text),
                metadata={
                    "sentences": len(current_chunk),
                    "length": current_length
                }
            ))
        
        return chunks

    async def _process_chunk(self, chunk: Chunk, models: Dict[str, Type[BaseModel]]) -> ExtractedData:
        """Process a text chunk using LLM-based extraction guided by the ontology."""
        try:
            # Generate the extraction prompt
            prompt = self.generate_extraction_prompt(chunk.text)
            ExtractionModel = models["Extraction"]
            
            # Get structured extraction from LLM
            messages = [
                {"role": "system", "content": """You are an expert at extracting structured information from text.
                Your task is to identify entities and relationships that match the given ontology definitions.
                Only extract items with high confidence that clearly match the ontology.
                Follow the exact structure of the models and maintain all relationships.
                Ensure all required fields are filled with appropriate values."""},
                {"role": "user", "content": prompt}
            ]

            # Extract using appropriate LLM provider
            if self.llm_service.provider in ['openai', 'anthropic', 'vertexai']:
                extraction = self.llm_service.client.chat.completions.create(
                    model=self.llm_service.model,
                    messages=messages,
                    response_model=ExtractionModel
                )
            else:  # Gemini
                extraction = self.llm_service.client.chat.completions.create(
                    messages=messages,
                    response_model=ExtractionModel
                )

            # Convert the nested extraction result to flat ExtractedData format
            entities = []
            relationships = []
            
            # Process root nodes first
            for field_name, field_value in extraction.model_dump(exclude={'metadata'}).items():
                if field_value is None:
                    continue
                    
                # Create entity for the root node
                root_entity = Entity(
                    id=f"{chunk.id}_{field_name}_{len(entities)}",
                    type=field_name.upper(),  # Convert to proper node type
                    value=self._get_entity_value(field_value),
                    confidence=extraction.metadata.confidence,
                    metadata={
                        "chunk_id": chunk.id,
                        "source_text": extraction.metadata.source_text
                    }
                )
                entities.append(root_entity)
                
                # Process nested relationships
                nested_entities, nested_rels = self._process_nested_data(
                    chunk.id,
                    root_entity.id,
                    field_value,
                    extraction.metadata.confidence
                )
                
                entities.extend(nested_entities)
                relationships.extend(nested_rels)
            
            return ExtractedData(
                entities=entities,
                relationships=relationships,
                chunk_id=chunk.id
            )
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.id}: {str(e)}")
            raise

    def _process_nested_data(
        self,
        chunk_id: str,
        parent_id: str,
        data: Any,
        confidence: float
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Process nested data from extraction results"""
        entities = []
        relationships = []
        
        # Handle different types of nested data
        if isinstance(data, BaseModel):
            # Process each field of the model
            for field_name, field_value in data.model_dump().items():
                if field_value is None or field_name == 'metadata':
                    continue
                
                if isinstance(field_value, list):
                    # Handle list of nested objects
                    for idx, item in enumerate(field_value):
                        entity_id = f"{chunk_id}_{field_name}_{idx}"
                        entity = Entity(
                            id=entity_id,
                            type=field_name.upper(),
                            value=self._get_entity_value(item),
                            confidence=confidence,
                            metadata={"chunk_id": chunk_id}
                        )
                        entities.append(entity)
                        
                        # Create relationship to parent
                        rel = Relationship(
                            source_id=parent_id,
                            target_id=entity_id,
                            type=f"HAS_{field_name.upper()}",
                            confidence=confidence
                        )
                        relationships.append(rel)
                        
                        # Process any nested data in the item
                        nested_entities, nested_rels = self._process_nested_data(
                            chunk_id,
                            entity_id,
                            item,
                            confidence
                        )
                        entities.extend(nested_entities)
                        relationships.extend(nested_rels)
                
                elif isinstance(field_value, BaseModel):
                    # Handle nested single object
                    entity_id = f"{chunk_id}_{field_name}"
                    entity = Entity(
                        id=entity_id,
                        type=field_name.upper(),
                        value=self._get_entity_value(field_value),
                        confidence=confidence,
                        metadata={"chunk_id": chunk_id}
                    )
                    entities.append(entity)
                    
                    # Create relationship to parent
                    rel = Relationship(
                        source_id=parent_id,
                        target_id=entity_id,
                        type=f"HAS_{field_name.upper()}",
                        confidence=confidence
                    )
                    relationships.append(rel)
                    
                    # Process nested data
                    nested_entities, nested_rels = self._process_nested_data(
                        chunk_id,
                        entity_id,
                        field_value,
                        confidence
                    )
                    entities.extend(nested_entities)
                    relationships.extend(nested_rels)
        
        return entities, relationships

    def _get_entity_value(self, data: Any) -> str:
        """Convert entity data to string value"""
        if isinstance(data, BaseModel):
            # Convert model to string representation
            return str(data.model_dump())
        elif isinstance(data, list):
            return str(data)
        else:
            return str(data)
    
    def _combine_extractions(self, extractions: List[ExtractedData]) -> ExtractedData:
        """Combine extractions from multiple chunks."""
        all_entities = []
        all_relationships = []
        
        # Combine entities with deduplication
        entity_map = {}  # Track entities by normalized value
        for ext in extractions:
            for entity in ext.entities:
                norm_value = entity.value.lower().strip()
                if norm_value not in entity_map:
                    entity_map[norm_value] = entity
                    all_entities.append(entity)
        
        # Combine relationships
        rel_set = set()  # Track unique relationships
        for ext in extractions:
            for rel in ext.relationships:
                rel_key = (rel.source_id, rel.target_id, rel.type)
                if rel_key not in rel_set:
                    rel_set.add(rel_key)
                    all_relationships.append(rel)
        
        return ExtractedData(
            entities=all_entities,
            relationships=all_relationships,
            chunk_id="combined"
        )