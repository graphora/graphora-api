from typing import List, Dict, Optional, Tuple
from datetime import datetime
from instructor import patch
from openai import OpenAI
from app.utils.logger import logger
from dataclasses import dataclass
from app.schemas.extraction import (
    ExtractedEntity, ExtractedRelationship,
    ChunkExtraction
)
from app.schemas.document import (
    Entity, 
    Relationship
)
from app.utils.llm_client_service import LLMClientService
from app.services.ontology_generator_service import Neo4jOntology
import spacy

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
        
        # Process each chunk
        all_extractions = []
        for chunk in chunks:
            extracted_data = await self._process_chunk(chunk, ontology)
            all_extractions.append(extracted_data)
        
        # Combine extractions
        return self._combine_extractions(all_extractions)
    
    def generate_extraction_prompt(self, text: str, ontology: Neo4jOntology) -> str:
        """Generate a prompt for the LLM to extract entities and relationships."""
        # Convert ontology to a clear text format for the LLM
        node_descriptions = []
        for node in ontology.nodes:
            # Format properties with their types and descriptions
            props = [
                f"- {prop.name} ({prop.type.value}): {prop.description}"
                f"{' (Required)' if prop.required else ''}"
                f"{' (Examples: ' + ', '.join(prop.examples) + ')' if prop.examples else ''}"
                for prop in node.properties
            ]
            
            node_desc = f"""Type: {node.label.name}
            Description: {node.label.description}
            Properties:
            {chr(10).join(props) if props else '- No specific properties defined'}
            """
            if node.examples:
                node_desc += f"\nExamples:\n{chr(10).join(f'- {ex}' for ex in node.examples)}"
            
            node_descriptions.append(node_desc)

        relationship_descriptions = []
        for rel in ontology.relationships:
            # Format relationship properties if they exist
            props = []
            if rel.properties:
                props = [
                    f"- {prop.name} ({prop.type.value}): {prop.description}"
                    f"{' (Required)' if prop.required else ''}"
                    f"{' (Examples: ' + ', '.join(prop.examples) + ')' if prop.examples else ''}"
                    for prop in rel.properties
                ]
            
            direction_symbol = {
                "RIGHT": "->",
                "LEFT": "<-",
                "BOTH": "<->"
            }[rel.direction.value]

            rel_desc = f"""Type: {rel.label.name}
            Description: {rel.label.description}
            Pattern: ({rel.source_label}){direction_symbol}[:{rel.label.name}]({rel.target_label})
            Properties:
            {chr(10).join(props) if props else '- No specific properties defined'}
            """
            if rel.examples:
                rel_desc += f"\nExamples:\n{chr(10).join(f'- {ex}' for ex in rel.examples)}"
                
            relationship_descriptions.append(rel_desc)

        prompt = f"""Extract entities and relationships from the following text according to this ontology:

        # Node Types
        {chr(10).join(node_descriptions)}

        # Relationship Types
        {chr(10).join(relationship_descriptions)}

        Text to analyze:
        {text}

        Extract all entities and relationships that match the ontology definitions. For each extraction:
        1. Identify entities that match the node types defined above
        2. Include all required properties and any optional properties found in the text
        3. Identify relationships between entities following the patterns defined
        4. Ensure high confidence scores (>0.8) only for clear matches
        5. Include example matches that follow the patterns shown in the examples

        Format entities and relationships exactly according to the types and properties defined in the ontology."""

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
        MAX_CHUNK_LENGTH = 1000  # Characters
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

    async def _process_chunk(self, chunk: Chunk, ontology: Neo4jOntology) -> ExtractedData:
        """Process a text chunk using LLM-based extraction guided by the ontology."""
        try:
            # Generate the extraction prompt
            prompt = self.generate_extraction_prompt(chunk.text, ontology)
            
            # Get structured extraction from LLM
            extraction = self.llm_service.client.chat.completions.create(
                response_model=ChunkExtraction,
                messages=[
                    {"role": "system", "content": """You are an expert at extracting structured information from text.
Your task is to identify entities and relationships that match the given ontology definitions.
Only extract items with high confidence that clearly match the ontology.
Assign realistic confidence scores based on how clearly the text matches the definitions."""},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Convert to internal ExtractedData format
            entities = []
            for ent in extraction.entities:
                entity = Entity(
                    id=f"{chunk.id}_entity_{len(entities)}",
                    type=ent.type,
                    value=ent.value,
                    confidence=ent.confidence,
                    metadata={
                        "chunk_id": chunk.id,
                        "position": ent.metadata.get("position", None),
                        **ent.metadata
                    }
                )
                entities.append(entity)
            
            relationships = []
            for rel in extraction.relationships:
                # Ensure entity IDs reference extracted entities
                if self._validate_relationship(rel, entities):
                    relationship = Relationship(
                        source_id=rel.source_id,
                        target_id=rel.target_id,
                        type=rel.type,
                        confidence=rel.confidence
                    )
                    relationships.append(relationship)
            
            return ExtractedData(
                entities=entities,
                relationships=relationships,
                chunk_id=chunk.id
            )
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.id}: {str(e)}")
            raise

    def _validate_relationship(self, rel: ExtractedRelationship, entities: List[Entity]) -> bool:
        """Validate that a relationship references valid entities."""
        entity_ids = {entity.id for entity in entities}
        return rel.source_id in entity_ids and rel.target_id in entity_ids
    

    
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