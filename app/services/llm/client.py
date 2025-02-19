from typing import Any, Type
from pydantic import BaseModel
from app.utils.llm_client_service import extract

class LLMClient:
    """Client for LLM-based extraction"""
    
    async def extract_from_chunk(
        self,
        chunk: str,
        ontology_spec: str,
        response_model: Type[BaseModel]
    ) -> Any:
        """Extract entities and relationships from text chunk"""
        
        # Build prompt
        prompt = f"""Extract structured information from the text according to this ontology specification.
        
Format the output as a JSON object with the following structure:
1. For each entity type, include a list field named "<entity_type>_list" containing all instances
2. For each relationship type, include a list field named "<source>_<relationship>_<target>" containing all instances
3. Include metadata fields:
   - extraction_timestamp: Current timestamp
   - tokens_used: Number of tokens used (if available)
   - confidence_score: Overall confidence in extraction (0.0 to 1.0)

Ontology Specification:
{ontology_spec}

Text to process:
{chunk}

Remember:
- Only extract information that is explicitly present in the text
- Set confidence scores based on certainty of extraction
- Include all required fields for each entity/relationship
- Omit optional fields if information is not clearly present
"""
        
        # Call LLM with structured output
        return extract(prompt, response_model)
