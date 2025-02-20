from typing import Any, Type, Optional
from pydantic import BaseModel
from app.utils.llm_client_service import extract, generate_text

class LLMClient:
    """Client for LLM-based extraction"""
    async def generate_text(
        self,
        prompt: str,
        json_response: bool = True
    ) -> Any:
        result = generate_text(prompt, json_response)
        return result   

    async def extract_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        context: Optional[str] = None
    ) -> Any:
        """Extract entities and relationships from text chunk"""
        
        # Build prompt
        prompt = f"""Extract structured information from the text chunk according to the ontology specification.
        
Format the output as a JSON object with the following structure:
1. For each entity type, include a list field named "<entity_type>_list" containing all instances
2. For each relationship type, include a list field named "<source>_<relationship>_<target>" containing all instances
3. Include metadata fields:
   - extraction_timestamp: Current timestamp
   - tokens_used: Number of tokens used (if available)
   - confidence_score: Overall confidence in extraction (0.0 to 1.0)
4. Omit optional fields if information is not clearly present
5. No additional properties. Just the specified fields.

When extracting new information, maintain consistency with these previously identified entities. 
These were identified from the previous tex chunks of the same doc.
```
{context}
```

Chunk to process:
{chunk}

Remember:
- Only extract information that is explicitly present in the text
- Set confidence scores based on certainty of extraction
- Include all required fields for each entity/relationship
- Omit optional fields if information is not clearly present
"""
        # Call LLM with structured output
        result = extract(prompt, response_model)
        return result   