from typing import Dict, List, Any, Type
from datetime import datetime
from vertexai.generative_models import GenerativeModel
import vertexai
from pydantic import BaseModel

from app.config import settings
from app.services.transform.models import ExtractionConfidence
from app.services.llm.cache import ExtractionCache

# Initialize cache
_cache = ExtractionCache()

# Configure Vertex AI
if settings.VERTEXAI_PROJECT_ID and settings.VERTEXAI_LOCATION: 
    vertexai.init(
        project=settings.VERTEXAI_PROJECT_ID,
        location=settings.VERTEXAI_LOCATION
    )

class EntityExtraction(BaseModel):
    """Schema for entity extraction"""
    entity_type: str
    properties: Dict[str, Any]
    confidence_score: float
    property_scores: Dict[str, float]

class RelationshipExtraction(BaseModel):
    """Schema for relationship extraction"""
    source_type: str
    source_fields: Dict[str, Any]
    target_type: str
    target_fields: Dict[str, Any]
    relationship_type: str
    confidence_score: float

class ExtractionResponse(BaseModel):
    """Schema for LLM extraction response"""
    entities: List[EntityExtraction]
    relationships: List[RelationshipExtraction]

def create_extraction_prompt(
    text: str,
    models: Dict[str, Type[BaseModel]]
) -> str:
    """Create structured prompt for information extraction"""
    # Create schema description
    schema_desc = []
    for entity_name, model in models.items():
        fields = []
        for field_name, field in model.model_fields.items():
            if field_name not in ['id', 'type', 'provenance']:
                field_desc = {
                    'name': field_name,
                    'type': str(field.type_),
                    'required': field.required
                }
                fields.append(field_desc)
        
        schema_desc.append({
            'entity': entity_name,
            'fields': fields
        })
    
    prompt = f"""Extract structured information from the text below according to these entity definitions:

Entity Definitions:
{schema_desc}

Rules:
1. Extract ALL relevant entities and their relationships
2. Only include fields if you are confident about the information
3. Format numbers and dates appropriately
4. Include confidence scores (0.0 to 1.0) for each extraction
5. If information is ambiguous or unclear, set the field to null

Text:
{text}
"""
    return prompt.strip()

async def call_llm_gemini(
    text: str,
    models: Dict[str, Type[BaseModel]],
    track_metrics: bool = True,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Call Gemini API for structured information extraction"""
    try:
        # Check cache first
        if use_cache:
            cached = await _cache.get(text, "gemini-pro", models)
            if cached:
                return cached
        
        # Create model
        model = GenerativeModel('gemini-pro')
        
        # Generate prompt
        prompt = create_extraction_prompt(text, models)
        
        # Call API with structured output
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 40
            },
            response_mime_type='application/json',
            response_schema=ExtractionResponse
        )
        
        # Get parsed response
        extraction = response.parsed
        
        # Create confidence metadata
        confidence = ExtractionConfidence(
            overall_score=sum(
                e.confidence_score for e in extraction.entities
            ) / len(extraction.entities) if extraction.entities else 0.0,
            property_scores={
                f"{e.entity_type}.{prop}": score
                for e in extraction.entities
                for prop, score in e.property_scores.items()
            },
            extraction_method="gemini_structured",
            llm_model="gemini-pro",
            timestamp=datetime.now()
        )
        
        # Convert to expected format
        result = {}
        for entity in extraction.entities:
            if entity.entity_type not in result:
                result[entity.entity_type] = []
            
            # Add entity with confidence
            entity_data = {
                **entity.properties,
                'confidence_score': entity.confidence_score,
                'property_scores': entity.property_scores,
                'extraction_confidence': confidence
            }
            result[entity.entity_type].append(entity_data)
        
        # Cache the result
        if use_cache:
            await _cache.set(
                text,
                "gemini-pro",
                models,
                result,
                confidence.overall_score
            )
        
        return result
        
    except Exception as e:
        raise Exception(f"LLM API error: {str(e)}")

async def validate_llm_response(
    response: Dict[str, Any],
    models: Dict[str, Type[BaseModel]]
) -> bool:
    """Validate LLM response format and content"""
    try:
        # Validate each entity type
        for entity_type, entities in response.items():
            if entity_type not in models:
                return False
            
            model = models[entity_type]
            for entity in entities:
                # Check confidence score
                if not isinstance(
                    entity.get('confidence_score', 0),
                    (int, float)
                ):
                    return False
                
                # Check required fields
                for field_name, field in model.model_fields.items():
                    if field.required and field_name not in entity:
                        return False
        
        return True
        
    except Exception:
        return False
