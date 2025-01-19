from fastapi import APIRouter, HTTPException
from uuid import uuid4
import yaml
from app.config import settings
from app.schemas.ontology import OntologyRequest, OntologyResponse
from app.services.ontology_validator import parse_and_validate_yaml, OntologyValidationError

router = APIRouter(prefix=settings.API_V1_STR, tags=["Ontology"])

# In-memory cache for validated ontologies
# In production, this should be replaced with a proper caching solution
ontology_cache = {}

@router.post("/ontology", response_model=OntologyResponse)
async def validate_ontology(request: OntologyRequest) -> OntologyResponse:
    """
    Validate and process ontology YAML.
    
    Parameters:
    - text: String containing ontology definition in YAML format
    
    Returns:
    - success: Boolean indicating validation status
    - error: Error message if validation fails
    - uuid: Unique identifier for valid ontology
    
    Raises:
    - 400: Invalid YAML syntax or schema validation failure
    - 500: Internal server error during processing
    """
    try:
        # Parse and validate YAML
        ontology_dict = parse_and_validate_yaml(request.text)
        
        # Generate UUID for valid ontology
        ontology_id = str(uuid4())
        
        # Cache ontology for future use
        ontology_cache[ontology_id] = (request.text, ontology_dict)
        
        return OntologyResponse(
            success=True,
            uuid=ontology_id
        )
        
    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid YAML syntax: {str(e)}"
        )
        
    except OntologyValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Schema validation failed: {str(e)}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
