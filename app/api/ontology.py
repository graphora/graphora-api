import traceback
from app.services.transform.ontology_helper import OntologyParser
from fastapi import APIRouter, HTTPException
from uuid import uuid4
import yaml
import os
from pathlib import Path
from app.config import settings
from app.schemas.ontology import OntologyRequest, OntologyResponse
from app.services.ontology_validator import parse_and_validate_yaml, OntologyValidationError

router = APIRouter(prefix=settings.API_V1_STR, tags=["Ontology"])

def ensure_ontology_dir():
    """Ensure ontology directory exists"""
    Path(settings.ontology_dir).expanduser().mkdir(parents=True, exist_ok=True)

@router.post("/ontology", response_model=OntologyResponse)
async def validate_ontology(request: OntologyRequest) -> OntologyResponse:
    """
    Validate and process ontology YAML.
    
    Parameters:
    - text: String containing ontology definition in YAML format
    
    Returns:
    - id: Unique ID for the validated ontology
    """
    try:
        # Parse and validate YAML
        parse_and_validate_yaml(request.text)
        
        # Generate unique ID
        ontology_id = str(uuid4())
        
        # Ensure directory exists
        ensure_ontology_dir()
        
        # Save ontology to file
        ontology_path = Path(settings.ontology_dir).expanduser() / f"{ontology_id}.yaml"
        with open(ontology_path, 'w') as f:
            f.write(request.text)
            
        # Create Full Text Indexes for entities defined in Ontology
        await OntologyParser(ontology_path).build_full_text_indexes()
        
        return OntologyResponse(id=ontology_id)
        
    except OntologyValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ontology: {str(e)}"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing ontology: {str(e)}"
        )

@router.get("/ontology/{ontology_id}", response_model=OntologyRequest)
async def get_ontology(ontology_id: str) -> OntologyRequest:
    """
    Get ontology by ID
    
    Parameters:
    - ontology_id: ID of the ontology to retrieve
    
    Returns:
    - text: Ontology YAML text
    """
    try:
        # Check if ontology exists
        ontology_path = Path(settings.ontology_dir).expanduser() / f"{ontology_id}.yaml"
        if not ontology_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Ontology {ontology_id} not found"
            )
            
        # Read and validate ontology
        with open(ontology_path, 'r') as f:
            ontology_text = f.read()
            
        # Validate to ensure it's still valid
        parse_and_validate_yaml(ontology_text)
        
        return OntologyRequest(text=ontology_text)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving ontology: {str(e)}"
        )
