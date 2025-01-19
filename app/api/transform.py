from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from uuid import uuid4
from app.schemas.transform import UploadResponse, FileValidationError
from app.services.transform.transform import validate_file, save_files, initialize_processing
from app.services.ontology_validator import OntologyValidationError
from app.config import settings
from datetime import datetime

router = APIRouter(prefix=settings.API_V1_STR, tags=["Transform"])

@router.post("/transform/{ontology_id}/upload",
          response_model=UploadResponse)
async def upload_documents(
    ontology_id: str,
    files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload documents for transformation
    
    Parameters:
    - ontology_id: UUID of validated ontology
    - files: List of files to process
    
    Returns:
    - id: Transformation batch ID
    - status: Upload status
    - message: Optional status message
    
    Raises:
    - 400: Invalid ontology ID or file validation error
    - 500: Internal server error during processing
    """
    try:
        # Validate files first to avoid unnecessary processing
        for file in files:
            await validate_file(file)
            
        # Generate transformation ID
        transform_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Save files
        saved_files = await save_files(transform_id, files)
        
        # Initialize processing
        await initialize_processing(transform_id, ontology_id, saved_files)
        
        return UploadResponse(
            id=transform_id,
            status='success'
        )
        
    except FileValidationError as e:
        return UploadResponse(
            id='',
            status='error',
            message=str(e)
        )
        
    except OntologyValidationError:
        return UploadResponse(
            id='',
            status='error',
            message="Invalid ontology ID"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
