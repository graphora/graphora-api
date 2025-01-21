from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from typing import List
from app.schemas.transform import UploadResponse, FileValidationError
from app.services.transform.transform import validate_file, save_files, initialize_processing
from app.services.ontology_validator import OntologyValidationError
from app.services.job_manager import get_job_manager
from app.schemas.job import JobStatusResponse
from app.config import settings
from datetime import datetime
import traceback
import asyncio

router = APIRouter(prefix=settings.API_V1_STR, tags=["Transform"])

@router.post("/transform/{ontology_id}/upload",
          response_model=UploadResponse)
async def upload_documents(
    request: Request,
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
        
        # Get job manager
        job_manager = get_job_manager(request.app)
        
        # Create and start async job
        await job_manager.create_job(transform_id)
        asyncio.create_task(
            job_manager.run_async_job(
                transform_id,
                initialize_processing,
                transform_id,
                ontology_id,
                saved_files,
                request.app
            )
        )
        
        return UploadResponse(
            id=transform_id,
            status='success'
        )
        
    except FileValidationError as e:
        # Mark job as failed if it was created
        if 'transform_id' in locals():
            job_manager = get_job_manager(request.app)
            await job_manager.fail_job(transform_id, str(e))
            
        return UploadResponse(
            id=transform_id if 'transform_id' in locals() else '',
            status='error',
            message=str(e)
        )
        
    except OntologyValidationError as e:
        # Mark job as failed if it was created
        if 'transform_id' in locals():
            job_manager = get_job_manager(request.app)
            await job_manager.fail_job(transform_id, str(e))
            
        return UploadResponse(
            id=transform_id if 'transform_id' in locals() else '',
            status='error',
            message="Invalid ontology ID"
        )
        
    except Exception as e:
        # Mark job as failed if it was created
        if 'transform_id' in locals():
            job_manager = get_job_manager(request.app)
            await job_manager.fail_job(transform_id, str(e))
            
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/transform/status/{transform_id}",
         response_model=JobStatusResponse)
async def get_transform_status(
    request: Request,
    transform_id: str) -> JobStatusResponse:
    """
    Get status of a transformation job
    
    Parameters:
    - transform_id: ID of the transformation job
    
    Returns:
    - status: Current job status (processing, completed, failed)
    - progress: Progress percentage (0-100)
    
    Raises:
    - 404: Job not found
    """
    job_manager = get_job_manager(request.app)
    status = job_manager.get_job_status(transform_id)
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Job {transform_id} not found"
        )
    return JobStatusResponse(
        status=status.status,
        progress=status.progress
    )
