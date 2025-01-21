from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request, HTTPException
from typing import List
from datetime import datetime
from app.services.transform.transform import validate_file, save_files, initialize_processing
from app.services.ontology_validator import OntologyValidationError
from app.services.job_manager import get_job_manager
from app.schemas.job import JobStatusResponse
from app.config import settings
import traceback
from app.utils.logger import logger

router = APIRouter(prefix=settings.API_V1_STR, tags=["Transform"])

@router.post("/transform/{ontology_id}/upload", response_model=JobStatusResponse)
async def upload_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    ontology_id: str,
    files: List[UploadFile] = File(...)
) -> JobStatusResponse:
    """
    Upload documents for processing
    
    Args:
        request: FastAPI request object
        background_tasks: FastAPI background tasks
        ontology_id: Ontology ID to use for transformation
        files: List of files to process
        
    Returns:
        JobStatusResponse with id (transform_id) for tracking progress
    """
    try:
        # Generate transform ID
        transform_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Initialize job
        job_manager = get_job_manager(request.app)
        await job_manager.create_job(transform_id)
        
        # Validate files
        for file in files:
            await validate_file(file)
            
        # Save files
        file_paths = await save_files(transform_id, files)
        
        # Start processing in background
        background_tasks.add_task(
            initialize_processing,
            transform_id,
            ontology_id, 
            file_paths,
            request.app
        )
        
        # Return transform_id as job_id for tracking
        return JobStatusResponse(
            id=transform_id,
            status="processing",
            progress=15.0
        )
        
    except Exception as e:
        logger.error(f"Error uploading documents: {traceback.format_exc()}")
        if isinstance(e, OntologyValidationError):
            raise e
        raise

@router.get("/transform/status/{transform_id}",
         response_model=JobStatusResponse)
async def get_transform_status(
    request: Request,
    transform_id: str
) -> JobStatusResponse:
    """
    Get status of a transformation job
    
    Args:
        request: FastAPI request object
        transform_id: ID of the transformation job
        
    Returns:
        JobStatusResponse with current status
        
    Raises:
        HTTPException: If job not found or other error occurs
    """
    try:
        job_manager = get_job_manager(request.app)
        status = job_manager.get_job_status(transform_id)
        
        if status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {transform_id} not found"
            )
            
        return JobStatusResponse(
            id=transform_id,
            status=status.status,
            progress=status.progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transform status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving job status: {str(e)}"
        )
