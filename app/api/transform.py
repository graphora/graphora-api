from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request, HTTPException
from typing import List
from datetime import datetime, timezone
from pathlib import Path
import aiofiles
import uuid
from fastapi.responses import JSONResponse
from app.utils.logger import logger
from app.schemas.transform import (
    TransformInitResponse,
    DocumentMetadata,
    DocumentInfo,
    DocumentType,
    TransformStatus
)
from app.services.transform.flows import document_transformation_flow
from app.services.transform.validators import FileValidator
from app.services.transform.progress_tracker import ProgressTracker
from app.services.transform.status_models import DetailedTransformStatus
from app.config import settings

router = APIRouter(prefix=settings.API_V1_STR, tags=["Transform"])
progress_tracker = ProgressTracker()

@router.post("/transform/{ontology_id}/upload", response_model=TransformInitResponse)
async def upload_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    ontology_id: str,
    files: List[UploadFile] = File(...)
) -> TransformInitResponse:
    """
    Upload documents for processing using Prefect workflow
    
    Args:
        request: FastAPI request object
        background_tasks: FastAPI background tasks
        ontology_id: Ontology ID to use for transformation
        files: List of files to process
        
    Returns:
        TransformInitResponse with Prefect flow_id for tracking progress
    """
    try:
        # Generate transform ID
        transform_id = f"transform_{uuid.uuid4().hex}"
        
        # Validate files
        validator = FileValidator()
        temp_dir = Path(settings.UPLOAD_DIR) / transform_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created transform TMP directory {temp_dir}")
        
        file_paths = []
        doc_metadata = []
        
        for file in files:
            # Validate file
            validation_result = await validator.validate(file)
            logger.info(f"Validated file {file.filename}: {validation_result}")
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file {file.filename}: {validation_result.errors}"
                )
            
            # Save file temporarily
            temp_path = temp_dir / file.filename
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await file.read()
                await file.seek(0)
                await f.write(content)
            
            file_paths.append(temp_path)
            logger.info(f"File saved to TMP directory {temp_path}")
            
            # Create metadata
            metadata = DocumentMetadata(
                source=file.filename,
                document_type=DocumentType(Path(file.filename).suffix[1:]),
                tags=[ontology_id]
            )
            doc_metadata.append(metadata)
            logger.info(f"Created document metadata {metadata}")
            
            # Create document info for response
            doc_info = DocumentInfo(
                filename=file.filename,
                size=len(content),
                document_type=metadata.document_type,
                metadata=metadata
            )
        
        # Start Prefect flow in background
        background_tasks.add_task(
            document_transformation_flow,
            transform_id=transform_id,
            ontology_id=ontology_id,
            file_paths=file_paths,
            metadata=doc_metadata
        )
        
        return TransformInitResponse(
            id=transform_id,
            upload_timestamp=datetime.now(timezone.utc),
            status=TransformStatus.PENDING,
            document_info=doc_info
        )
        
    except Exception as e:
        # Clean up temp directory on error
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()
        raise

@router.get(
    "/transform/status/{transform_id}",
    response_model=DetailedTransformStatus,
    response_model_exclude_none=True
)
async def get_transform_status(
    request: Request,
    transform_id: str,
    include_metrics: bool = True
) -> DetailedTransformStatus:
    """
    Get detailed transformation status
    
    Args:
        transform_id: Transformation ID to get status for
        include_metrics: Whether to include resource metrics
        
    Returns:
        DetailedTransformStatus with current progress
        
    Raises:
        HTTPException: If transform not found or other error
    """
    try:
        # Get status)
        status = await progress_tracker.get_detailed_status(transform_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Transform {transform_id} not found"
            )
        
        # Optionally exclude metrics
        if not include_metrics:
            status.resource_metrics = None
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get transform status: {str(e)}"
        )

@router.post(
    "/transform/status/{transform_id}/cleanup",
    response_class=JSONResponse
)
async def cleanup_transform_status(
    request: Request,
    transform_id: str,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    """
    Clean up transformation status data
    
    Args:
        transform_id: Transformation ID to clean up
        
    Returns:
        JSONResponse confirming cleanup
    """
    try:
        # Add cleanup to background tasks
        background_tasks.add_task(
            progress_tracker.cleanup_transform,
            transform_id
        )
        
        return JSONResponse({
            "message": f"Cleanup scheduled for transform {transform_id}"
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule cleanup: {str(e)}"
        )
