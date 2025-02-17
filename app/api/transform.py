from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request, HTTPException
from typing import List
from datetime import datetime
from pathlib import Path
import aiofiles
import uuid

from app.schemas.transform import (
    TransformInitResponse,
    DocumentMetadata,
    DocumentInfo,
    DocumentType,
    TransformStatus
)
from app.services.transform.flows import document_transformation_flow
from app.services.transform.validators import FileValidator
from app.config import settings

router = APIRouter(prefix=settings.API_V1_STR, tags=["Transform"])

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
        temp_dir = Path(settings.TEMP_UPLOAD_DIR) / transform_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        doc_metadata = []
        
        for file in files:
            # Validate file
            validation_result = await validator.validate(file)
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
            
            # Create metadata
            metadata = DocumentMetadata(
                source=file.filename,
                document_type=DocumentType(Path(file.filename).suffix[1:]),
                tags=[ontology_id]
            )
            doc_metadata.append(metadata)
            
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
            file_paths=file_paths,
            metadata=doc_metadata
        )
        
        return TransformInitResponse(
            id=transform_id,
            upload_timestamp=datetime.utcnow(),
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

@router.get("/transform/status/{transform_id}",
         response_model=TransformInitResponse)
async def get_transform_status(
    request: Request,
    transform_id: str
) -> TransformInitResponse:
    """
    Get status of a transformation job using Prefect
    
    Args:
        request: FastAPI request object
        transform_id: ID of the transformation job
        
    Returns:
        TransformInitResponse with current status
        
    Raises:
        HTTPException: If job not found or other error occurs
    """
    try:
        from prefect.client import get_client
        
        async with get_client() as client:
            # Get the latest flow run for this transform_id
            flows = await client.read_flows(
                flow_filter={"name": {"any_": ["document-transformation"]}},
                flow_run_filter={"tags": {"all_": [transform_id]}}
            )
            
            if not flows:
                raise HTTPException(
                    status_code=404,
                    detail=f"Transform job {transform_id} not found"
                )
            
            # Get the latest flow run
            flow_run = flows[0].latest_flow_runs[0]
            
            # Map Prefect state to our TransformStatus
            status_map = {
                "PENDING": TransformStatus.PENDING,
                "RUNNING": TransformStatus.PROCESSING,
                "COMPLETED": TransformStatus.COMPLETED,
                "FAILED": TransformStatus.FAILED
            }
            
            transform_status = status_map.get(
                flow_run.state.type.value,
                TransformStatus.FAILED
            )
            
            # Get document info from flow run data
            doc_info = flow_run.state.data.get("doc_info")
            if not doc_info:
                raise HTTPException(
                    status_code=500,
                    detail="Document info not found in flow run data"
                )
            
            return TransformInitResponse(
                id=transform_id,
                upload_timestamp=flow_run.start_time,
                status=transform_status,
                document_info=doc_info
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving transform status: {str(e)}"
        )
