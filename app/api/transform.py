from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from typing import List, Optional
import aiofiles
import uuid
import traceback
import time
import json
import os
import re
from fastapi.responses import JSONResponse
from app.utils.logger import logger
from app.schemas.transform import (
    TransformInitResponse,
    DocumentMetadata,
    DocumentInfo,
    DocumentType,
    TransformStatus,
)
from app.services.transform.status_models import TransformationStage
from app.services.transform.validators import FileValidator
from app.services.transform.flows import document_transformation_flow, progress_tracker
from app.services.transform.status_models import DetailedTransformStatus
from app.config import settings
from pathlib import Path
from datetime import datetime, timezone
from app.services.audit_service import audit_service, OperationType
from app.services.chunking.config import ChunkingConfig
from app.auth import get_current_user_id

router = APIRouter(prefix=settings.API_V1_STR, tags=["Transform"])


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.

    Args:
        filename: The original filename from user input

    Returns:
        A safe filename with path traversal characters removed

    Raises:
        ValueError: If the filename is invalid or empty after sanitization
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Get just the basename to remove any directory components
    safe_name = os.path.basename(filename)

    # Check for path traversal attempts
    if ".." in filename or filename != safe_name:
        raise ValueError(f"Invalid filename: path traversal detected in '{filename}'")

    # Remove any remaining dangerous characters (keep alphanumeric, dots, hyphens, underscores)
    # Allow spaces but be strict about other characters
    if not re.match(r"^[\w\-. ]+$", safe_name):
        raise ValueError(
            f"Invalid filename: contains disallowed characters '{filename}'"
        )

    # Ensure filename is not empty after sanitization
    if not safe_name or safe_name in (".", ".."):
        raise ValueError(f"Invalid filename after sanitization: '{filename}'")

    return safe_name


async def run_transform_flow(
    transform_id: str,
    ontology_id: str,
    file_paths: List[Path],
    metadata: List[DocumentMetadata],
    user_id: str,
    audit_id: str,
    chunking_config: Optional[ChunkingConfig] = None,
):
    """Run the transform flow asynchronously with user context and audit logging"""
    flow_start_time = time.time()

    # Create separate audit ID for transform completion
    completion_audit_id = await audit_service.log_operation_start(
        user_id=user_id,
        operation_type=OperationType.TRANSFORM_COMPLETED,
        operation_id=transform_id,
        resource_name=f"Transform {transform_id[:8]}",
        metadata={"ontology_id": ontology_id, "files_count": len(file_paths)},
    )

    try:
        # Convert Path objects to strings
        file_paths_str = [str(path) for path in file_paths]

        # Run the flow with user ID context and chunking config (transforms use staging database)
        flow_state = await document_transformation_flow(
            transform_id=transform_id,
            ontology_id=ontology_id,
            file_paths=file_paths_str,
            metadata=metadata,
            user_id=user_id,  # Pass user ID to the flow
            chunking_config=chunking_config,  # Pass chunking configuration
        )

        logger.info(
            f"Completed transform flow for user {user_id} with state: {flow_state}"
        )

        # Log successful transform completion
        flow_duration_ms = int((time.time() - flow_start_time) * 1000)
        if completion_audit_id:
            await audit_service.log_operation_success(
                audit_id=completion_audit_id,
                duration_ms=flow_duration_ms,
                metadata={
                    "transform_id": transform_id,
                    "ontology_id": ontology_id,
                    "files_processed": len(file_paths),
                    "flow_state": str(flow_state),
                    "total_nodes": flow_state.get("total_nodes", 0),
                    "total_relationships": flow_state.get("total_relationships", 0),
                },
            )

    except Exception as e:
        logger.error(f"Failed to complete transform flow for user {user_id}: {str(e)}")

        # Log transform failure
        flow_duration_ms = int((time.time() - flow_start_time) * 1000)
        if completion_audit_id:
            await audit_service.log_operation_failure(
                audit_id=completion_audit_id,
                error_message=str(e),
                duration_ms=flow_duration_ms,
            )

        raise


@router.post("/transform/{ontology_id}/upload", response_model=TransformInitResponse)
async def upload_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    ontology_id: str,
    user_id: str = Depends(get_current_user_id),
    files: List[UploadFile] = File(...),
    chunking_config: Optional[str] = Form(
        None, description="JSON string of chunking configuration"
    ),
) -> TransformInitResponse:
    """
    Upload documents for processing using Prefect workflow

    Args:
        request: FastAPI request object
        background_tasks: FastAPI background tasks
        ontology_id: Ontology ID to use for transformation
        user_id: User's ID (from header)
        files: List of files to process

    Returns:
        TransformInitResponse with Prefect flow_id for tracking progress
    """
    start_time = time.time()
    temp_dir = Path(settings.UPLOAD_DIR)
    transform_id = f"transform_{uuid.uuid4().hex}"
    audit_id = ""

    try:
        logger.info(f"Starting document upload for user: {user_id}")

        # Parse chunking configuration if provided
        parsed_chunking_config = None
        if chunking_config:
            try:
                config_dict = json.loads(chunking_config)
                parsed_chunking_config = ChunkingConfig(**config_dict)
                logger.info(
                    f"Using custom chunking config: {parsed_chunking_config.strategy}"
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Invalid chunking config provided: {e}. Using defaults."
                )

        # Start audit trail for transform operation
        audit_id = await audit_service.log_operation_start(
            user_id=user_id,
            operation_type=OperationType.TRANSFORM_STARTED,
            operation_id=transform_id,
            resource_name=f"Transform {transform_id[:8]}",
            metadata={
                "ontology_id": ontology_id,
                "files_count": len(files),
                "file_names": [file.filename for file in files],
                "chunking_strategy": (
                    parsed_chunking_config.strategy.value
                    if parsed_chunking_config
                    else "default"
                ),
            },
        )

        # Initialize progress tracking
        await progress_tracker.initialize_transform(transform_id)

        # Validate files
        validator = FileValidator()
        temp_dir = Path(settings.UPLOAD_DIR) / transform_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created transform TMP directory {temp_dir} for user {user_id}")

        file_paths = []
        doc_metadata = []
        total_file_size = 0

        for file in files:
            # Sanitize filename to prevent path traversal attacks
            try:
                safe_filename = sanitize_filename(file.filename)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=str(e),
                )

            # Validate file
            validation_result = await validator.validate(file)
            logger.info(
                f"Validated file {safe_filename} for user {user_id}: {validation_result}"
            )
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file {safe_filename}: {validation_result.errors}",
                )

            # Save file temporarily using sanitized filename
            temp_path = temp_dir / safe_filename
            async with aiofiles.open(temp_path, "wb") as f:
                content = await file.read()
                await file.seek(0)
                await f.write(content)

            file_paths.append(temp_path)
            total_file_size += len(content)
            logger.info(f"File saved to TMP directory {temp_path} for user {user_id}")

            # Create metadata using sanitized filename
            metadata = DocumentMetadata(
                source=safe_filename,
                document_type=DocumentType(Path(safe_filename).suffix[1:]),
                tags=[ontology_id, user_id],  # Add user ID to tags
            )
            doc_metadata.append(metadata)
            logger.info(f"Created document metadata for user {user_id}: {metadata}")

            # Create document info for response
            doc_info = DocumentInfo(
                filename=safe_filename,
                size=len(content),
                document_type=metadata.document_type,
                metadata=metadata,
            )

        await progress_tracker.complete_stage(transform_id, TransformationStage.UPLOAD)

        # Start Prefect flow in background with user context, audit ID, and chunking config
        background_tasks.add_task(
            run_transform_flow,
            transform_id=transform_id,
            ontology_id=ontology_id,
            file_paths=file_paths,
            metadata=doc_metadata,
            user_id=user_id,
            audit_id=audit_id,
            chunking_config=parsed_chunking_config,
        )

        logger.info(
            f"Started transformation flow for user {user_id} with transform_id: {transform_id}"
        )

        # Log upload success (not the full transform completion yet)
        upload_duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_success(
                audit_id=audit_id,
                duration_ms=upload_duration_ms,
                metadata={
                    "transform_id": transform_id,
                    "upload_completed": True,
                    "total_file_size_bytes": total_file_size,
                    "temp_directory": str(temp_dir),
                },
            )

        return TransformInitResponse(
            id=transform_id,
            upload_timestamp=datetime.now(timezone.utc),
            status=TransformStatus.PENDING,
            document_info=doc_info,
        )

    except Exception as e:
        logger.error(f"Upload failed for user {user_id}: {str(e)}")
        traceback.print_exc()

        # Log upload failure
        upload_duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=upload_duration_ms
            )

        # Clean up temp directory on error
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()
        raise


@router.get(
    "/transform/status/{transform_id}",
    response_model=DetailedTransformStatus,
    response_model_exclude_none=True,
)
async def get_transform_status(
    request: Request,
    transform_id: str,
    user_id: str = Depends(get_current_user_id),
    include_metrics: bool = True,
) -> DetailedTransformStatus:
    """
    Get detailed transformation status

    Args:
        transform_id: Transformation ID to get status for
        user_id: User's ID (from header)
        include_metrics: Whether to include resource metrics

    Returns:
        DetailedTransformStatus with current progress

    Raises:
        HTTPException: If transform not found or other error
    """
    try:
        logger.info(
            f"Getting transform status for user {user_id}, transform_id: {transform_id}"
        )

        # Get status
        status = await progress_tracker.get_detailed_status(transform_id)

        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Transform {transform_id} not found for user {user_id}",
            )

        # Optionally exclude metrics
        if not include_metrics:
            status.resource_metrics = None

        return status

    except HTTPException:
        traceback.print_exc()
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error getting transform status for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get transform status: {str(e)}"
        )


@router.post("/transform/status/{transform_id}/cleanup", response_class=JSONResponse)
async def cleanup_transform_status(
    request: Request,
    transform_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """
    Clean up transformation status data

    Args:
        transform_id: Transformation ID to clean up
        user_id: User's ID (from header)

    Returns:
        JSONResponse confirming cleanup
    """
    try:
        logger.info(
            f"Scheduling cleanup for user {user_id}, transform_id: {transform_id}"
        )

        # Add cleanup to background tasks
        background_tasks.add_task(progress_tracker.cleanup_transform, transform_id)

        return JSONResponse(
            {
                "message": f"Cleanup scheduled for transform {transform_id} for user {user_id}"
            }
        )

    except Exception as e:
        logger.error(f"Failed to schedule cleanup for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to schedule cleanup: {str(e)}"
        )
