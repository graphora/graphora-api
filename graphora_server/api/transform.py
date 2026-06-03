from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field
from typing import List, Optional
import aiofiles
import uuid
import traceback
import time
import json
import os
import re
from fastapi.responses import JSONResponse
from graphora_server.utils.logger import logger
from graphora_server.schemas.transform import (
    TransformInitResponse,
    DocumentMetadata,
    DocumentInfo,
    DocumentType,
    TransformStatus,
)
from graphora_server.services.transform.status_models import TransformationStage
from graphora_server.services.transform.validators import FileValidator
from graphora_server.services.transform.flows import (
    document_transformation_flow,
    progress_tracker,
)
from graphora_server.services.transform.status_models import DetailedTransformStatus
from graphora_server.config import settings
from pathlib import Path
from datetime import datetime, timezone
from graphora_server.api.budgets import enforce_budget_preflight
from graphora_server.services.audit_service import audit_service, OperationType
from graphora_server.services.chunking.config import ChunkingConfig
from graphora_server.services.schema_inference import create_auto_schema_ontology
from graphora_server.services.document_parser import DocumentParser
from graphora_server.services.user_db_service import UserDatabaseService
from graphora_server.auth import get_current_user_id
from graphora_server.exceptions import AIConfigurationError

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


@router.post(
    "/transform/{ontology_id}/upload",
    response_model=TransformInitResponse,
    summary="Extract using a registered ontology",
)
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
    """Run extraction with a pre-registered ontology.

    The pipeline is biased toward the entity and relationship types
    declared in the ontology (registered via `POST /ontology`).

    **Schema-free alternatives** — no separate ontology step required:

    - `POST /transform/upload` — auto-infers an ontology from
      document content *before* extracting.
    - `POST /transform/schemaless/upload` — extracts with a generic
      schema; ontology emerges from results and is refined post-hoc
      via `GET /transform/{id}/inferred-ontology` +
      `POST /transform/{id}/finalize-ontology`.

    Example:

        curl -X POST /api/v1/transform/{ontology_id}/upload \\
            -F "files=@doc.pdf"
    """
    start_time = time.time()
    temp_dir = Path(settings.UPLOAD_DIR)
    transform_id = f"transform_{uuid.uuid4().hex}"
    audit_id = ""

    try:
        logger.info(f"Starting document upload for user: {user_id}")

        # B5-obs slice 2: preflight budget check. Raises 402 if the
        # user is over their monthly cap; no-ops otherwise. Runs
        # before any audit row / temp file is created — failing
        # here leaves no cleanup tail.
        await enforce_budget_preflight(user_id)

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

    except HTTPException:
        # Reviewer-flagged on commit 535f56d (B5-obs slice 2 P1):
        # without this guard the budget-preflight 402 (and any
        # other HTTPException raised before temp_dir narrows to
        # the per-transform subdirectory) would fall into the
        # broad except below and shutil.rmtree(temp_dir) — which
        # is still settings.UPLOAD_DIR — deleting every other
        # transform's working directory. Auto-schema and
        # schemaless endpoints already have this guard; the
        # ontology-supplied endpoint was missing it.
        raise
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
        try:
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            logger.warning(
                f"Failed to cleanup temp directory {temp_dir}: {cleanup_err}"
            )
        raise


@router.post(
    "/transform/upload",
    response_model=TransformInitResponse,
    summary="Extract with auto-inferred ontology (zero-config)",
)
async def upload_documents_auto_schema(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    files: List[UploadFile] = File(...),
    auto_schema: bool = Form(
        True, description="Auto-generate schema from document content"
    ),
    chunking_config: Optional[str] = Form(
        None, description="JSON string of chunking configuration"
    ),
) -> TransformInitResponse:
    """Zero-config extraction: auto-infer the ontology, then extract.

    This endpoint peeks at uploaded document text to infer an
    ontology, then runs extraction biased toward that ontology.
    **No `ontology_id` required** — there is no separate
    `POST /ontology` registration step.

    **Related modes**:

    - `POST /transform/{ontology_id}/upload` — when you have a
      pre-registered ontology you want to enforce.
    - `POST /transform/schemaless/upload` — when you want to
      defer ontology design until *after* seeing extraction
      results (no pre-extraction schema bias).

    Example:

        curl -X POST /api/v1/transform/upload -F "files=@doc.pdf"
    """
    start_time = time.time()
    temp_dir = Path(settings.UPLOAD_DIR)
    transform_id = f"transform_{uuid.uuid4().hex}"
    audit_id = ""

    try:
        logger.info(f"Starting auto-schema document upload for user: {user_id}")

        # B5-obs slice 2: preflight budget check. Same enforcement
        # contract as the ontology-supplied upload path.
        await enforce_budget_preflight(user_id)

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
                "auto_schema": auto_schema,
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
        text_chunks = []

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

            # Extract text for schema inference
            if auto_schema:
                try:
                    parser = DocumentParser()
                    text_content = await parser.parse_file(str(temp_path))
                    if text_content:
                        text_chunks.append(
                            text_content[:10000]
                        )  # Sample first 10k chars
                except Exception as parse_err:
                    logger.warning(
                        f"Failed to extract text for schema inference: {parse_err}"
                    )

            # Create metadata using sanitized filename
            metadata = DocumentMetadata(
                source=safe_filename,
                document_type=DocumentType(Path(safe_filename).suffix[1:]),
                tags=["auto-schema", user_id],
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

        # Generate schema from text if auto_schema is enabled
        ontology_id = None
        if auto_schema and text_chunks:
            logger.info(f"Inferring schema from {len(text_chunks)} text chunks")
            await progress_tracker.update_stage_progress(
                transform_id, TransformationStage.PARSING, 0, "Inferring schema..."
            )
            try:
                ontology_id = await create_auto_schema_ontology(
                    text_chunks=text_chunks,
                    user_id=user_id,
                    transform_id=transform_id,
                )
                logger.info(f"Auto-generated ontology: {ontology_id}")
            except AIConfigurationError as ai_err:
                logger.warning(f"AI configuration error for user {user_id}: {ai_err}")
                raise HTTPException(
                    status_code=400,
                    detail=ai_err.to_dict(),
                )
            except Exception as schema_err:
                logger.error(f"Schema inference failed: {schema_err}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate schema: {str(schema_err)}",
                )
        elif auto_schema:
            # No text extracted - use default generic schema
            from graphora_server.services.schema_inference import (
                get_default_generic_schema,
            )
            from graphora_server.services.ontology_storage_service import (
                ontology_storage_service,
            )

            generic_yaml = get_default_generic_schema()
            ontology_id = f"auto_{uuid.uuid4().hex[:12]}"
            await ontology_storage_service.store_ontology(
                user_id=user_id,
                ontology_id=ontology_id,
                yaml_content=generic_yaml,
                name=f"Generic Schema ({transform_id[:8]})",
                description="Default generic schema for entity extraction",
            )
            logger.info(f"Using default generic schema: {ontology_id}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Either provide ontology_id or enable auto_schema",
            )

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
            f"Started auto-schema transformation flow for user {user_id} with transform_id: {transform_id}"
        )

        # Log upload success (not the full transform completion yet)
        upload_duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_success(
                audit_id=audit_id,
                duration_ms=upload_duration_ms,
                metadata={
                    "transform_id": transform_id,
                    "ontology_id": ontology_id,
                    "auto_schema": True,
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto-schema upload failed for user {user_id}: {str(e)}")
        traceback.print_exc()

        # Log upload failure
        upload_duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=upload_duration_ms
            )

        # Clean up temp directory on error
        try:
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            logger.warning(
                f"Failed to cleanup temp directory {temp_dir}: {cleanup_err}"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process documents: {str(e)}",
        )


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


@router.get(
    "/transform/{transform_id}/inferred-ontology",
    response_class=JSONResponse,
)
async def get_inferred_ontology(
    transform_id: str,
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """Run post-hoc ontology inference on a completed extraction.

    Unlike the pre-extraction auto-schema path, this endpoint runs
    AFTER extraction and refines the ontology from emerged types.
    Side-effect-free: the response carries the YAML inline but the
    ontology is not persisted. Callers that want to save it should
    POST the YAML to the ontology endpoint explicitly.

    Args:
        transform_id: Completed transform to analyze.

    Returns:
        JSON: {ontology_yaml: str, ontology: dict, stats: {...}}
    """
    from graphora_server.services.schema_postprocess import (
        infer_ontology_from_graph,
        ontology_dict_to_yaml,
    )
    from graphora_server.services.storage.memory import InMemoryStorage
    from graphora_server.services.user_db_service import (
        is_memory_storage_enabled,
    )
    from graphora_server.services.storage.factory import user_has_staging_db

    graph_service = None
    try:
        use_in_memory = is_memory_storage_enabled() or not await user_has_staging_db(
            user_id
        )
        if use_in_memory:
            storage = InMemoryStorage(user_id=user_id)
            graph = await storage.get_transformation_data(transform_id)
        else:
            graph_service = await UserDatabaseService.get_staging_graph_service(user_id)
            graph = graph_service.get_graph_by_transform_id(
                transform_id=transform_id, limit=10000, skip=0
            )

        if not graph.nodes:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transform {transform_id} has no extracted nodes. "
                    "Wait for extraction to complete before inferring an ontology."
                ),
            )

        nodes_payload = [n.model_dump() for n in graph.nodes]
        edges_payload = [e.model_dump() for e in graph.edges]

        ontology_dict = await infer_ontology_from_graph(
            nodes=nodes_payload,
            edges=edges_payload,
            user_id=user_id,
        )
        yaml_content = ontology_dict_to_yaml(ontology_dict)

        return JSONResponse(
            {
                "transform_id": transform_id,
                "ontology_yaml": yaml_content,
                "ontology": ontology_dict,
                "stats": {
                    "node_count": len(nodes_payload),
                    "edge_count": len(edges_payload),
                    "entity_types": len(ontology_dict.get("entities", {})),
                    "relationship_types": len(ontology_dict.get("relationships", {})),
                },
            }
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Inferred-ontology request rejected for %s: %s", user_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except AIConfigurationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        traceback.print_exc()
        logger.error("Inferred-ontology failed for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=500, detail=f"Failed to infer ontology: {str(e)}"
        )
    finally:
        if graph_service is not None:
            graph_service.close()


@router.post(
    "/transform/schemaless/upload",
    response_model=TransformInitResponse,
    summary="Extract with generic schema, refine ontology post-hoc",
)
async def upload_documents_schemaless(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    files: List[UploadFile] = File(...),
    chunking_config: Optional[str] = Form(
        None, description="JSON string of chunking configuration"
    ),
) -> TransformInitResponse:
    """Extract first, refine the ontology after — no pre-extraction schema bias.

    Unlike `/transform/upload` (which peeks at document text to
    infer an ontology *before* extraction), this endpoint uses
    the default generic schema (`Person`, `Organization`,
    `Concept`, `Entity`) to run extraction, then leaves ontology
    refinement to the agent or user via:

        GET  /transform/{id}/inferred-ontology   (preview)
        POST /transform/{id}/finalize-ontology   (save refinement)

    Effect on the pipeline: the LLM is **not** biased by a
    pre-inferred category list — it extracts into broad buckets
    and the specific refined types emerge from what was actually
    surfaced. Use this when you don't trust pre-extraction schema
    inference to anticipate the corpus.

    **Related modes**:

    - `POST /transform/upload` — auto-infer ontology *before*
      extracting (faster, biases toward what's expected).
    - `POST /transform/{ontology_id}/upload` — when you already
      have a pre-registered ontology.

    Example:

        curl -X POST /api/v1/transform/schemaless/upload \\
            -F "files=@doc.pdf"
    """
    from graphora_server.services.schema_inference import get_default_generic_schema
    from graphora_server.services.ontology_storage_service import (
        ontology_storage_service,
    )

    start_time = time.time()
    temp_dir = Path(settings.UPLOAD_DIR)
    transform_id = f"transform_{uuid.uuid4().hex}"
    audit_id = ""

    try:
        logger.info("Starting schemaless upload for user: %s", user_id)

        # B5-obs slice 2: preflight budget check. Same enforcement
        # contract as the other two transform-upload endpoints.
        await enforce_budget_preflight(user_id)

        parsed_chunking_config = None
        if chunking_config:
            try:
                config_dict = json.loads(chunking_config)
                parsed_chunking_config = ChunkingConfig(**config_dict)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Invalid chunking config: %s. Using defaults.", e)

        audit_id = await audit_service.log_operation_start(
            user_id=user_id,
            operation_type=OperationType.TRANSFORM_STARTED,
            operation_id=transform_id,
            resource_name=f"Schemaless {transform_id[:8]}",
            metadata={
                "schemaless": True,
                "files_count": len(files),
                "file_names": [f.filename for f in files],
            },
        )

        await progress_tracker.initialize_transform(transform_id)

        validator = FileValidator()
        temp_dir = Path(settings.UPLOAD_DIR) / transform_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_paths: List[Path] = []
        doc_metadata: List[DocumentMetadata] = []
        total_file_size = 0
        doc_info: Optional[DocumentInfo] = None

        for file in files:
            try:
                safe_filename = sanitize_filename(file.filename)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            validation_result = await validator.validate(file)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file {safe_filename}: {validation_result.errors}",
                )

            temp_path = temp_dir / safe_filename
            async with aiofiles.open(temp_path, "wb") as fh:
                content = await file.read()
                await file.seek(0)
                await fh.write(content)

            file_paths.append(temp_path)
            total_file_size += len(content)

            metadata = DocumentMetadata(
                source=safe_filename,
                document_type=DocumentType(Path(safe_filename).suffix[1:]),
                tags=["schemaless", user_id],
            )
            doc_metadata.append(metadata)
            doc_info = DocumentInfo(
                filename=safe_filename,
                size=len(content),
                document_type=metadata.document_type,
                metadata=metadata,
            )

        await progress_tracker.complete_stage(transform_id, TransformationStage.UPLOAD)

        # Always use the permissive generic schema. No pre-extraction
        # text sampling — that's what separates schemaless from
        # auto-schema.
        generic_yaml = get_default_generic_schema()
        ontology_id = f"schemaless_{uuid.uuid4().hex[:12]}"
        await ontology_storage_service.store_ontology(
            user_id=user_id,
            ontology_id=ontology_id,
            yaml_content=generic_yaml,
            name=f"Schemaless base ({transform_id[:8]})",
            description=(
                "Permissive generic schema used for schemaless extraction. "
                "Run /finalize-ontology after extraction to get a refined version."
            ),
        )

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

        upload_duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_success(
                audit_id=audit_id,
                duration_ms=upload_duration_ms,
                metadata={
                    "transform_id": transform_id,
                    "ontology_id": ontology_id,
                    "schemaless": True,
                    "upload_completed": True,
                    "total_file_size_bytes": total_file_size,
                },
            )

        return TransformInitResponse(
            id=transform_id,
            upload_timestamp=datetime.now(timezone.utc),
            status=TransformStatus.PENDING,
            document_info=doc_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Schemaless upload failed for user %s: %s", user_id, e)
        traceback.print_exc()
        upload_duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_failure(
                audit_id=audit_id,
                error_message=str(e),
                duration_ms=upload_duration_ms,
            )
        try:
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            logger.warning("Failed to cleanup temp %s: %s", temp_dir, cleanup_err)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process schemaless upload: {str(e)}",
        )


class FinalizeOntologyRequest(BaseModel):
    """Optional request body for the finalize-ontology endpoint.

    When ``yaml_content`` is provided the server skips post-hoc
    inference and persists the supplied YAML verbatim — this is the
    UX path for "I edited the inferred ontology and want my edits
    saved, not re-overwritten by a fresh LLM call." When omitted
    (default) the server runs inference itself, mirroring the
    original endpoint behaviour.
    """

    yaml_content: Optional[str] = Field(
        default=None,
        description=(
            "User-edited ontology YAML. When set, persisted as-is "
            "after a basic shape check; LLM inference is skipped. "
            "When omitted, the server runs post-hoc inference on the "
            "extracted graph and persists that result."
        ),
    )


@router.post(
    "/transform/{transform_id}/finalize-ontology",
    response_class=JSONResponse,
)
async def finalize_inferred_ontology(
    transform_id: str,
    body: Optional[FinalizeOntologyRequest] = Body(default=None),
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """Persist a refined ontology for a completed transform.

    Two modes, controlled by the request body:

    * **No body / empty body** — runs post-hoc ontology inference on
      the extracted graph (LLM round-trip) and persists the result.
      Same behaviour as the original endpoint.
    * **Body with ``yaml_content``** — skips inference and persists
      the user-supplied YAML verbatim after a basic shape check.

    Returns the new ``ontology_id`` + full YAML + stats in both cases.
    """
    import yaml as yaml_lib

    from graphora_server.services.schema_postprocess import (
        infer_ontology_from_graph,
        ontology_dict_to_yaml,
    )
    from graphora_server.services.storage.memory import InMemoryStorage
    from graphora_server.services.user_db_service import is_memory_storage_enabled
    from graphora_server.services.storage.factory import user_has_staging_db
    from graphora_server.services.ontology_storage_service import (
        ontology_storage_service,
    )

    # Treat whitespace-only yaml_content the same as "no body" so an
    # accidentally-empty form field doesn't fail validation; preserve
    # the user's exact YAML otherwise (no strip — trailing newlines
    # round-trip cleanly through the ontology store).
    user_supplied_yaml = (
        body.yaml_content
        if body and body.yaml_content and body.yaml_content.strip()
        else None
    )

    graph_service = None
    try:
        use_in_memory = is_memory_storage_enabled() or not await user_has_staging_db(
            user_id
        )
        if use_in_memory:
            storage = InMemoryStorage(user_id=user_id)
            graph = await storage.get_transformation_data(transform_id)
        else:
            graph_service = await UserDatabaseService.get_staging_graph_service(user_id)
            graph = graph_service.get_graph_by_transform_id(
                transform_id=transform_id, limit=10000, skip=0
            )

        if not graph.nodes:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transform {transform_id} has no extracted nodes. "
                    "Cannot finalize ontology on an empty graph."
                ),
            )

        nodes_payload = [n.model_dump() for n in graph.nodes]
        edges_payload = [e.model_dump() for e in graph.edges]

        if user_supplied_yaml is not None:
            # Validate the YAML the user wants to persist. Mirrors the
            # shape check schema_postprocess applies to LLM output:
            # must parse, must be a dict, must have a non-empty
            # ``entities`` key. Other shape validation is left to the
            # downstream ontology storage / quality services so this
            # endpoint stays focused on the YAML→stored-row contract.
            try:
                ontology_dict = yaml_lib.safe_load(user_supplied_yaml)
            except yaml_lib.YAMLError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provided yaml_content is not valid YAML: {exc}",
                )
            if not isinstance(ontology_dict, dict):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Provided yaml_content must parse to a YAML mapping; "
                        f"got {type(ontology_dict).__name__}."
                    ),
                )
            if not ontology_dict.get("entities"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Provided yaml_content has no 'entities' key (or it is "
                        "empty). At least one entity type is required to "
                        "persist as an ontology."
                    ),
                )
            ontology_dict.setdefault("version", "0.1.0")
            ontology_dict.setdefault("relationships", {})
            yaml_content = user_supplied_yaml
        else:
            ontology_dict = await infer_ontology_from_graph(
                nodes=nodes_payload,
                edges=edges_payload,
                user_id=user_id,
            )
            yaml_content = ontology_dict_to_yaml(ontology_dict)

        new_ontology_id = f"auto_refined_{uuid.uuid4().hex[:12]}"
        await ontology_storage_service.store_ontology(
            user_id=user_id,
            ontology_id=new_ontology_id,
            yaml_content=yaml_content,
            name=f"Refined ({transform_id[:8]})",
            description=(
                "User-edited ontology persisted from transform " f"{transform_id}"
                if user_supplied_yaml is not None
                else f"Post-hoc inferred ontology from transform {transform_id}"
            ),
        )

        return JSONResponse(
            {
                "transform_id": transform_id,
                "ontology_id": new_ontology_id,
                "ontology_yaml": yaml_content,
                "ontology": ontology_dict,
                "source": "user_edit" if user_supplied_yaml is not None else "inferred",
                "stats": {
                    "node_count": len(nodes_payload),
                    "edge_count": len(edges_payload),
                    "entity_types": len(ontology_dict.get("entities", {})),
                    "relationship_types": len(
                        ontology_dict.get("relationships", {}) or {}
                    ),
                },
            }
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Finalize-ontology rejected for %s: %s", user_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except AIConfigurationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        traceback.print_exc()
        logger.error("Finalize-ontology failed for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=500, detail=f"Failed to finalize ontology: {str(e)}"
        )
    finally:
        if graph_service is not None:
            graph_service.close()
