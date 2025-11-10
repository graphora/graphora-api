import time
import logging
from app.services.transform.ontology_helper import OntologyParser
from fastapi import APIRouter, Depends, HTTPException
from uuid import uuid4
from pathlib import Path
from app.config import settings
from app.schemas.ontology import OntologyRequest, OntologyResponse
from app.services.ontology_validator import (
    parse_and_validate_yaml,
    OntologyValidationError,
)
from app.services.audit_service import audit_service, OperationType
from app.services.ontology_storage_service import ontology_storage_service
from app.auth import get_current_user_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix=settings.API_V1_STR, tags=["Ontology"])


def ensure_ontology_dir():
    """Ensure ontology directory exists"""
    Path(settings.ontology_dir).expanduser().mkdir(parents=True, exist_ok=True)


@router.post("/ontology", response_model=OntologyResponse)
async def validate_ontology(
    request: OntologyRequest, user_id: str = Depends(get_current_user_id)
) -> OntologyResponse:
    """
    Validate and process ontology YAML.

    Parameters:
    - request: Ontology request containing YAML text
    - user_id: User's ID (from header)

    Returns:
    - id: Unique ID for the validated ontology
    """
    start_time = time.time()
    ontology_id = str(uuid4())
    audit_id = ""

    try:
        # Start audit trail
        audit_id = await audit_service.log_operation_start(
            user_id=user_id,
            operation_type=OperationType.ONTOLOGY_STORED,
            operation_id=ontology_id,
            resource_name="Ontology",
            metadata={"size_bytes": len(request.text)},
        )

        # Parse and validate YAML
        parse_and_validate_yaml(request.text)

        # Store ontology in Supabase
        ontology_name = request.name if request.name else f"Ontology {ontology_id[:8]}"
        success = await ontology_storage_service.store_ontology(
            user_id=user_id,
            ontology_id=ontology_id,
            yaml_content=request.text,
            name=ontology_name,
            description="User-defined ontology",
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to store ontology")

        # Create file backup for backward compatibility
        await ontology_storage_service.create_file_backup(user_id, ontology_id)

        # Create Full Text Indexes for entities defined in Ontology
        # Use user's specific database configurations
        try:
            # For file backup compatibility, still create the file path for OntologyParser
            ontology_path = (
                Path(settings.ontology_dir).expanduser() / f"{ontology_id}.yaml"
            )
            await OntologyParser(
                ontology_path, user_id
            ).build_full_text_indexes_for_user(user_id)
        except ValueError as db_error:
            # User doesn't have database configuration set up yet
            # This is okay - we can still validate and save the ontology
            # The indexes will be created when the user sets up their databases
            print(
                f"Warning: Could not create full-text indexes for user {user_id}: {str(db_error)}"
            )

        # Log success
        duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_success(
                audit_id=audit_id,
                duration_ms=duration_ms,
                metadata={"ontology_id": ontology_id},
            )

        return OntologyResponse(id=ontology_id)

    except OntologyValidationError as e:
        # Log validation failure
        duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_failure(
                audit_id=audit_id,
                error_message=f"Validation error: {str(e)}",
                duration_ms=duration_ms,
            )

        raise HTTPException(status_code=400, detail=f"Invalid ontology: {str(e)}")
    except Exception as e:
        # Log general failure
        duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=duration_ms
            )

        raise HTTPException(
            status_code=500, detail=f"Error processing ontology: {str(e)}"
        )


@router.get("/ontology/{ontology_id}", response_model=OntologyRequest)
async def get_ontology(
    ontology_id: str, user_id: str = Depends(get_current_user_id)
) -> OntologyRequest:
    """
    Get ontology by ID from Supabase database only

    Parameters:
    - ontology_id: ID of the ontology to retrieve
    - user_id: User's ID (from header)

    Returns:
    - text: Ontology YAML text
    """
    try:
        # Get from Supabase database only
        ontology = await ontology_storage_service.get_ontology(user_id, ontology_id)

        if ontology:
            return OntologyRequest(text=ontology["yaml_content"])

        raise HTTPException(status_code=404, detail=f"Ontology {ontology_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving ontology: {str(e)}"
        )


@router.get("/ontologies")
async def list_ontologies(user_id: str = Depends(get_current_user_id)):
    """
    List all ontologies for a user from Supabase database only
    """
    try:
        ontologies = await ontology_storage_service.list_ontologies(user_id)

        for ontology in ontologies:
            ontology["source"] = "database"
            ontology.setdefault("file_name", f"{ontology['id']}.yaml")
            ontology.setdefault("metadata", {})

        return {"ontologies": ontologies, "total": len(ontologies)}

    except Exception as e:
        logger.error(f"Error listing ontologies: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing ontologies: {str(e)}"
        )


@router.get("/ontologies/{ontology_id}")
async def get_ontology_by_id(
    ontology_id: str, user_id: str = Depends(get_current_user_id)
):
    """
    Get a specific ontology by ID from Supabase database only
    """
    try:
        ontology = await ontology_storage_service.get_ontology(user_id, ontology_id)

        if ontology:
            return {
                "id": ontology["id"],
                "name": ontology.get("name", f"Ontology {ontology['id'][:8]}"),
                "file_name": ontology.get("file_name") or f"{ontology['id']}.yaml",
                "yaml_content": ontology["yaml_content"],
                "version": ontology["version"],
                "metadata": ontology.get("metadata", {}),
                "source": "database",
                "created_at": ontology.get("created_at"),
                "updated_at": ontology.get("updated_at"),
            }

        raise HTTPException(
            status_code=404, detail=f"Ontology with ID {ontology_id} not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ontology {ontology_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting ontology: {str(e)}")


@router.delete("/ontology/{ontology_id}")
async def delete_ontology(
    ontology_id: str, user_id: str = Depends(get_current_user_id)
):
    """
    Delete an ontology

    Parameters:
    - ontology_id: ID of the ontology to delete
    - user_id: User's ID (from header)
    """
    try:
        success = await ontology_storage_service.delete_ontology(user_id, ontology_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Ontology {ontology_id} not found"
            )

        return {"message": "Ontology deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting ontology: {str(e)}"
        )


@router.put("/ontology/{ontology_id}", response_model=OntologyResponse)
async def update_ontology(
    ontology_id: str,
    request: OntologyRequest,
    user_id: str = Depends(get_current_user_id),
) -> OntologyResponse:
    """
    Update an existing ontology by ID.

    Parameters:
    - ontology_id: ID of the ontology to update
    - request: Ontology request containing YAML text
    - user_id: User's ID (from header)

    Returns:
    - id: ID of the updated ontology
    """
    start_time = time.time()
    audit_id = ""

    try:
        # Check if ontology exists and belongs to user
        existing = await ontology_storage_service.get_ontology(user_id, ontology_id)
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Ontology {ontology_id} not found"
            )

        # Start audit trail
        audit_id = await audit_service.log_operation_start(
            user_id=user_id,
            operation_type=OperationType.ONTOLOGY_STORED,
            operation_id=ontology_id,
            resource_name="Ontology",
            metadata={"size_bytes": len(request.text), "update": True},
        )

        # Parse and validate YAML
        parse_and_validate_yaml(request.text)

        # Update ontology in Supabase with versioning
        # Use the provided name if available, otherwise keep the existing name
        updated_name = request.name if request.name else existing.get("name")
        success = await ontology_storage_service.store_ontology(
            user_id=user_id,
            ontology_id=ontology_id,
            yaml_content=request.text,
            name=updated_name,
            description=existing.get("description"),
            is_update=True,  # This will create a new version and deactivate the old one
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update ontology")

        # Update file backup for backward compatibility
        await ontology_storage_service.create_file_backup(user_id, ontology_id)

        # Update Full Text Indexes for entities defined in Ontology
        try:
            ontology_path = (
                Path(settings.ontology_dir).expanduser() / f"{ontology_id}.yaml"
            )
            await OntologyParser(
                ontology_path, user_id
            ).build_full_text_indexes_for_user(user_id)
        except ValueError as db_error:
            print(
                f"Warning: Could not update full-text indexes for user {user_id}: {str(db_error)}"
            )

        # Log success
        duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_success(
                audit_id=audit_id,
                duration_ms=duration_ms,
                metadata={"ontology_id": ontology_id, "updated": True},
            )

        return OntologyResponse(id=ontology_id)

    except OntologyValidationError as e:
        # Log validation failure
        duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_failure(
                audit_id=audit_id,
                error_message=f"Validation error: {str(e)}",
                duration_ms=duration_ms,
            )

        raise HTTPException(status_code=400, detail=f"Invalid ontology: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        # Log general failure
        duration_ms = int((time.time() - start_time) * 1000)
        if audit_id:
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=duration_ms
            )

        raise HTTPException(
            status_code=500, detail=f"Error updating ontology: {str(e)}"
        )


@router.get("/ontology/{ontology_id}/versions")
async def get_ontology_versions(
    ontology_id: str, user_id: str = Depends(get_current_user_id)
):
    """
    Get version history for a specific ontology

    Parameters:
    - ontology_id: ID of the ontology
    - user_id: User's ID (from header)

    Returns:
    - List of ontology versions
    """
    try:
        versions = await ontology_storage_service.get_ontology_versions(
            user_id, ontology_id
        )

        return {
            "ontology_id": ontology_id,
            "versions": versions,
            "total_versions": len(versions),
        }

    except Exception as e:
        logger.error(f"Error getting ontology versions {ontology_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting ontology versions: {str(e)}"
        )
