import logging
import time
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.schema import (
    SchemaGenerationRequest,
    SchemaGenerationResponse,
    SchemaSearchRequest,
    SchemaSearchResponse,
    QuestionConfigResponse,
    SchemaRefinementRequest,
    SchemaRefinementResponse,
    CreateSchemaRequest,
    UpdateSchemaRequest,
    StoredSchema,
)
from app.services.schema_generation_service import schema_generation_service
from app.services.schema_search_service import schema_search_service
from app.services.schema_storage_service import schema_storage_service
from app.services.audit_service import audit_service, OperationType
from app.config import settings
from app.auth import AuthContext, get_current_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.API_V1_STR, tags=["Schema Generation"])


@router.post("/schema/generate", response_model=SchemaGenerationResponse)
async def generate_schema(
    request: SchemaGenerationRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> SchemaGenerationResponse:
    """Generate a schema based on user responses and context"""

    start_time = time.time()
    operation_id = str(uuid.uuid4())
    audit_id = ""

    try:
        # Start audit trail
        audit_id = await audit_service.log_operation_start(
            user_id=auth.user_id,
            operation_type=OperationType.SCHEMA_GENERATION,
            operation_id=operation_id,
            resource_name="Schema Generation",
            metadata={
                "response_count": len(request.user_responses),
                "context": request.context.model_dump() if request.context else None,
            },
        )

        # Generate schema
        result = await schema_generation_service.generate_schema(
            user_id=auth.user_id, request=request
        )

        # Log success
        duration_ms = int((time.time() - start_time) * 1000)
        await audit_service.log_operation_success(
            audit_id=audit_id,
            duration_ms=duration_ms,
            metadata={
                "schema_id": result.id,
                "confidence": result.confidence,
                "related_schemas_count": len(result.related_schemas or []),
            },
        )

        logger.info(
            "Generated schema %s for user %s (confidence: %.2f, time: %sms)",
            result.id,
            auth.user_id,
            result.confidence,
            duration_ms,
        )

        return result

    except Exception as e:
        # Log failure
        if audit_id:
            duration_ms = int((time.time() - start_time) * 1000)
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=duration_ms
            )

        logger.error("Error generating schema for user %s: %s", auth.user_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to generate schema")


@router.post("/schema/search", response_model=SchemaSearchResponse)
async def search_schemas(
    request: SchemaSearchRequest, auth: AuthContext = Depends(get_current_auth)
) -> SchemaSearchResponse:
    """Search for schemas using text or vector similarity"""
    user_id = auth.user_id

    start_time = time.time()
    operation_id = str(uuid.uuid4())
    audit_id = ""

    try:
        # Start audit trail
        audit_id = await audit_service.log_operation_start(
            user_id=user_id,
            operation_type=OperationType.SCHEMA_SEARCH,
            operation_id=operation_id,
            resource_name="Schema Search",
            metadata={
                "query": request.query,
                "domain": request.domain,
                "limit": request.limit,
            },
        )

        # Search schemas
        result = await schema_search_service.search_schemas(
            user_id=user_id,
            query=request.query,
            domain=request.domain,
            limit=request.limit,
            threshold=request.threshold,
            include_content=request.include_content,
        )

        # Log success
        duration_ms = int((time.time() - start_time) * 1000)
        await audit_service.log_operation_success(
            audit_id=audit_id,
            duration_ms=duration_ms,
            metadata={"results_count": result.total, "took_ms": result.took_ms},
        )

        return result

    except Exception as e:
        # Log failure
        if audit_id:
            duration_ms = int((time.time() - start_time) * 1000)
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=duration_ms
            )

        logger.error("Error searching schemas for user %s: %s", auth.user_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to search schemas")


@router.get("/schema/search", response_model=SchemaSearchResponse)
async def get_popular_schemas(
    domain: Optional[str] = Query(None, description="Domain filter"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    auth: AuthContext = Depends(get_current_auth),
) -> SchemaSearchResponse:
    """Get popular schemas, optionally filtered by domain"""

    start_time = time.time()

    try:
        # Get popular schemas
        results = await schema_search_service.get_popular_schemas_by_domain(
            domain=domain, limit=limit
        )

        took_ms = int((time.time() - start_time) * 1000)

        return SchemaSearchResponse(
            results=results,
            total=len(results),
            query=f"popular_{domain}" if domain else "popular",
            took_ms=took_ms,
        )

    except Exception as e:
        logger.error(f"Error getting popular schemas: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get popular schemas")


@router.post("/schema/refine", response_model=SchemaRefinementResponse)
async def refine_schema(
    request: SchemaRefinementRequest, auth: AuthContext = Depends(get_current_auth)
) -> SchemaRefinementResponse:
    """Refine an existing schema based on user feedback"""
    user_id = auth.user_id

    start_time = time.time()
    operation_id = str(uuid.uuid4())
    audit_id = ""

    try:
        # Start audit trail
        audit_id = await audit_service.log_operation_start(
            user_id=user_id,
            operation_type=OperationType.SCHEMA_REFINEMENT,
            operation_id=operation_id,
            resource_name="Schema Refinement",
            metadata={
                "schema_id": request.schema_id,
                "feedback_length": len(request.user_feedback),
            },
        )

        # Refine schema
        refined_schema, changes_made, confidence, explanation = (
            await schema_generation_service.refine_schema(
                user_id=user_id,
                schema_id=request.schema_id,
                current_schema=request.current_schema,
                user_feedback=request.user_feedback,
                context=request.context,
            )
        )

        # Prepare response
        result = SchemaRefinementResponse(
            refined_schema=refined_schema,
            changes_made=changes_made,
            confidence=confidence,
            explanation=explanation,
        )

        # Log success
        duration_ms = int((time.time() - start_time) * 1000)
        await audit_service.log_operation_success(
            audit_id=audit_id,
            duration_ms=duration_ms,
            metadata={"confidence": confidence, "changes_count": len(changes_made)},
        )

        return result

    except Exception as e:
        # Log failure
        if audit_id:
            duration_ms = int((time.time() - start_time) * 1000)
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=duration_ms
            )

        logger.error("Error refining schema for user %s: %s", auth.user_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to refine schema")


@router.get("/schema/questions", response_model=QuestionConfigResponse)
async def get_question_configuration(
    domain: Optional[str] = Query(None, description="Domain to get questions for"),
    include_optional: bool = Query(True, description="Include optional questions"),
    auth: AuthContext = Depends(get_current_auth),
) -> QuestionConfigResponse:
    """Get the configuration of questions for schema generation"""

    try:
        from app.services.question_sets import get_question_sets_for_domain

        # Get filtered question sets
        question_sets = get_question_sets_for_domain(
            domain=domain, include_optional=include_optional
        )

        return QuestionConfigResponse(
            question_sets=question_sets,
            metadata={
                "domain": domain,
                "include_optional": include_optional,
                "total_sets": len(question_sets),
            },
        )

    except Exception as e:
        logger.error(f"Error getting question configuration: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to get question configuration"
        )


# Schema CRUD endpoints
@router.post("/schema", response_model=StoredSchema)
async def create_schema(
    request: CreateSchemaRequest, auth: AuthContext = Depends(get_current_auth)
) -> StoredSchema:
    """Create a new schema"""
    user_id = auth.user_id

    start_time = time.time()
    operation_id = str(uuid.uuid4())
    audit_id = ""

    try:
        # Start audit trail
        audit_id = await audit_service.log_operation_start(
            user_id=user_id,
            operation_type=OperationType.SCHEMA_CREATE,
            operation_id=operation_id,
            resource_name="Schema Creation",
            metadata={
                "title": request.title,
                "domain": request.domain,
                "is_public": request.is_public,
            },
        )

        # Create schema
        result = await schema_storage_service.create_schema(
            user_id=user_id, request=request
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to create schema")

        # Log success
        duration_ms = int((time.time() - start_time) * 1000)
        await audit_service.log_operation_success(
            audit_id=audit_id,
            duration_ms=duration_ms,
            metadata={"schema_id": result.id},
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        # Log failure
        if audit_id:
            duration_ms = int((time.time() - start_time) * 1000)
            await audit_service.log_operation_failure(
                audit_id=audit_id, error_message=str(e), duration_ms=duration_ms
            )

        logger.error("Error creating schema for user %s: %s", auth.user_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to create schema")


@router.get("/schema/{schema_id}", response_model=StoredSchema)
async def get_schema(
    schema_id: str, auth: AuthContext = Depends(get_current_auth)
) -> StoredSchema:
    """Get a specific schema by ID"""

    try:
        result = await schema_storage_service.get_schema(
            schema_id=schema_id, user_id=auth.user_id
        )

        if not result:
            raise HTTPException(status_code=404, detail="Schema not found")

        # Log usage event
        await schema_storage_service.log_usage_event(
            schema_id=schema_id, user_id=auth.user_id, event_type="view"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting schema {schema_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get schema")


@router.put("/schema/{schema_id}", response_model=StoredSchema)
async def update_schema(
    schema_id: str,
    request: UpdateSchemaRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> StoredSchema:
    """Update an existing schema"""
    user_id = auth.user_id

    try:
        result = await schema_storage_service.update_schema(
            schema_id=schema_id, user_id=user_id, request=request
        )

        if not result:
            raise HTTPException(
                status_code=404, detail="Schema not found or not authorized"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating schema {schema_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update schema")


@router.delete("/schema/{schema_id}")
async def delete_schema(
    schema_id: str, auth: AuthContext = Depends(get_current_auth)
) -> dict:
    """Delete a schema"""
    user_id = auth.user_id

    try:
        success = await schema_storage_service.delete_schema(
            schema_id=schema_id, user_id=user_id
        )

        if not success:
            raise HTTPException(
                status_code=404, detail="Schema not found or not authorized"
            )

        return {"message": "Schema deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting schema {schema_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete schema")


@router.get("/schemas", response_model=list[StoredSchema])
async def list_user_schemas(
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    include_public: bool = Query(True, description="Include public schemas"),
    auth: AuthContext = Depends(get_current_auth),
) -> list[StoredSchema]:
    """List schemas for the current user"""
    user_id = auth.user_id

    try:
        results = await schema_storage_service.list_user_schemas(
            user_id=user_id, limit=limit, offset=offset, include_public=include_public
        )

        return results

    except Exception as e:
        logger.error("Error listing schemas for user %s: %s", auth.user_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to list schemas")


@router.get("/schema/{schema_id}/related", response_model=list[StoredSchema])
async def get_related_schemas(
    schema_id: str,
    limit: int = Query(5, ge=1, le=20, description="Maximum results"),
    auth: AuthContext = Depends(get_current_auth),
) -> list[StoredSchema]:
    """Get schemas related to a specific schema"""

    try:
        results = await schema_search_service.get_related_schemas(
            schema_id=schema_id, user_id=auth.user_id, limit=limit
        )

        # Convert SchemaSearchResult to StoredSchema format
        # This is a simplified conversion - in practice you might want a different response model
        related_schemas = []
        for result in results:
            # Note: This creates a partial StoredSchema - you might want a dedicated response model
            schema = StoredSchema(
                id=result.id,
                title=result.title,
                description=result.description,
                content=result.content or "",
                domain=result.domain,
                tags=result.tags,
                user_id=result.user_id,
                is_public=True,  # Assume public if found in search
                usage_count=result.usage_count,
                created_at=result.created_at,
                updated_at=result.updated_at,
            )
            related_schemas.append(schema)

        return related_schemas

    except Exception as e:
        logger.error(f"Error getting related schemas for {schema_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get related schemas")
