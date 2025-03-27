"""API endpoints for merge operations"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body, Query
import logging
import uuid
from fastapi import status
import traceback

from app.api.graph import get_staging_graph_service
from app.schemas.graph import GraphResponse
from app.services.graph_service import GraphService
from app.services.merge.service import MergeService, merge_flow
from app.services.merge.models import MergeInitResponse, MergeStatus, MergeProgress, VerificationResult
from app.schemas.conflicts import (
    Conflict, ConflictListResponse, ConflictResolutionRequest,
    ConflictResolutionResponse
)
from app.dependencies import get_progress_tracker, get_merge_service
from app.config import settings
from pydantic import BaseModel, Field
from app.schemas.merge import (
    MergeStatisticsResponse,
    VerificationResultResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/merge",
    tags=["Merge"]
)

class StartMergeRequest(BaseModel):
    """Request model for starting a merge"""
    ontology_id: Optional[str] = None

@router.post("/{session_id}/{transform_id}/start",
            response_model=MergeInitResponse,
            description="Start graph merge process")
async def start_merge(
    session_id: str,
    transform_id: str,
    background_tasks: BackgroundTasks
) -> MergeInitResponse:
    """Start a new merge process"""
    try:
        # Validate inputs
        if not session_id or not transform_id:
            raise HTTPException(
                status_code=400,
                detail="session_id and transform_id are required"
            )
            
        # Generate merge ID
        merge_id = str(uuid.uuid4())
        
        # Initialize progress tracking
        async with get_progress_tracker() as progress_tracker:
            await progress_tracker.initialize_merge(merge_id)
        
        # Define background task
        async def run_merge_flow():
            try:
                # Create flow run directly
                flow_run = await merge_flow(
                    merge_id=merge_id,
                    session_id=session_id,
                    transform_id=transform_id,
                    ontology_id=session_id
                )
                logger.info(f"Started flow run {flow_run} for merge {merge_id}")
            except Exception as e:
                logger.error(f"Failed to start merge flow: {str(e)}")
                async with get_progress_tracker() as progress_tracker:
                    await progress_tracker.fail_merge(merge_id, str(e))
        
        # Add background task
        background_tasks.add_task(run_merge_flow)
        
        return MergeInitResponse(
            merge_id=merge_id,
            status=MergeStatus.PENDING,
            start_time=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to start merge: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start merge: {str(e)}"
        )

@router.get("/{merge_id}/status", response_model=MergeProgress)
async def get_merge_status(merge_id: str) -> MergeProgress:
    """Get current status of a merge process"""
    try:
        async with get_progress_tracker() as progress_tracker:
            status = await progress_tracker.get_progress(merge_id)
            
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Merge {merge_id} not found"
            )
            
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to get merge status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get merge status: {str(e)}"
        )

@router.get(
    "/{merge_id}/conflicts",
    response_model=ConflictListResponse,
    description="Get conflicts for a merge process"
)
async def get_conflicts(
    merge_id: str,
    conflict_type: Optional[str] = None,
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
) -> ConflictListResponse:
    """Get conflicts for a merge process"""
    try:
        async with get_merge_service() as merge_service:
            conflicts, total_count = await merge_service.get_conflicts(
                merge_id=merge_id,
                conflict_type=conflict_type,
                severity=severity,
                resolved=resolved,
                limit=limit,
                offset=offset
            )
            
            summary = await merge_service.get_conflict_summary(merge_id)
            
            return ConflictListResponse(
                merge_id=merge_id,
                conflicts=conflicts,
                total_count=total_count,
                summary=summary,
                limit=limit,
                offset=offset
            )
            
    except Exception as e:
        logger.error(f"Failed to get conflicts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conflicts: {str(e)}"
        )

@router.get(
    "/conflicts/{merge_id}/{conflict_id}",
    response_model=Conflict,
    description="Get detailed information about a specific conflict"
)
async def get_conflict_detail(
    merge_id: str,
    conflict_id: str,
    merge_service: MergeService = Depends(get_merge_service)
) -> Conflict:
    """Get detailed information about a specific conflict"""
    try:
        # Get conflict
        async with get_merge_service() as merge_service:
            conflict = await merge_service.get_conflict(merge_id, conflict_id)
            
            if not conflict:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conflict {conflict_id} not found for merge {merge_id}"
                )
                
            return conflict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conflict: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conflict: {str(e)}"
        )

@router.post(
    "/{merge_id}/auto-resolve",
    response_model=Dict[str, Any],
    description="Automatically resolve minor conflicts"
)
async def auto_resolve_conflicts(
    merge_id: str,
    request: Optional[Dict[str, Any]] = Body(None)
) -> Dict[str, Any]:
    """Automatically resolve minor conflicts"""
    try:
        async with get_merge_service() as merge_service:
            result = await merge_service.auto_resolve_conflicts(merge_id, request)
            return result
    except Exception as e:
        logger.error(f"Auto-resolution failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Auto-resolution failed: {str(e)}"
        )


@router.post(
    "/{merge_id}/conflicts/{conflict_id}/resolve",
    response_model=ConflictResolutionResponse,
    description="Apply a resolution to a specific conflict"
)
async def resolve_conflict(
    merge_id: str,
    conflict_id: str,
    request: ConflictResolutionRequest,
    merge_service: MergeService = Depends(get_merge_service)
) -> ConflictResolutionResponse:
    """
    Apply a resolution to a specific conflict
    
    Parameters:
    - merge_id: ID of the merge process
    - conflict_id: ID of the conflict to resolve
    - request: Resolution request with resolution type and data
    
    Returns:
    - Result of the resolution application
    """
    try:
        async with get_merge_service() as merge_service:
            result = await merge_service.apply_conflict_resolution(
                merge_id=merge_id,
                conflict_id=conflict_id,
                resolution_id=request.resolution_id,
                resolution_type=request.resolution_type,
                resolution_data=request.additional_data,
                resolved_by=request.resolved_by
            )
        
            return ConflictResolutionResponse(
                merge_id=merge_id,
                conflict_id=conflict_id,
                resolution_id=request.resolution_id,
                success=result.success,
                resolved=result.resolved,
                error=result.error
            )
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error resolving conflict: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving conflict: {str(e)}"
        )


@router.post(
    "/resolution/{resolution_id}/feedback",
    description="Update resolution success and feedback"
)
async def update_resolution_feedback(
    resolution_id: str,
    success: bool,
    feedback: Optional[str] = None,
    merge_service: MergeService = Depends(get_merge_service)
) -> Dict[str, bool]:
    """Update success status and feedback for a resolution"""
    async with get_merge_service() as merge_service:    
        result = await merge_service.resolution_history.update_resolution_success(
            resolution_id=resolution_id,
            success=success,
            feedback=feedback
        )
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Resolution {resolution_id} not found"
        )
        
    return {"updated": True}


class SimilarResolutionRequest(BaseModel):
    """Request model for finding similar resolutions"""
    conflict_id: str = Field(..., description="ID of the conflict to find similar resolutions for")
    limit: int = Field(5, description="Maximum number of similar resolutions to return")
    min_similarity: float = Field(0.7, description="Minimum similarity score threshold")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters to apply")


class SimilarResolutionResponse(BaseModel):
    """Response model for similar resolutions"""
    conflict_id: str = Field(..., description="ID of the original conflict")
    similar_resolutions: List[Dict[str, Any]] = Field(..., description="List of similar resolutions")
    total_found: int = Field(..., description="Total number of similar resolutions found")


class BatchSimilarResolutionRequest(BaseModel):
    """Request model for batch similar resolution search"""
    conflict_ids: List[str] = Field(..., description="IDs of conflicts to find similar resolutions for")
    limit_per_conflict: int = Field(5, description="Maximum number of similar resolutions per conflict")
    min_similarity: float = Field(0.7, description="Minimum similarity score threshold")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters to apply")


@router.get(
    "/statistics/{merge_id}",
    response_model=MergeStatisticsResponse,
    description="Get detailed statistics of a merge operation"
)
async def get_merge_statistics(
    merge_id: str,
    merge_service: MergeService = Depends(get_merge_service)
) -> MergeStatisticsResponse:
    """Get detailed statistics of a merge operation"""
    try:
        async with get_merge_service() as merge_service:
            statistics = await merge_service.get_merge_statistics(merge_id)
            if not statistics:
                raise HTTPException(status_code=404, detail=f"Merge {merge_id} not found")
            return statistics
    except ValueError as e:
        # Check if this is a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Merge {merge_id} not found")
        # For other value errors, return a 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error
        logger.error(f"Error getting merge statistics: {str(e)}")
        # Return a 500 error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post(
    "/{merge_id}/{session_id}/{transform_id}/finalise",
    response_model=VerificationResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Finalise a merge operation",
    description="Finalise the merge operation by storing the merged data in the database."
)
async def finalise_merge(
    merge_id: str,
    session_id: str,
    transform_id: str,
    merge_service: MergeService = Depends(get_merge_service),
    graph_service: GraphService = Depends(get_staging_graph_service)
) -> VerificationResult:
    """Finalise a merge operation
    
    Args:
        merge_id: ID of the merge operation
        session_id: ID of the session
        transform_id: ID of the transformation
        
    Returns:
        VerificationResult with verification results
    """
    try:
        # Get merge service
        async with get_merge_service() as merge_service:
            # Verify merge
            verification_result = await merge_service.finalise_and_verify_merge(
                graph_service=graph_service,
                merge_id=merge_id,
                session_id=session_id,
                transform_id=transform_id
            )
            return verification_result
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error verifying merge {merge_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error verifying merge: {str(e)}"
        )


@router.get("/graph/{merge_id}/{transform_id}",
         response_model=GraphResponse,
         description="Retrieve nodes by transform ID and their relationships")
async def get_graph_by_merge_transform_id(
    merge_id: str,
    transform_id: str,
    limit: Optional[int] = 1000,
    skip: Optional[int] = 0,
    graph_service: GraphService = Depends(get_staging_graph_service)
) -> GraphResponse:
    try:
        # Validate inputs
        if limit < 0 or skip < 0:
            raise HTTPException(
                status_code=400,
                detail="Limit and skip must be non-negative"
            )
        
        if limit > 10000:
            raise HTTPException(
                status_code=400,
                detail="Maximum limit is 10000 nodes"
            )

        # Get graph data
        async with get_merge_service() as merge_service:
            response = await merge_service.get_merge_graph(
                graph_service=graph_service,
                transform_id=transform_id,
                merge_id=merge_id,
                limit=limit,
                skip=skip
            )
            
            if not response.nodes:
                logger.warning(f"No nodes found with transform_id: {transform_id}")
                
            return response
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error retrieving graph data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving graph data: {str(e)}"
        )