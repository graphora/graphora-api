"""API endpoints for merge operations"""
from datetime import datetime, timezone
from typing import Annotated, Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body, Query
import logging
import uuid
from prefect import get_client

from app.services.merge.service import MergeService, merge_flow
from app.services.merge.models import MergeInitResponse, MergeStatus, MergeProgress, MergeStage
from app.schemas.conflicts import (
    Conflict, ConflictListResponse, ConflictResolutionRequest,
    PendingConflictsResponse, ConflictResolutionResponse,
    BulkResolutionRequest, BulkResolutionResponse,
    ResolutionRequest, ResolutionResult, BatchResolutionRequest, BatchResolutionResult,
    ConflictType
)
from app.dependencies import get_progress_tracker, get_merge_service
from app.config import settings
from pydantic import BaseModel
from app.schemas.resolution_history import ResolutionHistoryEntry

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
    request: StartMergeRequest,
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
                client = get_client()
                # Create flow run directly
                flow_run = await client.create_flow_run(
                    flow=merge_flow,
                    parameters={
                        "merge_id": merge_id,
                        "session_id": session_id,
                        "transform_id": transform_id,
                        "ontology_id": request.ontology_id
                    }
                )
                logger.info(f"Started flow run {flow_run.id} for merge {merge_id}")
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

@router.get("/status/{merge_id}", response_model=MergeProgress)
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
        logger.error(f"Failed to get merge status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get merge status: {str(e)}"
        )

@router.get("/{merge_id}/status", response_model=MergeProgress)
async def get_merge_status_alt(merge_id: str) -> MergeProgress:
    """Get current status of a merge process (alternative path)"""
    return await get_merge_status(merge_id)

@router.get(
    "/conflicts/{merge_id}",
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
    "/{merge_id}/conflicts",
    response_model=ConflictListResponse,
    description="Get conflicts for a merge process (alternative path)"
)
async def get_conflicts_alt(
    merge_id: str,
    conflict_type: Optional[str] = None,
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
) -> ConflictListResponse:
    """Get conflicts for a merge process (alternative path)"""
    return await get_conflicts(
        merge_id=merge_id,
        conflict_type=conflict_type,
        severity=severity,
        resolved=resolved,
        limit=limit,
        offset=offset
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
    "/conflicts/{merge_id}/{conflict_id}/resolve",
    response_model=ResolutionResult,
    description="Apply a resolution to a conflict"
)
async def resolve_conflict(
    merge_id: str,
    conflict_id: str,
    resolution: ResolutionRequest,
    merge_service: MergeService = Depends(get_merge_service)
) -> ResolutionResult:
    """
    Apply a specific resolution to a conflict
    
    Parameters:
    - merge_id: ID of the merge process
    - conflict_id: ID of the conflict to resolve
    - resolution: Resolution request with resolution_id
    
    Returns:
    - Result of the resolution application with verification status
    """
    try:
        result = await merge_service.apply_conflict_resolution(
            conflict_id=conflict_id,
            resolution_id=resolution.resolution_id
        )
        
        return ResolutionResult(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error resolving conflict: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving conflict: {str(e)}"
        )

@router.post("/{merge_id}/conflicts/analyze", response_model=Dict[str, Any])
async def analyze_conflicts_with_llm(
    merge_id: str,
    conflict_ids: Optional[List[str]] = Body(None),
    merge_service: MergeService = Depends(get_merge_service)
):
    """
    Analyze conflicts using LLM and generate intelligent resolution options.
    
    If conflict_ids is provided, only these specific conflicts will be analyzed.
    Otherwise, all unresolved conflicts will be analyzed.
    """
    return await merge_service.analyze_conflicts_with_llm(merge_id, conflict_ids)

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
    "/{merge_id}/select-strategies",
    response_model=Dict[str, Any],
    description="Select resolution strategies for conflicts"
)
async def select_resolution_strategies(
    merge_id: str,
    request: Optional[Dict[str, Any]] = Body(None)
) -> Dict[str, Any]:
    """Select resolution strategies for conflicts"""
    try:
        config = request.get("config") if request else None
        async with get_merge_service() as merge_service:
            result = await merge_service.select_resolution_strategies(merge_id, config)
            return result
    except Exception as e:
        logger.error(f"Strategy selection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Strategy selection failed: {str(e)}"
        )

@router.post(
    "/{merge_id}/apply-strategies",
    response_model=Dict[str, Any],
    description="Apply selected resolution strategies"
)
async def apply_selected_strategies(
    merge_id: str,
    request: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Apply selected resolution strategies"""
    try:
        min_confidence = request.get("min_confidence", 0.7)
        async with get_merge_service() as merge_service:
            result = await merge_service.apply_selected_strategies(merge_id, min_confidence)
            return result
    except Exception as e:
        logger.error(f"Applying strategies failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Applying strategies failed: {str(e)}"
        )

@router.get(
    "/merge/{merge_id}/pending-conflicts",
    response_model=PendingConflictsResponse,
    summary="Get pending conflicts requiring human review",
    description="Retrieves a list of conflicts that require human review, with filtering and pagination options."
)
async def get_pending_conflicts(
    merge_id: str,
    conflict_type: Optional[str] = None,
    severity: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    merge_service: MergeService = Depends(get_merge_service)
):
    """
    Get conflicts that require human review with filtering and pagination.
    
    - **merge_id**: ID of the merge process
    - **conflict_type**: Filter by conflict type
    - **severity**: Filter by severity level
    - **entity_type**: Filter by entity type
    - **limit**: Maximum number of conflicts to return
    - **offset**: Pagination offset
    """
    try:
        # Get conflicts that are not resolved
        conflicts, total_count = await merge_service.get_conflicts(
            merge_id=merge_id,
            conflict_type=conflict_type,
            severity=severity,
            entity_type=entity_type,
            resolved=False,
            limit=limit,
            offset=offset
        )
        
        return PendingConflictsResponse(
            merge_id=merge_id,
            conflicts=conflicts,
            total=total_count,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error getting pending conflicts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pending conflicts: {str(e)}"
        )

@router.post(
    "/conflicts/{merge_id}/batch-resolve",
    response_model=BatchResolutionResult,
    description="Apply multiple resolutions at once"
)
async def batch_resolve_conflicts(
    merge_id: str,
    resolutions: BatchResolutionRequest,
    merge_service: MergeService = Depends(get_merge_service)
) -> BatchResolutionResult:
    """
    Apply multiple resolutions at once
    
    Parameters:
    - merge_id: ID of the merge process
    - resolutions: List of conflict_id/resolution_id pairs to apply
    
    Returns:
    - Summary of resolution application results
    """
    try:
        result = await merge_service.apply_batch_resolutions(
            merge_id=merge_id,
            resolutions=resolutions.resolutions
        )
        
        return BatchResolutionResult(**result)
        
    except Exception as e:
        logger.error(f"Error in batch resolution: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch resolution: {str(e)}"
        )

@router.post(
    "/merge/{merge_id}/conflicts/bulk-resolve",
    response_model=BulkResolutionResponse,
    description="Apply the same resolution to multiple conflicts"
)
async def bulk_resolve_conflicts(
    merge_id: str,
    request: BulkResolutionRequest,
    merge_service: MergeService = Depends(get_merge_service)
) -> BulkResolutionResponse:
    """
    Apply the same resolution to multiple conflicts
    
    Parameters:
    - merge_id: ID of the merge process
    - request: Bulk resolution request with conflict IDs and resolution type
    
    Returns:
    - Summary of resolution application results
    """
    try:
        results = await merge_service.apply_bulk_conflict_resolution(
            merge_id=merge_id,
            conflict_ids=request.conflict_ids,
            resolution_type=request.resolution_type,
            resolution_data=request.additional_data,
            resolved_by=request.resolved_by
        )
        
        # Count resolved conflicts
        resolved_count = sum(1 for r in results if r.resolved)
        
        return BulkResolutionResponse(
            merge_id=merge_id,
            total=len(request.conflict_ids),
            resolved=resolved_count,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error in bulk resolution: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in bulk resolution: {str(e)}"
        )

@router.post(
    "/merge/{merge_id}/conflicts/{conflict_id}/resolve",
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
        result = await merge_service.apply_conflict_resolution(
            merge_id=merge_id,
            conflict_id=conflict_id,
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
        logger.error(f"Error resolving conflict: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving conflict: {str(e)}"
        )

@router.get(
    "/resolution/history",
    response_model=List[ResolutionHistoryEntry],
    description="Get resolution history"
)
async def get_resolution_history(
    merge_id: Optional[str] = None,
    conflict_type: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    merge_service: MergeService = Depends(get_merge_service)
) -> List[ResolutionHistoryEntry]:
    """Get resolution history with optional filtering"""
    # Convert string to enum if provided
    conflict_type_enum = None
    if conflict_type:
        try:
            conflict_type_enum = ConflictType(conflict_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid conflict type: {conflict_type}"
            )
    
    return await merge_service.resolution_history.get_resolution_history(
        merge_id=merge_id,
        conflict_type=conflict_type_enum,
        entity_type=entity_type,
        limit=limit,
        offset=offset
    )

@router.get(
    "/resolution/stats",
    description="Get resolution statistics"
)
async def get_resolution_stats(
    merge_service: MergeService = Depends(get_merge_service)
) -> Dict[str, Any]:
    """Get statistics about resolutions"""
    return await merge_service.resolution_history.get_resolution_stats()

@router.get(
    "/conflicts/{merge_id}/{conflict_id}/suggestions",
    description="Get resolution suggestions for a conflict"
)
async def get_resolution_suggestions(
    merge_id: str,
    conflict_id: str,
    merge_service: MergeService = Depends(get_merge_service)
) -> List[Dict[str, Any]]:
    """Get resolution suggestions based on similar past resolutions"""
    try:
        return await merge_service.get_resolution_suggestions(
            merge_id=merge_id,
            conflict_id=conflict_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting resolution suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting suggestions: {str(e)}"
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
