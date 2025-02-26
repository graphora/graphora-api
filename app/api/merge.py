"""API endpoints for merge operations"""
from datetime import datetime, timezone
from typing import Annotated, Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body
import logging

from app.services.merge.service import MergeService
from app.services.merge.models import MergeInitResponse, MergeStatus, MergeProgress
from app.schemas.conflicts import Conflict, ConflictListResponse
from app.dependencies import get_storage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/merge",
    tags=["merge"]
)

class ConflictResolutionRequest(BaseModel):
    """Request model for conflict resolution"""
    resolution_id: str

async def get_merge_service(
    storage=Depends(get_storage)
) -> MergeService:
    """Get MergeService instance"""
    return MergeService(storage=storage)

@router.post("/{session_id}/{transform_id}/start",
            response_model=MergeInitResponse,
            description="Start graph merge process")
async def start_merge(
    session_id: str,
    transform_id: str,
    background_tasks: BackgroundTasks,
    merge_service: Annotated[MergeService, Depends(get_merge_service)]
) -> MergeInitResponse:
    """Start the merge process from staging to production"""
    try:
        # Validate parameters
        if not session_id or not transform_id:
            raise HTTPException(
                status_code=400,
                detail="session_id and transform_id are required"
            )
            
        # Start merge flow
        merge_id = await merge_service.start_merge_flow(
            session_id=session_id,
            transform_id=transform_id
        )
        
        return MergeInitResponse(
            merge_id=merge_id,
            status=MergeStatus.PENDING,
            start_time=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Failed to start merge process: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start merge process: {str(e)}"
        )

@router.get("/status/{merge_id}", response_model=MergeProgress)
async def get_merge_status(
    merge_id: str,
    merge_service: MergeService = Depends(get_merge_service)
) -> MergeProgress:
    """Get current status and progress of merge process"""
    status = await merge_service.get_merge_progress(merge_id)
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Merge {merge_id} not found"
        )
    return status

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
    offset: int = 0,
    merge_service: MergeService = Depends(get_merge_service)
) -> ConflictListResponse:
    """Get conflicts for a merge process with filtering and pagination"""
    try:
        # Get conflicts
        conflicts, total_count = await merge_service.get_conflicts(
            merge_id=merge_id,
            conflict_type=conflict_type,
            severity=severity,
            resolved=resolved,
            limit=limit,
            offset=offset
        )
        
        # Get summary
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
        logger.error(f"Error retrieving conflicts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conflicts: {str(e)}"
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
    response_model=dict,
    description="Resolve a specific conflict"
)
async def resolve_conflict(
    merge_id: str,
    conflict_id: str,
    resolution_request: ConflictResolutionRequest,
    merge_service: MergeService = Depends(get_merge_service)
) -> dict:
    """Resolve a specific conflict using the provided resolution option"""
    try:
        # Resolve conflict
        success = await merge_service.resolve_conflict(
            merge_id=merge_id,
            conflict_id=conflict_id,
            resolution_id=resolution_request.resolution_id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to resolve conflict {conflict_id}"
            )
            
        return {
            "success": True,
            "merge_id": merge_id,
            "conflict_id": conflict_id,
            "resolution_id": resolution_request.resolution_id,
            "message": "Conflict resolved successfully"
        }
        
    except HTTPException:
        raise
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
