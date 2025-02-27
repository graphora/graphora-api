"""API endpoints for merge operations"""
from datetime import datetime, timezone
from typing import Annotated, Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body
import logging
import uuid
from prefect import get_client

from app.services.merge.service import MergeService, merge_flow
from app.services.merge.models import MergeInitResponse, MergeStatus, MergeProgress, MergeStage
from app.schemas.conflicts import Conflict, ConflictListResponse, ConflictResolutionRequest
from app.dependencies import get_merge_service, get_progress_tracker
from app.config import settings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/merge",
    tags=["Merge"]
)

class ConflictResolutionRequest(BaseModel):
    """Request model for conflict resolution"""
    resolution_id: str

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
