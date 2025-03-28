"""API endpoints for merge operations"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging
import uuid
import traceback
from app.schemas.graph import GraphResponse
from app.services.merge.new_merger import (
  merge_flow, get_human_review_items, get_merge_status, apply_resolution, 
  get_merge_statistics, get_merge_graph, log_merge_failure
)
from app.services.merge.models import MergeInitResponse, MergeStatus, ChangeLog
from app.config import settings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/merge",
    tags=["Merge"]
)

@router.post("/{session_id}/{transform_id}/start",
            response_model=MergeStatus,
            description="Start graph merge process")
async def start_merge(
    merge_id: Optional[str],
    session_id: str,
    transform_id: str,
    background_tasks: BackgroundTasks
) -> MergeStatus:
    """Start a new merge process"""
    try:
        # Validate inputs
        if not session_id or not transform_id:
            raise HTTPException(
                status_code=400,
                detail="session_id and transform_id are required"
            )
            
        # Generate merge ID
        if not merge_id:
            merge_id = str(uuid.uuid4())
        
        # Define background task
        async def run_merge_flow():
            try:
                # Create flow run directly
                flow_run = await merge_flow(
                    merge_id=merge_id,
                    transform_id=transform_id,
                    ontology_id=session_id
                )
                logger.info(f"Started flow run {flow_run} for merge {merge_id}")
            except Exception as e:
                logger.error(f"Failed to start merge flow: {str(e)}")
                log_merge_failure(merge_id, str(e))
        
        # Add background task
        background_tasks.add_task(run_merge_flow)
        
        return MergeStatus.STARTED
        
    except Exception as e:
        logger.error(f"Failed to start merge: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start merge: {str(e)}"
        )

@router.get("/{merge_id}/status", response_model=MergeStatus)
def get_merge_status_api(merge_id: str) -> MergeStatus:
    """Get current status of a merge process"""
    try:
        status = get_merge_status(merge_id)
            
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
    response_model=List[ChangeLog],
    description="Get conflicts for a merge process"
)
async def get_conflicts(
    merge_id: str
) -> List[ChangeLog]:
    """Get conflicts for a merge process"""
    try:
        return get_human_review_items(merge_id)
    except Exception as e:
        logger.error(f"Failed to get conflicts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conflicts: {str(e)}"
        )

@router.post(
    "/{merge_id}/conflicts/{conflict_id}/resolve",
    response_model=bool,
    description="Apply a resolution to a specific conflict"
)
async def resolve_conflict(
    merge_id: str,
    conflict_id: str,
    changed_props: Dict[str, Any],
    learning_comment: str
) -> bool:
    """
    Apply a resolution to a specific conflict
    
    Parameters:
    - merge_id: ID of the merge process
    - conflict_id: ID of the conflict to resolve
    - changed_props: Properties to update
    - learning_comment: Comment on the resolution
    
    Returns:
    - True if the resolution was applied successfully, False otherwise
    """
    try:
        return await apply_resolution(merge_id, conflict_id, changed_props, learning_comment)
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error resolving conflict: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving conflict: {str(e)}"
        )

@router.get(
    "/statistics/{merge_id}",
    response_model=Dict[str, Any],
    description="Get detailed statistics of a merge operation"
)
async def get_merge_statistics(
    merge_id: str
) -> Dict[str, Any]:
    """Get detailed statistics of a merge operation"""
    try:
        statistics = get_merge_statistics(merge_id)
        if not statistics:
            raise HTTPException(status_code=404, detail=f"Merge {merge_id} not found")
        return statistics
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error getting merge statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting merge statistics: {str(e)}"
        )

@router.get("/graph/{merge_id}/{transform_id}",
         response_model=GraphResponse,
         description="Retrieve nodes by transform ID and their relationships")
async def get_graph_by_merge_id(
    merge_id: str,
    transform_id: str
) -> GraphResponse:
    try:
        return await get_merge_graph(merge_id, transform_id)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error retrieving graph data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving graph data: {str(e)}"
        )