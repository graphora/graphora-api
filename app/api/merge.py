"""API endpoints for merge operations"""
from datetime import datetime, timezone
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import logging

from app.services.merge.service import MergeService
from app.services.merge.models import MergeInitResponse, MergeStatus, MergeProgress
from app.dependencies import get_storage

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/merge",
    tags=["merge"]
)

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
