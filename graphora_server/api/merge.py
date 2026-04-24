"""API endpoints for merge operations"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from graphora_server.baml_client.types import ResolutionStrategy
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import logging
import uuid
import traceback
from graphora_server.schemas.graph import GraphResponse
from graphora_server.services.merge.new_merger import (
    merge_flow,
    get_human_review_items,
    get_merge_status,
    apply_resolution,
    get_merge_statistics,
    get_merge_graph,
    log_merge_failure_task,
)
from graphora_server.services.merge.models import MergeInitResponse, MergeStatus, ChangeLog
from graphora_server.config import settings
from graphora_server.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{settings.API_V1_STR}/merge", tags=["Merge"])


@router.post(
    "/{session_id}/{transform_id}/start",
    response_model=MergeInitResponse,
    description="Start graph merge process",
)
async def start_merge(
    merge_id: Optional[str],
    session_id: str,
    transform_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> MergeInitResponse:
    """Start a new merge process for user's production database"""
    try:
        # Validate inputs
        if not session_id or not transform_id:
            raise HTTPException(
                status_code=400, detail="session_id and transform_id are required"
            )

        logger.info(
            f"Starting merge for user {user_id}, session: {session_id}, transform: {transform_id}"
        )

        # Generate merge ID
        if not merge_id or merge_id == "new":
            merge_id = str(uuid.uuid4())

        # Define background task with user context (merges use production database)
        async def run_merge_flow():
            try:
                # Create flow run directly with user ID
                await merge_flow(
                    merge_id=merge_id,
                    transform_id=transform_id,
                    ontology_id=session_id,
                    user_id=user_id,  # Pass user ID to merge flow
                )
                logger.info(
                    f"Started merge flow for user {user_id} with merge_id: {merge_id}"
                )
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Failed to start merge flow for user {user_id}: {str(e)}")
                log_merge_failure_task(merge_id, str(e))

        # Add background task
        background_tasks.add_task(run_merge_flow)

        return MergeInitResponse(
            merge_id=merge_id, status=MergeStatus.STARTED, start_time=datetime.now()
        )

    except Exception as e:
        logger.error(f"Failed to start merge for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start merge: {str(e)}")


@router.get("/{merge_id}/status", response_model=MergeStatus)
def get_merge_status_api(
    merge_id: str, user_id: str = Depends(get_current_user_id)
) -> MergeStatus:
    """Get current status of a merge process for user"""
    try:
        logger.info(f"Getting merge status for user {user_id}, merge_id: {merge_id}")

        status = get_merge_status(merge_id)

        if not status:
            raise HTTPException(
                status_code=404, detail=f"Merge {merge_id} not found for user {user_id}"
            )

        return status

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to get merge status for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get merge status: {str(e)}"
        )


@router.get(
    "/{merge_id}/conflicts",
    response_model=List[ChangeLog],
    description="Get conflicts for a merge process",
)
async def get_conflicts(
    merge_id: str, user_id: str = Depends(get_current_user_id)
) -> List[ChangeLog]:
    """Get conflicts for a merge process for user"""
    try:
        logger.info(f"Getting conflicts for user {user_id}, merge_id: {merge_id}")
        return get_human_review_items(merge_id)
    except Exception as e:
        logger.error(f"Failed to get conflicts for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get conflicts: {str(e)}"
        )


@router.post(
    "/{merge_id}/conflicts/{conflict_id}/resolve",
    response_model=bool,
    description="Apply a resolution to a specific conflict",
)
async def resolve_conflict(
    merge_id: str,
    conflict_id: str,
    changed_props: Dict[str, Any],
    resolution: ResolutionStrategy,
    learning_comment: str,
    user_id: str = Depends(get_current_user_id),
) -> bool:
    """
    Apply a resolution to a specific conflict for user

    Parameters:
    - merge_id: ID of the merge process
    - conflict_id: ID of the conflict to resolve
    - changed_props: Properties that were changed
    - resolution: The resolution decision
    - learning_comment: Comment on the resolution
    - user_id: User's ID (from header)

    Returns:
    - True if the resolution was applied successfully, False otherwise
    """
    try:
        logger.info(
            f"Resolving conflict for user {user_id}, merge_id: {merge_id}, conflict_id: {conflict_id}"
        )

        return await apply_resolution(
            merge_id, conflict_id, changed_props, resolution, learning_comment, user_id
        )

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error resolving conflict for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error resolving conflict: {str(e)}"
        )


@router.get(
    "/statistics/{merge_id}",
    response_model=Dict[str, Any],
    description="Get detailed statistics of a merge operation",
)
async def get_merge_statistics_api(
    merge_id: str, user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get detailed statistics of a merge operation for user"""
    try:
        logger.info(
            f"Getting merge statistics for user {user_id}, merge_id: {merge_id}"
        )

        statistics = await get_merge_statistics(merge_id)
        if not statistics:
            # Return empty statistics if not available yet (merge in progress)
            return {
                "message": "Statistics not available yet - merge in progress",
                "nodes_stored": None,
                "edges_stored": None,
            }
        return statistics
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error getting merge statistics for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting merge statistics: {str(e)}"
        )


@router.get(
    "/graph/{merge_id}/{transform_id}",
    response_model=GraphResponse,
    description="Retrieve nodes by transform ID and their relationships",
)
async def get_graph_by_merge_id(
    merge_id: str, transform_id: str, user_id: str = Depends(get_current_user_id)
) -> GraphResponse:
    """Get graph data for merge operation for user (from production database)"""
    try:
        logger.info(
            f"Getting merge graph for user {user_id}, merge_id: {merge_id}, transform_id: {transform_id}"
        )

        graph = await get_merge_graph(merge_id, transform_id, user_id)
        if graph:
            return graph
        else:
            return GraphResponse(nodes=[], edges=[])
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error retrieving graph data for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving graph data: {str(e)}"
        )
