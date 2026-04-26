"""Audit Trail API endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from graphora_server.config import settings
from graphora_server.db import postgres as db
from graphora_server.services.audit_service import audit_service, OperationType
from graphora_server.auth import get_current_user_id

router = APIRouter(prefix=settings.API_V1_STR, tags=["Audit Trail"])


@router.get("/audit/summary")
async def get_audit_summary(user_id: str = Depends(get_current_user_id)):
    """
    Get audit trail summary for dashboard

    Parameters:
    - user_id: User's ID (from header)

    Returns:
    - Summary of audit operations for the user
    """
    try:
        # Get audit summary from service
        summary = await audit_service.get_audit_summary(user_id)
        return summary

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching audit summary: {str(e)}"
        )


@router.get("/audit/trail")
async def get_audit_trail(
    user_id: str = Depends(get_current_user_id),
    operation_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Get detailed audit trail for a user

    Parameters:
    - user_id: User's ID (from header)
    - operation_type: Filter by operation type (optional)
    - limit: Maximum number of records to return (default: 50)
    - offset: Number of records to skip (default: 0)

    Returns:
    - List of audit trail records
    """
    try:
        operation_type_enum = None
        if operation_type:
            try:
                operation_type_enum = OperationType(operation_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid operation type: {operation_type}. Valid types: {[op.value for op in OperationType]}",
                )

        trail = await audit_service.get_user_audit_trail(
            user_id=user_id,
            operation_type=operation_type_enum,
            limit=limit,
            offset=offset,
        )

        return {"records": trail, "total": len(trail), "offset": offset, "limit": limit}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching audit trail: {str(e)}"
        )


@router.get("/audit/conflicts")
async def get_conflicts_summary(user_id: str = Depends(get_current_user_id)):
    """
    Get conflicts summary for dashboard

    Parameters:
    - user_id: User's ID (from header)

    Returns:
    - Summary of merge conflicts for the user
    """
    try:
        user_merges = await db.fetch(
            """
            SELECT operation_id
            FROM audit_trail
            WHERE user_id = %s AND operation_type = %s
            """,
            user_id,
            OperationType.MERGE_STARTED.value,
        )

        user_merge_ids = [merge["operation_id"] for merge in user_merges or []]

        if not user_merge_ids:
            # User has no merges, return empty summary
            return {
                "total_conflicts": 0,
                "conflicts_by_merge": [],
                "recent_conflicts": [],
            }

        conflicts = await db.fetch(
            """
            SELECT merge_id, node_type, need_human_review, previous_props, changed_props
            FROM change_logs
            WHERE need_human_review = TRUE AND merge_id = ANY(%s)
            """,
            user_merge_ids,
        )

        # Group by merge_id and node_type
        conflicts_by_merge = {}
        for conflict in conflicts:
            merge_id = conflict["merge_id"]
            if merge_id not in conflicts_by_merge:
                conflicts_by_merge[merge_id] = {
                    "merge_id": merge_id,
                    "total_conflicts": 0,
                    "by_type": {},
                }

            node_type = conflict["node_type"]
            if node_type not in conflicts_by_merge[merge_id]["by_type"]:
                conflicts_by_merge[merge_id]["by_type"][node_type] = 0

            conflicts_by_merge[merge_id]["by_type"][node_type] += 1
            conflicts_by_merge[merge_id]["total_conflicts"] += 1

        return {
            "total_conflicts": len(conflicts),
            "conflicts_by_merge": list(conflicts_by_merge.values()),
            "recent_conflicts": conflicts[:10],  # Latest 10 conflicts
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching conflicts summary: {str(e)}"
        )
