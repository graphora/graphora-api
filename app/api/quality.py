"""Quality validation API endpoints."""

import traceback
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timezone
import logging

# Import quality models and services
try:
    from app.services.quality.models import (
        QualityResults,
        QualityViolation,
        QualityRuleType,
        QualitySeverity,
    )
    from app.services.quality.service import QualityService
    from app.services.user_db_service import UserDatabaseService
    from app.services.feedback_service import feedback_service, FeedbackType
    from app.auth import get_current_user_id

    QUALITY_API_AVAILABLE = True
except ImportError:
    QUALITY_API_AVAILABLE = False


# Request models
class RejectQualityRequest(BaseModel):
    rejection_reason: str


class ApprovalRequest(BaseModel):
    approval_comment: Optional[str] = None


logger = logging.getLogger(__name__)

# Create router with conditional registration
if QUALITY_API_AVAILABLE:
    router = APIRouter(prefix="/api/v1/quality", tags=["quality"])

    class QualitySummaryItem(BaseModel):
        transform_id: str
        overall_score: Optional[float] = None
        grade: Optional[str] = None
        requires_review: Optional[bool] = None
        total_violations: Optional[int] = None
        created_at: Optional[datetime] = None
        status: str


    async def _get_quality_service(user_id: str) -> QualityService:
        user_config = await UserDatabaseService.get_user_config(user_id)

        try:
            from app.services.storage.neo4j import Neo4jStorage
        except Exception as storage_error:  # pragma: no cover - import failure handled at runtime
            logger.error(
                "Neo4j storage backend unavailable for quality operations: %s",
                storage_error,
            )
            raise HTTPException(
                status_code=503,
                detail="Graph storage backend unavailable",
            ) from storage_error

        neo4j_storage = Neo4jStorage(
            uri=user_config.stagingDb.uri,
            username=user_config.stagingDb.username,
            password=user_config.stagingDb.password,
            database="neo4j",
        )

        return QualityService(neo4j_storage)

    @router.get("/results/{transform_id}", response_model=QualityResults)
    async def get_quality_results(
        transform_id: str, user_id: str = Depends(get_current_user_id)
    ):
        """Get quality validation results for a transform."""
        try:
            quality_service = await _get_quality_service(user_id)

            logger.info(
                f"Retrieving quality results for transform {transform_id}, user {user_id}"
            )
            results = await quality_service.get_quality_results(transform_id, user_id)
            if not results:
                logger.warning(
                    f"Quality results not found for transform {transform_id}, user {user_id}"
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Quality results not found for transform {transform_id}",
                )
            logger.info(
                f"Successfully retrieved quality results for transform {transform_id}"
            )
            return results
        except HTTPException:
            # Re-raise HTTP exceptions (like 404) without wrapping them
            raise
        except Exception as e:
            logger.error(f"Failed to get quality results for {transform_id}: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail="Failed to retrieve quality results"
            )

    @router.post("/approve/{transform_id}")
    async def approve_quality_results(
        transform_id: str,
        request: ApprovalRequest,
        user_id: str = Depends(get_current_user_id),
    ):
        """User approves quality results and proceeds to merge."""
        try:
            quality_service = await _get_quality_service(user_id)

            await quality_service.approve_quality_results(
                transform_id, user_id, request.approval_comment
            )

            # Store approval feedback in Supabase
            try:
                await feedback_service.store_quality_feedback(
                    user_id=user_id,
                    transform_id=transform_id,
                    feedback_type=FeedbackType.QUALITY_APPROVAL,
                    feedback_content=request.approval_comment
                    or "Quality approved for merge",
                    metadata={
                        "action": "approve",
                        "source": "quality_dashboard",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                logger.info(
                    f"Quality approval feedback stored for transform {transform_id}"
                )

            except Exception as feedback_error:
                logger.warning(f"Failed to store approval feedback: {feedback_error}")
                # Don't fail the main operation if feedback storage fails

            # TODO: Trigger merge process here
            # merge_id = await merge_service.start_auto_merge(transform_id, user_id)

            return {
                "message": "Quality approved successfully",
                "transform_id": transform_id,
                "status": "approved",
                # "merge_id": merge_id  # When merge integration is ready
            }
        except Exception as e:
            logger.error(f"Failed to approve quality results for {transform_id}: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to approve quality results"
            )

    @router.post("/reject/{transform_id}")
    async def reject_quality_results(
        transform_id: str,
        request: RejectQualityRequest,
        user_id: str = Depends(get_current_user_id),
    ):
        """User rejects quality results - stops the process."""
        try:
            quality_service = await _get_quality_service(user_id)

            await quality_service.reject_quality_results(
                transform_id, request.rejection_reason, user_id
            )

            # Store feedback in Supabase
            try:
                await feedback_service.store_quality_feedback(
                    user_id=user_id,
                    transform_id=transform_id,
                    feedback_type=FeedbackType.QUALITY_REJECTION,
                    feedback_content=request.rejection_reason,
                    metadata={
                        "action": "reject",
                        "source": "quality_dashboard",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                logger.info(
                    f"Quality rejection feedback stored for transform {transform_id}"
                )

            except Exception as feedback_error:
                logger.warning(f"Failed to store feedback: {feedback_error}")
                # Don't fail the main operation if feedback storage fails

            # TODO: Mark transform as failed in transform service
            # await transform_service.mark_transform_failed(transform_id, "User rejected quality results")

            return {
                "message": "Quality rejected successfully",
                "transform_id": transform_id,
                "status": "rejected",
                "reason": request.rejection_reason,
            }
        except Exception as e:
            logger.error(f"Failed to reject quality results for {transform_id}: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to reject quality results"
            )

    @router.get(
        "/violations/{transform_id}", response_model=List[QualityViolation]
    )
    async def list_quality_violations(
        transform_id: str,
        violation_type: Optional[QualityRuleType] = Query(
            default=None, description="Filter by quality rule type"
        ),
        severity: Optional[QualitySeverity] = Query(
            default=None, description="Filter by violation severity"
        ),
        entity_type: Optional[str] = Query(
            default=None, description="Filter by entity type"
        ),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        user_id: str = Depends(get_current_user_id),
    ):
        """List persisted quality violations for a transform with optional filtering."""

        try:
            quality_service = await _get_quality_service(user_id)
            quality_results = await quality_service.get_quality_results(
                transform_id, user_id
            )

            if not quality_results:
                raise HTTPException(
                    status_code=404,
                    detail=f"Quality results not found for transform {transform_id}",
                )

            violations = await quality_service.get_violations(
                transform_id=transform_id,
                user_id=user_id,
                violation_type=violation_type,
                severity=severity,
                entity_type=entity_type,
                limit=limit,
                offset=offset,
                quality_results=quality_results,
            )

            logger.info(
                "Retrieved %s quality violations for transform %s (offset=%s, limit=%s)",
                len(violations),
                transform_id,
                offset,
                limit,
            )

            return violations
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to list quality violations for transform %s: %s",
                transform_id,
                exc,
            )
            raise HTTPException(
                status_code=500, detail="Failed to retrieve quality violations"
            ) from exc

    @router.get("/summary", response_model=List[QualitySummaryItem])
    async def get_quality_summary_endpoint(
        limit: int = Query(10, ge=1, le=100),
        user_id: str = Depends(get_current_user_id),
    ):
        """Return recent quality summaries for the authenticated user."""

        try:
            quality_service = await _get_quality_service(user_id)
            summary_rows = await quality_service.get_quality_summary(user_id, limit)

            summaries: List[QualitySummaryItem] = []
            for row in summary_rows:
                created_at_raw = row.get("created_at")
                created_at_value: Optional[datetime] = None
                if isinstance(created_at_raw, datetime):
                    created_at_value = created_at_raw
                elif isinstance(created_at_raw, str):
                    try:
                        created_at_value = datetime.fromisoformat(created_at_raw)
                    except ValueError:
                        created_at_value = None

                summaries.append(
                    QualitySummaryItem(
                        transform_id=row.get("transform_id"),
                        overall_score=row.get("overall_score"),
                        grade=row.get("grade"),
                        requires_review=row.get("requires_review"),
                        total_violations=row.get("total_violations"),
                        created_at=created_at_value,
                        status=row.get("status", "pending"),
                    )
                )

            logger.info(
                "Retrieved %s quality summaries for user %s",
                len(summaries),
                user_id,
            )

            return summaries
        except Exception as exc:
            logger.error(
                "Failed to retrieve quality summary for user %s: %s", user_id, exc
            )
            raise HTTPException(
                status_code=500, detail="Failed to retrieve quality summary"
            ) from exc

    @router.delete("/results/{transform_id}")
    async def delete_quality_results(
        transform_id: str,
        user_id: str = Depends(get_current_user_id),
    ):
        """Delete quality results for a transform."""
        try:
            quality_service = await _get_quality_service(user_id)

            deleted = await quality_service.delete_quality_results(
                transform_id, user_id
            )

            if deleted:
                return {
                    "message": "Quality results deleted successfully",
                    "transform_id": transform_id,
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Quality results not found for transform {transform_id}",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete quality results for {transform_id}: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to delete quality results"
            )

    @router.get("/health")
    async def quality_api_health():
        """Health check for quality API."""
        return {
            "status": "healthy",
            "quality_api_available": True,
            "message": "Quality validation API is operational",
        }

else:
    # Create a minimal router when quality API is not available
    router = APIRouter(prefix="/api/v1/quality", tags=["quality"])

    @router.get("/health")
    async def quality_api_health():
        """Health check - indicates quality API is not available."""
        return {
            "status": "unavailable",
            "quality_api_available": False,
            "message": "Quality validation modules not installed",
        }

    @router.get("/{path:path}")
    async def quality_not_available():
        """Catch-all for quality endpoints when not available."""
        raise HTTPException(
            status_code=503,
            detail="Quality validation feature is not available. Please check module installation.",
        )
