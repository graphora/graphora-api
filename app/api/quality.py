"""Quality validation API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from typing import Optional, List
import logging

# Import quality models and services
try:
    from app.services.quality.models import QualityResults, QualityViolation, QualityRuleType, QualitySeverity
    from app.services.quality.service import QualityService
    from app.services.storage.neo4j import Neo4jService
    QUALITY_API_AVAILABLE = True
except ImportError:
    QUALITY_API_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router with conditional registration
if QUALITY_API_AVAILABLE:
    router = APIRouter(prefix="/api/v1/quality", tags=["quality"])
    
    # Dependency to get quality service
    def get_quality_service() -> QualityService:
        neo4j_service = Neo4jService()
        return QualityService(neo4j_service)
    
    @router.get("/results/{transform_id}", response_model=QualityResults)
    async def get_quality_results(
        transform_id: str,
        user_id: str = Header(..., alias="user-id"),
        quality_service: QualityService = Depends(get_quality_service)
    ):
        """Get quality validation results for a transform."""
        try:
            results = await quality_service.get_quality_results(transform_id, user_id)
            if not results:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Quality results not found for transform {transform_id}"
                )
            return results
        except Exception as e:
            logger.error(f"Failed to get quality results for {transform_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve quality results")
    
    @router.post("/approve/{transform_id}")
    async def approve_quality_results(
        transform_id: str,
        approval_comment: Optional[str] = None,
        user_id: str = Header(..., alias="user-id"),
        quality_service: QualityService = Depends(get_quality_service)
    ):
        """User approves quality results and proceeds to merge."""
        try:
            await quality_service.approve_quality_results(transform_id, user_id, approval_comment)
            
            # TODO: Trigger merge process here
            # merge_id = await merge_service.start_auto_merge(transform_id, user_id)
            
            return {
                "message": "Quality approved successfully",
                "transform_id": transform_id,
                "status": "approved"
                # "merge_id": merge_id  # When merge integration is ready
            }
        except Exception as e:
            logger.error(f"Failed to approve quality results for {transform_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to approve quality results")
    
    @router.post("/reject/{transform_id}")
    async def reject_quality_results(
        transform_id: str,
        rejection_reason: str,
        user_id: str = Header(..., alias="user-id"),
        quality_service: QualityService = Depends(get_quality_service)
    ):
        """User rejects quality results - stops the process."""
        try:
            await quality_service.reject_quality_results(transform_id, rejection_reason, user_id)
            
            # TODO: Mark transform as failed in transform service
            # await transform_service.mark_transform_failed(transform_id, "User rejected quality results")
            
            return {
                "message": "Quality rejected successfully",
                "transform_id": transform_id,
                "status": "rejected",
                "reason": rejection_reason
            }
        except Exception as e:
            logger.error(f"Failed to reject quality results for {transform_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to reject quality results")
    
    @router.get("/violations/{transform_id}")
    async def get_detailed_violations(
        transform_id: str,
        user_id: str = Header(..., alias="user-id"),
        violation_type: Optional[QualityRuleType] = Query(None, description="Filter by violation type"),
        severity: Optional[QualitySeverity] = Query(None, description="Filter by severity"),
        entity_type: Optional[str] = Query(None, description="Filter by entity type"),
        limit: int = Query(100, ge=1, le=1000, description="Number of violations to return"),
        offset: int = Query(0, ge=0, description="Number of violations to skip"),
        quality_service: QualityService = Depends(get_quality_service)
    ):
        """Get filtered list of quality violations."""
        try:
            violations = await quality_service.get_violations(
                transform_id=transform_id,
                user_id=user_id,
                violation_type=violation_type,
                severity=severity,
                entity_type=entity_type,
                limit=limit,
                offset=offset
            )
            
            return {
                "transform_id": transform_id,
                "violations": violations,
                "total_returned": len(violations),
                "filters_applied": {
                    "violation_type": violation_type,
                    "severity": severity,
                    "entity_type": entity_type
                }
            }
        except Exception as e:
            logger.error(f"Failed to get violations for {transform_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve violations")
    
    @router.get("/summary")
    async def get_quality_summary(
        user_id: str = Header(..., alias="user-id"),
        limit: int = Query(10, ge=1, le=50, description="Number of recent results to return"),
        quality_service: QualityService = Depends(get_quality_service)
    ):
        """Get summary of recent quality results for a user."""
        try:
            summaries = await quality_service.get_quality_summary(user_id, limit)
            
            return {
                "user_id": user_id,
                "recent_quality_results": summaries,
                "total_returned": len(summaries)
            }
        except Exception as e:
            logger.error(f"Failed to get quality summary for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve quality summary")
    
    @router.delete("/results/{transform_id}")
    async def delete_quality_results(
        transform_id: str,
        user_id: str = Header(..., alias="user-id"),
        quality_service: QualityService = Depends(get_quality_service)
    ):
        """Delete quality results for a transform."""
        try:
            deleted = await quality_service.delete_quality_results(transform_id, user_id)
            
            if deleted:
                return {
                    "message": "Quality results deleted successfully",
                    "transform_id": transform_id
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Quality results not found for transform {transform_id}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete quality results for {transform_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete quality results")
    
    @router.get("/health")
    async def quality_api_health():
        """Health check for quality API."""
        return {
            "status": "healthy",
            "quality_api_available": True,
            "message": "Quality validation API is operational"
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
            "message": "Quality validation modules not installed"
        }
    
    @router.get("/{path:path}")
    async def quality_not_available():
        """Catch-all for quality endpoints when not available."""
        raise HTTPException(
            status_code=503,
            detail="Quality validation feature is not available. Please check module installation."
        )