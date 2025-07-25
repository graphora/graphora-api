"""Quality validation API endpoints."""

import traceback
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime, timezone
import logging

# Import quality models and services
try:
    from app.services.quality.models import QualityResults, QualityViolation, QualityRuleType, QualitySeverity
    from app.services.quality.service import QualityService
    from app.services.storage.neo4j import Neo4jStorage
    from app.services.user_db_service import UserDatabaseService
    from app.services.feedback_service import feedback_service, FeedbackType
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
    
    @router.get("/results/{transform_id}", response_model=QualityResults)
    async def get_quality_results(
        transform_id: str,
        user_id: str = Header(..., alias="user-id")
    ):
        """Get quality validation results for a transform."""
        try:
            # Get user's database configuration
            user_config = await UserDatabaseService.get_user_config(user_id)
            
            # Create Neo4j storage with user's staging database configuration
            neo4j_storage = Neo4jStorage(
                uri=user_config.stagingDb.uri,
                username=user_config.stagingDb.username,
                password=user_config.stagingDb.password,
                database="neo4j"  # Default database name
            )
            quality_service = QualityService(neo4j_storage)
            
            logger.info(f"Retrieving quality results for transform {transform_id}, user {user_id}")
            results = await quality_service.get_quality_results(transform_id, user_id)
            if not results:
                logger.warning(f"Quality results not found for transform {transform_id}, user {user_id}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Quality results not found for transform {transform_id}"
                )
            logger.info(f"Successfully retrieved quality results for transform {transform_id}")
            return results
        except HTTPException:
            # Re-raise HTTP exceptions (like 404) without wrapping them
            raise
        except Exception as e:
            logger.error(f"Failed to get quality results for {transform_id}: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Failed to retrieve quality results")
    
    @router.post("/approve/{transform_id}")
    async def approve_quality_results(
        transform_id: str,
        request: ApprovalRequest,
        user_id: str = Header(..., alias="user-id")
    ):
        """User approves quality results and proceeds to merge."""
        try:
            # Get user's database configuration
            user_config = await UserDatabaseService.get_user_config(user_id)
            
            # Create Neo4j storage with user's staging database configuration
            neo4j_storage = Neo4jStorage(
                uri=user_config.stagingDb.uri,
                username=user_config.stagingDb.username,
                password=user_config.stagingDb.password,
                database="neo4j"  # Default database name
            )
            quality_service = QualityService(neo4j_storage)
            
            await quality_service.approve_quality_results(transform_id, user_id, request.approval_comment)
            
            # Store approval feedback in Supabase  
            try:
                await feedback_service.store_quality_feedback(
                    user_id=user_id,
                    transform_id=transform_id,
                    feedback_type=FeedbackType.QUALITY_APPROVAL,
                    feedback_content=request.approval_comment or "Quality approved for merge",
                    metadata={
                        "action": "approve",
                        "source": "quality_dashboard",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                logger.info(f"Quality approval feedback stored for transform {transform_id}")
                
            except Exception as feedback_error:
                logger.warning(f"Failed to store approval feedback: {feedback_error}")
                # Don't fail the main operation if feedback storage fails
            
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
        request: RejectQualityRequest,
        user_id: str = Header(..., alias="user-id")
    ):
        """User rejects quality results - stops the process."""
        try:
            # Get user's database configuration
            user_config = await UserDatabaseService.get_user_config(user_id)
            
            # Create Neo4j storage with user's staging database configuration
            neo4j_storage = Neo4jStorage(
                uri=user_config.stagingDb.uri,
                username=user_config.stagingDb.username,
                password=user_config.stagingDb.password,
                database="neo4j"  # Default database name
            )
            quality_service = QualityService(neo4j_storage)
            
            await quality_service.reject_quality_results(transform_id, request.rejection_reason, user_id)
            
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
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                logger.info(f"Quality rejection feedback stored for transform {transform_id}")
                
            except Exception as feedback_error:
                logger.warning(f"Failed to store feedback: {feedback_error}")
                # Don't fail the main operation if feedback storage fails
            
            # TODO: Mark transform as failed in transform service
            # await transform_service.mark_transform_failed(transform_id, "User rejected quality results")
            
            return {
                "message": "Quality rejected successfully",
                "transform_id": transform_id,
                "status": "rejected",
                "reason": request.rejection_reason
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
    ):
        """Get filtered list of quality violations."""
        try:
            # Get user's database configuration
            user_config = await UserDatabaseService.get_user_config(user_id)
            
            # Create Neo4j storage with user's staging database configuration
            neo4j_storage = Neo4jStorage(
                uri=user_config.stagingDb.uri,
                username=user_config.stagingDb.username,
                password=user_config.stagingDb.password,
                database="neo4j"  # Default database name
            )
            quality_service = QualityService(neo4j_storage)
            
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