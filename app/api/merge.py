"""API endpoints for merge operations"""
from datetime import datetime, timezone
from typing import Annotated, Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body, Query
import logging
import uuid
from prefect import get_client
from fastapi import status
import traceback

from app.services.merge.service import MergeService, merge_flow
from app.services.merge.models import MergeInitResponse, MergeStatus, MergeProgress, MergeStage, RollbackOptions, RollbackResponse, VerificationResult
from app.services.merge.batch_resolver import BatchResolver
from app.services.merge.resolution_search import ResolutionPatternSearchService
from app.services.storage.vector_storage import QdrantResolutionStorage
from app.schemas.conflicts import (
    Conflict, ConflictListResponse, ConflictResolutionRequest,
    PendingConflictsResponse, ConflictResolutionResponse,
    BulkResolutionRequest, BulkResolutionResponse,
    ResolutionRequest, ResolutionResult, BatchResolutionRequest, BatchResolutionResult,
    ConflictType, GroupBatchResolutionRequest
)
from app.dependencies import get_progress_tracker, get_merge_service
from app.config import settings
from pydantic import BaseModel, Field
from app.schemas.resolution_history import ResolutionHistoryEntry
from app.schemas.merge import (
    MergeProgressResponse,
    MergeStatisticsResponse,
    MergeSummaryResponse,
    VerificationResultResponse
)

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
                # Create flow run directly
                flow_run = await merge_flow(
                    merge_id=merge_id,
                    session_id=session_id,
                    transform_id=transform_id,
                    ontology_id=session_id
                )
                logger.info(f"Started flow run {flow_run} for merge {merge_id}")
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
        traceback.print_exc()
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
        async with get_merge_service() as merge_service:
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
    "/{merge_id}/pending-conflicts",
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
        async with get_merge_service() as merge_service:
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
        async with get_merge_service() as merge_service:
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
    "/{merge_id}/conflicts/bulk-resolve",
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
        async with get_merge_service() as merge_service:
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
    "/{merge_id}/conflicts/{conflict_id}/resolve",
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
        async with get_merge_service() as merge_service:
            result = await merge_service.apply_conflict_resolution(
                merge_id=merge_id,
                conflict_id=conflict_id,
                resolution_id=request.resolution_id,
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
    async with get_merge_service() as merge_service:
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
        async with get_merge_service() as merge_service:
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
    async with get_merge_service() as merge_service:    
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

@router.get(
    "/conflicts/{merge_id}/groups",
    response_model=Dict[str, List[Dict[str, Any]]],
    description="Get grouped conflicts for batch resolution"
)
async def get_conflict_groups(
    merge_id: str,
    grouping_strategy: str = "type_and_entity",
    similarity_threshold: float = 0.8,
    merge_service: MergeService = Depends(get_merge_service)
) -> Dict[str, List[Dict[str, Any]]]:
    """Get grouped conflicts for batch resolution"""
    try:
        # Create batch resolver
        resolver = BatchResolver(merge_service)
        
        # Get grouped conflicts
        groups = await resolver.group_similar_conflicts(
            merge_id=merge_id,
            grouping_strategy=grouping_strategy,
            similarity_threshold=similarity_threshold
        )
        
        # Convert conflicts to dictionaries for response
        result = {}
        for group_key, conflicts in groups.items():
            result[group_key] = [conflict.model_dump() for conflict in conflicts]
            
        return result
        
    except Exception as e:
        logger.error(f"Error getting conflict groups: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting conflict groups: {str(e)}"
        )

@router.post(
    "/conflicts/{merge_id}/resolve-batch",
    response_model=Dict[str, Any],
    description="Resolve a batch of conflicts with the same strategy"
)
async def resolve_batch_conflicts(
    merge_id: str,
    request: GroupBatchResolutionRequest,
    merge_service: MergeService = Depends(get_merge_service)
) -> Dict[str, Any]:
    """Resolve a batch of conflicts with the same strategy"""
    try:
        # Create batch resolver
        resolver = BatchResolver(merge_service)
        
        # Apply batch resolution
        result = await resolver.apply_batch_resolution(
            merge_id=merge_id,
            group_key=request.group_key,
            resolution_option=request.resolution_option,
            exceptions=request.exceptions
        )
            
        return result
        
    except Exception as e:
        logger.error(f"Error resolving batch conflicts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving batch conflicts: {str(e)}"
        )

class SimilarResolutionRequest(BaseModel):
    """Request model for finding similar resolutions"""
    conflict_id: str = Field(..., description="ID of the conflict to find similar resolutions for")
    limit: int = Field(5, description="Maximum number of similar resolutions to return")
    min_similarity: float = Field(0.7, description="Minimum similarity score threshold")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters to apply")


class SimilarResolutionResponse(BaseModel):
    """Response model for similar resolutions"""
    conflict_id: str = Field(..., description="ID of the original conflict")
    similar_resolutions: List[Dict[str, Any]] = Field(..., description="List of similar resolutions")
    total_found: int = Field(..., description="Total number of similar resolutions found")


class BatchSimilarResolutionRequest(BaseModel):
    """Request model for batch similar resolution search"""
    conflict_ids: List[str] = Field(..., description="IDs of conflicts to find similar resolutions for")
    limit_per_conflict: int = Field(5, description="Maximum number of similar resolutions per conflict")
    min_similarity: float = Field(0.7, description="Minimum similarity score threshold")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters to apply")


@router.post(
    "/resolutions/similar",
    response_model=SimilarResolutionResponse,
    description="Find similar past resolutions for a conflict"
)
async def find_similar_resolutions(
    request: SimilarResolutionRequest,
    merge_service: MergeService = Depends(get_merge_service)
) -> SimilarResolutionResponse:
    """Find similar past resolutions for a conflict"""
    try:
        # Get the conflict
        async with get_merge_service() as merge_service:
            conflict = await merge_service.get_conflict(request.conflict_id)
            if not conflict:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conflict with ID {request.conflict_id} not found"
                )
            
            # Initialize the resolution pattern search service
            vector_storage = QdrantResolutionStorage()
            search_service = ResolutionPatternSearchService(
                vector_storage=vector_storage,
                similarity_threshold=request.min_similarity
            )
            
            # Find similar resolutions
            similar_results = await search_service.find_similar_resolutions(
                conflict=conflict,
                limit=request.limit,
                filters=request.filters
            )
            
            # Format the response
            similar_resolutions = []
            for pattern, score in similar_results:
                similar_resolutions.append({
                    "id": pattern.id,
                    "similarity_score": score,
                    "conflict_type": pattern.conflict_type,
                    "entity_types": pattern.entity_types,
                    "property_names": pattern.property_names,
                    "relationship_types": pattern.relationship_types,
                    "resolution_strategy": pattern.resolution_strategy,
                    "resolution_data": pattern.resolution_data,
                    "confidence": pattern.confidence,
                    "original_conflict_id": pattern.original_conflict_id,
                    "original_merge_id": pattern.original_merge_id,
                    "created_at": pattern.created_at.isoformat() if pattern.created_at else None
                })
            
            return SimilarResolutionResponse(
                conflict_id=request.conflict_id,
                similar_resolutions=similar_resolutions,
                total_found=len(similar_resolutions)
            )
    
    except Exception as e:
        logger.error(f"Error finding similar resolutions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar resolutions: {str(e)}"
        )


@router.post(
    "/resolutions/batch-similar",
    response_model=Dict[str, SimilarResolutionResponse],
    description="Find similar past resolutions for multiple conflicts"
)
async def batch_find_similar_resolutions(
    request: BatchSimilarResolutionRequest,
    merge_service: MergeService = Depends(get_merge_service)
) -> Dict[str, SimilarResolutionResponse]:
    """Find similar past resolutions for multiple conflicts"""
    try:
        # Get all conflicts
        async with get_merge_service() as merge_service:
            conflicts = []
            for conflict_id in request.conflict_ids:
                conflict = await merge_service.get_conflict(conflict_id)
                if conflict:
                    conflicts.append(conflict)
            
            if not conflicts:
                raise HTTPException(
                    status_code=404,
                    detail="No valid conflicts found"
                )
            
            # Initialize the resolution pattern search service
            vector_storage = QdrantResolutionStorage()
            search_service = ResolutionPatternSearchService(
                vector_storage=vector_storage,
                similarity_threshold=request.min_similarity
            )
            
            # Find similar resolutions for all conflicts
            batch_results = await search_service.batch_find_similar_resolutions(
                conflicts=conflicts,
                limit_per_conflict=request.limit_per_conflict,
                filters=request.filters
            )
            
            # Format the response
            response = {}
            for conflict in conflicts:
                similar_results = batch_results.get(conflict.id, [])
                similar_resolutions = []
                
                for pattern, score in similar_results:
                    similar_resolutions.append({
                        "id": pattern.id,
                        "similarity_score": score,
                        "conflict_type": pattern.conflict_type,
                        "entity_types": pattern.entity_types,
                        "property_names": pattern.property_names,
                        "relationship_types": pattern.relationship_types,
                        "resolution_strategy": pattern.resolution_strategy,
                        "resolution_data": pattern.resolution_data,
                        "confidence": pattern.confidence,
                        "original_conflict_id": pattern.original_conflict_id,
                        "original_merge_id": pattern.original_merge_id,
                        "created_at": pattern.created_at.isoformat() if pattern.created_at else None
                    })
                
                response[conflict.id] = SimilarResolutionResponse(
                    conflict_id=conflict.id,
                    similar_resolutions=similar_resolutions,
                    total_found=len(similar_resolutions)
                )
            
            return response
    
    except Exception as e:
        logger.error(f"Error finding similar resolutions in batch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar resolutions in batch: {str(e)}"
        )

@router.post(
    "/{merge_id}/rollback", 
    response_model=RollbackResponse,
    description="Rollback a merge operation"
)
async def rollback_merge(
    merge_id: str,
    options: RollbackOptions,
    merge_service: MergeService = Depends(get_merge_service)
) -> RollbackResponse:
    """
    Rollback a merge operation completely or partially
    
    Parameters:
    - merge_id: ID of the merge to rollback
    - options: Configuration options for rollback
    """
    try:
        logger.info(f"Rollback requested for merge {merge_id}")
        
        # Check if merge exists
        async with get_merge_service() as merge_service:
            progress = await merge_service.get_merge_progress(merge_id)
            if not progress:
                raise HTTPException(
                    status_code=404,
                    detail=f"Merge {merge_id} not found"
                )
            
            # Check if rollback is already in progress
            if progress.overall_status == MergeStatus.CANCELLED and "rollback" in (progress.error or ""):
                raise HTTPException(
                    status_code=409,
                    detail=f"Rollback already in progress for merge {merge_id}"
                )
            
            # Execute rollback
            result = await merge_service.rollback_merge(merge_id, options)
            
            return result
        
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Rollback validation error: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
        
    except Exception as e:
        # Handle other errors
        logger.error(f"Rollback error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rollback merge: {str(e)}"
        )

@router.get("/progress/{merge_id}", response_model=MergeProgressResponse)
async def get_merge_progress(
    merge_id: str,
    merge_service: MergeService = Depends(get_merge_service)
):
    """
    Get the current progress of an active merge operation
    """
    try:
        async with get_merge_service() as merge_service:
            progress = await merge_service.get_merge_progress(merge_id)
            return progress
    except ValueError as e:
        # Check if this is a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Merge {merge_id} not found")
        # For other value errors, return a 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error
        logger.error(f"Error getting merge progress: {str(e)}")
        # Try to rollback if needed
        try:
            await merge_service.rollback_merge(merge_id)
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {str(rollback_error)}")
        # Return a 500 error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get(
    "/statistics/{merge_id}",
    response_model=MergeStatisticsResponse,
    description="Get detailed statistics of a merge operation"
)
async def get_merge_statistics(
    merge_id: str,
    merge_service: MergeService = Depends(get_merge_service)
) -> MergeStatisticsResponse:
    """Get detailed statistics of a merge operation"""
    try:
        async with get_merge_service() as merge_service:
            statistics = await merge_service.get_merge_statistics(merge_id)
            if not statistics:
                raise HTTPException(status_code=404, detail=f"Merge {merge_id} not found")
            return statistics
    except ValueError as e:
        # Check if this is a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=f"Merge {merge_id} not found")
        # For other value errors, return a 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error
        logger.error(f"Error getting merge statistics: {str(e)}")
        # Return a 500 error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get(
    "/history",
    response_model=List[MergeSummaryResponse],
    description="Get history of merge operations"
)
async def get_merge_history(
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    transform_id: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    merge_service: MergeService = Depends(get_merge_service)
) -> List[MergeSummaryResponse]:
    """Get history of merge operations with filtering"""
    async with get_merge_service() as merge_service:
        history = await merge_service.get_merge_history(
            status=status,
            start_date=start_date,
            end_date=end_date,
            transform_id=transform_id,
            limit=limit,
            offset=offset
        )
        return history

@router.post(
    "/{merge_id}/{session_id}/{transform_id}/finalise",
    response_model=VerificationResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Finalise a merge operation",
    description="Finalise the merge operation by storing the merged data in the database."
)
async def finalise_merge(
    merge_id: str,
    session_id: str,
    transform_id: str,
    merge_service: MergeService = Depends(get_merge_service)
) -> VerificationResult:
    """Finalise a merge operation
    
    Args:
        merge_id: ID of the merge operation
        session_id: ID of the session
        transform_id: ID of the transformation
        
    Returns:
        VerificationResult with verification results
    """
    try:
        # Get merge service
        async with get_merge_service() as merge_service:
            # Verify merge
            verification_result = await merge_service.finalise_and_verify_merge(merge_id, session_id, transform_id)
            return verification_result
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error verifying merge {merge_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error verifying merge: {str(e)}"
        )
