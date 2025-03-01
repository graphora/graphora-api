from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.dependencies import get_merge_service
from app.schemas.conflicts import ConflictType
from app.schemas.resolution_history import ResolutionHistoryEntry
from app.services.resolution_history_service import ResolutionHistoryService
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/resolutions", tags=["resolutions"])

# Pydantic models for request and response
class ResolutionFilter(BaseModel):
    """Filter parameters for resolution queries"""
    conflict_type: Optional[ConflictType] = Field(None, description="Filter by conflict type")
    resolution_type: Optional[str] = Field(None, description="Filter by resolution strategy used")
    start_date: Optional[datetime] = Field(None, description="Filter by start date (inclusive)")
    end_date: Optional[datetime] = Field(None, description="Filter by end date (inclusive)")
    user: Optional[str] = Field(None, description="Filter by user who applied resolution")
    effectiveness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Filter by effectiveness rating")
    entity_type: Optional[str] = Field(None, description="Filter by entity type")

class PaginationParams(BaseModel):
    """Pagination parameters"""
    limit: int = Field(10, ge=1, le=100, description="Maximum number of items to return")
    offset: int = Field(0, ge=0, description="Number of items to skip")
    sort_by: str = Field("applied_at", description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order (asc or desc)")

class ResolutionResponse(BaseModel):
    """Response model for resolution queries"""
    items: List[ResolutionHistoryEntry] = Field(..., description="List of resolution history entries")
    total: int = Field(..., description="Total number of items matching the query")
    limit: int = Field(..., description="Maximum number of items returned")
    offset: int = Field(..., description="Number of items skipped")

class ResolutionStats(BaseModel):
    """Statistics about resolutions"""
    total_resolutions: int = Field(..., description="Total number of resolutions")
    by_conflict_type: Dict[str, int] = Field(..., description="Resolutions by conflict type")
    by_resolution_type: Dict[str, int] = Field(..., description="Resolutions by resolution type")
    by_entity_type: Dict[str, int] = Field(..., description="Resolutions by entity type")
    by_user: Dict[str, int] = Field(..., description="Resolutions by user")
    success_rate: float = Field(..., description="Overall success rate")
    average_effectiveness: float = Field(..., description="Average effectiveness rating")
    time_distribution: Dict[str, int] = Field(..., description="Distribution by time period")

@router.get("/stats", response_model=ResolutionStats)
async def get_resolution_stats(
    start_date: Optional[datetime] = Query(None, description="Filter by start date (inclusive)"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date (inclusive)"),
    merge_service = Depends(get_merge_service)
):
    """
    Get statistics about resolutions.
    
    This endpoint returns various statistics about resolutions, such as counts by type,
    success rates, and effectiveness metrics.
    """
    # Get resolution history service from merge service
    resolution_service: ResolutionHistoryService = merge_service.resolution_history
    
    # Get statistics
    stats = await resolution_service.get_resolution_stats(
        start_date=start_date,
        end_date=end_date
    )
    
    return ResolutionStats(**stats)

@router.get("/{merge_id}", response_model=ResolutionResponse)
async def get_resolutions_by_merge_id(
    merge_id: str = Path(..., description="ID of the merge process"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    sort_by: str = Query("applied_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    merge_service = Depends(get_merge_service)
):
    """
    Retrieve all resolutions for a specific merge process.
    
    This endpoint returns all resolution history entries associated with a specific merge ID,
    with support for pagination and sorting.
    """
    # Get resolution history service from merge service
    resolution_service: ResolutionHistoryService = merge_service.resolution_history
    
    # Get resolutions for the merge ID
    resolutions = await resolution_service.get_resolution_history(
        merge_id=merge_id,
        limit=limit,
        offset=offset
    )
    
    # Get total count for pagination
    total_count = await resolution_service.get_resolution_count(merge_id=merge_id)
    
    return ResolutionResponse(
        items=resolutions,
        total=total_count,
        limit=limit,
        offset=offset
    )

@router.get("", response_model=ResolutionResponse)
async def filter_resolutions(
    conflict_type: Optional[ConflictType] = Query(None, description="Filter by conflict type"),
    resolution_type: Optional[str] = Query(None, description="Filter by resolution strategy used"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date (inclusive)"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date (inclusive)"),
    user: Optional[str] = Query(None, description="Filter by user who applied resolution"),
    effectiveness: Optional[float] = Query(None, ge=0.0, le=1.0, description="Filter by effectiveness rating"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    sort_by: str = Query("applied_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    merge_service = Depends(get_merge_service)
):
    """
    Query resolutions with filtering, pagination, and sorting.
    
    This endpoint allows filtering resolutions by various criteria, with support for
    pagination and sorting.
    """
    # Get resolution history service from merge service
    resolution_service: ResolutionHistoryService = merge_service.resolution_history
    
    # Create filter object
    filter_params = ResolutionFilter(
        conflict_type=conflict_type,
        resolution_type=resolution_type,
        start_date=start_date,
        end_date=end_date,
        user=user,
        effectiveness=effectiveness,
        entity_type=entity_type
    )
    
    # Create pagination params
    pagination_params = PaginationParams(
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    # Get filtered resolutions
    resolutions, total_count = await resolution_service.filter_resolutions(
        filter_params=filter_params,
        pagination_params=pagination_params
    )
    
    return ResolutionResponse(
        items=resolutions,
        total=total_count,
        limit=limit,
        offset=offset
    ) 