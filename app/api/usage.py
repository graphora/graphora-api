"""
API endpoints for usage tracking and reporting.

This module provides endpoints for viewing usage statistics,
checking limits, and generating billing reports.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
from datetime import datetime, timedelta
import traceback
from app.config import get_settings
from app.services.usage_tracking import usage_tracking_service
from app.schemas.usage import UsageReport, LimitCheckResult, ModelProviderSchema, ModelPricingSchema
from app.utils.logger import logger
from app.auth import get_current_user_id
from decimal import Decimal

settings = get_settings()
router = APIRouter(prefix=settings.API_V1_STR, tags=["Usage & Billing"])


@router.get("/usage/limits", response_model=LimitCheckResult)
async def check_usage_limits(
    user_id: str = Depends(get_current_user_id)
) -> LimitCheckResult:
    """
    Check current usage against tier limits.
    
    Args:
        user_id: User's ID (from header)
        
    Returns:
        LimitCheckResult: Current usage and limit status
        
    Raises:
        HTTPException: 500 on error
    """
    try:
        limit_result = await usage_tracking_service.check_user_limits(user_id)
        return limit_result
        
    except Exception as e:
        logger.error(f"Error checking usage limits for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/usage/report", response_model=UsageReport)
async def get_usage_report(
    user_id: str = Depends(get_current_user_id),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    days: Optional[int] = Query(30, description="Number of days back from today")
) -> UsageReport:
    """
    Get usage report for a time period.
    
    Args:
        user_id: User's ID (from header)
        start_date: Report start date (YYYY-MM-DD format)
        end_date: Report end date (YYYY-MM-DD format)
        days: Days back from today if dates not specified
        
    Returns:
        UsageReport: Detailed usage report
        
    Raises:
        HTTPException: 400 for invalid dates, 500 on error
    """
    try:
        # Parse dates
        if start_date and end_date:
            try:
                period_start = datetime.fromisoformat(start_date)
                period_end = datetime.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use YYYY-MM-DD"
                )
        else:
            # Default to last N days
            period_end = datetime.now()
            period_start = period_end - timedelta(days=days)
        
        # Validate date range
        if period_start >= period_end:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        if (period_end - period_start).days > 365:
            raise HTTPException(
                status_code=400,
                detail="Date range cannot exceed 365 days"
            )
        
        report = await usage_tracking_service.get_usage_report(
            user_id=user_id,
            period_start=period_start,
            period_end=period_end
        )
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating usage report for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/usage/summary")
async def get_usage_summary(
    user_id: str = Depends(get_current_user_id)
) -> JSONResponse:
    """
    Get a quick usage summary for the current billing period.
    
    Args:
        user_id: User's ID (from header)
        
    Returns:
        JSONResponse: Summary statistics
        
    Raises:
        HTTPException: 500 on error
    """
    try:
        # Get current month usage
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        report = await usage_tracking_service.get_usage_report(
            user_id=user_id,
            period_start=month_start,
            period_end=now
        )
        
        # Get limit check
        limits = await usage_tracking_service.check_user_limits(user_id)
        
        summary = {
            "current_period": {
                "start": month_start.isoformat(),
                "end": now.isoformat(),
                "documents_processed": report.total_documents,
                "pages_processed": report.total_pages,
                "llm_calls": report.total_llm_calls,
                "tokens_used": report.total_tokens,
                "estimated_cost_usd": str(report.estimated_total_cost_usd)
            },
            "limits": {
                "tier": limits.tier_name,
                "within_limits": limits.within_limits,
                "document_usage": f"{limits.current_documents}/{limits.document_limit or 'unlimited'}",
                "page_usage": f"{limits.current_pages}/{limits.page_limit or 'unlimited'}",
                "token_usage": f"{limits.current_tokens}/{limits.token_limit or 'unlimited'}",
                "warnings": limits.warnings
            },
            "performance": {
                "avg_processing_time_ms": report.avg_processing_time_ms,
                "success_rate": str(report.success_rate) if report.success_rate else None
            }
        }
        
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error generating usage summary for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/usage/models")
async def get_model_usage_breakdown(
    user_id: str = Depends(get_current_user_id),
    days: int = Query(30, description="Number of days back from today")
) -> JSONResponse:
    """
    Get detailed breakdown of usage by LLM model.
    
    Args:
        user_id: User's ID (from header)
        days: Days back from today
        
    Returns:
        JSONResponse: Model usage breakdown
        
    Raises:
        HTTPException: 500 on error
    """
    try:
        period_end = datetime.now()
        period_start = period_end - timedelta(days=days)
        
        report = await usage_tracking_service.get_usage_report(
            user_id=user_id,
            period_start=period_start,
            period_end=period_end
        )
        
        # Format model usage for better readability
        formatted_models = {}
        for model_key, usage in report.model_usage.items():
            provider, model = model_key.split(':', 1) if ':' in model_key else (model_key, 'unknown')
            
            if provider not in formatted_models:
                formatted_models[provider] = {}
            
            formatted_models[provider][model] = {
                "calls": usage["calls"],
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "estimated_cost_usd": usage["estimated_cost_usd"],
                "avg_tokens_per_call": round(usage["total_tokens"] / usage["calls"], 2) if usage["calls"] > 0 else 0
            }
        
        breakdown = {
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
                "days": days
            },
            "totals": {
                "total_calls": report.total_llm_calls,
                "total_tokens": report.total_tokens,
                "total_cost_usd": str(report.estimated_total_cost_usd)
            },
            "by_provider": formatted_models
        }
        
        return JSONResponse(content=breakdown)
        
    except Exception as e:
        logger.error(f"Error generating model usage breakdown for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/usage/documents")
async def get_document_usage_history(
    user_id: str = Depends(get_current_user_id),
    days: int = Query(30, description="Number of days back from today"),
    limit: int = Query(50, description="Maximum number of documents to return"),
    status: Optional[str] = Query(None, description="Filter by processing status")
) -> JSONResponse:
    """
    Get recent document processing history.
    
    Args:
        user_id: User's ID (from header)
        days: Days back from today
        limit: Maximum documents to return
        status: Filter by processing status (success, failed, etc.)
        
    Returns:
        JSONResponse: Document processing history
        
    Raises:
        HTTPException: 500 on error
    """
    try:
        # This would typically query the database directly for more detailed document info
        # For now, we'll use the report data
        period_end = datetime.now()
        period_start = period_end - timedelta(days=days)
        
        # Note: This is a simplified implementation
        # In production, you'd want to query document_usage table directly
        # for more detailed and paginated results
        
        response = {
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            },
            "filters": {
                "status": status,
                "limit": limit
            },
            "message": "Document history endpoint - would query document_usage table directly in full implementation",
            "note": "Use the usage report endpoint for aggregated document statistics"
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error getting document usage history for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/usage/pricing/providers", response_model=List[ModelProviderSchema])
async def get_model_providers() -> List[ModelProviderSchema]:
    """
    Get all active model providers.
    
    Returns:
        List[ModelProviderSchema]: List of active providers
        
    Raises:
        HTTPException: 500 on error
    """
    try:
        providers = await usage_tracking_service.get_all_model_providers()
        return providers
        
    except Exception as e:
        logger.error(f"Error getting model providers: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/usage/pricing/providers/{provider_name}/models", response_model=List[ModelPricingSchema])
async def get_provider_pricing(provider_name: str) -> List[ModelPricingSchema]:
    """
    Get all pricing for a specific provider.
    
    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')
        
    Returns:
        List[ModelPricingSchema]: List of model pricing for the provider
        
    Raises:
        HTTPException: 500 on error
    """
    try:
        pricing = await usage_tracking_service.get_model_pricing_by_provider(provider_name)
        return pricing
        
    except Exception as e:
        logger.error(f"Error getting pricing for provider {provider_name}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
