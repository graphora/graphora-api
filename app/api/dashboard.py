"""Dashboard analytics endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth import get_current_user_id
from app.config import settings
from app.services.quality.service import QualityService
from app.services.usage_tracking import UsageTrackingService
from app.services.quality.models import QualityResults


router = APIRouter(prefix=settings.API_V1_STR, tags=["Dashboard"])


class LLMUsageSummary(BaseModel):
    """Aggregated LLM usage for a transform run."""

    total_calls: int = Field(default=0, description="Total LLM invocations")
    input_tokens: int = Field(default=0, description="Total input tokens")
    output_tokens: int = Field(default=0, description="Total output tokens")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")
    estimated_cost_usd: Optional[float] = Field(
        default=None, description="Estimated total cost in USD"
    )
    models_used: List[str] = Field(
        default_factory=list, description="Distinct provider/model pairs"
    )


class TransformRunSummary(BaseModel):
    """Summary information for a transform run used by the dashboard."""

    transform_id: str
    session_id: Optional[str] = None
    document_name: str
    document_type: str
    document_size_bytes: int
    page_count: int
    processing_status: str
    processing_started_at: datetime
    processing_completed_at: Optional[datetime] = None
    processing_duration_ms: Optional[int] = None
    chunks_created: int = 0
    nodes_extracted: int = 0
    relationships_extracted: int = 0

    quality_score: Optional[float] = None
    quality_gate_status: Optional[str] = None
    quality_requires_review: Optional[bool] = None
    quality_gate_reasons: List[str] = Field(default_factory=list)

    llm_usage: LLMUsageSummary = Field(
        default_factory=LLMUsageSummary,
        description="Summarised LLM usage for the transform",
    )


class RecentRunsResponse(BaseModel):
    """Response model for dashboard recent runs endpoint."""

    runs: List[TransformRunSummary]
    window_start: datetime
    window_end: datetime


usage_tracking_service: Optional[UsageTrackingService] = None


def _get_usage_service() -> UsageTrackingService:
    global usage_tracking_service  # noqa: PLW0603 - intentional singleton
    if usage_tracking_service is None:
        usage_tracking_service = UsageTrackingService()
    return usage_tracking_service


async def _get_quality_service(user_id: str) -> QualityService:
    from app.services.user_db_service import UserDatabaseService
    from app.services.storage.neo4j import Neo4jStorage

    user_config = await UserDatabaseService.get_user_config(user_id)

    storage = Neo4jStorage(
        uri=user_config.stagingDb.uri,
        username=user_config.stagingDb.username,
        password=user_config.stagingDb.password,
        database="neo4j",
    )
    return QualityService(storage)


def _decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_quality(quality: Optional[QualityResults]) -> Dict[str, Any]:
    if not quality:
        return {
            "quality_score": None,
            "quality_gate_status": None,
            "quality_requires_review": None,
            "quality_gate_reasons": [],
        }

    return {
        "quality_score": quality.overall_score,
        "quality_gate_status": getattr(quality, "quality_gate_status", None),
        "quality_requires_review": quality.requires_review,
        "quality_gate_reasons": getattr(quality, "quality_gate_reasons", []) or [],
    }


@router.get("/dashboard/runs", response_model=RecentRunsResponse)
async def get_recent_transform_runs(
    limit: int = Query(20, ge=1, le=50),
    days: int = Query(14, ge=1, le=90),
    user_id: str = Depends(get_current_user_id),
):
    """Return the most recent transform runs with quality and usage context."""

    usage_service = _get_usage_service()

    try:
        supabase = usage_service.supabase
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    window_end = datetime.utcnow()
    window_start = window_end - timedelta(days=days)

    try:
        response = (
            supabase.table("document_usage")
            .select("*")
            .eq("user_id", user_id)
            .gte("processing_started_at", window_start.isoformat())
            .order("processing_started_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:  # pragma: no cover - supabase errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    records: List[Dict[str, Any]] = response.data or []

    quality_service: Optional[QualityService] = None
    runs: List[TransformRunSummary] = []

    for record in records:
        transform_id = record.get("transform_id")
        if not transform_id:
            continue

        # Aggregate LLM usage for the transform
        llm_usage_summary = LLMUsageSummary()
        try:
            llm_response = (
                supabase.table("llm_usage")
                .select(
                    "model_provider, model_name, input_tokens, output_tokens, total_tokens, estimated_cost_usd"
                )
                .eq("transform_id", transform_id)
                .execute()
            )
            llm_rows = llm_response.data or []
            model_set = set()
            for row in llm_rows:
                llm_usage_summary.total_calls += 1
                llm_usage_summary.input_tokens += int(row.get("input_tokens", 0) or 0)
                llm_usage_summary.output_tokens += int(row.get("output_tokens", 0) or 0)
                llm_usage_summary.total_tokens += int(row.get("total_tokens", 0) or 0)
                estimated = row.get("estimated_cost_usd")
                if estimated is not None:
                    llm_usage_summary.estimated_cost_usd = (
                        (llm_usage_summary.estimated_cost_usd or 0.0)
                        + _decimal_to_float(Decimal(str(estimated)))
                    )
                provider = row.get("model_provider")
                model = row.get("model_name")
                if provider and model:
                    model_set.add(f"{provider}:{model}")
            if model_set:
                llm_usage_summary.models_used = sorted(model_set)
            if llm_usage_summary.estimated_cost_usd is not None:
                llm_usage_summary.estimated_cost_usd = round(
                    llm_usage_summary.estimated_cost_usd, 4
                )
        except Exception as exc:  # pragma: no cover - supabase errors
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Fetch quality results if available
        quality_details: Dict[str, Any] = {
            "quality_score": None,
            "quality_gate_status": None,
            "quality_requires_review": None,
            "quality_gate_reasons": [],
        }

        try:
            if quality_service is None:
                quality_service = await _get_quality_service(user_id)
            quality_result = await quality_service.get_quality_results(
                transform_id, user_id
            )
            quality_details = _normalise_quality(quality_result)
        except HTTPException:  # type: ignore[bare-except] - reuse FastAPI errors
            quality_details = _normalise_quality(None)
        except Exception:
            quality_details = _normalise_quality(None)

        runs.append(
            TransformRunSummary(
                transform_id=transform_id,
                session_id=record.get("session_id"),
                document_name=record.get("document_name", "Unknown"),
                document_type=record.get("document_type", ""),
                document_size_bytes=int(record.get("document_size_bytes", 0) or 0),
                page_count=int(record.get("page_count", 0) or 0),
                processing_status=(record.get("processing_status") or "").lower(),
                processing_started_at=datetime.fromisoformat(
                    record["processing_started_at"].replace("Z", "+00:00")
                ),
                processing_completed_at=(
                    datetime.fromisoformat(
                        record["processing_completed_at"].replace("Z", "+00:00")
                    )
                    if record.get("processing_completed_at")
                    else None
                ),
                processing_duration_ms=
                    int(record.get("processing_duration_ms", 0) or 0)
                    if record.get("processing_duration_ms") is not None
                    else None,
                chunks_created=int(record.get("chunks_created", 0) or 0),
                nodes_extracted=int(record.get("nodes_extracted", 0) or 0),
                relationships_extracted=int(
                    record.get("relationships_extracted", 0) or 0
                ),
                llm_usage=llm_usage_summary,
                **quality_details,
            )
        )

    return RecentRunsResponse(
        runs=runs,
        window_start=window_start,
        window_end=window_end,
    )
