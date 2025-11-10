"""Dashboard analytics endpoints."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth import get_current_user_id
from app.config import settings
from app.db import postgres as db
from app.services.quality.models import QualityResults, QualitySeverity
from app.services.quality.service import QualityService
from app.services.usage_tracking import UsageTrackingService

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


class DashboardSummaryResponse(BaseModel):
    """Aggregate pipeline snapshot for the dashboard top-line KPIs."""

    window_start: datetime
    window_end: datetime
    total_runs: int
    completed_runs: int
    failed_runs: int
    running_runs: int
    pass_count: int
    warn_count: int
    fail_count: int
    requires_review_count: int
    average_duration_ms: Optional[float]
    p50_duration_ms: Optional[float]
    p95_duration_ms: Optional[float]
    average_tokens_per_run: Optional[float]
    total_tokens: int
    total_llm_calls: int
    total_estimated_cost_usd: Optional[float]
    runs_per_day: Optional[float]
    recent_gate_reasons: List[str] = Field(default_factory=list)


class PerformanceTimeseriesPoint(BaseModel):
    """Daily performance metrics for dashboard charts."""

    date: date
    runs: int
    average_duration_ms: Optional[float]
    p95_duration_ms: Optional[float]
    total_tokens: int
    total_llm_calls: int
    total_estimated_cost_usd: Optional[float]


class DashboardPerformanceResponse(BaseModel):
    window_start: datetime
    window_end: datetime
    total_runs: int
    total_tokens: int
    total_llm_calls: int
    total_estimated_cost_usd: Optional[float]
    timeseries: List[PerformanceTimeseriesPoint]


class QualityReasonStat(BaseModel):
    reason: str
    count: int


class QualityRuleStat(BaseModel):
    rule_id: str
    severity: str
    count: int


class EntityCoverageStat(BaseModel):
    entity_type: str
    count: int


class EntityConfidenceStat(BaseModel):
    entity_type: str
    average_confidence: float


class DashboardQualityResponse(BaseModel):
    window_start: datetime
    window_end: datetime
    average_score: Optional[float]
    p50_score: Optional[float]
    p95_score: Optional[float]
    pass_count: int
    warn_count: int
    fail_count: int
    requires_review_count: int
    recent_reasons: List[QualityReasonStat]
    top_rules: List[QualityRuleStat]
    entity_coverage: List[EntityCoverageStat]
    entity_confidence: List[EntityConfidenceStat]


usage_tracking_service: Optional[UsageTrackingService] = None


def _get_usage_service() -> UsageTrackingService:
    global usage_tracking_service  # noqa: PLW0603 - intentional singleton
    if usage_tracking_service is None:
        usage_tracking_service = UsageTrackingService()
    return usage_tracking_service


async def _get_quality_service(user_id: str) -> QualityService:
    from app.services.storage.neo4j import Neo4jStorage
    from app.services.user_db_service import UserDatabaseService

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


def _quality_reason_priority(
    status: Optional[str],
    requires_review: Optional[bool],
) -> int:
    if requires_review:
        return 0
    normalised = (status or "").lower()
    if normalised == "fail":
        return 1
    if normalised in {"warn", "warning"}:
        return 2
    if normalised == "pass":
        return 3
    return 4


def _sort_reasons(
    counter: Counter[str],
    priorities: Dict[str, int],
) -> List[Tuple[str, int]]:
    items = list(counter.items())
    items.sort(
        key=lambda item: (
            priorities.get(item[0], 4),
            -item[1],
            item[0].lower(),
        )
    )
    return items


def _percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(
        min(percent / 100.0 * (len(sorted_values) - 1), len(sorted_values) - 1), 0
    )
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return mean(values) if values else None


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


async def _hydrate_runs(
    user_id: str,
    records: Iterable[Dict[str, Any]],
) -> Tuple[List[TransformRunSummary], List[Optional[QualityResults]]]:
    records = list(records)
    if not records:
        return [], []

    transform_ids = [
        record.get("transform_id") for record in records if record.get("transform_id")
    ]

    llm_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if transform_ids:
        llm_rows = await db.fetch(
            """
            SELECT transform_id, model_provider, model_name,
                   input_tokens, output_tokens, total_tokens, estimated_cost_usd
            FROM llm_usage
            WHERE transform_id = ANY(%s)
            """,
            transform_ids,
        )
        for row in llm_rows or []:
            key = row.get("transform_id")
            if key:
                llm_map[key].append(row)

    quality_service: Optional[QualityService] = None
    try:
        quality_service = await _get_quality_service(user_id)
    except Exception:
        quality_service = None

    summaries: List[TransformRunSummary] = []
    quality_results: List[Optional[QualityResults]] = []

    for record in records:
        transform_id = record.get("transform_id")
        if not transform_id:
            continue

        llm_usage_summary = LLMUsageSummary()
        try:
            llm_rows = llm_map.get(transform_id, [])
            model_set = set()
            for row in llm_rows:
                llm_usage_summary.total_calls += 1
                llm_usage_summary.input_tokens += int(row.get("input_tokens", 0) or 0)
                llm_usage_summary.output_tokens += int(row.get("output_tokens", 0) or 0)
                llm_usage_summary.total_tokens += int(row.get("total_tokens", 0) or 0)
                estimated = row.get("estimated_cost_usd")
                if estimated is not None:
                    llm_usage_summary.estimated_cost_usd = (
                        llm_usage_summary.estimated_cost_usd or 0.0
                    ) + _decimal_to_float(Decimal(str(estimated)))
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
        except Exception as exc:  # pragma: no cover - db errors
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        quality_details: Dict[str, Any]
        quality_result: Optional[QualityResults] = None
        try:
            if quality_service is not None:
                quality_result = await quality_service.get_quality_results(
                    transform_id, user_id
                )
        except Exception:
            quality_result = None

        quality_details = _normalise_quality(quality_result)
        quality_results.append(quality_result)

        started_at = _coerce_datetime(record.get("processing_started_at"))
        completed_at = _coerce_datetime(record.get("processing_completed_at"))

        summaries.append(
            TransformRunSummary(
                transform_id=transform_id,
                session_id=record.get("session_id"),
                document_name=record.get("document_name", "Unknown"),
                document_type=record.get("document_type", ""),
                document_size_bytes=int(record.get("document_size_bytes", 0) or 0),
                page_count=int(record.get("page_count", 0) or 0),
                processing_status=(record.get("processing_status") or "").lower(),
                processing_started_at=started_at or datetime.utcnow(),
                processing_completed_at=completed_at,
                processing_duration_ms=(
                    int(record.get("processing_duration_ms", 0) or 0)
                    if record.get("processing_duration_ms") is not None
                    else None
                ),
                chunks_created=int(record.get("chunks_created", 0) or 0),
                nodes_extracted=int(record.get("nodes_extracted", 0) or 0),
                relationships_extracted=int(
                    record.get("relationships_extracted", 0) or 0
                ),
                llm_usage=llm_usage_summary,
                **quality_details,
            )
        )

    return summaries, quality_results


def _query_document_usage(
    user_id: str,
    days: int,
    limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], datetime, datetime]:
    window_end = datetime.utcnow()
    window_start = window_end - timedelta(days=days)

    try:
        query = """
            SELECT *
            FROM document_usage
            WHERE user_id = %s AND processing_started_at >= %s
            ORDER BY processing_started_at DESC
        """
        params: List[Any] = [user_id, window_start]
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)
        records = db.sync_fetch(query, *params)
    except Exception as exc:  # pragma: no cover - db errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    records = records or []
    return records, window_start, window_end


def _aggregate_llm_totals(
    runs: Iterable[TransformRunSummary],
) -> Tuple[int, int, Optional[float]]:
    total_tokens = 0
    total_calls = 0
    total_cost: Optional[float] = 0.0

    for run in runs:
        total_tokens += run.llm_usage.total_tokens
        total_calls += run.llm_usage.total_calls
        if run.llm_usage.estimated_cost_usd is not None:
            total_cost = (total_cost or 0.0) + run.llm_usage.estimated_cost_usd

    return total_tokens, total_calls, (round(total_cost, 4) if total_cost else None)


@router.get("/dashboard/runs", response_model=RecentRunsResponse)
async def get_recent_transform_runs(
    limit: int = Query(20, ge=1, le=50),
    days: int = Query(14, ge=1, le=90),
    user_id: str = Depends(get_current_user_id),
):
    """Return the most recent transform runs with quality and usage context."""

    records, window_start, window_end = _query_document_usage(user_id, days, limit)
    summaries, _ = await _hydrate_runs(user_id, records)

    return RecentRunsResponse(
        runs=summaries,
        window_start=window_start,
        window_end=window_end,
    )


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(
    days: int = Query(14, ge=1, le=90),
    max_runs: int = Query(200, ge=1, le=1000),
    user_id: str = Depends(get_current_user_id),
):
    records, window_start, window_end = _query_document_usage(user_id, days, max_runs)
    summaries, quality_results = await _hydrate_runs(user_id, records)

    if not summaries:
        return DashboardSummaryResponse(
            window_start=window_start,
            window_end=window_end,
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            running_runs=0,
            pass_count=0,
            warn_count=0,
            fail_count=0,
            requires_review_count=0,
            average_duration_ms=None,
            p50_duration_ms=None,
            p95_duration_ms=None,
            average_tokens_per_run=None,
            total_tokens=0,
            total_llm_calls=0,
            total_estimated_cost_usd=None,
            runs_per_day=None,
            recent_gate_reasons=[],
        )

    durations = [
        run.processing_duration_ms
        for run in summaries
        if run.processing_duration_ms is not None
    ]

    total_runs = len(summaries)
    completed_runs = sum(1 for run in summaries if run.processing_status == "success")
    failed_runs = sum(1 for run in summaries if run.processing_status == "failed")
    running_runs = total_runs - completed_runs - failed_runs

    pass_count = sum(
        1 for run in summaries if (run.quality_gate_status or "").lower() == "pass"
    )
    warn_count = sum(
        1
        for run in summaries
        if (run.quality_gate_status or "").lower() in {"warn", "warning"}
    )
    fail_count = sum(
        1 for run in summaries if (run.quality_gate_status or "").lower() == "fail"
    )

    requires_review_count = sum(
        1
        for run in summaries
        if run.quality_requires_review
        or (run.quality_gate_status or "").lower() in {"warn", "warning", "fail"}
    )

    durations_sec = [d for d in durations if d is not None]
    average_duration_ms = float(mean(durations_sec)) if durations_sec else None
    p50_duration_ms = _percentile(durations_sec, 50.0)
    p95_duration_ms = _percentile(durations_sec, 95.0)

    total_tokens, total_calls, total_cost = _aggregate_llm_totals(summaries)
    average_tokens_per_run = (
        total_tokens / total_runs if total_runs and total_tokens else None
    )

    diff_days = max((window_end - window_start).total_seconds() / 86400.0, 1.0)
    runs_per_day = round(total_runs / diff_days, 2)

    reason_counter: Counter[str] = Counter()
    reason_priorities: Dict[str, int] = {}
    for run, result in zip(summaries, quality_results):
        if not result or not result.quality_gate_reasons:
            continue

        priority = _quality_reason_priority(
            run.quality_gate_status,
            run.quality_requires_review,
        )
        for reason in (
            reason.strip() for reason in result.quality_gate_reasons if reason.strip()
        ):
            reason_counter[reason] += 1
            reason_priorities[reason] = min(
                reason_priorities.get(reason, priority),
                priority,
            )

    recent_reasons = [
        reason for reason, _ in _sort_reasons(reason_counter, reason_priorities)[:5]
    ]

    return DashboardSummaryResponse(
        window_start=window_start,
        window_end=window_end,
        total_runs=total_runs,
        completed_runs=completed_runs,
        failed_runs=failed_runs,
        running_runs=running_runs,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        requires_review_count=requires_review_count,
        average_duration_ms=average_duration_ms,
        p50_duration_ms=p50_duration_ms,
        p95_duration_ms=p95_duration_ms,
        average_tokens_per_run=average_tokens_per_run,
        total_tokens=total_tokens,
        total_llm_calls=total_calls,
        total_estimated_cost_usd=total_cost,
        runs_per_day=runs_per_day,
        recent_gate_reasons=recent_reasons,
    )


@router.get("/dashboard/performance", response_model=DashboardPerformanceResponse)
async def get_dashboard_performance(
    days: int = Query(14, ge=1, le=90),
    max_runs: int = Query(200, ge=1, le=1000),
    user_id: str = Depends(get_current_user_id),
):
    records, window_start, window_end = _query_document_usage(user_id, days, max_runs)
    summaries, _ = await _hydrate_runs(user_id, records)

    if not summaries:
        return DashboardPerformanceResponse(
            window_start=window_start,
            window_end=window_end,
            total_runs=0,
            total_tokens=0,
            total_llm_calls=0,
            total_estimated_cost_usd=None,
            timeseries=[],
        )

    buckets: Dict[date, Dict[str, Any]] = defaultdict(
        lambda: {
            "durations": [],
            "tokens": 0,
            "calls": 0,
            "cost": 0.0,
            "runs": 0,
        }
    )

    for run in summaries:
        started_at = run.processing_started_at.date()
        bucket = buckets[started_at]
        bucket["runs"] += 1
        if run.processing_duration_ms is not None:
            bucket["durations"].append(run.processing_duration_ms)
        bucket["tokens"] += run.llm_usage.total_tokens
        bucket["calls"] += run.llm_usage.total_calls
        if run.llm_usage.estimated_cost_usd is not None:
            bucket["cost"] += run.llm_usage.estimated_cost_usd

    total_tokens, total_calls, total_cost = _aggregate_llm_totals(summaries)

    timeseries = [
        PerformanceTimeseriesPoint(
            date=day,
            runs=data["runs"],
            average_duration_ms=_safe_mean(data["durations"]),
            p95_duration_ms=_percentile(data["durations"], 95.0),
            total_tokens=data["tokens"],
            total_llm_calls=data["calls"],
            total_estimated_cost_usd=(round(data["cost"], 4) if data["cost"] else None),
        )
        for day, data in sorted(buckets.items())
    ]

    return DashboardPerformanceResponse(
        window_start=window_start,
        window_end=window_end,
        total_runs=len(summaries),
        total_tokens=total_tokens,
        total_llm_calls=total_calls,
        total_estimated_cost_usd=total_cost,
        timeseries=timeseries,
    )


@router.get("/dashboard/quality", response_model=DashboardQualityResponse)
async def get_dashboard_quality(
    days: int = Query(14, ge=1, le=90),
    max_runs: int = Query(200, ge=1, le=1000),
    user_id: str = Depends(get_current_user_id),
):
    records, window_start, window_end = _query_document_usage(user_id, days, max_runs)
    summaries, quality_results = await _hydrate_runs(user_id, records)

    if not summaries:
        return DashboardQualityResponse(
            window_start=window_start,
            window_end=window_end,
            average_score=None,
            p50_score=None,
            p95_score=None,
            pass_count=0,
            warn_count=0,
            fail_count=0,
            requires_review_count=0,
            recent_reasons=[],
            top_rules=[],
            entity_coverage=[],
            entity_confidence=[],
        )

    scores = [run.quality_score for run in summaries if run.quality_score is not None]
    pass_count = sum(
        1 for run in summaries if (run.quality_gate_status or "").lower() == "pass"
    )
    warn_count = sum(
        1
        for run in summaries
        if (run.quality_gate_status or "").lower() in {"warn", "warning"}
    )
    fail_count = sum(
        1 for run in summaries if (run.quality_gate_status or "").lower() == "fail"
    )
    requires_review_count = sum(
        1
        for run in summaries
        if run.quality_requires_review
        or (run.quality_gate_status or "").lower() in {"warn", "warning", "fail"}
    )

    reason_counter: Counter[str] = Counter()
    reason_priorities: Dict[str, int] = {}
    rule_counter: Counter[Tuple[str, str]] = Counter()
    entity_coverage_counter: Counter[str] = Counter()
    confidence_totals: Dict[str, List[float]] = defaultdict(list)

    for result in quality_results:
        if not result:
            continue

        if result.quality_gate_reasons:
            priority = _quality_reason_priority(
                result.quality_gate_status,
                result.requires_review,
            )
            for reason in (
                reason.strip()
                for reason in result.quality_gate_reasons
                if reason.strip()
            ):
                reason_counter[reason] += 1
                reason_priorities[reason] = min(
                    reason_priorities.get(reason, priority),
                    priority,
                )

        for violation in result.violations:
            severity = (
                violation.severity.value
                if isinstance(violation.severity, QualitySeverity)
                else str(violation.severity)
            )
            rule_counter[(violation.rule_id, severity)] += 1

        if result.metrics.entity_type_coverage:
            entity_coverage_counter.update(result.metrics.entity_type_coverage)

        for entity_type, confidence in (
            result.metrics.confidence_scores_by_type or {}
        ).items():
            confidence_totals[entity_type].append(confidence)

    average_score = (sum(scores) / len(scores)) if scores else None

    return DashboardQualityResponse(
        window_start=window_start,
        window_end=window_end,
        average_score=average_score,
        p50_score=_percentile(scores, 50.0),
        p95_score=_percentile(scores, 95.0),
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        requires_review_count=requires_review_count,
        recent_reasons=[
            QualityReasonStat(reason=reason, count=count)
            for reason, count in _sort_reasons(reason_counter, reason_priorities)[:5]
        ],
        top_rules=[
            QualityRuleStat(rule_id=rule_id, severity=severity, count=count)
            for (rule_id, severity), count in rule_counter.most_common(5)
        ],
        entity_coverage=[
            EntityCoverageStat(entity_type=entity_type, count=count)
            for entity_type, count in entity_coverage_counter.most_common()
        ],
        entity_confidence=[
            EntityConfidenceStat(
                entity_type=entity_type,
                average_confidence=round(mean(values), 3),
            )
            for entity_type, values in sorted(confidence_totals.items())
        ],
    )
