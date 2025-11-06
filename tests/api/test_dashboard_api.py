from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.api.dashboard import get_current_user_id
from app.main import app
from app.services.quality.models import (
    QualityMetrics,
    QualityResults,
    QualityRuleType,
    QualitySeverity,
    QualityViolation,
)


class FakeQuery:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows
        self._filters: List[Dict[str, Any]] = []
        self._order: Dict[str, Any] | None = None
        self._limit: int | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, column: str, value: Any):
        self._filters.append({"op": "eq", "column": column, "value": value})
        return self

    def gte(self, column: str, value: str):
        self._filters.append({"op": "gte", "column": column, "value": value})
        return self

    def order(self, column: str, desc: bool = False):
        self._order = {"column": column, "desc": desc}
        return self

    def limit(self, value: int):
        self._limit = value
        return self

    def execute(self):
        data = list(self._rows)
        for filt in self._filters:
            if filt["op"] == "eq":
                data = [row for row in data if row.get(filt["column"]) == filt["value"]]
            elif filt["op"] == "gte":
                threshold = datetime.fromisoformat(filt["value"].replace("Z", "+00:00"))
                data = [
                    row
                    for row in data
                    if datetime.fromisoformat(
                        row[filt["column"]].replace("Z", "+00:00")
                    )
                    >= threshold
                ]
        if self._order:
            column = self._order["column"]
            reverse = bool(self._order.get("desc"))
            data = sorted(data, key=lambda row: row[column], reverse=reverse)
        if self._limit is not None:
            data = data[: self._limit]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, tables: Dict[str, List[Dict[str, Any]]]):
        self._tables = tables

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(list(self._tables.get(name, [])))


class FakeUsageService:
    def __init__(self, supabase: FakeSupabase) -> None:
        self.supabase = supabase


def _build_quality_results(
    transform_id: str,
    *,
    score: float,
    gate_status: str,
    requires_review: bool,
    reasons: List[str],
    violations: List[QualityViolation],
) -> QualityResults:
    metrics = QualityMetrics(
        total_entities=2,
        total_relationships=1,
        total_properties=3,
        entities_with_violations=len(violations),
        relationships_with_violations=0,
        total_violations=len(violations),
        entity_violation_rate=float(len(violations) > 0),
        relationship_violation_rate=0.0,
        overall_violation_rate=float(len(violations) > 0),
        avg_entity_confidence=0.85,
        avg_relationship_confidence=0.8,
        confidence_scores_by_type={"Company": 0.9, "Risk": 0.8},
        property_completeness_rate=92.0,
        entity_type_coverage={"Company": 2, "Risk": 1},
        property_fill_rates_by_entity={"Company": 0.95},
    )

    severity_counts = {
        QualitySeverity.ERROR: len(
            [v for v in violations if v.severity == QualitySeverity.ERROR]
        ),
        QualitySeverity.WARNING: len(
            [v for v in violations if v.severity == QualitySeverity.WARNING]
        ),
    }

    return QualityResults(
        transform_id=transform_id,
        overall_score=score,
        grade="A" if score >= 90 else "B",
        requires_review=requires_review,
        violations=violations,
        metrics=metrics,
        violations_by_type={QualityRuleType.BUSINESS: len(violations)},
        violations_by_severity=severity_counts,
        violations_by_entity_type={"Company": len(violations)},
        entity_quality_summary={},
        validation_duration_ms=25,
        rules_applied=3,
        validation_config={"threshold": 0.8},
        quality_gate_status=gate_status,
        quality_gate_reasons=reasons,
        validation_timestamp=datetime.utcnow(),
    )


@pytest.fixture()
def dashboard_client(monkeypatch):
    now = datetime.utcnow()
    document_rows = [
        {
            "transform_id": "tx-1",
            "session_id": "sess-1",
            "user_id": "user-1",
            "document_name": "alpha.pdf",
            "document_type": "PDF",
            "document_size_bytes": 1024,
            "page_count": 2,
            "processing_status": "success",
            "processing_started_at": (now - timedelta(days=1)).isoformat(),
            "processing_completed_at": (
                now - timedelta(days=1) + timedelta(minutes=5)
            ).isoformat(),
            "processing_duration_ms": 5 * 60 * 1000,
            "chunks_created": 4,
            "nodes_extracted": 12,
            "relationships_extracted": 5,
        },
        {
            "transform_id": "tx-2",
            "session_id": "sess-2",
            "user_id": "user-1",
            "document_name": "beta.txt",
            "document_type": "TXT",
            "document_size_bytes": 2048,
            "page_count": 1,
            "processing_status": "failed",
            "processing_started_at": (now - timedelta(days=2)).isoformat(),
            "processing_completed_at": (
                now - timedelta(days=2) + timedelta(minutes=2)
            ).isoformat(),
            "processing_duration_ms": 2 * 60 * 1000,
            "chunks_created": 2,
            "nodes_extracted": 4,
            "relationships_extracted": 0,
        },
    ]

    llm_rows = [
        {
            "transform_id": "tx-1",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "estimated_cost_usd": 0.12,
        },
        {
            "transform_id": "tx-1",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "input_tokens": 60,
            "output_tokens": 15,
            "total_tokens": 75,
            "estimated_cost_usd": 0.05,
        },
        {
            "transform_id": "tx-2",
            "model_provider": "anthropic",
            "model_name": "claude",
            "input_tokens": 80,
            "output_tokens": 20,
            "total_tokens": 100,
            "estimated_cost_usd": 0.09,
        },
    ]

    supabase = FakeSupabase(
        {
            "document_usage": document_rows,
            "llm_usage": llm_rows,
        }
    )

    usage_service = FakeUsageService(supabase)
    monkeypatch.setattr("app.api.dashboard.usage_tracking_service", usage_service)

    violation = QualityViolation(
        rule_id="Company.name.missing",
        rule_type=QualityRuleType.BUSINESS,
        severity=QualitySeverity.ERROR,
        entity_type="Company",
        entity_id="company-1",
        property_name="name",
        message="Company name missing",
        expected="Non-empty",
        actual="",
        confidence=0.7,
    )

    quality_map = {
        "tx-1": _build_quality_results(
            "tx-1",
            score=96.5,
            gate_status="pass",
            requires_review=False,
            reasons=["High confidence across entities"],
            violations=[],
        ),
        "tx-2": _build_quality_results(
            "tx-2",
            score=72.1,
            gate_status="warn",
            requires_review=True,
            reasons=["Missing required property"],
            violations=[violation],
        ),
    }

    class FakeQualityService:
        async def get_quality_results(self, transform_id: str, user_id: str):
            assert user_id == "user-1"
            return quality_map.get(transform_id)

    async def _fake_quality_service(_user_id: str):
        return FakeQualityService()

    monkeypatch.setattr("app.api.dashboard._get_quality_service", _fake_quality_service)

    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_user_id, None)
        import app.api.dashboard as dashboard_module

        dashboard_module.usage_tracking_service = None


def test_dashboard_summary_endpoint_returns_metrics(dashboard_client):
    response = dashboard_client.get("/api/v1/dashboard/summary", params={"days": 7})
    assert response.status_code == 200
    payload = response.json()

    assert payload["total_runs"] == 2
    assert payload["completed_runs"] == 1
    assert payload["failed_runs"] == 1
    assert payload["warn_count"] == 1
    assert payload["pass_count"] == 1
    assert payload["requires_review_count"] == 1
    assert pytest.approx(payload["total_tokens"], rel=1e-3) == 325
    assert payload["recent_gate_reasons"][0] == "Missing required property"


def test_dashboard_performance_endpoint_timeseries(dashboard_client):
    response = dashboard_client.get("/api/v1/dashboard/performance", params={"days": 7})
    assert response.status_code == 200
    payload = response.json()

    assert payload["total_runs"] == 2
    timeseries = payload["timeseries"]
    assert len(timeseries) == 2
    # entries should be sorted ascending by date
    assert timeseries[0]["runs"] == 1
    assert timeseries[0]["total_llm_calls"] >= 1


def test_dashboard_quality_endpoint_stats(dashboard_client):
    response = dashboard_client.get("/api/v1/dashboard/quality", params={"days": 7})
    assert response.status_code == 200
    payload = response.json()

    assert payload["pass_count"] == 1
    assert payload["warn_count"] == 1
    assert payload["requires_review_count"] == 1
    assert payload["recent_reasons"][0]["reason"] == "Missing required property"
    assert payload["top_rules"][0]["rule_id"] == "Company.name.missing"
    coverage = {row["entity_type"]: row["count"] for row in payload["entity_coverage"]}
    assert coverage["Company"] == 4
