from datetime import datetime
from typing import List

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from graphora_server.api.quality import get_current_user_id
from graphora_server.main import app
from graphora_server.services.quality.models import (
    QualityMetrics,
    QualityResults,
    QualityRuleType,
    QualitySeverity,
    QualityViolation,
)
from graphora_server.services.quality.service import QualityService


class _NoopNeo4j:
    async def execute_query(self, query, params=None):  # pragma: no cover - test stub
        return []


def _build_quality_results(violations: List[QualityViolation]) -> QualityResults:
    metrics = QualityMetrics(
        total_entities=1,
        total_relationships=0,
        total_properties=1,
        entities_with_violations=len(violations),
        relationships_with_violations=0,
        total_violations=len(violations),
        entity_violation_rate=float(len(violations) > 0),
        relationship_violation_rate=0.0,
        overall_violation_rate=float(len(violations) > 0),
        avg_entity_confidence=0.9,
        avg_relationship_confidence=0.0,
        confidence_scores_by_type={},
        property_completeness_rate=100.0,
        entity_type_coverage={"Company": 1},
        property_fill_rates_by_entity={"Company": 1.0},
    )

    return QualityResults(
        transform_id="transform-1",
        overall_score=97.2,
        grade="A",
        requires_review=False,
        violations=violations,
        metrics=metrics,
        violations_by_type={},
        violations_by_severity={
            QualitySeverity.ERROR: len(
                [v for v in violations if v.severity == QualitySeverity.ERROR]
            ),
            QualitySeverity.WARNING: len(
                [v for v in violations if v.severity == QualitySeverity.WARNING]
            ),
        },
        violations_by_entity_type={},
        entity_quality_summary={},
        validation_duration_ms=10,
        rules_applied=2,
        validation_config={"gating": {}},
        quality_gate_status="pass",
        quality_gate_reasons=[],
        validation_timestamp=datetime.utcnow(),
    )


@pytest.fixture
def test_client():
    app.dependency_overrides[get_current_user_id] = lambda: "user-123"
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_user_id, None)


def _patch_quality_service(monkeypatch, service: QualityService):
    async def _get_service(user_id: str) -> QualityService:
        return service

    monkeypatch.setattr("graphora_server.api.quality._get_quality_service", _get_service)


def test_list_quality_violations_endpoint_filters(test_client, monkeypatch):
    violation_error = QualityViolation(
        rule_id="Company.missing_name",
        rule_type=QualityRuleType.BUSINESS,
        severity=QualitySeverity.ERROR,
        entity_type="Company",
        entity_id="company-1",
        property_name="name",
        message="Company name is required",
        expected="Non-empty",
        actual="",
        confidence=0.8,
    )

    violation_warning = QualityViolation(
        rule_id="Company.description.short",
        rule_type=QualityRuleType.FORMAT,
        severity=QualitySeverity.WARNING,
        entity_type="Company",
        entity_id="company-1",
        property_name="description",
        message="Description too short",
        expected=">= 20 chars",
        actual="short",
        confidence=0.7,
    )

    quality_results = _build_quality_results([violation_error, violation_warning])

    service = QualityService(_NoopNeo4j())
    service.get_quality_results = AsyncMock(return_value=quality_results)

    _patch_quality_service(monkeypatch, service)

    response = test_client.get(
        "/api/v1/quality/violations/transform-1",
        params={
            "violation_type": QualityRuleType.BUSINESS.value,
            "severity": QualitySeverity.ERROR.value,
            "entity_type": "Company",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["rule_id"] == "Company.missing_name"

    service.get_quality_results.assert_awaited_once()


def test_quality_summary_endpoint_returns_items(test_client, monkeypatch):
    service = QualityService(_NoopNeo4j())
    service.get_quality_results = AsyncMock(return_value=None)
    service.get_quality_summary = AsyncMock(
        return_value=[
            {
                "transform_id": "transform-123",
                "overall_score": 96.4,
                "grade": "A",
                "requires_review": False,
                "total_violations": 2,
                "created_at": "2024-01-01T00:00:00",
                "status": "approved",
            }
        ]
    )

    _patch_quality_service(monkeypatch, service)

    response = test_client.get("/api/v1/quality/summary", params={"limit": 5})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    summary = payload[0]
    assert summary["transform_id"] == "transform-123"
    assert summary["status"] == "approved"
    assert summary["created_at"] == "2024-01-01T00:00:00"

    assert service.get_quality_summary.await_args.args[1] == 5
