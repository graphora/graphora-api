from datetime import datetime

import pytest

from app.services.quality.models import (
    QualityMetrics,
    QualityResults,
    QualitySeverity,
    QualityRuleType,
)
from app.services.quality.service import QualityService
class DummyNeo4jStorage:
    async def execute_query(self, *args, **kwargs):
        return []


@pytest.fixture
def quality_service():
    return QualityService(neo4j_storage=DummyNeo4jStorage())


@pytest.fixture
def base_quality_results():
    metrics = QualityMetrics(
        total_entities=1,
        total_relationships=0,
        total_properties=1,
        entities_with_violations=0,
        relationships_with_violations=0,
        total_violations=0,
        entity_violation_rate=0.0,
        relationship_violation_rate=0.0,
        overall_violation_rate=0.0,
        avg_entity_confidence=0.9,
        avg_relationship_confidence=0.0,
        confidence_scores_by_type={},
        property_completeness_rate=100.0,
        entity_type_coverage={"Company": 1},
        property_fill_rates_by_entity={"Company": 1.0},
    )

    return QualityResults(
        transform_id="transform-1",
        overall_score=99.0,
        grade="A",
        requires_review=False,
        violations=[],
        metrics=metrics,
        violations_by_type={},
        violations_by_severity={QualitySeverity.ERROR: 0, QualitySeverity.WARNING: 0},
        violations_by_entity_type={},
        entity_quality_summary={},
        validation_duration_ms=10,
        rules_applied=0,
        validation_config={"gating": {}},
        quality_gate_status="pass",
        quality_gate_reasons=["Quality gate passed"],
        validation_timestamp=datetime.utcnow(),
    )


def test_auto_approval_requires_gate_pass(quality_service, base_quality_results):
    results = base_quality_results.model_copy(update={"quality_gate_status": "warn"})
    eligibility = quality_service.calculate_auto_approval_eligibility(results)

    assert eligibility["eligible"] is False
    assert "Quality gate status is fail" not in eligibility["reasons"]
    assert any("warnings" in reason.lower() for reason in eligibility["reasons"])


def test_auto_approval_passes_for_gate_pass(quality_service, base_quality_results):
    eligibility = quality_service.calculate_auto_approval_eligibility(base_quality_results)

    assert eligibility["eligible"] is True
    assert not eligibility["reasons"]
