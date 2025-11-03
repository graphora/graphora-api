from datetime import datetime

import pytest

from app.services.quality.models import (
    QualityMetrics,
    QualityResults,
    QualitySeverity,
    QualityRuleType,
    QualityViolation,
)
from app.services.quality.service import QualityService


class DummyNeo4jStorage:
    def __init__(self):
        self.queries = []

    async def execute_query(self, query, params=None):
        self.queries.append({"query": query, "params": params})
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


@pytest.mark.asyncio
async def test_store_quality_results_persists_violations(
    quality_service, base_quality_results
):
    violation = QualityViolation(
        rule_id="Company.name.required",
        rule_type=QualityRuleType.BUSINESS,
        severity=QualitySeverity.ERROR,
        entity_type="Company",
        entity_id="company-1",
        property_name="name",
        message="Company name is required",
        expected="Non-empty name",
        actual="",
        confidence=0.5,
        suggestion="Provide the company name",
    )

    results = base_quality_results.model_copy(
        update={
            "violations": [violation],
            "metrics": base_quality_results.metrics.model_copy(update={"total_violations": 1}),
        }
    )

    await quality_service.store_quality_results("transform-123", results, "user-42")

    queries = quality_service.neo4j.queries

    assert len(queries) == 3
    assert "MERGE (qr:QualityResults" in queries[0]["query"]
    assert "DETACH DELETE v" in queries[1]["query"]
    assert "UNWIND violations AS violation" in queries[2]["query"]

    payload = queries[2]["params"]["violations"]
    assert len(payload) == 1
    stored_violation = payload[0]
    assert stored_violation["rule_id"] == "Company.name.required"
    assert stored_violation["severity"] == QualitySeverity.ERROR.value
    assert stored_violation["entity_id"] == "company-1"
    assert stored_violation["violation_index"] == 0
    assert stored_violation["violation_id"]


@pytest.mark.asyncio
async def test_store_quality_results_clears_existing_violations(
    quality_service, base_quality_results
):
    await quality_service.store_quality_results("transform-xyz", base_quality_results, "user-1")

    queries = quality_service.neo4j.queries
    assert len(queries) == 2
    assert "DETACH DELETE v" in queries[1]["query"]


@pytest.mark.asyncio
async def test_get_violations_filters_and_pagination(
    quality_service, base_quality_results
):
    violation_error = QualityViolation(
        rule_id="Company.missing_name",
        rule_type=QualityRuleType.BUSINESS,
        severity=QualitySeverity.ERROR,
        entity_type="Company",
        entity_id="company-1",
        property_name="name",
        message="Company name is required",
        expected="Non-empty name",
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
        message="Description is shorter than recommended",
        expected="At least 20 characters",
        actual="Short",
        confidence=0.7,
    )

    violation_other = QualityViolation(
        rule_id="Product.sku.unique",
        rule_type=QualityRuleType.BUSINESS,
        severity=QualitySeverity.ERROR,
        entity_type="Product",
        entity_id="product-1",
        property_name="sku",
        message="Duplicate SKU detected",
        expected="Unique SKU",
        actual="SKU-123",
        confidence=0.9,
    )

    rich_results = base_quality_results.model_copy(
        update={
            "violations": [violation_error, violation_warning, violation_other],
            "metrics": base_quality_results.metrics.model_copy(
                update={"total_violations": 3}
            ),
        }
    )

    filtered = await quality_service.get_violations(
        transform_id="transform-1",
        user_id="user-1",
        violation_type=QualityRuleType.BUSINESS,
        severity=QualitySeverity.ERROR,
        entity_type="Company",
        quality_results=rich_results,
    )

    assert len(filtered) == 1
    assert filtered[0].rule_id == "Company.missing_name"

    paged = await quality_service.get_violations(
        transform_id="transform-1",
        user_id="user-1",
        limit=1,
        offset=1,
        quality_results=rich_results,
    )

    assert len(paged) == 1
    assert paged[0].rule_id == "Company.description.short"
