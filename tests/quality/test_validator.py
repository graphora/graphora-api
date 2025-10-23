import pytest

from app.services.quality.models import QualityRuleType, QualitySeverity
from app.services.quality.validator import QualityValidator
from app.services.transform.models import BaseNode, DocumentKnowledgeGraph


@pytest.mark.asyncio
async def test_quality_validator_captures_entity_violations():
    ontology = {
        "version": "1.0.0",
        "entities": {
            "Company": {
                "properties": {
                    "name": {
                        "required": True,
                        "quality": {
                            "format": {"caseFormat": "titleCase"},
                        },
                    },
                    "ticker": {
                        "quality": {
                            "format": {"pattern": "^[A-Z]{3}$"},
                            "business": {"forbiddenValues": ["N/A"]},
                        }
                    },
                }
            }
        },
    }

    validator = QualityValidator(ontology)

    valid_company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "Acme Corp", "ticker": "ACM"},
        confidence_score=0.95,
    )
    invalid_company = BaseNode(
        id="company-2",
        type="Company",
        properties={"name": "", "ticker": "N/A"},
        confidence_score=0.4,
    )

    knowledge_graph = DocumentKnowledgeGraph(nodes=[valid_company, invalid_company])

    results = await validator.validate_extraction(knowledge_graph, "transform-xyz")

    violation_ids = {violation.rule_id for violation in results.violations}

    assert "Company.name.required" in violation_ids
    assert "Company.ticker.forbidden_values" in violation_ids
    assert "Company.ticker.pattern" in violation_ids
    assert QualityRuleType.BUSINESS in results.violations_by_type
    assert QualitySeverity.ERROR in results.violations_by_severity
    assert results.violations_by_severity[QualitySeverity.WARNING] >= 1
    assert results.metrics.entities_with_violations == 1
    assert results.requires_review is True
    assert results.metrics.property_completeness_rate == pytest.approx(50.0)
