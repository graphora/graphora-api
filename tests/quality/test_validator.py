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
    assert results.quality_gate_status == "fail"


@pytest.mark.asyncio
async def test_quality_validator_warn_band_allows_retry():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "gating": {
                "passScore": 95,
                "warnScore": 90,
                "hardFailScore": 60,
                "maxWarnings": 0,
            }
        },
        "entities": {
            "Company": {
                "properties": {
                    "name": {
                        "quality": {
                            "format": {"pattern": "^[A-Z][a-z]+"}
                        }
                    }
                }
            }
        },
    }

    validator = QualityValidator(ontology)

    company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "acme"},
        confidence_score=0.95,
    )

    knowledge_graph = DocumentKnowledgeGraph(nodes=[company])

    results = await validator.validate_extraction(knowledge_graph, "transform-warn")

    assert results.quality_gate_status == "warn"
    assert results.requires_review is False
    assert results.quality_gate_reasons
    assert any("warnings" in reason.lower() for reason in results.quality_gate_reasons)


@pytest.mark.asyncio
async def test_relationship_presence_rule_creates_violation():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "distributionRules": {
                "relationshipRequirements": [
                    {
                        "entityType": "Company",
                        "relationshipType": "HAS_PRODUCT",
                        "direction": "outbound",
                        "minCount": 1,
                        "severity": "error",
                    }
                ]
            },
            "gating": {
                "hardFailScore": 90,
                "warnScore": 95,
                "maxErrors": 0,
            },
        },
        "entities": {
            "Company": {
                "properties": {
                    "name": {"required": True}
                },
                "relationships": {
                    "HAS_PRODUCT": {"target": "Product"}
                },
            },
            "Product": {"properties": {"name": {"required": True}}},
        },
    }

    validator = QualityValidator(ontology)

    company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "Acme"},
        confidence_score=0.9,
    )
    product = BaseNode(
        id="product-1",
        type="Product",
        properties={"name": "Widget"},
        confidence_score=0.9,
    )

    knowledge_graph = DocumentKnowledgeGraph(
        nodes=[company, product],
        relationships=[],
    )

    results = await validator.validate_extraction(knowledge_graph, "transform-rel")

    assert any(
        violation.rule_id.startswith("global.relationship_requirement")
        for violation in results.violations
    )
    assert results.quality_gate_status == "fail"
    assert results.requires_review is True


@pytest.mark.asyncio
async def test_date_window_rule_enforces_range():
    ontology = {
        "version": "1.0.0",
        "entities": {
            "Filing": {
                "properties": {
                    "fiscal_year": {
                        "quality": {
                            "validation": {
                                "dateWindow": {
                                    "earliest": "2010-01-01",
                                    "latest": "2020-12-31",
                                    "severity": "warning",
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    validator = QualityValidator(ontology)

    filing = BaseNode(
        id="filing-1",
        type="Filing",
        properties={"fiscal_year": "2023-05-01"},
        confidence_score=0.8,
    )

    knowledge_graph = DocumentKnowledgeGraph(nodes=[filing])

    results = await validator.validate_extraction(knowledge_graph, "transform-date")

    assert any(
        violation.rule_id == "Filing.fiscal_year.date_window"
        for violation in results.violations
    )
