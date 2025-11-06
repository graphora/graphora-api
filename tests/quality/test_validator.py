from typing import List

import pytest

from app.services.quality.models import QualityRuleType, QualitySeverity
from app.services.quality.validator import QualityValidator
from app.services.transform.models import (
    BaseNode,
    DocumentKnowledgeGraph,
    RelationshipInstance,
)


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
                    "name": {"quality": {"format": {"pattern": "^[A-Z][a-z]+"}}}
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
                "properties": {"name": {"required": True}},
                "relationships": {"HAS_PRODUCT": {"target": "Product"}},
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
async def test_quality_validator_pass_gate():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "gating": {
                "passScore": 90,
                "warnScore": 80,
                "hardFailScore": 70,
                "maxWarnings": 1,
            }
        },
        "entities": {
            "Company": {
                "properties": {
                    "name": {"required": True},
                }
            }
        },
    }

    validator = QualityValidator(ontology)

    company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "Acme"},
        confidence_score=0.99,
    )

    graph = DocumentKnowledgeGraph(nodes=[company])
    results = await validator.validate_extraction(graph, "transform-pass")

    assert results.quality_gate_status == "pass"
    assert results.requires_review is False
    assert not results.violations


@pytest.mark.asyncio
async def test_entity_completeness_rule_detects_low_fill_rate():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "gating": {
                "hardFailScore": 85,
                "warnScore": 90,
                "maxWarnings": 5,
            }
        },
        "entities": {
            "Company": {
                "properties": {
                    "name": {"required": True},
                    "ticker": {"type": "string"},
                },
                "quality": {
                    "entityLevel": {
                        "completeness": {
                            "minPropertyFillRate": 0.75,
                            "severity": "error",
                        }
                    }
                },
            }
        },
    }

    validator = QualityValidator(ontology)

    company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "Acme"},
        confidence_score=0.6,
    )

    graph = DocumentKnowledgeGraph(nodes=[company])
    results = await validator.validate_extraction(graph, "transform-completeness")

    assert any(
        violation.rule_id == "Company.completeness.min_fill_rate"
        for violation in results.violations
    )
    assert results.quality_gate_status == "fail"


@pytest.mark.asyncio
async def test_symmetric_relationship_rule_detects_missing_inverse():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "global": {},
            "gating": {
                "passScore": 95,
                "warnScore": 85,
                "hardFailScore": 75,
            },
            "crossValidationRules": [
                {
                    "name": "company_industry_symmetry",
                    "ruleType": "symmetric_relationship",
                    "relationshipType": "ASSOCIATED_WITH",
                    "severity": "warning",
                }
            ],
        },
        "entities": {
            "Company": {
                "properties": {"name": {"required": True}},
                "relationships": {"ASSOCIATED_WITH": {"target": "Industry"}},
            },
            "Industry": {"properties": {"name": {"required": True}}},
        },
    }

    validator = QualityValidator(ontology)

    company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "Acme"},
        confidence_score=0.8,
    )
    industry = BaseNode(
        id="industry-1",
        type="Industry",
        properties={"name": "Tech"},
        confidence_score=0.8,
    )

    relationships = []
    relationships.append(
        RelationshipInstance(
            id="rel-1",
            type="ASSOCIATED_WITH",
            source_id=company.id,
            target_id=industry.id,
            source_type="Company",
            target_type="Industry",
        )
    )

    graph = DocumentKnowledgeGraph(
        nodes=[company, industry], relationships=relationships
    )
    results = await validator.validate_extraction(graph, "transform-symmetric")

    assert any(
        violation.rule_id.startswith("global.symmetric_relationship")
        for violation in results.violations
    )


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


@pytest.mark.asyncio
async def test_cross_entity_consistency_rule_detects_mismatch():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "crossValidationRules": [
                {
                    "ruleType": "property_alignment",
                    "relationshipType": "CLASSIFIED_AS",
                    "sourceProperty": "classificationStandard",
                    "targetProperty": "classification",
                    "severity": "error",
                }
            ]
        },
        "entities": {
            "Company": {
                "properties": {"classificationStandard": {"type": "string"}},
                "relationships": {"CLASSIFIED_AS": {"target": "Industry"}},
            },
            "Industry": {"properties": {"classification": {"type": "string"}}},
        },
    }

    validator = QualityValidator(ontology)

    company = BaseNode(
        id="company-1",
        type="Company",
        properties={"classificationStandard": "GICS"},
    )
    industry = BaseNode(
        id="industry-1",
        type="Industry",
        properties={"classification": "NAICS"},
    )
    relationship = RelationshipInstance(
        id="rel-1",
        type="CLASSIFIED_AS",
        source_id=company.id,
        source_type="Company",
        target_id=industry.id,
        target_type="Industry",
        properties={},
    )

    knowledge_graph = DocumentKnowledgeGraph(
        nodes=[company, industry], relationships=[relationship]
    )

    results = await validator.validate_extraction(
        knowledge_graph, "transform-consistency"
    )

    assert any(
        violation.rule_id.startswith("global.property_alignment")
        for violation in results.violations
    )
    assert results.violations_by_severity[QualitySeverity.ERROR] >= 1


@pytest.mark.asyncio
async def test_property_coverage_rule_flags_low_fill_rate():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "coverageRules": [
                {
                    "entityType": "Company",
                    "property": "description",
                    "minCoverage": 0.75,
                    "severity": "warning",
                }
            ]
        },
        "entities": {
            "Company": {
                "properties": {
                    "description": {"type": "string"},
                }
            }
        },
    }

    validator = QualityValidator(ontology)

    nodes = [
        BaseNode(id=f"company-{idx}", type="Company", properties={}) for idx in range(3)
    ]
    nodes.append(
        BaseNode(
            id="company-4",
            type="Company",
            properties={"description": "A fully described company."},
        )
    )

    knowledge_graph = DocumentKnowledgeGraph(nodes=nodes, relationships=[])

    results = await validator.validate_extraction(knowledge_graph, "transform-coverage")

    assert any(
        violation.rule_id.startswith("global.property_coverage")
        for violation in results.violations
    )


@pytest.mark.asyncio
async def test_global_date_window_rule_detects_out_of_range():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "temporalRules": [
                {
                    "entityType": "Industry",
                    "property": "effectiveDate",
                    "earliest": "2020-01-01",
                    "latest": "2024-12-31",
                    "allowFuture": False,
                    "allowMissing": False,
                }
            ]
        },
        "entities": {"Industry": {"properties": {"effectiveDate": {"type": "string"}}}},
    }

    validator = QualityValidator(ontology)

    industry_ok = BaseNode(
        id="industry-1",
        type="Industry",
        properties={"effectiveDate": "2021-06-30"},
    )
    industry_bad = BaseNode(
        id="industry-2",
        type="Industry",
        properties={"effectiveDate": "2010-01-01"},
    )

    knowledge_graph = DocumentKnowledgeGraph(
        nodes=[industry_ok, industry_bad], relationships=[]
    )

    results = await validator.validate_extraction(knowledge_graph, "transform-temporal")

    assert any(
        violation.rule_id.startswith("global.temporal")
        for violation in results.violations
    )


@pytest.mark.asyncio
async def test_confidence_threshold_rule_flags_low_confidence():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "extractionStandards": {
                "confidenceThresholds": {
                    "entityExtraction": 0.8,
                    "relationshipExtraction": 0.7,
                }
            }
        },
        "entities": {
            "Company": {
                "properties": {"name": {"required": True}},
                "relationships": {"HAS_SUBSIDIARY": {"target": "Company"}},
            }
        },
    }

    validator = QualityValidator(ontology)

    strong_company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "Acme"},
        confidence_score=0.95,
    )
    weak_company = BaseNode(
        id="company-2",
        type="Company",
        properties={"name": "Subsidiary"},
        confidence_score=0.5,
    )

    low_conf_relationship = RelationshipInstance(
        id="rel-1",
        type="HAS_SUBSIDIARY",
        source_id=strong_company.id,
        target_id=weak_company.id,
        source_type="Company",
        target_type="Company",
        confidence_score=0.4,
    )

    knowledge_graph = DocumentKnowledgeGraph(
        nodes=[strong_company, weak_company],
        relationships=[low_conf_relationship],
    )

    results = await validator.validate_extraction(
        knowledge_graph, "transform-confidence"
    )

    confidence_violations = [
        violation
        for violation in results.violations
        if violation.rule_id == "global.confidence_threshold"
    ]

    assert confidence_violations, "Expected confidence threshold violations"
    assert QualityRuleType.DISTRIBUTION in results.violations_by_type
    assert results.violations_by_severity[QualitySeverity.ERROR] >= 1


@pytest.mark.asyncio
async def test_minimum_entities_rule_enforces_threshold():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "extractionStandards": {
                "completenessRequirements": {
                    "minEntitiesPerDocument": 3,
                    "severity": "error",
                }
            }
        },
        "entities": {"Company": {"properties": {"name": {"required": True}}}},
    }

    validator = QualityValidator(ontology)

    company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "Acme"},
        confidence_score=0.9,
    )

    knowledge_graph = DocumentKnowledgeGraph(nodes=[company], relationships=[])

    results = await validator.validate_extraction(
        knowledge_graph, "transform-min-entities"
    )

    assert any(
        violation.rule_id == "global.completeness.min_entities"
        for violation in results.violations
    )
    assert results.violations_by_type[QualityRuleType.CROSS_ENTITY] >= 1


@pytest.mark.asyncio
async def test_required_entity_types_rule_detects_missing_types():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "extractionStandards": {
                "completenessRequirements": {
                    "requiredEntityTypes": ["Company", "Industry"],
                    "severity": "error",
                }
            }
        },
        "entities": {
            "Company": {"properties": {"name": {"required": True}}},
            "Industry": {"properties": {"name": {"required": True}}},
        },
    }

    validator = QualityValidator(ontology)

    company = BaseNode(
        id="company-1",
        type="Company",
        properties={"name": "Acme"},
        confidence_score=0.9,
    )

    knowledge_graph = DocumentKnowledgeGraph(nodes=[company], relationships=[])

    results = await validator.validate_extraction(
        knowledge_graph, "transform-required-types"
    )

    assert any(
        violation.rule_id == "global.completeness.required_entity_types"
        for violation in results.violations
    )
    assert results.requires_review is True


@pytest.mark.asyncio
async def test_entity_balance_rule_enforces_expected_ratios():
    ontology = {
        "version": "1.0.0",
        "dataQualityConfig": {
            "distributionRules": {
                "entityBalance": {
                    "maxSingleTypeRatio": 0.6,
                    "expectedRatios": {
                        "Company": {"min": 0.3, "max": 0.6},
                        "Industry": {"min": 0.2},
                    },
                }
            }
        },
        "entities": {
            "Company": {"properties": {"name": {"required": True}}},
            "Industry": {"properties": {"name": {"required": True}}},
        },
    }

    validator = QualityValidator(ontology)

    nodes: List[BaseNode] = []
    for idx in range(8):
        nodes.append(
            BaseNode(
                id=f"company-{idx}",
                type="Company",
                properties={"name": f"Company {idx}"},
                confidence_score=0.9,
            )
        )

    nodes.append(
        BaseNode(
            id="industry-1",
            type="Industry",
            properties={"name": "Tech"},
            confidence_score=0.9,
        )
    )

    knowledge_graph = DocumentKnowledgeGraph(nodes=nodes, relationships=[])

    results = await validator.validate_extraction(
        knowledge_graph, "transform-entity-balance"
    )

    assert any(
        violation.rule_id == "global.entity_balance" for violation in results.violations
    )
    assert results.violations_by_type[QualityRuleType.DISTRIBUTION] >= 1
