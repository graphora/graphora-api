import pytest
from types import SimpleNamespace

from app.services.transform.models import BaseNode
from app.services.transform.helpers import (
    _build_canonical_properties,
    _prepare_entities_for_deduplication,
    _create_splink_dataframe,
    _create_splink_comparisons,
    _create_blocking_rules,
    transform_as_relationships,
)
from app.utils.constants import SYSTEM_PROPERTIES


@pytest.fixture(autouse=True)
def stub_splink(monkeypatch):
    from app.services import transform as transform_pkg  # noqa: F401 to ensure package loaded
    import app.services.transform.helpers as helpers

    class DummyComparison:
        def __init__(self, column: str, kind: str):
            self.column = column
            self.kind = kind

    comparison_module = SimpleNamespace(
        ExactMatch=lambda col: DummyComparison(col, "exact"),
        JaroWinklerAtThresholds=lambda col, thresholds: DummyComparison(col, "jaro"),
        LevenshteinAtThresholds=lambda col, thresholds: DummyComparison(col, "lev"),
    )

    def fake_block_on(column):
        return SimpleNamespace(blocking_rule_sql=str(column))

    monkeypatch.setattr(helpers, "cl", comparison_module, raising=False)
    monkeypatch.setattr(helpers, "block_on", fake_block_on, raising=False)


def test_canonical_properties_normalize_names():
    ontology = {
        "entities": {
            "Company": {
                "properties": {
                    "name": {
                        "type": "string",
                        "unique": True,
                        "canonicalization": {
                            "strip_suffixes": ["Inc", "LLC"],
                            "strip_punctuation": True,
                        },
                    },
                    "ticker": {"type": "string", "unique": True},
                }
            }
        }
    }

    raw = {"name": "Acme, Inc.", "ticker": "ACM"}
    canonical = _build_canonical_properties(ontology, "Company", raw, raw)

    assert canonical["name"] == "acme"
    assert canonical["ticker"] == "acm"


def test_default_canonicalization_lowercases_without_suffix_rules():
    ontology = {
        "entities": {
            "Person": {
                "properties": {
                    "first_name": {"type": "string"},
                    "age": {"type": "integer"},
                }
            }
        }
    }

    raw = {"first_name": " Alice  ", "age": 42}
    canonical = _build_canonical_properties(ontology, "Person", raw, raw)

    assert canonical["first_name"] == "alice"
    assert canonical["age"] == "42"


def test_splink_uses_canonical_and_unique_first():
    ontology = {
        "entities": {
            "Company": {
                "properties": {
                    "name": {
                        "type": "string",
                        "canonicalization": {
                            "strip_suffixes": ["Inc"],
                            "strip_punctuation": True,
                        },
                    },
                    "ticker": {"type": "string", "unique": True},
                }
            }
        }
    }

    raw = {"name": "Acme, Inc.", "ticker": "ACM"}
    canonical = _build_canonical_properties(ontology, "Company", raw, raw)
    node = BaseNode(
        id="1",
        type="Company",
        properties=raw,
        canonical_properties=canonical,
    )

    entities = _prepare_entities_for_deduplication([node], parsed_ontology=ontology)
    df, columns = _create_splink_dataframe(entities, SYSTEM_PROPERTIES)

    assert df.loc[0, "name"] == "acme"
    assert df.loc[0, "ticker"] == "acm"
    assert df.loc[0, "canonical__name"] == "acme"
    assert df.loc[0, "canonical__ticker"] == "acm"

    comparisons = _create_splink_comparisons(columns, df, "Company", ontology)
    assert comparisons
    assert comparisons[0].column == "canonical__ticker"

    blocking_rules = _create_blocking_rules(columns, df, "Company", ontology)
    assert any(rule.blocking_rule_sql == "canonical__ticker" for rule in blocking_rules)


def test_blocking_prefers_canonical_name_columns():
    ontology = {
        "entities": {
            "Organisation": {
                "properties": {
                    "name": {
                        "type": "string",
                        "canonicalization": {
                            "strip_company_suffixes": True,
                            "strip_punctuation": True,
                        },
                    },
                    "registration_code": {
                        "type": "string",
                        "unique": True,
                        "canonicalization": {"remove_non_alnum": True},
                    },
                }
            }
        }
    }

    raw = {"name": "Acme Limited", "registration_code": " 123-456 "}
    canonical = _build_canonical_properties(ontology, "Organisation", raw, raw)
    node = BaseNode(
        id="1",
        type="Organisation",
        properties=raw,
        canonical_properties=canonical,
    )

    entities = _prepare_entities_for_deduplication([node], parsed_ontology=ontology)
    df, columns = _create_splink_dataframe(entities, SYSTEM_PROPERTIES)

    blocking_rules = _create_blocking_rules(columns, df, "Organisation", ontology)
    rule_columns = {rule.blocking_rule_sql for rule in blocking_rules}
    assert "canonical__name" in rule_columns
    assert "canonical__registration_code" in rule_columns


def test_relationships_resolve_canonical_identifiers():
    ontology = {
        "entities": {
            "Company": {
                "properties": {
                    "name": {
                        "type": "string",
                        "canonicalization": {
                            "strip_company_suffixes": True,
                            "strip_punctuation": True,
                        },
                    }
                },
                "relationships": {
                    "HAS_PRODUCT": {
                        "target": "Product",
                        "properties": {},
                    }
                },
            },
            "Product": {
                "properties": {
                    "name": {"type": "string"},
                }
            },
        }
    }

    company_node = BaseNode(
        id="node-company",
        type="Company",
        properties={"name": "Acme Inc."},
        canonical_properties={"name": "acme"},
        canonical_key="Company:name=acme",
        canonical_id="canon-company",
    )

    product_node = BaseNode(
        id="node-product",
        type="Product",
        properties={"name": "Gizmo"},
        canonical_properties={"name": "gizmo"},
        canonical_key="Product:name=gizmo",
        canonical_id="canon-product",
    )

    relationship_result = SimpleNamespace(
        confidence_score=0.85,
        Company_HAS_PRODUCT_Product=[
            SimpleNamespace(
                source_id="canon-company",
                target_id="canon-product",
                properties=None,
                confidence_score=0.9,
            )
        ]
    )

    relationships = transform_as_relationships(
        ontology,
        [company_node, product_node],
        relationship_result,
    )

    assert relationships, "Relationship should be created when canonical IDs are provided"
    assert relationships[0].source_id == "node-company"
    assert relationships[0].target_id == "node-product"
