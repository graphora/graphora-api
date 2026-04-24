import pytest
from types import SimpleNamespace

from graphora_server.services.transform.models import BaseNode
from graphora_server.services.transform.helpers import (
    _build_canonical_properties,
    _prepare_entities_for_deduplication,
    _create_splink_dataframe,
    _create_splink_comparisons,
    _create_blocking_rules,
    transform_as_relationships,
)
from graphora_server.utils.constants import SYSTEM_PROPERTIES
from graphora_server.config import settings


@pytest.fixture(autouse=True)
def stub_splink(monkeypatch):
    import graphora_server.services.transform.helpers as helpers

    class DummyComparison:
        def __init__(self, column: str, kind: str, levels: int):
            self.column = column
            self.kind = kind
            self._levels = levels
            self.m_probabilities = None
            self.u_probabilities = None

        @property
        def num_non_null_levels(self) -> int:
            return self._levels

        def configure(self, *, m_probabilities=None, u_probabilities=None):
            if m_probabilities is not None:
                self.m_probabilities = m_probabilities
            if u_probabilities is not None:
                self.u_probabilities = u_probabilities
            return self

    comparison_module = SimpleNamespace(
        ExactMatch=lambda col: DummyComparison(col, "exact", 2),
        JaroWinklerAtThresholds=lambda col, _thresholds: DummyComparison(
            col, "jaro", 4
        ),
        LevenshteinAtThresholds=lambda col, _thresholds: DummyComparison(col, "lev", 3),
    )

    def fake_block_on(column):
        return SimpleNamespace(blocking_rule_sql=str(column))

    monkeypatch.setattr(helpers, "cl", comparison_module, raising=False)
    monkeypatch.setattr(helpers, "block_on", fake_block_on, raising=False)


def _row_count(df) -> int:
    shape = getattr(df, "shape", None)
    if shape is not None:
        try:
            return int(shape[0])
        except Exception:
            pass

    try:
        return len(df)
    except Exception:
        pass

    index = getattr(df, "index", None)
    if index is not None:
        try:
            return len(index)
        except Exception:
            pass

    rows = getattr(df, "rows", None)
    if rows is not None:
        return len(rows)

    internal_rows = getattr(df, "_rows", None)
    if internal_rows is not None:
        return len(internal_rows)

    raise AssertionError("Unable to determine row count for dataframe stub")


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


def test_canonicalization_respects_setting(monkeypatch):
    ontology = {
        "entities": {
            "Company": {
                "properties": {
                    "name": {
                        "type": "string",
                        "canonicalization": {
                            "strip_suffixes": ["Inc", "LLC"],
                            "strip_punctuation": True,
                        },
                    }
                }
            }
        }
    }

    raw = {"name": "Acme, Inc."}

    monkeypatch.setattr(settings, "ENTITY_CANONICALIZATION_ENABLED", False)
    canonical = _build_canonical_properties(ontology, "Company", raw, raw)

    assert canonical["name"] == "acme, inc."


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
    allowed = set(ontology["entities"]["Company"]["properties"].keys())
    df, columns = _create_splink_dataframe(
        entities,
        SYSTEM_PROPERTIES,
        allowed_properties=allowed,
    )

    assert df.loc[0, "name"] == "acme"
    assert df.loc[0, "ticker"] == "acm"
    assert df.loc[0, "canonical__name"] == "acme"
    assert df.loc[0, "canonical__ticker"] == "acm"

    comparisons, text_columns = _create_splink_comparisons(
        columns,
        df,
        _row_count(df),
        "Company",
        ontology,
    )
    assert comparisons
    assert comparisons[0].column == "canonical__ticker"

    blocking_rules = _create_blocking_rules(
        columns,
        df,
        _row_count(df),
        "Company",
        ontology,
    )
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
    allowed = set(ontology["entities"]["Organisation"]["properties"].keys())
    df, columns = _create_splink_dataframe(
        entities,
        SYSTEM_PROPERTIES,
        allowed_properties=allowed,
    )

    blocking_rules = _create_blocking_rules(
        columns,
        df,
        _row_count(df),
        "Organisation",
        ontology,
    )
    rule_columns = {rule.blocking_rule_sql for rule in blocking_rules}
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
        ],
    )

    relationships = transform_as_relationships(
        ontology,
        [company_node, product_node],
        relationship_result,
    )

    assert (
        relationships
    ), "Relationship should be created when canonical IDs are provided"
    assert relationships[0].source_id == "node-company"
    assert relationships[0].target_id == "node-product"
