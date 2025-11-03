import pytest
from types import SimpleNamespace

from app.services.transform.models import BaseNode
from app.services.transform.helpers import (
    _build_canonical_properties,
    _prepare_entities_for_deduplication,
    _create_splink_dataframe,
    _create_splink_comparisons,
    _create_blocking_rules,
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

    comparisons = _create_splink_comparisons(columns, df, "Company", ontology)
    assert comparisons
    assert comparisons[0].column == "ticker"

    blocking_rules = _create_blocking_rules(columns, df, "Company", ontology)
    assert any(rule.blocking_rule_sql == "ticker" for rule in blocking_rules)
