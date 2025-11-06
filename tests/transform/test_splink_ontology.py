import pytest
from types import SimpleNamespace

from app.services.transform.helpers import (
    deduplicate_entities_with_splink,
    _prepare_entities_for_deduplication,
    _create_splink_dataframe,
    _create_splink_comparisons,
    _base_property_from_column,
)
from app.utils.constants import SYSTEM_PROPERTIES
from app.services.transform.models import BaseNode


@pytest.fixture(autouse=True)
def stub_splink(monkeypatch):
    import app.services.transform.helpers as helpers

    class DummyComparison:
        def __init__(self, column: str, kind: str, levels: int):
            self.column = column
            self.kind = kind
            self._levels = levels
            self.m_probabilities = None
            self.u_probabilities = None
            self.col_expressions = {
                "default": SimpleNamespace(raw_sql_expression=column)
            }

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
        JaroWinklerAtThresholds=lambda col, thresholds: DummyComparison(col, "jaro", 4),
        LevenshteinAtThresholds=lambda col, thresholds: DummyComparison(col, "lev", 3),
    )

    def fake_block_on(column):
        return SimpleNamespace(blocking_rule_sql=str(column))

    monkeypatch.setattr(helpers, "cl", comparison_module, raising=False)
    monkeypatch.setattr(helpers, "block_on", fake_block_on, raising=False)


def build_node(
    *,
    node_id: str,
    entity_type: str,
    properties: dict,
    canonical_properties: dict,
    canonical_id: str | None = None,
):
    return BaseNode(
        id=node_id,
        type=entity_type,
        properties=properties,
        canonical_properties=canonical_properties,
        canonical_id=canonical_id,
    )


@pytest.mark.asyncio
async def test_small_group_dedupes_on_unique_canonical_value():
    ontology = {
        "entities": {
            "EntityAlpha": {
                "properties": {
                    "primary_identifier": {
                        "type": "string",
                        "unique": True,
                    },
                    "descriptor": {
                        "type": "string",
                    },
                }
            }
        }
    }

    first = build_node(
        node_id="alpha-1",
        entity_type="EntityAlpha",
        properties={
            "primary_identifier": "ID-001 ",
            "descriptor": "Original descriptor",
        },
        canonical_properties={
            "primary_identifier": "id-001",
            "descriptor": "original descriptor",
        },
        canonical_id="entityalpha:id-001",
    )

    duplicate = build_node(
        node_id="alpha-2",
        entity_type="EntityAlpha",
        properties={
            "primary_identifier": "id-001",
            "descriptor": "Descriptor variant",
        },
        canonical_properties={
            "primary_identifier": "id-001",
            "descriptor": "descriptor variant",
        },
    )

    deduped, _ = await deduplicate_entities_with_splink(
        [first, duplicate],
        parsed_ontology=ontology,
    )

    assert len(deduped) == 1
    assert deduped[0].id == "alpha-1"


@pytest.mark.asyncio
async def test_small_group_keeps_distinct_unique_values():
    ontology = {
        "entities": {
            "EntityAlpha": {
                "properties": {
                    "primary_identifier": {
                        "type": "string",
                        "unique": True,
                    }
                }
            }
        }
    }

    first = build_node(
        node_id="alpha-10",
        entity_type="EntityAlpha",
        properties={"primary_identifier": "ID-010"},
        canonical_properties={"primary_identifier": "id-010"},
    )

    second = build_node(
        node_id="alpha-11",
        entity_type="EntityAlpha",
        properties={"primary_identifier": "ID-011"},
        canonical_properties={"primary_identifier": "id-011"},
    )

    deduped, _ = await deduplicate_entities_with_splink(
        [first, second],
        parsed_ontology=ontology,
    )

    assert len(deduped) == 2
    assert {node.id for node in deduped} == {"alpha-10", "alpha-11"}


@pytest.mark.asyncio
async def test_indexed_property_guides_splink_blocking():
    ontology = {
        "entities": {
            "EntityBeta": {
                "properties": {
                    "tracking_label": {
                        "type": "string",
                        "index": True,
                    },
                    "status": {
                        "type": "string",
                    },
                }
            }
        }
    }

    nodes = [
        build_node(
            node_id="beta-1",
            entity_type="EntityBeta",
            properties={
                "tracking_label": "Segment-A",
                "status": "Active",
            },
            canonical_properties={
                "tracking_label": "segment-a",
                "status": "active",
            },
        ),
        build_node(
            node_id="beta-2",
            entity_type="EntityBeta",
            properties={
                "tracking_label": "segment-a",
                "status": "Active",
            },
            canonical_properties={
                "tracking_label": "segment-a",
                "status": "active",
            },
        ),
        build_node(
            node_id="beta-3",
            entity_type="EntityBeta",
            properties={
                "tracking_label": "Segment-B",
                "status": "Dormant",
            },
            canonical_properties={
                "tracking_label": "segment-b",
                "status": "dormant",
            },
        ),
        build_node(
            node_id="beta-4",
            entity_type="EntityBeta",
            properties={
                "tracking_label": "Segment-C",
                "status": "Active",
            },
            canonical_properties={
                "tracking_label": "segment-c",
                "status": "active",
            },
        ),
    ]

    deduped, _ = await deduplicate_entities_with_splink(
        nodes,
        parsed_ontology=ontology,
    )

    assert len(deduped) == 3
    assert any(node.id == "beta-1" for node in deduped)
    assert not any(node.id == "beta-2" for node in deduped)


def _allowed_properties_for(ontology, entity_type: str) -> set[str]:
    return set(
        ontology["entities"].get(entity_type, {}).get("properties", {}).keys()
    )


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


def _prepare_df(nodes, ontology, entity_type: str):
    entities = _prepare_entities_for_deduplication(nodes, parsed_ontology=ontology)
    allowed = _allowed_properties_for(ontology, entity_type)
    df, columns = _create_splink_dataframe(
        entities,
        SYSTEM_PROPERTIES,
        allowed_properties=allowed if allowed else None,
    )
    return df, columns, _row_count(df)


def test_dataframe_uses_only_ontology_defined_properties():
    ontology = {
        "EntitySigma": {
            "properties": {
                "uid": {"type": "string", "unique": True},
                "label": {"type": "string"},
            }
        }
    }

    nodes = [
        build_node(
            node_id="sigma-1",
            entity_type="EntitySigma",
            properties={
                "uid": "ABC123",
                "label": "Primary",
                "adhoc": "should be ignored",
            },
            canonical_properties={
                "uid": "abc123",
                "label": "primary",
                "adhoc": "should be ignored",
            },
        )
    ]

    df, columns, record_count = _prepare_df(nodes, {"entities": ontology}, "EntitySigma")

    allowed = _allowed_properties_for({"entities": ontology}, "EntitySigma")
    assert allowed == {"uid", "label"}
    assert all(
        _base_property_from_column(column) in allowed for column in columns
    ), columns


def test_comparisons_apply_priors_from_ontology():
    ontology = {
        "entities": {
            "EntityTheta": {
                "properties": {
                    "code": {"type": "string", "unique": True},
                    "description": {"type": "string"},
                }
            }
        }
    }

    nodes = [
        build_node(
            node_id=f"theta-{idx}",
            entity_type="EntityTheta",
            properties={
                "code": f"ID-{idx:03d}",
                "description": f"Description {idx}",
            },
            canonical_properties={
                "code": f"id-{idx:03d}",
                "description": f"description {idx}",
            },
        )
        for idx in range(1, 9)
    ]

    df, columns, record_count = _prepare_df(nodes, ontology, "EntityTheta")

    comparisons = _create_splink_comparisons(
        columns,
        df,
        record_count,
        "EntityTheta",
        ontology,
    )

    unique_matches = [
        comp
        for comp in comparisons
        if comp.kind == "exact" and "code" in comp.column
    ]

    assert unique_matches, "Expected unique comparison for canonical code"
    for comp in unique_matches:
        assert comp.m_probabilities == [0.97, 0.03]
        assert comp.u_probabilities == [0.02, 0.98]

    string_matches = [comp for comp in comparisons if comp.kind == "jaro"]
    assert string_matches, "Expected fuzzy comparison for description"
    for comp in string_matches:
        assert comp.num_non_null_levels == len(comp.m_probabilities)
        assert comp.m_probabilities[:2] == [0.85, 0.1]
        assert comp.u_probabilities[:2] == [0.05, 0.1]


def test_small_groups_skip_string_comparisons_when_unique_available():
    ontology = {
        "entities": {
            "EntityKappa": {
                "properties": {
                    "identifier": {"type": "string", "unique": True},
                    "notes": {"type": "string"},
                }
            }
        }
    }

    nodes = [
        build_node(
            node_id=f"kappa-{idx}",
            entity_type="EntityKappa",
            properties={
                "identifier": f"PK-{idx if idx < 3 else 99}",
                "notes": f"Variant {idx}",
            },
            canonical_properties={
                "identifier": f"pk-{idx if idx < 3 else 99}",
                "notes": f"variant {idx}",
            },
        )
        for idx in range(1, 4)
    ]

    df, columns, record_count = _prepare_df(nodes, ontology, "EntityKappa")

    comparisons = _create_splink_comparisons(
        columns,
        df,
        record_count,
        "EntityKappa",
        ontology,
    )

    assert any(comp.kind == "exact" for comp in comparisons)
    string_matches = [comp for comp in comparisons if comp.kind == "jaro"]
    assert string_matches, "Expected fallback fuzzy comparison even for small sets"
    for comp in string_matches:
        assert comp.m_probabilities[:2] == [0.85, 0.1]
        assert comp.u_probabilities[:2] == [0.05, 0.1]
