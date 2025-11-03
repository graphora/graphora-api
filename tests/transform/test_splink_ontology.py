import pytest

from app.services.transform.helpers import deduplicate_entities_with_splink
from app.services.transform.models import BaseNode


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
