"""End-to-end integration tests for the Apache AGE storage adapter.

Round-trips the adapter against a real ``apache/age`` Docker
container brought up by ``conftest.py::age_container``. Validates
that the Cypher patterns the unit tests pin actually work against
real AGE, not just our mock-shaped expectations.

Marked ``integration`` so the unit suite stays Docker-free; run
explicitly with::

    uv run pytest tests/integration/storage/ -m integration

Or skip integration in the default suite::

    uv run pytest -m "not integration"
"""

from __future__ import annotations

import pytest

from graphora_server.services.storage.models import StorageStage
from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_checkpoint_round_trip(age_storage):
    """update_checkpoint then get_storage_status returns the same
    transform_id / last_processed_index / stage. Tightest
    smoke-test of the cypher() helper + agtype parsing on real AGE."""
    transform_id = "round-trip-checkpoint"

    none_initial = await age_storage.get_storage_status(transform_id)
    assert none_initial is None

    result = await age_storage.update_checkpoint(
        transform_id, last_index=42, stage=StorageStage.NODES
    )
    assert result.success is True
    assert result.items_processed == 1

    checkpoint = await age_storage.get_storage_status(transform_id)
    assert checkpoint is not None
    assert checkpoint.transform_id == transform_id
    assert checkpoint.last_processed_index == 42
    assert checkpoint.stage == StorageStage.NODES


@pytest.mark.asyncio
async def test_store_and_read_nodes(age_storage):
    """Bulk-write nodes via UNWIND batch, read them back via
    get_transformation_data. Pins that the bucket-by-type write
    + transform_id-keyed read both work end-to-end."""
    transform_id = "round-trip-nodes"
    nodes = [
        BaseNode(type="Person", properties={"name": "Alice"}),
        BaseNode(type="Person", properties={"name": "Bob"}),
        BaseNode(type="Company", properties={"name": "Acme"}),
    ]

    result = await age_storage.store_nodes(
        nodes, batch_index=0, transform_id=transform_id
    )
    assert result.success is True
    assert result.items_processed == 3

    response = await age_storage.get_transformation_data(transform_id)
    assert response.total_nodes == 3
    types = sorted(n.type for n in response.nodes)
    assert types == ["Company", "Person", "Person"]


@pytest.mark.asyncio
async def test_store_and_read_relationships(age_storage):
    """Write nodes + edges, read both back. Edges keyed on the
    same __tid metadata, so the transform-scoped read pulls them
    together with their endpoint nodes."""
    transform_id = "round-trip-rels"
    alice = BaseNode(type="Person", properties={"name": "Alice"})
    acme = BaseNode(type="Company", properties={"name": "Acme"})

    await age_storage.store_nodes(
        [alice, acme], batch_index=0, transform_id=transform_id
    )

    rel = RelationshipInstance(
        type="WORKS_AT",
        source_id=alice.id,
        target_id=acme.id,
        source_type="Person",
        target_type="Company",
        properties={"role": "engineer"},
    )
    rel_result = await age_storage.store_relationships(
        [rel], batch_index=0, transform_id=transform_id
    )
    assert rel_result.success is True

    response = await age_storage.get_transformation_data(transform_id)
    assert response.total_nodes == 2
    assert response.total_edges == 1
    edge = response.edges[0]
    assert edge.type == "WORKS_AT"
    assert edge.source == alice.id
    assert edge.target == acme.id


@pytest.mark.asyncio
async def test_get_node_by_id(age_storage):
    """Single-node lookup by user-facing id (not AGE's internal
    numeric id)."""
    transform_id = "round-trip-get-by-id"
    alice = BaseNode(type="Person", properties={"name": "Alice"})
    await age_storage.store_nodes([alice], batch_index=0, transform_id=transform_id)

    found = await age_storage.get_node_by_id(alice.id)
    assert found is not None
    assert found.id == alice.id
    assert found.type == "Person"
    assert found.properties.get("name") == "Alice"

    missing = await age_storage.get_node_by_id("does-not-exist")
    assert missing is None


@pytest.mark.asyncio
async def test_find_similar_nodes_real(age_storage):
    """Cypher CONTAINS scoring against real AGE. Two Person nodes
    with overlapping name properties; query a near-match string."""
    transform_id = "round-trip-find-similar"
    alice = BaseNode(type="Person", properties={"name": "Alice Smith"})
    alic = BaseNode(type="Person", properties={"name": "Alic"})
    bob = BaseNode(type="Person", properties={"name": "Bob Jones"})

    await age_storage.store_nodes(
        [alice, alic, bob], batch_index=0, transform_id=transform_id
    )

    matches = await age_storage.find_similar_nodes(
        label="Person",
        properties={"name": "alic"},
        similarity_threshold=0.5,
        max_results=10,
    )
    matched_names = {m.properties.get("name") for m in matches}
    assert "Alice Smith" in matched_names
    assert "Alic" in matched_names
    assert "Bob Jones" not in matched_names


@pytest.mark.asyncio
async def test_provenance_fields_round_trip(age_storage):
    """Nodes with NodeProvenance fields write + read back without
    losing the source-span / decision-trail metadata. The merge
    flow + Explorer Evidence tab depend on this round-trip."""
    transform_id = "round-trip-prov"
    node = BaseNode(
        type="Person",
        properties={"name": "Alice"},
        provenance=NodeProvenance(
            chunk_ids=["c-1"],
            extraction_timestamp="2026-04-28T00:00:00+00:00",
            source_file="report.pdf",
            extractor_model="gemini-1.5-pro",
            prompt_version="v1.0.0",
            validator_score=0.92,
        ),
    )
    await age_storage.store_nodes([node], batch_index=0, transform_id=transform_id)

    fetched = await age_storage.get_node_by_id(node.id)
    assert fetched is not None
    # Provenance fields land in the property bag via
    # _attach_provenance_properties at the extraction layer; the
    # adapter's own writepath helper folds NodeProvenance fields
    # in via setdefault. Pin both source-span and decision-trail
    # fields make it through round-trip.
    assert fetched.properties.get("source_file") == "report.pdf"
    assert fetched.properties.get("extractor_model") == "gemini-1.5-pro"
    assert fetched.properties.get("prompt_version") == "v1.0.0"
    assert fetched.properties.get("validator_score") == 0.92


@pytest.mark.asyncio
async def test_create_or_replace_ft_index_for_node_real(age_storage):
    """Real GIN/pg_trgm DDL against AGE-managed entity table.

    Pre-condition: at least one node of the target type must
    exist before the index can be created (AGE creates the
    underlying ag_label table lazily on first insert).
    """
    transform_id = "round-trip-ft-index"
    alice = BaseNode(type="Person", properties={"name": "Alice Smith"})
    await age_storage.store_nodes([alice], batch_index=0, transform_id=transform_id)

    # First call creates the GIN index. Second call (same name,
    # same shape) exercises the DROP IF EXISTS + CREATE idempotency.
    await age_storage.create_or_replace_ft_index_for_node(
        "ix_person_name_test", "Person", ["name"]
    )
    await age_storage.create_or_replace_ft_index_for_node(
        "ix_person_name_test", "Person", ["name"]
    )

    # Verify the index actually exists in the catalog. Skip the
    # check if pg_trgm wasn't bootstrapped (apache/age:latest may
    # not ship it; the adapter's already-warned no-op path took
    # over and left no index to find).
    if not age_storage._has_pg_trgm:
        pytest.skip("pg_trgm not available in this AGE container")

    async with age_storage._get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = %s AND indexname = %s",
                (age_storage.graph_name, "ix_person_name_test"),
            )
            rows = await cur.fetchall()
    assert len(rows) == 1, "expected GIN index to be present in pg_indexes"


@pytest.mark.asyncio
async def test_create_or_replace_ft_index_for_relationship_real(age_storage):
    """Same shape for the relationship variant."""
    transform_id = "round-trip-ft-rel-index"
    alice = BaseNode(type="Person", properties={"name": "Alice"})
    acme = BaseNode(type="Company", properties={"name": "Acme"})
    await age_storage.store_nodes(
        [alice, acme], batch_index=0, transform_id=transform_id
    )

    rel = RelationshipInstance(
        type="WORKS_AT",
        source_id=alice.id,
        target_id=acme.id,
        source_type="Person",
        target_type="Company",
        properties={"role": "engineer"},
    )
    await age_storage.store_relationships(
        [rel], batch_index=0, transform_id=transform_id
    )

    await age_storage.create_or_replace_ft_index_for_relationship(
        "ix_works_at_role_test", "Person", "WORKS_AT", "Company", ["role"]
    )

    if not age_storage._has_pg_trgm:
        pytest.skip("pg_trgm not available in this AGE container")

    async with age_storage._get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = %s AND indexname = %s",
                (age_storage.graph_name, "ix_works_at_role_test"),
            )
            rows = await cur.fetchall()
    assert len(rows) == 1
