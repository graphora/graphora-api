"""End-to-end integration tests for Neo4jStorage.

Round-trips the adapter against a real Neo4j Docker container
brought up by ``conftest.py::neo4j_container``. Validates that
the Cypher patterns the unit tests pin actually work against
real Neo4j, not just our mock-shaped expectations.

These tests are also the regression net for the storage perf
work tracked in CLAUDE.md (N+1 in store_nodes/store_relationships).
A future refactor that batches via UNWIND must keep the round-
trip behaviours pinned here intact — partial-failure semantics,
versioning logic for relationships, transform-scoped reads, etc.

Marked ``integration`` so the unit suite stays Docker-free. Run
explicitly with::

    make test-integration

(That target sets ``GRAPHORA_TEST_REAL_NEO4J=1`` so the
unit-suite stub is bypassed; running pytest directly without
that env var produces a clean skip message.)
"""

from __future__ import annotations

import pytest

from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_store_and_read_nodes_round_trip(neo4j_storage):
    """Bulk-write nodes via store_nodes, read them back via
    get_transformation_data. Covers the happy path of the per-
    node session pattern (pre-perf-fix) AND the future UNWIND-
    batched shape — both must produce a graph readable by the
    transform-scoped read.

    Pin the contract: 3 nodes in → 3 nodes out, types preserved,
    properties preserved."""
    transform_id = "round-trip-nodes"
    nodes = [
        BaseNode(type="Person", properties={"name": "Alice"}),
        BaseNode(type="Person", properties={"name": "Bob"}),
        BaseNode(type="Company", properties={"name": "Acme"}),
    ]

    result = await neo4j_storage.store_nodes(
        nodes, batch_index=0, transform_id=transform_id
    )
    assert result.success is True
    assert result.items_processed == 3

    response = await neo4j_storage.get_transformation_data(transform_id)
    assert response.total_nodes == 3
    types = sorted(n.type for n in response.nodes)
    assert types == ["Company", "Person", "Person"]
    names = sorted(n.properties.get("name") for n in response.nodes)
    assert names == ["Acme", "Alice", "Bob"]


@pytest.mark.asyncio
async def test_store_and_read_relationships_round_trip(neo4j_storage):
    """Write nodes + edges, read both back via the transform-scoped
    fetch. Pins the cross-axis __tid contract that the count-query
    fix series (e96c15c, 16c8ecb) addressed: a transform's
    relationships must come back from get_transformation_data
    when both endpoints AND the relationship itself carry the
    transform's __tid."""
    transform_id = "round-trip-rels"
    alice = BaseNode(type="Person", properties={"name": "Alice"})
    acme = BaseNode(type="Company", properties={"name": "Acme"})

    node_result = await neo4j_storage.store_nodes(
        [alice, acme], batch_index=0, transform_id=transform_id
    )
    assert node_result.success is True

    rel = RelationshipInstance(
        type="WORKS_AT",
        source_id=alice.id,
        target_id=acme.id,
        source_type="Person",
        target_type="Company",
        properties={"role": "engineer"},
    )
    rel_result = await neo4j_storage.store_relationships(
        [rel], batch_index=0, transform_id=transform_id
    )
    assert rel_result.success is True
    assert rel_result.items_processed == 1

    response = await neo4j_storage.get_transformation_data(transform_id)
    assert response.total_nodes == 2
    assert response.total_edges == 1
    edge = response.edges[0]
    assert edge.type == "WORKS_AT"
    assert edge.source == alice.id
    assert edge.target == acme.id
    # Properties survive the round-trip via the SET n += $properties
    # pattern in _build_relationship_query.
    assert edge.properties.get("role") == "engineer"


@pytest.mark.asyncio
async def test_get_transformation_data_scopes_to_single_transform(
    neo4j_storage,
):
    """Reviewer-flagged regression risk on the count-query series:
    a transform's read should never include nodes/edges from a
    different transform that happens to share the same Neo4j
    instance.

    Plant TWO transforms' worth of data; assert each transform's
    get_transformation_data sees only its own. This is the load-
    bearing contract that motivated commits e96c15c, 16c8ecb,
    and b1edc7c — pin it against a real database, not just
    against captured query strings."""
    tx_a = "scoping-tx-a"
    tx_b = "scoping-tx-b"

    a_alice = BaseNode(type="Person", properties={"name": "Alice (A)"})
    a_acme = BaseNode(type="Company", properties={"name": "Acme (A)"})
    b_bob = BaseNode(type="Person", properties={"name": "Bob (B)"})
    b_beta = BaseNode(type="Company", properties={"name": "Beta (B)"})

    await neo4j_storage.store_nodes([a_alice, a_acme], batch_index=0, transform_id=tx_a)
    await neo4j_storage.store_nodes([b_bob, b_beta], batch_index=0, transform_id=tx_b)

    # Edges exclusively within their own transform.
    await neo4j_storage.store_relationships(
        [
            RelationshipInstance(
                type="WORKS_AT",
                source_id=a_alice.id,
                target_id=a_acme.id,
                source_type="Person",
                target_type="Company",
                properties={},
            )
        ],
        batch_index=0,
        transform_id=tx_a,
    )
    await neo4j_storage.store_relationships(
        [
            RelationshipInstance(
                type="WORKS_AT",
                source_id=b_bob.id,
                target_id=b_beta.id,
                source_type="Person",
                target_type="Company",
                properties={},
            )
        ],
        batch_index=0,
        transform_id=tx_b,
    )

    a_data = await neo4j_storage.get_transformation_data(tx_a)
    assert a_data.total_nodes == 2
    assert a_data.total_edges == 1
    a_names = sorted(n.properties.get("name") for n in a_data.nodes)
    assert a_names == ["Acme (A)", "Alice (A)"]

    b_data = await neo4j_storage.get_transformation_data(tx_b)
    assert b_data.total_nodes == 2
    assert b_data.total_edges == 1
    b_names = sorted(n.properties.get("name") for n in b_data.nodes)
    assert b_names == ["Beta (B)", "Bob (B)"]


@pytest.mark.asyncio
async def test_node_provenance_round_trip(neo4j_storage):
    """A1-prov / B0-prov-extend fields write + read back without
    losing source-span / decision-trail metadata. Same contract
    the AGE integration test pins; mirrored here so both backends
    have parity coverage."""
    transform_id = "round-trip-prov"
    node = BaseNode(
        type="Person",
        properties={"name": "Alice"},
        provenance=NodeProvenance(
            chunk_ids=["c-1"],
            extraction_timestamp="2026-05-07T00:00:00+00:00",
            source_file="report.pdf",
            extractor_model="gemini-1.5-pro",
            prompt_version="v1.0.0",
            validator_score=0.92,
        ),
    )
    await neo4j_storage.store_nodes([node], batch_index=0, transform_id=transform_id)

    response = await neo4j_storage.get_transformation_data(transform_id)
    assert response.total_nodes == 1
    fetched = response.nodes[0]
    # Provenance fields ride on the node properties via the
    # _build_node_query path (line 252-253). Pin both source-span
    # and decision-trail fields make it through round-trip.
    assert fetched.properties.get("source_file") == "report.pdf"
    assert fetched.properties.get("extractor_model") == "gemini-1.5-pro"
    assert fetched.properties.get("prompt_version") == "v1.0.0"
    assert fetched.properties.get("validator_score") == 0.92


@pytest.mark.asyncio
async def test_relationship_versioning_keeps_active_only_in_count(
    neo4j_storage,
):
    """When the SAME relationship is stored twice with different
    properties, the adapter must:
      1. Set __valid_to on the existing edge (close v1)
      2. Create a NEW edge alongside (active v2)
      3. get_transformation_data must return only v2 (active),
         filtered by r.__valid_to IS NULL.

    Reviewer-flagged on commit ce22727: pre-fix this test only
    asserted (3) and could pass even when (1) and (2) didn't
    actually create a closed version — the production code's
    MERGE pattern was overwriting v1 in place rather than
    preserving history. The raw-DB assertions below pin all three
    so a regression to the overwrite-in-place behaviour fails
    loud, and strip the active-only filter from
    get_transformation_data fails the count assertion."""
    transform_id = "round-trip-versioning"
    alice = BaseNode(type="Person", properties={"name": "Alice"})
    acme = BaseNode(type="Company", properties={"name": "Acme"})
    await neo4j_storage.store_nodes(
        [alice, acme], batch_index=0, transform_id=transform_id
    )

    rel_v1 = RelationshipInstance(
        type="WORKS_AT",
        source_id=alice.id,
        target_id=acme.id,
        source_type="Person",
        target_type="Company",
        properties={"role": "engineer"},
    )
    await neo4j_storage.store_relationships(
        [rel_v1], batch_index=0, transform_id=transform_id
    )

    # Same logical edge (s, t, type), different role → triggers
    # the versioning path. rel.id is reused intentionally so the
    # adapter has to recognise this as 'same edge, new version'
    # rather than 'fresh edge'.
    rel_v2 = RelationshipInstance(
        id=rel_v1.id,
        type="WORKS_AT",
        source_id=alice.id,
        target_id=acme.id,
        source_type="Person",
        target_type="Company",
        properties={"role": "principal-engineer"},
    )
    await neo4j_storage.store_relationships(
        [rel_v2], batch_index=1, transform_id=transform_id
    )

    # (1) + (2): raw DB assertion. Two edges of the same type
    # exist between alice and acme — one closed (__valid_to set)
    # and one active (__valid_to NULL). Pre-fix the adapter would
    # MERGE-overwrite v1 in place, leaving exactly one edge with
    # __valid_to NULL — which would fail the closed-count assert
    # below. This is the assertion the reviewer specifically asked
    # for ('Add a raw DB assertion for one closed + one active
    # rel').
    async with neo4j_storage._get_session() as session:
        active_result = await session.run(
            "MATCH (s {id: $sid})-[r:WORKS_AT]->(t {id: $tid}) "
            "WHERE r.__valid_to IS NULL RETURN count(r) AS n",
            sid=alice.id,
            tid=acme.id,
        )
        active_count = (await active_result.single())["n"]

        closed_result = await session.run(
            "MATCH (s {id: $sid})-[r:WORKS_AT]->(t {id: $tid}) "
            "WHERE r.__valid_to IS NOT NULL RETURN count(r) AS n",
            sid=alice.id,
            tid=acme.id,
        )
        closed_count = (await closed_result.single())["n"]

    assert active_count == 1, (
        f"Expected exactly one ACTIVE WORKS_AT edge after versioning; "
        f"got {active_count}. The versioning path likely created an "
        f"edge without overwriting __valid_to=NULL."
    )
    assert closed_count == 1, (
        f"Expected exactly one CLOSED (versioned) WORKS_AT edge; got "
        f"{closed_count}. The MERGE-then-SET pattern in "
        f"_build_relationship_query was probably re-matching and "
        f"overwriting v1 instead of preserving history. Switch the "
        f"versioning call site in store_relationships to merge=False "
        f"so CREATE makes a distinct edge."
    )

    # (3): get_transformation_data must apply r.__valid_to IS NULL
    # in BOTH count and data queries. Pre-fix it filtered only by
    # r.__tid; with versioning now actually working, that would
    # double-count the closed edge.
    response = await neo4j_storage.get_transformation_data(transform_id)
    assert response.total_edges == 1, (
        f"get_transformation_data returned total_edges={response.total_edges}, "
        f"expected 1 active. Likely lost the r.__valid_to IS NULL "
        f"filter on the count query."
    )
    assert len(response.edges) == 1, (
        f"get_transformation_data returned {len(response.edges)} edges, "
        f"expected 1 active. Likely lost the r.__valid_to IS NULL "
        f"filter on the data query."
    )
    # And the surviving edge in the payload is the latest version.
    assert response.edges[0].properties.get("role") == "principal-engineer"
