"""Unit tests for emit_edge_property_claims (B1-prob slice 2b-edge).

The edge helper fires inside the same per-chunk extraction loops
as the node helper, just for relationships. It must:
  * Emit one claim per (target, property) for each extracted
    edge.
  * Key on the canonical edge signature
    ``"{src_cid}|{type}|{tgt_cid}"`` when both endpoints' canonical
    ids resolve through the nodes list (so cross-chunk extractions
    of the same logical relationship group correctly).
  * Resolve endpoint ids via ``node.id`` AND via aliases in
    ``original_extraction_ids`` (handles the post-merge case where
    a relationship's source_id is a pre-merge positional id).
  * Fall back to per-chunk source/target ids when canonical
    resolution fails — graceful degrade, not crash.
  * Filter SYSTEM_PROPERTIES (provenance fields aren't claims).
  * Skip None values.
  * Use the edge's confidence from provenance, defaulting to 1.0.
  * Log-and-swallow per-row failures.
  * No-op when claims_service / transform_id / user_id is None.
  * Mark every claim with target_kind=EDGE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from graphora_server.services.claims_service import (
    ClaimsService,
    TargetKind,
    _reset_default_memory_store_for_tests,
)
from graphora_server.services.transform.helpers import (
    emit_edge_property_claims,
)
from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
)


def _node(
    *,
    node_id: str,
    canonical_id: Optional[str] = None,
    type: str = "Person",
    original_extraction_ids: Optional[List[str]] = None,
) -> BaseNode:
    """A minimal BaseNode shaped like what
    ``transform_as_nodes`` produces. canonical_id is what the
    edge helper reads to build the canonical edge signature;
    original_extraction_ids carries any pre-merge aliases."""
    return BaseNode(
        id=node_id,
        type=type,
        properties={"name": node_id},
        canonical_id=canonical_id,
        canonical_key=(
            f"{type}:name={node_id.lower()}" if canonical_id else None
        ),
        original_extraction_ids=original_extraction_ids or [],
    )


def _edge(
    *,
    edge_id: str = "edge-1",
    edge_type: str = "WORKS_AT",
    source_id: str,
    target_id: str,
    properties: Optional[Dict[str, Any]] = None,
    chunk_id: str = "chunk-1",
    extractor_model: str = "gemini-2.5-flash",
    prompt_version: str = "v1.0",
    confidence: float = 0.9,
) -> RelationshipInstance:
    """Build a RelationshipInstance with provenance populated —
    what transform_as_relationships produces."""
    return RelationshipInstance(
        id=edge_id,
        type=edge_type,
        source_id=source_id,
        target_id=target_id,
        source_type="Person",
        target_type="Organization",
        properties=properties if properties is not None else {"role": "Engineer"},
        provenance=NodeProvenance(
            chunk_ids=[chunk_id],
            confidence_score=confidence,
            extractor_model=extractor_model,
            prompt_version=prompt_version,
        ),
    )


@pytest.fixture
def memory_service(monkeypatch):
    """Real ClaimsService with an isolated in-memory store. Lets
    tests assert post-append state through the service's read
    methods rather than poking the mock's call list."""
    from graphora_server.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    _reset_default_memory_store_for_tests()
    yield ClaimsService(memory_store=[])
    _reset_default_memory_store_for_tests()


# ============================================================
# Happy-path emission
# ============================================================


@pytest.mark.asyncio
async def test_emits_one_claim_per_user_property(memory_service):
    """For a single edge with two user-visible properties, the
    helper appends two claims — one per property. Mirrors the
    per-property granularity invariant of the node helper."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(
        source_id="alice",
        target_id="acme",
        properties={"role": "Engineer", "since": "2020"},
    )

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    property_keys = {c.property_key for c in claims}
    assert property_keys == {"role", "since"}


@pytest.mark.asyncio
async def test_uses_canonical_edge_signature_as_target_id(memory_service):
    """Cross-chunk grouping invariant. The target_id must be the
    canonical edge signature derived from the endpoints'
    canonical_ids — NOT the edge's per-chunk id. Otherwise two
    chunks claiming "Alice WORKS_AT Acme with role=X" vs
    "role=Y" stay in separate contradiction groups."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(
        edge_id="per-chunk-uuid-irrelevant",
        source_id="alice",
        target_id="acme",
    )

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    assert len(claims) == 1
    assert claims[0].target_id == "cid-alice|WORKS_AT|cid-acme"


@pytest.mark.asyncio
async def test_resolves_endpoints_through_extraction_aliases(memory_service):
    """Post-merge nodes carry pre-merge ids in
    ``original_extraction_ids``. The helper must resolve a
    relationship whose source_id is a pre-merge alias against
    the post-merge canonical_id. Pin so a refactor that only
    matches by ``node.id`` regresses — the relationship-
    extraction LLM emits source_id values that came from the
    nodes-context, and that context can carry pre-merge alias
    ids depending on when the context was snapshotted."""
    # alice_chunk_1 and alice_chunk_2 merged into a single
    # canonical Alice; both aliases survive on the merged node.
    alice = _node(
        node_id="alice_post_merge",
        canonical_id="cid-alice",
        original_extraction_ids=["alice_chunk_1", "alice_chunk_2"],
    )
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    # The LLM cited Alice by her chunk-2 alias — the helper must
    # still resolve to cid-alice.
    edge = _edge(source_id="alice_chunk_2", target_id="acme")

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    assert len(claims) == 1


@pytest.mark.asyncio
async def test_falls_back_when_endpoint_canonical_id_missing(memory_service):
    """Ontologies without ``unique``-flagged properties don't
    produce stable canonical_ids on their nodes — node.canonical_id
    is None and the lookup uses node.id directly. Pin that the
    edge claim still emits (keyed on per-chunk ids in the
    signature) rather than crashing. Cross-chunk grouping
    degrades but the claim row still lands."""
    # Both endpoints have no canonical_id.
    alice = _node(node_id="alice_42", canonical_id=None)
    acme = _node(node_id="acme_7", canonical_id=None, type="Organization")
    edge = _edge(source_id="alice_42", target_id="acme_7")

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    # Fallback signature uses the node.id values (since
    # canonical_lookup mapped each to its own id).
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="alice_42|WORKS_AT|acme_7",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    assert len(claims) == 1


@pytest.mark.asyncio
async def test_falls_back_when_endpoint_node_missing(memory_service):
    """The LLM occasionally emits a relationship whose source_id
    or target_id doesn't match any node we extracted (phantom
    endpoint). Downstream validation drops the edge, but at the
    pre-dedup emit point it's still in the list. The helper
    must NOT crash — it should fall back to the raw per-chunk
    id and still emit the claim, so the claim row records what
    the LLM saw even though it won't match anything in the
    graph."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    # acme deliberately missing from nodes list.
    edge = _edge(source_id="alice", target_id="acme_phantom")

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice],
        transform_id="tx-1",
        user_id="user-1",
    )

    # target side falls back to the raw id.
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|acme_phantom",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    assert len(claims) == 1


@pytest.mark.asyncio
async def test_marks_claim_target_kind_as_edge(memory_service):
    """target_kind=EDGE is the discriminator that lets the
    contradictions endpoint and the diff service tell node
    claims from edge claims. Pin so a copy-paste from the node
    helper that forgot to change EDGE→NODE regresses (both
    surfaces would treat edge claims as node claims and the
    grouping would silently collapse)."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(source_id="alice", target_id="acme")

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    assert len(claims) == 1
    assert claims[0].target_kind == TargetKind.EDGE


@pytest.mark.asyncio
async def test_propagates_provenance_fields(memory_service):
    """source_chunk_id / source_extractor_model /
    source_prompt_version must reach the Claim from the edge's
    provenance — same expectation as the node helper, mirrored
    for edges."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(
        source_id="alice",
        target_id="acme",
        chunk_id="chunk-7",
        extractor_model="gemini-2.5-pro",
        prompt_version="v2.1",
    )

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    claim = claims[0]
    assert claim.source_chunk_id == "chunk-7"
    assert claim.source_extractor_model == "gemini-2.5-pro"
    assert claim.source_prompt_version == "v2.1"


@pytest.mark.asyncio
async def test_uses_provenance_confidence(memory_service):
    """The Claim's confidence comes from edge.provenance.confidence_score.
    Mirrors the node-helper invariant."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(source_id="alice", target_id="acme", confidence=0.55)

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    assert claims[0].confidence == 0.55


# ============================================================
# Filtering
# ============================================================


@pytest.mark.asyncio
async def test_filters_system_properties(memory_service):
    """SYSTEM_PROPERTIES (source_chunk_id, extractor_model,
    extraction_confidence, etc.) are observability, not claims —
    same filter as the node helper. transform_as_relationships
    mirrors provenance into edge.properties; without the filter,
    every cross-chunk provenance difference would surface as a
    bogus contradiction."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(
        source_id="alice",
        target_id="acme",
        properties={
            "role": "Engineer",
            "source_chunk_id": "chunk-1",
            "extractor_model": "gemini",
            "extraction_confidence": 0.9,
        },
    )

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    property_keys = {c.property_key for c in claims}
    assert property_keys == {"role"}


@pytest.mark.asyncio
async def test_skips_none_values(memory_service):
    """Mirrors the node-helper skip. LLM extraction can emit
    None for unset fields; storing "claim that this edge has
    property K = None" is misleading."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(
        source_id="alice",
        target_id="acme",
        properties={"role": "Engineer", "since": None},
    )

    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    property_keys = {c.property_key for c in claims}
    assert property_keys == {"role"}


# ============================================================
# No-op guards
# ============================================================


@pytest.mark.asyncio
async def test_noop_when_claims_service_is_none():
    """Older callers without claim wiring pass claims_service=None.
    Must not raise."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(source_id="alice", target_id="acme")

    await emit_edge_property_claims(
        None,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_noop_when_user_id_is_none(memory_service):
    """user_id is NOT NULL at the Claim layer — passing None
    would raise out of Claim construction. Mirrors the node
    helper's pre-construction no-op."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(source_id="alice", target_id="acme")

    pre_count = len(memory_service._memory_store)
    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id=None,
    )
    assert len(memory_service._memory_store) == pre_count


@pytest.mark.asyncio
async def test_noop_when_transform_id_is_none(memory_service):
    """Claims are keyed by transform_id; without it the rows
    are unfindable by the contradictions endpoint. No-op."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(source_id="alice", target_id="acme")

    pre_count = len(memory_service._memory_store)
    await emit_edge_property_claims(
        memory_service,
        [edge],
        [alice, acme],
        transform_id=None,
        user_id="user-1",
    )
    assert len(memory_service._memory_store) == pre_count


# ============================================================
# Failure isolation
# ============================================================


@pytest.mark.asyncio
async def test_swallows_append_failures():
    """Per-row append failure must NOT propagate — same posture
    as the node helper. The extractor must keep running even if
    the claims store is down."""
    failing_service = ClaimsService(memory_store=[])
    failing_service.append = AsyncMock(side_effect=RuntimeError("DB on fire"))

    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    edge = _edge(
        source_id="alice",
        target_id="acme",
        properties={"role": "Engineer", "since": "2020"},
    )

    # Must NOT raise.
    await emit_edge_property_claims(
        failing_service,
        [edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    # Both append attempts were made (loop didn't bail on first
    # failure).
    assert failing_service.append.await_count == 2


@pytest.mark.asyncio
async def test_emits_for_each_edge_in_batch(memory_service):
    """A chunk can produce multiple edges; the helper iterates
    the batch."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    bob = _node(node_id="bob", canonical_id="cid-bob")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    alice_edge = _edge(
        edge_id="e1",
        source_id="alice",
        target_id="acme",
        properties={"role": "Engineer"},
    )
    bob_edge = _edge(
        edge_id="e2",
        source_id="bob",
        target_id="acme",
        properties={"role": "Designer"},
    )

    await emit_edge_property_claims(
        memory_service,
        [alice_edge, bob_edge],
        [alice, bob, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    alice_claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    bob_claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-bob|WORKS_AT|cid-acme",
        target_kind=TargetKind.EDGE,
        user_id="user-1",
    )
    assert len(alice_claims) == 1
    assert len(bob_claims) == 1


# ============================================================
# Integration with contradiction detection
# ============================================================


@pytest.mark.asyncio
async def test_cross_chunk_disagreement_surfaces_as_contradiction(
    memory_service,
):
    """End-to-end pin: two chunks both claim "Alice WORKS_AT
    Acme" but disagree on role. After the helper fires for each
    chunk, the contradiction detector groups them by canonical
    edge signature and surfaces the disagreement.

    Load-bearing for the slice — the surface is meaningless
    without the canonical-signature grouping working. The
    per-chunk edge ids are deliberately different (real chunks
    don't share edge ids) so a regression that keys on edge.id
    instead of the canonical signature would NOT group them and
    the test fails."""
    alice = _node(node_id="alice", canonical_id="cid-alice")
    acme = _node(node_id="acme", canonical_id="cid-acme", type="Organization")
    chunk_1_edge = _edge(
        edge_id="edge-chunk-1-uuid",
        source_id="alice",
        target_id="acme",
        properties={"role": "Engineer"},
        chunk_id="chunk-1",
    )
    chunk_2_edge = _edge(
        edge_id="edge-chunk-2-uuid",
        source_id="alice",
        target_id="acme",
        properties={"role": "Senior Engineer"},
        chunk_id="chunk-2",
    )

    await emit_edge_property_claims(
        memory_service,
        [chunk_1_edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )
    await emit_edge_property_claims(
        memory_service,
        [chunk_2_edge],
        [alice, acme],
        transform_id="tx-1",
        user_id="user-1",
    )

    contradictions = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1"
    )
    role_contradictions = [c for c in contradictions if c.property_key == "role"]
    assert len(role_contradictions) == 1
    c = role_contradictions[0]
    assert c.target_id == "cid-alice|WORKS_AT|cid-acme"
    assert c.target_kind == TargetKind.EDGE
    assert c.severity == 2
    values = {claim.value for claim in c.competing_claims}
    assert values == {"Engineer", "Senior Engineer"}
