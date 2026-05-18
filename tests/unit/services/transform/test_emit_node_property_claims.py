"""Unit tests for emit_node_property_claims (B1-prob slice 2b).

The hook fires inside graph_transformer's per-chunk extraction
loop. It must:
  * Emit one claim per (target, property) for each extracted node.
  * Key on canonical_id when available (so cross-chunk extractions
    of the same logical entity group correctly).
  * Filter SYSTEM_PROPERTIES (provenance fields aren't claims).
  * Skip None values.
  * Use the node's confidence from provenance, defaulting to 1.0
    when missing.
  * Log-and-swallow per-row failures — extraction must never
    break because a claim row couldn't write.
  * No-op when claims_service / transform_id / user_id is None.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from graphora_server.services.claims_service import (
    ClaimsService,
    TargetKind,
    _reset_default_memory_store_for_tests,
)
from graphora_server.services.transform.helpers import (
    emit_node_property_claims,
)
from graphora_server.services.transform.models import BaseNode, NodeProvenance


def _node(
    *,
    node_id: str = "alice_42",
    canonical_id: str = "cid-alice",
    properties: Dict[str, Any] = None,
    chunk_id: str = "chunk-1",
    extractor_model: str = "gemini-2.5-flash",
    prompt_version: str = "v1.0",
    confidence: float = 0.9,
) -> BaseNode:
    """Build a BaseNode with provenance populated — what
    transform_as_nodes produces post-_attach_provenance_properties."""
    return BaseNode(
        id=node_id,
        type="Person",
        properties=properties if properties is not None else {"name": "Alice"},
        canonical_id=canonical_id,
        canonical_key="Person:name=alice",
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
    methods rather than poking the mock's call list — more
    representative of the real hook path."""
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
    """For a single node with two user-visible properties, the
    helper appends two claims — one per property. Pin so a
    refactor that batches into a single multi-value claim
    regresses (the contradiction detector keys on property_key
    and would lose granularity)."""
    node = _node(
        properties={
            "name": "Alice",
            "title": "Engineer",
        }
    )

    await emit_node_property_claims(
        memory_service,
        [node],
        transform_id="tx-1",
        user_id="user-1",
    )

    # Read back via the service's tenant-scoped lookup.
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    property_keys = {c.property_key for c in claims}
    assert property_keys == {"name", "title"}


@pytest.mark.asyncio
async def test_uses_canonical_id_as_target_id(memory_service):
    """Cross-chunk grouping invariant: claims must key on
    ``canonical_id`` (the stable hash of unique properties)
    NOT the per-chunk ``id``. Otherwise two chunks extracting
    "Alice" with different per-chunk ids would stay in
    separate contradiction groups and the disagreement would
    be invisible."""
    node = _node(node_id="alice_chunk_5", canonical_id="cid-stable")
    await emit_node_property_claims(
        memory_service,
        [node],
        transform_id="tx-1",
        user_id="user-1",
    )
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-stable",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    assert len(claims) == 1
    assert claims[0].target_id == "cid-stable"


@pytest.mark.asyncio
async def test_falls_back_to_id_when_no_canonical_id(memory_service):
    """Ontologies without ``unique``-flagged properties don't
    produce a stable canonical_id. The helper must still emit
    the claim — keyed on the per-chunk id — so the contradiction
    detector at least sees the claim exist, even if it can't
    group cross-chunk for entities without unique properties.
    Graceful degrade, not crash."""
    node = _node(canonical_id=None, node_id="alice_42")
    await emit_node_property_claims(
        memory_service,
        [node],
        transform_id="tx-1",
        user_id="user-1",
    )
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="alice_42",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    assert len(claims) == 1


@pytest.mark.asyncio
async def test_propagates_provenance_fields(memory_service):
    """source_chunk_id / source_extractor_model /
    source_prompt_version must reach the Claim from the node's
    provenance. The contradictions surface uses these to render
    "which extractor + which chunk emitted this claim" — losing
    them would defeat the purpose of keeping competing claims."""
    node = _node(
        chunk_id="chunk-7",
        extractor_model="gemini-2.5-pro",
        prompt_version="v2.1",
    )
    await emit_node_property_claims(
        memory_service,
        [node],
        transform_id="tx-1",
        user_id="user-1",
    )
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    claim = claims[0]
    assert claim.source_chunk_id == "chunk-7"
    assert claim.source_extractor_model == "gemini-2.5-pro"
    assert claim.source_prompt_version == "v2.1"


@pytest.mark.asyncio
async def test_uses_provenance_confidence(memory_service):
    """The Claim's confidence comes from node.provenance.confidence_score.
    Pin so a refactor that hardcodes 1.0 (losing the extractor's
    self-assessment) regresses."""
    node = _node(confidence=0.55)
    await emit_node_property_claims(
        memory_service,
        [node],
        transform_id="tx-1",
        user_id="user-1",
    )
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    assert claims[0].confidence == 0.55


# ============================================================
# Filtering
# ============================================================


@pytest.mark.asyncio
async def test_filters_system_properties(memory_service):
    """SYSTEM_PROPERTIES (source_chunk_id, extractor_model,
    extraction_confidence, etc.) are observability, not claims.
    Emitting them as claims would surface every cross-chunk
    provenance difference as a "contradiction" — exactly the
    wrong signal. Pin the filter so a refactor that drops it
    regresses noisily."""
    # Include a system property AND a real one.
    node = _node(
        properties={
            "name": "Alice",
            "source_chunk_id": "chunk-1",
            "extractor_model": "gemini",
            "extraction_confidence": 0.9,
        }
    )
    await emit_node_property_claims(
        memory_service,
        [node],
        transform_id="tx-1",
        user_id="user-1",
    )
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    property_keys = {c.property_key for c in claims}
    # Only the user-visible property survives.
    assert property_keys == {"name"}


@pytest.mark.asyncio
async def test_skips_none_values(memory_service):
    """LLM extraction sometimes emits None for unset fields.
    Storing "claim that X has property K = None" is misleading
    — the LLM didn't claim K=None, it just didn't claim K.
    Pin the skip so a refactor that emits sentinel None claims
    regresses."""
    node = _node(
        properties={
            "name": "Alice",
            "title": None,
        }
    )
    await emit_node_property_claims(
        memory_service,
        [node],
        transform_id="tx-1",
        user_id="user-1",
    )
    claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    property_keys = {c.property_key for c in claims}
    assert property_keys == {"name"}


# ============================================================
# No-op guards
# ============================================================


@pytest.mark.asyncio
async def test_noop_when_claims_service_is_none():
    """When the caller didn't construct a ClaimsService (older
    test paths, partial wiring), the helper must do nothing
    rather than raise. Pin so the optional-dependency contract
    stays intact and a refactor that hardens the type doesn't
    break older callers."""
    # No claims_service. Must not raise.
    await emit_node_property_claims(
        None,
        [_node()],
        transform_id="tx-1",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_noop_when_user_id_is_none(memory_service):
    """user_id is required at the Claim layer (NOT NULL in
    migration 20, required by the dataclass since commit
    00e7476). The hook must NOT try to write a claim without
    user_id — that would either crash on Claim construction
    or land an orphan row. Pin the no-op so a refactor that
    forwards user_id=None doesn't surface the bug at append
    time."""
    # Save the store size before.
    pre_count = len(memory_service._memory_store)
    await emit_node_property_claims(
        memory_service,
        [_node()],
        transform_id="tx-1",
        user_id=None,
    )
    # Nothing written.
    assert len(memory_service._memory_store) == pre_count


@pytest.mark.asyncio
async def test_noop_when_transform_id_is_none(memory_service):
    """Claims are keyed by transform_id; without it the rows are
    unfindable by the contradictions endpoint. Pin the no-op
    so the helper degrades gracefully rather than writing
    orphan rows. Mirrors B0-log's similar guard."""
    pre_count = len(memory_service._memory_store)
    await emit_node_property_claims(
        memory_service,
        [_node()],
        transform_id=None,
        user_id="user-1",
    )
    assert len(memory_service._memory_store) == pre_count


# ============================================================
# Failure isolation
# ============================================================


@pytest.mark.asyncio
async def test_swallows_append_failures(monkeypatch):
    """Per-row append failure must NOT propagate — claim writes
    are observability, not a correctness gate. The pipeline
    must continue when the DB hiccups. Pin so a refactor that
    raises out of the loop breaks the extraction-availability
    contract.

    Same posture as B0-log's append (commit 8cbc76b)."""
    failing_service = ClaimsService(memory_store=[])
    # Force every append to fail.
    failing_service.append = AsyncMock(side_effect=RuntimeError("DB on fire"))

    # Helper must NOT raise even though every row fails.
    await emit_node_property_claims(
        failing_service,
        [_node(properties={"name": "Alice", "title": "Engineer"})],
        transform_id="tx-1",
        user_id="user-1",
    )

    # All append attempts were made (the loop didn't bail on
    # the first failure).
    assert failing_service.append.await_count == 2


@pytest.mark.asyncio
async def test_emits_for_each_node_in_batch(memory_service):
    """A chunk can produce multiple nodes; the helper iterates
    the batch and emits per-node claims. Pin so a refactor that
    only walks the first node regresses."""
    nodes: List[BaseNode] = [
        _node(canonical_id="cid-alice", properties={"name": "Alice"}),
        _node(canonical_id="cid-bob", properties={"name": "Bob"}),
    ]
    await emit_node_property_claims(
        memory_service,
        nodes,
        transform_id="tx-1",
        user_id="user-1",
    )

    # Both target_ids have their claim.
    alice_claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-alice",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    bob_claims = await memory_service.for_target(
        transform_id="tx-1",
        target_id="cid-bob",
        target_kind=TargetKind.NODE,
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
    """End-to-end pin: simulate two chunks extracting "Alice"
    with different titles. After the helper fires for each
    chunk, the contradiction detector groups them by
    canonical_id and surfaces the disagreement.

    This is the load-bearing scenario for the whole slice —
    the slice 2a surface is meaningless without this
    integration working. Pin so any regression in either the
    helper OR the contradiction detector that breaks the
    cross-chunk grouping invariant is caught."""
    # Chunk 1 extraction.
    chunk_1_node = _node(
        canonical_id="cid-alice",
        node_id="alice_chunk_1",
        properties={"name": "Alice", "title": "Engineer"},
        chunk_id="chunk-1",
    )
    # Chunk 2 extraction — same canonical_id (Alice), different title.
    chunk_2_node = _node(
        canonical_id="cid-alice",
        node_id="alice_chunk_2",
        properties={"name": "Alice", "title": "Senior Engineer"},
        chunk_id="chunk-2",
    )

    # Both extractions go through the helper.
    await emit_node_property_claims(
        memory_service,
        [chunk_1_node],
        transform_id="tx-1",
        user_id="user-1",
    )
    await emit_node_property_claims(
        memory_service,
        [chunk_2_node],
        transform_id="tx-1",
        user_id="user-1",
    )

    # The contradiction detector picks up the title disagreement.
    contradictions = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1"
    )
    title_contradictions = [c for c in contradictions if c.property_key == "title"]
    assert len(title_contradictions) == 1
    c = title_contradictions[0]
    assert c.target_id == "cid-alice"
    assert c.severity == 2
    values = {claim.value for claim in c.competing_claims}
    assert values == {"Engineer", "Senior Engineer"}
    # name had the same value in both chunks → no contradiction.
    name_contradictions = [c for c in contradictions if c.property_key == "name"]
    assert name_contradictions == []
