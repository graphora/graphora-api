"""Unit tests for ClaimsService (B1-prob slice 1).

Claims keep the full extraction distribution per (target,
property), unlike the existing Decision Log which only records
the pipeline's chosen winner. These tests pin three layers:

  * Append + read round-trip on the memory backend (zero-
    config dev path). Confidence-DESC ordering is load-bearing
    for the Contradictions tab and gets a dedicated test.
  * Contradiction-detection algorithm: 1 distinct value isn't
    a contradiction; 2+ distinct values is; min_confidence
    gates noise out.
  * Postgres-shape SQL parameter binding + DB-exception
    propagation (mirrors the ScenarioService posture —
    claims are user data, not best-effort observability).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from graphora_server.config import settings
from graphora_server.services.claims_service import (
    Claim,
    ClaimsService,
    TargetKind,
    _DEFAULT_MEMORY_STORE,
    _reset_default_memory_store_for_tests,
)


def _claim(
    *,
    transform_id: str = "tx-1",
    target_id: str = "node-alice",
    target_kind: TargetKind = TargetKind.NODE,
    property_key: str = "title",
    value: str = "Senior Engineer",
    confidence: float = 0.9,
    user_id: str = "user-1",
    source_chunk_id: str = "chunk-1",
    source_extractor_model: str = "gemini-2.5-flash",
    source_prompt_version: str = "v1.0",
) -> Claim:
    """Test helper. Defaults form a self-consistent Alice/title
    claim so tests can focus on the differential field they're
    pinning, not the boilerplate."""
    return Claim(
        transform_id=transform_id,
        target_id=target_id,
        target_kind=target_kind,
        property_key=property_key,
        value=value,
        confidence=confidence,
        user_id=user_id,
        source_chunk_id=source_chunk_id,
        source_extractor_model=source_extractor_model,
        source_prompt_version=source_prompt_version,
    )


@pytest.fixture
def memory_service(monkeypatch):
    """Force memory mode + explicit per-instance store. Mirrors
    the decision_log / scenario test fixtures."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    return ClaimsService(memory_store=[])


# ============================================================
# Confidence-range invariant
# ============================================================


def test_claim_rejects_confidence_below_zero():
    """The DB CHECK enforces 0..1; the Python dataclass enforces
    it earlier at the service boundary so a buggy writer fails
    fast instead of crashing the SQL layer with an opaque
    constraint-violation error."""
    with pytest.raises(ValueError, match="confidence"):
        _claim(confidence=-0.1)


def test_claim_rejects_confidence_above_one():
    """Same invariant, upper bound. Pin so a refactor that
    swaps the comparison operator regresses."""
    with pytest.raises(ValueError, match="confidence"):
        _claim(confidence=1.5)


def test_claim_accepts_boundary_values():
    """0.0 and 1.0 are valid — the bound is inclusive on both
    sides. Pin so a refactor that uses strict comparison
    accidentally locks out legitimate boundary claims."""
    _claim(confidence=0.0)
    _claim(confidence=1.0)


# ============================================================
# Append + read round-trip on memory backend
# ============================================================


@pytest.mark.asyncio
async def test_memory_append_and_for_target_roundtrip(memory_service):
    """append populates id + created_at; for_target returns the
    written row. Minimum contract: writes are readable in the
    same service instance."""
    written = await memory_service.append(_claim())

    assert written.id is not None, "id must be auto-populated"
    assert written.created_at is not None, "created_at must be auto-populated"

    fetched = await memory_service.for_target(
        transform_id="tx-1", target_id="node-alice", user_id="user-1"
    )
    assert len(fetched) == 1
    assert fetched[0].id == written.id


@pytest.mark.asyncio
async def test_for_target_orders_by_confidence_desc(memory_service):
    """The "highest confidence first" ordering is load-bearing —
    the Contradictions tab assumes the winning claim renders at
    the top. Pin so a refactor that swaps to created_at-ASC
    (intuitive but wrong) regresses."""
    await memory_service.append(_claim(confidence=0.55, source_chunk_id="chunk-a"))
    await memory_service.append(_claim(confidence=0.95, source_chunk_id="chunk-b"))
    await memory_service.append(_claim(confidence=0.7, source_chunk_id="chunk-c"))

    fetched = await memory_service.for_target(
        transform_id="tx-1", target_id="node-alice", user_id="user-1"
    )
    # Highest confidence first.
    confidences = [c.confidence for c in fetched]
    assert confidences == sorted(confidences, reverse=True), (
        f"for_target must return claims sorted by confidence DESC; "
        f"got {confidences}"
    )


@pytest.mark.asyncio
async def test_for_target_filters_by_tenant(memory_service):
    """Cross-tenant claims must NOT appear in for_target. Pin
    so a regression that drops the user_id filter is caught at
    the boundary."""
    await memory_service.append(_claim(user_id="user-1"))
    await memory_service.append(_claim(user_id="user-2"))

    user1_claims = await memory_service.for_target(
        transform_id="tx-1", target_id="node-alice", user_id="user-1"
    )
    user2_claims = await memory_service.for_target(
        transform_id="tx-1", target_id="node-alice", user_id="user-2"
    )

    assert len(user1_claims) == 1
    assert len(user2_claims) == 1
    assert user1_claims[0].user_id == "user-1"
    assert user2_claims[0].user_id == "user-2"


@pytest.mark.asyncio
async def test_for_transform_groups_by_target_and_property(memory_service):
    """for_transform's ordering must group by (target_id,
    property_key) so the contradiction detector's group walk is
    a single linear pass. Pin the ordering invariant the
    detector depends on."""
    # Two targets, two properties each, two claims per property.
    for target_id in ("a", "b"):
        for property_key in ("name", "title"):
            for confidence in (0.9, 0.7):
                await memory_service.append(
                    _claim(
                        target_id=target_id,
                        property_key=property_key,
                        confidence=confidence,
                        value=f"{target_id}-{property_key}-{confidence}",
                    )
                )

    all_claims = await memory_service.for_transform(
        transform_id="tx-1", user_id="user-1"
    )
    assert len(all_claims) == 8

    # Walk the list: target_ids must be in ASC order; within a
    # target, property_keys ASC; within a property, confidence
    # DESC. Reconstructing the expected key sequence is the
    # cleanest pin.
    actual_keys = [(c.target_id, c.property_key, c.confidence) for c in all_claims]
    expected_keys = [
        ("a", "name", 0.9),
        ("a", "name", 0.7),
        ("a", "title", 0.9),
        ("a", "title", 0.7),
        ("b", "name", 0.9),
        ("b", "name", 0.7),
        ("b", "title", 0.9),
        ("b", "title", 0.7),
    ]
    assert actual_keys == expected_keys


# ============================================================
# Contradiction detection
# ============================================================


@pytest.mark.asyncio
async def test_contradictions_detects_two_distinct_values(memory_service):
    """The core algorithm pin: 2+ distinct claimed values for
    the same (target, property) is a contradiction. Pin so a
    refactor that uses claim-count (instead of distinct-value-
    count) silently breaks the detector for duplicate-but-
    consistent claims (which should NOT be contradictions)."""
    # Alice's title: two different values from two different
    # chunks — this is a real contradiction.
    await memory_service.append(
        _claim(
            property_key="title",
            value="Senior Engineer",
            confidence=0.9,
            source_chunk_id="chunk-1",
        )
    )
    await memory_service.append(
        _claim(
            property_key="title",
            value="Staff Engineer",
            confidence=0.8,
            source_chunk_id="chunk-2",
        )
    )

    contradictions = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1"
    )
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.target_id == "node-alice"
    assert c.property_key == "title"
    assert c.severity == 2
    # Competing claims sorted by confidence DESC: winning value
    # first.
    assert c.competing_claims[0].value == "Senior Engineer"
    assert c.competing_claims[1].value == "Staff Engineer"


@pytest.mark.asyncio
async def test_contradictions_ignores_duplicate_consistent_claims(
    memory_service,
):
    """The same value from multiple chunks is NOT a
    contradiction — the pipeline can legitimately re-emit the
    same fact across chunks. Pin so the detector counts
    DISTINCT values, not claim-rows.

    This is the subtle bug a naïve implementation would have:
    "group has more than one claim → contradiction." The right
    invariant is "group has more than one DISTINCT value →
    contradiction.""" ""
    # Three claims, same value — consistent agreement, not a
    # contradiction.
    for chunk in ("a", "b", "c"):
        await memory_service.append(
            _claim(
                property_key="title",
                value="Senior Engineer",
                confidence=0.9,
                source_chunk_id=f"chunk-{chunk}",
            )
        )

    contradictions = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1"
    )
    assert contradictions == [], (
        "Three claims with the SAME value must NOT register as a "
        "contradiction; got "
        f"{contradictions}"
    )


@pytest.mark.asyncio
async def test_contradictions_min_confidence_filter(memory_service):
    """min_confidence gates low-confidence noise out of the
    contradictions surface. Pin so a refactor that drops the
    filter floods the API with junk."""
    await memory_service.append(_claim(value="Senior Engineer", confidence=0.95))
    # Low-confidence dissenting claim — should be filtered out.
    await memory_service.append(_claim(value="Janitor", confidence=0.1))

    # No filter: contradiction surfaces.
    unfiltered = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1", min_confidence=0.0
    )
    assert len(unfiltered) == 1

    # Filter at 0.5: low-confidence claim drops out, leaving
    # only one claim above floor → no contradiction.
    filtered = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1", min_confidence=0.5
    )
    assert filtered == []


@pytest.mark.asyncio
async def test_contradictions_filters_by_tenant(memory_service):
    """Tenant scoping on the contradiction detector. Pin so a
    refactor that surfaces another user's contradictions
    regresses — would be a real cross-tenant leak."""
    await memory_service.append(_claim(user_id="user-1", value="A", confidence=0.9))
    await memory_service.append(_claim(user_id="user-1", value="B", confidence=0.8))
    # user-2 has their own (non-contradicting) claim.
    await memory_service.append(_claim(user_id="user-2", value="X"))

    user1 = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1"
    )
    user2 = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-2"
    )
    assert len(user1) == 1
    assert user2 == []


@pytest.mark.asyncio
async def test_contradictions_handles_complex_values(memory_service):
    """Claim values can be dicts/lists — JSON-equal comparison
    must group them correctly. Pin so a refactor using Python's
    ``==`` (which compares by hashable identity for unhashable
    types) regresses on structured values.

    Two dict claims with the same keys/values but different
    insertion order are EQUAL contradictions-wise — sort_keys=True
    in the JSON serialization handles this."""
    await memory_service.append(
        _claim(
            property_key="address",
            value={"city": "SF", "country": "US"},
            confidence=0.9,
        )
    )
    # Same dict, different insertion order — should NOT be a
    # contradiction.
    await memory_service.append(
        _claim(
            property_key="address",
            value={"country": "US", "city": "SF"},
            confidence=0.8,
        )
    )
    # Truly different value — IS a contradiction.
    await memory_service.append(
        _claim(
            property_key="address",
            value={"city": "NYC", "country": "US"},
            confidence=0.7,
        )
    )

    contradictions = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1"
    )
    # Three claims, two distinct values (SF and NYC) → severity 2.
    assert len(contradictions) == 1
    assert contradictions[0].severity == 2


@pytest.mark.asyncio
async def test_contradictions_node_and_edge_with_same_id_stay_separate(
    memory_service,
):
    """target_id is unique per target_kind: a node and an edge
    could theoretically share the same id string. The detector
    keys on (target_id, target_kind, property_key), so the
    edge and node contradictions stay in separate groups
    even when the ids match.

    This is a defensive pin against a future refactor that
    drops target_kind from the group key — looks harmless
    but would conflate node and edge contradictions."""
    await memory_service.append(
        _claim(
            target_id="same-id",
            target_kind=TargetKind.NODE,
            value="A",
            confidence=0.9,
        )
    )
    await memory_service.append(
        _claim(
            target_id="same-id",
            target_kind=TargetKind.EDGE,
            value="B",
            confidence=0.9,
        )
    )

    contradictions = await memory_service.contradictions_for_transform(
        transform_id="tx-1", user_id="user-1"
    )
    # Two separate single-value groups, NOT one cross-kind
    # contradiction. Both groups have 1 distinct value → 0
    # contradictions.
    assert contradictions == []


# ============================================================
# Shared dev-mode store
# ============================================================


@pytest.fixture
def _shared_store_reset(monkeypatch):
    """Force memory mode AND clear the module-level store before/
    after each test. Mirrors the ScenarioService fixture
    convention. Leading underscore so vulture's dead-code scan
    skips the test-parameter name (vulture doesn't grok pytest's
    fixture DI)."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    _reset_default_memory_store_for_tests()
    yield
    _reset_default_memory_store_for_tests()


@pytest.mark.asyncio
async def test_default_constructor_shares_memory_across_instances(
    _shared_store_reset,
):
    """Default-constructed services (no ``memory_store=`` arg)
    must share the same underlying list so dev-mode writes
    survive cross-request reads — each API request gets its own
    instance. Same defense as ScenarioService (commit 088692b)."""
    writer = ClaimsService()
    reader = ClaimsService()

    written = await writer.append(_claim())
    fetched = await reader.for_target(
        transform_id="tx-1", target_id="node-alice", user_id="user-1"
    )
    assert len(fetched) == 1
    assert fetched[0].id == written.id, (
        "Default-constructed ClaimsService instances must share "
        "the dev-mode store. The reader saw an empty list, which "
        "means writes are lost between API requests."
    )


@pytest.mark.asyncio
async def test_explicit_memory_store_isolates_from_shared(
    _shared_store_reset,
):
    """The ``memory_store=[]`` test escape hatch must NOT share
    state with the module-level dev store. Pin so a refactor
    that always binds to the shared list breaks tests."""
    isolated = ClaimsService(memory_store=[])
    shared = ClaimsService()

    await isolated.append(_claim())
    shared_fetched = await shared.for_target(
        transform_id="tx-1", target_id="node-alice", user_id="user-1"
    )
    assert shared_fetched == []
    assert _DEFAULT_MEMORY_STORE == []


# ============================================================
# Postgres-shape SQL pins + exception propagation
# ============================================================


@pytest.mark.asyncio
async def test_postgres_append_binds_all_fields(monkeypatch):
    """Pin the Postgres write path: all 12 INSERT parameters
    must bind in the expected order. A refactor that reorders
    or drops a column would silently corrupt data — keep the
    contract at the SQL boundary."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://fake/url")
    service = ClaimsService()
    assert service._enabled

    fake_execute = AsyncMock()
    with patch(
        "graphora_server.services.claims_service.db.execute",
        new=fake_execute,
    ):
        await service.append(
            _claim(
                user_id="user-pg",
                transform_id="tx-pg",
                target_id="node-pg",
                property_key="title",
                value="Engineer",
                confidence=0.85,
            )
        )

    fake_execute.assert_called_once()
    args = fake_execute.await_args.args
    sql = args[0]
    assert "INSERT INTO claims" in sql
    # Positional params after sql: id, user_id, transform_id,
    # target_id, target_kind, property_key, value(Json),
    # confidence, source_chunk_id, source_extractor_model,
    # source_prompt_version, created_at.
    assert args[2] == "user-pg"  # user_id
    assert args[3] == "tx-pg"  # transform_id
    assert args[4] == "node-pg"  # target_id
    assert args[5] == "node"  # target_kind (enum value)
    assert args[6] == "title"  # property_key
    # value is wrapped in Json() — confirm the type so a
    # refactor passing a raw value (which psycopg would reject
    # for non-JSONable types) regresses here.
    from psycopg.types.json import Json

    assert isinstance(args[7], Json)
    assert args[8] == 0.85  # confidence


@pytest.mark.asyncio
async def test_postgres_for_target_propagates_db_exception(monkeypatch):
    """Claims are user data, not best-effort observability. A
    DB outage must NOT masquerade as "no claims for this
    target." Pin propagation so the API layer surfaces 5xx
    correctly — same posture ScenarioService got (commit
    088692b)."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://fake/url")
    service = ClaimsService()

    fake_fetch = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("graphora_server.services.claims_service.db.fetch", new=fake_fetch):
        with pytest.raises(RuntimeError, match="connection refused"):
            await service.for_target(transform_id="tx", target_id="node", user_id="u")


@pytest.mark.asyncio
async def test_postgres_contradictions_propagates_db_exception(monkeypatch):
    """contradictions_for_transform internally calls
    for_transform, which hits the DB. A DB exception in the
    underlying read must propagate through the detector — a
    swallowed exception would silently return [] (no
    contradictions found) for a DB that's actually on fire."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://fake/url")
    service = ClaimsService()

    fake_fetch = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("graphora_server.services.claims_service.db.fetch", new=fake_fetch):
        with pytest.raises(RuntimeError, match="connection refused"):
            await service.contradictions_for_transform(transform_id="tx", user_id="u")
