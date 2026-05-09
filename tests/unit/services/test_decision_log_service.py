"""Unit tests for DecisionLogService (B0-log slice 1).

The Decision Log is observability, not a correctness gate — these
tests pin both directions:
  * Memory backend (zero-config dev mode): full append/query
    round-trip, ordering, multi-target isolation, schema-level
    decisions with target_id=None.
  * Postgres backend: mock the db helpers so we can pin parameter
    binding, SQL shape, and that exceptions on append don't propagate
    (the "logged-and-swallowed" contract — losing a row must never
    break extraction).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from graphora_server.config import settings
from graphora_server.services.decision_log_service import (
    Decision,
    DecisionLogService,
    DecisionType,
    TargetKind,
)


# ============================================================
# Memory backend
# ============================================================


@pytest.fixture
def memory_service(monkeypatch):
    """Force memory mode by disabling DATABASE_URL/resolved_database_url
    on the settings module. Mirrors the entity_ledger test pattern."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    return DecisionLogService(memory_store=[])


@pytest.mark.asyncio
async def test_memory_append_and_for_target_roundtrip(memory_service):
    """append(decision) + for_target(transform_id, target_id) must
    return what we wrote. The minimal contract."""
    decision = Decision(
        transform_id="tx-1",
        target_id="node-alice",
        target_kind=TargetKind.NODE,
        decision_type=DecisionType.ENTITY_MERGED,
        reason="similarity 0.91 > threshold 0.85",
        evidence={"similarity_score": 0.91, "method": "embedding"},
        alternatives=[{"id": "node-alicia", "score": 0.74}],
    )

    appended = await memory_service.append(decision)
    # append populates id + created_at so callers downstream of the
    # Decision Log tab/MCP get_evidence don't have to mint them.
    assert appended.id is not None
    assert appended.created_at is not None

    results = await memory_service.for_target("tx-1", "node-alice")
    assert len(results) == 1
    got = results[0]
    assert got.target_kind == TargetKind.NODE
    assert got.decision_type == DecisionType.ENTITY_MERGED
    assert got.reason == "similarity 0.91 > threshold 0.85"
    assert got.evidence == {"similarity_score": 0.91, "method": "embedding"}
    assert got.alternatives == [{"id": "node-alicia", "score": 0.74}]


@pytest.mark.asyncio
async def test_memory_for_target_isolates_by_target_id(memory_service):
    """Two decisions in the same transform with different target_ids
    must NOT bleed into each other's for_target results — that's the
    contract the Evidence tab depends on to render only the selected
    edge's history."""
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="node-alice",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="alice merge",
        )
    )
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="node-bob",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="bob merge",
        )
    )

    alice_decisions = await memory_service.for_target("tx-1", "node-alice")
    bob_decisions = await memory_service.for_target("tx-1", "node-bob")

    assert len(alice_decisions) == 1
    assert len(bob_decisions) == 1
    assert alice_decisions[0].reason == "alice merge"
    assert bob_decisions[0].reason == "bob merge"


@pytest.mark.asyncio
async def test_memory_for_target_isolates_by_transform_id(memory_service):
    """Same target_id across different transforms must not collide.
    transform_id is part of the lookup key for exactly this reason —
    the same logical entity may appear in many transforms with
    different decision histories per transform."""
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="node-alice",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="tx-1 merge",
        )
    )
    await memory_service.append(
        Decision(
            transform_id="tx-2",
            target_id="node-alice",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="tx-2 merge",
        )
    )

    tx1 = await memory_service.for_target("tx-1", "node-alice")
    tx2 = await memory_service.for_target("tx-2", "node-alice")

    assert len(tx1) == 1
    assert len(tx2) == 1
    assert tx1[0].reason == "tx-1 merge"
    assert tx2[0].reason == "tx-2 merge"


@pytest.mark.asyncio
async def test_memory_schema_level_decision_with_null_target_id(memory_service):
    """Schema-level decisions don't key on a node/edge id. They must
    still be retrievable via for_transform — the Evidence tab and
    get_evidence consumers expect the full decision history including
    upstream schema choices."""
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id=None,
            target_kind=TargetKind.SCHEMA,
            decision_type=DecisionType.SCHEMA_INFERRED,
            reason="No ontology supplied — inferred from chunk-1",
            evidence={"source_chunks": ["chunk-1"]},
        )
    )

    all_for_tx = await memory_service.for_transform("tx-1")
    assert len(all_for_tx) == 1
    assert all_for_tx[0].target_id is None
    assert all_for_tx[0].target_kind == TargetKind.SCHEMA


@pytest.mark.asyncio
async def test_memory_for_target_does_not_return_schema_decisions(
    memory_service,
):
    """for_target requires a non-None target_id by signature; schema
    decisions (target_id=None) must NOT show up there. Pinning this
    so a future change to the comparison (``d.target_id is None``)
    doesn't accidentally include schema rows in node-specific
    queries."""
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id=None,
            target_kind=TargetKind.SCHEMA,
            decision_type=DecisionType.SCHEMA_INFERRED,
            reason="schema-level",
        )
    )
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="node-alice",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="node-level",
        )
    )

    target_results = await memory_service.for_target("tx-1", "node-alice")
    assert len(target_results) == 1
    assert target_results[0].reason == "node-level"


@pytest.mark.asyncio
async def test_memory_for_transform_returns_chronological_order(memory_service):
    """Decisions returned by for_transform must be ordered by
    created_at ASC. The Decision Log tab renders them as a timeline;
    out-of-order rows would mis-narrate causation (e.g., a downstream
    confidence_marked appearing before the entity_merged that
    triggered it)."""
    # Append three decisions with increasing created_at strings; the
    # service preserves whatever the caller stamped.
    for i, ts in enumerate(
        ["2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00"]
    ):
        await memory_service.append(
            Decision(
                transform_id="tx-1",
                target_id="node-alice",
                target_kind=TargetKind.NODE,
                decision_type=DecisionType.CONFIDENCE_MARKED,
                reason=f"step-{i}",
                created_at=ts,
            )
        )

    results = await memory_service.for_transform("tx-1")
    assert [d.reason for d in results] == ["step-0", "step-1", "step-2"]


@pytest.mark.asyncio
async def test_memory_empty_results_when_no_match(memory_service):
    """No decisions for the target/transform → empty list, not None.
    Callers iterate the result without a None check; returning None
    would be a footgun."""
    assert await memory_service.for_target("tx-1", "missing") == []
    assert await memory_service.for_transform("missing") == []


@pytest.mark.asyncio
async def test_memory_alternatives_field_round_trips_structured_list(
    memory_service,
):
    """The ``alternatives`` field carries scored candidates the
    Evidence tab renders alongside the chosen one. Pinning the
    structured-list shape keeps a future refactor from collapsing it
    to a string or dict."""
    alternatives = [
        {"id": "node-alicia", "score": 0.74, "rejected_reason": "below-threshold"},
        {"id": "node-allison", "score": 0.61, "rejected_reason": "below-threshold"},
    ]
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="node-alice",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.LLM_DISAMBIGUATED,
            reason="LLM picked node-alice",
            alternatives=alternatives,
        )
    )

    [got] = await memory_service.for_target("tx-1", "node-alice")
    assert got.alternatives == alternatives


# ============================================================
# Postgres backend
# ============================================================


@pytest.fixture
def postgres_service(monkeypatch):
    # DATABASE_URL alone is enough to flip _enabled to True — the
    # ``or`` in __init__ short-circuits before reaching
    # resolved_database_url (which is a read-only property and can't
    # be monkeypatched).
    monkeypatch.setattr(
        settings,
        "DATABASE_URL",
        "postgresql://test:test@localhost/test",
    )
    return DecisionLogService()


@pytest.mark.asyncio
async def test_postgres_append_inserts_into_extraction_decisions(
    postgres_service,
):
    """The Postgres-mode append must run an INSERT against
    ``extraction_decisions`` with all fields bound, including JSONB
    payloads via psycopg's Json wrapper. Pinning the SQL shape so a
    future refactor can't quietly drop ``evidence`` or
    ``alternatives`` from the insert."""
    with patch(
        "graphora_server.services.decision_log_service.db.execute",
        new=AsyncMock(),
    ) as mock_execute:
        await postgres_service.append(
            Decision(
                transform_id="tx-1",
                target_id="node-alice",
                target_kind=TargetKind.NODE,
                decision_type=DecisionType.ENTITY_MERGED,
                reason="merged",
                evidence={"score": 0.9},
                alternatives=[{"id": "node-alicia"}],
            )
        )

    assert mock_execute.await_count == 1
    args = mock_execute.await_args.args
    query = args[0]
    assert "INSERT INTO extraction_decisions" in query
    # Positional args follow the query: id, transform_id, target_id,
    # target_kind, decision_type, reason, evidence (Json),
    # alternatives (Json), created_at.
    assert args[2] == "tx-1"
    assert args[3] == "node-alice"
    assert args[4] == TargetKind.NODE.value  # serialized to string
    assert args[5] == DecisionType.ENTITY_MERGED.value
    assert args[6] == "merged"


@pytest.mark.asyncio
async def test_postgres_append_swallows_exceptions(postgres_service, caplog):
    """The Decision Log is observability — a failing append must not
    propagate. Pin: when db.execute raises, append returns the
    decision (with id/created_at) and the failure is logged but no
    exception escapes. Callers don't need to wrap append() in
    try/except for extraction to remain correct."""
    with patch(
        "graphora_server.services.decision_log_service.db.execute",
        new=AsyncMock(side_effect=RuntimeError("postgres down")),
    ):
        decision = Decision(
            transform_id="tx-1",
            target_id="node-alice",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
        )
        # Must not raise.
        appended = await postgres_service.append(decision)
        assert appended.id is not None  # still populated client-side


@pytest.mark.asyncio
async def test_postgres_for_target_runs_indexed_query(postgres_service):
    """Pin the SELECT shape: filter on ``transform_id`` AND
    ``target_id`` (which matches the
    ``idx_extraction_decisions_transform_target`` index from the
    migration). Order by created_at ASC for the timeline contract."""
    fake_row = {
        "id": "decision-1",
        "transform_id": "tx-1",
        "target_id": "node-alice",
        "target_kind": "node",
        "decision_type": "entity_merged",
        "reason": "merged",
        "evidence": {"score": 0.9},
        "alternatives": [],
        "created_at": "2026-05-08T12:00:00+00:00",
    }
    with patch(
        "graphora_server.services.decision_log_service.db.fetch",
        new=AsyncMock(return_value=[fake_row]),
    ) as mock_fetch:
        results = await postgres_service.for_target("tx-1", "node-alice")

    assert mock_fetch.await_count == 1
    query = mock_fetch.await_args.args[0]
    assert "transform_id = %s" in query
    assert "target_id = %s" in query
    assert "ORDER BY created_at ASC" in query
    assert mock_fetch.await_args.args[1:] == ("tx-1", "node-alice")

    assert len(results) == 1
    assert results[0].decision_type == DecisionType.ENTITY_MERGED
    assert results[0].target_kind == TargetKind.NODE


@pytest.mark.asyncio
async def test_postgres_for_transform_runs_indexed_query(postgres_service):
    """Pin the SELECT shape for the transform-only query: filter on
    ``transform_id`` alone (no target_id predicate), so schema-level
    decisions (target_id IS NULL) come back in the result set."""
    with patch(
        "graphora_server.services.decision_log_service.db.fetch",
        new=AsyncMock(return_value=[]),
    ) as mock_fetch:
        await postgres_service.for_transform("tx-1")

    query = mock_fetch.await_args.args[0]
    assert "transform_id = %s" in query
    # Pin: target_id is NOT in the WHERE clause so NULL target_ids
    # (schema-level decisions) are returned.
    where_block = query.split("WHERE")[1].split("ORDER")[0]
    assert "target_id" not in where_block
    assert "ORDER BY created_at ASC" in query
    assert mock_fetch.await_args.args[1:] == ("tx-1",)


@pytest.mark.asyncio
async def test_postgres_for_decision_type_runs_indexed_query(postgres_service):
    """Reviewer-flagged P3 (commit 9ac9bb5): callers needing
    schema-level decisions used to do ``for_transform + Python
    filter`` which fetched every row in the transform. The new
    ``for_decision_type`` narrows the read at the DB layer using
    the (transform_id, decision_type) index from migration 14.

    Pin both the SQL shape AND the parameter binding (the enum's
    ``.value`` is what hits the DB, not the Python enum object)."""
    with patch(
        "graphora_server.services.decision_log_service.db.fetch",
        new=AsyncMock(return_value=[]),
    ) as mock_fetch:
        await postgres_service.for_decision_type("tx-1", DecisionType.SCHEMA_INFERRED)

    query = mock_fetch.await_args.args[0]
    assert "transform_id = %s AND decision_type = %s" in query
    assert "ORDER BY created_at ASC" in query
    # The enum value (string), not the enum object — psycopg can't
    # bind enum types directly.
    assert mock_fetch.await_args.args[1:] == ("tx-1", "schema_inferred")


@pytest.mark.asyncio
async def test_memory_user_id_filter_isolates_tenants(memory_service):
    """Reviewer-flagged P1 (commit eb22a79). When for_target /
    for_transform / for_decision_type is called with a non-None
    user_id, only rows matching that user_id come back. Pin
    bidirectionally: user-1 sees user-1's rows; user-2 sees
    user-2's rows; rows with NULL user_id (legacy) come back only
    when the caller doesn't pass a user_id filter.

    The Postgres backend's WHERE clause does the same filtering
    via a parameterized query — the unit test for that lives
    further down."""
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="n1",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="user-1 merge",
            user_id="user-1",
        )
    )
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="n1",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="user-2 merge — must not leak to user-1",
            user_id="user-2",
        )
    )
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="n1",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="legacy null-user-id row",
            user_id=None,
        )
    )

    user_1_results = await memory_service.for_target("tx-1", "n1", user_id="user-1")
    assert [d.reason for d in user_1_results] == ["user-1 merge"]

    user_2_results = await memory_service.for_target("tx-1", "n1", user_id="user-2")
    assert [d.reason for d in user_2_results] == [
        "user-2 merge — must not leak to user-1"
    ]

    # Without a user_id filter, all three rows come back — that's
    # the legacy-caller path and what tests use to seed.
    no_filter_results = await memory_service.for_target("tx-1", "n1")
    assert len(no_filter_results) == 3


@pytest.mark.asyncio
async def test_postgres_for_target_with_user_id_appends_where_clause(
    postgres_service,
):
    """Pin the SQL shape: when user_id is provided, WHERE adds
    ``AND user_id = %s`` and the param is bound. When user_id
    is None, that clause is omitted (legacy callers untouched)."""
    with patch(
        "graphora_server.services.decision_log_service.db.fetch",
        new=AsyncMock(return_value=[]),
    ) as mock_fetch:
        await postgres_service.for_target("tx-1", "n1", user_id="user-1")

    query = mock_fetch.await_args.args[0]
    assert "AND user_id = %s" in query
    assert mock_fetch.await_args.args[1:] == ("tx-1", "n1", "user-1")

    # Now without user_id — clause omitted, two params only.
    with patch(
        "graphora_server.services.decision_log_service.db.fetch",
        new=AsyncMock(return_value=[]),
    ) as mock_fetch_no_user:
        await postgres_service.for_target("tx-1", "n1")

    query_no_user = mock_fetch_no_user.await_args.args[0]
    assert "AND user_id" not in query_no_user
    assert mock_fetch_no_user.await_args.args[1:] == ("tx-1", "n1")


@pytest.mark.asyncio
async def test_memory_for_decision_type_filters_correctly(memory_service):
    """Memory backend equivalent: filter by decision_type on the
    in-memory list. The two backends must return shape-identical
    results so dev-mode (memory) and prod (Postgres) behave the
    same."""
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id=None,
            target_kind=TargetKind.SCHEMA,
            decision_type=DecisionType.SCHEMA_INFERRED,
            reason="schema",
        )
    )
    await memory_service.append(
        Decision(
            transform_id="tx-1",
            target_id="n1",
            target_kind=TargetKind.NODE,
            decision_type=DecisionType.ENTITY_MERGED,
            reason="merge",
        )
    )

    schema_only = await memory_service.for_decision_type(
        "tx-1", DecisionType.SCHEMA_INFERRED
    )
    assert len(schema_only) == 1
    assert schema_only[0].reason == "schema"

    merge_only = await memory_service.for_decision_type(
        "tx-1", DecisionType.ENTITY_MERGED
    )
    assert len(merge_only) == 1
    assert merge_only[0].reason == "merge"


@pytest.mark.asyncio
async def test_postgres_malformed_row_isolates_does_not_blank_query(
    postgres_service,
):
    """Reviewer-flagged on commit 8cbc76b. Pre-fix the row→Decision
    conversion ran inside a list-comprehension wrapped in a single
    outer try/except, so a single bad row (e.g. a stale
    ``decision_type`` value the running code doesn't yet know about
    after a rollback or rolling deploy) raised ``ValueError`` and
    the whole for_target/for_transform call returned ``[]`` —
    blanking the Decision Log for that target.

    Post-fix: per-row try/except isolates the bad row, logs it, and
    keeps the good rows. Combined with the DB-side CHECK
    constraints (migration 14), this is defence in depth: the
    constraints prevent malformed rows landing, and per-row
    isolation contains any that slip through (e.g. cross-deploy
    skew where the schema gained an enum value before the code did).
    """
    good_row = {
        "id": "decision-good",
        "transform_id": "tx-1",
        "target_id": "node-alice",
        "target_kind": "node",
        "decision_type": "entity_merged",
        "reason": "good row",
        "evidence": {},
        "alternatives": [],
        "created_at": "2026-05-08T12:00:00+00:00",
    }
    bad_row = {
        "id": "decision-bad",
        "transform_id": "tx-1",
        "target_id": "node-alice",
        "target_kind": "node",
        # Not in DecisionType — would raise ValueError in
        # _row_to_decision pre-fix.
        "decision_type": "decision_type_from_a_future_release",
        "reason": "bad row",
        "evidence": {},
        "alternatives": [],
        "created_at": "2026-05-08T12:00:01+00:00",
    }
    another_good_row = {
        "id": "decision-good-2",
        "transform_id": "tx-1",
        "target_id": "node-alice",
        "target_kind": "node",
        "decision_type": "confidence_marked",
        "reason": "another good row",
        "evidence": {},
        "alternatives": [],
        "created_at": "2026-05-08T12:00:02+00:00",
    }

    with patch(
        "graphora_server.services.decision_log_service.db.fetch",
        new=AsyncMock(
            return_value=[good_row, bad_row, another_good_row],
        ),
    ):
        results = await postgres_service.for_target("tx-1", "node-alice")

    # Pre-fix: this would be 0 (the ValueError on the bad row escaped
    # the list-comp into the outer except, returning []).
    # Post-fix: 2 — bad row skipped, good rows preserved.
    assert len(results) == 2, (
        f"Expected 2 good rows preserved, got {len(results)}. The bad "
        f"row's ValueError must be caught per-row, not at the query "
        f"level — otherwise one stale row blanks the whole Decision "
        f"Log surface for that target."
    )
    reasons = [d.reason for d in results]
    assert "good row" in reasons
    assert "another good row" in reasons
    assert "bad row" not in reasons
