"""Unit tests for DisputedPairsService (B2-active backend slice A).

Pins three concerns:
  * Tenant isolation — every read/write filters on user_id;
    cross-tenant access returns None / empty (never leaks).
  * State-machine: enqueue forces PENDING; label sets the
    matching status + labeled_at + labeled_by_user_id; re-label
    overwrites prior labels.
  * Both backends (Postgres + in-memory dev) behave identically
    on the surface API.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from graphora_server.config import settings
from graphora_server.services.disputed_pairs_service import (
    Decision,
    DisputedPair,
    DisputedPairsService,
    SourceStage,
    Status,
)


# ============================================================
# Memory backend
# ============================================================


@pytest.fixture
def memory_service(monkeypatch):
    """Force memory mode by disabling DATABASE_URL."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    return DisputedPairsService(memory_store=[])


def _make_pair(
    user_id: str = "user-1",
    transform_id: str = "tx-1",
    node_a: str = "n-a",
    node_b: str = "n-b",
    entity_type: str = "Person",
    source_stage: SourceStage = SourceStage.EMBEDDING_BLOCKER,
    similarity: float | None = None,
) -> DisputedPair:
    return DisputedPair(
        user_id=user_id,
        transform_id=transform_id,
        node_a_id=node_a,
        node_b_id=node_b,
        entity_type=entity_type,
        source_stage=source_stage,
        similarity_score=Decimal(str(similarity)) if similarity is not None else None,
    )


@pytest.mark.asyncio
async def test_enqueue_populates_id_and_created_at(memory_service):
    """The service is the source of truth for these fields —
    callers don't have to mint UUIDs themselves. Pin so a
    future refactor that drops the mint logic surfaces here."""
    pair = _make_pair()
    enqueued = await memory_service.enqueue(pair)
    assert enqueued.id is not None
    assert enqueued.created_at is not None
    assert enqueued.status == Status.PENDING


@pytest.mark.asyncio
async def test_enqueue_forces_pending_even_if_caller_passes_labeled_status(
    memory_service,
):
    """A pair enqueued with a status other than PENDING must be
    coerced to PENDING. Allowing a labeled-at-enqueue path
    would skip the created_at vs labeled_at audit trail the
    dashboard relies on."""
    pair = _make_pair()
    pair.status = Status.LABELED_MATCH
    pair.labeled_at = "2026-01-01T00:00:00+00:00"
    pair.labeled_by_user_id = "user-x"

    enqueued = await memory_service.enqueue(pair)

    assert enqueued.status == Status.PENDING
    assert enqueued.labeled_at is None
    assert enqueued.labeled_by_user_id is None


@pytest.mark.asyncio
async def test_get_isolates_by_user_id(memory_service):
    """Fetching another tenant's pair returns None — never leak
    the existence of pairs that belong to other users. The
    endpoint translates None to 404 without distinguishing
    'does not exist' from 'belongs to another tenant'."""
    pair = await memory_service.enqueue(_make_pair(user_id="user-1"))

    same_user = await memory_service.get(pair.id, "user-1")
    assert same_user is not None
    assert same_user.id == pair.id

    other_user = await memory_service.get(pair.id, "user-2")
    assert other_user is None, (
        "Cross-tenant fetch leaked. ``service.get(id, user_id)`` "
        "must only return pairs owned by the supplied user_id."
    )


@pytest.mark.asyncio
async def test_label_match_transitions_status_and_stamps_metadata(
    memory_service,
):
    """A successful match label moves the pair from PENDING to
    LABELED_MATCH and stamps labeled_at + labeled_by_user_id +
    label_reason. Pinning all three so the audit trail stays
    consistent across the dual-backend write."""
    pair = await memory_service.enqueue(_make_pair(user_id="user-1"))

    updated = await memory_service.label(
        pair_id=pair.id,
        user_id="user-1",
        decision=Decision.MATCH,
        reason="same person, just different aliases",
    )

    assert updated is not None
    assert updated.status == Status.LABELED_MATCH
    assert updated.labeled_at is not None
    assert updated.labeled_by_user_id == "user-1"
    assert updated.label_reason == "same person, just different aliases"


@pytest.mark.asyncio
async def test_label_not_match_uses_corresponding_status(memory_service):
    pair = await memory_service.enqueue(_make_pair(user_id="user-1"))
    updated = await memory_service.label(
        pair.id, user_id="user-1", decision=Decision.NOT_MATCH
    )
    assert updated.status == Status.LABELED_NOT_MATCH


@pytest.mark.asyncio
async def test_label_skip_marks_skipped(memory_service):
    """SKIP is a distinct terminal state — it lets the queue UX
    distinguish 'I reviewed this and labeled it' from 'I deferred
    this for later'. Pin the mapping so a future refactor that
    collapses Skip into the not_match bucket fails loud."""
    pair = await memory_service.enqueue(_make_pair(user_id="user-1"))
    updated = await memory_service.label(
        pair.id, user_id="user-1", decision=Decision.SKIP
    )
    assert updated.status == Status.SKIPPED


@pytest.mark.asyncio
async def test_label_blocks_cross_tenant(memory_service):
    """A label request with the WRONG user_id must not succeed.
    The service returns None (no row matched the
    (pair_id, user_id) tuple); the endpoint translates that to
    404 without leaking the cross-tenant existence."""
    pair = await memory_service.enqueue(_make_pair(user_id="user-1"))

    blocked = await memory_service.label(
        pair_id=pair.id,
        user_id="user-2",  # WRONG tenant
        decision=Decision.MATCH,
    )

    assert blocked is None
    # Confirm the pair itself wasn't mutated.
    still_pending = await memory_service.get(pair.id, "user-1")
    assert still_pending.status == Status.PENDING


@pytest.mark.asyncio
async def test_re_label_overwrites_prior_decision(memory_service):
    """Operators change their minds. A previously labeled pair
    can be re-labeled with a different decision; labeled_at +
    labeled_by_user_id update to reflect the latest write.
    Without this, a separate 'undo' endpoint would be needed."""
    pair = await memory_service.enqueue(_make_pair(user_id="user-1"))

    first = await memory_service.label(
        pair.id, user_id="user-1", decision=Decision.MATCH
    )
    assert first.status == Status.LABELED_MATCH
    first_labeled_at = first.labeled_at

    second = await memory_service.label(
        pair.id,
        user_id="user-1",
        decision=Decision.NOT_MATCH,
        reason="actually different",
    )
    assert second.status == Status.LABELED_NOT_MATCH
    # The new label_reason replaces the prior one (or None).
    assert second.label_reason == "actually different"
    # labeled_at advanced (or at least is non-None and matches
    # the new write).
    assert second.labeled_at is not None
    assert second.labeled_at >= first_labeled_at


@pytest.mark.asyncio
async def test_list_pending_filters_by_user_and_status(memory_service):
    """The pending queue surface returns only this user's
    PENDING pairs — labeled pairs and other users' pairs both
    excluded."""
    user_1_pending = await memory_service.enqueue(
        _make_pair(user_id="user-1", node_a="a", node_b="b")
    )
    user_1_labeled = await memory_service.enqueue(
        _make_pair(user_id="user-1", node_a="c", node_b="d")
    )
    await memory_service.label(
        user_1_labeled.id, user_id="user-1", decision=Decision.MATCH
    )
    await memory_service.enqueue(_make_pair(user_id="user-2", node_a="e", node_b="f"))

    pending = await memory_service.list_pending("user-1")

    assert len(pending) == 1
    assert pending[0].id == user_1_pending.id


@pytest.mark.asyncio
async def test_list_pending_orders_newest_first(memory_service):
    """Index-aligned ordering pin: ``created_at DESC`` so the
    paginated read is index-only and the dashboard shows the
    most-recent ER guesses first."""
    older = await memory_service.enqueue(
        _make_pair(user_id="user-1", node_a="old-a", node_b="old-b")
    )
    older.created_at = "2026-01-01T00:00:00+00:00"
    newer = await memory_service.enqueue(
        _make_pair(user_id="user-1", node_a="new-a", node_b="new-b")
    )
    newer.created_at = "2026-05-01T00:00:00+00:00"

    pending = await memory_service.list_pending("user-1")

    assert [p.id for p in pending] == [newer.id, older.id]


@pytest.mark.asyncio
async def test_list_pending_filters_by_transform_id(memory_service):
    """The transform_id filter is opt-in; when supplied, only
    pairs from that run come back. Used by the per-transform
    review view."""
    tx_a = await memory_service.enqueue(
        _make_pair(user_id="user-1", transform_id="tx-a", node_a="a")
    )
    await memory_service.enqueue(
        _make_pair(user_id="user-1", transform_id="tx-b", node_a="b")
    )

    filtered = await memory_service.list_pending("user-1", transform_id="tx-a")
    assert [p.id for p in filtered] == [tx_a.id]


@pytest.mark.asyncio
async def test_list_for_transform_returns_all_statuses(memory_service):
    """Per-transform view returns all statuses (pending +
    labeled + skipped), not just pending. Pin so a future
    refactor doesn't accidentally inherit the pending-only
    filter from list_pending."""
    pending = await memory_service.enqueue(_make_pair(user_id="user-1", node_a="p"))
    labeled = await memory_service.enqueue(_make_pair(user_id="user-1", node_a="l"))
    await memory_service.label(labeled.id, user_id="user-1", decision=Decision.MATCH)

    pairs = await memory_service.list_for_transform("user-1", "tx-1")
    ids = {p.id for p in pairs}
    assert ids == {pending.id, labeled.id}


@pytest.mark.asyncio
async def test_list_for_transform_isolates_by_user(memory_service):
    """Cross-tenant block on the per-transform view too — a
    user with a guessed transform_id from another tenant's run
    gets an empty list."""
    await memory_service.enqueue(_make_pair(user_id="user-1", transform_id="tx-1"))
    pairs = await memory_service.list_for_transform("user-2", "tx-1")
    assert pairs == []


# ============================================================
# Postgres backend (mocked db helpers)
# ============================================================


@pytest.fixture
def postgres_service(monkeypatch):
    """Force the Postgres path. The ``_enabled`` flag depends on
    DATABASE_URL OR resolved_database_url being truthy; setting
    DATABASE_URL is enough."""
    monkeypatch.setattr(
        settings,
        "DATABASE_URL",
        "postgresql://test:test@localhost/test",
    )
    return DisputedPairsService()


@pytest.mark.asyncio
async def test_postgres_enqueue_inserts_pending_row(postgres_service):
    """Pin the SQL shape: INSERT into disputed_pairs with status
    forced to ``pending`` regardless of the caller's value."""
    with patch(
        "graphora_server.services.disputed_pairs_service.db.execute",
        new=AsyncMock(),
    ) as mock_execute:
        pair = await postgres_service.enqueue(_make_pair())
    assert mock_execute.await_count == 1
    args = mock_execute.await_args.args
    query = args[0]
    assert "INSERT INTO disputed_pairs" in query
    # status arg is the 11th positional (after id, user_id,
    # transform_id, node_a_id, node_b_id, entity_type,
    # node_a_canonical_key, node_b_canonical_key,
    # similarity_score, source_stage).
    assert args[11] == "pending", (
        f"Status arg wasn't forced to pending: {args[11]!r}. "
        "Allowing caller-supplied status would skip the audit "
        "trail's created_at vs labeled_at distinction."
    )
    assert pair.status == Status.PENDING


@pytest.mark.asyncio
async def test_postgres_label_returns_none_on_cross_tenant_attempt(
    postgres_service,
):
    """The UPDATE ... WHERE id = %s AND user_id = %s returns
    zero rows when the pair belongs to another tenant. The
    fetchrow returns None; the service returns None; the
    endpoint translates to 404."""
    with patch(
        "graphora_server.services.disputed_pairs_service.db.fetchrow",
        new=AsyncMock(return_value=None),
    ):
        result = await postgres_service.label(
            pair_id="some-pair-id",
            user_id="user-1",
            decision=Decision.MATCH,
        )
    assert result is None


@pytest.mark.asyncio
async def test_postgres_list_pending_filters_user_and_status(
    postgres_service,
):
    """SQL-shape pin for the pending-queue read. Both ``user_id``
    and ``status = 'pending'`` are in the WHERE clause so the
    (user_id, status, created_at DESC) index from migration 17
    actually covers the query."""
    with patch(
        "graphora_server.services.disputed_pairs_service.db.fetch",
        new=AsyncMock(return_value=[]),
    ) as mock_fetch:
        await postgres_service.list_pending("user-1", limit=10, offset=5)

    query = mock_fetch.await_args.args[0]
    assert "WHERE user_id = %s AND status = %s" in query
    assert "ORDER BY created_at DESC" in query
    assert "LIMIT %s OFFSET %s" in query
    # First two args are user_id + status; limit/offset come last.
    args_after_query = mock_fetch.await_args.args[1:]
    assert args_after_query[0] == "user-1"
    assert args_after_query[1] == "pending"
    assert args_after_query[-2:] == (10, 5)


@pytest.mark.asyncio
async def test_dev_mode_shares_memory_store_across_service_instances(monkeypatch):
    """Reviewer-flagged P2 on commit 26d3e89. Pre-fix every
    ``DisputedPairsService()`` instance got a fresh empty list,
    and the API constructs a new service inside each handler.
    So in dev mode (no DATABASE_URL): a pair enqueued via one
    request was invisible to GET / POST in subsequent requests
    — the queue always looked empty, labels always 404'd.

    Pin: in dev mode, two service instances constructed without
    an explicit ``memory_store`` argument share the module-level
    default. Enqueue via one, read via the other, the pair is
    visible.

    Tests that need isolation continue to pass
    ``memory_store=[]`` — that path bypasses the shared default
    so parallel test runs don't trample each other (existing
    tests in this file all use the per-test isolated list)."""
    from graphora_server.services.disputed_pairs_service import (
        _reset_default_memory_store_for_tests,
    )

    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    # Reset between tests so the module-level default doesn't
    # carry state across the suite.
    _reset_default_memory_store_for_tests()

    # Two distinct service instances — both constructed without
    # ``memory_store=``, mirroring how the API endpoints build
    # their services per-request.
    writer = DisputedPairsService()
    reader = DisputedPairsService()
    assert writer is not reader

    pair = await writer.enqueue(_make_pair(user_id="user-1"))

    # The reader must see the just-enqueued pair.
    pending = await reader.list_pending("user-1")
    assert len(pending) == 1, (
        "Pre-fix: each service instance had a private list, so "
        "the reader saw nothing. Module-level shared store fixes "
        "this. Got pending pairs: "
        f"{[p.id for p in pending]!r}"
    )
    assert pending[0].id == pair.id

    # Cross-tenant filtering still works on the shared store.
    other_tenant_pending = await reader.list_pending("user-other")
    assert other_tenant_pending == [], (
        "Shared store leaked across tenants. user_id filter must " "still apply."
    )

    # Cleanup so the next test sees an empty shared store.
    _reset_default_memory_store_for_tests()


@pytest.mark.asyncio
async def test_explicit_memory_store_isolates_from_shared_default(monkeypatch):
    """Tests that pass ``memory_store=[]`` must NOT see writes
    from the module-level default — otherwise parallel test
    runs would trample each other. Pin the isolation
    contract."""
    from graphora_server.services.disputed_pairs_service import (
        _reset_default_memory_store_for_tests,
    )

    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    _reset_default_memory_store_for_tests()

    # Default-store writer.
    default_writer = DisputedPairsService()
    await default_writer.enqueue(
        _make_pair(user_id="user-1", node_a="default-store-pair")
    )

    # Explicit-store reader. Should NOT see the default-store
    # pair.
    isolated = DisputedPairsService(memory_store=[])
    pending = await isolated.list_pending("user-1")
    assert pending == [], (
        "Explicit memory_store=[] must isolate from the module-"
        "level default. Got: "
        f"{[p.node_a_id for p in pending]!r}"
    )

    _reset_default_memory_store_for_tests()


@pytest.mark.asyncio
async def test_postgres_enqueue_failures_propagate(postgres_service):
    """A degraded Postgres on the write path raises. The service
    deliberately does NOT swallow — losing an enqueued pair means
    silently losing an ER decision that the pipeline already
    deferred. Mirrors the budget-service fail-closed contract."""
    with patch(
        "graphora_server.services.disputed_pairs_service.db.execute",
        new=AsyncMock(side_effect=RuntimeError("postgres down")),
    ):
        with pytest.raises(RuntimeError, match="postgres down"):
            await postgres_service.enqueue(_make_pair())
