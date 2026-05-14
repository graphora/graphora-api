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
    forced to ``pending`` regardless of the caller's value, plus
    the ON CONFLICT clause that backs idempotency (P2 fix on
    commit 150677a). Without the conflict clause the queue
    accumulates duplicate pending rows from task retries."""
    # Mock fetchrow to return the just-inserted row so the
    # service's _row_to_pair path produces a valid DisputedPair.
    inserted_row = {
        "id": "pair-id-1",
        "user_id": "user-1",
        "transform_id": "tx-1",
        "node_a_id": "n-a",
        "node_b_id": "n-b",
        "entity_type": "Person",
        "node_a_canonical_key": None,
        "node_b_canonical_key": None,
        "similarity_score": None,
        "source_stage": "embedding_blocker",
        "status": "pending",
        "labeled_at": None,
        "labeled_by_user_id": None,
        "label_reason": None,
        "created_at": "2026-05-14T00:00:00+00:00",
    }
    with patch(
        "graphora_server.services.disputed_pairs_service.db.fetchrow",
        new=AsyncMock(return_value=inserted_row),
    ) as mock_fetchrow:
        pair = await postgres_service.enqueue(_make_pair())
    assert mock_fetchrow.await_count == 1, (
        "On the insert path (no conflict) we expect a single "
        "fetchrow call. A second call only happens on the "
        "conflict-fallback SELECT path."
    )
    args = mock_fetchrow.await_args.args
    query = args[0]
    assert "INSERT INTO disputed_pairs" in query
    assert "ON CONFLICT" in query, (
        "P2 pin (commit 150677a): the INSERT must carry an "
        "ON CONFLICT clause so duplicate enqueues don't pile up "
        "pending rows. Without it, task retries / re-extractions "
        "would create N rows for the same gray-zone pair."
    )
    assert "DO NOTHING" in query
    assert "LEAST(node_a_id, node_b_id)" in query and (
        "GREATEST(node_a_id, node_b_id)" in query
    ), (
        "ON CONFLICT must reference the unordered pair expression "
        "matching migration 18's unique index — otherwise "
        "(node_a, node_b) and (node_b, node_a) bypass dedup."
    )
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
    deferred. Mirrors the budget-service fail-closed contract.

    Note: the upstream hook in graph_transformer.py wraps THIS
    call in a try/except that swallows for observability. The
    service itself stays fail-closed; the wrapping decision lives
    at the call site."""
    with patch(
        "graphora_server.services.disputed_pairs_service.db.fetchrow",
        new=AsyncMock(side_effect=RuntimeError("postgres down")),
    ):
        with pytest.raises(RuntimeError, match="postgres down"):
            await postgres_service.enqueue(_make_pair())


# ============================================================
# Idempotency (reviewer P2 on commit 150677a)
#
# Slice B's write hook fires every time a 2-node candidate
# resolves as all-singletons. Task retries / re-extractions
# surface the SAME pair multiple times with deterministic node
# IDs — without an unordered-pair unique key the queue
# accumulates duplicates and labeling one leaves siblings
# pending. These tests pin the dedup contract on both backends.
# ============================================================


@pytest.mark.asyncio
async def test_memory_duplicate_enqueue_is_idempotent(memory_service):
    """Calling enqueue twice with the same (user_id, transform_id,
    source_stage, node pair) must NOT create a second row. The
    second call returns the SAME pair instance (same id,
    same created_at) as the first."""
    first = await memory_service.enqueue(_make_pair(node_a="x", node_b="y"))
    second = await memory_service.enqueue(_make_pair(node_a="x", node_b="y"))

    assert first.id == second.id, (
        "Duplicate enqueue returned a different id — the dedup "
        "logic missed the existing row. Memory backend must "
        "mirror the Postgres ON CONFLICT semantics."
    )
    pending = await memory_service.list_pending("user-1")
    assert len(pending) == 1, (
        f"Expected one pending row after duplicate enqueue; got "
        f"{len(pending)}. Pre-fix: every call appended a new row."
    )


@pytest.mark.asyncio
async def test_memory_enqueue_is_order_independent(memory_service):
    """(node_a, node_b) and (node_b, node_a) represent the same
    candidate pair — the disputed-pair concept is unordered.
    Migration 18's index uses LEAST/GREATEST to canonicalize;
    the memory backend mirrors with sorted() on the canonical
    key. Pin so a future "simpler" refactor that drops the
    sort doesn't quietly start duplicating reversed pairs."""
    first = await memory_service.enqueue(_make_pair(node_a="alpha", node_b="beta"))
    reversed_pair = await memory_service.enqueue(
        _make_pair(node_a="beta", node_b="alpha")
    )

    assert first.id == reversed_pair.id, (
        "Reversed enqueue created a new row — the canonical key "
        "isn't order-independent. (a,b) and (b,a) must dedup."
    )
    pending = await memory_service.list_pending("user-1")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_memory_same_pair_different_stage_creates_separate_rows(memory_service):
    """The same node pair surfaced by ``property_blocker`` vs
    ``embedding_blocker`` is two different signals — both are
    worth showing on the queue surface so reviewers can see
    which blocker raised each pair. Pin source_stage as part of
    the dedup key."""
    property_pair = await memory_service.enqueue(
        _make_pair(node_a="x", node_b="y", source_stage=SourceStage.PROPERTY_BLOCKER)
    )
    embedding_pair = await memory_service.enqueue(
        _make_pair(node_a="x", node_b="y", source_stage=SourceStage.EMBEDDING_BLOCKER)
    )

    assert property_pair.id != embedding_pair.id, (
        "source_stage was incorrectly collapsed into a single row "
        "— reviewers lose the diagnostic signal of which blocker "
        "raised the candidate."
    )
    pending = await memory_service.list_pending("user-1")
    assert len(pending) == 2


@pytest.mark.asyncio
async def test_memory_re_enqueue_preserves_label_sticky(memory_service):
    """If a user already labeled a pair and the pipeline
    re-extracts (different chunk batch, task retry, etc.), the
    re-enqueue must NOT reset the row to pending. The user's
    decision sticks. Returning the existing labeled row on
    conflict (rather than overwriting) is what enforces this."""
    first = await memory_service.enqueue(_make_pair(node_a="x", node_b="y"))
    labeled = await memory_service.label(
        pair_id=first.id, user_id="user-1", decision=Decision.MATCH
    )
    assert labeled.status == Status.LABELED_MATCH

    # Re-extract surfaces the same pair again. The pipeline
    # blindly enqueues; the service must keep the label intact.
    re_enqueued = await memory_service.enqueue(_make_pair(node_a="x", node_b="y"))

    assert re_enqueued.id == first.id
    assert re_enqueued.status == Status.LABELED_MATCH, (
        "Re-enqueue overwrote a labeled pair back to pending. "
        "The user's labeling work would be silently destroyed "
        "by every task retry — the queue is supposed to be the "
        "user's audit trail, not the pipeline's scratchpad."
    )
    # And the queue still shows only one row total — labeled,
    # not pending, so list_pending must return nothing.
    pending = await memory_service.list_pending("user-1")
    assert pending == []


@pytest.mark.asyncio
async def test_memory_different_transforms_have_independent_pairs(memory_service):
    """transform_id IS part of the dedup key — the same node
    pair from a different transform_id is a different pair
    (different extraction run, different reviewer context). Pin
    so a future "global pair store" refactor surfaces the
    design decision intentionally."""
    pair_a = await memory_service.enqueue(
        _make_pair(transform_id="tx-1", node_a="x", node_b="y")
    )
    pair_b = await memory_service.enqueue(
        _make_pair(transform_id="tx-2", node_a="x", node_b="y")
    )

    assert pair_a.id != pair_b.id
    pending = await memory_service.list_pending("user-1")
    assert len(pending) == 2


@pytest.mark.asyncio
async def test_postgres_enqueue_returns_existing_on_conflict(postgres_service):
    """On the Postgres path, when the INSERT collides with the
    unique-pair index, ``DO NOTHING`` swallows the insert and
    RETURNING is empty. The service then fetches the existing
    row by the canonical key and returns it.

    Two fetchrow calls expected: the INSERT (returns None) and
    the followup SELECT (returns the existing labeled row). The
    returned pair reflects the EXISTING row's state — including
    any prior label — not the input pair."""
    existing_labeled_row = {
        "id": "existing-pair-id",
        "user_id": "user-1",
        "transform_id": "tx-1",
        "node_a_id": "n-a",
        "node_b_id": "n-b",
        "entity_type": "Person",
        "node_a_canonical_key": None,
        "node_b_canonical_key": None,
        "similarity_score": None,
        "source_stage": "embedding_blocker",
        "status": "labeled_match",  # User already labeled this
        "labeled_at": "2026-05-13T12:00:00+00:00",
        "labeled_by_user_id": "user-1",
        "label_reason": "same Alice",
        "created_at": "2026-05-13T11:00:00+00:00",
    }
    # First fetchrow (INSERT ON CONFLICT) returns None; second
    # (SELECT existing) returns the labeled row.
    fetchrow_mock = AsyncMock(side_effect=[None, existing_labeled_row])
    with patch(
        "graphora_server.services.disputed_pairs_service.db.fetchrow",
        new=fetchrow_mock,
    ):
        returned = await postgres_service.enqueue(_make_pair())

    assert fetchrow_mock.await_count == 2, (
        "Expected two fetchrow calls on the conflict path: "
        "INSERT ON CONFLICT DO NOTHING (returns None) + SELECT "
        "for the existing row. Got: "
        f"{fetchrow_mock.await_count}"
    )
    # The second call must be a SELECT against the canonical
    # key — order-independent via LEAST/GREATEST so node_a/node_b
    # in either order finds the existing row.
    second_call_query = fetchrow_mock.await_args_list[1].args[0]
    assert "SELECT" in second_call_query
    assert "LEAST(node_a_id, node_b_id) = %s" in second_call_query
    assert "GREATEST(node_a_id, node_b_id) = %s" in second_call_query

    # And the returned pair reflects the existing labeled state,
    # not the caller's freshly-minted PENDING input.
    assert returned.id == "existing-pair-id"
    assert returned.status == Status.LABELED_MATCH, (
        "On conflict the service returned the input pair (PENDING) "
        "instead of the existing labeled row. Re-extractions would "
        "appear to silently re-open labeled work."
    )
    assert returned.label_reason == "same Alice"


@pytest.mark.asyncio
async def test_postgres_enqueue_handles_missing_existing_row_defensively(
    postgres_service, caplog
):
    """Edge case: ON CONFLICT fires (INSERT returned None) but
    the followup SELECT also returns None (e.g., concurrent
    DELETE — slice A has no DELETE endpoint, but defense in
    depth). The service must log a warning and return the
    caller's input pair rather than crashing or returning None
    (which the typed return wouldn't allow)."""
    fetchrow_mock = AsyncMock(side_effect=[None, None])
    with patch(
        "graphora_server.services.disputed_pairs_service.db.fetchrow",
        new=fetchrow_mock,
    ):
        returned = await postgres_service.enqueue(_make_pair())

    assert returned is not None
    assert any(
        "no matching row found" in rec.message for rec in caplog.records
    ), "Defensive branch must log a warning so operators can spot the anomaly."


# ============================================================
# B2-active slice C: label → merge_learning_service feedback
#
# After a successful label transition, DisputedPairsService.label
# nudges the per-(user, entity_type) threshold in merge_learning
# via apply_pair_label. These tests pin:
#   * the hook fires with the right args on successful labels
#   * cross-tenant labels (which return None) do NOT trigger the
#     hook — otherwise an attacker could move another tenant's
#     thresholds without touching their data
#   * apply_pair_label failures are swallowed (label is the
#     primary persisted artifact; threshold update is bookkeeping)
# ============================================================


@pytest.fixture
def _reset_merge_learning():
    """Reset the module-level merge_learning_service stats before
    each test so prior tests' bootstrap entries don't pollute the
    assertions."""
    from graphora_server.services.merge.learning import merge_learning_service

    merge_learning_service.reset()
    yield merge_learning_service
    merge_learning_service.reset()


@pytest.mark.asyncio
async def test_memory_label_match_invokes_merge_learning(
    memory_service, _reset_merge_learning
):
    """Slice C contract: a successful match label fires
    apply_pair_label with the pair's entity_type and the
    decision. The merge_learning singleton mutates as a result —
    verifiable via snapshot()."""
    pair = await memory_service.enqueue(_make_pair(entity_type="Company"))

    labeled = await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.MATCH
    )
    assert labeled.status == Status.LABELED_MATCH

    # The hook should have bootstrapped a stats slot for
    # (user-1, Company). Pre-fix: the label persisted but
    # merge_learning saw nothing.
    snapshot = _reset_merge_learning.snapshot()
    assert ("user-1", "Company") in snapshot, (
        "Slice-C hook didn't fire: merge_learning has no stats "
        f"slot for the labeled (user, type) tuple. Got: "
        f"{list(snapshot.keys())!r}"
    )


@pytest.mark.asyncio
async def test_memory_label_skip_does_not_change_threshold(
    memory_service, _reset_merge_learning
):
    """SKIP labels reach apply_pair_label (the service is the
    dispatcher) but apply_pair_label is a no-op on SKIP. End
    result: skipping a pair does NOT shift the threshold, which
    is the right UX — deferring review shouldn't silently
    change ER behavior."""
    pair = await memory_service.enqueue(_make_pair(entity_type="Person"))

    await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.SKIP
    )

    snapshot = _reset_merge_learning.snapshot()
    assert ("user-1", "Person") not in snapshot, (
        "SKIP shifted the threshold — the no-op contract is "
        "broken. Deferring a pair must not mutate the merge "
        "calibration for that user/type."
    )


@pytest.mark.asyncio
async def test_cross_tenant_label_miss_does_not_fire_hook(
    memory_service, _reset_merge_learning
):
    """When the label call returns None (cross-tenant attempt,
    missing pair), the merge_learning hook MUST NOT fire. A
    malicious caller probing pair IDs from another tenant
    could otherwise shift that tenant's thresholds — the label
    persistence wisely returns None for cross-tenant, but the
    side-effect would silently leak data influence.

    Pin so the hook stays inside the success branch."""
    # Pair owned by user-1.
    await memory_service.enqueue(_make_pair(user_id="user-1", entity_type="Company"))
    # user-2 tries to label it.
    result = await memory_service.label(
        pair_id="bogus-id", user_id="user-2", decision=Decision.MATCH
    )
    assert result is None
    # No stats slot for either tenant — the hook didn't fire.
    snapshot = _reset_merge_learning.snapshot()
    assert snapshot == {}, (
        "Hook fired on a cross-tenant/missing label. An attacker "
        "could move another user's thresholds by spamming label "
        "calls with guessed pair IDs."
    )


@pytest.mark.asyncio
async def test_label_with_merge_learning_failure_still_returns_labeled_pair(
    memory_service, caplog
):
    """The label is the user's primary action — it must succeed
    independent of the merge_learning side-effect. Pin the
    swallow contract: if apply_pair_label raises, label()
    returns the labeled pair anyway and logs a warning.

    Mirrors the observability-vs-correctness split applied to
    DecisionLogService.append in B0-log slice 2."""
    pair = await memory_service.enqueue(_make_pair(entity_type="Company"))

    with patch(
        "graphora_server.services.disputed_pairs_service."
        "merge_learning_service.apply_pair_label",
        new=AsyncMock(side_effect=RuntimeError("learning down")),
    ):
        labeled = await memory_service.label(
            pair_id=pair.id, user_id="user-1", decision=Decision.MATCH
        )

    assert labeled is not None, (
        "Label returned None when the merge_learning hook failed. "
        "The label is the persisted artifact and must not be "
        "abandoned just because the threshold bookkeeping crashed."
    )
    assert labeled.status == Status.LABELED_MATCH
    assert any(
        "label persisted, threshold unchanged" in rec.message for rec in caplog.records
    ), "Swallowed merge_learning failure must surface in logs."


@pytest.mark.asyncio
async def test_get_threshold_reflects_match_label_end_to_end(
    memory_service, _reset_merge_learning
):
    """End-to-end pin matching the slice C exit signal: enqueue
    a pair, label it MATCH, then call get_threshold for the
    same (user, type) tuple — the returned threshold reflects
    the label. Without this, the active-learning feedback loop
    isn't observable from the public service surface."""
    pair = await memory_service.enqueue(_make_pair(entity_type="Company"))
    await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.MATCH
    )

    default_threshold = 0.95
    threshold = await _reset_merge_learning.get_threshold(
        "user-1", "Company", default_threshold
    )
    # Bootstrap: ema=0.95 (1.0 + default match_nudge of -0.05).
    # get_threshold logic: ema (0.95) >= default - margin (0.90)
    # → adaptive sticks at default. Pin instead on the
    # snapshot: the stats slot exists with ema below 1.0, which
    # IS the directional update.
    snapshot = _reset_merge_learning.snapshot()
    stats = snapshot[("user-1", "Company")]
    assert stats.ema_low_score < 1.0, (
        f"After a MATCH label the ema_low_score should be below "
        f"the 1.0 prior; got {stats.ema_low_score}. The feedback "
        "loop didn't take effect."
    )
    # threshold itself may still equal default on the first
    # label (the EMA is still close to the default-margin
    # boundary); we DON'T assert threshold < default here. The
    # apply_pair_label unit tests in test_merge_learning.py pin
    # the threshold-change path explicitly; this end-to-end test
    # just confirms the wiring fires.
    assert threshold == default_threshold or threshold < default_threshold


# ============================================================
# Reviewer-flagged P2 (commit 72381b4): label() must be
# idempotent w.r.t. merge_learning. A client retry, browser
# double-submit, or re-saving the same MATCH label was moving
# the threshold N times for one human decision. The fix:
# label() captures the PRE-update status and passes it to
# apply_pair_label, which now operates on a (old, new)
# transition rather than a single decision. These tests pin
# the idempotency + transition semantics from the service
# surface.
# ============================================================


@pytest.mark.asyncio
async def test_double_submit_match_label_does_not_double_nudge(
    memory_service, _reset_merge_learning
):
    """The central P2 pin. Pre-fix: labeling the same pair MATCH
    twice (browser retry, double-click, etc.) moved the
    Company threshold from 0.95 to 0.90 — one nudge per call.
    Post-fix: the second call sees old_status=LABELED_MATCH and
    new_status=LABELED_MATCH, computes delta=0, no-ops the
    merge_learning hook."""
    pair = await memory_service.enqueue(_make_pair(entity_type="Company"))

    # First submit.
    await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.MATCH
    )
    after_first = _reset_merge_learning.snapshot()[("user-1", "Company")].ema_low_score

    # Double-submit: same pair, same decision.
    await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.MATCH
    )
    after_second = _reset_merge_learning.snapshot()[("user-1", "Company")].ema_low_score

    assert pytest.approx(after_first, rel=1e-9) == after_second, (
        "Double-submit moved the threshold twice. Pre-fix: "
        f"after_first={after_first}, after_second={after_second} "
        "— the threshold should remain steady on idempotent "
        "re-labels because delta=0 for same-status transitions."
    )


@pytest.mark.asyncio
async def test_match_to_not_match_relabel_applies_full_swing(
    memory_service, _reset_merge_learning
):
    """User changes their mind from MATCH to NOT_MATCH. The
    label endpoint already supports re-labeling (overwrite);
    the threshold must reflect BOTH undoing the prior match
    contribution AND applying the new not_match contribution
    in a single transition.

    With default nudges (±0.05): pending→match seeds ema at 0.95;
    match→not_match delta = +0.05 - (-0.05) = +0.10. ema goes
    to 0.95 + 0.10 = 1.05, clamped to 1.00.

    Pre-fix bug: each label call applied its OWN nudge in
    isolation, so MATCH then NOT_MATCH would have left the ema
    at 0.95 + 0.05 = 1.00 by accident. Post-fix, the test
    structure makes the transition-aware semantics explicit."""
    pair = await memory_service.enqueue(_make_pair(entity_type="Company"))

    # MATCH first.
    await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.MATCH
    )
    after_match = _reset_merge_learning.snapshot()[("user-1", "Company")].ema_low_score
    # Bootstrap: 1.0 + (-0.05) = 0.95.
    assert pytest.approx(after_match, rel=1e-9) == 0.95

    # User re-labels NOT_MATCH.
    await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.NOT_MATCH
    )
    after_relabel = _reset_merge_learning.snapshot()[
        ("user-1", "Company")
    ].ema_low_score
    # delta = +0.05 - (-0.05) = +0.10. 0.95 + 0.10 = 1.05 →
    # clamped at 1.0.
    assert pytest.approx(after_relabel, rel=1e-9) == 1.0, (
        f"Match→not_match transition didn't apply the full swing. "
        f"Got ema={after_relabel}. Expected 1.0 (0.95 + 2*0.05 "
        "clamped at 1.0). The transition delta must include "
        "BOTH the undo of the prior match contribution and the "
        "new not_match contribution."
    )


@pytest.mark.asyncio
async def test_match_to_skip_undoes_match_contribution(
    memory_service, _reset_merge_learning
):
    """User downgrades a MATCH label to SKIP. The match
    contribution is undone (delta = 0 - match_nudge = +nudge),
    moving the threshold back toward neutral.

    Pre-fix: SKIP was a hard no-op regardless of prior status,
    so a MATCH followed by SKIP left the threshold lowered
    forever. Post-fix: SKIP correctly undoes any prior labeled
    contribution."""
    pair = await memory_service.enqueue(_make_pair(entity_type="Company"))

    await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.MATCH
    )
    after_match = _reset_merge_learning.snapshot()[("user-1", "Company")].ema_low_score
    assert pytest.approx(after_match, rel=1e-9) == 0.95

    # Now skip — should undo the match contribution.
    await memory_service.label(
        pair_id=pair.id, user_id="user-1", decision=Decision.SKIP
    )
    after_skip = _reset_merge_learning.snapshot()[("user-1", "Company")].ema_low_score
    # delta = 0 - (-0.05) = +0.05. 0.95 + 0.05 = 1.00.
    assert pytest.approx(after_skip, rel=1e-9) == 1.00, (
        f"Match→skip didn't undo the prior match contribution. "
        f"Got ema={after_skip}. Pre-fix: skip was a hard no-op "
        "and the match nudge would have stuck around forever."
    )


@pytest.mark.asyncio
async def test_postgres_label_query_uses_locking_cte_for_old_status(postgres_service):
    """Postgres SQL shape pin. The pre-update status capture must
    use a LOCK-BEARING read so concurrent label submits don't
    both read old_status='pending' before either UPDATE commits.

    Reviewer-flagged P2 (commit a1775c5): the prior two-CTE
    pattern used a plain SELECT for ``old``, which gave correct
    semantics for SEQUENTIAL double-submits (idempotency) but
    leaked the same double-nudge bug at concurrent execution.
    The fix wraps the ``old`` read in ``MATERIALIZED`` +
    ``FOR UPDATE`` so PG's row-level lock serializes the two
    statements; the second-in-line re-reads the post-commit
    state and apply_pair_label sees a same-status transition.

    Pin BOTH keywords because dropping either compromises the
    contract:
      * Without FOR UPDATE: no lock, concurrent reads race.
      * Without MATERIALIZED: PG 12+ may inline the CTE in a
        way that doesn't preserve lock semantics through future
        planner changes.

    Patches db.fetchrow to inspect the query string."""
    row = {
        "id": "p-1",
        "user_id": "user-1",
        "transform_id": "tx-1",
        "node_a_id": "n-a",
        "node_b_id": "n-b",
        "entity_type": "Company",
        "node_a_canonical_key": None,
        "node_b_canonical_key": None,
        "similarity_score": None,
        "source_stage": "embedding_blocker",
        "status": "labeled_match",
        "labeled_at": "2026-05-14T01:00:00+00:00",
        "labeled_by_user_id": "user-1",
        "label_reason": None,
        "created_at": "2026-05-14T00:00:00+00:00",
        "old_status": "pending",
    }
    fetchrow_mock = AsyncMock(return_value=row)
    with patch(
        "graphora_server.services.disputed_pairs_service.db.fetchrow",
        new=fetchrow_mock,
    ):
        await postgres_service.label(
            pair_id="p-1", user_id="user-1", decision=Decision.MATCH
        )

    query = fetchrow_mock.await_args.args[0]
    # Lock-bearing read.
    assert "FOR UPDATE" in query, (
        "Label SQL must use SELECT ... FOR UPDATE in the old "
        "CTE so concurrent submits serialize on the row lock. "
        "Without it, two clients can both read old_status="
        "'pending' before either UPDATE commits and each "
        "applies a full nudge for a single human decision."
    )
    # Explicit materialization so future PG planner changes
    # can't inline the CTE in a way that drops the lock scope.
    assert "MATERIALIZED" in query, (
        "Label SQL must mark the old CTE MATERIALIZED so PG "
        "12+ doesn't inline it. Implicit inlining could "
        "theoretically reorder evaluation in a way that loses "
        "the FOR UPDATE lock contract."
    )
    # CTE-then-UPDATE-FROM pattern (vs the older two-CTE +
    # final SELECT shape) — the UPDATE references old via FROM
    # so RETURNING can surface dp.* alongside old.status.
    assert "WITH old AS" in query
    assert "UPDATE disputed_pairs dp" in query
    assert "FROM old" in query
    assert (
        "old.status AS old_status" in query
    ), "RETURNING must surface old.status alongside dp.* so the merge_learning hook receives the pre-update status."


@pytest.mark.asyncio
async def test_postgres_label_threads_old_status_to_merge_learning(
    postgres_service, _reset_merge_learning
):
    """End-to-end Postgres path: when the CTE returns a row with
    old_status=labeled_match and the user re-submits MATCH,
    apply_pair_label sees old=new=labeled_match and treats it
    as a no-op. Pin for the Postgres parity with the memory
    backend's idempotency contract."""
    # Pre-seed merge_learning with a labeled_match contribution
    # so we can verify the no-op (no further change after the
    # re-label).
    await _reset_merge_learning.apply_pair_label(
        "user-1",
        "Company",
        old_status="pending",
        new_status="labeled_match",
    )
    seeded_ema = _reset_merge_learning.snapshot()[("user-1", "Company")].ema_low_score
    assert pytest.approx(seeded_ema, rel=1e-9) == 0.95

    # Mock returns the row with old_status="labeled_match"
    # (i.e., the pair was already labeled MATCH).
    row = {
        "id": "p-1",
        "user_id": "user-1",
        "transform_id": "tx-1",
        "node_a_id": "n-a",
        "node_b_id": "n-b",
        "entity_type": "Company",
        "node_a_canonical_key": None,
        "node_b_canonical_key": None,
        "similarity_score": None,
        "source_stage": "embedding_blocker",
        "status": "labeled_match",
        "labeled_at": "2026-05-14T01:00:00+00:00",
        "labeled_by_user_id": "user-1",
        "label_reason": None,
        "created_at": "2026-05-14T00:00:00+00:00",
        "old_status": "labeled_match",  # already labeled
    }
    with patch(
        "graphora_server.services.disputed_pairs_service.db.fetchrow",
        new=AsyncMock(return_value=row),
    ):
        await postgres_service.label(
            pair_id="p-1", user_id="user-1", decision=Decision.MATCH
        )

    # The double-label should be a merge_learning no-op.
    after_relabel = _reset_merge_learning.snapshot()[
        ("user-1", "Company")
    ].ema_low_score
    assert pytest.approx(after_relabel, rel=1e-9) == seeded_ema, (
        "Postgres double-label moved the threshold. The CTE "
        "returned old_status=labeled_match and the service "
        "passed it through, but apply_pair_label still applied "
        "a delta — the no-op semantics are broken on the "
        "Postgres path."
    )
