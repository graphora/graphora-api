"""Unit tests for BudgetService (B5-obs slice 2).

The service has three responsibilities pinned independently:
  * CRUD on the user_budgets table (PK upserts, deletes).
  * Status aggregation — current spend rolled up from llm_usage
    in the UTC-calendar-month window, plus the under/near/over
    state label.
  * Preflight enforcement — check_can_proceed returns
    allowed=False when state==OVER, allowed=True otherwise.

Tests force DATABASE_URL on so the service hits the real code
path (not the dev-mode short-circuit), then mock db.fetchrow /
db.fetch / db.execute to assert SQL shape + result handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from graphora_server.config import settings
from graphora_server.services.budget_service import (
    BudgetCheckResult,
    BudgetReadError,
    BudgetService,
    BudgetState,
    BudgetStatus,
    _current_month_window,
)


@pytest.fixture(autouse=True)
def _ensure_db_enabled(monkeypatch):
    """Service short-circuits when neither DATABASE_URL nor
    resolved_database_url is set. Force the enabled path so we
    exercise the real SQL shape under mocked db helpers."""
    monkeypatch.setattr(
        settings, "DATABASE_URL", "postgresql://test:test@localhost/test"
    )


@pytest.fixture
def service():
    return BudgetService()


# ---- _current_month_window --------------------------------------------------


def test_month_window_is_inclusive_start_exclusive_end():
    """Start is the first instant of the month, end is the first
    instant of the next month — exclusive bound so timestamps at
    exactly month-end midnight don't double-count."""
    now = datetime(2026, 5, 15, 12, 30, tzinfo=timezone.utc)
    start, end = _current_month_window(now)
    assert start == datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)


def test_month_window_handles_december_year_rollover():
    """A naive +30/+31 day calculation breaks across year
    boundaries. The +28/+4 trick (snap to day 28, advance 4 days,
    snap to day 1) is month-length-agnostic — pin the rollover so
    a future refactor can't regress."""
    now = datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)
    start, end = _current_month_window(now)
    assert start == datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2027, 1, 1, 0, 0, tzinfo=timezone.utc)


# ---- CRUD ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_budget_returns_none_when_not_set(service):
    """An unset budget is a legitimate state, not an error. The
    /me endpoint returns null in this case so the dashboard can
    render an 'add budget' CTA."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(return_value=None),
    ):
        result = await service.get_budget("user-1")
    assert result is None


@pytest.mark.asyncio
async def test_get_budget_parses_row_into_typed_dataclass(service):
    """Postgres returns Decimal for NUMERIC(10,4); the dataclass
    preserves that without lossy float conversion. Timestamps
    serialize to ISO8601 strings so the API response is
    JSON-ready."""
    row = {
        "user_id": "user-1",
        "monthly_cap_usd": Decimal("25.0000"),
        "created_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 5, 10, tzinfo=timezone.utc),
    }
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(return_value=row),
    ):
        result = await service.get_budget("user-1")
    assert result is not None
    assert result.user_id == "user-1"
    assert result.monthly_cap_usd == Decimal("25.0000")
    assert result.created_at == "2026-05-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_set_budget_upserts_via_on_conflict(service):
    """Pin the SQL shape: INSERT ... ON CONFLICT (user_id) DO
    UPDATE. A naive INSERT would 23505 on the second set; a
    naive UPDATE-only would silently fail when the user hadn't
    set a budget yet."""
    returning_row = {
        "user_id": "user-1",
        "monthly_cap_usd": Decimal("50.0000"),
        "created_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 5, 10, tzinfo=timezone.utc),
    }
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(return_value=returning_row),
    ) as mock_fetchrow:
        result = await service.set_budget("user-1", Decimal("50"))

    query = mock_fetchrow.await_args.args[0]
    assert "INSERT INTO user_budgets" in query
    assert "ON CONFLICT (user_id) DO UPDATE" in query
    assert "RETURNING" in query
    assert mock_fetchrow.await_args.args[1:] == ("user-1", Decimal("50"))
    assert result is not None
    assert result.monthly_cap_usd == Decimal("50.0000")


@pytest.mark.asyncio
async def test_set_budget_rejects_negative_cap(service):
    """Defense in depth: the DB CHECK constraint enforces
    non-negative, and Pydantic at the API layer rejects negative
    request bodies, but the service raises ValueError if a caller
    bypasses both. Cheap belt-and-braces."""
    with pytest.raises(ValueError, match="non-negative"):
        await service.set_budget("user-1", Decimal("-1"))


@pytest.mark.asyncio
async def test_delete_budget_returns_true_when_row_existed(service):
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(return_value={"user_id": "user-1"}),
    ):
        assert await service.delete_budget("user-1") is True


@pytest.mark.asyncio
async def test_delete_budget_returns_false_when_nothing_to_delete(service):
    """Idempotent: deleting a budget the user doesn't have isn't
    an error. Returning False (not 404 / raising) keeps the
    endpoint simple — operators can re-issue DELETE without
    branching on prior state."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(return_value=None),
    ):
        assert await service.delete_budget("user-1") is False


# ---- get_status ------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_unset_when_no_budget_row(service):
    """State==UNSET surfaces in the response so the UI can
    render an 'add budget' CTA. Distinct from UNDER because UNDER
    requires a cap to compare against."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(side_effect=[None, {"total": Decimal("5.00")}]),
    ):
        status = await service.get_status("user-1")
    assert status.state == BudgetState.UNSET
    # Even unset, current spend is rolled up so the UI shows
    # 'currently spending $5' alongside the CTA.
    assert status.current_spend_usd == Decimal("5.00")
    assert status.cap_usd is None


@pytest.mark.asyncio
async def test_status_under_when_below_80_percent(service):
    """$5 spent against a $100 cap = 5% — well under the
    _NEAR_THRESHOLD = 80%."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(
            side_effect=[
                {
                    "user_id": "user-1",
                    "monthly_cap_usd": Decimal("100"),
                    "created_at": None,
                    "updated_at": None,
                },
                {"total": Decimal("5.00")},
            ]
        ),
    ):
        status = await service.get_status("user-1")
    assert status.state == BudgetState.UNDER
    assert status.cap_usd == Decimal("100")


@pytest.mark.asyncio
async def test_status_near_when_at_or_above_80_percent(service):
    """$80 of $100 = 80% — right on the threshold, should be
    NEAR (the dashboard renders an amber badge)."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(
            side_effect=[
                {
                    "user_id": "user-1",
                    "monthly_cap_usd": Decimal("100"),
                    "created_at": None,
                    "updated_at": None,
                },
                {"total": Decimal("80")},
            ]
        ),
    ):
        status = await service.get_status("user-1")
    assert status.state == BudgetState.NEAR


@pytest.mark.asyncio
async def test_status_over_when_spend_meets_or_exceeds_cap(service):
    """$100 spent against $100 cap = OVER (>= cap, not just >).
    Pin the inclusive boundary so an exactly-at-cap user is
    blocked on the next call."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(
            side_effect=[
                {
                    "user_id": "user-1",
                    "monthly_cap_usd": Decimal("100"),
                    "created_at": None,
                    "updated_at": None,
                },
                {"total": Decimal("100")},
            ]
        ),
    ):
        status = await service.get_status("user-1")
    assert status.state == BudgetState.OVER


@pytest.mark.asyncio
async def test_status_zero_cap_is_always_over(service):
    """cap=0 is a deliberate kill-switch: "block all spend on
    this account". The status must report OVER regardless of
    whether the user has any LLM rows yet — otherwise a brand-new
    user with cap=0 and zero spend would slip through."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(
            side_effect=[
                {
                    "user_id": "user-1",
                    "monthly_cap_usd": Decimal("0"),
                    "created_at": None,
                    "updated_at": None,
                },
                {"total": Decimal("0")},
            ]
        ),
    ):
        status = await service.get_status("user-1")
    assert status.state == BudgetState.OVER


# ---- check_can_proceed (preflight) -----------------------------------------


@pytest.mark.asyncio
async def test_check_allows_when_unset(service):
    """Opt-in semantics: users without a budget are allowed. The
    legacy install (no budget set) keeps working without forcing
    operators to set caps before the migration applies."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(side_effect=[None, {"total": Decimal("0")}]),
    ):
        result = await service.check_can_proceed("user-1")
    assert result.allowed is True
    assert result.state == BudgetState.UNSET


@pytest.mark.asyncio
async def test_check_blocks_when_over(service):
    """The single binary enforcement gate. allowed=False AND a
    reason string the endpoint surfaces in the 402 body."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(
            side_effect=[
                {
                    "user_id": "user-1",
                    "monthly_cap_usd": Decimal("10"),
                    "created_at": None,
                    "updated_at": None,
                },
                {"total": Decimal("15")},
            ]
        ),
    ):
        result = await service.check_can_proceed("user-1")
    assert result.allowed is False
    assert result.state == BudgetState.OVER
    assert result.reason is not None
    assert "exceeded" in result.reason.lower()


@pytest.mark.asyncio
async def test_check_allows_when_near_but_not_over(service):
    """NEAR is a warning state, not a block state. The agent
    rendering layer may want to surface the warning to the user,
    but transforms still go through."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(
            side_effect=[
                {
                    "user_id": "user-1",
                    "monthly_cap_usd": Decimal("100"),
                    "created_at": None,
                    "updated_at": None,
                },
                {"total": Decimal("85")},
            ]
        ),
    ):
        result = await service.check_can_proceed("user-1")
    assert result.allowed is True
    assert result.state == BudgetState.NEAR


# ---- SQL window pin --------------------------------------------------------


@pytest.mark.asyncio
async def test_get_current_spend_uses_month_window_with_user_filter(service):
    """Pin both the user_id scoping AND the half-open month window
    (>= start, < end). The < end is load-bearing: without it, the
    last second of the month gets counted twice when month-roll
    timestamps land at exactly midnight."""
    captured = {}

    async def capture_fetchrow(query, *args):
        captured["query"] = query
        captured["args"] = args
        return {"total": Decimal("12.34")}

    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(side_effect=capture_fetchrow),
    ):
        spend = await service.get_current_spend("user-1")

    assert spend == Decimal("12.34")
    query = captured["query"]
    assert "WHERE user_id = %s" in query
    assert "created_at >= %s" in query
    assert "created_at < %s" in query
    # First arg = user_id; positional args follow.
    assert captured["args"][0] == "user-1"
    # End > start.
    assert captured["args"][2] > captured["args"][1]


# ---- Dev-mode short-circuit ------------------------------------------------


@pytest.mark.asyncio
async def test_check_allows_when_database_disabled(monkeypatch):
    """Dev mode (no DATABASE_URL) skips enforcement entirely.
    Setting up a Postgres for ``make dev`` shouldn't be a
    requirement for the rest of the API to work — budget reads
    return None, status is UNSET, check_can_proceed allows."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    service = BudgetService()
    result = await service.check_can_proceed("user-1")
    assert result.allowed is True
    assert result.state == BudgetState.UNSET


# ---- Fail-closed on DB errors (P1 reviewer fix on commit 535f56d) ---------


@pytest.mark.asyncio
async def test_get_budget_raises_budget_read_error_on_db_failure(service):
    """Reviewer-flagged P1: the previous behaviour swallowed every
    DB exception and returned None, which the status path then
    treated as UNSET → allowed → silent enforcement bypass.

    Pin the fail-closed contract: a configured-but-failing DB read
    propagates BudgetReadError so the API layer can translate to
    HTTP 503. The alternative — return None — would re-introduce
    the silent-bypass bug."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(side_effect=RuntimeError("connection refused")),
    ):
        with pytest.raises(BudgetReadError, match="connection refused"):
            await service.get_budget("user-1")


@pytest.mark.asyncio
async def test_get_current_spend_raises_budget_read_error_on_db_failure(service):
    """Mirror pin for the spend aggregation path. Pre-fix, this
    returned Decimal(0), which collapsed against any non-zero cap
    to spend < cap → UNDER → allowed. Post-fix it propagates so
    the gate fails closed."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(side_effect=RuntimeError("relation llm_usage does not exist")),
    ):
        with pytest.raises(BudgetReadError, match="llm_usage"):
            await service.get_current_spend("user-1")


@pytest.mark.asyncio
async def test_check_can_proceed_fails_closed_when_budget_read_fails(service):
    """End-to-end pin via the enforcement helper. The pre-fix
    pattern was particularly nasty here: get_budget swallowed →
    returned None → state==UNSET → allowed=True. A degraded
    Postgres with a configured DATABASE_URL silently disabled
    every budget cap.

    Post-fix the BudgetReadError propagates all the way out of
    check_can_proceed, and the API helper translates to 503."""
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(side_effect=RuntimeError("postgres unreachable")),
    ):
        with pytest.raises(BudgetReadError):
            await service.check_can_proceed("user-1")


@pytest.mark.asyncio
async def test_check_can_proceed_fails_closed_when_spend_read_fails(service):
    """The second leg of the read: the budget row was fetched
    successfully, but the spend rollup failed. The composite
    get_status / check_can_proceed flow must still propagate so
    we don't accidentally let the user through with cap-known +
    spend-unknown."""
    budget_row = {
        "user_id": "user-1",
        "monthly_cap_usd": Decimal("100"),
        "created_at": None,
        "updated_at": None,
    }

    # First fetchrow returns the budget; second raises on the
    # spend rollup. Pre-fix the spend rollup would have been
    # swallowed → 0 → allowed=True (under any positive cap).
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(
            side_effect=[budget_row, RuntimeError("network partition")],
        ),
    ):
        with pytest.raises(BudgetReadError):
            await service.check_can_proceed("user-1")


@pytest.mark.asyncio
async def test_check_allows_unset_user_without_reading_spend(service):
    """Reviewer-flagged P2 on commit 5b78f85. The fail-closed
    spend read meant an unset-budget user got a 503 if llm_usage
    was unavailable — breaking the stated opt-in semantics
    ("legacy users keep working").

    Pin the short-circuit contract: when budget is None,
    check_can_proceed allows WITHOUT touching get_current_spend.
    A capped user's spend read can fail closed; an unset user
    shouldn't see that read at all.

    Two assertions: (1) the result is allowed=UNSET, (2) the
    spend read mock was never invoked. The second assertion is
    the load-bearing one — a future refactor that re-introduces
    the eager spend roll-up would fail loud here even if the
    return value happened to look right."""
    spend_mock = AsyncMock(
        side_effect=AssertionError(
            "get_current_spend must not be called when budget is unset"
        )
    )
    with (
        patch(
            "graphora_server.services.budget_service.db.fetchrow",
            new=AsyncMock(return_value=None),  # No budget row.
        ),
        patch.object(service, "get_current_spend", new=spend_mock),
    ):
        result = await service.check_can_proceed("user-no-budget")

    assert result.allowed is True
    assert result.state == BudgetState.UNSET
    spend_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_allows_unset_user_even_when_spend_read_would_fail(
    service,
):
    """End-to-end version of the opt-in pin: even with llm_usage
    actively failing (the exact scenario the reviewer flagged),
    an unset user passes through. The short-circuit happens
    BEFORE the spend read, so the read never gets a chance to
    raise."""
    # First fetchrow returns None (no budget); second WOULD
    # raise on the spend read but must never be called.
    fetchrow_mock = AsyncMock(
        side_effect=[
            None,  # get_budget — unset
            RuntimeError("llm_usage unavailable — must not be reached"),
        ]
    )
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=fetchrow_mock,
    ):
        result = await service.check_can_proceed("user-1")

    assert result.allowed is True
    assert result.state == BudgetState.UNSET
    # Only ONE fetchrow call — the budget lookup. The spend
    # lookup was never reached.
    assert fetchrow_mock.await_count == 1


@pytest.mark.asyncio
async def test_dev_mode_still_allows_through_when_db_is_unconfigured(
    monkeypatch,
):
    """The fail-closed change must not affect dev mode. When
    DATABASE_URL is intentionally unset (``_enabled=False``), the
    service short-circuits before touching the DB — no
    BudgetReadError, no 503. Operators running ``make dev``
    without a Postgres get the same behaviour as before the fix."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    dev_service = BudgetService()

    # Even if we wire a raising mock, dev mode shouldn't reach it.
    with patch(
        "graphora_server.services.budget_service.db.fetchrow",
        new=AsyncMock(side_effect=RuntimeError("should not be called")),
    ):
        # All three reads succeed without raising.
        assert await dev_service.get_budget("user-1") is None
        assert await dev_service.get_current_spend("user-1") == Decimal("0")
        result = await dev_service.check_can_proceed("user-1")
        assert result.allowed is True
        assert result.state == BudgetState.UNSET


# Sanity that the dataclasses re-export cleanly for downstream use.
def test_dataclasses_exported():
    """If the imports landed wrong this module wouldn't load —
    but a no-op test pins that the public shape stays stable."""
    assert BudgetService
    assert BudgetState.UNSET.value == "unset"
    assert BudgetStatus
    assert BudgetCheckResult
    assert BudgetReadError
