"""B5-obs slice 2: project budget caps.

Closes the loop on the cost-observability story from slice 1: the
/cost endpoint shows operators what they spent; this service lets
them set a cap and have the system enforce it before kicking off
expensive operations.

Three primitives:
  * Storage: one budget per user (PK on user_id), expressed as a
    monthly cap in USD. Period semantics are UTC-calendar-month —
    the most common operator model and trivial to compute against
    the existing llm_usage.created_at column.
  * Reads: get_budget, get_status. Status includes the
    current-period spend rolled up from llm_usage and a state
    label so the rendering layer can show "under" / "near" / "over"
    without recomputing.
  * Enforcement: check_can_proceed at transform-start. Returns a
    BudgetCheckResult that the API endpoint translates to 402 on
    block, 200 on allow. Centralizing the check here means every
    transform-upload route inherits the same enforcement contract.

Out of scope (future B5-obs slices):
  * Per-document / per-project budgets (this service is per-user).
  * Multiple periods (daily / weekly / quarterly).
  * Soft-cap warnings via email / webhook (slice 3 territory).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional

from graphora_server.config import settings
from graphora_server.db import postgres as db

logger = logging.getLogger(__name__)


class BudgetState(str, Enum):
    """Three-valued state the rendering layer / preflight gate
    consume. ``unset`` is distinct from ``under`` because operators
    may want to render a CTA ("set a budget") rather than
    "currently $0 / $0 spent"."""

    UNSET = "unset"
    UNDER = "under"
    NEAR = "near"
    OVER = "over"


class BudgetReadError(RuntimeError):
    """Raised when a budget read fails against a configured database.

    Reviewer-flagged on commit 535f56d (B5-obs slice 2 P1): the
    previous behaviour swallowed every DB error and treated it as
    "no budget set" / "zero spend", which silently disabled
    enforcement when migration 16 was missing or Postgres was
    degraded. The enforcement gate is a correctness boundary and
    MUST fail closed when it can't tell — that's what this
    exception signals to the API layer (which translates it to
    HTTP 503).

    Dev mode (``_enabled=False``) does NOT raise — it short-
    circuits before touching the DB. Only the configured-but-
    failing path raises."""


# 80% of cap triggers the ``near`` state — early-warning headroom
# the rendering layer uses to render an amber badge before the
# operator hits the hard wall.
_NEAR_THRESHOLD = Decimal("0.8")


@dataclass
class Budget:
    user_id: str
    monthly_cap_usd: Decimal
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class BudgetStatus:
    state: BudgetState
    current_spend_usd: Decimal
    cap_usd: Optional[Decimal]
    period_start: str
    period_end: str


@dataclass
class BudgetCheckResult:
    allowed: bool
    state: BudgetState
    current_spend_usd: Decimal
    cap_usd: Optional[Decimal]
    reason: Optional[str] = None


def _current_month_window(now: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """Return [start_of_month_utc, start_of_next_month_utc) as the
    aggregation window. End is exclusive so the WHERE clause uses
    ``< end`` and avoids the off-by-one ambiguity of BETWEEN with
    timestamps near midnight."""
    now = now or datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # First-of-next-month: add 4 days to the 28th, then snap to day=1.
    # Works for every month length without month-specific branching.
    end_seed = start.replace(day=28) + timedelta(days=4)
    end = end_seed.replace(day=1)
    return start, end


class BudgetService:
    """Per-user monthly budget caps backed by Postgres.

    The Decision Log and Entity Ledger use a dual backend
    (Postgres + in-memory dict) for zero-config dev mode. Budgets
    are deliberately Postgres-only: a budget that doesn't survive
    a process restart isn't a meaningful cap. Dev/test mode skips
    enforcement entirely by returning "unset" + allowed=True for
    every call.
    """

    def __init__(self) -> None:
        self._enabled = bool(settings.DATABASE_URL or settings.resolved_database_url)

    # Reads ----------------------------------------------------------------

    async def get_budget(self, user_id: str) -> Optional[Budget]:
        """Fetch this user's budget, or None when they haven't
        set one.

        Raises BudgetReadError when the DB is configured but the
        read fails — see class docstring on the fail-closed
        invariant. Dev mode (``_enabled=False``) short-circuits
        before touching the DB and returns None instead."""
        if not self._enabled or not user_id:
            return None
        try:
            row = await db.fetchrow(
                """
                SELECT user_id, monthly_cap_usd, created_at, updated_at
                FROM user_budgets
                WHERE user_id = %s
                """,
                user_id,
            )
        except Exception as exc:
            logger.error("Failed to fetch budget for user %s: %s", user_id, exc)
            raise BudgetReadError(
                f"Budget DB read failed for user {user_id}: {exc}"
            ) from exc
        if not row:
            return None
        return _row_to_budget(row)

    async def get_current_spend(self, user_id: str) -> Decimal:
        """Sum of estimated_cost_usd across this user's llm_usage
        rows in the current UTC-calendar-month window. Unpriced
        rows (NULL estimated_cost_usd) contribute zero — they're
        unbillable but tracked elsewhere via models_used.

        Raises BudgetReadError when the DB is configured but the
        read fails — same fail-closed invariant as get_budget."""
        if not self._enabled or not user_id:
            return Decimal("0")
        start, end = _current_month_window()
        try:
            row = await db.fetchrow(
                """
                SELECT COALESCE(SUM(estimated_cost_usd), 0) AS total
                FROM llm_usage
                WHERE user_id = %s
                  AND created_at >= %s
                  AND created_at < %s
                """,
                user_id,
                start,
                end,
            )
        except Exception as exc:
            logger.error(
                "Failed to compute current spend for user %s: %s", user_id, exc
            )
            raise BudgetReadError(
                f"llm_usage read failed for user {user_id}: {exc}"
            ) from exc
        if not row or row.get("total") is None:
            return Decimal("0")
        return Decimal(str(row["total"]))

    async def get_status(self, user_id: str) -> BudgetStatus:
        """Composite read used by the agent / dashboard surface:
        current spend rolled up from llm_usage, plus a state label
        for rendering."""
        start, end = _current_month_window()
        period_start = start.isoformat()
        period_end = end.isoformat()

        budget = await self.get_budget(user_id)
        spend = await self.get_current_spend(user_id)

        if budget is None:
            return BudgetStatus(
                state=BudgetState.UNSET,
                current_spend_usd=spend,
                cap_usd=None,
                period_start=period_start,
                period_end=period_end,
            )

        cap = budget.monthly_cap_usd
        # cap=0 is a deliberate "block everything". Any spend >= 0
        # against a zero cap is OVER — that branch falls through
        # the next check (spend >= cap is always True when cap=0).
        if spend >= cap:
            state = BudgetState.OVER
        elif spend / cap >= _NEAR_THRESHOLD:
            state = BudgetState.NEAR
        else:
            state = BudgetState.UNDER

        return BudgetStatus(
            state=state,
            current_spend_usd=spend,
            cap_usd=cap,
            period_start=period_start,
            period_end=period_end,
        )

    # Writes ---------------------------------------------------------------

    async def set_budget(
        self, user_id: str, monthly_cap_usd: Decimal
    ) -> Optional[Budget]:
        """Upsert the user's monthly cap. Returns the resulting
        row, or None when the DB isn't configured (dev mode)."""
        if not self._enabled:
            return None
        if monthly_cap_usd < 0:
            raise ValueError("monthly_cap_usd must be non-negative")
        try:
            row = await db.fetchrow(
                """
                INSERT INTO user_budgets (user_id, monthly_cap_usd)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET monthly_cap_usd = EXCLUDED.monthly_cap_usd,
                    updated_at = NOW()
                RETURNING user_id, monthly_cap_usd, created_at, updated_at
                """,
                user_id,
                monthly_cap_usd,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to set budget for user %s: %s", user_id, exc)
            raise
        return _row_to_budget(row) if row else None

    async def delete_budget(self, user_id: str) -> bool:
        """Remove the user's budget. Returns True when a row was
        deleted, False when there was nothing to remove."""
        if not self._enabled:
            return False
        try:
            row = await db.fetchrow(
                "DELETE FROM user_budgets WHERE user_id = %s RETURNING user_id",
                user_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to delete budget for user %s: %s", user_id, exc)
            return False
        return row is not None

    # Enforcement ----------------------------------------------------------

    async def check_can_proceed(self, user_id: str) -> BudgetCheckResult:
        """Preflight check called at transform-start. Allows when:
          * No budget configured (UNSET — opt-in semantics; legacy
            users keep working).
          * DB isn't configured (dev mode — no enforcement).
          * Current spend < cap.

        Blocks when current spend >= cap. The endpoint translates
        a blocked result into a 402 Payment Required."""
        status = await self.get_status(user_id)

        if status.state == BudgetState.OVER:
            cap_str = f"${status.cap_usd}" if status.cap_usd is not None else "unset"
            spend_str = f"${status.current_spend_usd}"
            return BudgetCheckResult(
                allowed=False,
                state=status.state,
                current_spend_usd=status.current_spend_usd,
                cap_usd=status.cap_usd,
                reason=(
                    f"Monthly budget exceeded: spent {spend_str} of "
                    f"{cap_str} cap. Increase or remove the cap to "
                    f"continue."
                ),
            )
        return BudgetCheckResult(
            allowed=True,
            state=status.state,
            current_spend_usd=status.current_spend_usd,
            cap_usd=status.cap_usd,
            reason=None,
        )


def _row_to_budget(row: Dict[str, Any]) -> Budget:
    """Convert a Postgres row dict into a typed Budget. Decimals
    pass through unchanged; timestamps serialize to ISO strings so
    the API response is JSON-ready without ad-hoc conversion."""
    return Budget(
        user_id=row["user_id"],
        monthly_cap_usd=Decimal(str(row["monthly_cap_usd"])),
        created_at=_iso_or_none(row.get("created_at")),
        updated_at=_iso_or_none(row.get("updated_at")),
    )


def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
