"""B5-obs slice 2: budgets API.

Three routes under /api/v1/budgets/me:
  * GET    /api/v1/budgets/me        — current user's budget (or None)
  * PUT    /api/v1/budgets/me        — set / update the user's cap
  * DELETE /api/v1/budgets/me        — remove the cap (back to unset)
  * GET    /api/v1/budgets/me/status — current spend vs cap, with state

Tenant-scoped via auth.user_id, matching the pattern from
/decisions (commit 65fceac) and /cost (commit 34e29d7). The
read/write surface is /me only — no admin endpoint that takes a
user_id path param. RBAC (operator manages tenant budgets) lands
in Gate 6.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.config import settings
from graphora_server.services.budget_service import BudgetService
from graphora_server.utils.logger import logger

router = APIRouter(prefix=f"{settings.API_V1_STR}/budgets", tags=["Budgets"])


class BudgetWriteRequest(BaseModel):
    """Body shape for PUT /budgets/me. Pydantic's Decimal type
    accepts JSON numbers and strings; the service layer enforces
    non-negative via a CHECK constraint at the DB layer as
    defense-in-depth."""

    monthly_cap_usd: Decimal = Field(
        ...,
        ge=0,
        description="Monthly spend cap in USD. 0 blocks all spend.",
    )


class BudgetResponse(BaseModel):
    user_id: str
    monthly_cap_usd: str = Field(
        ...,
        description=(
            "Cap as a string for Decimal precision — same convention "
            "as /cost's estimated_cost_usd."
        ),
    )
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class BudgetStatusResponse(BaseModel):
    state: str = Field(..., description="One of: unset, under, near, over")
    current_spend_usd: str = Field(
        ...,
        description=(
            "Spend rolled up from llm_usage in the current period, "
            "as a string for Decimal precision."
        ),
    )
    cap_usd: Optional[str] = None
    period_start: str
    period_end: str


@router.get(
    "/me",
    description="Get the authenticated user's monthly budget cap.",
)
async def get_my_budget(
    auth: AuthContext = Depends(get_current_auth),
) -> Optional[BudgetResponse]:
    """Returns the budget row, or null when the user hasn't set
    one. Returning null (not 404) lets the dashboard render an
    "add budget" CTA without paying a 404-as-not-error round."""
    service = BudgetService()
    budget = await service.get_budget(auth.user_id)
    if budget is None:
        return None
    return BudgetResponse(
        user_id=budget.user_id,
        monthly_cap_usd=str(budget.monthly_cap_usd),
        created_at=budget.created_at,
        updated_at=budget.updated_at,
    )


@router.put(
    "/me",
    response_model=BudgetResponse,
    description="Set or update the authenticated user's monthly budget cap.",
)
async def set_my_budget(
    body: BudgetWriteRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> BudgetResponse:
    service = BudgetService()
    try:
        budget = await service.set_budget(auth.user_id, body.monthly_cap_usd)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if budget is None:
        # Dev mode (DATABASE_URL unset) — surface a clear error so
        # operators don't silently lose their write.
        raise HTTPException(
            status_code=503,
            detail=(
                "Budgets require DATABASE_URL to be configured. "
                "Cannot persist in memory-only mode."
            ),
        )
    return BudgetResponse(
        user_id=budget.user_id,
        monthly_cap_usd=str(budget.monthly_cap_usd),
        created_at=budget.created_at,
        updated_at=budget.updated_at,
    )


@router.delete(
    "/me",
    description="Remove the authenticated user's monthly budget cap.",
)
async def delete_my_budget(
    auth: AuthContext = Depends(get_current_auth),
) -> Dict[str, bool]:
    service = BudgetService()
    deleted = await service.delete_budget(auth.user_id)
    return {"deleted": deleted}


@router.get(
    "/me/status",
    response_model=BudgetStatusResponse,
    description=(
        "Current period's spend vs cap for the authenticated user, "
        "with a state label the UI / agent can use to decide whether "
        "to render an amber warning or block further spend."
    ),
)
async def get_my_budget_status(
    auth: AuthContext = Depends(get_current_auth),
) -> BudgetStatusResponse:
    service = BudgetService()
    status = await service.get_status(auth.user_id)
    return BudgetStatusResponse(
        state=status.state.value,
        current_spend_usd=str(status.current_spend_usd),
        cap_usd=str(status.cap_usd) if status.cap_usd is not None else None,
        period_start=status.period_start,
        period_end=status.period_end,
    )


# ---- Helpers used by preflight at the transform endpoints ----------------


async def enforce_budget_preflight(user_id: str) -> None:
    """Helper called at the top of transform-upload endpoints.

    Raises HTTPException(402) when the user is over budget. Returns
    None on allow (so the call site stays a one-liner ``await
    enforce_budget_preflight(user_id)``).

    Centralizing the check here means the three transform-upload
    routes (ontology-supplied, auto-schema, schemaless) all
    inherit the same enforcement contract — a future fourth
    upload route just needs to call this helper, no copy/paste."""
    service = BudgetService()
    result = await service.check_can_proceed(user_id)
    if not result.allowed:
        logger.info(
            "Blocking transform start for user %s: %s",
            user_id,
            result.reason,
        )
        raise HTTPException(
            status_code=402,
            detail={
                "error": "budget_exceeded",
                "reason": result.reason,
                "state": result.state.value,
                "current_spend_usd": str(result.current_spend_usd),
                "cap_usd": (
                    str(result.cap_usd) if result.cap_usd is not None else None
                ),
            },
        )


# Re-export for tests + other modules that want to construct
# without importing the implementation module directly.
__all__ = [
    "router",
    "enforce_budget_preflight",
    "BudgetWriteRequest",
    "BudgetResponse",
    "BudgetStatusResponse",
]
