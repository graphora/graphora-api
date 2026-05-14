"""B2-active backend slice A: disputed-pairs queue API.

Four routes under /api/v1/disputed-pairs:
  * GET    /api/v1/disputed-pairs              — list pending queue
                                                  (optional transform_id filter)
  * GET    /api/v1/disputed-pairs/{pair_id}    — single pair
  * POST   /api/v1/disputed-pairs/{pair_id}/label — apply a label
  * GET    /api/v1/disputed-pairs/transform/{transform_id} — per-transform review

Tenant-scoped via auth.user_id, mirroring /budgets and /decisions
patterns. The label POST accepts a closed-set decision enum
(match / not_match / skip) so the wire shape stays stable across
the agent / dashboard / CLI surfaces.

Slice A scope: data surface only. The hook that POPULATES the
queue from B2-er's gray-zone candidates lands in slice B; the
feedback loop into Splink weight updates lands in slice C."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from graphora_server.auth import AuthContext, get_current_auth
from graphora_server.config import settings
from graphora_server.services.disputed_pairs_service import (
    Decision,
    DisputedPair,
    DisputedPairsService,
)
from graphora_server.utils.logger import logger

router = APIRouter(
    prefix=f"{settings.API_V1_STR}/disputed-pairs",
    tags=["Disputed Pairs"],
)


class DisputedPairResponse(BaseModel):
    """Wire shape for a single disputed pair. Decimals serialize
    as strings for precision (same convention as /cost's
    estimated_cost_usd)."""

    id: str
    user_id: str
    transform_id: str
    node_a_id: str
    node_b_id: str
    entity_type: str
    node_a_canonical_key: Optional[str] = None
    node_b_canonical_key: Optional[str] = None
    similarity_score: Optional[str] = None
    source_stage: str = Field(
        ...,
        description=(
            "One of: property_blocker, embedding_blocker, " "splink_blocker, llm_review"
        ),
    )
    status: str = Field(
        ...,
        description=("One of: pending, labeled_match, labeled_not_match, skipped"),
    )
    labeled_at: Optional[str] = None
    labeled_by_user_id: Optional[str] = None
    label_reason: Optional[str] = None
    created_at: Optional[str] = None


class LabelRequest(BaseModel):
    """Body shape for POST /label. Closed-set decision enum at
    the wire layer too — Pydantic enforces it before the service
    sees the value."""

    decision: str = Field(
        ...,
        description="One of: match, not_match, skip",
    )
    reason: Optional[str] = Field(
        None,
        description="Optional free-text reason the operator/agent supplies.",
    )


def _pair_to_response(pair: DisputedPair) -> DisputedPairResponse:
    return DisputedPairResponse(
        id=pair.id,
        user_id=pair.user_id,
        transform_id=pair.transform_id,
        node_a_id=pair.node_a_id,
        node_b_id=pair.node_b_id,
        entity_type=pair.entity_type,
        node_a_canonical_key=pair.node_a_canonical_key,
        node_b_canonical_key=pair.node_b_canonical_key,
        similarity_score=(
            str(pair.similarity_score) if pair.similarity_score is not None else None
        ),
        source_stage=pair.source_stage.value,
        status=pair.status.value,
        labeled_at=pair.labeled_at,
        labeled_by_user_id=pair.labeled_by_user_id,
        label_reason=pair.label_reason,
        created_at=pair.created_at,
    )


@router.get(
    "",
    description=(
        "Pending disputed pairs for the authenticated user, "
        "newest first. Optional transform_id filter restricts to "
        "a single run."
    ),
)
async def list_pending_pairs(
    transform_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth: AuthContext = Depends(get_current_auth),
) -> List[DisputedPairResponse]:
    service = DisputedPairsService()
    try:
        pairs = await service.list_pending(
            user_id=auth.user_id,
            transform_id=transform_id,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error(
            "Error listing pending disputed pairs for user %s: %s",
            auth.user_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error listing disputed pairs: {exc}",
        )
    return [_pair_to_response(p) for p in pairs]


@router.get(
    "/transform/{transform_id}",
    description=(
        "All disputed pairs (any status) for a specific transform "
        "owned by the authenticated user. Used by the per-transform "
        "review view."
    ),
)
async def list_pairs_for_transform(
    transform_id: str,
    auth: AuthContext = Depends(get_current_auth),
) -> List[DisputedPairResponse]:
    service = DisputedPairsService()
    try:
        pairs = await service.list_for_transform(
            user_id=auth.user_id, transform_id=transform_id
        )
    except Exception as exc:
        logger.error(
            "Error listing disputed pairs for user %s transform %s: %s",
            auth.user_id,
            transform_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error listing disputed pairs: {exc}",
        )
    return [_pair_to_response(p) for p in pairs]


@router.get(
    "/{pair_id}",
    description="Fetch a single disputed pair.",
)
async def get_pair(
    pair_id: str,
    auth: AuthContext = Depends(get_current_auth),
) -> DisputedPairResponse:
    service = DisputedPairsService()
    try:
        pair = await service.get(pair_id, auth.user_id)
    except Exception as exc:
        logger.error(
            "Error fetching disputed pair %s for user %s: %s",
            pair_id,
            auth.user_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching disputed pair: {exc}",
        )
    if pair is None:
        # 404 covers both "doesn't exist" and "belongs to another
        # tenant" — never leak the distinction.
        raise HTTPException(status_code=404, detail="Disputed pair not found")
    return _pair_to_response(pair)


@router.post(
    "/{pair_id}/label",
    description=(
        "Apply a label to a disputed pair. Decision is one of "
        "match / not_match / skip. Re-labeling overwrites prior "
        "labels (supports the 'I changed my mind' UX without a "
        "separate undo endpoint)."
    ),
)
async def label_pair(
    pair_id: str,
    body: LabelRequest,
    auth: AuthContext = Depends(get_current_auth),
) -> DisputedPairResponse:
    # Validate the decision string against the closed set BEFORE
    # touching the service. Pydantic could enforce this via an
    # Enum field, but staying as ``str`` keeps the wire JSON
    # plain — agents posting `{"decision": "match"}` are common,
    # and a Pydantic Enum field would either reject or coerce
    # surprisingly depending on the version.
    try:
        decision = Decision(body.decision)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid decision: {body.decision!r}. Must be one "
                "of: match, not_match, skip."
            ),
        )

    service = DisputedPairsService()
    try:
        pair = await service.label(
            pair_id=pair_id,
            user_id=auth.user_id,
            decision=decision,
            reason=body.reason,
        )
    except Exception as exc:
        logger.error(
            "Error labeling disputed pair %s for user %s: %s",
            pair_id,
            auth.user_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error labeling disputed pair: {exc}",
        )
    if pair is None:
        raise HTTPException(status_code=404, detail="Disputed pair not found")
    return _pair_to_response(pair)


__all__ = [
    "router",
    "DisputedPairResponse",
    "LabelRequest",
]
