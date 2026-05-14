"""B2-active backend slice A: foundation for the disputed-pairs queue.

Gate 4 exit signal: ER's disputed-pairs queue has > 0 labels
applied by a dogfooding user, and Splink weights have updated.
This slice lands the FOUNDATION — the data model, service CRUD,
API + MCP surface — so that:

  * Slice B can wire the write hook into ``_compare_and_merge_nodes``
    (or wherever B2-er surfaces uncertain pairs) to populate the
    queue.
  * Slice C can wire labels back to Splink weight updates.

Architecture mirrors BudgetService (commit 535f56d) and
DecisionLogService (commit 8cbc76b): dual backend (Postgres when
DATABASE_URL is set, in-memory dict otherwise for dev mode),
tenant-scoped via user_id on every read/write, closed-set enums
for status / source_stage / decision.

The queue is conceptually a TODO list of ER decisions the
pipeline wasn't confident enough to auto-apply. Each row carries:
  * A candidate pair (node_a_id, node_b_id) within a transform
  * The stage that surfaced them (property / embedding / splink /
    llm_review)
  * An optional similarity score from that stage
  * A status that transitions pending → {labeled_match,
    labeled_not_match, skipped} when an operator/agent labels it

Operators consume via the Active Learning UI (graphora-fe);
agents via the MCP tools. Both go through the REST endpoint —
MCP stays a pure HTTP client (architectural rule reinforced
across the B0/B5 commits)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from graphora_server.config import settings
from graphora_server.db import postgres as db
from graphora_server.services.merge.learning import merge_learning_service

logger = logging.getLogger(__name__)


class SourceStage(str, Enum):
    """The B2-er stage that surfaced this candidate pair as
    needing review. Closed set — extending requires both the
    enum and the DB CHECK constraint (migration 17). Forcing a
    code change rather than letting arbitrary strings land
    keeps the dashboard / agent rendering layer stable."""

    PROPERTY_BLOCKER = "property_blocker"
    EMBEDDING_BLOCKER = "embedding_blocker"
    SPLINK_BLOCKER = "splink_blocker"
    LLM_REVIEW = "llm_review"


class Status(str, Enum):
    """Pair status. Closed set, mirrored by the DB CHECK
    constraint. State machine: pending → {labeled_match,
    labeled_not_match, skipped}. No transitions BACK to pending —
    a labeled pair can be re-labeled (overwriting), but never
    un-labeled to ambiguous."""

    PENDING = "pending"
    LABELED_MATCH = "labeled_match"
    LABELED_NOT_MATCH = "labeled_not_match"
    SKIPPED = "skipped"


class Decision(str, Enum):
    """Inputs from the operator/agent when labeling. The service
    translates these to the matching status values. Skip is a
    valid decision — operators may legitimately defer a pair
    they don't have enough context for, and we track that
    separately from "labeled" so the queue surface can
    distinguish "reviewed" from "skipped-for-now"."""

    MATCH = "match"
    NOT_MATCH = "not_match"
    SKIP = "skip"


_DECISION_TO_STATUS: Dict[Decision, Status] = {
    Decision.MATCH: Status.LABELED_MATCH,
    Decision.NOT_MATCH: Status.LABELED_NOT_MATCH,
    Decision.SKIP: Status.SKIPPED,
}


@dataclass
class DisputedPair:
    user_id: str
    transform_id: str
    node_a_id: str
    node_b_id: str
    entity_type: str
    source_stage: SourceStage
    status: Status = Status.PENDING
    node_a_canonical_key: Optional[str] = None
    node_b_canonical_key: Optional[str] = None
    similarity_score: Optional[Decimal] = None
    labeled_at: Optional[str] = None
    labeled_by_user_id: Optional[str] = None
    label_reason: Optional[str] = None
    id: Optional[str] = None
    created_at: Optional[str] = None


# Module-level shared store for dev mode (no DATABASE_URL).
# See DisputedPairsService.__init__ for the rationale —
# instances constructed by different request handlers share
# this list so writes survive cross-request reads. Tenant
# isolation is enforced at read time via user_id filtering, so
# the shared list is safe across tenants in dev.
_DEFAULT_MEMORY_STORE: List[DisputedPair] = []


def _canonical_pair_key(
    user_id: str,
    transform_id: str,
    source_stage: SourceStage,
    node_a_id: str,
    node_b_id: str,
) -> tuple:
    """Order-independent canonical key for dedup.

    Mirrors the Postgres expression unique index from migration 18
    (user_id, transform_id, source_stage, LEAST(node_a, node_b),
    GREATEST(node_a, node_b)). Used by the memory backend to match
    the Postgres ON CONFLICT semantics so both backends behave
    identically when the hook fires twice on the same pair (task
    retries, re-extractions)."""
    lo, hi = sorted([node_a_id, node_b_id])
    return (user_id, transform_id, source_stage, lo, hi)


def _reset_default_memory_store_for_tests() -> None:
    """Clear the dev-mode shared memory store. Test fixtures
    that exercise the production path (no ``memory_store=`` arg)
    should call this between scenarios to keep tests isolated.
    Most tests pass ``memory_store=[]`` directly and don't need
    this — it's only for direct-DB-mock tests that fall back to
    constructing a default service."""
    _DEFAULT_MEMORY_STORE.clear()


class DisputedPairsService:
    """CRUD + label-transition for the disputed-pairs queue.

    Dual backend (Postgres / in-memory) mirrors BudgetService and
    DecisionLogService. Tenant-scoped: every read/write filters
    on user_id so a malicious request with another user's
    pair_id returns nothing rather than leaking the pair.

    Failures on the Postgres path: enqueue propagates errors
    (correctness gate — losing an uncertain pair means silently
    losing an ER decision the pipeline already deferred). Reads
    propagate too — a degraded DB on the review UI should
    surface as 5xx rather than as an empty queue, otherwise
    operators would think there's nothing to review."""

    TABLE_NAME = "disputed_pairs"

    def __init__(
        self,
        memory_store: Optional[List[DisputedPair]] = None,
    ) -> None:
        self._enabled = bool(settings.DATABASE_URL or settings.resolved_database_url)
        # Reviewer-flagged on commit 26d3e89. When DATABASE_URL
        # isn't configured, every endpoint constructs its own
        # service instance (mirroring the budget / decision-log
        # pattern). If each instance got a fresh list, a pair
        # enqueued by the transform pipeline in one service
        # instance would be invisible to GET /disputed-pairs +
        # POST /label in another — the queue would always look
        # empty and labels would 404.
        #
        # Module-level ``_DEFAULT_MEMORY_STORE`` shares the dev-
        # mode list across all instances in the same process so
        # writes survive cross-request reads. Tenant filtering
        # happens at the read methods (user_id on every pair), so
        # tenants stay isolated even on the shared list.
        #
        # Tests pass ``memory_store=[]`` explicitly for isolation
        # — the per-test list bypasses the module default. That
        # contract keeps the test suite parallel-safe.
        if memory_store is not None:
            self._memory_store = memory_store
        else:
            self._memory_store = _DEFAULT_MEMORY_STORE

    # Writes -------------------------------------------------------------------

    async def enqueue(self, pair: DisputedPair) -> DisputedPair:
        """Persist a new pending pair OR return the existing row
        when the (user_id, transform_id, source_stage, unordered
        node pair) key already exists.

        Returns the pair with id and created_at populated. When
        the call inserted a new row, those are the values just
        minted. When the call collided with an existing row, those
        reflect the EXISTING row — including any prior label, which
        is preserved (re-extractions don't re-open labeled pairs).

        Caller MUST set ``user_id`` to the owning tenant. Status
        on the insert path is forced to PENDING — labels happen
        via ``label()``, not at enqueue. On the conflict path the
        existing row's status (which may be labeled) is returned
        verbatim.

        Idempotency is critical: the slice-B write hook in
        ``_compare_and_merge_nodes`` fires every time a 2-node
        candidate comes back as all-singletons. Task retries +
        re-extractions surface the same candidate multiple times
        with deterministic node IDs — without ON CONFLICT DO NOTHING
        the queue accumulates duplicate PENDING rows and labeling
        one leaves siblings still pending. Migration 18 adds the
        backing unique index; this method does the conflict
        handling."""
        if pair.id is None:
            pair.id = str(uuid.uuid4())
        if pair.created_at is None:
            pair.created_at = datetime.now(timezone.utc).isoformat()
        # Force pending — enqueued pairs are always pending by
        # definition. Allowing the caller to enqueue a labeled
        # pair would skip the audit-trail (created_at vs
        # labeled_at) the dashboard relies on.
        pair.status = Status.PENDING
        pair.labeled_at = None
        pair.labeled_by_user_id = None
        pair.label_reason = None

        if self._enabled:
            try:
                # ON CONFLICT references the same expression list as
                # the unique index from migration 18. Postgres infers
                # the index from this expression list. RETURNING is
                # empty on conflict (DO NOTHING swallows the insert);
                # the followup SELECT fetches the existing row so the
                # service contract — "always return a real persisted
                # pair" — is preserved.
                insert_query = """
                    INSERT INTO disputed_pairs (
                        id,
                        user_id,
                        transform_id,
                        node_a_id,
                        node_b_id,
                        entity_type,
                        node_a_canonical_key,
                        node_b_canonical_key,
                        similarity_score,
                        source_stage,
                        status,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (
                        user_id,
                        transform_id,
                        source_stage,
                        LEAST(node_a_id, node_b_id),
                        GREATEST(node_a_id, node_b_id)
                    ) DO NOTHING
                    RETURNING id, user_id, transform_id, node_a_id, node_b_id,
                              entity_type, node_a_canonical_key,
                              node_b_canonical_key, similarity_score,
                              source_stage, status, labeled_at,
                              labeled_by_user_id, label_reason, created_at
                """
                row = await db.fetchrow(
                    insert_query,
                    pair.id,
                    pair.user_id,
                    pair.transform_id,
                    pair.node_a_id,
                    pair.node_b_id,
                    pair.entity_type,
                    pair.node_a_canonical_key,
                    pair.node_b_canonical_key,
                    pair.similarity_score,
                    pair.source_stage.value,
                    pair.status.value,
                    pair.created_at,
                )
                if row is not None:
                    return _row_to_pair(row)
                # Conflict path: fetch the existing canonical row.
                # Computing lo/hi Python-side keeps params primitive
                # and the WHERE clause's LEAST/GREATEST evaluation
                # matches the unique-index expression — Postgres
                # uses the index for an index-only scan.
                node_lo, node_hi = sorted([pair.node_a_id, pair.node_b_id])
                existing_query = """
                    SELECT id, user_id, transform_id, node_a_id, node_b_id,
                           entity_type, node_a_canonical_key,
                           node_b_canonical_key, similarity_score,
                           source_stage, status, labeled_at,
                           labeled_by_user_id, label_reason, created_at
                    FROM disputed_pairs
                    WHERE user_id = %s
                      AND transform_id = %s
                      AND source_stage = %s
                      AND LEAST(node_a_id, node_b_id) = %s
                      AND GREATEST(node_a_id, node_b_id) = %s
                """
                existing_row = await db.fetchrow(
                    existing_query,
                    pair.user_id,
                    pair.transform_id,
                    pair.source_stage.value,
                    node_lo,
                    node_hi,
                )
                if existing_row is None:
                    # Defensive: ON CONFLICT fired but the row isn't
                    # visible. Shouldn't happen outside of a concurrent
                    # DELETE race that the queue surface doesn't expose
                    # (slice A's API has no DELETE). Log + return the
                    # caller's input so the surface contract holds.
                    logger.warning(
                        "Disputed-pair conflict on enqueue but no "
                        "matching row found (user=%s, transform=%s, "
                        "stage=%s, pair=%s/%s). Possible concurrent "
                        "delete or schema drift.",
                        pair.user_id,
                        pair.transform_id,
                        pair.source_stage.value,
                        node_lo,
                        node_hi,
                    )
                    return pair
                return _row_to_pair(existing_row)
            except Exception as exc:
                logger.error(
                    "Failed to enqueue disputed pair for user %s: %s",
                    pair.user_id,
                    exc,
                )
                raise
        # Memory backend — mirror the canonical-key dedup so both
        # backends behave identically when the hook fires twice.
        canon = _canonical_pair_key(
            pair.user_id,
            pair.transform_id,
            pair.source_stage,
            pair.node_a_id,
            pair.node_b_id,
        )
        for existing in self._memory_store:
            if (
                _canonical_pair_key(
                    existing.user_id,
                    existing.transform_id,
                    existing.source_stage,
                    existing.node_a_id,
                    existing.node_b_id,
                )
                == canon
            ):
                return existing
        self._memory_store.append(pair)
        return pair

    async def label(
        self,
        pair_id: str,
        user_id: str,
        decision: Decision,
        reason: Optional[str] = None,
    ) -> Optional[DisputedPair]:
        """Apply a label to a pair. Returns the updated pair, or
        None when no row matched the (pair_id, user_id) tuple
        (either the pair doesn't exist OR it belongs to another
        tenant — the endpoint should translate this to 404
        rather than leaking the distinction).

        Re-labeling is allowed: a previously labeled pair can be
        relabeled with a different decision, overwriting
        labeled_at + labeled_by_user_id with the latest. This
        supports the "I changed my mind" UX without forcing a
        separate "undo" endpoint."""
        new_status = _DECISION_TO_STATUS[decision]
        labeled_at = datetime.now(timezone.utc).isoformat()
        labeled_pair: Optional[DisputedPair] = None

        if self._enabled:
            try:
                query = """
                    UPDATE disputed_pairs
                    SET status = %s,
                        labeled_at = %s,
                        labeled_by_user_id = %s,
                        label_reason = %s
                    WHERE id = %s AND user_id = %s
                    RETURNING id, user_id, transform_id, node_a_id, node_b_id,
                              entity_type, node_a_canonical_key,
                              node_b_canonical_key, similarity_score,
                              source_stage, status, labeled_at,
                              labeled_by_user_id, label_reason, created_at
                """
                row = await db.fetchrow(
                    query,
                    new_status.value,
                    labeled_at,
                    user_id,
                    reason,
                    pair_id,
                    user_id,
                )
                if row:
                    labeled_pair = _row_to_pair(row)
            except Exception as exc:
                logger.error(
                    "Failed to label disputed pair %s for user %s: %s",
                    pair_id,
                    user_id,
                    exc,
                )
                raise
        else:
            # Memory backend.
            for pair in self._memory_store:
                if pair.id == pair_id and pair.user_id == user_id:
                    pair.status = new_status
                    pair.labeled_at = labeled_at
                    pair.labeled_by_user_id = user_id
                    pair.label_reason = reason
                    labeled_pair = pair
                    break

        if labeled_pair is None:
            return None

        # B2-active slice C: close the active-learning feedback
        # loop. The user has just judged a candidate pair; that
        # judgment is a high-confidence signal for merge_learning
        # to adjust the per-(user, entity_type) threshold used by
        # cluster_entities_with_splink. MATCH labels nudge the
        # threshold lower (more permissive); NOT_MATCH labels
        # nudge it higher (more strict); SKIP is a no-op.
        #
        # Failures here are swallowed — the user's label is the
        # primary persisted artifact and MUST succeed independent
        # of an in-memory bookkeeping side-effect. Mirrors the
        # observability-vs-correctness split applied to
        # DecisionLogService.append.
        try:
            await merge_learning_service.apply_pair_label(
                user_id=user_id,
                entity_type=labeled_pair.entity_type,
                decision=decision,
            )
        except Exception as exc:
            logger.warning(
                "Failed to apply disputed-pair label %s to "
                "merge_learning for pair=%s user=%s type=%s: %s — "
                "label persisted, threshold unchanged",
                decision.value,
                pair_id,
                user_id,
                labeled_pair.entity_type,
                exc,
            )

        return labeled_pair

    # Reads --------------------------------------------------------------------

    async def get(self, pair_id: str, user_id: str) -> Optional[DisputedPair]:
        """Fetch a single pair scoped to the requesting user.
        Returns None when no row matches — the endpoint should
        return 404 without leaking whether the pair exists for
        another tenant."""
        if self._enabled:
            try:
                row = await db.fetchrow(
                    """
                    SELECT id, user_id, transform_id, node_a_id, node_b_id,
                           entity_type, node_a_canonical_key,
                           node_b_canonical_key, similarity_score,
                           source_stage, status, labeled_at,
                           labeled_by_user_id, label_reason, created_at
                    FROM disputed_pairs
                    WHERE id = %s AND user_id = %s
                    """,
                    pair_id,
                    user_id,
                )
                return _row_to_pair(row) if row else None
            except Exception as exc:
                logger.error(
                    "Failed to fetch disputed pair %s for user %s: %s",
                    pair_id,
                    user_id,
                    exc,
                )
                raise
        for pair in self._memory_store:
            if pair.id == pair_id and pair.user_id == user_id:
                return pair
        return None

    async def list_pending(
        self,
        user_id: str,
        *,
        transform_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DisputedPair]:
        """Return the user's pending queue. ``transform_id``
        filters to a single run; omitting it returns pending
        pairs across all of the user's transforms.

        Ordered by ``created_at DESC`` (newest first) — matches
        the (user_id, status, created_at DESC) index so the
        paginated read is index-only."""
        if self._enabled:
            try:
                base = [
                    "SELECT id, user_id, transform_id, node_a_id, node_b_id,",
                    "       entity_type, node_a_canonical_key,",
                    "       node_b_canonical_key, similarity_score,",
                    "       source_stage, status, labeled_at,",
                    "       labeled_by_user_id, label_reason, created_at",
                    "FROM disputed_pairs",
                    "WHERE user_id = %s AND status = %s",
                ]
                params: List[Any] = [user_id, Status.PENDING.value]
                if transform_id is not None:
                    base.append("AND transform_id = %s")
                    params.append(transform_id)
                base.append("ORDER BY created_at DESC")
                base.append("LIMIT %s OFFSET %s")
                params.extend([limit, offset])
                rows = await db.fetch("\n".join(base), *params)
                return [_row_to_pair(r) for r in rows]
            except Exception as exc:
                logger.error(
                    "Failed to list pending disputed pairs for user %s: %s",
                    user_id,
                    exc,
                )
                raise
        # Memory backend.
        filtered = [
            p
            for p in self._memory_store
            if p.user_id == user_id and p.status == Status.PENDING
        ]
        if transform_id is not None:
            filtered = [p for p in filtered if p.transform_id == transform_id]
        filtered.sort(key=lambda p: p.created_at or "", reverse=True)
        return filtered[offset : offset + limit]

    async def list_for_transform(
        self, user_id: str, transform_id: str
    ) -> List[DisputedPair]:
        """All pairs for a specific transform, regardless of
        status. Used by the per-transform review view (vs the
        global pending queue)."""
        if self._enabled:
            try:
                rows = await db.fetch(
                    """
                    SELECT id, user_id, transform_id, node_a_id, node_b_id,
                           entity_type, node_a_canonical_key,
                           node_b_canonical_key, similarity_score,
                           source_stage, status, labeled_at,
                           labeled_by_user_id, label_reason, created_at
                    FROM disputed_pairs
                    WHERE user_id = %s AND transform_id = %s
                    ORDER BY created_at DESC
                    """,
                    user_id,
                    transform_id,
                )
                return [_row_to_pair(r) for r in rows]
            except Exception as exc:
                logger.error(
                    "Failed to list disputed pairs for user %s transform %s: %s",
                    user_id,
                    transform_id,
                    exc,
                )
                raise
        filtered = [
            p
            for p in self._memory_store
            if p.user_id == user_id and p.transform_id == transform_id
        ]
        filtered.sort(key=lambda p: p.created_at or "", reverse=True)
        return filtered


# Helpers --------------------------------------------------------------------


def _row_to_pair(row: Dict[str, Any]) -> DisputedPair:
    """Convert a Postgres row dict into a typed DisputedPair.

    Enums on unknown values raise ValueError — mirrors the
    DecisionLogService convention. A future schema drift would
    fail loud here rather than silently downgrading to the
    string."""
    return DisputedPair(
        id=str(row["id"]),
        user_id=row["user_id"],
        transform_id=row["transform_id"],
        node_a_id=row["node_a_id"],
        node_b_id=row["node_b_id"],
        entity_type=row["entity_type"],
        node_a_canonical_key=row.get("node_a_canonical_key"),
        node_b_canonical_key=row.get("node_b_canonical_key"),
        similarity_score=(
            Decimal(str(row["similarity_score"]))
            if row.get("similarity_score") is not None
            else None
        ),
        source_stage=SourceStage(row["source_stage"]),
        status=Status(row["status"]),
        labeled_at=_iso_or_none(row.get("labeled_at")),
        labeled_by_user_id=row.get("labeled_by_user_id"),
        label_reason=row.get("label_reason"),
        created_at=_iso_or_none(row.get("created_at")),
    )


def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


__all__ = [
    "Decision",
    "DisputedPair",
    "DisputedPairsService",
    "SourceStage",
    "Status",
]
