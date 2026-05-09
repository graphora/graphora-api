"""B0-log slice 1: persisted Decision Log for extraction-time decisions.

The Decision Log is the data surface that turns Graphora from "looks
like a graph" into "I'd defend this." Every automated decision the
pipeline makes — schema inferred, entity merged, relationship
accepted/rejected, confidence marked — should be appendable here, and
later surfaced to the user via Evidence Explorer's Decision Log tab
and the MCP ``get_evidence`` tool.

Slice 1 (this file) lands the foundation: data model, dual-backend
service (Postgres when DATABASE_URL is set, in-memory dict otherwise
to mirror EntityLedgerService), append + query API. Hooks at
specific decision sites (entity-merge, schema-inference, relationship
accept/reject) come in subsequent slices once the data shape proves
stable against real call sites.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from psycopg.types.json import Json

from graphora_server.config import settings
from graphora_server.db import postgres as db

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Closed set of decision types the pipeline can emit.

    Defining the full set up front (rather than letting callers pass
    arbitrary strings) gives the Decision Log tab and ``get_evidence``
    a stable contract to render against. Adding a new decision type
    is a deliberate code change in two places: this enum + the
    rendering code that needs to know how to display it.
    """

    SCHEMA_INFERRED = "schema_inferred"
    ENTITY_MERGED = "entity_merged"
    RELATIONSHIP_ACCEPTED = "relationship_accepted"
    RELATIONSHIP_REJECTED = "relationship_rejected"
    CONFIDENCE_MARKED = "confidence_marked"
    LLM_DISAMBIGUATED = "llm_disambiguated"


class TargetKind(str, Enum):
    """Whether a decision attaches to a node, an edge, or the schema.

    Schema-level decisions (e.g., schema_inferred) carry
    ``target_id=None`` because the inferred schema isn't keyed by an
    individual node/edge id. Keeping the kind explicit lets queries
    filter correctly without inspecting target_id."""

    NODE = "node"
    EDGE = "edge"
    SCHEMA = "schema"


@dataclass
class Decision:
    transform_id: str
    target_kind: TargetKind
    decision_type: DecisionType
    target_id: Optional[str] = None
    reason: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    id: Optional[str] = None
    created_at: Optional[str] = None


class DecisionLogService:
    """Append-only log of pipeline decisions, keyed for retrieval by
    ``(transform_id, target_id)`` and by ``transform_id`` alone.

    Dual backend mirrors :class:`EntityLedgerService`:
      * Postgres when ``DATABASE_URL`` is configured — the production
        path. Decisions survive process restarts and are visible to
        downstream tools.
      * In-memory list otherwise — the zero-config developer path.
        Decisions live for the lifetime of the service instance only;
        good enough for tests and local exploration.

    Append is fire-and-forget on the Postgres path: failures are
    logged but don't propagate. The Decision Log is observability,
    not a correctness gate — losing a row is regrettable but must
    never block extraction itself.
    """

    TABLE_NAME = "extraction_decisions"

    def __init__(
        self,
        memory_store: Optional[List[Decision]] = None,
    ) -> None:
        self._enabled = bool(settings.DATABASE_URL or settings.resolved_database_url)
        self._memory_store: List[Decision] = (
            memory_store if memory_store is not None else []
        )

    # Public API -----------------------------------------------------------------

    async def append(self, decision: Decision) -> Decision:
        """Persist a decision. Returns the decision with id +
        created_at populated (server-side defaults are echoed on the
        Postgres path; minted client-side on the memory path so both
        backends produce equally-shaped objects).

        Logged-and-swallowed on Postgres failures: see class
        docstring for why this is observability rather than a
        correctness gate.
        """
        if decision.id is None:
            decision.id = str(uuid.uuid4())
        if decision.created_at is None:
            decision.created_at = datetime.now(timezone.utc).isoformat()

        if self._enabled:
            try:
                query = """
                    INSERT INTO extraction_decisions (
                        id,
                        transform_id,
                        target_id,
                        target_kind,
                        decision_type,
                        reason,
                        evidence,
                        alternatives,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                await db.execute(
                    query,
                    decision.id,
                    decision.transform_id,
                    decision.target_id,
                    decision.target_kind.value,
                    decision.decision_type.value,
                    decision.reason,
                    Json(decision.evidence),
                    Json(decision.alternatives),
                    decision.created_at,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to append extraction decision: %s", exc)
        else:
            self._memory_store.append(decision)
        return decision

    async def for_target(
        self,
        transform_id: str,
        target_id: str,
    ) -> List[Decision]:
        """All decisions for a specific node/edge in the given
        transform, ordered by created_at ASC. Empty list if none."""
        if self._enabled:
            try:
                query = """
                    SELECT id, transform_id, target_id, target_kind,
                           decision_type, reason, evidence, alternatives,
                           created_at
                    FROM extraction_decisions
                    WHERE transform_id = %s AND target_id = %s
                    ORDER BY created_at ASC
                """
                rows = await db.fetch(query, transform_id, target_id)
                return self._rows_to_decisions(rows)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to fetch decisions for target: %s", exc)
                return []
        return sorted(
            (
                d
                for d in self._memory_store
                if d.transform_id == transform_id and d.target_id == target_id
            ),
            key=lambda d: d.created_at or "",
        )

    async def for_transform(self, transform_id: str) -> List[Decision]:
        """All decisions for a transform — including schema-level
        decisions whose target_id is None. Ordered by created_at ASC."""
        if self._enabled:
            try:
                query = """
                    SELECT id, transform_id, target_id, target_kind,
                           decision_type, reason, evidence, alternatives,
                           created_at
                    FROM extraction_decisions
                    WHERE transform_id = %s
                    ORDER BY created_at ASC
                """
                rows = await db.fetch(query, transform_id)
                return self._rows_to_decisions(rows)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to fetch decisions for transform: %s", exc)
                return []
        return sorted(
            (d for d in self._memory_store if d.transform_id == transform_id),
            key=lambda d: d.created_at or "",
        )

    async def for_decision_type(
        self,
        transform_id: str,
        decision_type: DecisionType,
    ) -> List[Decision]:
        """All decisions of a specific type for a transform, ordered
        by created_at ASC.

        Reviewer-flagged on commit 9ac9bb5 (B0-explain): the previous
        ``for_transform + Python filter`` pattern fetched every
        decision in the transform just to surface schema-level ones
        (one row out of potentially thousands). This method narrows
        the read at the DB layer using the existing
        ``idx_extraction_decisions_transform_type`` index from
        migration 14, so node-evidence lookups don't scale with the
        full transform decision log."""
        if self._enabled:
            try:
                query = """
                    SELECT id, transform_id, target_id, target_kind,
                           decision_type, reason, evidence, alternatives,
                           created_at
                    FROM extraction_decisions
                    WHERE transform_id = %s AND decision_type = %s
                    ORDER BY created_at ASC
                """
                rows = await db.fetch(query, transform_id, decision_type.value)
                return self._rows_to_decisions(rows)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to fetch decisions for transform/type: %s", exc)
                return []
        return sorted(
            (
                d
                for d in self._memory_store
                if d.transform_id == transform_id and d.decision_type == decision_type
            ),
            key=lambda d: d.created_at or "",
        )

    # Helpers --------------------------------------------------------------------

    @classmethod
    def _rows_to_decisions(cls, rows: List[Dict[str, Any]]) -> List[Decision]:
        """Per-row conversion with isolation: one malformed row (e.g.
        a stale ``decision_type`` from a rolled-back deploy, an enum
        value the running code doesn't yet know about) gets logged
        and skipped instead of poisoning the whole query result.

        Reviewer-flagged on commit 8cbc76b: pre-fix we did
        ``[self._row_to_decision(row) for row in rows]`` inside an
        outer try/except, so a single ``ValueError`` from
        ``DecisionType(unknown)`` would blank the entire Decision
        Log for that target/transform. Combined with the new DB-side
        CHECK constraints (migration 14), this gives defence in
        depth: the constraints prevent malformed rows landing in the
        first place, and per-row isolation contains any that slip
        through (e.g. across rolling deploys with mismatched code
        and schema).
        """
        decisions: List[Decision] = []
        for row in rows:
            try:
                decisions.append(cls._row_to_decision(row))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed extraction_decisions row " "(id=%s): %s",
                    row.get("id") if isinstance(row, dict) else None,
                    exc,
                )
        return decisions

    @staticmethod
    def _row_to_decision(row: Dict[str, Any]) -> Decision:
        """Convert a single Postgres row dict into a typed Decision.

        Raises ``ValueError`` on unknown enum values — that's the
        signal _rows_to_decisions uses to skip the row. The
        alternative (silently downgrading to the string) would let
        stale prod rows render as opaque tokens in the Evidence tab;
        forcing a code change when the enum legitimately drifts is
        the right tradeoff."""
        return Decision(
            id=str(row["id"]),
            transform_id=row["transform_id"],
            target_id=row["target_id"],
            target_kind=TargetKind(row["target_kind"]),
            decision_type=DecisionType(row["decision_type"]),
            reason=row["reason"],
            evidence=row["evidence"] or {},
            alternatives=row["alternatives"] or [],
            created_at=(
                row["created_at"].isoformat()
                if hasattr(row["created_at"], "isoformat")
                else row["created_at"]
            ),
        )
