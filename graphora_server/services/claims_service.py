"""B1-prob slice 1: persisted probabilistic claims for the
extraction pipeline.

A :class:`Claim` is an extractor's property-level assertion: a
specific extractor, reading a specific source chunk, concluded
that a specific node or edge had a specific property value with
a specific confidence. The pipeline today resolves conflicting
claims by picking a single winner per (target, property); claims
keep the full distribution so contradictions stay queryable.

Slice 1 (this file) lands the foundation: data model + dual-
backend service + contradiction detector. Slice 2 wires the
extraction pipeline to emit claims at write time and adds a
``/contradictions`` REST + MCP surface. Slice 3 puts the
Contradictions tab in the Evidence Explorer.

The service mirrors :class:`ScenarioService` (commits d7a1f6e /
088692b) in three deliberate ways:
  * Module-level ``_DEFAULT_MEMORY_STORE`` so dev-mode CRUD
    survives across the per-request service instances FastAPI
    constructs (reviewer-flagged High on d7a1f6e).
  * DB exceptions propagate from reads — claims are user data,
    not best-effort observability. The API layer surfaces them
    as 5xx rather than masquerading as "no claims" / "not
    found" (reviewer-flagged High on d7a1f6e).
  * Tenant scoping is enforced on every read; cross-tenant
    queries see empty results, never a partial leak.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from psycopg.types.json import Json

from graphora_server.config import settings
from graphora_server.db import postgres as db

logger = logging.getLogger(__name__)


class TargetKind(str, Enum):
    """The two things a claim can be about. Matches B0-log's
    TargetKind closed set so callers can pass-through enum
    values from one surface to the other without remapping.
    Schema-level claims are intentionally not in scope —
    pipeline ontology decisions go to the Decision Log, not
    here."""

    NODE = "node"
    EDGE = "edge"


@dataclass
class Claim:
    """One extractor's assertion about a target's property.

    Slice 1 keeps the shape narrow: target_id + property_key +
    value identify what's being claimed; confidence + source_*
    identify who claimed it and how strongly. The same target
    can carry many claims across (extractor, chunk) pairs; the
    contradiction detector groups by (target_id, property_key)
    and looks at the distinct-value set.
    """

    transform_id: str
    target_id: str
    target_kind: TargetKind
    property_key: str
    value: Any
    confidence: float
    # Reviewer-flagged Medium on commit 5331a46. user_id is
    # ``TEXT NOT NULL`` in migration 20 — the dataclass must
    # match the SQL schema. Pre-fix this was Optional[str], so
    # in memory mode a Claim(user_id=None) was silently
    # accepted by ``append`` and then dropped at read time
    # (tenant filter excluded the orphan row); in Postgres
    # mode the INSERT would 500 with a NOT NULL violation.
    # No legacy data to worry about — this is a fresh table
    # — so requiring the field outright is the right pin.
    # Note: this field has to live above the optional-with-
    # default fields below; Python dataclasses don't allow a
    # non-default field after a defaulted one.
    user_id: str
    source_chunk_id: Optional[str] = None
    source_extractor_model: Optional[str] = None
    source_prompt_version: Optional[str] = None
    # Server-side defaults — populated by ``append`` so callers
    # don't have to mint them. Matches DecisionLogService /
    # ScenarioService conventions.
    id: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        # Enforce the DB CHECK at the service boundary too —
        # better to fail fast in the writer than have the SQL
        # layer return an opaque "constraint violation" error
        # downstream. Mirrors the JSON-cleanliness rule the
        # /golden/score scorer uses for zero-denominator F1.
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0]; got {self.confidence!r}"
            )
        # The dataclass type annotation enforces "must pass a
        # value" but Python doesn't reject empty strings — pin
        # the non-empty invariant here so a buggy writer that
        # passes ``""`` fails at construction rather than
        # quietly writing an orphan row.
        if not self.user_id:
            raise ValueError("user_id is required and must be non-empty")


@dataclass
class Contradiction:
    """A target+property carrying multiple distinct claimed values.

    Two claims with the same JSON-equal value are NOT a
    contradiction even if they came from different chunks — the
    pipeline can legitimately re-emit the same fact. Two claims
    with different values mean the extractor disagreed with
    itself (across chunks) or with another extractor (across
    routing); both are signals an operator should review.

    ``severity`` is a heuristic on the value distribution. Slice
    1 keeps it simple: severity = number of distinct values
    above the confidence threshold. Slice 2 may swap to a
    confidence-weighted entropy measure once the data lands.
    """

    target_id: str
    target_kind: TargetKind
    property_key: str
    # Ordered: highest-confidence claim first. The "winning"
    # value the pipeline picked is typically the first entry.
    competing_claims: List[Claim] = field(default_factory=list)
    severity: int = 0


# Module-level shared dev-mode store. Mirrors ScenarioService's
# pattern (commit 088692b). Per-request service instances must
# share the dev-mode list so writes survive cross-request reads.
# Tenant filtering at read time keeps users isolated even
# though the underlying list is process-wide.
_DEFAULT_MEMORY_STORE: List[Claim] = []


def _reset_default_memory_store_for_tests() -> None:
    """Clear the dev-mode shared store. Test fixtures that fall
    through to the production constructor (no ``memory_store=``
    arg) should call this between scenarios to keep tests
    isolated."""
    _DEFAULT_MEMORY_STORE.clear()


class ClaimsService:
    """Append-only log of extraction-time property claims.

    Dual backend: Postgres when ``DATABASE_URL`` is configured
    (production), in-memory list otherwise (zero-config dev).
    Tenant-scoped on every read.

    Failure posture: DB exceptions propagate from reads. Claims
    are user data — losing them to a swallowed exception would
    silently break the contradiction-detection downstream
    surface. The append path also propagates (a Postgres
    constraint violation indicates a buggy writer, not a
    transient observability blip). Mirrors ScenarioService's
    posture for the same reasons.
    """

    TABLE_NAME = "claims"

    def __init__(
        self,
        memory_store: Optional[List[Claim]] = None,
    ) -> None:
        self._enabled = bool(settings.DATABASE_URL or settings.resolved_database_url)
        # See the ScenarioService comment for the full rationale:
        # the API constructs a fresh service per request, and a
        # per-instance empty list would lose writes across
        # request boundaries in dev mode. Tests pass
        # ``memory_store=[]`` explicitly when they need
        # isolation.
        if memory_store is not None:
            self._memory_store = memory_store
        else:
            self._memory_store = _DEFAULT_MEMORY_STORE

    # Writes ---------------------------------------------------------------------

    async def append(self, claim: Claim) -> Claim:
        """Persist a claim. Returns the claim with id +
        created_at populated.

        Postgres failures propagate (see class docstring). The
        per-DB CHECK on confidence range catches the same
        invariant the ``Claim.__post_init__`` does, but at the
        SQL layer — defence in depth for any writer that
        bypasses the dataclass (manual SQL, future
        microservice, etc.).
        """
        if claim.id is None:
            claim.id = str(uuid.uuid4())
        if claim.created_at is None:
            claim.created_at = datetime.now(timezone.utc).isoformat()

        if self._enabled:
            await db.execute(
                """
                INSERT INTO claims (
                    id, user_id, transform_id, target_id, target_kind,
                    property_key, value, confidence,
                    source_chunk_id, source_extractor_model,
                    source_prompt_version, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                claim.id,
                claim.user_id,
                claim.transform_id,
                claim.target_id,
                claim.target_kind.value,
                claim.property_key,
                Json(claim.value),
                claim.confidence,
                claim.source_chunk_id,
                claim.source_extractor_model,
                claim.source_prompt_version,
                claim.created_at,
            )
        else:
            self._memory_store.append(claim)
        return claim

    # Reads ----------------------------------------------------------------------

    async def for_target(
        self,
        transform_id: str,
        target_id: str,
        target_kind: TargetKind,
        user_id: str,
    ) -> List[Claim]:
        """All claims for a specific node OR edge in the given
        transform, ordered by confidence DESC then created_at ASC.

        Reviewer-flagged Medium on commit 5331a46. ``target_kind``
        is required because target_id is unique only per kind: a
        node and an edge can theoretically share the same id
        string. Pre-fix this method only filtered on target_id,
        so a node+edge collision would return both row sets
        interleaved and confuse downstream consumers (slice 2's
        node-evidence vs edge-evidence surfaces would silently
        cross-pollinate). The contradiction detector already
        keys on (target_id, target_kind, property_key) — this
        change makes the single-target lookup use the same
        identity rule.

        The "highest confidence first" ordering is load-bearing
        for downstream consumers: the Contradictions tab shows
        the winning claim at the top, and the contradiction
        detector trusts the same ordering to identify the
        "default" claim vs the alternatives.
        """
        if self._enabled:
            rows = await db.fetch(
                """
                SELECT id, user_id, transform_id, target_id, target_kind,
                       property_key, value, confidence,
                       source_chunk_id, source_extractor_model,
                       source_prompt_version, created_at
                FROM claims
                WHERE transform_id = %s
                  AND target_id = %s
                  AND target_kind = %s
                  AND user_id = %s
                ORDER BY confidence DESC, created_at ASC
                """,
                transform_id,
                target_id,
                target_kind.value,
                user_id,
            )
            return self._rows_to_claims(rows)
        return sorted(
            (
                c
                for c in self._memory_store
                if c.transform_id == transform_id
                and c.target_id == target_id
                and c.target_kind == target_kind
                and c.user_id == user_id
            ),
            # Two-key sort to match the Postgres ORDER BY: high
            # confidence first, ties broken by earlier
            # created_at (stable across runs).
            key=lambda c: (-(c.confidence or 0.0), c.created_at or ""),
        )

    async def for_transform(
        self,
        transform_id: str,
        user_id: str,
    ) -> List[Claim]:
        """All claims for a transform, ordered by (target_id,
        property_key, confidence DESC) so callers walking the
        list see all claims about one target grouped together.
        Useful for batch contradiction detection."""
        if self._enabled:
            rows = await db.fetch(
                """
                SELECT id, user_id, transform_id, target_id, target_kind,
                       property_key, value, confidence,
                       source_chunk_id, source_extractor_model,
                       source_prompt_version, created_at
                FROM claims
                WHERE transform_id = %s AND user_id = %s
                ORDER BY target_id ASC, property_key ASC, confidence DESC
                """,
                transform_id,
                user_id,
            )
            return self._rows_to_claims(rows)
        return sorted(
            (
                c
                for c in self._memory_store
                if c.transform_id == transform_id and c.user_id == user_id
            ),
            key=lambda c: (
                c.target_id,
                c.property_key,
                -(c.confidence or 0.0),
            ),
        )

    async def contradictions_for_transform(
        self,
        transform_id: str,
        user_id: str,
        *,
        min_confidence: float = 0.0,
    ) -> List[Contradiction]:
        """Find (target, property) pairs with multiple distinct
        claimed values above ``min_confidence``.

        Slice 1 algorithm: group claims by (target_id,
        property_key), filter by confidence floor, count
        distinct JSON-equal values. A group with 2+ distinct
        values becomes a Contradiction; its competing_claims
        list carries all qualifying claims sorted by
        confidence DESC so the winning value is first.

        ``min_confidence`` keeps low-confidence noise out of the
        contradictions surface. Default 0.0 returns everything;
        the API layer can tune via a query parameter once slice
        2 ships the read endpoint.
        """
        claims = await self.for_transform(transform_id, user_id)

        # Group by (target_id, target_kind, property_key).
        # target_kind goes in the key because the same id
        # string could theoretically belong to a node OR an
        # edge — keep them in separate contradiction groups.
        groups: Dict[Tuple[str, TargetKind, str], List[Claim]] = defaultdict(list)
        for claim in claims:
            if claim.confidence < min_confidence:
                continue
            groups[(claim.target_id, claim.target_kind, claim.property_key)].append(
                claim
            )

        contradictions: List[Contradiction] = []
        for (target_id, target_kind, property_key), group in groups.items():
            # Distinct values: serialize via json.dumps so list/
            # dict values compare structurally, not by Python
            # identity. ``sort_keys=True`` makes dict ordering
            # irrelevant to the equality check.
            import json as _json

            distinct = set()
            for claim in group:
                key = _json.dumps(claim.value, sort_keys=True, default=str)
                distinct.add(key)
            if len(distinct) < 2:
                continue
            contradictions.append(
                Contradiction(
                    target_id=target_id,
                    target_kind=target_kind,
                    property_key=property_key,
                    # Group is already sorted by confidence
                    # DESC from for_transform's ORDER BY.
                    competing_claims=group,
                    severity=len(distinct),
                )
            )
        # Stable output ordering: most-severe first, ties by
        # target_id + property_key for diffability across runs.
        contradictions.sort(key=lambda c: (-c.severity, c.target_id, c.property_key))
        return contradictions

    # Helpers --------------------------------------------------------------------

    @classmethod
    def _rows_to_claims(cls, rows: List[Dict[str, Any]]) -> List[Claim]:
        """Per-row conversion with isolation: one malformed row
        gets logged + skipped instead of poisoning the whole
        query result. Mirrors DecisionLogService's pattern
        (commit 8cbc76b)."""
        claims: List[Claim] = []
        for row in rows:
            try:
                claims.append(cls._row_to_claim(row))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed claims row (id=%s): %s",
                    row.get("id") if isinstance(row, dict) else None,
                    exc,
                )
        return claims

    @staticmethod
    def _row_to_claim(row: Dict[str, Any]) -> Claim:
        created_at = row["created_at"]
        return Claim(
            id=str(row["id"]),
            user_id=row["user_id"],
            transform_id=row["transform_id"],
            target_id=row["target_id"],
            target_kind=TargetKind(row["target_kind"]),
            property_key=row["property_key"],
            value=row["value"],
            confidence=float(row["confidence"]),
            source_chunk_id=row.get("source_chunk_id"),
            source_extractor_model=row.get("source_extractor_model"),
            source_prompt_version=row.get("source_prompt_version"),
            created_at=(
                created_at.isoformat()
                if hasattr(created_at, "isoformat")
                else created_at
            ),
        )
