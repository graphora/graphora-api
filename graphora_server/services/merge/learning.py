"""Adaptive merge threshold learning for entity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from graphora_server.utils.logger import logger


_GLOBAL_USER = "__global__"


@dataclass
class EntityMergeLearningStats:
    """Rolling statistics for an entity type."""

    ema_low_score: float
    sample_count: int
    last_observed: float
    last_updated: datetime


class MergeLearningService:
    """Tracks observed merge scores so thresholds can adapt over time."""

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        floor: float = 0.7,
        margin: float = 0.05,
        label_match_nudge: float = -0.05,
        label_not_match_nudge: float = 0.05,
    ) -> None:
        self._alpha = alpha
        self._floor = floor
        self._margin = margin
        # B2-active slice C: per-label nudge magnitude. A MATCH
        # label nudges ema_low_score DOWN (be more permissive next
        # run — the user confirmed this similarity-level was a real
        # match); NOT_MATCH nudges UP (be more strict — the user
        # rejected this similarity-level). Magnitude is small so a
        # single label doesn't swing the threshold; the effect
        # accumulates across labels. Defaults mirror the heuristic
        # margin so one match label moves the threshold by exactly
        # one margin-width.
        self._label_match_nudge = label_match_nudge
        self._label_not_match_nudge = label_not_match_nudge
        self._stats: Dict[Tuple[str, str], EntityMergeLearningStats] = {}

    def _key(self, user_id: Optional[str], entity_type: str) -> Tuple[str, str]:
        return (user_id or _GLOBAL_USER, entity_type)

    async def get_threshold(
        self,
        user_id: Optional[str],
        entity_type: str,
        default_threshold: float,
    ) -> float:
        """Return the active threshold for the given user/entity type."""

        stats = self._stats.get(self._key(user_id, entity_type))
        if not stats:
            return default_threshold

        if stats.ema_low_score >= default_threshold - self._margin:
            adaptive = default_threshold
        else:
            adaptive = max(stats.ema_low_score - self._margin, self._floor)
            adaptive = min(adaptive, default_threshold)
        logger.debug(
            "Adaptive merge threshold",
            extra={
                "user": user_id or _GLOBAL_USER,
                "entity_type": entity_type,
                "default": default_threshold,
                "ema_low_score": stats.ema_low_score,
                "adaptive": adaptive,
            },
        )
        return adaptive

    async def record_outcome(
        self,
        user_id: Optional[str],
        entity_type: str,
        match_scores: List[float],
    ) -> None:
        """Update learning statistics from observed match scores."""

        if not match_scores:
            return

        key = self._key(user_id, entity_type)
        low_score = max(min(match_scores), 0.0)
        timestamp = datetime.now(timezone.utc)
        stats = self._stats.get(key)

        if stats is None:
            stats = EntityMergeLearningStats(
                ema_low_score=low_score,
                sample_count=len(match_scores),
                last_observed=low_score,
                last_updated=timestamp,
            )
        else:
            stats.ema_low_score = (
                1 - self._alpha
            ) * stats.ema_low_score + self._alpha * low_score
            stats.sample_count += len(match_scores)
            stats.last_observed = low_score
            stats.last_updated = timestamp

        self._stats[key] = stats

        logger.debug(
            "Recorded merge scores",
            extra={
                "user": key[0],
                "entity_type": entity_type,
                "low_score": low_score,
                "ema_low_score": stats.ema_low_score,
                "sample_count": stats.sample_count,
            },
        )

    async def apply_pair_label(
        self,
        user_id: Optional[str],
        entity_type: str,
        decision: Any,
    ) -> Optional[Tuple[float, float]]:
        """B2-active slice C: close the active-learning feedback
        loop. When a user/agent labels a previously-disputed pair,
        nudge the adaptive merge threshold in the direction the
        label implies.

        Decision semantics:
          * ``MATCH``  — the blocker grouped this pair correctly.
            Lower ema_low_score so future runs accept similar
            pairs more readily (a small, accumulating nudge —
            ``self._label_match_nudge``).
          * ``NOT_MATCH`` — the blocker grouped this pair WRONGLY.
            Raise ema_low_score so future runs reject similar
            pairs (``self._label_not_match_nudge``).
          * Anything else (``SKIP``, unknown): no-op. Skip is a
            valid review outcome but carries no directional signal.

        On first label for a (user, type) pair the stats slot is
        bootstrapped from the nudge alone (seed = 1.0 + nudge,
        clamped to [floor, 1.0]). After that, the nudge is added
        to the current ema_low_score and clamped. Clamping at
        ``self._floor`` prevents arbitrarily-low thresholds from
        accumulated MATCH labels; clamping at 1.0 prevents
        impossible-to-meet thresholds from accumulated NOT_MATCH
        labels.

        Returns ``(old_ema_low_score, new_ema_low_score)`` so the
        caller can log/audit the adjustment. Returns ``None`` when
        no adjustment was made (SKIP / unrecognized decision).

        Type-flexible on ``decision`` (accepts ``Any``) to avoid a
        circular import between ``merge.learning`` and
        ``disputed_pairs_service`` — the actual matching is by
        string value, mirroring how the Postgres CHECK constraint
        identifies decisions."""
        decision_value = getattr(decision, "value", decision)
        if decision_value == "match":
            nudge = self._label_match_nudge
        elif decision_value == "not_match":
            nudge = self._label_not_match_nudge
        else:
            # SKIP, unknown, or None — no directional signal to
            # apply. Returning None lets the caller surface
            # "no-op" in their own logging without inspecting
            # the threshold directly.
            return None

        key = self._key(user_id, entity_type)
        timestamp = datetime.now(timezone.utc)
        stats = self._stats.get(key)

        if stats is None:
            # Bootstrap: no prior observation. Seed with 1.0
            # (perfect-match prior) + nudge so the first label
            # has its full directional effect from a neutral
            # baseline. Clamp into [floor, 1.0].
            seed = max(self._floor, min(1.0, 1.0 + nudge))
            stats = EntityMergeLearningStats(
                ema_low_score=seed,
                sample_count=1,
                last_observed=seed,
                last_updated=timestamp,
            )
            self._stats[key] = stats
            logger.debug(
                "Bootstrapped merge threshold from first user label",
                extra={
                    "user": key[0],
                    "entity_type": entity_type,
                    "decision": decision_value,
                    "ema_low_score": seed,
                },
            )
            return (1.0, seed)

        old = stats.ema_low_score
        new = max(self._floor, min(1.0, old + nudge))
        stats.ema_low_score = new
        stats.sample_count += 1
        stats.last_observed = new
        stats.last_updated = timestamp

        logger.debug(
            "Nudged merge threshold from user label",
            extra={
                "user": key[0],
                "entity_type": entity_type,
                "decision": decision_value,
                "old_ema_low_score": old,
                "new_ema_low_score": new,
            },
        )
        return (old, new)

    def snapshot(self) -> Dict[Tuple[str, str], EntityMergeLearningStats]:
        """Expose a shallow copy of stats for testing or diagnostics."""

        return dict(self._stats)

    def reset(self) -> None:
        """Clear in-memory statistics (used in tests)."""

        self._stats.clear()


merge_learning_service = MergeLearningService()
