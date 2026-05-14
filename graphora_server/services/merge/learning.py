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

    def _status_contribution(self, status: Any) -> float:
        """Map a disputed-pair status to its threshold contribution.

        Only terminal LABELED_* statuses carry directional signal —
        pending and skipped contribute zero. Match labels pull the
        threshold toward more-permissive (negative contribution);
        not_match labels pull toward more-strict (positive). The
        transition logic in ``apply_pair_label`` then applies a
        NET delta of ``contribution(new) - contribution(old)``,
        which makes re-labels idempotent (same status → zero delta)
        and re-decisions correct (match→not_match swings by the
        sum of both nudges).

        Accepts ``Any`` (raw string OR ``Status`` enum) to avoid a
        circular import with ``disputed_pairs_service``; matches by
        ``.value`` so both representations work."""
        status_value = getattr(status, "value", status)
        if status_value == "labeled_match":
            return self._label_match_nudge
        if status_value == "labeled_not_match":
            return self._label_not_match_nudge
        # pending / skipped / None / unknown → zero contribution.
        return 0.0

    async def apply_pair_label(
        self,
        user_id: Optional[str],
        entity_type: str,
        *,
        old_status: Any,
        new_status: Any,
    ) -> Optional[Tuple[float, float]]:
        """B2-active slice C (transition-aware): close the
        active-learning feedback loop by applying the NET delta of
        a status transition to the per-(user, entity_type)
        threshold.

        Reviewer-flagged P2 on commit 72381b4: the previous
        decision-based signature applied a full nudge on every
        successful label call, including idempotent re-labels of
        the same pair. A client retry or double-submit of the same
        MATCH label moved the threshold twice for one human
        decision. The transition-aware version computes
        ``delta = contribution(new_status) - contribution(old_status)``
        so:

          * pending → labeled_match: one match nudge applied
          * labeled_match → labeled_match: delta == 0, no-op
          * labeled_match → labeled_not_match: ``+2*|nudge|`` swing
            (undo the prior match nudge, apply the not_match nudge)
          * labeled_match → skipped: undo the match nudge (return
            to neutral)
          * skipped → labeled_match: apply the match nudge

        Where "contribution" maps:
          * labeled_match → ``label_match_nudge`` (negative default)
          * labeled_not_match → ``label_not_match_nudge`` (positive)
          * pending / skipped / unknown → 0.0

        Bootstrap path (stats slot empty): seed = 1.0 + delta,
        clamped into ``[floor, 1.0]``. Clamping at ``floor`` prevents
        accumulated MATCH labels from driving the threshold to 0
        (accept-everything); clamping at 1.0 prevents accumulated
        NOT_MATCH labels from making any future merge impossible.

        Returns ``(old_ema_low_score, new_ema_low_score)`` when
        stats mutated. Returns ``None`` when delta is 0 (same-status
        transition, or transition between non-labeled states like
        pending→skipped). Caller treats None as "no threshold
        change — nothing to audit"."""
        delta = self._status_contribution(new_status) - self._status_contribution(
            old_status
        )
        if delta == 0:
            # Same-status transition (idempotent re-label) or a
            # transition between two non-labeled states. Either way
            # nothing directional to apply.
            return None

        key = self._key(user_id, entity_type)
        timestamp = datetime.now(timezone.utc)
        stats = self._stats.get(key)

        if stats is None:
            # Bootstrap: no prior observation. Seed with 1.0
            # (perfect-match prior) + delta so the first transition
            # has its full directional effect from a neutral baseline.
            seed = max(self._floor, min(1.0, 1.0 + delta))
            stats = EntityMergeLearningStats(
                ema_low_score=seed,
                sample_count=1,
                last_observed=seed,
                last_updated=timestamp,
            )
            self._stats[key] = stats
            logger.debug(
                "Bootstrapped merge threshold from first label transition",
                extra={
                    "user": key[0],
                    "entity_type": entity_type,
                    "old_status": getattr(old_status, "value", old_status),
                    "new_status": getattr(new_status, "value", new_status),
                    "delta": delta,
                    "ema_low_score": seed,
                },
            )
            return (1.0, seed)

        old = stats.ema_low_score
        new = max(self._floor, min(1.0, old + delta))
        stats.ema_low_score = new
        stats.sample_count += 1
        stats.last_observed = new
        stats.last_updated = timestamp

        logger.debug(
            "Applied label-transition delta to merge threshold",
            extra={
                "user": key[0],
                "entity_type": entity_type,
                "old_status": getattr(old_status, "value", old_status),
                "new_status": getattr(new_status, "value", new_status),
                "delta": delta,
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
