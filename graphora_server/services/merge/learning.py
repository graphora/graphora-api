"""Adaptive merge threshold learning for entity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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
        self, *, alpha: float = 0.3, floor: float = 0.7, margin: float = 0.05
    ) -> None:
        self._alpha = alpha
        self._floor = floor
        self._margin = margin
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

    def snapshot(self) -> Dict[Tuple[str, str], EntityMergeLearningStats]:
        """Expose a shallow copy of stats for testing or diagnostics."""

        return dict(self._stats)

    def reset(self) -> None:
        """Clear in-memory statistics (used in tests)."""

        self._stats.clear()


merge_learning_service = MergeLearningService()
