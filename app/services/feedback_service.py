"""Feedback Service for storing quality validation feedback in Postgres."""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from enum import Enum

from psycopg.types.json import Json

from app.config import settings
from app.db import postgres as db

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback"""

    QUALITY_REJECTION = "quality_rejection"
    QUALITY_APPROVAL = "quality_approval"
    GENERAL_FEEDBACK = "general_feedback"


class FeedbackService:
    """Service for managing user feedback in Postgres."""

    def __init__(self):
        if not (settings.DATABASE_URL or settings.resolved_database_url):
            logger.warning(
                "Database credentials not configured, feedback service will be disabled"
            )
            self.enabled = False
        else:
            self.enabled = True

    async def store_quality_feedback(
        self,
        user_id: str,
        transform_id: str,
        feedback_type: FeedbackType,
        feedback_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store quality validation feedback"""
        if not self.enabled:
            logger.warning("Feedback service disabled, skipping feedback storage")
            return False

        try:
            row = await db.fetchrow(
                """
                INSERT INTO quality_feedback (
                    user_id,
                    transform_id,
                    feedback_type,
                    feedback_content,
                    metadata,
                    created_at,
                    source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                user_id,
                transform_id,
                feedback_type.value,
                feedback_content,
                Json(metadata or {}),
                datetime.now(timezone.utc),
                "quality_dashboard",
            )

            if row:
                logger.info(
                    f"Quality feedback stored successfully for user {user_id}, transform {transform_id}"
                )
                return True
            else:
                logger.error(
                    "Failed to store feedback for user %s, transform %s",
                    user_id,
                    transform_id,
                )
                return False

        except Exception as e:
            logger.error(f"Error storing quality feedback: {e}")
            return False

    async def get_user_feedback(self, user_id: str, limit: int = 50) -> list:
        """Get feedback history for a user"""
        if not self.enabled:
            return []

        try:
            rows = await db.fetch(
                """
                SELECT *
                FROM quality_feedback
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                user_id,
                limit,
            )

            return rows or []

        except Exception as e:
            logger.error(f"Error retrieving user feedback: {e}")
            return []

    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        if not self.enabled:
            return {}

        try:
            rows = await db.fetch(
                """
                SELECT feedback_type, COUNT(*) AS count
                FROM quality_feedback
                GROUP BY feedback_type
                """
            )

            stats = {row["feedback_type"]: row["count"] for row in rows or []}

            return stats

        except Exception as e:
            logger.error(f"Error retrieving feedback stats: {e}")
            return {}


# Global instance
feedback_service = FeedbackService()
