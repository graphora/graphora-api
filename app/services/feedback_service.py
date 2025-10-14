"""Feedback Service for storing quality validation feedback in Supabase"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from enum import Enum

from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback"""

    QUALITY_REJECTION = "quality_rejection"
    QUALITY_APPROVAL = "quality_approval"
    GENERAL_FEEDBACK = "general_feedback"


class FeedbackService:
    """Service for managing user feedback in Supabase"""

    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning(
                "Supabase credentials not configured, feedback service will be disabled"
            )
            self.client = None
        else:
            self.client: Client = create_client(
                settings.SUPABASE_URL, settings.SUPABASE_KEY
            )

    async def store_quality_feedback(
        self,
        user_id: str,
        transform_id: str,
        feedback_type: FeedbackType,
        feedback_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store quality validation feedback"""
        if not self.client:
            logger.warning("Supabase client not available, skipping feedback storage")
            return False

        try:
            feedback_data = {
                "user_id": user_id,
                "transform_id": transform_id,
                "feedback_type": feedback_type.value,
                "feedback_content": feedback_content,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": "quality_dashboard",
            }

            # Insert into feedback table
            result = (
                self.client.table("quality_feedback").insert(feedback_data).execute()
            )

            if result.data:
                logger.info(
                    f"Quality feedback stored successfully for user {user_id}, transform {transform_id}"
                )
                return True
            else:
                logger.error(f"Failed to store feedback: {result}")
                return False

        except Exception as e:
            logger.error(f"Error storing quality feedback: {e}")
            return False

    async def get_user_feedback(self, user_id: str, limit: int = 50) -> list:
        """Get feedback history for a user"""
        if not self.client:
            return []

        try:
            result = (
                self.client.table("quality_feedback")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error retrieving user feedback: {e}")
            return []

    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        if not self.client:
            return {}

        try:
            # Get counts by feedback type
            result = (
                self.client.table("quality_feedback")
                .select("feedback_type, count")
                .execute()
            )

            stats = {}
            if result.data:
                for row in result.data:
                    stats[row["feedback_type"]] = row.get("count", 0)

            return stats

        except Exception as e:
            logger.error(f"Error retrieving feedback stats: {e}")
            return {}


# Global instance
feedback_service = FeedbackService()
