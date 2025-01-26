from typing import List, Optional, Dict
import asyncio
from datetime import datetime, timedelta
import uuid
from app.schemas.global_merge import ReviewItem, NodeResolutionResult, ReviewStatus

class HumanReviewQueue:
    def __init__(self):
        self.queue: Dict[str, ReviewItem] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, resolution_result: NodeResolutionResult) -> str:
        """Add a new item to the review queue"""
        async with self._lock:
            review_id = str(uuid.uuid4())
            review_item = ReviewItem(
                id=review_id,
                staging_node=resolution_result.staging_node,
                prod_node_id=resolution_result.prod_node_id,
                issues=resolution_result.issues,
                status=ReviewStatus.PENDING,
                confidence=resolution_result.confidence
            )
            self.queue[review_id] = review_item
            return review_id

    async def get_pending_reviews(self, limit: int = 10) -> List[ReviewItem]:
        """Get pending review items"""
        async with self._lock:
            pending = [item for item in self.queue.values()
                      if item.status == ReviewStatus.PENDING]
            return pending[:limit]

    async def assign_review(self, review_id: str, reviewer_id: str) -> ReviewItem:
        """Assign a review to a specific reviewer"""
        async with self._lock:
            if review_id not in self.queue:
                raise ValueError(f"Review {review_id} not found")

            review = self.queue[review_id]
            if review.status != ReviewStatus.PENDING:
                raise ValueError(f"Review {review_id} is not pending")

            review.assigned_to = reviewer_id
            review.updated_at = datetime.now()
            return review

    async def submit_review(
        self,
        review_id: str,
        reviewer_id: str,
        status: ReviewStatus,
        notes: Optional[str] = None,
        modifications: Optional[Dict] = None
    ) -> ReviewItem:
        """Submit a review decision"""
        async with self._lock:
            if review_id not in self.queue:
                raise ValueError(f"Review {review_id} not found")

            review = self.queue[review_id]
            if review.assigned_to != reviewer_id:
                raise ValueError(f"Review {review_id} not assigned to {reviewer_id}")

            review.status = status
            review.reviewer_notes = notes
            review.updated_at = datetime.now()

            if status == ReviewStatus.MODIFIED and modifications:
                # Apply modifications to the staging node
                review.staging_node.properties.update(modifications)

            return review

    async def get_review_stats(self) -> Dict:
        """Get statistics about the review queue"""
        async with self._lock:
            stats = {
                "total": len(self.queue),
                "by_status": {status: 0 for status in ReviewStatus},
                "average_time_to_review": timedelta(0)
            }

            completed_reviews = []

            for review in self.queue.values():
                stats["by_status"][review.status] += 1
                if review.status != ReviewStatus.PENDING:
                    completed_reviews.append(review)

            if completed_reviews:
                total_time = sum(
                    (review.updated_at - review.created_at
                     for review in completed_reviews),
                    timedelta(0)
                )
                stats["average_time_to_review"] = total_time / len(completed_reviews)

            return stats

    async def cleanup_old_reviews(self, days: int = 30):
        """Clean up old resolved reviews"""
        cutoff = datetime.now() - timedelta(days=days)
        async with self._lock:
            to_remove = [
                review_id for review_id, review in self.queue.items()
                if review.status != ReviewStatus.PENDING
                and review.updated_at < cutoff
            ]
            for review_id in to_remove:
                del self.queue[review_id]