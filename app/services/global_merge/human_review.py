from typing import List, Optional, Dict
import asyncio
from datetime import datetime, timedelta
import uuid
from app.schemas.global_merge import ReviewItem, NodeResolutionResult, ReviewStatus

class HumanReviewQueue:
    def __init__(self):
        self.queue: Dict[str, ReviewItem] = {}
        self._lock = asyncio.Lock()

    def _format_node_changes(self, node: Dict) -> str:
        """Format node details in a user-friendly way"""
        props = node.get('properties', {})
        details = []
        
        if props.get('name'):
            details.append(f"Name: {props['name']}")
        if props.get('description'):
            details.append(f"Description: {props['description']}")
            
        # Add other relevant properties
        for key, value in props.items():
            if key not in ['name', 'description', '_merged_ids'] and not key.startswith('_'):
                details.append(f"{key.replace('_', ' ').title()}: {value}")
                
        return '\n'.join(details)

    def _format_change_message(self, change: Dict) -> str:
        """Format a single change in a user-friendly way"""
        node = change.get('node', {})
        change_type = change.get('type', '').upper()
        node_type = node.get('labels', ['Unknown'])[0]
        
        icon = {
            'CREATE': '➕',
            'UPDATE': '✏️',
            'DELETE': '🗑️'
        }.get(change_type, '🔹')
        
        return f"{icon} {change_type} {node_type}:\n{self._format_node_changes(node)}"

    async def enqueue(self, resolution_result: NodeResolutionResult) -> str:
        """Add a new item to the review queue"""
        async with self._lock:
            review_id = str(uuid.uuid4())
            
            # Format changes for review
            changes = []
            
            # Add staging node change
            node_type = resolution_result.staging_node.labels[0] if resolution_result.staging_node.labels else "Unknown"
            
            if resolution_result.prod_node_id:
                # This is an update
                changes.append({
                    "type": "update",
                    "node": {
                        "labels": resolution_result.staging_node.labels,
                        "properties": resolution_result.staging_node.properties
                    },
                    "prod_id": resolution_result.prod_node_id
                })
            else:
                # This is a create
                changes.append({
                    "type": "create",
                    "node": {
                        "labels": resolution_result.staging_node.labels,
                        "properties": resolution_result.staging_node.properties
                    }
                })
            
            # Format review message
            changes_text = "\n\n".join(self._format_change_message(change) for change in changes)
            review_msg = f"📋 Please review the following changes:\n\n{changes_text}"
            
            if resolution_result.issues:
                review_msg += "\n\n⚠️ Issues to review:\n" + "\n".join(f"- {issue}" for issue in resolution_result.issues)
            
            if resolution_result.confidence < 0.85:
                review_msg += f"\n\n⚠️ Low confidence score: {resolution_result.confidence:.2f}"
            
            review_item = ReviewItem(
                id=review_id,
                staging_node=resolution_result.staging_node,
                prod_node_id=resolution_result.prod_node_id,
                issues=resolution_result.issues,
                status=ReviewStatus.PENDING,
                confidence=resolution_result.confidence,
                changes=changes,
                content=review_msg
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