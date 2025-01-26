from app.schemas.global_merge import ERState, ResolutionStatus
from app.services.global_merge.human_review import HumanReviewQueue

class HumanReviewAgent:
    """Agent for managing human review workflow"""

    def __init__(self, review_queue: HumanReviewQueue):
        self.review_queue = review_queue

    async def run(self, state: ERState) -> ERState:
        """Manage review workflow"""
        for result in state.processed_nodes:
            if (result.status == ResolutionStatus.NEEDS_REVIEW.value or
                result.confidence < 0.85 or
                len(result.issues) > 0):
                # Don't use to_review_item, use enqueue directly
                review_id = await self.review_queue.enqueue(result)
                state.metadata[f"review_{result.staging_node.id}"] = review_id
        return state