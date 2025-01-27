from fastapi import Depends
from functools import lru_cache
from app.services.merge_service import MergeService

@lru_cache()
def get_merge_service() -> MergeService:
    """Get singleton instance of MergeService"""
    return MergeService()
