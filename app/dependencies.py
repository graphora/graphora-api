from fastapi import Depends
from app.services.merge_service import MergeService

def get_merge_service() -> MergeService:
    """Dependency for getting MergeService instance"""
    return MergeService()
