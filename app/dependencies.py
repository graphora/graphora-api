from functools import lru_cache
from app.services.merge.service import MergeService
from app.services.storage.interface import Neo4jStorage

@lru_cache()
def get_merge_service() -> MergeService:
    """Get singleton instance of MergeService"""
    return MergeService(get_storage())

@lru_cache()
def get_storage() -> Neo4jStorage:
    """Get singleton instance of Storage"""
    return Neo4jStorage()