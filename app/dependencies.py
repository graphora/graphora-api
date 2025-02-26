from functools import lru_cache
from app.services.merge.service import MergeService
from app.services.storage.neo4j import Neo4jStorage
from app.config import settings

@lru_cache()
def get_staging_storage() -> Neo4jStorage:
    """Get singleton instance of staging storage"""
    return Neo4jStorage(
        uri=settings.STAGING_NEO4J_URI,
        username=settings.STAGING_NEO4J_USER,
        password=settings.STAGING_NEO4J_PASSWORD,
        database=settings.STAGING_NEO4J_DATABASE
    )

@lru_cache()
def get_production_storage() -> Neo4jStorage:
    """Get singleton instance of production storage"""
    return Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )

@lru_cache()
def get_merge_service() -> MergeService:
    """Get singleton instance of MergeService"""
    staging_storage = get_staging_storage()
    prod_storage = get_production_storage()
    return MergeService(staging_storage=staging_storage, prod_storage=prod_storage)

# For backward compatibility with existing code
def get_storage() -> Neo4jStorage:
    """Get singleton instance of Storage (defaults to production)"""
    return get_production_storage()