from functools import lru_cache
from app.services.merge.service import MergeService
from app.services.storage.neo4j import Neo4jStorage
from app.services.merge.progress import ProgressTracker
from app.config import settings
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_staging_storage():
    """Get instance of staging storage with proper async context management"""
    storage = Neo4jStorage(
        uri=settings.STAGING_NEO4J_URI,
        username=settings.STAGING_NEO4J_USER,
        password=settings.STAGING_NEO4J_PASSWORD,
        database=settings.STAGING_NEO4J_DATABASE
    )
    try:
        await storage.__aenter__()
        yield storage
    finally:
        await storage.__aexit__(None, None, None)

@asynccontextmanager
async def get_production_storage():
    """Get instance of production storage with proper async context management"""
    storage = Neo4jStorage(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DB
    )
    try:
        await storage.__aenter__()
        yield storage
    finally:
        await storage.__aexit__(None, None, None)

@asynccontextmanager
async def get_progress_tracker():
    """Get instance of ProgressTracker with proper async context management"""
    tracker = ProgressTracker()
    try:
        await tracker.__aenter__()
        yield tracker
    finally:
        await tracker.__aexit__(None, None, None)

@asynccontextmanager
async def get_merge_service():
    """Get instance of MergeService with proper async context management"""
    async with get_staging_storage() as staging_storage, \
              get_production_storage() as production_storage, \
              get_progress_tracker() as progress_tracker:
        
        service = MergeService(
            staging_storage=staging_storage,
            production_storage=production_storage,
            progress_tracker=progress_tracker
        )
        yield service

# For backward compatibility with existing code
@asynccontextmanager
async def get_storage():
    """Get instance of Storage (defaults to production) with proper async context management"""
    async with get_production_storage() as storage:
        yield storage