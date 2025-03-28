from app.services.storage.neo4j import Neo4jStorage
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

# For backward compatibility with existing code
@asynccontextmanager
async def get_storage():
    """Get instance of Storage (defaults to production) with proper async context management"""
    async with get_production_storage() as storage:
        yield storage