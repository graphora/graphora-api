"""Redis utility functions for the application"""
import redis.asyncio as redis
from app.config import settings

async def get_redis_client() -> redis.Redis:
    """
    Get a Redis client instance
    
    Returns:
        redis.Redis: Redis client instance
    """
    return redis.Redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    ) 