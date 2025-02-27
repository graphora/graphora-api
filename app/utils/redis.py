"""Redis utility functions for the application"""
import redis.asyncio as redis
from app.config import settings
import json
from datetime import datetime
from typing import Any

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

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO string
        return super().default(obj)  # Fall back to default for other types