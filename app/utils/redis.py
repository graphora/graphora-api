"""Redis utility functions for the application"""
from pydantic import BaseModel
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
    
def is_serializable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
    
def to_json(model: BaseModel) -> str: 
    data = model.model_dump()
    serializable_data = {k: v for k, v in data.items() if is_serializable(v)}
    json_output = json.dumps(serializable_data)
    
    return json_output

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO string
        return super().default(obj)  # Fall back to default for other types