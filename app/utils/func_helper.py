import asyncio
import random
from functools import wraps

from app.utils.logger import logger

# Retry decorator
def retry_async(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 1
            last_exception = None
            while attempt <= max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        raise last_exception  # Re-raise the last exception
                    sleep_time = delay * (backoff ** (attempt - 1)) + random.uniform(0, 0.1)
                    logger.debug(f"Attempt {attempt} failed with {e}. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    attempt += 1
        return wrapper
    return decorator
