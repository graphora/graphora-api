"""Base caching infrastructure with TTL support and optional Redis backing.

Provides a dual-tier caching system:
- Primary: In-memory LRU cache for fast access
- Optional: Redis backing for cross-worker sharing
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    max_entries: int = 1000
    ttl_seconds: int = 3600  # 1 hour default
    redis_url: Optional[str] = None
    key_prefix: str = "graphora"
    enable_stats: bool = True


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    last_access: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "hit_rate": f"{self.hit_rate:.2%}",
            "last_access": self.last_access,
        }


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with TTL tracking."""

    value: T
    created_at: float
    ttl_seconds: int

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds


class AsyncCache(ABC, Generic[T]):
    """Abstract base class for async caches."""

    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL override."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries from cache."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class InMemoryCache(AsyncCache[T]):
    """Thread-safe in-memory LRU cache with TTL support.

    Features:
    - LRU eviction when max_entries is reached
    - TTL-based expiration
    - Statistics tracking
    - Async-compatible (uses asyncio.Lock)
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize in-memory cache.

        Args:
            config: Cache configuration. Uses defaults if not provided.
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache, returning None if not found or expired."""
        async with self._lock:
            self._stats.last_access = time.time()

            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            # Move to end for LRU ordering
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return entry.value

    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL override."""
        async with self._lock:
            ttl_seconds = ttl if ttl is not None else self.config.ttl_seconds

            # Remove if exists to reset position
            if key in self._cache:
                del self._cache[key]

            # Evict oldest entries if at capacity
            while len(self._cache) >= self.config.max_entries:
                self._cache.popitem(last=False)
                self._stats.evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl_seconds,
            )
            self._stats.size = len(self._cache)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries from cache."""
        async with self._lock:
            self._cache.clear()
            self._stats.size = 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            self._stats.size = len(self._cache)
            return len(expired_keys)


class RedisCache(AsyncCache[T]):
    """Redis-backed cache for cross-worker sharing.

    Falls back gracefully to no-op when Redis is unavailable.
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize Redis cache.

        Args:
            config: Cache configuration with redis_url.
        """
        self.config = config or CacheConfig()
        self._redis: Any = None
        self._stats = CacheStats()
        self._initialized = False

    async def _get_redis(self) -> Any:
        """Get or create Redis connection."""
        if self._redis is None and self.config.redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                )
                self._initialized = True
            except ImportError:
                logger.warning("redis package not installed, cache disabled")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        return self._redis

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.config.key_prefix}:{key}"

    async def get(self, key: str) -> Optional[T]:
        """Get value from Redis cache."""
        redis = await self._get_redis()
        if redis is None:
            self._stats.misses += 1
            return None

        try:
            full_key = self._make_key(key)
            data = await redis.get(full_key)

            if data is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            self._stats.last_access = time.time()
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            self._stats.misses += 1
            return None

    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        redis = await self._get_redis()
        if redis is None:
            return

        try:
            full_key = self._make_key(key)
            ttl_seconds = ttl if ttl is not None else self.config.ttl_seconds

            await redis.set(
                full_key,
                json.dumps(value, default=str),
                ex=ttl_seconds,
            )
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        redis = await self._get_redis()
        if redis is None:
            return False

        try:
            full_key = self._make_key(key)
            result = await redis.delete(full_key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False

    async def clear(self) -> None:
        """Clear all entries with this cache's prefix."""
        redis = await self._get_redis()
        if redis is None:
            return

        try:
            pattern = f"{self.config.key_prefix}:*"
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class DualTierCache(AsyncCache[T]):
    """Dual-tier cache combining in-memory and Redis.

    Reads check in-memory first, then Redis (populating in-memory on hit).
    Writes go to both tiers.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        memory_cache: Optional[InMemoryCache[T]] = None,
        redis_cache: Optional[RedisCache[T]] = None,
    ) -> None:
        """Initialize dual-tier cache.

        Args:
            config: Cache configuration.
            memory_cache: Optional pre-configured memory cache.
            redis_cache: Optional pre-configured Redis cache.
        """
        self.config = config or CacheConfig()
        self._memory = memory_cache or InMemoryCache[T](self.config)
        self._redis = redis_cache

        if self._redis is None and self.config.redis_url:
            self._redis = RedisCache[T](self.config)

    async def get(self, key: str) -> Optional[T]:
        """Get from memory first, then Redis."""
        # Check memory cache first
        value = await self._memory.get(key)
        if value is not None:
            return value

        # Check Redis if available
        if self._redis is not None:
            value = await self._redis.get(key)
            if value is not None:
                # Populate memory cache
                await self._memory.set(key, value)
                return value

        return None

    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set in both tiers."""
        await self._memory.set(key, value, ttl)
        if self._redis is not None:
            await self._redis.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete from both tiers."""
        memory_deleted = await self._memory.delete(key)
        redis_deleted = False
        if self._redis is not None:
            redis_deleted = await self._redis.delete(key)
        return memory_deleted or redis_deleted

    async def clear(self) -> None:
        """Clear both tiers."""
        await self._memory.clear()
        if self._redis is not None:
            await self._redis.clear()

    def get_stats(self) -> CacheStats:
        """Get combined statistics."""
        memory_stats = self._memory.get_stats()
        redis_stats = self._redis.get_stats() if self._redis else CacheStats()

        return CacheStats(
            hits=memory_stats.hits + redis_stats.hits,
            misses=memory_stats.misses,  # Only count final misses
            evictions=memory_stats.evictions,
            size=memory_stats.size,
            last_access=max(memory_stats.last_access, redis_stats.last_access),
        )


def make_cache_key(*parts: Any) -> str:
    """Create a cache key from multiple parts.

    Args:
        *parts: Components to include in the key.

    Returns:
        MD5 hash of the concatenated parts.
    """
    key_data = ":".join(str(part) for part in parts)
    return hashlib.md5(key_data.encode()).hexdigest()


# Global cache instances
_caches: Dict[str, AsyncCache[Any]] = {}
_cache_lock = asyncio.Lock()


async def get_cache(
    name: str,
    config: Optional[CacheConfig] = None,
) -> AsyncCache[Any]:
    """Get or create a named cache instance.

    Args:
        name: Unique name for this cache.
        config: Cache configuration.

    Returns:
        Cache instance.
    """
    async with _cache_lock:
        if name not in _caches:
            cfg = config or CacheConfig(key_prefix=f"graphora:{name}")
            _caches[name] = DualTierCache(cfg)
        return _caches[name]
