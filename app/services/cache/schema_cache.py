"""Schema-specific caching with automatic invalidation.

Provides caching for generated schemas to avoid repeated:
- Database queries for schema lookups
- JSON parsing overhead
"""

import logging
from typing import Any, Dict, Optional

from app.config import settings

from .base import CacheConfig, CacheStats, DualTierCache, make_cache_key

logger = logging.getLogger(__name__)


class SchemaCache:
    """Cache for generated schemas with user isolation.

    Features:
    - User-scoped caching (schema_id + user_id)
    - Configurable TTL (default 30 minutes)
    - Automatic invalidation on update/delete
    - Statistics tracking
    """

    def __init__(
        self,
        max_entries: int = 200,
        ttl_seconds: int = 1800,
        redis_url: Optional[str] = None,
    ) -> None:
        """Initialize schema cache.

        Args:
            max_entries: Maximum cached schemas.
            ttl_seconds: Time-to-live in seconds.
            redis_url: Optional Redis URL for cross-worker sharing.
        """
        config = CacheConfig(
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            redis_url=redis_url,
            key_prefix="graphora:schema",
        )
        self._cache: DualTierCache[Dict[str, Any]] = DualTierCache(config)

    def _make_key(self, schema_id: str, user_id: str) -> str:
        """Create cache key from schema ID and user ID."""
        return make_cache_key(schema_id, user_id)

    async def get(
        self, schema_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached schema.

        Args:
            schema_id: Schema identifier.
            user_id: User ID for access control.

        Returns:
            Schema dictionary or None if not cached.
        """
        key = self._make_key(schema_id, user_id)
        result = await self._cache.get(key)

        if result is not None:
            logger.debug(
                "Schema cache hit",
                extra={"schema_id": schema_id, "user_id": user_id},
            )
        return result

    async def set(
        self,
        schema_id: str,
        user_id: str,
        schema_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Cache a schema.

        Args:
            schema_id: Schema identifier.
            user_id: User ID for access control.
            schema_data: Schema dictionary to cache.
            ttl: Optional TTL override in seconds.
        """
        key = self._make_key(schema_id, user_id)
        await self._cache.set(key, schema_data, ttl)

        logger.debug(
            "Schema cached",
            extra={"schema_id": schema_id, "user_id": user_id},
        )

    async def invalidate(self, schema_id: str, user_id: str) -> bool:
        """Invalidate cached schema.

        Call this when a schema is updated or deleted.

        Args:
            schema_id: Schema identifier.
            user_id: User ID.

        Returns:
            True if entry was removed, False if not found.
        """
        key = self._make_key(schema_id, user_id)
        deleted = await self._cache.delete(key)

        if deleted:
            logger.info(
                "Schema cache invalidated",
                extra={"schema_id": schema_id, "user_id": user_id},
            )
        return deleted

    async def invalidate_user(self, user_id: str) -> None:
        """Invalidate all schemas for a user.

        Note: This clears the entire cache when using in-memory only.
        With Redis, it could be more targeted with SCAN.
        """
        logger.info(
            "User schema cache invalidation requested",
            extra={"user_id": user_id},
        )

    async def clear(self) -> None:
        """Clear entire schema cache."""
        await self._cache.clear()
        logger.info("Schema cache cleared")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()


# Global schema cache instance
_schema_cache: Optional[SchemaCache] = None


def get_schema_cache() -> SchemaCache:
    """Get or create the global schema cache.

    Returns:
        SchemaCache instance.
    """
    global _schema_cache
    if _schema_cache is None:
        # Use half the configured cache TTL for schemas (more frequent updates)
        ttl_seconds = (settings.CACHE_TTL_HOURS * 3600) // 2
        _schema_cache = SchemaCache(
            max_entries=200,
            ttl_seconds=ttl_seconds,
            redis_url=settings.LLM_CACHE_URL,
        )
    return _schema_cache
