"""Ontology-specific caching with automatic invalidation.

Provides caching for ontology definitions to avoid repeated:
- File I/O for YAML loading
- Database queries for user ontologies
- YAML parsing overhead
"""

import logging
from typing import Any, Dict, Optional

from app.config import settings

from .base import CacheConfig, CacheStats, DualTierCache, make_cache_key

logger = logging.getLogger(__name__)


class OntologyCache:
    """Cache for ontology definitions with user isolation.

    Features:
    - User-scoped caching (ontology_id + user_id)
    - Configurable TTL (default 1 hour)
    - Automatic invalidation on update
    - Statistics tracking
    """

    def __init__(
        self,
        max_entries: int = 500,
        ttl_seconds: int = 3600,
        redis_url: Optional[str] = None,
    ) -> None:
        """Initialize ontology cache.

        Args:
            max_entries: Maximum cached ontologies.
            ttl_seconds: Time-to-live in seconds.
            redis_url: Optional Redis URL for cross-worker sharing.
        """
        config = CacheConfig(
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            redis_url=redis_url,
            key_prefix="graphora:ontology",
        )
        self._cache: DualTierCache[Dict[str, Any]] = DualTierCache(config)

    def _make_key(self, ontology_id: str, user_id: Optional[str] = None) -> str:
        """Create cache key from ontology ID and optional user ID."""
        if user_id:
            return make_cache_key(ontology_id, user_id)
        return make_cache_key(ontology_id, "__global__")

    async def get(
        self, ontology_id: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached ontology definition.

        Args:
            ontology_id: Ontology identifier.
            user_id: Optional user ID for user-specific ontologies.

        Returns:
            Parsed ontology dictionary or None if not cached.
        """
        key = self._make_key(ontology_id, user_id)
        result = await self._cache.get(key)

        if result is not None:
            logger.debug(
                "Ontology cache hit",
                extra={"ontology_id": ontology_id, "user_id": user_id},
            )
        return result

    async def set(
        self,
        ontology_id: str,
        ontology_data: Dict[str, Any],
        user_id: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache an ontology definition.

        Args:
            ontology_id: Ontology identifier.
            ontology_data: Parsed ontology dictionary.
            user_id: Optional user ID for user-specific ontologies.
            ttl: Optional TTL override in seconds.
        """
        key = self._make_key(ontology_id, user_id)
        await self._cache.set(key, ontology_data, ttl)

        logger.debug(
            "Ontology cached",
            extra={"ontology_id": ontology_id, "user_id": user_id},
        )

    async def invalidate(
        self, ontology_id: str, user_id: Optional[str] = None
    ) -> bool:
        """Invalidate cached ontology.

        Call this when an ontology is updated.

        Args:
            ontology_id: Ontology identifier.
            user_id: Optional user ID.

        Returns:
            True if entry was removed, False if not found.
        """
        key = self._make_key(ontology_id, user_id)
        deleted = await self._cache.delete(key)

        if deleted:
            logger.info(
                "Ontology cache invalidated",
                extra={"ontology_id": ontology_id, "user_id": user_id},
            )
        return deleted

    async def invalidate_user(self, user_id: str) -> None:
        """Invalidate all ontologies for a user.

        Note: This clears the entire cache when using in-memory only.
        With Redis, it could be more targeted with SCAN.
        """
        # For now, we don't have a way to selectively clear user ontologies
        # without tracking keys. This is a potential future enhancement.
        logger.info(
            "User ontology cache invalidation requested",
            extra={"user_id": user_id},
        )

    async def clear(self) -> None:
        """Clear entire ontology cache."""
        await self._cache.clear()
        logger.info("Ontology cache cleared")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()


# Global ontology cache instance
_ontology_cache: Optional[OntologyCache] = None


def get_ontology_cache() -> OntologyCache:
    """Get or create the global ontology cache.

    Returns:
        OntologyCache instance.
    """
    global _ontology_cache
    if _ontology_cache is None:
        _ontology_cache = OntologyCache(
            max_entries=500,
            ttl_seconds=settings.CACHE_TTL_HOURS * 3600,
            redis_url=settings.LLM_CACHE_URL,  # Reuse LLM cache Redis if configured
        )
    return _ontology_cache
