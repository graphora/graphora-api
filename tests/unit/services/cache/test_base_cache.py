"""Unit tests for the base caching infrastructure."""

import pytest

from graphora_server.services.cache import (
    CacheConfig,
    CacheStats,
    InMemoryCache,
    make_cache_key,
)


class TestCacheKey:
    """Tests for cache key generation."""

    def test_should_generate_consistent_keys(self):
        """Same inputs should produce same key."""
        key1 = make_cache_key("ontology", "user123")
        key2 = make_cache_key("ontology", "user123")
        assert key1 == key2

    def test_should_generate_different_keys_for_different_inputs(self):
        """Different inputs should produce different keys."""
        key1 = make_cache_key("ontology1", "user123")
        key2 = make_cache_key("ontology2", "user123")
        assert key1 != key2


class TestCacheStats:
    """Tests for cache statistics."""

    def test_should_calculate_hit_rate_correctly(self):
        """Hit rate should be hits / total."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8

    def test_should_handle_zero_requests(self):
        """Hit rate should be 0 when no requests."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_should_convert_to_dict(self):
        """Stats should serialize to dictionary."""
        stats = CacheStats(hits=10, misses=5, evictions=2, size=15)
        result = stats.to_dict()
        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["evictions"] == 2
        assert result["size"] == 15
        assert "hit_rate" in result


class TestInMemoryCache:
    """Tests for in-memory cache implementation."""

    @pytest.fixture
    def cache(self):
        """Create a small test cache."""
        config = CacheConfig(max_entries=5, ttl_seconds=60)
        return InMemoryCache[str](config)

    @pytest.mark.asyncio
    async def test_should_store_and_retrieve_values(self, cache):
        """Cache should store and retrieve values."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_should_return_none_for_missing_keys(self, cache):
        """Cache should return None for missing keys."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_should_delete_keys(self, cache):
        """Cache should delete keys."""
        await cache.set("key1", "value1")
        deleted = await cache.delete("key1")
        assert deleted is True
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_should_clear_all_entries(self, cache):
        """Cache should clear all entries."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_should_evict_oldest_when_full(self, cache):
        """Cache should evict oldest entries when full."""
        # Fill cache to capacity
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        # Add one more - should evict key0
        await cache.set("key5", "value5")

        # key0 should be evicted
        assert await cache.get("key0") is None
        # key5 should exist
        assert await cache.get("key5") == "value5"

    @pytest.mark.asyncio
    async def test_should_track_statistics(self, cache):
        """Cache should track hit/miss statistics."""
        await cache.set("key1", "value1")

        # Hit
        await cache.get("key1")
        # Miss
        await cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1

    @pytest.mark.asyncio
    async def test_should_update_lru_order_on_access(self, cache):
        """Accessing a key should move it to end of LRU queue."""
        # Fill cache
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        # Access key0 to make it recently used
        await cache.get("key0")

        # Add new entry - should evict key1 (oldest unaccessed)
        await cache.set("key5", "value5")

        # key0 should still exist
        assert await cache.get("key0") == "value0"
        # key1 should be evicted
        assert await cache.get("key1") is None
