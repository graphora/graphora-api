"""Unit tests for the ontology cache."""

import pytest

from app.services.cache import OntologyCache


class TestOntologyCache:
    """Tests for ontology-specific caching."""

    @pytest.fixture
    def cache(self):
        """Create a test ontology cache."""
        return OntologyCache(max_entries=10, ttl_seconds=60)

    @pytest.mark.asyncio
    async def test_should_cache_ontology_by_id(self, cache):
        """Cache should store and retrieve ontology by ID."""
        ontology_data = {
            "name": "test_ontology",
            "entities": {"Person": {"properties": {"name": {"type": "string"}}}},
        }

        await cache.set("test-ontology", ontology_data)
        result = await cache.get("test-ontology")

        assert result == ontology_data
        assert result["name"] == "test_ontology"

    @pytest.mark.asyncio
    async def test_should_isolate_by_user_id(self, cache):
        """Cache should isolate ontologies by user ID."""
        ontology_user1 = {"name": "user1_ontology"}
        ontology_user2 = {"name": "user2_ontology"}

        await cache.set("shared-id", ontology_user1, user_id="user1")
        await cache.set("shared-id", ontology_user2, user_id="user2")

        result1 = await cache.get("shared-id", user_id="user1")
        result2 = await cache.get("shared-id", user_id="user2")

        assert result1["name"] == "user1_ontology"
        assert result2["name"] == "user2_ontology"

    @pytest.mark.asyncio
    async def test_should_return_none_for_uncached(self, cache):
        """Cache should return None for uncached ontologies."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_should_invalidate_cached_ontology(self, cache):
        """Invalidation should remove cached ontology."""
        ontology_data = {"name": "test"}
        await cache.set("test-ontology", ontology_data, user_id="user1")

        deleted = await cache.invalidate("test-ontology", user_id="user1")
        assert deleted is True

        result = await cache.get("test-ontology", user_id="user1")
        assert result is None

    @pytest.mark.asyncio
    async def test_should_track_statistics(self, cache):
        """Cache should track hit/miss statistics."""
        ontology_data = {"name": "test"}
        await cache.set("test-ontology", ontology_data)

        # Hit
        await cache.get("test-ontology")
        # Miss
        await cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1

    @pytest.mark.asyncio
    async def test_should_clear_all_entries(self, cache):
        """Clear should remove all cached ontologies."""
        await cache.set("ontology1", {"name": "one"})
        await cache.set("ontology2", {"name": "two"})

        await cache.clear()

        assert await cache.get("ontology1") is None
        assert await cache.get("ontology2") is None
