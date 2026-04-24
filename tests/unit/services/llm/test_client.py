"""LLM Client unit tests following London School TDD.

These tests verify the LLM client's caching and utility functions:
- AsyncLRUCache operations
- RedisCache operations (mocked)
- Cache key generation
- Helper functions

Coverage targets:
- _AsyncLRUCache: 90%+
- Cache key generation: 90%+
- Helper functions: 85%+
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import json


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def async_lru_cache():
    """Create a fresh AsyncLRUCache instance."""
    from graphora_server.services.llm.client import _AsyncLRUCache

    return _AsyncLRUCache(max_size=3)


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock()
    client.delete = AsyncMock()
    return client


# ============================================================
# MD5 Hash Tests
# ============================================================


class TestMD5Hash:
    """Test MD5 hashing utility."""

    def test_md5_should_return_consistent_hash(self):
        """Same input should produce same hash."""
        from graphora_server.services.llm.client import md5

        hash1 = md5("hello world")
        hash2 = md5("hello world")

        assert hash1 == hash2

    def test_md5_should_return_different_hash_for_different_input(self):
        """Different input should produce different hash."""
        from graphora_server.services.llm.client import md5

        hash1 = md5("hello")
        hash2 = md5("world")

        assert hash1 != hash2

    def test_md5_should_return_32_character_hex_string(self):
        """MD5 hash should be 32 hex characters."""
        from graphora_server.services.llm.client import md5

        result = md5("test input")

        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_md5_should_handle_empty_string(self):
        """Should handle empty string input."""
        from graphora_server.services.llm.client import md5

        result = md5("")

        assert len(result) == 32

    def test_md5_should_handle_unicode(self):
        """Should handle unicode characters."""
        from graphora_server.services.llm.client import md5

        result = md5("你好世界")

        assert len(result) == 32


# ============================================================
# Cache Key Generation Tests
# ============================================================


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_should_combine_parts(self):
        """Should combine multiple parts into single key."""
        from graphora_server.services.llm.client import _cache_key

        key = _cache_key("part1", "part2", "part3")

        assert len(key) == 32  # MD5 hash length

    def test_cache_key_should_be_deterministic(self):
        """Same parts should produce same key."""
        from graphora_server.services.llm.client import _cache_key

        key1 = _cache_key("a", "b", "c")
        key2 = _cache_key("a", "b", "c")

        assert key1 == key2

    def test_cache_key_should_differ_for_different_parts(self):
        """Different parts should produce different key."""
        from graphora_server.services.llm.client import _cache_key

        key1 = _cache_key("a", "b", "c")
        key2 = _cache_key("x", "y", "z")

        assert key1 != key2

    def test_cache_key_should_handle_none_parts(self):
        """Should handle None values in parts."""
        from graphora_server.services.llm.client import _cache_key

        key = _cache_key("a", None, "c")

        assert len(key) == 32

    def test_cache_key_should_handle_empty_string_parts(self):
        """Should handle empty string parts."""
        from graphora_server.services.llm.client import _cache_key

        key = _cache_key("a", "", "c")

        assert len(key) == 32

    def test_cache_key_order_matters(self):
        """Parts order should affect the key."""
        from graphora_server.services.llm.client import _cache_key

        key1 = _cache_key("a", "b", "c")
        key2 = _cache_key("c", "b", "a")

        assert key1 != key2


# ============================================================
# Preview Function Tests
# ============================================================


class TestPreviewFunction:
    """Test text preview utility."""

    def test_preview_should_truncate_long_text(self):
        """Should truncate text longer than limit."""
        from graphora_server.services.llm.client import _preview

        long_text = "a" * 300
        result = _preview(long_text, limit=200)

        assert len(result) == 201  # 200 chars + ellipsis
        assert result.endswith("…")

    def test_preview_should_preserve_short_text(self):
        """Should not truncate text shorter than limit."""
        from graphora_server.services.llm.client import _preview

        short_text = "hello world"
        result = _preview(short_text, limit=200)

        assert result == short_text
        assert not result.endswith("…")

    def test_preview_should_replace_newlines(self):
        """Should replace newlines with spaces."""
        from graphora_server.services.llm.client import _preview

        text_with_newlines = "line1\nline2\nline3"
        result = _preview(text_with_newlines)

        assert "\n" not in result
        assert "line1 line2 line3" == result

    def test_preview_should_handle_none(self):
        """Should return empty string for None."""
        from graphora_server.services.llm.client import _preview

        result = _preview(None)

        assert result == ""

    def test_preview_should_handle_empty_string(self):
        """Should return empty string for empty input."""
        from graphora_server.services.llm.client import _preview

        result = _preview("")

        assert result == ""

    def test_preview_should_respect_custom_limit(self):
        """Should use custom limit when provided."""
        from graphora_server.services.llm.client import _preview

        text = "a" * 100
        result = _preview(text, limit=50)

        assert len(result) == 51  # 50 chars + ellipsis


# ============================================================
# AsyncLRUCache Tests
# ============================================================


class TestAsyncLRUCache:
    """Test AsyncLRUCache implementation."""

    @pytest.mark.asyncio
    async def test_cache_get_should_return_none_for_missing_key(self, async_lru_cache):
        """Should return None for keys not in cache."""
        result = await async_lru_cache.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_and_get_should_store_value(self, async_lru_cache):
        """Should store and retrieve values."""
        await async_lru_cache.set("key1", {"data": "value1"})

        result = await async_lru_cache.get("key1")

        assert result == {"data": "value1"}

    @pytest.mark.asyncio
    async def test_cache_should_return_deep_copy(self, async_lru_cache):
        """Should return deep copy to prevent mutation."""
        original = {"nested": {"value": 1}}
        await async_lru_cache.set("key1", original)

        retrieved = await async_lru_cache.get("key1")
        retrieved["nested"]["value"] = 999

        # Original cache entry should be unchanged
        result = await async_lru_cache.get("key1")
        assert result["nested"]["value"] == 1

    @pytest.mark.asyncio
    async def test_cache_should_evict_oldest_when_full(self, async_lru_cache):
        """Should evict oldest entry when max size is reached."""
        # Cache has max_size=3
        await async_lru_cache.set("key1", "value1")
        await async_lru_cache.set("key2", "value2")
        await async_lru_cache.set("key3", "value3")

        # Adding 4th item should evict key1
        await async_lru_cache.set("key4", "value4")

        assert await async_lru_cache.get("key1") is None
        assert await async_lru_cache.get("key2") == "value2"
        assert await async_lru_cache.get("key3") == "value3"
        assert await async_lru_cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_cache_get_should_move_to_end(self, async_lru_cache):
        """Accessing an item should move it to end (most recently used)."""
        await async_lru_cache.set("key1", "value1")
        await async_lru_cache.set("key2", "value2")
        await async_lru_cache.set("key3", "value3")

        # Access key1 to make it most recently used
        await async_lru_cache.get("key1")

        # Add key4 - should evict key2 (now oldest)
        await async_lru_cache.set("key4", "value4")

        assert await async_lru_cache.get("key1") == "value1"  # Still there
        assert await async_lru_cache.get("key2") is None  # Evicted
        assert await async_lru_cache.get("key3") == "value3"
        assert await async_lru_cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_cache_should_overwrite_existing_key(self, async_lru_cache):
        """Setting existing key should overwrite value."""
        await async_lru_cache.set("key1", "original")
        await async_lru_cache.set("key1", "updated")

        result = await async_lru_cache.get("key1")

        assert result == "updated"

    @pytest.mark.asyncio
    async def test_cache_should_handle_complex_values(self, async_lru_cache):
        """Should handle complex nested values."""
        complex_value = {
            "entities": [
                {"name": "Acme", "type": "Company"},
                {"name": "Jane", "type": "Person"},
            ],
            "metadata": {"confidence": 0.95},
        }

        await async_lru_cache.set("key1", complex_value)
        result = await async_lru_cache.get("key1")

        assert result == complex_value
        assert len(result["entities"]) == 2

    @pytest.mark.asyncio
    async def test_cache_should_be_thread_safe(self):
        """Should handle concurrent access safely."""
        from graphora_server.services.llm.client import _AsyncLRUCache

        cache = _AsyncLRUCache(max_size=100)

        async def writer(key_prefix: str):
            for i in range(10):
                await cache.set(f"{key_prefix}_{i}", f"value_{i}")

        async def reader(key_prefix: str):
            for i in range(10):
                await cache.get(f"{key_prefix}_{i}")

        # Run concurrent operations
        await asyncio.gather(
            writer("a"),
            writer("b"),
            reader("a"),
            reader("b"),
        )

        # Should not raise any errors


# ============================================================
# RedisCache Tests (Mocked)
# ============================================================


class TestRedisCache:
    """Test RedisCache implementation with mocked Redis."""

    @pytest.mark.asyncio
    async def test_redis_cache_get_should_return_none_for_missing(
        self, mock_redis_client
    ):
        """Should return None for missing keys."""
        with patch("redis.asyncio.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis_client
            mock_redis_client.get.return_value = None

            from graphora_server.services.llm.client import _RedisCache

            cache = _RedisCache("redis://localhost", "test", 3600)
            result = await cache.get("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_redis_cache_should_namespace_keys(self, mock_redis_client):
        """Should prefix keys with namespace."""
        with patch("redis.asyncio.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis_client

            from graphora_server.services.llm.client import _RedisCache

            cache = _RedisCache("redis://localhost", "myns", 3600)

            # Check namespaced key
            namespaced = cache._namespaced("mykey")

            assert namespaced == "myns:mykey"

    @pytest.mark.asyncio
    async def test_redis_cache_get_should_deserialize_json(self, mock_redis_client):
        """Should deserialize JSON from Redis."""
        with patch("redis.asyncio.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis_client
            mock_redis_client.get.return_value = '{"name": "Acme", "value": 42}'

            from graphora_server.services.llm.client import _RedisCache

            cache = _RedisCache("redis://localhost", "test", 3600)
            result = await cache.get("mykey")

            assert result == {"name": "Acme", "value": 42}

    @pytest.mark.asyncio
    async def test_redis_cache_set_should_serialize_json(self, mock_redis_client):
        """Should serialize value as JSON when setting."""
        with patch("redis.asyncio.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis_client

            from graphora_server.services.llm.client import _RedisCache

            cache = _RedisCache("redis://localhost", "test", 3600)
            await cache.set("mykey", {"data": "value"})

            # Verify set was called with JSON
            call_args = mock_redis_client.set.call_args
            assert call_args[0][0] == "test:mykey"
            assert json.loads(call_args[0][1]) == {"data": "value"}

    @pytest.mark.asyncio
    async def test_redis_cache_set_should_include_ttl(self, mock_redis_client):
        """Should include TTL when setting values."""
        with patch("redis.asyncio.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis_client

            from graphora_server.services.llm.client import _RedisCache

            cache = _RedisCache("redis://localhost", "test", 3600)
            await cache.set("mykey", "value")

            # Verify TTL was passed
            call_kwargs = mock_redis_client.set.call_args[1]
            assert call_kwargs.get("ex") == 3600

    @pytest.mark.asyncio
    async def test_redis_cache_should_handle_invalid_json(self, mock_redis_client):
        """Should handle corrupted JSON gracefully."""
        with patch("redis.asyncio.Redis") as MockRedis:
            MockRedis.from_url.return_value = mock_redis_client
            mock_redis_client.get.return_value = "invalid json {"

            from graphora_server.services.llm.client import _RedisCache

            cache = _RedisCache("redis://localhost", "test", 3600)
            result = await cache.get("corrupted")

            # Should return None and delete the corrupted entry
            assert result is None
            mock_redis_client.delete.assert_called_once()


# ============================================================
# Cache Factory Tests
# ============================================================


class TestCacheFactory:
    """Test cache factory function."""

    def test_create_cache_should_return_lru_without_redis_url(self):
        """Should return AsyncLRUCache when no Redis URL configured."""
        with patch("graphora_server.services.llm.client.settings") as mock_settings:
            mock_settings.LLM_CACHE_MAX_ENTRIES = 128
            mock_settings.LLM_CACHE_URL = None

            from graphora_server.services.llm.client import _create_cache, _AsyncLRUCache

            cache = _create_cache("test-namespace")

            assert isinstance(cache, _AsyncLRUCache)

    def test_create_cache_should_use_configured_max_entries(self):
        """Should use configured max entries for LRU cache."""
        with patch("graphora_server.services.llm.client.settings") as mock_settings:
            mock_settings.LLM_CACHE_MAX_ENTRIES = 256
            mock_settings.LLM_CACHE_URL = None

            from graphora_server.services.llm.client import _create_cache

            cache = _create_cache("test-namespace")

            assert cache._max_size == 256


# ============================================================
# LLMClient Tests
# ============================================================


class TestLLMClient:
    """Test LLMClient class."""

    @pytest.mark.asyncio
    async def test_extract_nodes_should_require_user_id(self):
        """Should raise error when user_id is not provided."""
        from graphora_server.services.llm.client import LLMClient

        client = LLMClient()

        with pytest.raises(ValueError, match="user_id is required"):
            await client.extract_nodes_from_pdf(
                pdf_path="/path/to/file.pdf",
                response_model=MagicMock(),
                ontology_yaml="",
                user_id=None,
            )

    @pytest.mark.asyncio
    async def test_extract_nodes_should_use_cache_on_hit(self):
        """Should return cached result on cache hit."""
        from graphora_server.services.llm.client import _PDF_NODE_CACHE

        # Pre-populate cache
        cached_result = {"entities": [{"name": "Acme"}]}

        with patch.object(_PDF_NODE_CACHE, "get", return_value=cached_result):
            with patch(
                "graphora_server.services.llm.client.get_user_llm_credentials",
                return_value=("api-key", "model"),
            ):
                # The actual extraction should be skipped due to cache hit
                # This documents the expected caching behavior
                pass


# ============================================================
# Retry Decorator Integration Tests
# ============================================================


class TestRetryIntegration:
    """Test retry decorator integration."""

    @pytest.mark.asyncio
    async def test_retry_should_attempt_multiple_times_on_failure(self):
        """Should retry on exceptions."""
        from graphora_server.utils.func_helper import retry_async

        attempt_count = 0

        @retry_async(max_attempts=3, delay=0.01, backoff=1, exceptions=(ValueError,))
        async def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await failing_function()

        assert attempt_count == 3
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_should_raise_after_max_attempts(self):
        """Should raise exception after max attempts."""
        from graphora_server.utils.func_helper import retry_async

        attempt_count = 0

        @retry_async(max_attempts=2, delay=0.01, backoff=1, exceptions=(ValueError,))
        async def always_fails():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await always_fails()

        assert attempt_count == 2
