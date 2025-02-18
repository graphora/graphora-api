import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.services.llm.cache import (
    ExtractionCache,
    CacheEntry
)
from app.services.llm.client import call_llm_gemini

# Test-specific Redis settings
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use DB 15 for testing
TEST_CACHE_TTL = 1  # 1 hour for faster testing

@pytest.fixture
def redis_mock():
    """Mock Redis client"""
    with patch('redis.from_url') as mock:
        mock.return_value = Mock()
        # Setup basic Redis mock functionality
        mock.return_value.ping.return_value = True
        mock.return_value.get.return_value = None
        mock.return_value.keys.return_value = []
        yield mock.return_value

@pytest.fixture
def cache(redis_mock):
    """Test cache instance"""
    return ExtractionCache(
        redis_url=TEST_REDIS_URL,
        ttl_hours=TEST_CACHE_TTL
    )

def test_cache_key_generation(cache, test_models):
    """Test cache key generation"""
    text = "Sample text"
    model_name = "test-model"
    
    # Generate key
    schema_hash = cache._compute_schema_hash(test_models)
    key = cache._compute_cache_key(text, model_name, schema_hash)
    
    # Verify key format
    assert key.startswith("extraction:")
    assert len(key) > 20
    
    # Verify key consistency
    key2 = cache._compute_cache_key(text, model_name, schema_hash)
    assert key == key2
    
    # Verify key differs with different inputs
    key3 = cache._compute_cache_key("Different text", model_name, schema_hash)
    assert key != key3

def test_schema_hash_generation(cache, test_models):
    """Test schema hash generation"""
    hash1 = cache._compute_schema_hash(test_models)
    
    # Verify hash format
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hash
    
    # Verify hash consistency
    hash2 = cache._compute_schema_hash(test_models)
    assert hash1 == hash2
    
    # Verify hash differs with different schema
    modified_models = test_models.copy()
    modified_models["NewEntity"] = test_models["TestEntity"]
    hash3 = cache._compute_schema_hash(modified_models)
    assert hash1 != hash3

def test_cache_set_get(cache, redis_mock, test_models):
    """Test setting and getting cache entries"""
    text = "Test text"
    model_name = "test-model"
    extraction = {"TestEntity": [{"name": "Test", "age": 30}]}
    confidence = 0.9
    
    # Set cache entry
    cache.set(
        text,
        model_name,
        test_models,
        extraction,
        confidence
    )
    
    # Verify Redis call
    assert redis_mock.setex.called
    call_args = redis_mock.setex.call_args[0]
    assert call_args[0].startswith("extraction:")
    
    # Setup mock for get
    entry = CacheEntry(
        extraction=extraction,
        timestamp=datetime.now(),
        model_name=model_name,
        confidence_score=confidence
    )
    redis_mock.get.return_value = entry.model_dump_json()
    
    # Get cache entry
    result = cache.get(text, model_name, test_models)
    
    # Verify result
    assert result == extraction
    assert redis_mock.get.called

def test_cache_expiration(cache, redis_mock, test_models):
    """Test cache entry expiration"""
    text = "Test text"
    model_name = "test-model"
    
    # Create expired entry
    entry = CacheEntry(
        extraction={"TestEntity": []},
        timestamp=datetime.now() - timedelta(hours=25),
        model_name=model_name,
        confidence_score=0.8
    )
    redis_mock.get.return_value = entry.model_dump_json()
    
    # Try to get expired entry
    result = cache.get(text, model_name, test_models)
    
    # Verify entry was not returned
    assert result is None
    assert redis_mock.delete.called

def test_cache_statistics(cache, redis_mock):
    """Test cache statistics collection"""
    # Setup mock data
    keys = [f"extraction:key{i}" for i in range(3)]
    entries = [
        CacheEntry(
            extraction={"TestEntity": []},
            timestamp=datetime.now() - timedelta(hours=i),
            model_name=f"model{i}",
            confidence_score=0.8 + i/10
        )
        for i in range(3)
    ]
    
    redis_mock.keys.return_value = keys
    redis_mock.get.side_effect = [
        entry.model_dump_json() for entry in entries
    ]
    
    # Get stats
    stats = cache.get_stats()
    
    # Verify stats
    assert stats["total_entries"] == 3
    assert 0.8 <= stats["avg_confidence"] <= 1.0
    assert len(stats["model_distribution"]) == 3
    assert len(stats["age_distribution"]) == 4
    assert sum(stats["age_distribution"].values()) == 1.0

def test_llm_integration(test_data, test_models):
    """Test LLM integration with cache"""
    # First call should miss cache
    result1 = call_llm_gemini(
        test_data[0]["text"],
        test_models,
        use_cache=True
    )
    
    # Second call should hit cache
    result2 = call_llm_gemini(
        test_data[0]["text"],
        test_models,
        use_cache=True
    )
    
    # Results should be identical
    assert result1 == result2
    
    # Call without cache should give fresh result
    result3 = call_llm_gemini(
        test_data[0]["text"],
        test_models,
        use_cache=False
    )
    
    # Results might differ slightly due to LLM non-determinism
    assert isinstance(result3, dict)

def test_cache_error_handling(cache, redis_mock, test_models):
    """Test cache error handling"""
    # Simulate Redis errors
    redis_mock.get.side_effect = Exception("Redis error")
    redis_mock.setex.side_effect = Exception("Redis error")
    
    # Get should return None on error
    result = cache.get(
        "test",
        "test-model",
        test_models
    )
    assert result is None
    
    # Set should not raise error
    cache.set(
        "test",
        "test-model",
        test_models,
        {"TestEntity": []},
        0.8
    )
    
    # Stats should return empty dict on error
    redis_mock.keys.side_effect = Exception("Redis error")
    stats = cache.get_stats()
    assert stats == {}
