"""Unified caching infrastructure for Graphora API.

This module provides:
- TTL-based in-memory caching with optional Redis backing
- Specialized caches for ontologies, schemas, and query results
- Cache statistics and monitoring
"""

from .base import (
    AsyncCache,
    CacheConfig,
    CacheEntry,
    CacheStats,
    DualTierCache,
    InMemoryCache,
    RedisCache,
    get_cache,
    make_cache_key,
)
from .ontology_cache import OntologyCache, get_ontology_cache
from .schema_cache import SchemaCache, get_schema_cache

__all__ = [
    # Base cache
    "AsyncCache",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "DualTierCache",
    "InMemoryCache",
    "RedisCache",
    "get_cache",
    "make_cache_key",
    # Specialized caches
    "OntologyCache",
    "get_ontology_cache",
    "SchemaCache",
    "get_schema_cache",
]
