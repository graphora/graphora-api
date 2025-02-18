from typing import Dict, Any, Optional
import hashlib
from datetime import datetime, timedelta
import redis
from pydantic import BaseModel

from app.config import settings

class CacheEntry(BaseModel):
    """Cache entry for LLM extractions"""
    extraction: Dict[str, Any]
    timestamp: datetime
    model_name: str
    confidence_score: float
    token_usage: Optional[Dict[str, int]] = None

class ExtractionCache:
    """Cache for LLM extractions"""
    
    def __init__(
        self,
        redis_url: str | None = None,
        ttl_hours: int | None = None
    ):
        """Initialize cache with settings"""
        try:
            self.redis = redis.from_url(
                redis_url or settings.REDIS_URL,
                decode_responses=True
            )
            # Test connection
            self.redis.ping()
        except redis.ConnectionError as e:
            print(f"Redis connection error: {str(e)}")
            self.redis = None
        
        self.ttl = timedelta(
            hours=ttl_hours or settings.CACHE_TTL_HOURS
        )
    
    def _compute_cache_key(
        self,
        text: str,
        model_name: str,
        schema_hash: str
    ) -> str:
        """Compute cache key for text and schema"""
        # Normalize text
        normalized_text = " ".join(text.lower().split())
        
        # Create composite key
        key_parts = [
            normalized_text,
            model_name,
            schema_hash
        ]
        
        # Hash the key
        key_str = "|".join(key_parts)
        return f"extraction:{hashlib.sha256(key_str.encode()).hexdigest()}"
    
    def _compute_schema_hash(
        self,
        models: Dict[str, BaseModel]
    ) -> str:
        """Compute hash of schema definition"""
        schema_dict = {
            name: {
                field: str(field_info.annotation)
                for field, field_info in model.model_fields.items()
                if field not in ['id', 'type', 'provenance']
            }
            for name, model in models.items()
        }
        return hashlib.sha256(
            str(sorted(schema_dict.items())).encode()
        ).hexdigest()
    
    def get(
        self,
        text: str,
        model_name: str,
        models: Dict[str, BaseModel]
    ) -> Optional[Dict[str, Any]]:
        """Get cached extraction if available"""
        if not self.redis:
            return None
            
        try:
            # Compute cache key
            schema_hash = self._compute_schema_hash(models)
            cache_key = self._compute_cache_key(
                text,
                model_name,
                schema_hash
            )
            
            # Get from cache
            cached = self.redis.get(cache_key)
            if not cached:
                return None
            
            # Parse entry
            entry = CacheEntry.model_validate_json(cached)
            
            # Check if expired
            if datetime.now() - entry.timestamp > self.ttl:
                self.redis.delete(cache_key)
                return None
            
            return entry.extraction
            
        except Exception as e:
            print(f"Cache get error: {str(e)}")
            return None
    
    def set(
        self,
        text: str,
        model_name: str,
        models: Dict[str, BaseModel],
        extraction: Dict[str, Any],
        confidence_score: float,
        token_usage: Optional[Dict[str, int]] = None
    ) -> None:
        """Cache extraction result"""
        if not self.redis:
            return
            
        try:
            # Compute cache key
            schema_hash = self._compute_schema_hash(models)
            cache_key = self._compute_cache_key(
                text,
                model_name,
                schema_hash
            )
            
            # Create cache entry
            entry = CacheEntry(
                extraction=extraction,
                timestamp=datetime.now(),
                model_name=model_name,
                confidence_score=confidence_score,
                token_usage=token_usage
            )
            
            # Store in cache
            self.redis.setex(
                cache_key,
                int(self.ttl.total_seconds()),
                entry.model_dump_json()
            )
            
        except Exception as e:
            print(f"Cache set error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis:
            return {}
            
        try:
            # Get all cache keys
            keys = self.redis.keys("extraction:*")
            
            # Collect stats
            total_entries = len(keys)
            if total_entries == 0:
                return {
                    "total_entries": 0,
                    "avg_confidence": 0.0,
                    "model_distribution": {},
                    "age_distribution": {
                        "1h": 0,
                        "6h": 0,
                        "24h": 0,
                        "older": 0
                    }
                }
            
            # Process entries
            confidence_scores = []
            model_counts = {}
            age_distribution = {
                "1h": 0,
                "6h": 0,
                "24h": 0,
                "older": 0
            }
            
            for key in keys:
                entry_data = self.redis.get(key)
                if entry_data:
                    entry = CacheEntry.model_validate_json(entry_data)
                    
                    # Confidence
                    confidence_scores.append(entry.confidence_score)
                    
                    # Model distribution
                    model_counts[entry.model_name] = (
                        model_counts.get(entry.model_name, 0) + 1
                    )
                    
                    # Age distribution
                    age = datetime.now() - entry.timestamp
                    if age <= timedelta(hours=1):
                        age_distribution["1h"] += 1
                    elif age <= timedelta(hours=6):
                        age_distribution["6h"] += 1
                    elif age <= timedelta(hours=24):
                        age_distribution["24h"] += 1
                    else:
                        age_distribution["older"] += 1
            
            return {
                "total_entries": total_entries,
                "avg_confidence": (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores else 0.0
                ),
                "model_distribution": {
                    model: count / total_entries
                    for model, count in model_counts.items()
                },
                "age_distribution": {
                    age: count / total_entries
                    for age, count in age_distribution.items()
                }
            }
            
        except Exception as e:
            print(f"Cache stats error: {str(e)}")
            return {}
