from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChunkQualityMetrics(BaseModel):
    """Enhanced quality metrics for chunks"""
    semantic_coherence: float  # Average semantic similarity within chunk
    boundary_smoothness: float  # How well chunk boundaries align with natural breaks
    content_density: float  # Ratio of meaningful content to total size
    readability_score: float  # Measure of chunk readability
    topic_consistency: float  # How well the chunk maintains a single topic
    formatting_quality: float  # Quality of text formatting and structure

class ChunkProcessingMetrics(BaseModel):
    """Metrics for parallel processing"""
    worker_id: str
    start_time: datetime
    end_time: datetime
    processing_time_ms: float
    memory_used_mb: float
    cpu_usage_percent: float
    queue_wait_time_ms: float

class ChunkMetadata(BaseModel):
    """Metadata for a single chunk"""
    chunk_id: str
    start_pos: int
    end_pos: int
    content_hash: str
    semantic_score: Optional[float] = None
    is_forced_split: bool = False
    is_merged: bool = False
    quality_metrics: Optional[ChunkQualityMetrics] = None
    processing_metrics: Optional[ChunkProcessingMetrics] = None
    tokens: Optional[int] = None
    sentences: Optional[int] = None
    language_detected: Optional[str] = None

class ChunkingMetrics(BaseModel):
    """Overall chunking metrics"""
    # Performance metrics
    total_processing_time_ms: float
    embedding_time_ms: float
    chunk_refinement_time_ms: float
    embedding_model_latency_ms: float
    parallel_processing_time_ms: float
    worker_count: int
    queue_depth: int
    
    # Document metrics
    original_doc_size: int
    total_chunks: int
    avg_chunk_size: float
    max_chunk_size: int
    min_chunk_size: int
    total_tokens: Optional[int] = None
    
    # Quality metrics
    semantic_splits: int
    forced_splits: int
    merged_chunks: int
    avg_semantic_coherence: float
    avg_boundary_smoothness: float
    avg_content_density: float
    avg_readability_score: float
    topic_consistency_score: float
    
    # Resource metrics
    peak_memory_mb: float
    avg_cpu_usage_percent: float
    total_embedding_calls: int
    cache_hit_rate: float

class ChunkingResult(BaseModel):
    """Result of document chunking process"""
    transform_id: str
    chunks: List[str]
    chunk_metadata: List[ChunkMetadata]
    metrics: Dict[str, Any]
    timestamp: str
    worker_metrics: Optional[List[ChunkProcessingMetrics]] = None
