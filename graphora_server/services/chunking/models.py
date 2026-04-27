from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


class ChunkQualityMetrics(BaseModel):
    """Quality metrics for a document chunk"""

    chunk_id: Optional[str] = None
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    size_score: float = 0.0
    # Additional fields for compatibility with hybrid chunker
    token_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    semantic_score: float = 0.0


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

    # Core identification
    transform_id: str = ""
    chunk_id: Optional[str] = None
    chunk_index: int = 0

    # Source provenance (A1-prov). Populated by the chunker when the
    # source file is known. ``page_number`` is set when the chunker
    # can derive it (currently: PDF chunks whose split filename
    # follows the ``page_<uuid>_<n>.pdf`` pattern emitted by
    # split_pdf in flows.py).
    source_file: Optional[str] = None
    page_number: Optional[int] = None

    # Position and content
    start_position: int = 0  # Renamed for consistency
    end_position: int = 0  # Renamed for consistency
    chunk_size: int = 0  # Added for size tracking
    chunk_hash: str = ""  # Renamed from content_hash

    # Quality and processing
    semantic_score: Optional[float] = None
    is_forced_split: bool = False
    is_merged: bool = False
    quality_metrics: Optional[ChunkQualityMetrics] = None
    processing_metrics: Optional[ChunkProcessingMetrics] = None
    processing_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Content analysis
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
    chunks: List[str] = Field(default_factory=list)  # Default empty list
    num_chunks: int = 0  # Added for compatibility
    total_tokens: int = 0  # Added for compatibility
    semantic_processing_time: float = 0.0  # Added for compatibility
    chunk_processing_time: float = 0.0  # Added for compatibility
    chunk_metadata: List[ChunkMetadata] = Field(
        default_factory=list
    )  # Default empty list
    metrics: Dict[str, Any] = Field(default_factory=dict)  # Default empty dict
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )  # Default to current ISO timestamp
    worker_metrics: Optional[List[ChunkProcessingMetrics]] = None
