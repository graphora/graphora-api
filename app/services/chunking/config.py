from pydantic import BaseModel, Field
from enum import Enum


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""

    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    RECURSIVE = "recursive"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""

    # Strategy selection
    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.HYBRID, description="Chunking strategy to use"
    )

    # Size constraints
    min_chunk_size: int = Field(
        default=500, ge=100, le=2000, description="Minimum chunk size in characters"
    )

    max_chunk_size: int = Field(
        default=3000, ge=1000, le=10000, description="Maximum chunk size in characters"
    )

    # Semantic chunking options
    semantic_threshold: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Semantic similarity threshold for chunking",
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="HuggingFace embedding model for semantic chunking",
    )

    # Structural chunking options
    preserve_lists: bool = Field(
        default=True, description="Keep list items together in chunks"
    )

    preserve_headings: bool = Field(
        default=True, description="Keep headings with their content"
    )

    preserve_quotes: bool = Field(
        default=True, description="Keep quotes as complete units"
    )

    # Recursive chunking options
    chunk_overlap: int = Field(
        default=200, ge=0, le=500, description="Character overlap between chunks"
    )

    # Advanced options
    force_strategy: bool = Field(
        default=False,
        description="Force use of specified strategy (disable auto-detection)",
    )

    quality_threshold: float = Field(
        default=0.6, ge=0.1, le=1.0, description="Minimum quality score for chunks"
    )


# User preferences removed - using analyze-then-pass approach instead

# Default configurations for different document types
DEFAULT_CONFIGS = {
    "structured": ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURAL,
        min_chunk_size=300,
        max_chunk_size=2000,
        preserve_lists=True,
        preserve_headings=True,
        preserve_quotes=True,
    ),
    "narrative": ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        min_chunk_size=1000,
        max_chunk_size=4000,
        semantic_threshold=0.75,
        chunk_overlap=150,
    ),
    "technical": ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        min_chunk_size=800,
        max_chunk_size=3500,
        chunk_overlap=250,
        preserve_headings=True,
    ),
    "hybrid": ChunkingConfig(
        strategy=ChunkingStrategy.HYBRID,
        min_chunk_size=500,
        max_chunk_size=3000,
        semantic_threshold=0.7,
        preserve_lists=True,
        preserve_headings=True,
        preserve_quotes=True,
        chunk_overlap=200,
    ),
}
