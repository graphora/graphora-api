from typing import List, Tuple, Optional
import traceback

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from app.services.chunking.models import (
    ChunkingResult,
    ChunkMetadata,
    ChunkQualityMetrics,
)
from app.services.chunking.hybrid_chunker import HybridDocumentChunker, ChunkingStrategy
from app.services.chunking.config import ChunkingConfig, DEFAULT_CONFIGS
from app.utils.logger import logger
from app.config import settings

# Global embedding model cache - initialized once per process
_embedding_cache = None
_text_splitter_cache = None


def _get_cached_embeddings():
    """Get or create cached embedding model"""
    global _embedding_cache
    if _embedding_cache is None:
        logger.info(f"Initializing embedding model cache: {settings.EMBEDDING_MODEL}")
        _embedding_cache = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        logger.info("Embedding model cached successfully")
    return _embedding_cache


def _get_cached_text_splitter():
    """Get or create cached semantic text splitter"""
    global _text_splitter_cache
    if _text_splitter_cache is None:
        logger.info("Initializing semantic text splitter cache")
        embeddings = _get_cached_embeddings()
        _text_splitter_cache = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="gradient",
            min_chunk_size=settings.MIN_SEMANTIC_CHUNK_SIZE,
        )
        logger.info("Semantic text splitter cached successfully")
    return _text_splitter_cache


def _create_semantic_chunker_with_config(config: ChunkingConfig):
    """Create a semantic chunker with specific configuration"""
    embeddings = _get_cached_embeddings()

    # Create semantic chunker with configuration
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",  # Use percentile for better threshold control
        breakpoint_threshold_amount=config.semantic_threshold,
        min_chunk_size=config.min_chunk_size,
    )

    logger.info(
        f"Created semantic chunker with threshold={config.semantic_threshold}, min_size={config.min_chunk_size}"
    )
    return semantic_chunker


class DocumentChunker:
    """Advanced document chunker with multiple strategies"""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker with configurable strategy"""
        try:
            # Use provided config or default hybrid config
            self.config = config or DEFAULT_CONFIGS["hybrid"]

            # Initialize semantic components if needed
            if self.config.strategy in [
                ChunkingStrategy.SEMANTIC,
                ChunkingStrategy.HYBRID,
            ]:
                self.embeddings = _get_cached_embeddings()
                # Use config-specific semantic chunker instead of cached version
                self.text_splitter = _create_semantic_chunker_with_config(self.config)
                logger.info(
                    f"Created semantic chunker with min_chunk_size={self.config.min_chunk_size}"
                )
            else:
                self.embeddings = None
                self.text_splitter = None

            # Initialize hybrid chunker with config
            self.hybrid_chunker = HybridDocumentChunker(
                semantic_chunker=self.text_splitter,
                strategy=self.config.strategy,
                config=self.config,
            )

            logger.info(f"Initialized chunker with {self.config.strategy} strategy")
        except Exception as e:
            logger.error(f"Failed to initialize chunker: {str(e)}")
            traceback.print_exc()
            raise

    def _compute_chunk_quality(self, chunk_text: str) -> ChunkQualityMetrics:
        """Compute quality metrics for a chunk"""
        words = chunk_text.split()
        sentences = [s for s in chunk_text.split(".") if s.strip()]

        return ChunkQualityMetrics(
            token_count=len(words),
            sentence_count=len(sentences),
            avg_sentence_length=len(words) / max(len(sentences), 1),
            semantic_score=1.0,  # Set by SemanticChunker's internal scoring
        )

    async def process_document(
        self,
        text: str,
        transform_id: str,
        strategy_override: Optional[ChunkingStrategy] = None,
    ) -> Tuple[ChunkingResult, List[ChunkMetadata]]:
        """Process document using configured or hybrid chunking strategy"""
        try:
            logger.info(f"Processing document with {len(text)} characters")

            # Use hybrid chunker for intelligent processing
            result, metadata = await self.hybrid_chunker.process_document(
                text=text,
                transform_id=transform_id,
                strategy_override=strategy_override,
            )

            # Log results for debugging
            logger.info(f"Generated {result.num_chunks} chunks")
            logger.info(f"Chunk sizes: {[len(c) for c in result.chunks]}")

            return result, metadata

        except Exception as e:
            logger.error(f"Failed to process document: {str(e)}")
            traceback.print_exc()
            raise


class ChunkingError(Exception):
    """Base exception for chunking errors"""

    pass
