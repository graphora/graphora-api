from typing import List, Tuple, Optional, Any
import asyncio
from pathlib import Path

import aiofiles
from app.services.chunking.chunker import DocumentChunker, ChunkingError
from app.services.chunking.models import ChunkMetadata, ChunkingResult
from app.services.chunking.config import ChunkingConfig, ChunkingStrategy
from app.config import settings
from app.utils.logger import logger
from prefect import task


@task(
    name="document-chunking",
    retries=settings.CHUNKING_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
)
async def chunk_document(
    file_path: str,
    transform_id: str,
    config: Optional[ChunkingConfig] = None,
    strategy_override: Optional[ChunkingStrategy] = None,
) -> Tuple[ChunkingResult, List[ChunkMetadata]]:
    """
    Chunk document using configurable chunking strategies

    Args:
        file_path: Path to document file
        transform_id: Transform ID
        config: Optional chunking configuration
        strategy_override: Optional strategy override

    Returns:
        Tuple of (ChunkingResult, List[ChunkMetadata])
    """
    try:
        # Initialize chunker with provided configuration
        chunker = DocumentChunker(config=config)

        # Read file content
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()

        logger.info(
            "Starting document chunking with strategy: %s",
            strategy_override or (config.strategy if config else "hybrid"),
        )
        result = await chunker.process_document(
            content, transform_id, strategy_override=strategy_override
        )

        if not result:
            raise ChunkingError("Chunking failed")

        return result

    except Exception as e:
        logger.error(
            "Chunking failed: %s",
            str(e),
            extra={"transform_id": transform_id, "file_path": str(file_path)},
        )
        raise ChunkingError(f"Failed to chunk document: {str(e)}")


@task(
    name="chunk-quality-check",
    retries=settings.CHUNKING_RETRIES,
    retry_delay_seconds=settings.RETRY_DELAY_SECONDS,
)
async def check_chunk_quality(chunks: List[ChunkMetadata]) -> bool:
    """
    Check quality of document chunks

    Args:
        chunks: List of chunk metadata

    Returns:
        True if chunks pass quality check, False otherwise
    """
    try:
        if not chunks:
            return False

        # Check basic quality metrics
        min_coherence = 0.6
        min_relevance = 0.6
        min_size_score = 0.5

        for chunk in chunks:
            if (
                chunk.quality_metrics.coherence_score < min_coherence
                or chunk.quality_metrics.relevance_score < min_relevance
                or chunk.quality_metrics.size_score < min_size_score
            ):
                return False

        return True

    except Exception as e:
        logger.error(f"Quality check failed: {str(e)}")
        return False


async def chunk_documents(
    transform_id: str,
    processed_paths: List[str],
    chunking_config: Optional[Any] = None,
) -> List[Tuple[ChunkingResult, List[ChunkMetadata]]]:
    """Chunk multiple documents concurrently with deterministic strategy selection."""

    if not processed_paths:
        return []

    semaphore = asyncio.Semaphore(settings.CHUNKING_MAX_CONCURRENCY)
    results: List[Tuple[ChunkingResult, List[ChunkMetadata]]] = []
    failures: List[Tuple[str, str]] = []

    async def _chunk(path: str):
        strategy_override: Optional[ChunkingStrategy] = None
        suffix = Path(path).suffix.lower()
        if suffix in {".md", ".markdown", ".txt"}:
            strategy_override = ChunkingStrategy.STRUCTURAL

        async with semaphore:
            return (
                path,
                await chunk_document(
                    file_path=Path(path),
                    transform_id=transform_id,
                    config=chunking_config,
                    strategy_override=strategy_override,
                ),
            )

    task_map = {asyncio.create_task(_chunk(path)): path for path in processed_paths}

    for future in asyncio.as_completed(task_map):
        source_path = task_map[future]
        try:
            path, (chunk_result, chunk_metadata) = await future
            if chunk_result and chunk_metadata:
                logger.info(
                    "Chunked %s into %s segments",
                    Path(path).name,
                    len(chunk_metadata),
                )
                results.append((chunk_result, chunk_metadata))
            else:
                failures.append((path, "No chunks produced"))
        except ChunkingError as exc:
            failures.append((source_path, str(exc)))
        except Exception as exc:  # pragma: no cover - defensive
            failures.append((source_path, str(exc)))

    if failures:
        logger.warning(
            "Chunking completed with %s failure(s)",
            len(failures),
            extra={"transform_id": transform_id, "failures": failures},
        )

    return results
