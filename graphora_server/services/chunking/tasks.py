from typing import List, Tuple, Optional, Any
import asyncio
import re
from pathlib import Path

import aiofiles
from graphora_server.services.chunking.chunker import DocumentChunker, ChunkingError
from graphora_server.services.chunking.models import ChunkMetadata, ChunkingResult
from graphora_server.services.chunking.config import ChunkingConfig, ChunkingStrategy
from graphora_server.config import settings
from graphora_server.utils.logger import logger
from prefect import task


# Pattern matches the per-page split filenames emitted by
# graphora_server/services/transform/flows.py::split_pdf
# (``page_<uuid>_<n>.pdf``). The trailing 1-based page index is the
# only payload we need; uuid and prefix are positional anchors.
_PDF_PAGE_FILENAME_RE = re.compile(r"^page_[a-f0-9]+_(\d+)\.pdf$", re.IGNORECASE)


def _page_number_from_path(path: Path) -> Optional[int]:
    """Extract the 1-based page number from a PDF split filename.

    Returns None for any path that doesn't match the split-pdf
    convention — non-PDF inputs, future format changes, or PDFs that
    weren't split (e.g., single-page documents that bypass split_pdf).
    Callers treat None as "page is unknown" rather than an error.
    """
    match = _PDF_PAGE_FILENAME_RE.match(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover — regex guarantees digits
        return None


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

        # A1-prov: stamp per-chunk provenance on every ChunkMetadata
        # so downstream extraction can copy these onto node/edge
        # properties. The chunker itself doesn't know the source path
        # or whether the chunk came from a per-page PDF split — we
        # add that here, where file_path is the authoritative source
        # of truth.
        chunk_result, chunk_metadatas = result
        path = Path(file_path)
        source_file = path.name
        page_number = _page_number_from_path(path)
        for cm in chunk_metadatas:
            if cm.source_file is None:
                cm.source_file = source_file
            if cm.page_number is None and page_number is not None:
                cm.page_number = page_number
        if hasattr(chunk_result, "chunk_metadata") and chunk_result.chunk_metadata:
            for cm in chunk_result.chunk_metadata:
                if cm.source_file is None:
                    cm.source_file = source_file
                if cm.page_number is None and page_number is not None:
                    cm.page_number = page_number

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
        # Allow hybrid/semantic chunking for all file types to reduce chunk counts
        # and improve performance (especially for .txt files)
        strategy_override: Optional[ChunkingStrategy] = None

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
