import pytest

from graphora_server.config import settings
from graphora_server.services.chunking.config import (
    ChunkingConfig,
    ChunkingStrategy as ConfigChunkingStrategy,
)
from graphora_server.services.chunking.hybrid_chunker import HybridDocumentChunker, ChunkingStrategy
from graphora_server.services.transform.graph_transformer import _build_nodes_context
from graphora_server.services.transform.models import BaseNode


def _make_config(max_chunk_size: int) -> ChunkingConfig:
    return ChunkingConfig(
        strategy=ConfigChunkingStrategy.STRUCTURAL,
        min_chunk_size=max(500, max_chunk_size // 2),
        max_chunk_size=max_chunk_size,
        chunk_overlap=0,
        preserve_lists=False,
        preserve_headings=False,
        preserve_quotes=False,
    )


def test_chunk_count_respects_cap(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TXT_CHUNKS", 3)
    config = _make_config(max_chunk_size=1600)
    chunker = HybridDocumentChunker(
        semantic_chunker=None,
        strategy=ChunkingStrategy.STRUCTURAL,
        config=config,
    )

    raw_chunks = [f"Paragraph {i}\n" + ("x" * 300) for i in range(10)]

    bounded = chunker._enforce_size_limits(raw_chunks)
    assert len(bounded) <= 3
    # Re-running should be deterministic
    assert bounded == chunker._enforce_size_limits(raw_chunks)


@pytest.mark.asyncio
async def test_nodes_context_truncation_is_deterministic(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 120)

    nodes = [
        BaseNode(
            id=str(i),
            type="Company",
            properties={
                "name": f"Company {i}",
                "description": "".join(["detail"] * 20),
            },
        )
        for i in range(8)
    ]

    context = await _build_nodes_context(nodes)
    assert len(context) <= 120
    assert "...[truncated]..." in context

    reversed_context = await _build_nodes_context(list(reversed(nodes)))
    assert context == reversed_context
