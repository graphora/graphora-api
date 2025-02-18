import pytest

from app.services.chunking.chunker import (
    HybridChunker,
    ChunkingError
)
from app.services.chunking.models import ChunkingResult

@pytest.fixture
def sample_text():
    return """
    This is a sample text for testing the chunking functionality. 
    It contains multiple sentences with varying lengths.
    Some sentences are short.
    Others are much longer and contain more detailed information that needs to be processed carefully.
    We also include some technical terms like Python, FastAPI, and Neural Networks.
    Finally, we add some numbers 123, 456, 789 to test different content types.
    """.strip()

@pytest.fixture
def long_text():
    # Generate a long text with 100 sentences
    sentences = []
    for i in range(100):
        sentences.append(f"This is test sentence number {i+1} with some varying content.")
    return " ".join(sentences)

@pytest.fixture
def chunker():
    return HybridChunker(
        max_chunk_size=200,
        min_chunk_size=50,
        semantic_threshold=0.7
    )

async def test_basic_chunking(chunker, sample_text):
    """Test basic chunking functionality"""
    result = await chunker.process_document(sample_text, "test_transform")
    
    assert isinstance(result, ChunkingResult)
    assert len(result.chunks) > 0
    assert all(50 <= len(chunk) <= 200 for chunk in result.chunks)
    assert result.transform_id == "test_transform"
    assert result.metrics is not None

async def test_semantic_breakpoints(chunker, sample_text):
    """Test semantic breakpoint detection"""
    semantic_chunker = chunker.semantic_chunker
    breakpoints = semantic_chunker.find_breakpoints(sample_text)
    
    assert isinstance(breakpoints, list)
    assert all(isinstance(bp, int) for bp in breakpoints)
    assert all(bp < len(sample_text) for bp in breakpoints)

async def test_large_document(chunker, long_text):
    """Test chunking of large documents"""
    result = await chunker.process_document(long_text, "test_large")
    
    assert len(result.chunks) > 10  # Should create multiple chunks
    assert result.metrics["total_chunks"] > 10
    assert result.metrics["peak_memory_mb"] > 0

async def test_edge_cases(chunker):
    """Test edge cases"""
    # Empty text
    with pytest.raises(ChunkingError):
        await chunker.process_document("", "test_empty")
    
    # Single short sentence
    result = await chunker.process_document("Short text.", "test_short")
    assert len(result.chunks) == 1
    
    # Very long sentence
    long_sentence = "word " * 1000
    result = await chunker.process_document(long_sentence, "test_long_sentence")
    assert all(len(chunk) <= chunker.max_chunk_size for chunk in result.chunks)

async def test_chunk_metadata(chunker, sample_text):
    """Test chunk metadata generation"""
    result = await chunker.process_document(sample_text, "test_metadata")
    
    assert len(result.chunk_metadata) == len(result.chunks)
    for metadata in result.chunk_metadata:
        assert metadata.chunk_id is not None
        assert metadata.start_pos >= 0
        assert metadata.end_pos > metadata.start_pos
        assert metadata.content_hash is not None

async def test_memory_tracking(chunker, long_text):
    """Test memory usage tracking"""
    result = await chunker.process_document(long_text, "test_memory")
    
    assert result.metrics["peak_memory_mb"] > 0
    assert isinstance(result.metrics["peak_memory_mb"], float)

@pytest.mark.parametrize("chunk_size", [100, 200, 500])
async def test_chunk_size_constraints(chunk_size, sample_text):
    """Test different chunk size constraints"""
    chunker = HybridChunker(
        max_chunk_size=chunk_size,
        min_chunk_size=chunk_size // 4
    )
    result = await chunker.process_document(sample_text, "test_size")
    
    assert all(len(chunk) <= chunk_size for chunk in result.chunks)
    assert all(len(chunk) >= chunk_size // 4 for chunk in result.chunks)

async def test_forced_splits(chunker):
    """Test forced splitting of long content"""
    # Create text without natural breakpoints
    text = "word" * 1000
    result = await chunker.process_document(text, "test_forced")
    
    assert result.metrics["forced_splits"] > 0
    assert all(len(chunk) <= chunker.max_chunk_size for chunk in result.chunks)

async def test_semantic_similarity(chunker):
    """Test semantic similarity calculations"""
    text = """
    Python is a programming language.
    FastAPI is a web framework.
    Neural networks are used in AI.
    Machine learning is a subset of AI.
    """
    
    semantic_chunker = chunker.semantic_chunker
    embeddings = semantic_chunker._compute_embeddings([
        "Python is a programming language",
        "FastAPI is a web framework",
        "Neural networks are used in AI",
        "Machine learning is a subset of AI"
    ])
    
    # Similar sentences should have higher similarity
    sim1 = semantic_chunker._compute_similarity(embeddings[2], embeddings[3])
    sim2 = semantic_chunker._compute_similarity(embeddings[0], embeddings[2])
    
    assert sim1 > sim2  # AI-related sentences should be more similar
