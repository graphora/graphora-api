import pytest
from pathlib import Path
import tempfile
import os
import json
from unittest.mock import Mock, patch

from app.services.chunking.tasks import (
    chunk_document,
    check_chunk_quality,
    store_chunking_result
)
from app.services.chunking.models import ChunkingResult, ChunkMetadata
from app.config import settings

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_document(temp_dir):
    content = """
    This is a test document.
    It has multiple lines.
    We will use it for testing the chunking tasks.
    Some lines are short.
    Others are longer and contain more information.
    """
    file_path = temp_dir / "test.txt"
    with open(file_path, "w") as f:
        f.write(content)
    return file_path

@pytest.fixture
def mock_logger():
    return Mock()

@pytest.fixture
def sample_chunking_result():
    return ChunkingResult(
        transform_id="test_transform",
        chunks=["Chunk 1", "Chunk 2", "Chunk 3"],
        chunk_metadata=[
            ChunkMetadata(
                chunk_id=f"chunk_{i}",
                start_pos=i*10,
                end_pos=(i+1)*10,
                content_hash="hash"
            )
            for i in range(3)
        ],
        metrics={
            "total_chunks": 3,
            "avg_chunk_size": 7,
            "max_chunk_size": 10,
            "min_chunk_size": 5
        },
        timestamp="2025-02-17T14:00:00Z"
    )

async def test_chunk_document(sample_document, mock_logger):
    """Test document chunking task"""
    with patch("app.services.chunking.tasks.get_run_logger", return_value=mock_logger):
        chunk_paths = await chunk_document(sample_document, "test_transform")
    
    assert chunk_paths is not None
    assert len(chunk_paths) > 0
    assert all(isinstance(p, Path) for p in chunk_paths)
    assert mock_logger.info.call_count >= 2

async def test_store_chunking_result(temp_dir, sample_chunking_result):
    """Test storing chunking results"""
    chunk_paths = await store_chunking_result(
        sample_chunking_result,
        "test_transform",
        temp_dir
    )
    
    assert len(chunk_paths) == len(sample_chunking_result.chunks)
    assert all(p.exists() for p in chunk_paths)
    
    # Check metadata file
    metadata_path = temp_dir / "test_transform" / "chunking_metadata.json"
    assert metadata_path.exists()
    
    with open(metadata_path) as f:
        metadata = json.load(f)
        assert metadata["transform_id"] == "test_transform"
        assert len(metadata["chunks"]) == 3

async def test_check_chunk_quality(temp_dir, mock_logger):
    """Test chunk quality checking"""
    # Create test chunks
    chunks_dir = temp_dir / "test_transform" / "chunks"
    chunks_dir.mkdir(parents=True)
    
    # Valid chunks
    for i in range(3):
        with open(chunks_dir / f"chunk_{i}.txt", "w") as f:
            f.write("A" * settings.MIN_CHUNK_SIZE)
    
    # Create metadata
    metadata = {
        "total_chunks": 3,
        "metrics": {
            "min_chunk_size": settings.MIN_CHUNK_SIZE,
            "max_chunk_size": settings.MIN_CHUNK_SIZE
        }
    }
    
    with open(temp_dir / "test_transform" / "chunking_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    with patch("app.services.chunking.tasks.get_run_logger", return_value=mock_logger):
        result = await check_chunk_quality(
            [chunks_dir / f"chunk_{i}.txt" for i in range(3)],
            "test_transform"
        )
    
    assert result is True

async def test_chunk_quality_failure(temp_dir, mock_logger):
    """Test quality check failure cases"""
    chunks_dir = temp_dir / "test_transform" / "chunks"
    chunks_dir.mkdir(parents=True)
    
    # Create invalid chunk (too small)
    with open(chunks_dir / "chunk_0.txt", "w") as f:
        f.write("A" * (settings.MIN_CHUNK_SIZE - 1))
    
    metadata = {
        "total_chunks": 1,
        "metrics": {
            "min_chunk_size": settings.MIN_CHUNK_SIZE,
            "max_chunk_size": settings.MAX_CHUNK_SIZE
        }
    }
    
    with open(temp_dir / "test_transform" / "chunking_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    with patch("app.services.chunking.tasks.get_run_logger", return_value=mock_logger):
        result = await check_chunk_quality(
            [chunks_dir / "chunk_0.txt"],
            "test_transform"
        )
    
    assert result is False
    assert mock_logger.warning.call_count >= 1

async def test_error_handling(temp_dir, mock_logger):
    """Test error handling in tasks"""
    # Non-existent file
    with patch("app.services.chunking.tasks.get_run_logger", return_value=mock_logger):
        with pytest.raises(Exception):
            await chunk_document(
                temp_dir / "nonexistent.txt",
                "test_transform"
            )
    
    assert mock_logger.error.call_count >= 1
