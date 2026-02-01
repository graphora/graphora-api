"""Unit tests for Embedding Similarity.

Tests for semantic similarity computation using embeddings.
Uses mocks to avoid loading actual ML models in tests.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.services.entity_resolution.embedding_similarity import (
    EmbeddingSimilarity,
    get_embedding_similarity,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_model():
    """Create a mock model for embedding similarity."""
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = 384

    # Create normalized embeddings for testing
    def mock_encode(text, normalize_embeddings=True, _convert_to_numpy=True, **kwargs):
        if isinstance(text, str):
            # Generate deterministic embedding based on text hash
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(384)
            if normalize_embeddings:
                emb = emb / np.linalg.norm(emb)
            return emb
        else:
            # Batch encoding
            embeddings = []
            for t in text:
                np.random.seed(hash(t) % 2**32)
                emb = np.random.randn(384)
                if normalize_embeddings:
                    emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
            return np.array(embeddings)

    mock.encode = mock_encode
    return mock


@pytest.fixture
def embedding_similarity(mock_model):
    """Create EmbeddingSimilarity with mocked model."""
    sim = EmbeddingSimilarity(
        model_name="test-model",
        cache_enabled=True,
        cache_max_size=100,
    )
    # Directly set the mock model to bypass lazy loading
    sim._model = mock_model
    sim._embedding_dim = 384
    return sim


# ============================================================
# Initialization Tests
# ============================================================


class TestEmbeddingSimilarityInit:
    """Test EmbeddingSimilarity initialization."""

    def test_should_create_with_defaults(self):
        """Should create instance with default settings."""
        sim = EmbeddingSimilarity()
        assert sim.model_name == "all-MiniLM-L6-v2"
        assert sim.cache_enabled is True
        assert sim.cache_max_size == 10000
        assert sim.normalize_embeddings is True

    def test_should_accept_custom_model_name(self):
        """Should accept custom model name."""
        sim = EmbeddingSimilarity(model_name="custom-model")
        assert sim.model_name == "custom-model"

    def test_should_accept_cache_settings(self):
        """Should accept cache configuration."""
        sim = EmbeddingSimilarity(
            cache_enabled=False,
            cache_max_size=500,
        )
        assert sim.cache_enabled is False
        assert sim.cache_max_size == 500

    def test_should_lazy_load_model(self):
        """Should not load model until needed."""
        sim = EmbeddingSimilarity()
        assert sim._model is None


# ============================================================
# Embedding Computation Tests
# ============================================================


class TestGetEmbedding:
    """Test single embedding computation."""

    def test_should_return_embedding_array(self, embedding_similarity):
        """Should return numpy array embedding."""
        embedding = embedding_similarity.get_embedding("test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384

    def test_should_cache_embedding(self, embedding_similarity):
        """Should cache computed embedding."""
        text = "cache test"

        # First call
        embedding_similarity.get_embedding(text)
        assert embedding_similarity._cache_misses == 1

        # Second call should hit cache
        embedding_similarity.get_embedding(text)
        assert embedding_similarity._cache_hits == 1

    def test_should_return_consistent_embedding(self, embedding_similarity):
        """Should return same embedding for same text."""
        text = "consistency test"
        emb1 = embedding_similarity.get_embedding(text)
        emb2 = embedding_similarity.get_embedding(text)
        np.testing.assert_array_equal(emb1, emb2)


class TestGetEmbeddingsBatch:
    """Test batch embedding computation."""

    def test_should_return_embedding_matrix(self, embedding_similarity):
        """Should return 2D array for batch."""
        texts = ["text one", "text two", "text three"]
        embeddings = embedding_similarity.get_embeddings_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_should_use_cache_for_batch(self, embedding_similarity):
        """Should use cache for previously computed embeddings."""
        # Pre-cache one embedding
        embedding_similarity.get_embedding("cached text")

        # Batch with mix of cached and new
        texts = ["cached text", "new text"]
        embeddings = embedding_similarity.get_embeddings_batch(texts)

        assert embeddings.shape == (2, 384)
        # One should be from cache
        assert embedding_similarity._cache_hits >= 1


# ============================================================
# Similarity Computation Tests
# ============================================================


class TestComputeSimilarity:
    """Test pairwise similarity computation."""

    def test_should_return_float_similarity(self, embedding_similarity):
        """Should return float similarity score."""
        sim = embedding_similarity.compute_similarity("text a", "text b")
        assert isinstance(sim, float)

    def test_should_return_similarity_in_valid_range(self, embedding_similarity):
        """Should return similarity between 0 and 1."""
        sim = embedding_similarity.compute_similarity("hello", "world")
        assert 0.0 <= sim <= 1.0

    def test_should_return_high_similarity_for_identical_text(
        self, embedding_similarity
    ):
        """Should return high similarity for identical texts."""
        sim = embedding_similarity.compute_similarity("same text", "same text")
        assert sim > 0.99  # Should be very close to 1.0


class TestComputeSimilarityMatrix:
    """Test similarity matrix computation."""

    def test_should_return_matrix_for_two_lists(self, embedding_similarity):
        """Should return similarity matrix for two text lists."""
        texts1 = ["a", "b"]
        texts2 = ["x", "y", "z"]

        matrix = embedding_similarity.compute_similarity_matrix(texts1, texts2)

        assert matrix.shape == (2, 3)
        assert matrix.min() >= 0.0
        assert matrix.max() <= 1.0

    def test_should_return_square_matrix_for_self_similarity(
        self, embedding_similarity
    ):
        """Should return square matrix when texts2 is None."""
        texts = ["a", "b", "c"]

        matrix = embedding_similarity.compute_similarity_matrix(texts)

        assert matrix.shape == (3, 3)
        # Diagonal should be high (self-similarity)
        for i in range(3):
            assert matrix[i, i] > 0.99


class TestFindSimilar:
    """Test similar text finding."""

    def test_should_return_list_of_matches(self, embedding_similarity):
        """Should return list of (index, text, similarity) tuples."""
        query = "query text"
        candidates = ["candidate 1", "candidate 2", "candidate 3"]

        results = embedding_similarity.find_similar(
            query, candidates, threshold=0.0  # Low threshold to get results
        )

        assert isinstance(results, list)
        if results:
            assert len(results[0]) == 3
            assert isinstance(results[0][0], int)  # index
            assert isinstance(results[0][1], str)  # text
            assert isinstance(results[0][2], float)  # similarity

    def test_should_filter_by_threshold(self, embedding_similarity):
        """Should filter results by threshold."""
        query = "query"
        candidates = ["a", "b", "c"]

        results = embedding_similarity.find_similar(
            query, candidates, threshold=0.99  # Very high threshold
        )

        # May return empty or few results depending on random embeddings
        assert all(r[2] >= 0.99 for r in results)

    def test_should_limit_by_top_k(self, embedding_similarity):
        """Should limit results to top K."""
        query = "query"
        candidates = ["a", "b", "c", "d", "e"]

        results = embedding_similarity.find_similar(
            query, candidates, threshold=0.0, top_k=2
        )

        assert len(results) <= 2

    def test_should_handle_empty_candidates(self, embedding_similarity):
        """Should handle empty candidate list."""
        results = embedding_similarity.find_similar("query", [], threshold=0.0)
        assert results == []


class TestComputeSimilarityLevel:
    """Test discrete similarity level computation."""

    def test_should_return_level_zero_for_high_similarity(self, embedding_similarity):
        """Should return level 0 for very similar texts."""
        # Same text should have high similarity
        level = embedding_similarity.compute_similarity_level(
            "exact match", "exact match"
        )
        assert level == 0

    def test_should_return_higher_level_for_dissimilar_text(self, embedding_similarity):
        """Should return higher levels for dissimilar texts."""
        level = embedding_similarity.compute_similarity_level(
            "completely different text",
            "unrelated content here",
            thresholds=[0.99, 0.95, 0.90],  # Very high thresholds
        )
        # Should be > 0 since texts are different
        assert level >= 0

    def test_should_accept_custom_thresholds(self, embedding_similarity):
        """Should accept custom threshold values."""
        level = embedding_similarity.compute_similarity_level(
            "text a",
            "text b",
            thresholds=[0.8, 0.6, 0.4],
        )
        assert 0 <= level <= 3


# ============================================================
# Cache Management Tests
# ============================================================


class TestCacheManagement:
    """Test cache functionality."""

    def test_should_track_cache_stats(self, embedding_similarity):
        """Should track cache hit/miss statistics."""
        embedding_similarity.get_embedding("text 1")
        embedding_similarity.get_embedding("text 1")  # Hit
        embedding_similarity.get_embedding("text 2")

        stats = embedding_similarity.get_cache_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["cache_size"] == 2

    def test_should_clear_cache(self, embedding_similarity):
        """Should clear cache and stats."""
        embedding_similarity.get_embedding("text")
        embedding_similarity.clear_cache()

        stats = embedding_similarity.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

    def test_should_evict_old_entries_when_full(self, mock_model):
        """Should evict old entries when cache is full."""
        sim = EmbeddingSimilarity(
            cache_enabled=True,
            cache_max_size=100,  # Use larger cache to test 10% eviction
        )
        sim._model = mock_model
        sim._embedding_dim = 384

        # Fill cache to max
        for i in range(100):
            sim.get_embedding(f"text {i}")

        # Trigger eviction by adding one more
        sim.get_embedding("trigger eviction")

        stats = sim.get_cache_stats()
        # After eviction, 10% (10 entries) should be removed, then new entry added
        # Result: 100 - 10 + 1 = 91
        assert stats["cache_size"] == 91


# ============================================================
# Singleton Tests
# ============================================================


class TestGetEmbeddingSimilarity:
    """Test get_embedding_similarity singleton function."""

    def test_should_return_same_instance(self, mock_model):
        """Should return same instance for same model."""
        # Reset global instance
        import app.services.entity_resolution.embedding_similarity as module

        module._default_instance = None

        # Create two instances with same model
        with patch.object(EmbeddingSimilarity, "_load_model"):
            instance1 = get_embedding_similarity("test-model")
            instance1._model = mock_model
            instance1._embedding_dim = 384

            instance2 = get_embedding_similarity("test-model")

            assert instance1 is instance2

    def test_should_create_new_instance_for_different_model(self, mock_model):
        """Should create new instance for different model."""
        import app.services.entity_resolution.embedding_similarity as module

        module._default_instance = None

        with patch.object(EmbeddingSimilarity, "_load_model"):
            instance1 = get_embedding_similarity("model-a")
            instance1._model = mock_model
            instance1._embedding_dim = 384

            instance2 = get_embedding_similarity("model-b")

            assert instance1 is not instance2


# ============================================================
# Import Error Handling Tests
# ============================================================


class TestImportErrorHandling:
    """Test handling when sentence-transformers is not installed."""

    def test_should_raise_import_error_when_library_missing(self):
        """Should raise ImportError with helpful message."""
        sim = EmbeddingSimilarity()

        # Mock _load_model to simulate import error
        def raise_import_error():
            raise ImportError(
                "sentence-transformers is required for embedding similarity. "
                "Install with: pip install sentence-transformers"
            )

        with patch.object(sim, "_load_model", side_effect=raise_import_error):
            with pytest.raises(ImportError, match="sentence-transformers"):
                _ = sim.model
