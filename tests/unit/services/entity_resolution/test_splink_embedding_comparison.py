"""Unit tests for Splink Embedding Comparison Factory.

Tests for embedding-aware comparison creation and DataFrame preparation.
Uses mocks to avoid loading actual ML models in tests.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from app.services.entity_resolution.splink_embedding_comparison import (
    EmbeddingAwareComparisonFactory,
    EMBEDDING_PRIOR,
    DEFAULT_EMBEDDING_THRESHOLDS,
    _is_prop_type_text,
    create_embedding_factory,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_embedding_similarity():
    """Create a mock embedding similarity instance."""
    mock = MagicMock()
    mock.embedding_dim = 384
    mock.model_name = "test-model"

    def mock_get_embeddings_batch(texts):
        embeddings = []
        for t in texts:
            np.random.seed(hash(t) % 2**32)
            emb = np.random.randn(384)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)

    mock.get_embeddings_batch = mock_get_embeddings_batch
    return mock


@pytest.fixture
def embedding_factory(mock_embedding_similarity):
    """Create EmbeddingAwareComparisonFactory with mocked embedding similarity."""
    factory = EmbeddingAwareComparisonFactory(
        embedding_model="test-model",
        cache_enabled=True,
    )
    factory._embedding_similarity = mock_embedding_similarity
    return factory


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    # Create as list of dicts to work with MiniDataFrame mock
    data = [
        {
            "id": "1",
            "name": "John Smith",
            "description": "Software engineer with 10 years experience",
            "email": "john@example.com",
        },
        {
            "id": "2",
            "name": "Jane Doe",
            "description": "Data scientist specializing in ML",
            "email": "jane@example.com",
        },
        {
            "id": "3",
            "name": "Bob Wilson",
            "description": "Product manager in tech industry",
            "email": "bob@example.com",
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_ontology():
    """Create a sample ontology with TEXT properties."""
    return {
        "entities": {
            "Person": {
                "properties": {
                    "name": {"type": "string", "unique": True},
                    "description": {"type": "text"},
                    "email": {"type": "email"},
                }
            }
        }
    }


# ============================================================
# Initialization Tests
# ============================================================


class TestEmbeddingAwareComparisonFactoryInit:
    """Test EmbeddingAwareComparisonFactory initialization."""

    def test_should_create_with_defaults(self):
        """Should create instance with default settings."""
        factory = EmbeddingAwareComparisonFactory()
        assert factory.cache_enabled is True
        assert factory.similarity_thresholds == DEFAULT_EMBEDDING_THRESHOLDS

    def test_should_accept_custom_model_name(self):
        """Should accept custom embedding model."""
        factory = EmbeddingAwareComparisonFactory(embedding_model="custom-model")
        assert factory.model_name == "custom-model"

    def test_should_accept_custom_thresholds(self):
        """Should accept custom similarity thresholds."""
        thresholds = [0.9, 0.8, 0.6]
        factory = EmbeddingAwareComparisonFactory(similarity_thresholds=thresholds)
        assert factory.similarity_thresholds == thresholds

    def test_should_lazy_load_embedding_similarity(self):
        """Should not load embedding similarity until needed."""
        factory = EmbeddingAwareComparisonFactory()
        assert factory._embedding_similarity is None


# ============================================================
# Embedding Precomputation Tests
# ============================================================


class TestPrecomputeEmbeddings:
    """Test embedding precomputation."""

    def test_should_return_embeddings_dict(self, embedding_factory):
        """Should return dictionary of embeddings."""
        # Create a simple mock DataFrame with just the needed interface
        mock_df = MagicMock()
        mock_df.columns = ["id", "description"]
        mock_df.__getitem__ = MagicMock(
            return_value=MagicMock(
                fillna=MagicMock(
                    return_value=MagicMock(
                        astype=MagicMock(
                            return_value=MagicMock(
                                tolist=MagicMock(
                                    return_value=["text1", "text2", "text3"]
                                )
                            )
                        )
                    )
                )
            )
        )

        embeddings = embedding_factory.precompute_embeddings(mock_df, ["description"])

        assert isinstance(embeddings, dict)
        assert "description" in embeddings

    def test_should_return_correct_shape(self, embedding_factory):
        """Should return embeddings with correct shape."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "description"]
        mock_df.__getitem__ = MagicMock(
            return_value=MagicMock(
                fillna=MagicMock(
                    return_value=MagicMock(
                        astype=MagicMock(
                            return_value=MagicMock(
                                tolist=MagicMock(
                                    return_value=["text1", "text2", "text3"]
                                )
                            )
                        )
                    )
                )
            )
        )

        embeddings = embedding_factory.precompute_embeddings(mock_df, ["description"])

        assert embeddings["description"].shape == (3, 384)

    def test_should_handle_missing_column(self, embedding_factory):
        """Should handle missing columns gracefully."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "name"]  # No "nonexistent_column"

        embeddings = embedding_factory.precompute_embeddings(
            mock_df, ["nonexistent_column"]
        )

        assert "nonexistent_column" not in embeddings

    def test_should_handle_empty_texts(self, embedding_factory):
        """Should handle empty text values."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "text"]
        mock_df.__getitem__ = MagicMock(
            return_value=MagicMock(
                fillna=MagicMock(
                    return_value=MagicMock(
                        astype=MagicMock(
                            return_value=MagicMock(
                                tolist=MagicMock(return_value=["", "valid text"])
                            )
                        )
                    )
                )
            )
        )

        embeddings = embedding_factory.precompute_embeddings(mock_df, ["text"])

        assert embeddings["text"].shape == (2, 384)

    def test_should_cache_precomputed_embeddings(self, embedding_factory):
        """Should cache precomputed embeddings for reuse."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "description"]
        mock_df.__getitem__ = MagicMock(
            return_value=MagicMock(
                fillna=MagicMock(
                    return_value=MagicMock(
                        astype=MagicMock(
                            return_value=MagicMock(
                                tolist=MagicMock(return_value=["text1", "text2"])
                            )
                        )
                    )
                )
            )
        )

        embedding_factory.precompute_embeddings(mock_df, ["description"])

        assert "description" in embedding_factory._precomputed_embeddings


# ============================================================
# Similarity Level Computation Tests
# ============================================================


class TestComputePairwiseSimilarityLevels:
    """Test pairwise similarity level computation."""

    def test_should_return_integer_levels(self, embedding_factory):
        """Should return matrix of integer levels."""
        np.random.seed(42)
        embeddings = np.random.randn(5, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        levels = embedding_factory.compute_pairwise_similarity_levels(embeddings)

        assert levels.dtype in [np.int64, np.int32, np.intp]
        assert levels.shape == (5, 5)

    def test_should_return_zero_for_self_similarity(self, embedding_factory):
        """Should return level 0 for identical embeddings (diagonal)."""
        # Use explicit unit vectors to ensure similarity = 1.0
        emb1 = np.zeros(384)
        emb1[0] = 1.0  # Unit vector
        embeddings = np.array([emb1, emb1.copy(), emb1.copy()])

        levels = embedding_factory.compute_pairwise_similarity_levels(embeddings)

        # Diagonal should be 0 (exact match with similarity = 1.0)
        assert levels[0, 0] == 0
        # All vectors are identical
        assert levels[0, 1] == 0
        assert levels[1, 2] == 0

    def test_should_respect_thresholds(self, embedding_factory):
        """Should map similarities to correct levels based on thresholds."""
        # Create embeddings with known similarities
        emb1 = np.zeros(384)
        emb1[0] = 1.0  # Unit vector along first axis
        embeddings = np.array([emb1, emb1.copy()])

        levels = embedding_factory.compute_pairwise_similarity_levels(embeddings)

        # emb1 and emb2 are identical -> level 0 (similarity = 1.0 >= 0.95)
        assert levels[0, 1] == 0
        assert levels[1, 0] == 0


# ============================================================
# Embedding Signature Tests
# ============================================================


class TestComputeEmbeddingSignatures:
    """Test LSH signature computation."""

    def test_should_return_string_signatures(self, embedding_factory):
        """Should return list of string signatures."""
        embeddings = np.random.randn(5, 384)

        signatures = embedding_factory._compute_embedding_signatures(embeddings)

        assert isinstance(signatures, list)
        assert len(signatures) == 5
        assert all(isinstance(s, str) for s in signatures)

    def test_should_return_consistent_signatures(self, embedding_factory):
        """Should return same signature for same embedding."""
        embeddings = np.random.randn(3, 384)

        sig1 = embedding_factory._compute_embedding_signatures(embeddings)
        sig2 = embedding_factory._compute_embedding_signatures(embeddings)

        assert sig1 == sig2

    def test_should_return_binary_string(self, embedding_factory):
        """Should return binary string signature."""
        embeddings = np.random.randn(2, 384)

        signatures = embedding_factory._compute_embedding_signatures(embeddings)

        # Signatures should only contain 0 and 1
        for sig in signatures:
            assert all(c in "01" for c in sig)


# ============================================================
# DataFrame Preparation Tests
# ============================================================


class TestAddEmbeddingSimilarityColumn:
    """Test adding embedding columns to DataFrame."""

    def test_should_add_signature_column(self, embedding_factory):
        """Should add embedding signature column."""
        # Create mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ["id", "description"]
        mock_df.__getitem__ = MagicMock(
            return_value=MagicMock(
                fillna=MagicMock(
                    return_value=MagicMock(
                        astype=MagicMock(
                            return_value=MagicMock(
                                tolist=MagicMock(
                                    return_value=["text1", "text2", "text3"]
                                )
                            )
                        )
                    )
                )
            )
        )
        mock_df.__setitem__ = MagicMock()

        embeddings = np.random.randn(3, 384)
        embedding_factory._precomputed_embeddings["description"] = embeddings

        _result_df = embedding_factory.add_embedding_similarity_column(
            mock_df, "description", embeddings
        )

        # Should have called setitem to add the column
        mock_df.__setitem__.assert_called()

    def test_should_use_cached_embeddings(self, embedding_factory):
        """Should use cached embeddings if not provided."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "description"]
        mock_df.__setitem__ = MagicMock()

        # Pre-cache embeddings
        embeddings = np.random.randn(3, 384)
        embedding_factory._precomputed_embeddings["description"] = embeddings

        embedding_factory.add_embedding_similarity_column(mock_df, "description")

        # Should have used cached embeddings and added column
        mock_df.__setitem__.assert_called()


class TestPrepareDataframeWithEmbeddings:
    """Test full DataFrame preparation."""

    def test_should_add_all_signature_columns(self, embedding_factory):
        """Should add signature columns for all text columns."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "description", "name"]
        mock_df.__getitem__ = MagicMock(
            return_value=MagicMock(
                fillna=MagicMock(
                    return_value=MagicMock(
                        astype=MagicMock(
                            return_value=MagicMock(
                                tolist=MagicMock(
                                    return_value=["text1", "text2", "text3"]
                                )
                            )
                        )
                    )
                )
            )
        )
        mock_df.__setitem__ = MagicMock()

        df, embeddings = embedding_factory.prepare_dataframe_with_embeddings(
            mock_df, ["description", "name"]
        )

        # Should have embeddings for both columns
        assert "description" in embeddings
        assert "name" in embeddings

    def test_should_return_embeddings_dict(self, embedding_factory):
        """Should return embeddings dictionary."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "description"]
        mock_df.__getitem__ = MagicMock(
            return_value=MagicMock(
                fillna=MagicMock(
                    return_value=MagicMock(
                        astype=MagicMock(
                            return_value=MagicMock(
                                tolist=MagicMock(
                                    return_value=["text1", "text2", "text3"]
                                )
                            )
                        )
                    )
                )
            )
        )
        mock_df.__setitem__ = MagicMock()

        df, embeddings = embedding_factory.prepare_dataframe_with_embeddings(
            mock_df, ["description"]
        )

        assert "description" in embeddings
        assert embeddings["description"].shape == (3, 384)

    def test_should_handle_empty_text_columns(self, embedding_factory):
        """Should handle empty text columns list."""
        mock_df = MagicMock()

        df, embeddings = embedding_factory.prepare_dataframe_with_embeddings(
            mock_df, []
        )

        assert embeddings == {}


# ============================================================
# Comparison Creation Tests
# ============================================================


class TestCreateEmbeddingComparison:
    """Test embedding comparison creation."""

    def test_should_return_comparison_dict(self, embedding_factory):
        """Should return comparison configuration dictionary."""
        comparison = embedding_factory.create_embedding_comparison("description")

        assert isinstance(comparison, dict)
        assert comparison["column"] == "description"

    def test_should_include_embedding_column(self, embedding_factory):
        """Should include embedding column name."""
        comparison = embedding_factory.create_embedding_comparison("description")

        assert comparison["embedding_column"] == "description_emb_signature"

    def test_should_use_default_prior(self, embedding_factory):
        """Should use default embedding prior."""
        comparison = embedding_factory.create_embedding_comparison("description")

        assert comparison["prior"] == EMBEDDING_PRIOR


# ============================================================
# Text Property Detection Tests
# ============================================================


class TestGetTextPropertyColumns:
    """Test TEXT property column detection."""

    def test_should_identify_text_columns(self, embedding_factory, sample_ontology):
        """Should identify columns with TEXT type."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "name", "description", "email"]

        text_columns = embedding_factory.get_text_property_columns(
            mock_df, sample_ontology, "Person"
        )

        assert "description" in text_columns

    def test_should_exclude_non_text_columns(self, embedding_factory, sample_ontology):
        """Should exclude columns with non-TEXT types."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "name", "description", "email"]

        text_columns = embedding_factory.get_text_property_columns(
            mock_df, sample_ontology, "Person"
        )

        assert "name" not in text_columns
        assert "email" not in text_columns

    def test_should_handle_missing_entity_type(
        self, embedding_factory, sample_ontology
    ):
        """Should handle missing entity type gracefully."""
        mock_df = MagicMock()
        mock_df.columns = ["id", "name"]

        text_columns = embedding_factory.get_text_property_columns(
            mock_df, sample_ontology, "NonexistentType"
        )

        assert text_columns == []


# ============================================================
# Helper Function Tests
# ============================================================


class TestIsPropTypeText:
    """Test _is_prop_type_text helper function."""

    def test_should_return_true_for_text(self):
        """Should return True for 'text' type."""
        assert _is_prop_type_text("text") is True
        assert _is_prop_type_text("TEXT") is True
        assert _is_prop_type_text("Text") is True

    def test_should_return_false_for_other_types(self):
        """Should return False for non-text types."""
        assert _is_prop_type_text("string") is False
        assert _is_prop_type_text("number") is False
        assert _is_prop_type_text("date") is False

    def test_should_handle_none(self):
        """Should return False for None."""
        assert _is_prop_type_text(None) is False


class TestCreateEmbeddingFactory:
    """Test factory function."""

    def test_should_create_factory_instance(self):
        """Should create EmbeddingAwareComparisonFactory instance."""
        factory = create_embedding_factory()
        assert isinstance(factory, EmbeddingAwareComparisonFactory)

    def test_should_accept_model_override(self):
        """Should accept model name override."""
        factory = create_embedding_factory(embedding_model="custom-model")
        assert factory.model_name == "custom-model"


# ============================================================
# Cache Statistics Tests
# ============================================================


class TestGetCacheStats:
    """Test cache statistics retrieval."""

    def test_should_return_not_initialized_when_no_model(self):
        """Should return not_initialized when embedding similarity not loaded."""
        factory = EmbeddingAwareComparisonFactory()
        stats = factory.get_cache_stats()
        assert stats["status"] == "not_initialized"

    def test_should_include_precomputed_info(
        self, embedding_factory, mock_embedding_similarity
    ):
        """Should include precomputed embedding information."""
        # Mock the get_cache_stats on the embedding similarity
        mock_embedding_similarity.get_cache_stats = MagicMock(
            return_value={
                "cache_enabled": True,
                "cache_size": 10,
            }
        )

        # Pre-cache some embeddings
        embedding_factory._precomputed_embeddings["description"] = np.random.randn(
            3, 384
        )
        stats = embedding_factory.get_cache_stats()

        assert "precomputed_columns" in stats
        assert "description" in stats["precomputed_columns"]
