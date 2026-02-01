"""Unit tests for Chunking Configuration.

Phase 4: Service Layer Tests - Chunking Config
Tests for configuration validation and defaults.
"""

import pytest
from pydantic import ValidationError

from app.services.chunking.config import (
    ChunkingConfig,
    ChunkingStrategy,
    DEFAULT_CONFIGS,
)


# ============================================================
# ChunkingStrategy Enum Tests
# ============================================================


class TestChunkingStrategy:
    """Test ChunkingStrategy enum."""

    def test_should_have_all_strategies(self):
        """Should include all chunking strategies."""
        assert ChunkingStrategy.SEMANTIC.value == "semantic"
        assert ChunkingStrategy.STRUCTURAL.value == "structural"
        assert ChunkingStrategy.HYBRID.value == "hybrid"
        assert ChunkingStrategy.RECURSIVE.value == "recursive"

    def test_should_be_string_enum(self):
        """Strategy values should be strings."""
        for strategy in ChunkingStrategy:
            assert isinstance(strategy.value, str)

    def test_should_have_four_strategies(self):
        """Should have exactly four strategies."""
        assert len(ChunkingStrategy) == 4


# ============================================================
# ChunkingConfig Default Values Tests
# ============================================================


class TestChunkingConfigDefaults:
    """Test ChunkingConfig default values."""

    def test_should_use_hybrid_strategy_by_default(self):
        """Default strategy should be HYBRID."""
        config = ChunkingConfig()
        assert config.strategy == ChunkingStrategy.HYBRID

    def test_should_have_sensible_chunk_size_defaults(self):
        """Should have reasonable min/max chunk size defaults."""
        config = ChunkingConfig()

        assert config.min_chunk_size == 500
        assert config.max_chunk_size == 6000
        assert config.min_chunk_size < config.max_chunk_size

    def test_should_have_semantic_chunking_defaults(self):
        """Should have semantic chunking defaults."""
        config = ChunkingConfig()

        assert config.semantic_threshold == 0.7
        assert config.semantic_min_length == 2000
        assert "mpnet" in config.embedding_model.lower()

    def test_should_enable_structural_preservation_by_default(self):
        """Should preserve lists, headings, quotes by default."""
        config = ChunkingConfig()

        assert config.preserve_lists is True
        assert config.preserve_headings is True
        assert config.preserve_quotes is True

    def test_should_have_overlap_default(self):
        """Should have default chunk overlap."""
        config = ChunkingConfig()
        assert config.chunk_overlap == 150

    def test_should_not_force_strategy_by_default(self):
        """Should not force strategy by default."""
        config = ChunkingConfig()
        assert config.force_strategy is False

    def test_should_have_quality_threshold_default(self):
        """Should have default quality threshold."""
        config = ChunkingConfig()
        assert config.quality_threshold == 0.6


# ============================================================
# ChunkingConfig Validation Tests
# ============================================================


class TestChunkingConfigValidation:
    """Test ChunkingConfig validation."""

    def test_should_accept_all_valid_strategies(self):
        """Should accept all valid strategies."""
        for strategy in ChunkingStrategy:
            config = ChunkingConfig(strategy=strategy)
            assert config.strategy == strategy

    def test_should_reject_invalid_strategy(self):
        """Should reject invalid strategy values."""
        with pytest.raises(ValidationError):
            ChunkingConfig(strategy="invalid_strategy")

    def test_should_enforce_min_chunk_size_minimum(self):
        """Minimum chunk size must be >= 100."""
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=50)

        # Edge case - exactly 100 should work
        config = ChunkingConfig(min_chunk_size=100)
        assert config.min_chunk_size == 100

    def test_should_enforce_min_chunk_size_maximum(self):
        """Minimum chunk size must be <= 2000."""
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=2500)

    def test_should_enforce_max_chunk_size_minimum(self):
        """Maximum chunk size must be >= 1000."""
        with pytest.raises(ValidationError):
            ChunkingConfig(max_chunk_size=500)

    def test_should_enforce_max_chunk_size_maximum(self):
        """Maximum chunk size must be <= 10000."""
        with pytest.raises(ValidationError):
            ChunkingConfig(max_chunk_size=15000)

    def test_should_enforce_semantic_threshold_range(self):
        """Semantic threshold must be between 0.1 and 1.0."""
        with pytest.raises(ValidationError):
            ChunkingConfig(semantic_threshold=0.05)

        with pytest.raises(ValidationError):
            ChunkingConfig(semantic_threshold=1.5)

        # Valid range
        config = ChunkingConfig(semantic_threshold=0.5)
        assert config.semantic_threshold == 0.5

    def test_should_enforce_chunk_overlap_range(self):
        """Chunk overlap must be between 0 and 500."""
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_overlap=-10)

        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_overlap=600)

        # Valid range
        config = ChunkingConfig(chunk_overlap=0)
        assert config.chunk_overlap == 0

    def test_should_enforce_quality_threshold_range(self):
        """Quality threshold must be between 0.1 and 1.0."""
        with pytest.raises(ValidationError):
            ChunkingConfig(quality_threshold=0.0)

        with pytest.raises(ValidationError):
            ChunkingConfig(quality_threshold=1.5)

    def test_should_allow_semantic_min_length_zero(self):
        """Semantic min length can be zero."""
        config = ChunkingConfig(semantic_min_length=0)
        assert config.semantic_min_length == 0


# ============================================================
# ChunkingConfig Serialization Tests
# ============================================================


class TestChunkingConfigSerialization:
    """Test ChunkingConfig serialization."""

    def test_should_serialize_to_dict(self):
        """Should serialize to dictionary."""
        config = ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC)
        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["strategy"] == "semantic"
        assert "min_chunk_size" in data
        assert "max_chunk_size" in data

    def test_should_deserialize_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "strategy": "structural",
            "min_chunk_size": 300,
            "max_chunk_size": 2000,
        }

        config = ChunkingConfig(**data)

        assert config.strategy == ChunkingStrategy.STRUCTURAL
        assert config.min_chunk_size == 300

    def test_should_serialize_to_json_compatible(self):
        """Should produce JSON-serializable output."""
        import json

        config = ChunkingConfig()
        data = config.model_dump()

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


# ============================================================
# DEFAULT_CONFIGS Tests
# ============================================================


class TestDefaultConfigs:
    """Test pre-defined default configurations."""

    def test_should_have_structured_config(self):
        """Should have structured document config."""
        config = DEFAULT_CONFIGS["structured"]

        assert config.strategy == ChunkingStrategy.STRUCTURAL
        assert config.preserve_lists is True
        assert config.preserve_headings is True

    def test_should_have_narrative_config(self):
        """Should have narrative document config."""
        config = DEFAULT_CONFIGS["narrative"]

        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.semantic_threshold >= 0.7

    def test_should_have_technical_config(self):
        """Should have technical document config."""
        config = DEFAULT_CONFIGS["technical"]

        assert config.strategy == ChunkingStrategy.RECURSIVE
        assert config.preserve_headings is True

    def test_should_have_hybrid_config(self):
        """Should have hybrid document config."""
        config = DEFAULT_CONFIGS["hybrid"]

        assert config.strategy == ChunkingStrategy.HYBRID
        assert config.preserve_lists is True
        assert config.preserve_headings is True
        assert config.preserve_quotes is True

    def test_default_configs_should_be_valid(self):
        """All default configs should be valid ChunkingConfig instances."""
        for name, config in DEFAULT_CONFIGS.items():
            assert isinstance(config, ChunkingConfig), f"{name} config is invalid"
            # Should not raise on dump
            config.model_dump()

    def test_default_configs_should_have_four_presets(self):
        """Should have exactly four preset configurations."""
        assert len(DEFAULT_CONFIGS) == 4
        assert set(DEFAULT_CONFIGS.keys()) == {
            "structured",
            "narrative",
            "technical",
            "hybrid",
        }


# ============================================================
# Strategy-Specific Configuration Tests
# ============================================================


class TestStrategySpecificConfig:
    """Test strategy-specific configuration combinations."""

    def test_semantic_strategy_should_use_embedding_model(self):
        """Semantic strategy should configure embedding model."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            embedding_model="custom-model",
        )

        assert config.embedding_model == "custom-model"

    def test_recursive_strategy_should_use_overlap(self):
        """Recursive strategy should use chunk overlap."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_overlap=250,
        )

        assert config.chunk_overlap == 250

    def test_structural_strategy_should_preserve_structure(self):
        """Structural strategy should preserve document structure."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURAL,
            preserve_lists=True,
            preserve_headings=True,
            preserve_quotes=True,
        )

        assert config.preserve_lists is True
        assert config.preserve_headings is True
        assert config.preserve_quotes is True

    def test_hybrid_strategy_should_combine_features(self):
        """Hybrid strategy should support features from multiple strategies."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            semantic_threshold=0.8,
            preserve_lists=True,
            chunk_overlap=200,
        )

        assert config.semantic_threshold == 0.8
        assert config.preserve_lists is True
        assert config.chunk_overlap == 200


# ============================================================
# Edge Case Tests
# ============================================================


class TestChunkingConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_should_handle_max_values(self):
        """Should handle maximum allowed values."""
        config = ChunkingConfig(
            min_chunk_size=2000,
            max_chunk_size=10000,
            semantic_threshold=1.0,
            chunk_overlap=500,
            quality_threshold=1.0,
        )

        assert config.min_chunk_size == 2000
        assert config.max_chunk_size == 10000
        assert config.semantic_threshold == 1.0

    def test_should_handle_min_values(self):
        """Should handle minimum allowed values."""
        config = ChunkingConfig(
            min_chunk_size=100,
            max_chunk_size=1000,
            semantic_threshold=0.1,
            chunk_overlap=0,
            quality_threshold=0.1,
        )

        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 1000
        assert config.semantic_threshold == 0.1

    def test_should_allow_disabling_structural_preservation(self):
        """Should allow disabling all structural preservation."""
        config = ChunkingConfig(
            preserve_lists=False,
            preserve_headings=False,
            preserve_quotes=False,
        )

        assert config.preserve_lists is False
        assert config.preserve_headings is False
        assert config.preserve_quotes is False

    def test_should_allow_custom_embedding_model(self):
        """Should allow custom embedding model path."""
        config = ChunkingConfig(
            embedding_model="organization/custom-embedding-model-v2",
        )

        assert config.embedding_model == "organization/custom-embedding-model-v2"

    def test_should_allow_force_strategy(self):
        """Should allow forcing a specific strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            force_strategy=True,
        )

        assert config.force_strategy is True
