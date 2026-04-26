"""Embedding-Aware Comparison Factory for Splink.

Integrates semantic embedding similarity into the Splink deduplication pipeline.
Precomputes embeddings for TEXT properties and creates comparison rules that
leverage semantic similarity alongside traditional string matching.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from graphora_server.services.entity_resolution.embedding_similarity import (
    EmbeddingSimilarity,
    get_embedding_similarity,
)
from graphora_server.services.entity_resolution.models import (
    ComparisonMethod,
    ComparisonPrior,
)
from graphora_server.config import settings

logger = logging.getLogger(__name__)


# Priors for embedding-based comparisons (4 levels: exact, high, medium, low)
EMBEDDING_PRIOR = ComparisonPrior(
    m=(0.80, 0.12, 0.05, 0.03),  # P(level | match)
    u=(0.10, 0.15, 0.20, 0.55),  # P(level | non-match)
)

# Default embedding similarity thresholds
DEFAULT_EMBEDDING_THRESHOLDS = [0.95, 0.85, 0.70]


class EmbeddingAwareComparisonFactory:
    """Factory for creating Splink comparisons with embedding similarity support.

    This factory extends standard Splink comparisons by:
    1. Precomputing embeddings for TEXT properties in batch
    2. Creating hybrid comparisons that combine exact/fuzzy string matching
       with embedding-based semantic similarity
    3. Adding embedding similarity columns to DataFrames for Splink processing
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        cache_enabled: bool = True,
        similarity_thresholds: Optional[List[float]] = None,
    ):
        """Initialize the embedding-aware comparison factory.

        Args:
            embedding_model: Name of the sentence-transformers model to use.
                            Defaults to settings.ENTITY_RESOLUTION_EMBEDDING_MODEL.
            cache_enabled: Whether to cache embeddings.
            similarity_thresholds: Thresholds for similarity levels.
                                  Defaults to [0.95, 0.85, 0.70].
        """
        self.model_name = embedding_model or settings.ENTITY_RESOLUTION_EMBEDDING_MODEL
        self.cache_enabled = cache_enabled
        self.similarity_thresholds = (
            similarity_thresholds or DEFAULT_EMBEDDING_THRESHOLDS
        )
        self._embedding_similarity: Optional[EmbeddingSimilarity] = None
        self._precomputed_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

    @property
    def embedding_similarity(self) -> EmbeddingSimilarity:
        """Get or create the embedding similarity instance (lazy loaded)."""
        if self._embedding_similarity is None:
            self._embedding_similarity = get_embedding_similarity(
                model_name=self.model_name,
                cache_enabled=self.cache_enabled,
            )
        return self._embedding_similarity

    def precompute_embeddings(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
    ) -> Dict[str, np.ndarray]:
        """Precompute embeddings for text columns in batch.

        Args:
            df: DataFrame with entity data.
            text_columns: List of column names containing TEXT data.

        Returns:
            Dictionary mapping column names to embedding matrices.
            Each matrix has shape (n_rows, embedding_dim).
        """
        embeddings: Dict[str, np.ndarray] = {}

        for column in text_columns:
            if column not in df.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue

            # Get non-null text values
            texts = df[column].fillna("").astype(str).tolist()

            # Filter out empty strings for embedding computation
            non_empty_mask = [bool(t.strip()) for t in texts]
            non_empty_texts = [t for t, m in zip(texts, non_empty_mask) if m]

            if not non_empty_texts:
                logger.debug(f"No non-empty texts in column {column}")
                embeddings[column] = np.zeros(
                    (len(texts), self.embedding_similarity.embedding_dim)
                )
                continue

            logger.info(
                f"Computing embeddings for {len(non_empty_texts)} texts in column {column}"
            )

            # Compute embeddings in batch
            non_empty_embeddings = self.embedding_similarity.get_embeddings_batch(
                non_empty_texts
            )

            # Create full embedding matrix with zeros for empty texts
            full_embeddings = np.zeros(
                (len(texts), self.embedding_similarity.embedding_dim)
            )
            non_empty_idx = 0
            for i, mask in enumerate(non_empty_mask):
                if mask:
                    full_embeddings[i] = non_empty_embeddings[non_empty_idx]
                    non_empty_idx += 1

            embeddings[column] = full_embeddings
            self._precomputed_embeddings[column] = full_embeddings

        return embeddings

    def compute_pairwise_similarity_levels(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise similarity matrix and convert to discrete levels.

        Args:
            embeddings: Embedding matrix of shape (n, embedding_dim).

        Returns:
            Similarity level matrix of shape (n, n) with integer levels:
            0 = exact (>= threshold[0])
            1 = high (>= threshold[1])
            2 = medium (>= threshold[2])
            3 = low (< threshold[2])
        """
        # Compute cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)  # Avoid division by zero
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)

        # Convert to levels - iterate in reverse so higher thresholds take precedence
        # Level 0 = highest similarity (>= threshold[0])
        # Level N = lowest similarity (< all thresholds)
        levels = np.full(similarity_matrix.shape, len(self.similarity_thresholds))
        for level in reversed(range(len(self.similarity_thresholds))):
            threshold = self.similarity_thresholds[level]
            levels = np.where(similarity_matrix >= threshold, level, levels)

        return levels

    def add_embedding_similarity_column(
        self,
        df: pd.DataFrame,
        column: str,
        embeddings: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Add an embedding similarity level column to the DataFrame.

        Creates a column with precomputed similarity levels that Splink
        can use for blocking and comparison.

        Args:
            df: DataFrame with entity data.
            column: Name of the text column.
            embeddings: Precomputed embeddings (uses cached if not provided).

        Returns:
            DataFrame with added embedding column.
        """
        if embeddings is None:
            embeddings = self._precomputed_embeddings.get(column)

        if embeddings is None:
            logger.warning(
                f"No embeddings available for column {column}. "
                "Call precompute_embeddings first."
            )
            return df

        # Create embedding signature column for blocking
        # Use first few PCA components or hash of embedding as blocking key
        embedding_col_name = f"{column}_emb_signature"
        signatures = self._compute_embedding_signatures(embeddings)
        df[embedding_col_name] = signatures

        return df

    def _compute_embedding_signatures(
        self,
        embeddings: np.ndarray,
        _n_buckets: int = 100,
    ) -> List[str]:
        """Compute locality-sensitive hash signatures for blocking.

        Args:
            embeddings: Embedding matrix of shape (n, embedding_dim).
            n_buckets: Number of hash buckets.

        Returns:
            List of signature strings for each embedding.
        """
        # Simple LSH: project onto random hyperplanes
        np.random.seed(42)  # Deterministic for reproducibility
        dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        if dim == 0:
            return [""] * len(embeddings)

        n_hyperplanes = 8  # 2^8 = 256 possible signatures
        hyperplanes = np.random.randn(dim, n_hyperplanes)

        # Compute hash
        projections = np.dot(embeddings, hyperplanes)
        bits = (projections > 0).astype(int)

        # Convert to string signatures
        signatures = []
        for row in bits:
            sig = "".join(str(b) for b in row)
            signatures.append(sig)

        return signatures

    def create_embedding_comparison(
        self,
        column: str,
        prior: Optional[ComparisonPrior] = None,
    ) -> Dict[str, Any]:
        """Create a Splink-compatible comparison definition for embedding similarity.

        Note: This creates a custom comparison that must be processed outside
        standard Splink comparisons, as Splink doesn't natively support
        embedding comparisons.

        Args:
            column: Name of the text column.
            prior: Comparison prior probabilities.

        Returns:
            Dictionary with comparison configuration.
        """
        return {
            "column": column,
            "comparison_method": ComparisonMethod.EMBEDDING,
            "prior": prior or EMBEDDING_PRIOR,
            "thresholds": self.similarity_thresholds,
            "embedding_column": f"{column}_emb_signature",
        }

    def get_text_property_columns(
        self,
        df: pd.DataFrame,
        parsed_ontology: Dict[str, Any],
        entity_type: str,
    ) -> List[str]:
        """Identify TEXT type columns from ontology definition.

        Args:
            df: DataFrame with entity data.
            parsed_ontology: Parsed ontology dictionary.
            entity_type: Entity type to look up.

        Returns:
            List of column names that are TEXT type.
        """
        text_columns = []

        entity_def = parsed_ontology.get("entities", {}).get(entity_type, {})
        property_defs = entity_def.get("properties", {})

        for prop_name, prop_def in property_defs.items():
            if not isinstance(prop_def, dict):
                continue

            prop_type = (prop_def.get("type") or "").lower()
            if prop_type == "text":
                # Check if column exists in DataFrame
                if prop_name in df.columns:
                    text_columns.append(prop_name)

        return text_columns

    def prepare_dataframe_with_embeddings(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Prepare DataFrame with embedding columns for Splink processing.

        Args:
            df: Original DataFrame with entity data.
            text_columns: List of TEXT property column names.

        Returns:
            Tuple of (enhanced DataFrame, embedding matrices dict).
        """
        if not text_columns:
            return df, {}

        # Precompute embeddings
        embeddings = self.precompute_embeddings(df, text_columns)

        # Add signature columns for blocking
        for column in text_columns:
            if column in embeddings:
                df = self.add_embedding_similarity_column(
                    df, column, embeddings[column]
                )

        return df, embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about embedding cache usage."""
        if self._embedding_similarity is None:
            return {"status": "not_initialized"}

        stats = self._embedding_similarity.get_cache_stats()
        stats["precomputed_columns"] = list(self._precomputed_embeddings.keys())
        stats["total_precomputed_vectors"] = sum(
            emb.shape[0] for emb in self._precomputed_embeddings.values()
        )
        return stats


def _is_prop_type_text(prop_type: Optional[str]) -> bool:
    """Check if property type indicates long-form text (uses embeddings)."""
    if not prop_type:
        return False
    return prop_type.lower() == "text"


def create_embedding_factory(
    embedding_model: Optional[str] = None,
) -> EmbeddingAwareComparisonFactory:
    """Create an EmbeddingAwareComparisonFactory with default settings.

    Args:
        embedding_model: Optional model name override.

    Returns:
        Configured EmbeddingAwareComparisonFactory instance.
    """
    return EmbeddingAwareComparisonFactory(
        embedding_model=embedding_model or settings.ENTITY_RESOLUTION_EMBEDDING_MODEL,
        cache_enabled=True,
        similarity_thresholds=DEFAULT_EMBEDDING_THRESHOLDS,
    )
