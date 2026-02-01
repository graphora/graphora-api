"""Embedding-Based Similarity for Entity Resolution.

Provides semantic similarity computation using sentence embeddings.
Domain-agnostic - works with any text property regardless of domain.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingSimilarity:
    """Compute semantic similarity using sentence embeddings.

    This class provides embedding-based comparison for entity resolution,
    enabling semantic matching beyond simple string similarity.

    The implementation is lazy-loaded to avoid importing heavy ML libraries
    when not needed.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_enabled: bool = True,
        cache_max_size: int = 10000,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize the embedding similarity computer.

        Args:
            model_name: The sentence-transformers model to use.
                       Common options:
                       - "all-MiniLM-L6-v2" (fast, good quality)
                       - "all-mpnet-base-v2" (better quality, slower)
                       - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
            cache_enabled: Whether to cache embeddings
            cache_max_size: Maximum number of embeddings to cache
            normalize_embeddings: Whether to L2-normalize embeddings
            device: Device to run model on ("cpu", "cuda", etc.)
        """
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.cache_max_size = cache_max_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device

        # Lazy-loaded model
        self._model = None
        self._embedding_dim: Optional[int] = None

        # Cache: hash -> embedding
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            # Load model to get dimension
            _ = self.model
        return self._embedding_dim

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model with dimension {self._embedding_dim}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers is required for embedding similarity. "
                "Install with: pip install sentence-transformers"
            )

    def _compute_hash(self, text: str) -> str:
        """Compute a hash for cache lookup."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available."""
        if not self.cache_enabled:
            return None

        cache_key = self._compute_hash(text)
        embedding = self._cache.get(cache_key)

        if embedding is not None:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        return embedding

    def _add_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Add embedding to cache."""
        if not self.cache_enabled:
            return

        # Simple LRU: if cache is full, remove oldest entries
        if len(self._cache) >= self.cache_max_size:
            # Remove 10% of oldest entries
            to_remove = self.cache_max_size // 10
            keys_to_remove = list(self._cache.keys())[:to_remove]
            for key in keys_to_remove:
                del self._cache[key]

        cache_key = self._compute_hash(text)
        self._cache[cache_key] = embedding

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        # Check cache first
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        # Cache and return
        self._add_to_cache(text, embedding)
        return embedding

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        # Separate cached and uncached
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute uncached embeddings in batch
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            )

            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                embedding = new_embeddings[i]
                results[idx] = embedding
                self._add_to_cache(text, embedding)

        return np.array(results)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Cosine similarity (embeddings are already normalized if normalize_embeddings=True)
        if self.normalize_embeddings:
            similarity = float(np.dot(emb1, emb2))
        else:
            similarity = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))

    def compute_similarity_matrix(
        self,
        texts1: List[str],
        texts2: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between two lists of texts.

        Args:
            texts1: First list of texts
            texts2: Second list of texts (if None, compute self-similarity)

        Returns:
            Similarity matrix of shape (len(texts1), len(texts2))
        """
        emb1 = self.get_embeddings_batch(texts1)

        if texts2 is None:
            emb2 = emb1
        else:
            emb2 = self.get_embeddings_batch(texts2)

        # Compute similarity matrix
        if self.normalize_embeddings:
            similarity_matrix = np.dot(emb1, emb2.T)
        else:
            # Normalize on the fly
            norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
            similarity_matrix = np.dot(emb1, emb2.T) / (norm1 * norm2.T)

        # Clamp to [0, 1]
        return np.clip(similarity_matrix, 0.0, 1.0)

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.7,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """Find similar texts from a list of candidates.

        Args:
            query: The query text
            candidates: List of candidate texts to search
            threshold: Minimum similarity threshold
            top_k: Return only top K results (None for all above threshold)

        Returns:
            List of (index, text, similarity) tuples, sorted by similarity descending
        """
        if not candidates:
            return []

        query_emb = self.get_embedding(query)
        candidate_embs = self.get_embeddings_batch(candidates)

        # Compute similarities
        if self.normalize_embeddings:
            similarities = np.dot(candidate_embs, query_emb)
        else:
            norm_query = np.linalg.norm(query_emb)
            norm_candidates = np.linalg.norm(candidate_embs, axis=1)
            similarities = np.dot(candidate_embs, query_emb) / (
                norm_candidates * norm_query
            )

        # Filter by threshold
        results = [
            (i, candidates[i], float(sim))
            for i, sim in enumerate(similarities)
            if sim >= threshold
        ]

        # Sort by similarity descending
        results.sort(key=lambda x: x[2], reverse=True)

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]

        return results

    def compute_similarity_level(
        self,
        text1: str,
        text2: str,
        thresholds: List[float] = [0.95, 0.85, 0.70],
    ) -> int:
        """Compute similarity and return a discrete level.

        Args:
            text1: First text
            text2: Second text
            thresholds: List of thresholds for levels (must be descending)

        Returns:
            Integer level:
            - 0: similarity >= thresholds[0] (exact match)
            - 1: thresholds[1] <= similarity < thresholds[0] (high similarity)
            - 2: thresholds[2] <= similarity < thresholds[1] (medium similarity)
            - 3: similarity < thresholds[2] (low similarity)
        """
        similarity = self.compute_similarity(text1, text2)

        for level, threshold in enumerate(thresholds):
            if similarity >= threshold:
                return level

        return len(thresholds)  # Below all thresholds

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self._cache),
            "cache_max_size": self.cache_max_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# Singleton instance for shared use
_default_instance: Optional[EmbeddingSimilarity] = None


def get_embedding_similarity(
    model_name: str = "all-MiniLM-L6-v2",
    cache_enabled: bool = True,
) -> EmbeddingSimilarity:
    """Get or create a shared EmbeddingSimilarity instance.

    Args:
        model_name: The model to use
        cache_enabled: Whether to enable caching

    Returns:
        Shared EmbeddingSimilarity instance
    """
    global _default_instance

    if _default_instance is None or _default_instance.model_name != model_name:
        _default_instance = EmbeddingSimilarity(
            model_name=model_name,
            cache_enabled=cache_enabled,
        )

    return _default_instance
