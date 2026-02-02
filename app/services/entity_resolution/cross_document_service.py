"""Cross-Document Entity Resolution Service.

Provides service-level integration for cross-document entity linking,
combining EntityStore, EmbeddingSimilarity, and CrossDocumentResolver
for comprehensive entity resolution across document boundaries.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.services.entity_resolution.embedding_similarity import (
    EmbeddingSimilarity,
    get_embedding_similarity,
)
from app.services.entity_resolution.entity_store import (
    CrossDocumentResolver,
    EntityStore,
)
from app.services.transform.models import BaseNode

logger = logging.getLogger(__name__)


class CrossDocumentResolutionService:
    """Service for cross-document entity resolution.

    This service provides:
    1. Entity lookup against a persistent cross-document entity store
    2. Embedding-based semantic similarity matching
    3. Two-stage resolution: exact key match (fast) then similarity search (thorough)
    4. Automatic storage of newly resolved entities for future matching
    """

    def __init__(
        self,
        user_id: str,
        namespace: str = "default",
        similarity_threshold: Optional[float] = None,
        embedding_model: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        """Initialize the cross-document resolution service.

        Args:
            user_id: User ID for entity isolation.
            namespace: Namespace for entity grouping (e.g., ontology name).
            similarity_threshold: Minimum similarity for entity matching.
                                 Defaults to settings.ENTITY_RESOLUTION_SIMILARITY_THRESHOLD.
            embedding_model: Sentence-transformers model for embeddings.
                            Defaults to settings.ENTITY_RESOLUTION_EMBEDDING_MODEL.
            enabled: Whether cross-document resolution is enabled.
                    Defaults to settings.ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED.
        """
        self.user_id = user_id
        self.namespace = namespace
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.ENTITY_RESOLUTION_SIMILARITY_THRESHOLD
        )
        self.embedding_model = (
            embedding_model or settings.ENTITY_RESOLUTION_EMBEDDING_MODEL
        )
        self.enabled = (
            enabled
            if enabled is not None
            else settings.ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED
        )

        # Lazily initialized components
        self._entity_store: Optional[EntityStore] = None
        self._embedding_similarity: Optional[EmbeddingSimilarity] = None
        self._resolver: Optional[CrossDocumentResolver] = None

        # Statistics
        self._stats = {
            "nodes_processed": 0,
            "exact_matches": 0,
            "similarity_matches": 0,
            "new_entities": 0,
        }

    @property
    def entity_store(self) -> EntityStore:
        """Get or create the entity store (lazy initialization)."""
        if self._entity_store is None:
            self._entity_store = EntityStore(
                user_id=self.user_id,
                namespace=self.namespace,
            )
        return self._entity_store

    @property
    def embedding_similarity(self) -> EmbeddingSimilarity:
        """Get or create the embedding similarity instance (lazy initialization)."""
        if self._embedding_similarity is None:
            self._embedding_similarity = get_embedding_similarity(
                model_name=self.embedding_model,
                cache_enabled=True,
            )
        return self._embedding_similarity

    @property
    def resolver(self) -> CrossDocumentResolver:
        """Get or create the cross-document resolver (lazy initialization)."""
        if self._resolver is None:
            self._resolver = CrossDocumentResolver(
                entity_store=self.entity_store,
                embedding_similarity=self.embedding_similarity,
                similarity_threshold=self.similarity_threshold,
            )
        return self._resolver

    async def resolve_nodes(
        self,
        nodes: List[BaseNode],
        source_document_id: Optional[str] = None,
    ) -> List[BaseNode]:
        """Resolve nodes against the cross-document entity store.

        This performs two-stage resolution:
        1. Stage 1 (fast): Exact canonical_key match
        2. Stage 2 (thorough): Embedding similarity search for unmatched

        Args:
            nodes: List of nodes to resolve.
            source_document_id: ID of the source document.

        Returns:
            List of nodes with updated canonical_ids where matches were found.
        """
        if not self.enabled or not nodes:
            return nodes

        logger.info(
            "Starting cross-document resolution for %d nodes (document: %s)",
            len(nodes),
            source_document_id,
        )

        resolved_nodes = []

        for node in nodes:
            self._stats["nodes_processed"] += 1

            # Stage 1: Try exact canonical_key match
            if node.canonical_id:
                existing = await self.entity_store.get_entity(node.canonical_id)
                if existing:
                    self._stats["exact_matches"] += 1
                    logger.debug(
                        "Exact match found for node %s (canonical_id: %s)",
                        node.id,
                        node.canonical_id,
                    )
                    resolved_nodes.append(node)
                    continue

            # Stage 2: Try embedding similarity search
            embedding = await self._compute_node_embedding(node)
            if embedding:
                similar = await self.entity_store.find_similar_entities(
                    query_embedding=embedding,
                    entity_type=node.type,
                    threshold=self.similarity_threshold,
                    top_k=1,
                )

                if similar:
                    matched_id, similarity, matched_entity = similar[0]
                    self._stats["similarity_matches"] += 1
                    logger.info(
                        "Similarity match for node %s: matched %s (score: %.3f)",
                        node.id,
                        matched_id,
                        similarity,
                    )
                    # Update node with matched canonical_id
                    node.canonical_id = matched_id
                    resolved_nodes.append(node)
                    continue

            # No match found - store as new entity
            self._stats["new_entities"] += 1
            await self._store_new_entity(node, embedding, source_document_id)
            resolved_nodes.append(node)

        logger.info(
            "Cross-document resolution complete: %d processed, "
            "%d exact matches, %d similarity matches, %d new",
            self._stats["nodes_processed"],
            self._stats["exact_matches"],
            self._stats["similarity_matches"],
            self._stats["new_entities"],
        )

        return resolved_nodes

    async def _compute_node_embedding(
        self,
        node: BaseNode,
    ) -> Optional[List[float]]:
        """Compute embedding from node's text properties.

        Args:
            node: The node to compute embedding for.

        Returns:
            Embedding vector or None if no text properties.
        """
        # Collect text from properties
        text_parts = []

        properties = node.properties or {}
        canonical_properties = node.canonical_properties or {}

        # Prioritize canonical properties for consistency
        for key, value in canonical_properties.items():
            if value and isinstance(value, str) and len(value) > 2:
                text_parts.append(str(value))

        # Add regular properties if not already included
        for key, value in properties.items():
            if key not in canonical_properties and value:
                if isinstance(value, str) and len(value) > 2:
                    text_parts.append(str(value))

        if not text_parts:
            return None

        # Combine text and compute embedding
        combined_text = " | ".join(text_parts[:5])  # Limit to first 5 properties
        embedding = self.embedding_similarity.get_embedding(combined_text)
        return embedding.tolist()

    async def _store_new_entity(
        self,
        node: BaseNode,
        embedding: Optional[List[float]],
        source_document_id: Optional[str],
    ) -> None:
        """Store a new entity in the cross-document store.

        Args:
            node: The node to store.
            embedding: Precomputed embedding (optional).
            source_document_id: Source document ID.
        """
        if not node.canonical_id:
            logger.debug("Skipping storage for node %s: no canonical_id", node.id)
            return

        await self.entity_store.store_entity(
            canonical_id=node.canonical_id,
            entity_type=node.type,
            properties=node.properties or {},
            embedding=embedding,
            source_document_id=source_document_id,
            confidence=node.confidence_score or 1.0,
        )

    async def resolve_and_link(
        self,
        nodes: List[BaseNode],
        source_document_id: Optional[str] = None,
    ) -> Tuple[List[BaseNode], Dict[str, str]]:
        """Resolve nodes and return linking map.

        Args:
            nodes: List of nodes to resolve.
            source_document_id: Source document ID.

        Returns:
            Tuple of (resolved nodes, id_to_canonical_id mapping).
        """
        resolved = await self.resolve_nodes(nodes, source_document_id)

        # Build mapping from original ID to canonical ID
        id_to_canonical = {}
        for node in resolved:
            if node.canonical_id and node.id != node.canonical_id:
                id_to_canonical[node.id] = node.canonical_id

        return resolved, id_to_canonical

    def get_stats(self) -> Dict[str, Any]:
        """Get resolution statistics.

        Returns:
            Dictionary with resolution statistics.
        """
        return {
            **self._stats,
            "enabled": self.enabled,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embedding_model,
        }

    def reset_stats(self) -> None:
        """Reset resolution statistics."""
        self._stats = {
            "nodes_processed": 0,
            "exact_matches": 0,
            "similarity_matches": 0,
            "new_entities": 0,
        }


async def create_cross_document_service(
    user_id: str,
    namespace: str = "default",
) -> CrossDocumentResolutionService:
    """Factory function to create a CrossDocumentResolutionService.

    Args:
        user_id: User ID for entity isolation.
        namespace: Namespace for entity grouping.

    Returns:
        Configured CrossDocumentResolutionService instance.
    """
    return CrossDocumentResolutionService(
        user_id=user_id,
        namespace=namespace,
    )
