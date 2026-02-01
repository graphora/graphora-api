"""Persistent Entity Store for Cross-Document Resolution.

Provides storage and retrieval of entity embeddings for
cross-document entity resolution.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EntityStore:
    """Persistent store for entity embeddings and metadata.

    Enables cross-document entity resolution by storing entity representations
    that can be searched across multiple documents and transforms.

    The store is domain-agnostic - it stores entities of any type
    with their embeddings and properties.
    """

    def __init__(
        self,
        user_id: str,
        namespace: str = "default",
        embedding_dim: int = 384,  # Default for MiniLM
    ):
        """Initialize the entity store.

        Args:
            user_id: User ID for isolation
            namespace: Namespace for entity grouping (e.g., ontology name)
            embedding_dim: Dimension of entity embeddings
        """
        self.user_id = user_id
        self.namespace = namespace
        self.embedding_dim = embedding_dim

        # In-memory index (to be replaced with persistent storage)
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._type_index: Dict[str, List[str]] = {}  # entity_type -> [canonical_ids]

    def _generate_storage_key(self, canonical_id: str) -> str:
        """Generate storage key for an entity."""
        return f"{self.user_id}:{self.namespace}:{canonical_id}"

    async def store_entity(
        self,
        canonical_id: str,
        entity_type: str,
        properties: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        source_document_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> str:
        """Store an entity with its embedding.

        Args:
            canonical_id: The canonical ID of the entity
            entity_type: The entity type (from ontology)
            properties: Entity properties
            embedding: Entity embedding vector
            source_document_id: ID of the source document
            confidence: Confidence score for the entity

        Returns:
            Storage key for the entity
        """
        storage_key = self._generate_storage_key(canonical_id)

        entity_record = {
            "canonical_id": canonical_id,
            "entity_type": entity_type,
            "properties": properties,
            "source_document_id": source_document_id,
            "confidence": confidence,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._entities[storage_key] = entity_record

        if embedding:
            self._embeddings[storage_key] = embedding

        # Update type index
        if entity_type not in self._type_index:
            self._type_index[entity_type] = []
        if canonical_id not in self._type_index[entity_type]:
            self._type_index[entity_type].append(canonical_id)

        logger.debug(f"Stored entity {canonical_id} of type {entity_type}")
        return storage_key

    async def get_entity(self, canonical_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity by canonical ID.

        Args:
            canonical_id: The canonical ID of the entity

        Returns:
            Entity record or None if not found
        """
        storage_key = self._generate_storage_key(canonical_id)
        return self._entities.get(storage_key)

    async def get_entity_embedding(
        self,
        canonical_id: str,
    ) -> Optional[List[float]]:
        """Get the embedding for an entity.

        Args:
            canonical_id: The canonical ID of the entity

        Returns:
            Embedding vector or None if not found
        """
        storage_key = self._generate_storage_key(canonical_id)
        return self._embeddings.get(storage_key)

    async def find_similar_entities(
        self,
        query_embedding: List[float],
        entity_type: Optional[str] = None,
        threshold: float = 0.7,
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find similar entities by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            entity_type: Filter by entity type (optional)
            threshold: Minimum similarity threshold
            top_k: Maximum number of results

        Returns:
            List of (canonical_id, similarity, entity_record) tuples
        """
        import numpy as np

        query_vec = np.array(query_embedding)

        # Filter by entity type if specified
        if entity_type and entity_type in self._type_index:
            candidate_ids = self._type_index[entity_type]
        else:
            candidate_ids = list(
                set(cid for ids in self._type_index.values() for cid in ids)
            )

        results = []
        for canonical_id in candidate_ids:
            storage_key = self._generate_storage_key(canonical_id)
            embedding = self._embeddings.get(storage_key)

            if embedding is None:
                continue

            # Compute cosine similarity
            entity_vec = np.array(embedding)
            similarity = float(
                np.dot(query_vec, entity_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(entity_vec) + 1e-10)
            )

            if similarity >= threshold:
                entity = self._entities.get(storage_key, {})
                results.append((canonical_id, similarity, entity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def find_by_properties(
        self,
        entity_type: str,
        properties: Dict[str, Any],
        match_threshold: float = 0.8,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find entities by property matching.

        Args:
            entity_type: The entity type to search
            properties: Properties to match
            match_threshold: Minimum property match ratio

        Returns:
            List of (canonical_id, match_score, entity_record) tuples
        """
        if entity_type not in self._type_index:
            return []

        results = []
        for canonical_id in self._type_index[entity_type]:
            storage_key = self._generate_storage_key(canonical_id)
            entity = self._entities.get(storage_key)

            if not entity:
                continue

            # Compute property match score
            match_score = self._compute_property_match(
                properties, entity.get("properties", {})
            )

            if match_score >= match_threshold:
                results.append((canonical_id, match_score, entity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _compute_property_match(
        self,
        query_props: Dict[str, Any],
        stored_props: Dict[str, Any],
    ) -> float:
        """Compute property match score between two property dicts.

        Args:
            query_props: Query properties
            stored_props: Stored entity properties

        Returns:
            Match score between 0 and 1
        """
        if not query_props:
            return 0.0

        matching = 0
        total = len(query_props)

        for key, value in query_props.items():
            if key in stored_props:
                stored_value = stored_props[key]
                if self._values_match(value, stored_value):
                    matching += 1

        return matching / total if total > 0 else 0.0

    def _values_match(self, v1: Any, v2: Any) -> bool:
        """Check if two property values match."""
        # Normalize strings for comparison
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.lower().strip() == v2.lower().strip()
        return v1 == v2

    async def get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get all entities of a specific type.

        Args:
            entity_type: The entity type
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of entity records
        """
        if entity_type not in self._type_index:
            return []

        canonical_ids = self._type_index[entity_type][offset : offset + limit]
        entities = []

        for canonical_id in canonical_ids:
            storage_key = self._generate_storage_key(canonical_id)
            entity = self._entities.get(storage_key)
            if entity:
                entities.append(entity)

        return entities

    async def update_entity(
        self,
        canonical_id: str,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Update an existing entity.

        Args:
            canonical_id: The canonical ID of the entity
            properties: New properties (merged with existing)
            embedding: New embedding vector
            confidence: New confidence score

        Returns:
            True if entity was updated, False if not found
        """
        storage_key = self._generate_storage_key(canonical_id)

        if storage_key not in self._entities:
            return False

        entity = self._entities[storage_key]

        if properties:
            entity["properties"].update(properties)

        if confidence is not None:
            entity["confidence"] = confidence

        entity["updated_at"] = datetime.now(timezone.utc).isoformat()

        if embedding:
            self._embeddings[storage_key] = embedding

        return True

    async def delete_entity(self, canonical_id: str) -> bool:
        """Delete an entity from the store.

        Args:
            canonical_id: The canonical ID of the entity

        Returns:
            True if entity was deleted, False if not found
        """
        storage_key = self._generate_storage_key(canonical_id)

        if storage_key not in self._entities:
            return False

        entity = self._entities.pop(storage_key)
        self._embeddings.pop(storage_key, None)

        # Update type index
        entity_type = entity.get("entity_type")
        if entity_type and entity_type in self._type_index:
            try:
                self._type_index[entity_type].remove(canonical_id)
            except ValueError:
                pass

        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the entity store.

        Returns:
            Dictionary with store statistics
        """
        return {
            "user_id": self.user_id,
            "namespace": self.namespace,
            "total_entities": len(self._entities),
            "entities_with_embeddings": len(self._embeddings),
            "entity_types": list(self._type_index.keys()),
            "entities_per_type": {t: len(ids) for t, ids in self._type_index.items()},
        }

    async def clear(self) -> None:
        """Clear all entities from the store."""
        self._entities.clear()
        self._embeddings.clear()
        self._type_index.clear()


class CrossDocumentResolver:
    """Resolve entities across documents using the entity store.

    This class provides cross-document entity resolution by:
    1. Looking up existing entities in the store
    2. Finding similar entities by embedding
    3. Merging with existing entities or creating new ones
    """

    def __init__(
        self,
        entity_store: EntityStore,
        embedding_similarity: Optional[Any] = None,
        similarity_threshold: float = 0.85,
    ):
        """Initialize the cross-document resolver.

        Args:
            entity_store: The entity store instance
            embedding_similarity: EmbeddingSimilarity instance (optional)
            similarity_threshold: Threshold for considering entities as matches
        """
        self.entity_store = entity_store
        self.embedding_similarity = embedding_similarity
        self.similarity_threshold = similarity_threshold

    async def resolve_entity(
        self,
        entity_type: str,
        properties: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> Tuple[Optional[str], bool, float]:
        """Resolve an entity against the store.

        Args:
            entity_type: The entity type
            properties: Entity properties
            embedding: Entity embedding (optional but improves accuracy)

        Returns:
            Tuple of (matched_canonical_id, is_new, confidence)
            - matched_canonical_id: ID of matched entity or None if new
            - is_new: True if this is a new entity
            - confidence: Confidence of the match (1.0 for exact, less for similar)
        """
        # First, try property-based lookup
        property_matches = await self.entity_store.find_by_properties(
            entity_type, properties, match_threshold=0.9
        )

        if property_matches:
            best_match = property_matches[0]
            return best_match[0], False, best_match[1]

        # If we have an embedding, try similarity search
        if embedding:
            similar = await self.entity_store.find_similar_entities(
                embedding,
                entity_type=entity_type,
                threshold=self.similarity_threshold,
                top_k=5,
            )

            if similar:
                best_match = similar[0]
                return best_match[0], False, best_match[1]

        # No match found - this is a new entity
        return None, True, 1.0

    async def resolve_and_store(
        self,
        canonical_id: str,
        entity_type: str,
        properties: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        source_document_id: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Resolve an entity and store it if new.

        Args:
            canonical_id: The canonical ID for this entity
            entity_type: The entity type
            properties: Entity properties
            embedding: Entity embedding (optional)
            source_document_id: Source document ID

        Returns:
            Tuple of (final_canonical_id, was_merged)
            - final_canonical_id: The canonical ID (may be different if merged)
            - was_merged: True if merged with existing entity
        """
        matched_id, is_new, confidence = await self.resolve_entity(
            entity_type, properties, embedding
        )

        if is_new:
            # Store as new entity
            await self.entity_store.store_entity(
                canonical_id=canonical_id,
                entity_type=entity_type,
                properties=properties,
                embedding=embedding,
                source_document_id=source_document_id,
                confidence=confidence,
            )
            return canonical_id, False
        else:
            # Merge with existing entity
            await self.entity_store.update_entity(
                canonical_id=matched_id,
                properties=properties,  # Merge properties
                embedding=embedding,
                confidence=max(confidence, 0.9),  # Keep high confidence
            )
            return matched_id, True
