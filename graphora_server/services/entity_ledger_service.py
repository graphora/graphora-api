"""Service for persisting canonical entity fingerprints across transforms."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from psycopg.types.json import Json

from graphora_server.config import settings
from graphora_server.db import postgres as db
from graphora_server.services.transform.models import BaseNode

logger = logging.getLogger(__name__)


@dataclass
class EntityLedgerEntry:
    user_id: str
    entity_type: str
    canonical_key: str
    canonical_id: str
    features: Dict[str, object]
    confidence: Optional[float] = None
    first_seen_at: Optional[str] = None
    updated_at: Optional[str] = None
    # Enhanced fields for cross-document resolution
    embedding: Optional[List[float]] = None
    # The sentence-transformer model used to produce ``embedding``.
    # Recorded so the similarity reader can ignore entries embedded
    # under a different model — mixing 384-dim MiniLM vectors with
    # 768-dim MPNet vectors silently returns garbage scores. Older
    # callers that don't set the field default to None and the
    # reader treats them as "unknown model -> skip".
    embedding_model: Optional[str] = None
    match_count: int = 1
    document_ids: Optional[List[str]] = None
    last_matched_at: Optional[str] = None


class EntityLedgerService:
    """Persist canonical entity fingerprints for reuse across transforms."""

    TABLE_NAME = "entity_ledger"

    def __init__(
        self,
        memory_store: Optional[Dict[tuple, EntityLedgerEntry]] = None,
    ) -> None:
        self._enabled = bool(settings.DATABASE_URL or settings.resolved_database_url)
        self._memory_store: Dict[tuple, EntityLedgerEntry] = memory_store or {}

    # Public API -----------------------------------------------------------------

    async def hydrate_nodes(
        self, user_id: Optional[str], nodes: Iterable[BaseNode]
    ) -> None:
        """Populate canonical_id overrides from the ledger if available."""

        if not user_id or not nodes:
            return

        nodes = list(nodes)
        lookup_map = await self._fetch_entries(user_id, nodes)

        for node in nodes:
            if not node.canonical_key:
                continue
            entry = lookup_map.get((node.type, node.canonical_key))
            if entry:
                node.canonical_id = entry.canonical_id

    async def record_nodes(
        self,
        user_id: Optional[str],
        nodes: Iterable[BaseNode],
    ) -> None:
        """Upsert canonical fingerprints for the supplied nodes.

        When ``ENTITY_RESOLUTION_EMBEDDING_ENABLED`` is set, this also
        precomputes a sentence-transformer embedding per node and
        stores it on the ledger row so the similarity-search path
        (``find_similar_entities``) doesn't have to re-embed every
        stored entry on every transform — that recomputation was the
        dominant cost of the lookup pre-slice-1."""

        if not user_id:
            return

        nodes = list(nodes)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Pair (record dict, source node) so we can compute embeddings
        # in one batched call after filtering. Storing the node
        # alongside lets the embedding step read from it without
        # tracking parallel index lists.
        pairs: List[tuple[Dict[str, object], BaseNode]] = []
        for node in nodes:
            if not node.canonical_key or not node.canonical_id:
                continue

            record = {
                "user_id": user_id,
                "entity_type": node.type,
                "canonical_key": node.canonical_key,
                "canonical_id": node.canonical_id,
                "features": {"canonical_properties": node.canonical_properties},
                "confidence": node.confidence_score,
                "first_seen_at": timestamp,
                "updated_at": timestamp,
                "embedding": None,
                "embedding_model": None,
            }
            pairs.append((record, node))

        if not pairs:
            return

        self._populate_embeddings_inplace(pairs)
        records = [pair[0] for pair in pairs]

        if self._enabled:
            try:
                chunk_size = 50
                query = """
                    INSERT INTO entity_ledger (
                        user_id,
                        entity_type,
                        canonical_key,
                        canonical_id,
                        features,
                        confidence,
                        first_seen_at,
                        updated_at,
                        embedding,
                        embedding_model
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, entity_type, canonical_key)
                    DO UPDATE SET
                        canonical_id = EXCLUDED.canonical_id,
                        features = EXCLUDED.features,
                        confidence = EXCLUDED.confidence,
                        updated_at = EXCLUDED.updated_at,
                        embedding = COALESCE(EXCLUDED.embedding, entity_ledger.embedding),
                        embedding_model = COALESCE(
                            EXCLUDED.embedding_model, entity_ledger.embedding_model
                        )
                """
                for idx in range(0, len(records), chunk_size):
                    chunk = records[idx : idx + chunk_size]
                    params = [
                        (
                            record["user_id"],
                            record["entity_type"],
                            record["canonical_key"],
                            record["canonical_id"],
                            Json(record["features"]),
                            record.get("confidence"),
                            record["first_seen_at"],
                            record["updated_at"],
                            (
                                Json(record["embedding"])
                                if record["embedding"] is not None
                                else None
                            ),
                            record.get("embedding_model"),
                        )
                        for record in chunk
                    ]
                    await db.executemany(query, params)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to upsert entity ledger entries: %s", exc)
        else:
            for record in records:
                key = (
                    record["user_id"],
                    record["entity_type"],
                    record["canonical_key"],
                )
                existing = self._memory_store.get(key)
                if existing:
                    existing.canonical_id = record["canonical_id"]  # type: ignore[assignment]
                    existing.features = record["features"]  # type: ignore[assignment]
                    existing.confidence = record.get("confidence")
                    existing.updated_at = timestamp
                    # Only overwrite embedding if we have a fresh one.
                    # Mirrors the COALESCE on the SQL side so disabling
                    # embeddings mid-stream doesn't wipe the cache.
                    if record["embedding"] is not None:
                        existing.embedding = record["embedding"]  # type: ignore[assignment]
                        existing.embedding_model = record["embedding_model"]  # type: ignore[assignment]
                else:
                    self._memory_store[key] = EntityLedgerEntry(
                        user_id=record["user_id"],  # type: ignore[arg-type]
                        entity_type=record["entity_type"],  # type: ignore[arg-type]
                        canonical_key=record["canonical_key"],  # type: ignore[arg-type]
                        canonical_id=record["canonical_id"],  # type: ignore[arg-type]
                        features=record["features"],  # type: ignore[arg-type]
                        confidence=record.get("confidence"),  # type: ignore[arg-type]
                        first_seen_at=timestamp,
                        updated_at=timestamp,
                        embedding=record["embedding"],  # type: ignore[arg-type]
                        embedding_model=record["embedding_model"],  # type: ignore[arg-type]
                    )

    def _populate_embeddings_inplace(
        self,
        pairs: List[tuple[Dict[str, object], BaseNode]],
    ) -> None:
        """Compute one batched embedding per record and stamp it onto
        the record dict (in-place). Silent no-op when embeddings are
        disabled, when the dependency is missing, or when no node has
        text to embed — those failure modes shouldn't break the main
        ledger write path."""
        if not settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED:
            return

        try:
            from graphora_server.services.entity_resolution.embedding_similarity import (
                get_embedding_similarity,
            )

            embedder = get_embedding_similarity(
                model_name=settings.ENTITY_RESOLUTION_EMBEDDING_MODEL,
            )
        except ImportError:
            logger.debug(
                "sentence-transformers not available; ledger embeddings skipped"
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialize embedding model: %s", exc)
            return

        texts: List[str] = []
        embeddable_indices: List[int] = []
        for idx, (_record, node) in enumerate(pairs):
            text = self._node_to_text(node)
            if text:
                texts.append(text)
                embeddable_indices.append(idx)

        if not texts:
            return

        try:
            embeddings = embedder.get_embeddings_batch(texts)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Embedding batch failed; ledger entries kept un-embedded: %s", exc
            )
            return

        for emb_idx, record_idx in enumerate(embeddable_indices):
            embedding_array = embeddings[emb_idx]
            pairs[record_idx][0]["embedding"] = embedding_array.tolist()
            pairs[record_idx][0][
                "embedding_model"
            ] = settings.ENTITY_RESOLUTION_EMBEDDING_MODEL

    async def hydrate_nodes_with_similarity(
        self,
        user_id: Optional[str],
        nodes: Iterable[BaseNode],
        similarity_threshold: float = 0.85,
    ) -> None:
        """Populate canonical_id using two-stage lookup: exact then similarity.

        Stage 1 (fast path): Exact canonical_key match
        Stage 2 (for unmatched): Embedding similarity search

        Args:
            user_id: User ID for isolation.
            nodes: Nodes to hydrate.
            similarity_threshold: Minimum similarity for Stage 2 matching.
        """
        if not user_id or not nodes:
            return

        nodes = list(nodes)

        # Stage 1: Exact key match (fast path)
        lookup_map = await self._fetch_entries(user_id, nodes)
        unmatched_nodes = []

        for node in nodes:
            if not node.canonical_key:
                unmatched_nodes.append(node)
                continue

            entry = lookup_map.get((node.type, node.canonical_key))
            if entry:
                node.canonical_id = entry.canonical_id
            else:
                unmatched_nodes.append(node)

        if not unmatched_nodes:
            logger.debug("All %d nodes matched via exact key lookup", len(nodes))
            return

        # Stage 2: Similarity search for unmatched nodes
        logger.debug(
            "Stage 2: Similarity search for %d unmatched nodes",
            len(unmatched_nodes),
        )

        similar_matches = await self.find_similar_entities(
            user_id, unmatched_nodes, similarity_threshold
        )

        for node in unmatched_nodes:
            match = similar_matches.get(node.id)
            if match:
                node.canonical_id = match["canonical_id"]
                logger.debug(
                    "Similarity match for node %s: %s (score: %.3f)",
                    node.id,
                    match["canonical_id"],
                    match["similarity"],
                )

    async def find_similar_entities(
        self,
        user_id: str,
        nodes: List[BaseNode],
        threshold: float = 0.85,
    ) -> Dict[str, Dict[str, object]]:
        """Find similar entities using embedding similarity.

        Pre-slice-1 this function recomputed an embedding for every
        stored ledger entry on every call — the dominant cost. Now
        we read the embedding column populated by ``record_nodes``
        and only compute embeddings for the query-side nodes (which
        are necessarily new).

        Entries embedded under a different model than the active
        ``ENTITY_RESOLUTION_EMBEDDING_MODEL`` are skipped — mixing
        vector spaces (e.g. 384-dim MiniLM with 768-dim MPNet)
        produces meaningless similarity scores. Pre-slice-1 entries
        with embedding=NULL are also skipped; they'll get
        backfilled the next time their canonical_key is upserted.

        Args:
            user_id: User ID for isolation.
            nodes: Nodes to find matches for.
            threshold: Minimum similarity threshold.

        Returns:
            Dictionary mapping node.id to match info
            {node_id: {"canonical_id": ..., "similarity": ..., "entry": ...}}
        """
        try:
            from graphora_server.services.entity_resolution.embedding_similarity import (
                get_embedding_similarity,
            )

            embedding_similarity = get_embedding_similarity(
                model_name=settings.ENTITY_RESOLUTION_EMBEDDING_MODEL,
            )
        except ImportError:
            logger.warning("Embedding similarity not available for entity ledger")
            return {}

        import numpy as np

        active_model = settings.ENTITY_RESOLUTION_EMBEDDING_MODEL
        matches: Dict[str, Dict[str, object]] = {}

        # Group nodes by type for efficient lookup
        nodes_by_type: Dict[str, List[BaseNode]] = {}
        for node in nodes:
            nodes_by_type.setdefault(node.type, []).append(node)

        for entity_type, type_nodes in nodes_by_type.items():
            existing_entries = await self._get_entries_by_type(user_id, entity_type)
            if not existing_entries:
                continue

            # Keep only entries with a usable, model-matched embedding.
            # The skip is by design — see the docstring.
            usable_entries: List[EntityLedgerEntry] = []
            stored_vectors: List[List[float]] = []
            for entry in existing_entries.values():
                if not entry.embedding:
                    continue
                if entry.embedding_model and entry.embedding_model != active_model:
                    continue
                usable_entries.append(entry)
                stored_vectors.append(entry.embedding)

            if not usable_entries:
                continue

            # Query-side: embed only the candidate nodes (always new).
            node_texts: List[str] = []
            for node in type_nodes:
                node_texts.append(self._node_to_text(node))
            if not any(node_texts):
                continue

            query_embeddings = embedding_similarity.get_embeddings_batch(node_texts)
            stored_matrix = np.array(stored_vectors, dtype=np.float32)
            # Both sides are L2-normalized by EmbeddingSimilarity (the
            # default config), so dot product = cosine similarity. We
            # clip to [0, 1] to mirror EmbeddingSimilarity.compute_
            # similarity_matrix's contract for downstream callers.
            similarity_matrix = np.clip(
                np.dot(query_embeddings, stored_matrix.T), 0.0, 1.0
            )

            for i, node in enumerate(type_nodes):
                if not node_texts[i]:
                    continue
                similarities = similarity_matrix[i]
                best_idx = int(similarities.argmax())
                best_similarity = float(similarities[best_idx])
                if best_similarity >= threshold:
                    best_entry = usable_entries[best_idx]
                    matches[node.id] = {
                        "canonical_id": best_entry.canonical_id,
                        "similarity": best_similarity,
                        "entry": best_entry,
                    }

        return matches

    async def _get_entries_by_type(
        self, user_id: str, entity_type: str
    ) -> Dict[str, EntityLedgerEntry]:
        """Get all ledger entries of a specific type for a user.

        Includes the persisted embedding + model when present —
        ``find_similar_entities`` reads these directly to skip the
        recomputation cost. Older rows (pre-slice-1) come back with
        embedding=None and the similarity reader skips them."""
        results: Dict[str, EntityLedgerEntry] = {}

        if self._enabled:
            try:
                rows = await db.fetch(
                    """
                    SELECT canonical_key, canonical_id, features, confidence,
                           first_seen_at, updated_at, embedding, embedding_model
                    FROM entity_ledger
                    WHERE user_id = %s AND entity_type = %s
                    LIMIT 1000
                    """,
                    user_id,
                    entity_type,
                )
                for row in rows or []:
                    results[row["canonical_key"]] = EntityLedgerEntry(
                        user_id=user_id,
                        entity_type=entity_type,
                        canonical_key=row["canonical_key"],
                        canonical_id=row["canonical_id"],
                        features=row.get("features", {}),
                        confidence=row.get("confidence"),
                        first_seen_at=row.get("first_seen_at"),
                        updated_at=row.get("updated_at"),
                        embedding=row.get("embedding"),
                        embedding_model=row.get("embedding_model"),
                    )
            except Exception as exc:
                logger.error("Failed to fetch entries by type: %s", exc)
        else:
            for key, entry in self._memory_store.items():
                if key[0] == user_id and key[1] == entity_type:
                    results[entry.canonical_key] = entry

        return results

    def _node_to_text(self, node: BaseNode) -> str:
        """Convert node properties to text for embedding."""
        parts = []
        canonical_props = node.canonical_properties or {}
        props = node.properties or {}

        # Prioritize canonical properties
        for key, value in canonical_props.items():
            if value and isinstance(value, str) and len(value) > 1:
                parts.append(str(value))

        # Add regular properties
        for key, value in props.items():
            if key not in canonical_props and value:
                if isinstance(value, str) and len(value) > 1:
                    parts.append(str(value))

        return " | ".join(parts[:5]) if parts else ""

    def _entry_to_text(self, entry: EntityLedgerEntry) -> str:
        """Convert ledger entry to text for embedding."""
        parts = []
        features = entry.features or {}
        canonical_props = features.get("canonical_properties", {})

        if isinstance(canonical_props, dict):
            for key, value in canonical_props.items():
                if value and isinstance(value, str) and len(value) > 1:
                    parts.append(str(value))

        return " | ".join(parts[:5]) if parts else entry.canonical_key

    # Internal helpers -----------------------------------------------------------

    async def _fetch_entries(
        self, user_id: str, nodes: Iterable[BaseNode]
    ) -> Dict[tuple, EntityLedgerEntry]:
        keys_by_type: Dict[str, List[str]] = {}
        for node in nodes:
            if node.canonical_key:
                keys_by_type.setdefault(node.type, []).append(node.canonical_key)

        if not keys_by_type:
            return {}

        results: Dict[tuple, EntityLedgerEntry] = {}

        if self._enabled:
            try:
                for entity_type, key_list in keys_by_type.items():
                    rows = await db.fetch(
                        """
                        SELECT canonical_key, canonical_id, features, confidence,
                               first_seen_at, updated_at, embedding, embedding_model
                        FROM entity_ledger
                        WHERE user_id = %s AND entity_type = %s AND canonical_key = ANY(%s)
                        """,
                        user_id,
                        entity_type,
                        key_list,
                    )
                    for row in rows or []:
                        results[(entity_type, row["canonical_key"])] = (
                            EntityLedgerEntry(
                                user_id=user_id,
                                entity_type=entity_type,
                                canonical_key=row["canonical_key"],
                                canonical_id=row["canonical_id"],
                                features=row.get("features", {}),
                                confidence=row.get("confidence"),
                                first_seen_at=row.get("first_seen_at"),
                                updated_at=row.get("updated_at"),
                                embedding=row.get("embedding"),
                                embedding_model=row.get("embedding_model"),
                            )
                        )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to fetch entity ledger entries: %s", exc)
        else:
            for entity_type, key_list in keys_by_type.items():
                for canonical_key in key_list:
                    entry = self._memory_store.get(
                        (user_id, entity_type, canonical_key)
                    )
                    if entry:
                        results[(entity_type, canonical_key)] = entry

        return results


entity_ledger_service = EntityLedgerService()
