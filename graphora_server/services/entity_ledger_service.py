"""Service for persisting canonical entity fingerprints across transforms."""

from __future__ import annotations

import json
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
        """Populate canonical_id overrides from the ledger if available.

        Slice 2 of cross-document linking: when
        ``ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED`` is set, this
        method dispatches to ``hydrate_nodes_with_similarity`` (Stage 1
        exact-key + Stage 2 embedding similarity). When the flag is
        off, the legacy exact-key-only path runs unchanged. Routing
        through the same public method keeps the existing call sites
        and their tests stable while letting operators opt in to the
        new behaviour without code changes."""

        if not user_id or not nodes:
            return

        if settings.ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED:
            await self.hydrate_nodes_with_similarity(
                user_id,
                nodes,
                similarity_threshold=settings.ENTITY_RESOLUTION_SIMILARITY_THRESHOLD,
            )
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
        total = len(nodes)

        # Stage 1: Exact key match (fast path)
        lookup_map = await self._fetch_entries(user_id, nodes)
        unmatched_nodes = []
        exact_matches = 0

        for node in nodes:
            if not node.canonical_key:
                unmatched_nodes.append(node)
                continue

            entry = lookup_map.get((node.type, node.canonical_key))
            if entry:
                node.canonical_id = entry.canonical_id
                exact_matches += 1
            else:
                unmatched_nodes.append(node)

        if not unmatched_nodes:
            self._log_hydrate_summary(
                user_id=user_id,
                total=total,
                exact_matches=exact_matches,
                similarity_matches=0,
                threshold=similarity_threshold,
            )
            return

        # Stage 2: Similarity search for unmatched nodes
        logger.debug(
            "Stage 2: Similarity search for %d unmatched nodes",
            len(unmatched_nodes),
        )

        similar_matches = await self.find_similar_entities(
            user_id,
            unmatched_nodes,
            similarity_threshold,
            review_threshold=settings.ENTITY_RESOLUTION_SIMILARITY_REVIEW_THRESHOLD,
        )

        # Apply auto-tier matches (above similarity_threshold) without
        # asking the LLM. The auto-tier is the high-confidence band
        # where the cost of an LLM disambiguation isn't worth it; we
        # only spend tokens on the gray zone.
        auto_match_count = 0
        review_node_ids: List[str] = []
        nodes_by_id: Dict[str, BaseNode] = {n.id: n for n in unmatched_nodes}
        for node in unmatched_nodes:
            match = similar_matches.get(node.id)
            if not match:
                continue
            if match["tier"] == "auto":
                node.canonical_id = match["canonical_id"]
                auto_match_count += 1
                logger.debug(
                    "Auto-tier similarity match for node %s: %s (score: %.3f)",
                    node.id,
                    match["canonical_id"],
                    match["similarity"],
                )
            elif match["tier"] == "review":
                review_node_ids.append(node.id)

        # Slice 3: gray-zone disambiguation. The LLM gets one batched
        # call per entity_type asking 'are these the same?' across
        # all review-tier candidates. Failures degrade silently —
        # auto-tier matches stay applied, review-tier candidates
        # just don't get linked.
        llm_resolved_count = 0
        llm_rejected_count = 0
        llm_skipped_for_cap = 0
        if review_node_ids:
            confirmed = await self._disambiguate_review_candidates(
                user_id=user_id,
                review_matches={nid: similar_matches[nid] for nid in review_node_ids},
                nodes_by_id=nodes_by_id,
            )
            for nid in review_node_ids:
                if nid in confirmed:
                    nodes_by_id[nid].canonical_id = confirmed[nid]["canonical_id"]
                    llm_resolved_count += 1
                else:
                    llm_rejected_count += 1
            # ``confirmed`` is empty if the cap fired — distinguish so
            # the telemetry helps an operator distinguish 'LLM said no'
            # from 'we didn't even try'.
            if confirmed.get("__skipped_for_cap__"):
                llm_skipped_for_cap = llm_rejected_count
                llm_rejected_count = 0
                confirmed.pop("__skipped_for_cap__", None)

        self._log_hydrate_summary(
            user_id=user_id,
            total=total,
            exact_matches=exact_matches,
            similarity_matches=auto_match_count,
            llm_resolved=llm_resolved_count,
            llm_rejected=llm_rejected_count,
            llm_skipped_for_cap=llm_skipped_for_cap,
            threshold=similarity_threshold,
        )

    def _log_hydrate_summary(
        self,
        *,
        user_id: str,
        total: int,
        exact_matches: int,
        similarity_matches: int,
        threshold: float,
        llm_resolved: int = 0,
        llm_rejected: int = 0,
        llm_skipped_for_cap: int = 0,
    ) -> None:
        """One-line INFO summary of the cross-document hydrate pass.

        Operators reading the log timeline need to see whether the
        feature is doing useful work without grep-ing debug-level
        output. Structured ``extra`` fields make the line tractable
        for log aggregation pipelines (Loki/CloudWatch/etc.) — one
        log entry per transform's hydrate call, with counts that
        sum to ``total``.

        Slice 3 added the LLM disambiguation buckets:
          * ``llm_resolved``      — LLM said 'yes, same entity'
          * ``llm_rejected``      — LLM said 'no, different entity'
          * ``llm_skipped_for_cap`` — disambiguation skipped because
            the gray-zone candidate count exceeded the per-call cap
            (operator hit the cost-bound)
        Operators tuning thresholds use these to spot 'LLM is
        rejecting 90% of gray-zone candidates' (threshold may be too
        permissive) or 'cap fires often' (raise cap or tighten
        threshold)."""
        unmatched = (
            total
            - exact_matches
            - similarity_matches
            - llm_resolved
            - llm_rejected
            - llm_skipped_for_cap
        )
        logger.info(
            "Cross-doc hydrate: %d nodes, %d exact, %d similarity, "
            "%d llm-resolved, %d llm-rejected, %d llm-skipped, "
            "%d unmatched (threshold=%.2f)",
            total,
            exact_matches,
            similarity_matches,
            llm_resolved,
            llm_rejected,
            llm_skipped_for_cap,
            unmatched,
            threshold,
            extra={
                "user_id": user_id,
                "ledger_total": total,
                "ledger_exact_matches": exact_matches,
                "ledger_similarity_matches": similarity_matches,
                "ledger_llm_resolved": llm_resolved,
                "ledger_llm_rejected": llm_rejected,
                "ledger_llm_skipped_for_cap": llm_skipped_for_cap,
                "ledger_unmatched": unmatched,
                "ledger_threshold": threshold,
            },
        )

    async def _disambiguate_review_candidates(
        self,
        *,
        user_id: str,
        review_matches: Dict[str, Dict[str, object]],
        nodes_by_id: Dict[str, BaseNode],
    ) -> Dict[str, Dict[str, object]]:
        """Slice 3: ask the LLM yes/no on gray-zone similarity matches.

        Per entity_type we make ONE batched call to the BAML
        ``ResolveEntities`` function — the same surface
        ``resolve_entity_group`` uses for in-document resolution. The
        LLM sees the query nodes alongside their candidate ledger
        entries (formatted as ``{id, properties, confidence}`` dicts)
        and returns groups; a query that ends up in a group with a
        candidate is confirmed to be that candidate's canonical
        entity. The candidate's ``canonical_id`` is the binding ID
        the LLM sees, so the result maps cleanly back without a
        second translation step.

        Cost guards:
          * Per-call cap on (queries + candidates). Beyond this the
            entire pass is skipped — a marker key is added to the
            return dict so the caller can distinguish 'cap fired'
            from 'LLM said no' for telemetry.
          * Each entity_type's call is wrapped in try/except. A
            failure for one type doesn't poison the others — same
            graceful-degradation contract as the embedding-side fix.

        Returns:
            ``{node_id: match_info}`` for queries the LLM confirmed,
            mirroring the ``find_similar_entities`` shape so
            ``hydrate_nodes_with_similarity`` can apply them
            uniformly. Empty dict means no LLM matches landed —
            either because the LLM said no, the call failed, or the
            cap fired (caller checks the ``__skipped_for_cap__``
            marker for the latter).
        """
        if not review_matches or not nodes_by_id:
            return {}

        # Group queries + candidates by entity_type so each LLM call
        # asks about one type. The LLM is type-aware via the
        # ``entity_type`` arg; sending mixed types in one call would
        # confuse it.
        queries_by_type: Dict[str, List[BaseNode]] = {}
        candidates_by_type: Dict[str, Dict[str, EntityLedgerEntry]] = {}
        for node_id, match in review_matches.items():
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            entry = match.get("entry")
            if not isinstance(entry, EntityLedgerEntry):
                continue
            queries_by_type.setdefault(node.type, []).append(node)
            candidates_by_type.setdefault(node.type, {})
            candidates_by_type[node.type][entry.canonical_id] = entry

        cap = settings.ENTITY_RESOLUTION_DISAMBIGUATION_CAP
        total_payload = sum(
            len(queries_by_type.get(t, [])) + len(candidates_by_type.get(t, {}))
            for t in queries_by_type
        )
        if total_payload > cap:
            logger.warning(
                "Cross-doc disambiguation skipped: payload %d exceeds "
                "cap %d (raise ENTITY_RESOLUTION_DISAMBIGUATION_CAP "
                "or tighten SIMILARITY_REVIEW_THRESHOLD)",
                total_payload,
                cap,
            )
            return {"__skipped_for_cap__": True}

        try:
            from graphora_server.services.llm.client import LLMClient

            llm_client = LLMClient()
        except ImportError:
            logger.warning(
                "LLM client unavailable for cross-doc disambiguation; "
                "review-tier matches dropped"
            )
            return {}

        confirmed: Dict[str, Dict[str, object]] = {}

        for entity_type, qnodes in queries_by_type.items():
            cands = candidates_by_type.get(entity_type, {})
            if not cands:
                continue

            node_dicts: List[Dict[str, object]] = []
            query_id_set = set()
            for n in qnodes:
                node_dicts.append(
                    {
                        "id": n.id,
                        "properties": n.properties or {},
                        "confidence": (
                            n.confidence_score
                            if n.confidence_score is not None
                            else 0.5
                        ),
                    }
                )
                query_id_set.add(n.id)
            for canonical_id, entry in cands.items():
                features = entry.features or {}
                canonical_props = (
                    features.get("canonical_properties", {})
                    if isinstance(features, dict)
                    else {}
                )
                node_dicts.append(
                    {
                        "id": canonical_id,
                        "properties": canonical_props,
                        "confidence": (
                            entry.confidence if entry.confidence is not None else 1.0
                        ),
                    }
                )

            try:
                results = await llm_client.resolve_entities(
                    entity_type=entity_type,
                    node_dicts_str=json.dumps(node_dicts),
                    user_id=user_id,
                )
            except Exception as exc:
                logger.warning(
                    "Cross-doc disambiguation LLM call failed "
                    "(entity_type=%s): %s; review-tier matches for "
                    "this type dropped",
                    entity_type,
                    exc,
                )
                continue

            candidate_id_set = set(cands.keys())
            for result in results or []:
                ids = list(result.matching_ids or [])
                qids_in_group = [i for i in ids if i in query_id_set]
                cids_in_group = [i for i in ids if i in candidate_id_set]
                if not qids_in_group or not cids_in_group:
                    continue
                # When the LLM groups multiple candidates with the same
                # query, pin the query to the FIRST candidate
                # deterministically. Multi-candidate groups are rare
                # (each candidate IS a canonical entity from a
                # different document; the LLM saying 'all three are the
                # same as the query' is an interesting edge case but
                # picking one canonical_id is the safe choice).
                chosen_canonical = cids_in_group[0]
                for qid in qids_in_group:
                    if qid in confirmed:
                        continue
                    orig = review_matches.get(qid, {})
                    confirmed[qid] = {
                        "canonical_id": chosen_canonical,
                        "similarity": orig.get("similarity"),
                        "entry": cands[chosen_canonical],
                        "tier": "llm_confirmed",
                    }
                    if result.explanation:
                        logger.info(
                            "Cross-doc disambiguation: query=%s -> %s "
                            "(score=%.3f, %s)",
                            qid,
                            chosen_canonical,
                            float(orig.get("similarity") or 0.0),
                            result.explanation,
                        )

        return confirmed

    async def find_similar_entities(
        self,
        user_id: str,
        nodes: List[BaseNode],
        threshold: float = 0.85,
        review_threshold: Optional[float] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Find similar entities using embedding similarity.

        Pre-slice-1 this function recomputed an embedding for every
        stored ledger entry on every call — the dominant cost. Now
        we read the embedding column populated by ``record_nodes``
        and only compute embeddings for the query-side nodes (which
        are necessarily new).

        Slice 3 (gray-zone tiering): matches are tagged with a
        ``tier`` field — ``"auto"`` when the similarity is at or
        above ``threshold`` (auto-accept), ``"review"`` when it
        falls in ``[review_threshold, threshold)`` (caller should
        send to LLM disambiguation). Matches below
        ``review_threshold`` are dropped outright. Pre-slice-3
        callers passing only ``threshold`` get behaviour identical
        to the pre-slice-3 contract: no review tier, no review_
        threshold defaults None → effectively skips the gray-zone
        path.

        Entries embedded under a different model than the active
        ``ENTITY_RESOLUTION_EMBEDDING_MODEL`` are skipped — mixing
        vector spaces (e.g. 384-dim MiniLM with 768-dim MPNet)
        produces meaningless similarity scores. Pre-slice-1 entries
        with embedding=NULL are also skipped; they'll get
        backfilled the next time their canonical_key is upserted.

        Args:
            user_id: User ID for isolation.
            nodes: Nodes to find matches for.
            threshold: Auto-accept similarity threshold.
            review_threshold: Lower bound of the gray-zone (review)
                tier. Pass None to disable review tiering — matches
                below ``threshold`` get dropped, mirroring pre-slice-3
                behaviour.

        Returns:
            Dictionary mapping node.id to match info:
            ``{node_id: {"canonical_id": ..., "similarity": ...,
            "entry": ..., "tier": "auto" | "review"}}``
        """
        try:
            from graphora_server.services.entity_resolution.embedding_similarity import (
                get_embedding_similarity,
            )

            embedding_similarity = get_embedding_similarity(
                model_name=settings.ENTITY_RESOLUTION_EMBEDDING_MODEL,
            )
            import numpy as np
        except ImportError:
            logger.warning("Embedding similarity not available for entity ledger")
            return {}

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
            #
            # ``embedding_model is None`` is also a skip: it means the
            # vector was written by an older code path that didn't
            # record the producing model. We can't safely assume the
            # embedding came from active_model — different models can
            # have different dimensions and different vector spaces, so
            # mixing them either crashes the dot product or silently
            # returns garbage. The guard is "match the model exactly,
            # or skip" rather than "skip on known mismatch".
            usable_entries: List[EntityLedgerEntry] = []
            stored_vectors: List[List[float]] = []
            for entry in existing_entries.values():
                if not entry.embedding:
                    continue
                if entry.embedding_model != active_model:
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

            # Runtime guard: the import succeeded, but the model load
            # itself is lazy — first call to ``get_embeddings_batch``
            # downloads weights, which can fail (HuggingFace
            # unreachable, missing/corrupted weights, GPU/CUDA error,
            # bad model name in settings). The dot/clip step can also
            # fail if a stored vector has an unexpected shape. We MUST
            # NOT let those propagate, because find_similar_entities
            # is called from inside hydrate_nodes — a transform
            # turning into a 500 because cross-doc resolution failed
            # is the worst possible failure mode for a feature that
            # is supposed to be additive on top of the legacy exact-
            # key path. Log and degrade: Stage 1 exact-key matches
            # already landed on nodes' canonical_id before we got
            # here, so returning {} preserves them; the caller just
            # doesn't get the Stage 2 boost.
            try:
                query_embeddings = embedding_similarity.get_embeddings_batch(node_texts)
                stored_matrix = np.array(stored_vectors, dtype=np.float32)
                # Both sides are L2-normalized by EmbeddingSimilarity
                # (the default config), so dot product = cosine
                # similarity. We clip to [0, 1] to mirror
                # EmbeddingSimilarity.compute_similarity_matrix's
                # contract for downstream callers.
                similarity_matrix = np.clip(
                    np.dot(query_embeddings, stored_matrix.T), 0.0, 1.0
                )
            except Exception as exc:
                logger.warning(
                    "Cross-document similarity failed at runtime "
                    "(model=%s, entity_type=%s): %s; degrading to "
                    "exact-key only for this hydrate call",
                    active_model,
                    entity_type,
                    exc,
                )
                return {}

            for i, node in enumerate(type_nodes):
                if not node_texts[i]:
                    continue
                similarities = similarity_matrix[i]
                best_idx = int(similarities.argmax())
                best_similarity = float(similarities[best_idx])

                if best_similarity >= threshold:
                    tier = "auto"
                elif (
                    review_threshold is not None and best_similarity >= review_threshold
                ):
                    tier = "review"
                else:
                    continue

                best_entry = usable_entries[best_idx]
                matches[node.id] = {
                    "canonical_id": best_entry.canonical_id,
                    "similarity": best_similarity,
                    "entry": best_entry,
                    "tier": tier,
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
                # ORDER BY updated_at DESC + LIMIT keeps the cap
                # deterministic when it's hit: we keep the most-recently-
                # touched entries, which are the most likely to match
                # newly-extracted nodes (reflect the current world). The
                # 10000 cap matches the migration-13 docstring ("ledger
                # sizes <= ~10k entries per (user, type), linear scan
                # <100ms"); pre-slice-1 the cap was 1000 with no order,
                # which (a) silently dropped candidates beyond the first
                # 1000 rows the planner happened to return and (b) made
                # different reads return different subsets. Slice 1
                # turned this query into the persisted similarity index,
                # so non-determinism here = non-deterministic match
                # outcomes downstream. NULLS LAST so rows missing the
                # column (defensive — shouldn't happen but cheap to
                # cover) sort to the end rather than the front.
                rows = await db.fetch(
                    """
                    SELECT canonical_key, canonical_id, features, confidence,
                           first_seen_at, updated_at, embedding, embedding_model
                    FROM entity_ledger
                    WHERE user_id = %s AND entity_type = %s
                    ORDER BY updated_at DESC NULLS LAST
                    LIMIT 10000
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
