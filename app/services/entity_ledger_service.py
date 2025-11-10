"""Service for persisting canonical entity fingerprints across transforms."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from psycopg.types.json import Json

from app.config import settings
from app.db import postgres as db
from app.services.transform.models import BaseNode

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
        """Upsert canonical fingerprints for the supplied nodes."""

        if not user_id:
            return

        nodes = list(nodes)
        timestamp = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, object]] = []

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
            }
            records.append(record)

        if not records:
            return

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
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, entity_type, canonical_key)
                    DO UPDATE SET
                        canonical_id = EXCLUDED.canonical_id,
                        features = EXCLUDED.features,
                        confidence = EXCLUDED.confidence,
                        updated_at = EXCLUDED.updated_at
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
                else:
                    self._memory_store[key] = EntityLedgerEntry(
                        user_id=record["user_id"],
                        entity_type=record["entity_type"],
                        canonical_key=record["canonical_key"],
                        canonical_id=record["canonical_id"],
                        features=record["features"],
                        confidence=record.get("confidence"),
                        first_seen_at=timestamp,
                        updated_at=timestamp,
                    )

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
                               first_seen_at, updated_at
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
