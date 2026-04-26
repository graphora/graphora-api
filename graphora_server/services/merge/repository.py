"""Database helpers for merge workflow persistence."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from psycopg.types.json import Json

from graphora_server.db import postgres as db
from graphora_server.services.merge.models import (
    ChangeLogRecord,
    ChangeLogResolution,
    MergeStatus,
)

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def _map_change_log(row: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(row)
    for key in ("id", "merge_id", "node_id", "prod_node_id"):
        if key in record:
            record[key] = _stringify(record[key])
    return record


def get_merge_status_row(merge_id: str) -> Optional[Dict[str, Any]]:
    """Return the merge_status row for the given merge_id."""

    return db.sync_fetchrow(
        "SELECT * FROM merge_status WHERE merge_id = %s",
        merge_id,
    )


def insert_merge_status(merge_id: str, transform_id: str, ontology_id: str) -> None:
    """Insert a merge_status row if it does not already exist."""

    db.sync_execute(
        """
        INSERT INTO merge_status (merge_id, transform_id, ontology_id, status)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (merge_id) DO NOTHING
        """,
        merge_id,
        transform_id,
        ontology_id,
        MergeStatus.STARTED.value,
    )


def update_merge_status(
    merge_id: str,
    *,
    status: Optional[MergeStatus] = None,
    statistics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Update fields on merge_status."""

    fields: List[str] = []
    params: List[Any] = []

    if status is not None:
        fields.append("status = %s")
        params.append(status.value if isinstance(status, MergeStatus) else status)
    if statistics is not None:
        fields.append("statistics = %s")
        params.append(Json(statistics))
    if error is not None:
        fields.append("error = %s")
        params.append(error)

    if not fields:
        return

    params.append(merge_id)

    db.sync_execute(
        f"UPDATE merge_status SET {', '.join(fields)} WHERE merge_id = %s",
        *params,
    )


def fetch_merge_statistics(merge_id: str) -> Optional[Dict[str, Any]]:
    row = db.sync_fetchrow(
        "SELECT statistics FROM merge_status WHERE merge_id = %s",
        merge_id,
    )
    if not row:
        return None
    return row.get("statistics")


def upsert_change_log(record: ChangeLogRecord) -> None:
    payload = record.to_row()
    created_at = payload.get("created_at") or datetime.utcnow()

    db.sync_execute(
        """
        INSERT INTO change_logs (
            id,
            merge_id,
            node_id,
            prod_node_id,
            node_type,
            previous_props,
            changed_props,
            need_human_review,
            created_at,
            match_confidence,
            match_strategy
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            merge_id = EXCLUDED.merge_id,
            node_id = EXCLUDED.node_id,
            prod_node_id = EXCLUDED.prod_node_id,
            node_type = EXCLUDED.node_type,
            previous_props = EXCLUDED.previous_props,
            changed_props = EXCLUDED.changed_props,
            need_human_review = EXCLUDED.need_human_review,
            match_confidence = EXCLUDED.match_confidence,
            match_strategy = EXCLUDED.match_strategy
        """,
        payload["id"],
        payload["merge_id"],
        payload["node_id"],
        payload.get("prod_node_id"),
        payload["node_type"],
        Json(payload["previous_props"]),
        Json(payload.get("changed_props") or {}),
        payload.get("need_human_review", False),
        created_at,
        payload.get("match_confidence"),
        payload.get("match_strategy"),
    )


def fetch_change_logs(merge_id: str) -> List[Dict[str, Any]]:
    rows = db.sync_fetch(
        "SELECT * FROM change_logs WHERE merge_id = %s ORDER BY created_at ASC",
        merge_id,
    )
    return [_map_change_log(row) for row in (rows or [])]


def fetch_change_log_by_id(change_log_id: str) -> Optional[Dict[str, Any]]:
    row = db.sync_fetchrow(
        "SELECT * FROM change_logs WHERE id = %s",
        change_log_id,
    )
    return _map_change_log(row) if row else None


def fetch_unresolved_change_logs(merge_id: str) -> List[Dict[str, Any]]:
    rows = db.sync_fetch(
        """
        SELECT *
        FROM change_logs
        WHERE merge_id = %s AND need_human_review = TRUE
        ORDER BY created_at ASC
        """,
        merge_id,
    )
    return [_map_change_log(row) for row in (rows or [])]


def mark_change_log_resolved(
    merge_id: str,
    change_log_id: str,
    *,
    changed_props: Optional[Dict[str, Any]] = None,
) -> None:
    db.sync_execute(
        """
        UPDATE change_logs
        SET need_human_review = FALSE,
            changed_props = %s
        WHERE merge_id = %s AND id = %s
        """,
        Json(changed_props or {}),
        merge_id,
        change_log_id,
    )


def insert_resolution(record: ChangeLogResolution) -> None:
    payload = record.to_row()

    db.sync_execute(
        """
        INSERT INTO resolutions (
            id,
            ontology_id,
            node_type,
            previous_props,
            changed_props,
            resolved_props,
            resolution,
            learning_comment,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            previous_props = EXCLUDED.previous_props,
            changed_props = EXCLUDED.changed_props,
            resolved_props = EXCLUDED.resolved_props,
            resolution = EXCLUDED.resolution,
            learning_comment = EXCLUDED.learning_comment
        """,
        payload["id"],
        payload["ontology_id"],
        payload.get("node_type"),
        Json(payload["previous_props"]),
        Json(payload["changed_props"]),
        Json(payload["resolved_props"]),
        payload["resolution"],
        payload.get("learning_comment"),
        payload.get("created_at", datetime.utcnow()),
    )


def fetch_resolutions_by_node_type(node_type: str) -> List[Dict[str, Any]]:
    return db.sync_fetch(
        "SELECT * FROM resolutions WHERE node_type = %s ORDER BY created_at DESC",
        node_type,
    )


def fetch_change_logs_raw(merge_id: str) -> List[Dict[str, Any]]:
    """Alias for fetch_change_logs to mirror previous naming."""

    return fetch_change_logs(merge_id)


def fetch_merge_info(merge_id: str) -> Optional[Dict[str, Any]]:
    return db.sync_fetchrow(
        "SELECT * FROM merge_status WHERE merge_id = %s",
        merge_id,
    )
