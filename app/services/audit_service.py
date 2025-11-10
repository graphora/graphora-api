"""Audit Trail Service for tracking operations in Postgres."""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from psycopg.types.json import Json

from app.config import settings
from app.db import postgres as db

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Types of operations to audit"""

    ONTOLOGY_STORED = "ontology_stored"
    TRANSFORM_STARTED = "transform_started"
    TRANSFORM_COMPLETED = "transform_completed"
    MERGE_STARTED = "merge_started"
    MERGE_COMPLETED = "merge_completed"
    SCHEMA_GENERATION = "schema_generation"
    SCHEMA_SEARCH = "schema_search"
    SCHEMA_REFINEMENT = "schema_refinement"
    SCHEMA_CREATE = "schema_create"

    # Chat-related operations
    CHAT_SESSION_STARTED = "chat_session_started"
    CHAT_MESSAGE_SENT = "chat_message_sent"
    CHAT_MESSAGE_RECEIVED = "chat_message_received"
    CHAT_SESSION_ENDED = "chat_session_ended"
    CHAT_SCHEMA_REFINEMENT = "chat_schema_refinement"


class OperationStatus(str, Enum):
    """Status of operations"""

    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


class AuditService:
    """Service for logging audit trails"""

    def __init__(self):
        if not (settings.DATABASE_URL or settings.resolved_database_url):
            if not settings.test_mode:
                raise ValueError("DATABASE_URL must be configured for audit service")

    async def log_operation_start(
        self,
        user_id: str,
        operation_type: OperationType,
        operation_id: str,
        resource_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log the start of an operation

        Returns:
            audit_id: ID of the created audit record
        """
        try:
            row = await db.fetchrow(
                """
                INSERT INTO audit_trail (
                    user_id,
                    operation_type,
                    operation_id,
                    resource_name,
                    status,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                user_id,
                operation_type.value,
                operation_id,
                resource_name,
                OperationStatus.IN_PROGRESS.value,
                Json(metadata or {}),
            )

            if row:
                audit_id = row["id"]
                logger.info(
                    f"Started audit trail {audit_id} for {operation_type.value} by user {user_id}"
                )
                return audit_id
            else:
                logger.error(f"Failed to create audit trail for {operation_type.value}")
                return ""

        except Exception as e:
            logger.error(f"Error logging operation start: {str(e)}")
            return ""

    async def log_operation_success(
        self,
        audit_id: str,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log successful completion of an operation"""
        try:
            merged_metadata = await self._merge_metadata(audit_id, metadata)

            row = await db.fetchrow(
                """
                UPDATE audit_trail
                SET status = %s,
                    duration_ms = COALESCE(%s, duration_ms),
                    metadata = COALESCE(%s::jsonb, metadata),
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id
                """,
                OperationStatus.SUCCESS.value,
                duration_ms,
                Json(merged_metadata) if merged_metadata is not None else None,
                audit_id,
            )

            if row:
                logger.info(f"Successfully completed audit trail {audit_id}")
                return True
            else:
                logger.error(f"Failed to update audit trail {audit_id}")
                return False

        except Exception as e:
            logger.error(f"Error logging operation success: {str(e)}")
            return False

    async def log_operation_failure(
        self,
        audit_id: str,
        error_message: str,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log failed completion of an operation"""
        try:
            merged_metadata = await self._merge_metadata(audit_id, metadata)

            row = await db.fetchrow(
                """
                UPDATE audit_trail
                SET status = %s,
                    error_message = %s,
                    duration_ms = COALESCE(%s, duration_ms),
                    metadata = COALESCE(%s::jsonb, metadata),
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id
                """,
                OperationStatus.FAILED.value,
                error_message,
                duration_ms,
                Json(merged_metadata) if merged_metadata is not None else None,
                audit_id,
            )

            if row:
                logger.info(
                    f"Logged failure for audit trail {audit_id}: {error_message}"
                )
                return True
            else:
                logger.error(f"Failed to update audit trail {audit_id}")
                return False

        except Exception as e:
            logger.error(f"Error logging operation failure: {str(e)}")
            return False

    async def log_operation_end(
        self,
        user_id: str,
        operation_id: str,
        status: OperationStatus,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Log the end of an operation by finding and updating the audit record"""
        try:
            # Find the audit record by user_id and operation_id
            record = await db.fetchrow(
                """
                SELECT id, metadata
                FROM audit_trail
                WHERE user_id = %s AND operation_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id,
                operation_id,
            )

            if not record:
                logger.error(
                    f"No audit record found for user {user_id} and operation {operation_id}"
                )
                return False

            merged_metadata: Optional[Dict[str, Any]] = None
            if metadata:
                existing_metadata = record.get("metadata") or {}
                merged_metadata = {**existing_metadata, **metadata}

            audit_id = record["id"]

            row = await db.fetchrow(
                """
                UPDATE audit_trail
                SET status = %s,
                    duration_ms = COALESCE(%s, duration_ms),
                    error_message = %s,
                    metadata = COALESCE(%s::jsonb, metadata),
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id
                """,
                status.value,
                duration_ms,
                error_message,
                Json(merged_metadata) if merged_metadata is not None else None,
                audit_id,
            )

            if row:
                logger.info(
                    f"Logged end of operation {operation_id} for user {user_id} with status {status.value}"
                )
                return True
            else:
                logger.error(f"Failed to update audit trail {audit_id}")
                return False

        except Exception as e:
            logger.error(f"Error logging operation end: {str(e)}")
            return False

    async def log_direct_operation(
        self,
        user_id: str,
        operation_type: OperationType,
        operation_id: str,
        status: OperationStatus,
        resource_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> str:
        """Log a complete operation in one call"""
        try:
            row = await db.fetchrow(
                """
                INSERT INTO audit_trail (
                    user_id,
                    operation_type,
                    operation_id,
                    resource_name,
                    status,
                    metadata,
                    error_message,
                    duration_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                user_id,
                operation_type.value,
                operation_id,
                resource_name,
                status.value,
                Json(metadata or {}),
                error_message,
                duration_ms,
            )

            if row:
                audit_id = row["id"]
                logger.info(
                    f"Logged audit trail {audit_id} for {operation_type.value} by user {user_id}"
                )
                return audit_id
            else:
                logger.error(f"Failed to create audit trail for {operation_type.value}")
                return ""

        except Exception as e:
            logger.error(f"Error logging direct operation: {str(e)}")
            return ""

    async def get_user_audit_trail(
        self,
        user_id: str,
        operation_type: Optional[OperationType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get audit trail for a user"""
        try:
            params: List[Any] = [user_id]
            filters = "WHERE user_id = %s"
            if operation_type:
                filters += " AND operation_type = %s"
                params.append(operation_type.value)

            params.extend([limit, offset])
            rows = await db.fetch(
                f"""
                SELECT *
                FROM audit_trail
                {filters}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                *params,
            )

            return rows or []

        except Exception as e:
            logger.error(f"Error fetching audit trail: {str(e)}")
            return []

    async def get_audit_summary(self, user_id: str) -> Dict[str, Any]:
        """Get audit trail summary for dashboard"""
        try:
            # Get counts by operation type
            operations = await db.fetch(
                """
                SELECT operation_type, status
                FROM audit_trail
                WHERE user_id = %s
                """,
                user_id,
            )

            summary = {
                "total_operations": len(operations),
                "by_type": {},
                "by_status": {},
                "recent_operations": [],
            }

            # Count by operation type (only successful operations) and status (all operations)
            for op in operations:
                op_type = op.get("operation_type", "unknown")
                status = op.get("status", "unknown")

                # For by_type, only count successful operations (for dashboard metrics)
                if status == "success":
                    if op_type not in summary["by_type"]:
                        summary["by_type"][op_type] = 0
                    summary["by_type"][op_type] += 1

                # For by_status, count all operations
                if status not in summary["by_status"]:
                    summary["by_status"][status] = 0
                summary["by_status"][status] += 1

            # Get recent operations
            summary["recent_operations"] = await db.fetch(
                """
                SELECT *
                FROM audit_trail
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 10
                """,
                user_id,
            )

            return summary

        except Exception as e:
            logger.error(f"Error fetching audit summary: {str(e)}")
            return {
                "total_operations": 0,
                "by_type": {},
                "by_status": {},
                "recent_operations": [],
            }

    async def _merge_metadata(
        self, audit_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not metadata:
            return None

        existing = await db.fetchrow(
            "SELECT metadata FROM audit_trail WHERE id = %s",
            audit_id,
        )
        existing_metadata = {}
        if existing and existing.get("metadata"):
            existing_metadata = existing["metadata"]

        return {**existing_metadata, **metadata}


# Create global instance
audit_service = AuditService()
