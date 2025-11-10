"""Chat Session Service using Audit Trail Infrastructure

This module provides chat session management functionality built on top of the audit trail
infrastructure. It enables multi-turn conversations with persistent session state.

Key Features:
- Session lifecycle management (start, message tracking, end)
- Multiple context types (schema generation, refinement, etc.)
- Message history retrieval with metadata preservation
- User session listing and filtering
- Built-in error handling and logging
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from app.db import postgres as db
from app.services.audit_service import AuditService, OperationType, OperationStatus

logger = logging.getLogger(__name__)


class ChatContextType(str, Enum):
    """Types of chat contexts"""

    SCHEMA_GENERATION = "schema_generation"
    SCHEMA_REFINEMENT = "schema_refinement"
    ONTOLOGY_DESIGN = "ontology_design"
    GENERAL = "general"


class MessageType(str, Enum):
    """Types of chat messages"""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"


class ChatSessionService:
    """Service for managing chat sessions using audit trail infrastructure

    This service provides a complete chat session management system that leverages
    the existing audit trail infrastructure for persistence. Each chat session and
    message is stored as audit entries with rich metadata.

    Attributes:
        audit_service: Instance of AuditService for data persistence
    """

    def __init__(self) -> None:
        """Initialize the chat session service with audit trail integration."""
        self.audit_service = AuditService()

    async def start_session(
        self,
        user_id: str,
        context_type: ChatContextType,
        context_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
    ) -> str:
        """Start a new chat session with persistent state.

        Creates a new chat session and logs it in the audit trail for persistence.
        The session can be associated with different context types and include
        initial context data for conversation continuity.

        Args:
            user_id: Unique identifier for the user starting the session
            context_type: Type of chat context (schema_generation, refinement, etc.)
            context_id: Optional ID of related resource (schema, ontology, etc.)
            initial_context: Optional initial conversation context data
            title: Optional human-readable session title

        Returns:
            str: Unique session identifier for subsequent operations

        Raises:
            Exception: If session creation fails due to audit service issues
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")

        session_id = str(uuid.uuid4())

        try:
            # Log session start in audit trail
            await self.audit_service.log_operation_start(
                user_id=user_id,
                operation_type=OperationType.CHAT_SESSION_STARTED,
                operation_id=session_id,
                resource_name=title or f"Chat Session - {context_type.value}",
                metadata={
                    "session_id": session_id,
                    "context_type": context_type.value,
                    "context_id": context_id,
                    "initial_context": initial_context or {},
                    "message_count": 0,
                    "started_at": datetime.utcnow().isoformat(),
                },
            )

            logger.info(
                f"Started chat session {session_id} for user {user_id} with context {context_type.value}"
            )
            return session_id

        except Exception as e:
            logger.error(f"Failed to start session for user {user_id}: {str(e)}")
            raise Exception(f"Session creation failed: {str(e)}") from e

    async def add_message(
        self,
        user_id: str,
        session_id: str,
        message_type: MessageType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a message to an existing chat session.

        Records a new message in the chat session using the audit trail.
        Messages are immediately marked as successful since they represent
        communication events rather than operations that can fail.

        Args:
            user_id: Unique identifier for the user
            session_id: Target session identifier
            message_type: Type of message (user, assistant, system)
            content: Message text content
            metadata: Optional additional message metadata

        Returns:
            str: Unique message identifier

        Raises:
            ValueError: If required parameters are missing or invalid
            Exception: If message logging fails
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")
        if not session_id or not session_id.strip():
            raise ValueError("session_id is required and cannot be empty")
        if not content or not content.strip():
            raise ValueError("message content is required and cannot be empty")

        message_id = str(uuid.uuid4())

        try:
            # Choose appropriate operation type based on message sender
            if message_type == MessageType.USER_MESSAGE:
                operation_type = OperationType.CHAT_MESSAGE_SENT
            else:
                operation_type = OperationType.CHAT_MESSAGE_RECEIVED

            # Log message in audit trail with comprehensive metadata
            await self.audit_service.log_operation_start(
                user_id=user_id,
                operation_type=operation_type,
                operation_id=message_id,
                resource_name=f"Message in session {session_id}",
                metadata={
                    "session_id": session_id,
                    "message_id": message_id,
                    "message_type": message_type.value,
                    "content": content,
                    "content_length": len(content),
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                },
            )

            # Mark as successful immediately for messages
            await self.audit_service.log_operation_end(
                user_id=user_id, operation_id=message_id, status=OperationStatus.SUCCESS
            )

            logger.debug(
                f"Added {message_type.value} message {message_id} to session {session_id}"
            )
            return message_id

        except Exception as e:
            logger.error(f"Failed to add message to session {session_id}: {str(e)}")
            raise Exception(f"Message creation failed: {str(e)}") from e

    async def get_session_history(
        self, user_id: str, session_id: str, limit: int = 50
    ) -> Dict[str, Any]:
        """Retrieve complete chat history for a specific session.

        Fetches all messages and session context from the audit trail for the
        specified session. Returns a structured response with session metadata
        and chronologically ordered messages.

        Args:
            user_id: User identifier for access control
            session_id: Target session identifier
            limit: Maximum number of messages to return (default: 50)

        Returns:
            Dict containing:
                - session_id: The requested session ID
                - session_context: Session metadata and initial context
                - messages: List of messages with full metadata
                - total_messages: Count of messages returned

        Raises:
            ValueError: If required parameters are missing
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")
        if not session_id or not session_id.strip():
            raise ValueError("session_id is required and cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        try:
            rows = await db.fetch(
                """
                SELECT *
                FROM audit_trail
                WHERE user_id = %s
                  AND metadata ->> 'session_id' = %s
                  AND operation_type IN (%s, %s, %s)
                ORDER BY created_at ASC
                LIMIT %s
                """,
                user_id,
                session_id,
                OperationType.CHAT_MESSAGE_SENT.value,
                OperationType.CHAT_MESSAGE_RECEIVED.value,
                OperationType.CHAT_SESSION_STARTED.value,
                limit + 1,
            )

            if not rows:
                logger.warning(
                    f"No session data found for session {session_id} and user {user_id}"
                )
                return self._create_empty_session_response(session_id)

            # Transform audit trail entries to structured chat data
            messages = []
            session_context = {}

            for entry in rows:
                metadata = entry.get("metadata", {})

                if entry["operation_type"] == OperationType.CHAT_SESSION_STARTED.value:
                    # Extract session context information
                    session_context = {
                        "context_type": metadata.get("context_type"),
                        "context_id": metadata.get("context_id"),
                        "initial_context": metadata.get("initial_context", {}),
                        "title": entry.get("resource_name"),
                        "started_at": metadata.get("started_at", entry["created_at"]),
                    }
                else:
                    # Regular message entry
                    message_metadata = {
                        k: v
                        for k, v in metadata.items()
                        if k
                        not in [
                            "session_id",
                            "message_id",
                            "message_type",
                            "content",
                            "timestamp",
                        ]
                    }

                    messages.append(
                        {
                            "id": metadata.get("message_id", entry["operation_id"]),
                            "type": metadata.get("message_type"),
                            "content": metadata.get("content", ""),
                            "timestamp": metadata.get("timestamp", entry["created_at"]),
                            "metadata": message_metadata if message_metadata else None,
                        }
                    )

            # Limit messages to requested count
            messages = messages[:limit]

            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")

            return {
                "session_id": session_id,
                "session_context": session_context,
                "messages": messages,
                "total_messages": len(messages),
            }

        except Exception as e:
            logger.error(f"Error retrieving session history for {session_id}: {str(e)}")
            return self._create_empty_session_response(session_id)

    def _create_empty_session_response(self, session_id: str) -> Dict[str, Any]:
        """Create an empty session response for error cases."""
        return {
            "session_id": session_id,
            "session_context": {},
            "messages": [],
            "total_messages": 0,
        }

    async def get_user_sessions(
        self,
        user_id: str,
        context_type: Optional[ChatContextType] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve list of chat sessions for a specific user.

        Fetches user's chat sessions with metadata and message counts.
        Results are ordered by creation time (most recent first) and can
        be filtered by context type.

        Args:
            user_id: User identifier for session filtering
            context_type: Optional filter by session context type
            limit: Maximum number of sessions to return (default: 20)

        Returns:
            List of session dictionaries containing:
                - session_id: Unique session identifier
                - title: Human-readable session title
                - context_type: Type of chat context
                - context_id: Related resource ID (if any)
                - started_at: Session creation timestamp
                - message_count: Number of messages in session

        Raises:
            ValueError: If required parameters are missing
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        try:
            # Build query for session start entries
            params: List[Any] = [user_id, OperationType.CHAT_SESSION_STARTED.value]
            context_filter = ""
            if context_type:
                context_filter = " AND metadata ->> 'context_type' = %s"
                params.append(context_type.value)

            params.append(limit)

            rows = await db.fetch(
                f"""
                SELECT *
                FROM audit_trail
                WHERE user_id = %s
                  AND operation_type = %s
                  {context_filter}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                *params,
            )

            if not rows:
                logger.info(f"No chat sessions found for user {user_id}")
                return []

            # Build session list with enriched metadata
            sessions = []
            for entry in rows:
                metadata = entry.get("metadata", {})
                session_id = metadata.get("session_id", entry["operation_id"])

                # Get message count for this session (with error handling)
                try:
                    message_count = await self._get_session_message_count(
                        user_id, session_id
                    )
                except Exception as count_error:
                    logger.warning(
                        f"Failed to get message count for session {session_id}: {count_error}"
                    )
                    message_count = 0

                sessions.append(
                    {
                        "session_id": session_id,
                        "title": entry.get("resource_name"),
                        "context_type": metadata.get("context_type"),
                        "context_id": metadata.get("context_id"),
                        "started_at": metadata.get("started_at", entry["created_at"]),
                        "message_count": message_count,
                    }
                )

            logger.debug(f"Retrieved {len(sessions)} sessions for user {user_id}")
            return sessions

        except Exception as e:
            logger.error(f"Error retrieving user sessions for {user_id}: {str(e)}")
            return []

    async def _get_session_message_count(self, user_id: str, session_id: str) -> int:
        """Get total message count for a specific session.

        Counts both sent and received messages in the session by querying
        the audit trail for message-related operation types.

        Args:
            user_id: User identifier for access control
            session_id: Target session identifier

        Returns:
            int: Total number of messages in the session

        Raises:
            Exception: If database query fails
        """
        if not user_id or not session_id:
            return 0

        try:
            count = await db.fetchval(
                """
                SELECT COUNT(*)
                FROM audit_trail
                WHERE user_id = %s
                  AND metadata ->> 'session_id' = %s
                  AND operation_type IN (%s, %s)
                """,
                user_id,
                session_id,
                OperationType.CHAT_MESSAGE_SENT.value,
                OperationType.CHAT_MESSAGE_RECEIVED.value,
            )
            count = count or 0
            logger.debug(f"Session {session_id} has {count} messages")
            return count

        except Exception as e:
            logger.error(f"Error counting messages for session {session_id}: {str(e)}")
            return 0

    async def end_session(
        self, user_id: str, session_id: str, summary: Optional[str] = None
    ) -> bool:
        """End a chat session and log final statistics.

        Properly closes a chat session by logging the end event with
        final session statistics including message count and optional summary.

        Args:
            user_id: User identifier for access control
            session_id: Target session identifier to end
            summary: Optional session summary or conclusion

        Returns:
            bool: True if session was successfully ended, False otherwise

        Raises:
            ValueError: If required parameters are missing
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")
        if not session_id or not session_id.strip():
            raise ValueError("session_id is required and cannot be empty")

        try:
            # Get final message count for session statistics
            message_count = await self._get_session_message_count(user_id, session_id)

            # Log session end with comprehensive metadata
            end_operation_id = f"{session_id}_end"
            await self.audit_service.log_operation_start(
                user_id=user_id,
                operation_type=OperationType.CHAT_SESSION_ENDED,
                operation_id=end_operation_id,
                resource_name=f"Session {session_id} ended",
                metadata={
                    "session_id": session_id,
                    "ended_at": datetime.utcnow().isoformat(),
                    "final_message_count": message_count,
                    "summary": summary,
                    "duration_messages": message_count,
                },
            )

            # Mark session end as successful
            await self.audit_service.log_operation_end(
                user_id=user_id,
                operation_id=end_operation_id,
                status=OperationStatus.SUCCESS,
            )

            logger.info(
                f"Successfully ended chat session {session_id} for user {user_id} "
                f"(final message count: {message_count})"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error ending session {session_id} for user {user_id}: {str(e)}"
            )
            return False


# Global service instance
chat_session_service = ChatSessionService()
