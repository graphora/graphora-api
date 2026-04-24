"""Chat Session API Endpoints

This module provides REST API endpoints for managing chat sessions and schema refinement.
Supports multi-turn conversations with persistent session state and context management.
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional

from graphora_server.schemas.chat import (
    StartSessionRequest,
    StartSessionResponse,
    SendMessageRequest,
    ChatSessionResponse,
    UserSessionsResponse,
    EndSessionRequest,
    SchemaRefinementRequest,
    SchemaRefinementResponse,
    GetCurrentSchemaResponse,
    OperationResponse,
)
from graphora_server.services.chat_session_service import (
    chat_session_service,
    ChatContextType,
    MessageType,
)
from graphora_server.services.schema_refinement_service import schema_refinement_service
from graphora_server.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat Sessions"])

# Chat Session Management Endpoints


@router.post("/sessions/start", response_model=StartSessionResponse)
async def start_chat_session(
    request: StartSessionRequest, user_id: str = Depends(get_current_user_id)
) -> StartSessionResponse:
    """
    Start a new chat session for multi-turn conversations.

    Args:
        request: Session creation parameters
        user_id: User identifier from header

    Returns:
        Session creation response with session ID

    Raises:
        HTTPException: If session creation fails or invalid parameters
    """
    try:
        # Validate context type
        try:
            context_type = ChatContextType(request.context_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid context_type. Must be one of: {[ct.value for ct in ChatContextType]}",
            )

        session_id = await chat_session_service.start_session(
            user_id=user_id,
            context_type=context_type,
            context_id=request.context_id,
            initial_context=request.initial_context,
            title=request.title,
        )

        logger.info(f"Started chat session {session_id} for user {user_id}")

        return StartSessionResponse(session_id=session_id, status="success")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting chat session for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start chat session")


@router.post("/sessions/{session_id}/messages", response_model=OperationResponse)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    user_id: str = Depends(get_current_user_id),
) -> OperationResponse:
    """
    Send a message to an existing chat session.

    Args:
        session_id: Target session identifier
        request: Message content and metadata
        user_id: User identifier from header

    Returns:
        Operation response with message ID

    Raises:
        HTTPException: If message sending fails or session not found
    """
    try:
        # Validate message type
        try:
            message_type = MessageType(request.message_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid message_type. Must be one of: {[mt.value for mt in MessageType]}",
            )

        message_id = await chat_session_service.add_message(
            user_id=user_id,
            session_id=session_id,
            message_type=message_type,
            content=request.message,
        )

        logger.info(
            f"Added message {message_id} to session {session_id} for user {user_id}"
        )

        return OperationResponse(
            status="success",
            message="Message sent successfully",
            data={"message_id": message_id},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error sending message to session {session_id} for user {user_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Failed to send message")


@router.get("/sessions/{session_id}/history", response_model=ChatSessionResponse)
async def get_session_history(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    limit: int = Query(
        50, ge=1, le=100, description="Maximum number of messages to return"
    ),
) -> ChatSessionResponse:
    """
    Retrieve chat history for a specific session.

    Args:
        session_id: Session identifier
        user_id: User identifier from header
        limit: Maximum number of messages to return

    Returns:
        Complete session data with message history

    Raises:
        HTTPException: If session not found or access denied
    """
    try:
        session_data = await chat_session_service.get_session_history(
            user_id=user_id, session_id=session_id, limit=limit
        )

        if not session_data or session_data.get("total_messages", 0) == 0:
            logger.warning(
                f"No session data found for session {session_id} and user {user_id}"
            )

        # Convert to response format
        from graphora_server.schemas.chat import ChatMessageResponse

        messages = [
            ChatMessageResponse(
                id=msg["id"],
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                metadata=msg.get("metadata"),
            )
            for msg in session_data.get("messages", [])
        ]

        return ChatSessionResponse(
            session_id=session_data["session_id"],
            session_context=session_data["session_context"],
            messages=messages,
            total_messages=session_data["total_messages"],
        )

    except Exception as e:
        logger.error(
            f"Error retrieving session history for {session_id} and user {user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve session history"
        )


@router.get("/sessions", response_model=UserSessionsResponse)
async def get_user_sessions(
    user_id: str = Depends(get_current_user_id),
    context_type: Optional[str] = Query(None, description="Filter by context type"),
    limit: int = Query(
        20, ge=1, le=50, description="Maximum number of sessions to return"
    ),
) -> UserSessionsResponse:
    """
    Get list of chat sessions for the authenticated user.

    Args:
        user_id: User identifier from header
        context_type: Optional filter by session context type
        limit: Maximum number of sessions to return

    Returns:
        List of user's chat sessions with metadata

    Raises:
        HTTPException: If retrieval fails or invalid context type
    """
    try:
        context_type_enum = None
        if context_type:
            try:
                context_type_enum = ChatContextType(context_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid context_type. Must be one of: {[ct.value for ct in ChatContextType]}",
                )

        sessions = await chat_session_service.get_user_sessions(
            user_id=user_id, context_type=context_type_enum, limit=limit
        )

        logger.info(f"Retrieved {len(sessions)} sessions for user {user_id}")

        return UserSessionsResponse(sessions=sessions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sessions for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user sessions")


@router.post("/sessions/{session_id}/end", response_model=OperationResponse)
async def end_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    request: Optional[EndSessionRequest] = None,
) -> OperationResponse:
    """
    End a chat session and optionally provide a summary.

    Args:
        session_id: Session to end
        user_id: User identifier from header
        request: Optional session summary

    Returns:
        Operation confirmation

    Raises:
        HTTPException: If session ending fails or session not found
    """
    try:
        summary = request.summary if request else None

        success = await chat_session_service.end_session(
            user_id=user_id, session_id=session_id, summary=summary
        )

        if success:
            logger.info(f"Successfully ended session {session_id} for user {user_id}")
            return OperationResponse(
                status="success", message="Session ended successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to end session")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session {session_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to end session")


# Schema Refinement Endpoints


@router.post("/schema-refinement", response_model=SchemaRefinementResponse)
async def refine_schema(
    request: SchemaRefinementRequest, user_id: str = Depends(get_current_user_id)
) -> SchemaRefinementResponse:
    """
    Refine a schema through conversational AI with multi-turn support.

    This endpoint supports both starting new refinement sessions and continuing
    existing conversations. The AI maintains context across interactions.

    Args:
        request: Schema refinement parameters and user request
        user_id: User identifier from header

    Returns:
        Refinement result with updated schema and explanation

    Raises:
        HTTPException: If refinement fails or invalid parameters
    """
    try:
        session_id = request.session_id

        # If no session provided, start a new refinement session
        if not session_id:
            if not request.initial_schema:
                raise HTTPException(
                    status_code=400,
                    detail="Either session_id or initial_schema must be provided",
                )

            session_id = await schema_refinement_service.start_refinement_session(
                user_id=user_id,
                initial_schema=request.initial_schema,
                schema_id=request.schema_id,
                context=request.context,
            )

            logger.info(
                f"Started new refinement session {session_id} for user {user_id}"
            )

        # Process the refinement request
        result = await schema_refinement_service.process_refinement_request(
            user_id=user_id, session_id=session_id, user_request=request.user_request
        )

        logger.info(
            f"Processed refinement request for session {session_id}, success: {result.get('success', False)}"
        )

        return SchemaRefinementResponse(session_id=session_id, **result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing schema refinement for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to process schema refinement"
        )


@router.get(
    "/schema-refinement/{session_id}/current", response_model=GetCurrentSchemaResponse
)
async def get_current_schema(
    session_id: str, user_id: str = Depends(get_current_user_id)
) -> GetCurrentSchemaResponse:
    """
    Get the current schema state from a refinement session.

    Args:
        session_id: Refinement session identifier
        user_id: User identifier from header

    Returns:
        Current schema content and metadata

    Raises:
        HTTPException: If session not found or schema unavailable
    """
    try:
        current_schema = await schema_refinement_service.get_current_schema(
            user_id=user_id, session_id=session_id
        )

        if current_schema is None:
            raise HTTPException(
                status_code=404, detail="Schema not found for the specified session"
            )

        logger.info(
            f"Retrieved current schema for session {session_id} and user {user_id}"
        )

        return GetCurrentSchemaResponse(
            session_id=session_id, current_schema=current_schema
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving current schema for session {session_id} and user {user_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve current schema")


# Freeflow Schema Chat Endpoints (Streaming)


@router.post("/schema-chat/start")
async def start_schema_chat(
    initial_schema: Optional[str] = None,
    title: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    """
    Start a new freeflow schema chat session.

    This endpoint creates a conversational session for designing knowledge graph
    schemas through natural language. Unlike the guided Q&A flow, this allows
    freeform conversation from the start.

    Args:
        initial_schema: Optional existing schema to start with
        title: Optional session title
        user_id: User identifier from header

    Returns:
        Session info with ID and welcome message
    """
    from graphora_server.services.schema_chat_service import schema_chat_service

    try:
        result = await schema_chat_service.start_chat_session(
            user_id=user_id,
            initial_schema=initial_schema,
            title=title,
        )

        logger.info(
            f"Started schema chat session {result['session_id']} for user {user_id}"
        )

        return result

    except Exception as e:
        logger.error(f"Error starting schema chat for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to start schema chat session"
        )


@router.post("/schema-chat/{session_id}/stream")
async def stream_schema_chat(
    session_id: str,
    message: str,
    current_schema: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    """
    Stream a chat response for schema generation/refinement.

    This endpoint streams the AI response in real-time using Server-Sent Events (SSE).
    Schema updates are sent as separate events when the AI generates or modifies a schema.

    Event types:
    - text: Regular text content
    - schema_update: New or updated schema YAML
    - done: Stream complete with final metadata
    - error: Error occurred

    Args:
        session_id: Chat session ID
        message: User's message
        current_schema: Current schema state (optional)
        user_id: User identifier from header

    Returns:
        StreamingResponse with SSE events
    """
    from graphora_server.services.schema_chat_service import schema_chat_service

    async def generate():
        try:
            async for chunk in schema_chat_service.stream_chat_response(
                user_id=user_id,
                session_id=session_id,
                message=message,
                current_schema=current_schema,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"Error in schema chat stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/schema-chat/{session_id}")
async def get_schema_chat_state(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Get the current state of a schema chat session.

    Args:
        session_id: Chat session ID
        user_id: User identifier from header

    Returns:
        Session state with messages and current schema
    """
    from graphora_server.services.schema_chat_service import schema_chat_service

    try:
        result = await schema_chat_service.get_session_state(
            user_id=user_id,
            session_id=session_id,
        )

        return result

    except Exception as e:
        logger.error(
            f"Error getting schema chat state for session {session_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Failed to get session state")
