"""Schema Chat Service with Streaming Support

Provides freeflow conversational schema generation and refinement with real-time
streaming responses. Users can describe their domain in natural language and
iteratively build schemas through conversation.
"""

import logging
import re
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime

from graphora_server.services.chat_session_service import (
    ChatSessionService,
    MessageType,
    ChatContextType,
)
from graphora_server.services.audit_service import AuditService

logger = logging.getLogger(__name__)

SCHEMA_CHAT_SYSTEM_PROMPT = """You are a helpful assistant that helps users design knowledge graph schemas through natural conversation.

Your role:
1. Understand the user's domain, data sources, and use cases
2. Suggest appropriate entity types and their properties
3. Identify relationships between entities
4. Iteratively refine the schema based on feedback
5. Output schema updates in YAML format when appropriate

When you create or update a schema, wrap it in:
```schema
<yaml content>
```

Schema Format Requirements:
- Use YAML format with version: "0.1.0"
- Structure: entities -> EntityName -> properties/relationships
- Properties have: type (str/int/float/bool/date), description, optionally: unique, required, index
- Relationships have: target entity, optionally properties
- Entity names: CamelCase (e.g., Person, Company)
- Relationship names: UPPER_SNAKE_CASE (e.g., WORKS_AT, REPORTS_TO)
- Property names: camelCase (e.g., firstName, createdAt)

Example schema:
```schema
version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
        description: "Full name of the person"
        required: true
      email:
        type: str
        description: "Email address"
        unique: true
    relationships:
      WORKS_AT:
        target: Company
        properties:
          startDate:
            type: date
  Company:
    properties:
      name:
        type: str
        required: true
      industry:
        type: str
```

Guidelines:
- Be conversational and helpful
- Ask clarifying questions when the domain is unclear
- Start simple and add complexity as needed
- Explain your schema choices
- When user asks for changes, show the updated schema
- If no schema exists yet, wait until you understand the domain before creating one
"""


class SchemaChatService:
    """Service for freeflow schema generation through streaming chat."""

    def __init__(self) -> None:
        self.chat_service = ChatSessionService()
        self.audit_service = AuditService()

    async def start_chat_session(
        self,
        user_id: str,
        initial_schema: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a new freeflow schema chat session.

        Args:
            user_id: User identifier
            initial_schema: Optional existing schema to start with
            title: Optional session title

        Returns:
            Dict with session_id and welcome message
        """
        session_id = await self.chat_service.start_session(
            user_id=user_id,
            context_type=ChatContextType.SCHEMA_GENERATION,
            initial_context={
                "current_schema": initial_schema,
                "schema_version": 1,
                "created_at": datetime.utcnow().isoformat(),
            },
            title=title or "Schema Design Session",
        )

        # Generate welcome message based on context
        if initial_schema:
            welcome = """I see you have an existing schema. I can help you:
- Add new entities or relationships
- Modify existing properties
- Restructure the schema
- Optimize for specific use cases

What would you like to change?"""
        else:
            welcome = """Hi! I'll help you design a knowledge graph schema.

Tell me about your domain:
- What kind of data do you work with?
- What are the main things (entities) you want to track?
- How are they connected?

For example: "I'm building a system to track research papers, their authors, and citations" or "I need to model an e-commerce platform with products, customers, and orders"."""

        # Store welcome message
        await self.chat_service.add_message(
            user_id=user_id,
            session_id=session_id,
            message_type=MessageType.ASSISTANT_MESSAGE,
            content=welcome,
            metadata={
                "message_type": "welcome",
                "has_initial_schema": bool(initial_schema),
            },
        )

        return {
            "session_id": session_id,
            "welcome_message": welcome,
            "current_schema": initial_schema,
        }

    async def stream_chat_response(
        self,
        user_id: str,
        session_id: str,
        message: str,
        current_schema: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat response for schema generation/refinement.

        Args:
            user_id: User identifier
            session_id: Chat session ID
            message: User's message
            current_schema: Current schema state (optional, will be fetched if not provided)

        Yields:
            Dict chunks with type (text/schema_update/done/error) and content
        """
        try:
            # Record user message
            await self.chat_service.add_message(
                user_id=user_id,
                session_id=session_id,
                message_type=MessageType.USER_MESSAGE,
                content=message,
                metadata={"timestamp": datetime.utcnow().isoformat()},
            )

            # Get conversation history for context
            session_data = await self.chat_service.get_session_history(
                user_id, session_id
            )

            # Get current schema from session if not provided
            if not current_schema:
                current_schema = await self._get_schema_from_session(session_data)

            # Format conversation history as a plain string (BAML's
            # template engine ingests strings, not Gemini's role/parts
            # message structure).
            conversation_history = self._build_conversation_history_text(session_data)

            # Cross-provider streaming via BAML — works for gemini,
            # openai, anthropic, ollama (#18 Phase 2).
            from graphora_server.baml_client.async_client import b as b_async
            from graphora_server.utils.llm_helper import (
                get_baml_registry_for_user,
            )

            registry, model_name, provider = await get_baml_registry_for_user(user_id)

            full_response = ""
            previous_partial = ""
            schema_content = None

            stream = b_async.stream.StreamSchemaChat(
                system_prompt=SCHEMA_CHAT_SYSTEM_PROMPT,
                conversation_history=conversation_history,
                current_schema=current_schema or "",
                user_message=message,
                baml_options={"client_registry": registry},
            )

            async for partial in stream:
                if not partial:
                    continue
                # BAML yields the GROWING accumulated string per partial;
                # compute the delta so we don't re-yield bytes the FE
                # already rendered.
                delta = partial[len(previous_partial) :]
                previous_partial = partial
                full_response = partial

                if delta:
                    yield {"type": "text", "content": delta}

                # Schema block detection — same regex as the pre-BAML
                # implementation, run against the accumulated response
                # so we surface the block as soon as both fences land.
                schema_match = re.search(
                    r"```schema\n(.*?)```", full_response, re.DOTALL
                )
                if schema_match and not schema_content:
                    schema_content = schema_match.group(1).strip()
                    yield {"type": "schema_update", "content": schema_content}

            # Approximate token counts (BAML's usage tracker covers the
            # call internally; we keep this here for parity with the
            # pre-BAML code path until the streaming-usage hook lands).
            logger.info(
                "Schema chat streamed via BAML",
                extra={
                    "user_id": user_id,
                    "provider": provider,
                    "model": model_name,
                    "response_chars": len(full_response),
                    "schema_extracted": bool(schema_content),
                },
            )

            # Store assistant response
            await self.chat_service.add_message(
                user_id=user_id,
                session_id=session_id,
                message_type=MessageType.ASSISTANT_MESSAGE,
                content=full_response,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "schema_updated": bool(schema_content),
                    "schema_content": schema_content,
                },
            )

            # Yield completion signal
            yield {
                "type": "done",
                "content": {
                    "schema": schema_content,
                    "message_length": len(full_response),
                },
            }

        except Exception as e:
            logger.error(f"Error in schema chat stream: {str(e)}")
            yield {"type": "error", "content": str(e)}

    def _build_conversation_history_text(
        self,
        session_data: Dict[str, Any],
    ) -> str:
        """Format the last 10 messages as a plain-text conversation log.

        Replaces the Gemini-specific ``_build_conversation_messages`` —
        BAML's template engine ingests strings, not role/parts message
        dicts. Returns an empty string when there's no history (the
        BAML template tolerates that — empty section, prompt still
        coherent).
        """
        history_messages = session_data.get("messages", [])[-10:]
        if not history_messages:
            return ""

        formatted = []
        for msg in history_messages:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            role = "User" if msg.get("type") == "user_message" else "Assistant"
            formatted.append(f"**{role}:** {content}")
        return "\n\n".join(formatted)

    async def _get_schema_from_session(
        self, session_data: Dict[str, Any]
    ) -> Optional[str]:
        """Extract the latest schema from session data."""
        # Check initial context
        initial_context = session_data.get("session_context", {}).get(
            "initial_context", {}
        )
        schema = initial_context.get("current_schema")

        # Look for schema updates in messages (newest first)
        for msg in reversed(session_data.get("messages", [])):
            if msg.get("type") == "assistant_message":
                metadata = msg.get("metadata", {})
                if metadata.get("schema_content"):
                    schema = metadata["schema_content"]
                    break

        return schema

    async def get_session_state(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get the current state of a chat session.

        Returns:
            Dict with messages, current_schema, and metadata
        """
        session_data = await self.chat_service.get_session_history(user_id, session_id)
        current_schema = await self._get_schema_from_session(session_data)

        return {
            "session_id": session_id,
            "messages": session_data.get("messages", []),
            "current_schema": current_schema,
            "message_count": len(session_data.get("messages", [])),
        }


# Global service instance
schema_chat_service = SchemaChatService()
