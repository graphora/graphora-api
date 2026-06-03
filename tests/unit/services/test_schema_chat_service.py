"""Unit tests for SchemaChatService — BAML streaming migration (#18 Phase 2).

Pins:
  * Streaming partials are decomposed into deltas (FE doesn't get
    duplicate bytes)
  * ```schema ... ``` blocks in the accumulated response surface
    as schema_update events
  * Registry from get_baml_registry_for_user is threaded through
    baml_options so the user's provider is honored
  * Conversation history text formatter handles empty / partial input
  * Stream failure surfaces as an error event (not propagated)
"""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphora_server.services.schema_chat_service import SchemaChatService


# ---- Helpers --------------------------------------------------------------


class _AsyncStreamMock:
    """Minimal AsyncIterable yielding pre-set partials.

    BAML's b_async.stream.X(...) returns an object that supports
    async for — this mock reproduces that contract for tests
    without bringing in the real BAML runtime.
    """

    def __init__(self, partials: List[str]):
        self._partials = partials

    def __aiter__(self):
        async def gen():
            for p in self._partials:
                yield p

        return gen()


@pytest.fixture
def service():
    return SchemaChatService()


@pytest.fixture(name="_chat_service_stub")
def chat_service_stub_fixture(service):
    """Mock the underlying chat session storage so we can focus tests
    on the streaming flow.

    The fixture is requested with a leading underscore in test method
    signatures so vulture's dead-code scan doesn't flag the parameter
    as unused — pytest injects the fixture for its side effect
    (``service.chat_service = stub``), not for the returned value.
    """
    stub = MagicMock()
    stub.add_message = AsyncMock()
    stub.get_session_history = AsyncMock(
        return_value={"messages": [], "session_context": {}}
    )
    service.chat_service = stub
    return stub


# ---- Tests ---------------------------------------------------------------


class TestStreamChatResponse:
    @pytest.mark.asyncio
    async def test_yields_text_deltas_not_duplicates(
        self, service, _chat_service_stub
    ) -> None:
        """BAML yields the GROWING accumulated string per partial.
        The service must compute the delta so the FE doesn't render
        duplicated bytes."""
        fake_b_async = MagicMock()
        fake_b_async.stream.StreamSchemaChat = MagicMock(
            return_value=_AsyncStreamMock(["Hello", "Hello world", "Hello world!"])
        )

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(return_value=(MagicMock(), "gpt-4o-mini", "openai")),
            ),
            patch("graphora_server.baml_client.async_client.b", fake_b_async),
        ):
            chunks: List[Any] = []
            async for event in service.stream_chat_response(
                user_id="u1", session_id="s1", message="hi"
            ):
                chunks.append(event)

        text_events = [c for c in chunks if c["type"] == "text"]
        assert [c["content"] for c in text_events] == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_schema_block_surfaces_as_schema_update_event(
        self, service, _chat_service_stub
    ) -> None:
        """When the accumulated response contains a schema block, a
        schema_update event fires with the extracted YAML."""
        fake_b_async = MagicMock()
        partials = [
            "Here is the schema:\n",
            "Here is the schema:\n```schema\nversion: 0.1.0\nentities:\n  Person: {}\n```",
            "Here is the schema:\n```schema\nversion: 0.1.0\nentities:\n  Person: {}\n```\nLet me know if you want changes.",
        ]
        fake_b_async.stream.StreamSchemaChat = MagicMock(
            return_value=_AsyncStreamMock(partials)
        )

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(
                    return_value=(MagicMock(), "claude-sonnet-4-6", "anthropic")
                ),
            ),
            patch("graphora_server.baml_client.async_client.b", fake_b_async),
        ):
            chunks: List[Any] = []
            async for event in service.stream_chat_response(
                user_id="u1", session_id="s1", message="design a schema"
            ):
                chunks.append(event)

        schema_events = [c for c in chunks if c["type"] == "schema_update"]
        assert len(schema_events) == 1
        assert "Person" in schema_events[0]["content"]
        assert schema_events[0]["content"].startswith("version: 0.1.0")

    @pytest.mark.asyncio
    async def test_registry_threaded_through_baml_options(
        self, service, _chat_service_stub
    ) -> None:
        """Provider honored — the registry from
        get_baml_registry_for_user must reach the BAML stream call."""
        fake_registry = object()
        fake_b_async = MagicMock()
        fake_b_async.stream.StreamSchemaChat = MagicMock(
            return_value=_AsyncStreamMock(["ok"])
        )

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(return_value=(fake_registry, "llama3.3:70b", "ollama")),
            ),
            patch("graphora_server.baml_client.async_client.b", fake_b_async),
        ):
            async for _ in service.stream_chat_response(
                user_id="u1", session_id="s1", message="hi"
            ):
                pass

        kwargs = fake_b_async.stream.StreamSchemaChat.call_args.kwargs
        assert kwargs["baml_options"]["client_registry"] is fake_registry

    @pytest.mark.asyncio
    async def test_baml_failure_surfaces_as_error_event(
        self, service, _chat_service_stub
    ) -> None:
        """BAML stream raises → service yields a single error event
        rather than letting the exception propagate."""
        fake_b_async = MagicMock()
        fake_b_async.stream.StreamSchemaChat = MagicMock(
            side_effect=RuntimeError("BAML upstream offline")
        )

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(return_value=(MagicMock(), "x", "gemini")),
            ),
            patch("graphora_server.baml_client.async_client.b", fake_b_async),
        ):
            chunks: List[Any] = []
            async for event in service.stream_chat_response(
                user_id="u1", session_id="s1", message="hi"
            ):
                chunks.append(event)

        assert any(c["type"] == "error" for c in chunks)
        assert any(
            "BAML upstream offline" in c["content"]
            for c in chunks
            if c["type"] == "error"
        )


class TestBuildConversationHistoryText:
    def test_empty_history_returns_empty_string(self, service):
        assert service._build_conversation_history_text({"messages": []}) == ""

    def test_formats_role_and_content(self, service):
        session = {
            "messages": [
                {"type": "user_message", "content": "Hi"},
                {"type": "assistant_message", "content": "Hello there"},
                {"type": "user_message", "content": "Make a schema"},
            ]
        }
        text = service._build_conversation_history_text(session)
        assert "**User:** Hi" in text
        assert "**Assistant:** Hello there" in text
        assert "**User:** Make a schema" in text

    def test_skips_empty_content(self, service):
        session = {
            "messages": [
                {"type": "user_message", "content": "  "},
                {"type": "user_message", "content": "Real message"},
            ]
        }
        text = service._build_conversation_history_text(session)
        assert "Real message" in text
        assert text.count("**User:**") == 1

    def test_keeps_only_last_10_messages(self, service):
        # Zero-pad so "msg-001" isn't a substring of "msg-010" etc.
        session = {
            "messages": [
                {"type": "user_message", "content": f"msg-{i:03d}"} for i in range(20)
            ]
        }
        text = service._build_conversation_history_text(session)
        for i in range(10):
            assert f"msg-{i:03d}" not in text
        for i in range(10, 20):
            assert f"msg-{i:03d}" in text
