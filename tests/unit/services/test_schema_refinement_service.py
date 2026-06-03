"""Unit tests for SchemaRefinementService — focused on the BAML migration.

PR #18 migrated ``_refine_schema_with_llm`` from a direct ``create_gemini_client``
call to BAML's ``RefineSchemaConversational`` so the flow works cross-
provider (gemini / openai / anthropic / ollama). These tests pin:

  * Happy path returns the typed 4-tuple unchanged
  * Conversation-context slicing happens in Python (BAML template gets
    pre-formatted strings)
  * BAML failure falls back to ``_fallback_refinement`` so chat
    refinement stays available
  * Provider is honored — the BAML registry from
    ``get_baml_registry_for_user`` is threaded through ``baml_options``
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphora_server.services.schema_refinement_service import (
    SchemaRefinementService,
)


@pytest.fixture
def service():
    return SchemaRefinementService()


def _baml_result(
    refined_schema: str = "version: 0.1.0\nentities: {}\n",
    changes_made=None,
    confidence: float = 0.85,
    explanation: str = "Test refinement",
):
    """Build a fake BAML result with the shape of SchemaRefinementResponse."""
    return SimpleNamespace(
        refined_schema=refined_schema,
        changes_made=changes_made or ["change-1"],
        confidence=confidence,
        explanation=explanation,
    )


class TestRefineSchemaWithLlm:
    @pytest.mark.asyncio
    async def test_happy_path_returns_four_tuple(self, service) -> None:
        """BAML success → returns (refined_schema, changes_made, confidence, explanation)."""
        fake_registry = MagicMock()
        fake_b = MagicMock()
        fake_b.RefineSchemaConversational = MagicMock(
            return_value=_baml_result(
                refined_schema="version: 0.1.0\nentities:\n  Person: {}\n",
                changes_made=["Added Person entity"],
                confidence=0.92,
                explanation="Added a Person entity per request",
            )
        )

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(return_value=(fake_registry, "gpt-4o-mini", "openai")),
            ),
            patch("graphora_server.baml_client.b", fake_b),
        ):
            result = await service._refine_schema_with_llm(
                user_id="u1",
                current_schema="version: 0.1.0\nentities: {}\n",
                user_request="Add a Person entity",
                conversation_context={},
            )

        refined, changes, confidence, explanation = result
        assert "Person" in refined
        assert changes == ["Added Person entity"]
        assert confidence == 0.92
        assert "Person" in explanation

    @pytest.mark.asyncio
    async def test_conversation_context_slicing_passes_through(self, service) -> None:
        """Previous requests get [-2:] sliced; changes history gets [-3:] sliced."""
        fake_registry = MagicMock()
        fake_b = MagicMock()
        fake_b.RefineSchemaConversational = MagicMock(return_value=_baml_result())

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(
                    return_value=(fake_registry, "claude-sonnet-4-6", "anthropic")
                ),
            ),
            patch("graphora_server.baml_client.b", fake_b),
        ):
            await service._refine_schema_with_llm(
                user_id="u1",
                current_schema="...",
                user_request="Latest change request",
                conversation_context={
                    "refinement_count": 5,
                    "previous_requests": ["r1", "r2", "r3", "r4"],
                    "changes_history": ["c1", "c2", "c3", "c4", "c5"],
                },
            )

        kwargs = fake_b.RefineSchemaConversational.call_args.kwargs
        # refinement_count is incremented (+1) so the prompt reads "#6"
        assert kwargs["refinement_count"] == 6
        # Slicing applied: last 2 previous requests
        assert kwargs["previous_requests_summary"] == "r3, r4"
        # Last 3 changes
        assert kwargs["changes_history_summary"] == "c3, c4, c5"

    @pytest.mark.asyncio
    async def test_empty_conversation_context_yields_none_string(self, service) -> None:
        """Missing/empty context fields → 'None' literal in the BAML prompt."""
        fake_registry = MagicMock()
        fake_b = MagicMock()
        fake_b.RefineSchemaConversational = MagicMock(return_value=_baml_result())

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(
                    return_value=(fake_registry, "gemini-2.5-flash", "gemini")
                ),
            ),
            patch("graphora_server.baml_client.b", fake_b),
        ):
            await service._refine_schema_with_llm(
                user_id="u1",
                current_schema="...",
                user_request="First request",
                conversation_context={},
            )

        kwargs = fake_b.RefineSchemaConversational.call_args.kwargs
        assert kwargs["refinement_count"] == 1  # 0 + 1
        assert kwargs["previous_requests_summary"] == "None"
        assert kwargs["changes_history_summary"] == "None"

    @pytest.mark.asyncio
    async def test_registry_threaded_through_baml_options(self, service) -> None:
        """The BAML call must receive client_registry so the provider is honored."""
        fake_registry = object()  # sentinel object — identity check
        fake_b = MagicMock()
        fake_b.RefineSchemaConversational = MagicMock(return_value=_baml_result())

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(return_value=(fake_registry, "llama3.3:70b", "ollama")),
            ),
            patch("graphora_server.baml_client.b", fake_b),
        ):
            await service._refine_schema_with_llm(
                user_id="u1",
                current_schema="...",
                user_request="...",
                conversation_context={},
            )

        kwargs = fake_b.RefineSchemaConversational.call_args.kwargs
        assert kwargs["baml_options"]["client_registry"] is fake_registry

    @pytest.mark.asyncio
    async def test_baml_failure_falls_back(self, service) -> None:
        """BAML raises → fallback path returns a result instead of propagating."""
        fake_b = MagicMock()
        fake_b.RefineSchemaConversational = MagicMock(
            side_effect=RuntimeError("BAML upstream offline")
        )
        fake_fallback = AsyncMock(
            return_value=("fallback-yaml", ["fb"], 0.5, "fb-expl")
        )

        with (
            patch(
                "graphora_server.utils.llm_helper.get_baml_registry_for_user",
                new=AsyncMock(return_value=(MagicMock(), "x", "gemini")),
            ),
            patch("graphora_server.baml_client.b", fake_b),
            patch.object(service, "_fallback_refinement", new=fake_fallback),
        ):
            result = await service._refine_schema_with_llm(
                user_id="u1",
                current_schema="some schema",
                user_request="some request",
                conversation_context={},
            )

        assert result == ("fallback-yaml", ["fb"], 0.5, "fb-expl")
        fake_fallback.assert_awaited_once_with("some schema", "some request")
