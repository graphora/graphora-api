"""Unit tests for the PDF extraction provider-aware routing (#18 Phase 3).

Pins:
  * Gemini-configured users → native PDF flow (create_gemini_client is
    called; PDF→text fallback is NOT)
  * Non-Gemini users (openai / anthropic / ollama) → fallback to
    extract_nodes_from_chunk with text from DocumentParser
  * Unparseable PDF on the fallback path → raises ValueError with a
    helpful provider-switch hint
  * Both ``extract_nodes_from_pdf`` and ``extract_relationships_from_pdf``
    share the same routing posture
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphora_server.exceptions import NoAIConfigurationError
from graphora_server.services.llm.client import LLMClient


@pytest.fixture
def client():
    return LLMClient()


def _patch_ai_config(provider_name: str, api_key: str = "k", model: str = "m"):
    """Patch AIConfigService so it returns the given provider tuple."""
    fake_service = MagicMock()
    fake_service.get_user_provider_secret = AsyncMock(
        return_value=(provider_name, api_key, model)
    )
    return patch(
        "graphora_server.services.ai_config_service.AIConfigService",
        return_value=fake_service,
    )


def _patch_ai_config_missing():
    fake_service = MagicMock()
    fake_service.get_user_provider_secret = AsyncMock(return_value=None)
    return patch(
        "graphora_server.services.ai_config_service.AIConfigService",
        return_value=fake_service,
    )


class TestExtractNodesFromPdfRouting:
    @pytest.mark.asyncio
    async def test_non_gemini_routes_to_chunk_via_text_fallback(self, client) -> None:
        """openai user → PDF parsed to text → extract_nodes_from_chunk."""
        sentinel_result = MagicMock(name="chunk-extraction-result")

        fake_parser = MagicMock()
        fake_parser.parse_file = AsyncMock(return_value="Extracted PDF text content")

        with (
            _patch_ai_config("openai"),
            patch(
                "graphora_server.services.document_parser.DocumentParser",
                return_value=fake_parser,
            ),
            patch.object(
                client,
                "extract_nodes_from_chunk",
                new=AsyncMock(return_value=sentinel_result),
            ) as mock_chunk_extract,
        ):
            result = await client.extract_nodes_from_pdf(
                pdf_path="/tmp/test.pdf",
                response_model=MagicMock(),
                ontology_yaml="entities: {}",
                user_id="u1",
            )

        assert result is sentinel_result
        # chunk extraction received the parsed PDF text as the chunk
        mock_chunk_extract.assert_awaited_once()
        kwargs = mock_chunk_extract.call_args.kwargs
        assert kwargs["chunk"] == "Extracted PDF text content"
        assert kwargs["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_unparseable_pdf_raises_with_provider_switch_hint(
        self, client
    ) -> None:
        """When the PDF can't be parsed for a non-Gemini user, the
        error should point them at the gemini-provider workaround."""
        fake_parser = MagicMock()
        fake_parser.parse_file = AsyncMock(return_value=None)  # parse failed

        with (
            _patch_ai_config("anthropic"),
            patch(
                "graphora_server.services.document_parser.DocumentParser",
                return_value=fake_parser,
            ),
            pytest.raises(ValueError, match="gemini provider"),
        ):
            await client.extract_nodes_from_pdf(
                pdf_path="/tmp/broken.pdf",
                response_model=MagicMock(),
                ontology_yaml="entities: {}",
                user_id="u1",
            )

    @pytest.mark.asyncio
    async def test_missing_ai_config_raises_no_ai_configuration_error(
        self, client
    ) -> None:
        """No user config → NoAIConfigurationError (preserves contract
        of the pre-routing implementation, which raised the same)."""
        with (
            _patch_ai_config_missing(),
            pytest.raises(NoAIConfigurationError),
        ):
            await client.extract_nodes_from_pdf(
                pdf_path="/tmp/test.pdf",
                response_model=MagicMock(),
                ontology_yaml="",
                user_id="u1",
            )


class TestExtractRelationshipsFromPdfRouting:
    @pytest.mark.asyncio
    async def test_non_gemini_routes_to_chunk_via_text_fallback(self, client) -> None:
        """ollama user → PDF parsed to text → extract_relationships_from_chunk."""
        sentinel_result = MagicMock(name="chunk-extraction-result")

        fake_parser = MagicMock()
        fake_parser.parse_file = AsyncMock(return_value="Some PDF text")

        with (
            _patch_ai_config("ollama"),
            patch(
                "graphora_server.services.document_parser.DocumentParser",
                return_value=fake_parser,
            ),
            patch.object(
                client,
                "extract_relationships_from_chunk",
                new=AsyncMock(return_value=sentinel_result),
            ) as mock_chunk_extract,
        ):
            result = await client.extract_relationships_from_pdf(
                pdf_path="/tmp/test.pdf",
                response_model=MagicMock(),
                ontology_yaml="entities: {}",
                user_id="u1",
            )

        assert result is sentinel_result
        kwargs = mock_chunk_extract.call_args.kwargs
        assert kwargs["chunk"] == "Some PDF text"
