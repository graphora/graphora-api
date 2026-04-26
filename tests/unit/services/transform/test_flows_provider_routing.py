"""Tests for provider-aware PDF routing in flows.py.

Stage 2 of the Ollama wiring: when LLM_PROVIDER=ollama (or the user's
DB provider is ollama), PDFs must be pre-extracted to text via
DocumentParser instead of going through the Gemini-binary path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphora_server.services.transform.flows import (
    _pdf_to_text_file,
    _should_pre_extract_pdfs,
)


# ---- _should_pre_extract_pdfs --------------------------------------------


class TestShouldPreExtractPdfs:
    @pytest.mark.asyncio
    async def test_env_var_ollama_returns_true_no_db_call(self) -> None:
        fake_settings = MagicMock(LLM_PROVIDER="ollama")
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.services.ai_config_service.AIConfigService"
            ) as ai_cls,
        ):
            assert await _should_pre_extract_pdfs("user-1") is True
        ai_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_env_var_ollama_case_insensitive(self) -> None:
        fake_settings = MagicMock(LLM_PROVIDER="OLLAMA")
        with patch("graphora_server.config.get_settings", return_value=fake_settings):
            assert await _should_pre_extract_pdfs("user-1") is True

    @pytest.mark.asyncio
    async def test_db_backed_ollama_returns_true(self) -> None:
        fake_settings = MagicMock(LLM_PROVIDER=None)
        fake_ai = MagicMock()
        fake_ai.get_user_provider_secret = AsyncMock(
            return_value=("ollama", "http://x:11434", "llama3.2")
        )
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.services.ai_config_service.AIConfigService",
                return_value=fake_ai,
            ),
        ):
            assert await _should_pre_extract_pdfs("user-1") is True

    @pytest.mark.asyncio
    async def test_db_backed_gemini_returns_false(self) -> None:
        """Default Gemini path: keep PDF binary fast path."""
        fake_settings = MagicMock(LLM_PROVIDER=None)
        fake_ai = MagicMock()
        fake_ai.get_user_provider_secret = AsyncMock(
            return_value=("gemini", "key", "gemini-2.5-flash")
        )
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.services.ai_config_service.AIConfigService",
                return_value=fake_ai,
            ),
        ):
            assert await _should_pre_extract_pdfs("user-1") is False

    @pytest.mark.asyncio
    async def test_no_user_config_returns_false(self) -> None:
        fake_settings = MagicMock(LLM_PROVIDER=None)
        fake_ai = MagicMock()
        fake_ai.get_user_provider_secret = AsyncMock(return_value=None)
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.services.ai_config_service.AIConfigService",
                return_value=fake_ai,
            ),
        ):
            assert await _should_pre_extract_pdfs("user-1") is False

    @pytest.mark.asyncio
    async def test_db_error_fails_closed_to_false(self) -> None:
        """Transient DB failure must NOT silently switch users to Ollama path."""
        fake_settings = MagicMock(LLM_PROVIDER=None)
        fake_ai = MagicMock()
        fake_ai.get_user_provider_secret = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.services.ai_config_service.AIConfigService",
                return_value=fake_ai,
            ),
        ):
            assert await _should_pre_extract_pdfs("user-1") is False


# ---- _pdf_to_text_file ----------------------------------------------------


class TestPdfToTextFile:
    @pytest.mark.asyncio
    async def test_writes_text_file_on_success(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "pdf-text"
        with patch(
            "graphora_server.services.document_parser.DocumentParser.parse_file",
            new_callable=AsyncMock,
            return_value="hello world",
        ):
            result = await _pdf_to_text_file("/some/where/source.pdf", out_dir)

        assert result is not None
        out_path = Path(result)
        assert out_path.exists()
        assert out_path.read_text() == "hello world"
        assert out_path.suffix == ".txt"

    @pytest.mark.asyncio
    async def test_returns_none_when_parser_yields_nothing(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "pdf-text"
        with patch(
            "graphora_server.services.document_parser.DocumentParser.parse_file",
            new_callable=AsyncMock,
            return_value=None,
        ):
            assert await _pdf_to_text_file("/x.pdf", out_dir) is None

    @pytest.mark.asyncio
    async def test_returns_none_on_whitespace_only(self, tmp_path: Path) -> None:
        with patch(
            "graphora_server.services.document_parser.DocumentParser.parse_file",
            new_callable=AsyncMock,
            return_value="   \n\n  ",
        ):
            assert await _pdf_to_text_file("/x.pdf", tmp_path) is None
