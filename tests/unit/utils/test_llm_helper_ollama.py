"""Tests for the Ollama provider abstraction in llm_helper.

Covers:
* Env-var fast path (LLM_PROVIDER=ollama) skips DB lookup
* DB-backed Ollama (provider=ollama in user config) uses api_key as host
* DB-backed Gemini still works
* The Ollama client adapter exposes the genai-shaped interface used by
  schema_inference and schema_postprocess
* Unsupported provider raises a clear error
* Missing LLM config raises NoAIConfigurationError
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphora_server.exceptions import (
    NoAIConfigurationError,
    UnsupportedProviderError,
)
from graphora_server.utils import llm_helper


# ---- The Ollama client adapter -------------------------------------------


class TestOllamaGenAICompat:
    """The adapter must mimic genai.Client.models.generate_content."""

    def _patched_ollama(self):
        """Build a fake ``ollama`` module so the adapter doesn't need
        the real package installed."""
        fake_module = MagicMock()
        fake_client = MagicMock()
        fake_client.generate.return_value = {"response": "hello world"}
        fake_module.Client.return_value = fake_client
        return fake_module, fake_client

    def test_generate_content_returns_text_attribute(self) -> None:
        fake_module, fake_client = self._patched_ollama()
        with patch.dict("sys.modules", {"ollama": fake_module}):
            adapter = llm_helper.create_ollama_client(
                "http://localhost:11434", "llama3.2"
            )
            result = adapter.models.generate_content(
                model="llama3.2",
                contents=["please summarize"],
                config=None,
            )
        assert result.text == "hello world"
        # Confirm the real ollama.Client.generate was called once with
        # the prompt joined and model passed through.
        fake_client.generate.assert_called_once()
        call_kwargs = fake_client.generate.call_args.kwargs
        assert call_kwargs["model"] == "llama3.2"
        assert call_kwargs["prompt"] == "please summarize"
        assert call_kwargs["stream"] is False

    def test_joins_list_contents_into_single_prompt(self) -> None:
        fake_module, fake_client = self._patched_ollama()
        with patch.dict("sys.modules", {"ollama": fake_module}):
            adapter = llm_helper.create_ollama_client("http://localhost:11434", "m")
            adapter.models.generate_content(
                contents=["line 1", "line 2"],
                config=None,
            )
        assert fake_client.generate.call_args.kwargs["prompt"] == "line 1\nline 2"

    def test_translates_genai_config_to_ollama_options(self) -> None:
        fake_module, fake_client = self._patched_ollama()
        with patch.dict("sys.modules", {"ollama": fake_module}):
            adapter = llm_helper.create_ollama_client("http://localhost:11434", "m")

            # Mimic types.GenerateContentConfig — the schema_postprocess
            # service uses temperature + max_output_tokens.
            class _FakeConfig:
                temperature = 0.2
                max_output_tokens = 4096

            adapter.models.generate_content(contents="x", config=_FakeConfig())
        opts = fake_client.generate.call_args.kwargs["options"]
        # max_output_tokens → num_predict (Ollama option name).
        assert opts == {"temperature": 0.2, "num_predict": 4096}

    def test_response_carries_token_counts_via_usage_metadata(self) -> None:
        """B5-obs reviewer fix (commit 9f8e5e7 → this fix): the
        Ollama adapter previously discarded
        ``prompt_eval_count`` / ``eval_count`` from the SDK
        response, so /cost saw 0 tokens for any Ollama-backed
        call. Pin: the response object exposes a Gemini-shaped
        ``usage_metadata`` so the same set_usage_from_response
        extractor in llm_usage_tracker works without provider-
        aware branching."""
        fake_module = MagicMock()
        fake_client = MagicMock()
        fake_client.generate.return_value = {
            "response": "hello",
            "prompt_eval_count": 200,
            "eval_count": 50,
        }
        fake_module.Client.return_value = fake_client
        with patch.dict("sys.modules", {"ollama": fake_module}):
            adapter = llm_helper.create_ollama_client("http://localhost:11434", "m")
            result = adapter.models.generate_content(contents="x", config=None)

        assert result.text == "hello"
        # Mirror Gemini's usage_metadata field names.
        assert result.usage_metadata.prompt_token_count == 200
        assert result.usage_metadata.candidates_token_count == 50
        assert result.usage_metadata.total_token_count == 250

    def test_response_handles_missing_token_counts_gracefully(self) -> None:
        """Older Ollama servers / cached responses may not include
        ``prompt_eval_count`` or ``eval_count``. Pin: the adapter
        falls back to 0 rather than raising — the call's tokens
        record as 0 in /cost (which is honest, not wrong)."""
        fake_module, fake_client = self._patched_ollama()
        with patch.dict("sys.modules", {"ollama": fake_module}):
            adapter = llm_helper.create_ollama_client("http://localhost:11434", "m")
            result = adapter.models.generate_content(contents="x", config=None)

        assert result.usage_metadata.prompt_token_count == 0
        assert result.usage_metadata.candidates_token_count == 0
        assert result.usage_metadata.total_token_count == 0


# ---- get_llm_client_for_user routing -------------------------------------


@pytest.mark.asyncio
async def test_env_var_ollama_skips_db_lookup() -> None:
    """When LLM_PROVIDER=ollama is set, the DB is never touched."""
    fake_module = MagicMock()
    fake_module.Client.return_value = MagicMock()

    fake_settings = MagicMock(
        LLM_PROVIDER="ollama",
        OLLAMA_HOST="http://localhost:11434",
        OLLAMA_MODEL="llama3.2",
    )
    with (
        patch.dict("sys.modules", {"ollama": fake_module}),
        patch("graphora_server.config.get_settings", return_value=fake_settings),
        patch("graphora_server.utils.llm_helper.AIConfigService") as ai_config_cls,
    ):
        client, model_name, provider = await llm_helper.get_llm_client_for_user(
            "user-1"
        )

    assert provider == "ollama"
    assert model_name == "llama3.2"
    # Adapter must expose .models.generate_content shape.
    assert hasattr(client, "models")
    assert hasattr(client.models, "generate_content")
    # AIConfigService must NOT be instantiated on the env-var path.
    ai_config_cls.assert_not_called()


@pytest.mark.asyncio
async def test_db_backed_gemini_returns_real_genai_client() -> None:
    """No env override → DB lookup → Gemini client constructed."""
    fake_settings = MagicMock(
        LLM_PROVIDER=None,
        OLLAMA_HOST="http://localhost:11434",
        OLLAMA_MODEL="x",
    )
    fake_ai_config = MagicMock()
    fake_ai_config.get_user_ai_config = AsyncMock(return_value={"some": "config"})
    fake_ai_config.get_user_provider_secret = AsyncMock(
        return_value=("gemini", "real-key", "gemini-2.5-flash")
    )

    sentinel_client: Any = object()
    with (
        patch("graphora_server.config.get_settings", return_value=fake_settings),
        patch(
            "graphora_server.utils.llm_helper.AIConfigService",
            return_value=fake_ai_config,
        ),
        patch(
            "graphora_server.utils.llm_helper.create_gemini_client",
            return_value=sentinel_client,
        ),
    ):
        client, model_name, provider = await llm_helper.get_llm_client_for_user(
            "user-1"
        )

    assert client is sentinel_client
    assert model_name == "gemini-2.5-flash"
    assert provider == "gemini"


@pytest.mark.asyncio
async def test_db_backed_ollama_uses_api_key_column_as_host() -> None:
    """When the user's stored provider is 'ollama', api_key holds the host."""
    fake_module = MagicMock()
    fake_module.Client.return_value = MagicMock()
    fake_settings = MagicMock(
        LLM_PROVIDER=None,
        OLLAMA_HOST="http://localhost:11434",
        OLLAMA_MODEL="default-fallback",
    )
    fake_ai_config = MagicMock()
    fake_ai_config.get_user_ai_config = AsyncMock(return_value={"some": "config"})
    fake_ai_config.get_user_provider_secret = AsyncMock(
        return_value=("ollama", "http://192.168.1.42:11434", "qwen2.5")
    )

    with (
        patch.dict("sys.modules", {"ollama": fake_module}),
        patch("graphora_server.config.get_settings", return_value=fake_settings),
        patch(
            "graphora_server.utils.llm_helper.AIConfigService",
            return_value=fake_ai_config,
        ),
    ):
        _client, model_name, provider = await llm_helper.get_llm_client_for_user(
            "user-1"
        )

    assert provider == "ollama"
    assert model_name == "qwen2.5"
    # Host from the stored config wins, not the env default.
    fake_module.Client.assert_called_once_with(host="http://192.168.1.42:11434")


@pytest.mark.asyncio
async def test_no_user_config_raises() -> None:
    fake_settings = MagicMock(
        LLM_PROVIDER=None,
        OLLAMA_HOST="x",
        OLLAMA_MODEL="x",
    )
    fake_ai_config = MagicMock()
    fake_ai_config.get_user_ai_config = AsyncMock(return_value=None)

    with (
        patch("graphora_server.config.get_settings", return_value=fake_settings),
        patch(
            "graphora_server.utils.llm_helper.AIConfigService",
            return_value=fake_ai_config,
        ),
    ):
        with pytest.raises(NoAIConfigurationError):
            await llm_helper.get_llm_client_for_user("user-1")


@pytest.mark.asyncio
async def test_unsupported_provider_raises() -> None:
    fake_settings = MagicMock(LLM_PROVIDER=None, OLLAMA_HOST="x", OLLAMA_MODEL="x")
    fake_ai_config = MagicMock()
    fake_ai_config.get_user_ai_config = AsyncMock(return_value={"x": 1})
    fake_ai_config.get_user_provider_secret = AsyncMock(
        return_value=("anthropic", "k", "claude-3")
    )

    with (
        patch("graphora_server.config.get_settings", return_value=fake_settings),
        patch(
            "graphora_server.utils.llm_helper.AIConfigService",
            return_value=fake_ai_config,
        ),
    ):
        with pytest.raises(UnsupportedProviderError):
            await llm_helper.get_llm_client_for_user("user-1")


# ---- create_baml_client_registry routing ---------------------------------


class TestBamlClientRegistry:
    def test_gemini_path_keeps_existing_shape(self) -> None:
        """Backward-compat: existing Gemini callers don't pass provider."""
        registry = llm_helper.create_baml_client_registry(
            api_key="k", model_name="gemini-2.5-flash"
        )
        # We can't introspect baml_py's internal state easily; just
        # confirm no exception was raised and a registry is returned.
        assert registry is not None

    def test_ollama_path_requires_base_url(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            llm_helper.create_baml_client_registry(
                api_key="",
                model_name="llama3.2",
                provider="ollama",
            )

    def test_ollama_path_succeeds_with_base_url(self) -> None:
        registry = llm_helper.create_baml_client_registry(
            api_key="",
            model_name="llama3.2",
            provider="ollama",
            base_url="http://localhost:11434",
        )
        assert registry is not None

    def test_unsupported_provider_raises(self) -> None:
        with pytest.raises(UnsupportedProviderError):
            llm_helper.create_baml_client_registry(
                api_key="k",
                model_name="x",
                provider="claude",
            )


# ---- get_baml_registry_for_user routing ----------------------------------


class TestBamlRegistryForUser:
    @pytest.mark.asyncio
    async def test_env_var_ollama_skips_db(self) -> None:
        fake_settings = MagicMock(
            LLM_PROVIDER="ollama",
            OLLAMA_HOST="http://localhost:11434",
            OLLAMA_MODEL="llama3.2",
        )
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch("graphora_server.utils.llm_helper.AIConfigService") as ai_cls,
        ):
            registry, model, provider = await llm_helper.get_baml_registry_for_user(
                "u1"
            )
        assert provider == "ollama"
        assert model == "llama3.2"
        assert registry is not None
        ai_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_backed_gemini(self) -> None:
        fake_settings = MagicMock(LLM_PROVIDER=None, OLLAMA_HOST="x", OLLAMA_MODEL="x")
        fake_ai = MagicMock()
        fake_ai.get_user_ai_config = AsyncMock(return_value={"x": 1})
        fake_ai.get_user_provider_secret = AsyncMock(
            return_value=("gemini", "real-key", "gemini-2.5-flash")
        )
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.utils.llm_helper.AIConfigService", return_value=fake_ai
            ),
        ):
            registry, model, provider = await llm_helper.get_baml_registry_for_user(
                "u1"
            )
        assert provider == "gemini"
        assert model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_db_backed_ollama_uses_api_key_as_host(self) -> None:
        fake_settings = MagicMock(
            LLM_PROVIDER=None,
            OLLAMA_HOST="http://default-host:11434",
            OLLAMA_MODEL="x",
        )
        fake_ai = MagicMock()
        fake_ai.get_user_ai_config = AsyncMock(return_value={"x": 1})
        fake_ai.get_user_provider_secret = AsyncMock(
            return_value=("ollama", "http://stored-host:11434", "qwen2.5")
        )
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.utils.llm_helper.AIConfigService", return_value=fake_ai
            ),
        ):
            _registry, model, provider = await llm_helper.get_baml_registry_for_user(
                "u1"
            )
        assert provider == "ollama"
        assert model == "qwen2.5"

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self) -> None:
        fake_settings = MagicMock(LLM_PROVIDER=None, OLLAMA_HOST="x", OLLAMA_MODEL="x")
        fake_ai = MagicMock()
        fake_ai.get_user_ai_config = AsyncMock(return_value={"x": 1})
        fake_ai.get_user_provider_secret = AsyncMock(
            return_value=("anthropic", "k", "claude")
        )
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.utils.llm_helper.AIConfigService", return_value=fake_ai
            ),
        ):
            with pytest.raises(UnsupportedProviderError):
                await llm_helper.get_baml_registry_for_user("u1")

    # ---- B5-obs slice 3: model_override routing ------------------------

    @pytest.mark.asyncio
    async def test_model_override_supersedes_user_stored_model_gemini(self) -> None:
        """B5-obs slice 3: when ``model_override`` is supplied, the
        registry is built with the override model_name instead of
        the user's stored one. Provider + auth stay intact — we
        don't want refinement-pass routing to accidentally swap
        provider or leak a different user's API key."""
        fake_settings = MagicMock(LLM_PROVIDER=None, OLLAMA_HOST="x", OLLAMA_MODEL="x")
        fake_ai = MagicMock()
        fake_ai.get_user_ai_config = AsyncMock(return_value={"x": 1})
        fake_ai.get_user_provider_secret = AsyncMock(
            return_value=("gemini", "real-key", "gemini-1.5-flash"),
        )
        captured: dict = {}

        def capture_registry(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.utils.llm_helper.AIConfigService", return_value=fake_ai
            ),
            patch(
                "graphora_server.utils.llm_helper.create_baml_client_registry",
                side_effect=capture_registry,
            ),
        ):
            _registry, model, provider = await llm_helper.get_baml_registry_for_user(
                "u1", model_override="gemini-2.5-pro"
            )

        assert provider == "gemini"
        # Returned model_name reflects the override — callers
        # logging "which model produced this fact" see the truth.
        assert model == "gemini-2.5-pro"
        # Registry was built with the override, not the stored model.
        assert captured["model_name"] == "gemini-2.5-pro"
        # Provider + auth survived.
        assert captured["provider"] == "gemini"
        assert captured["api_key"] == "real-key"

    @pytest.mark.asyncio
    async def test_model_override_none_preserves_user_model(self) -> None:
        """Pre-slice-3 behavior: when override is None, the user's
        stored model_name flows through unchanged. Pin so a refactor
        that accidentally always overrides surfaces here."""
        fake_settings = MagicMock(LLM_PROVIDER=None, OLLAMA_HOST="x", OLLAMA_MODEL="x")
        fake_ai = MagicMock()
        fake_ai.get_user_ai_config = AsyncMock(return_value={"x": 1})
        fake_ai.get_user_provider_secret = AsyncMock(
            return_value=("gemini", "k", "gemini-2.5-flash"),
        )
        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.utils.llm_helper.AIConfigService", return_value=fake_ai
            ),
        ):
            _registry, model, provider = await llm_helper.get_baml_registry_for_user(
                "u1"
            )
        assert provider == "gemini"
        assert model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_model_override_applies_on_ollama_env_path_too(self) -> None:
        """The env-var Ollama fast path (LLM_PROVIDER=ollama) skips
        the DB lookup entirely. Override must still apply — that's
        the whole point of routing: a deployment running Ollama for
        primary extraction can route refinement to a beefier model
        WITHOUT touching the user-config flow."""
        fake_settings = MagicMock(
            LLM_PROVIDER="ollama",
            OLLAMA_HOST="http://localhost:11434",
            OLLAMA_MODEL="llama3.2",
        )
        captured: dict = {}

        def capture_registry(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("graphora_server.config.get_settings", return_value=fake_settings),
            patch(
                "graphora_server.utils.llm_helper.create_baml_client_registry",
                side_effect=capture_registry,
            ),
        ):
            _registry, model, _provider = await llm_helper.get_baml_registry_for_user(
                "u1", model_override="qwen2.5:14b"
            )
        assert model == "qwen2.5:14b"
        assert captured["model_name"] == "qwen2.5:14b"
        # Host / provider stays — only the model_name changed.
        assert captured["provider"] == "ollama"
        assert captured["base_url"] == "http://localhost:11434"
