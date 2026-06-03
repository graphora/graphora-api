from typing import Any, Optional, Tuple
from google import genai
import baml_py
from graphora_server.services.ai_config_service import AIConfigService
from graphora_server.exceptions import (
    NoAIConfigurationError,
    InvalidAPIKeyError,
    UnsupportedProviderError,
)


async def get_user_llm_credentials(user_id: str) -> Tuple[str, str]:
    """
    Get user's LLM credentials from database

    Args:
        user_id: User's ID

    Returns:
        Tuple of (api_key, model_name)

    Raises:
        NoAIConfigurationError: If user has no LLM configuration
        InvalidAPIKeyError: If API key retrieval fails
        UnsupportedProviderError: If provider is not supported
    """
    ai_config_service = AIConfigService()
    user_config = await ai_config_service.get_user_ai_config(user_id)

    if not user_config:
        raise NoAIConfigurationError(user_id)

    provider_name, api_key, model_name = (
        await ai_config_service.get_user_provider_secret(user_id)
    )
    if not provider_name:
        raise InvalidAPIKeyError(provider="Unknown", user_id=user_id)

    if provider_name != "gemini":
        raise UnsupportedProviderError(provider_name)

    return api_key, model_name


def create_gemini_client(api_key: str) -> genai.Client:
    """
    Create a Gemini client with the provided API key

    Args:
        api_key: Gemini API key

    Returns:
        genai.Client: Configured Gemini client
    """
    return genai.Client(api_key=api_key)


def create_baml_client_registry(
    api_key: str,
    model_name: str,
    provider: str = "gemini",
    *,
    base_url: Optional[str] = None,
) -> baml_py.ClientRegistry:
    """Create a BAML ClientRegistry for the requested provider.

    Args:
        api_key: Provider API key (ignored for Ollama).
        model_name: Model name to drive the client with.
        provider: ``"gemini"`` (default, multimodal — sends PDFs natively)
            or ``"ollama"`` (text-only, requires ``base_url``).
        base_url: Ollama server URL. Required when provider="ollama".

    Returns:
        baml_py.ClientRegistry: Registry with the matching primary client.

    Raises:
        UnsupportedProviderError: If provider is anything other than the
            two supported strings.
        ValueError: If provider="ollama" without base_url.
    """
    client_registry = baml_py.ClientRegistry()

    if provider == "gemini":
        client_registry.add_llm_client(
            name="DynamicGemini",
            provider="google-ai",
            options={
                "model": model_name,
                "api_key": api_key,
                "generationConfig": {
                    "temperature": 0.0,
                    "topP": 0.0,
                    "topK": 1,
                    "candidateCount": 1,
                },
            },
        )
        client_registry.set_primary("DynamicGemini")
    elif provider == "ollama":
        if not base_url:
            raise ValueError("base_url is required when provider='ollama'")
        # BAML's openai-generic provider handles Ollama's OpenAI-compatible
        # /v1/chat/completions endpoint without needing a separate driver.
        client_registry.add_llm_client(
            name="DynamicOllama",
            provider="openai-generic",
            options={
                "base_url": base_url.rstrip("/") + "/v1",
                "model": model_name,
                "api_key": "ollama",  # Required by spec, ignored by Ollama.
                "default_options": {
                    "temperature": 0.0,
                },
            },
        )
        client_registry.set_primary("DynamicOllama")
    elif provider == "openai":
        # ``base_url`` lets users point at Azure OpenAI, OpenRouter, or
        # other OpenAI-compatible endpoints. None falls back to OpenAI's
        # canonical https://api.openai.com.
        options: dict = {
            "model": model_name,
            "api_key": api_key,
            "default_options": {"temperature": 0.0},
        }
        if base_url:
            options["base_url"] = base_url.rstrip("/")
        client_registry.add_llm_client(
            name="DynamicOpenAI",
            provider="openai",
            options=options,
        )
        client_registry.set_primary("DynamicOpenAI")
    elif provider == "anthropic":
        client_registry.add_llm_client(
            name="DynamicAnthropic",
            provider="anthropic",
            options={
                "model": model_name,
                "api_key": api_key,
                "default_options": {"temperature": 0.0},
            },
        )
        client_registry.set_primary("DynamicAnthropic")
    else:
        raise UnsupportedProviderError(provider)

    return client_registry


# ---- Provider abstraction (Ollama support) --------------------------------


class _OllamaGenAICompat:
    """Adapter exposing the genai.Client.models.generate_content shape.

    schema_inference.py and schema_postprocess.py speak in the Gemini
    SDK's idiom (``client.models.generate_content(model=..., contents=[...],
    config=...)``). Wrapping the ollama-python SDK in the same shape
    means the inference services don't have to know which provider
    they're talking to.

    Intentionally minimal: only the methods the inference services
    actually call are stubbed. Adding more provider-aware code paths
    means adding methods here, not branching at every callsite.
    """

    def __init__(self, host: str, default_model: str):
        try:
            import ollama  # type: ignore
        except ImportError as exc:  # pragma: no cover — exercised without [ollama]
            raise ImportError(
                "Ollama support requires the [ollama] extra. "
                "Install with: pip install 'graphora-server[ollama]'"
            ) from exc
        self._client = ollama.Client(host=host)
        self._default_model = default_model
        # Mirror genai.Client.models attribute access.
        self.models = self

    def generate_content(
        self,
        *,
        model: Optional[str] = None,
        contents: Any,
        config: Any = None,
    ) -> Any:
        """Run a one-shot generation, return an object with ``.text``."""
        # Normalize Gemini's contents-as-list-of-strings into a single
        # prompt. The Ollama API takes a flat string for ``prompt``.
        if isinstance(contents, list):
            prompt = "\n".join(str(c) for c in contents)
        else:
            prompt = str(contents)

        options: dict = {}
        if config is not None:
            # GenerateContentConfig fields the inference services set.
            for attr in ("temperature", "top_p", "max_output_tokens"):
                value = getattr(config, attr, None)
                if value is not None:
                    options["num_predict" if attr == "max_output_tokens" else attr] = (
                        value
                    )

        response = self._client.generate(
            model=model or self._default_model,
            prompt=prompt,
            options=options or None,
            stream=False,
        )
        # Pass through Ollama's token counts so the usage tracker
        # can surface real numbers for /cost. Ollama's SDK calls
        # them prompt_eval_count / eval_count; mirror them onto a
        # ``usage_metadata``-shaped object so the same Gemini-style
        # extractor in llm_usage_tracker.set_usage_from_response
        # picks them up unchanged.
        prompt_tokens = int(response.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(response.get("eval_count", 0) or 0)
        return _OllamaResponse(
            text=response.get("response", ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class _OllamaUsageMetadata:
    """Gemini-shaped token counts so set_usage_from_response works
    on the Ollama path without provider-aware branching there."""

    __slots__ = ("prompt_token_count", "candidates_token_count", "total_token_count")

    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens
        self.total_token_count = prompt_tokens + completion_tokens


class _OllamaResponse:
    """Mimic the ``response.text`` access pattern of genai responses,
    plus a Gemini-shaped ``usage_metadata`` so the same
    set_usage_from_response extractor picks up Ollama's token counts
    without provider-aware branching at the tracker boundary."""

    __slots__ = ("text", "usage_metadata")

    def __init__(self, text: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.text = text
        self.usage_metadata = _OllamaUsageMetadata(prompt_tokens, completion_tokens)


def create_ollama_client(host: str, model: str) -> _OllamaGenAICompat:
    """Create a Gemini-shaped adapter wrapping the Ollama SDK.

    Args:
        host: Ollama server URL (e.g., ``http://localhost:11434``).
        model: Default model name (e.g., ``llama3.2``). Callers can
            still override per-request via ``generate_content(model=...)``.
    """
    return _OllamaGenAICompat(host=host, default_model=model)


async def get_baml_registry_for_user(
    user_id: str,
    *,
    model_override: Optional[str] = None,
) -> Tuple[baml_py.ClientRegistry, str, str]:
    """Resolve a BAML ClientRegistry for the active provider.

    Mirrors ``get_llm_client_for_user`` but for the BAML extraction
    pipeline. Returns ``(registry, model_name, provider_name)``.

    Provider resolution:

        1. ``LLM_PROVIDER=ollama`` env var → DynamicOllama registry
        2. User's stored provider:
           - ``gemini`` → DynamicGemini registry
           - ``ollama`` → DynamicOllama; api_key column doubles as the
             host URL, falls back to OLLAMA_HOST env when empty.
             ``config_data.base_url`` also honored if set via the
             generic ``/ai-config/{provider}`` endpoint.
           - ``openai`` → DynamicOpenAI; optional ``config_data.base_url``
             routes through Azure / OpenRouter / other compatible endpoints.
           - ``anthropic`` → DynamicAnthropic.

    B5-obs slice 3: ``model_override`` lets the multi-pass extractor
    swap the model name without re-deriving the provider or
    re-fetching the user's API key. The override is applied to the
    final registry construction, AFTER provider resolution, so the
    user's auth and provider choice stay intact — only the model
    name changes. Returns the EFFECTIVE model_name (override when
    provided, user's stored name otherwise), so callers logging
    "which model was used for this call" see the truth.
    """
    from graphora_server.config import get_settings

    settings = get_settings()

    if (settings.LLM_PROVIDER or "").lower() == "ollama":
        effective_model = model_override or settings.OLLAMA_MODEL
        registry = create_baml_client_registry(
            api_key="",
            model_name=effective_model,
            provider="ollama",
            base_url=settings.OLLAMA_HOST,
        )
        return registry, effective_model, "ollama"

    ai_config_service = AIConfigService()
    user_config = await ai_config_service.get_user_ai_config(user_id)
    if not user_config:
        raise NoAIConfigurationError(user_id)

    provider_name, api_key, model_name = (
        await ai_config_service.get_user_provider_secret(user_id)
    )
    if not provider_name:
        raise InvalidAPIKeyError(provider="Unknown", user_id=user_id)

    effective_model = model_override or model_name

    if provider_name == "gemini":
        registry = create_baml_client_registry(
            api_key=api_key, model_name=effective_model, provider="gemini"
        )
        return registry, effective_model, "gemini"
    if provider_name == "ollama":
        extras = await ai_config_service.get_user_provider_extras(user_id) or {}
        host = _resolve_ollama_host(
            extras.get("base_url"), api_key, settings.OLLAMA_HOST
        )
        registry = create_baml_client_registry(
            api_key="",
            model_name=effective_model,
            provider="ollama",
            base_url=host,
        )
        return registry, effective_model, "ollama"
    if provider_name == "openai":
        extras = await ai_config_service.get_user_provider_extras(user_id) or {}
        registry = create_baml_client_registry(
            api_key=api_key,
            model_name=effective_model,
            provider="openai",
            base_url=extras.get("base_url"),
        )
        return registry, effective_model, "openai"
    if provider_name == "anthropic":
        registry = create_baml_client_registry(
            api_key=api_key,
            model_name=effective_model,
            provider="anthropic",
        )
        return registry, effective_model, "anthropic"
    raise UnsupportedProviderError(provider_name)


async def get_llm_client_for_user(
    user_id: str,
) -> Tuple[Any, str, str]:
    """Resolve the LLM client to use for ``user_id``.

    Provider resolution order:
        1. ``LLM_PROVIDER=ollama`` env var → return Ollama client built
           from ``OLLAMA_HOST`` + ``OLLAMA_MODEL``. Skips the DB
           entirely — this is the no-key local-dev fast path.
        2. Otherwise → fall through to ``get_user_provider_secret``
           and dispatch on the provider name stored for that user.

    Returns:
        ``(client, model_name, provider_name)`` where ``client`` exposes
        ``client.models.generate_content(model=..., contents=...,
        config=...)`` regardless of provider.

    Raises:
        NoAIConfigurationError, UnsupportedProviderError as appropriate.
    """
    # Local import to avoid a top-level config dependency in this util.
    from graphora_server.config import get_settings

    settings = get_settings()

    if (settings.LLM_PROVIDER or "").lower() == "ollama":
        client = create_ollama_client(settings.OLLAMA_HOST, settings.OLLAMA_MODEL)
        return client, settings.OLLAMA_MODEL, "ollama"

    ai_config_service = AIConfigService()
    user_config = await ai_config_service.get_user_ai_config(user_id)
    if not user_config:
        raise NoAIConfigurationError(user_id)

    provider_name, api_key, model_name = (
        await ai_config_service.get_user_provider_secret(user_id)
    )
    if not provider_name:
        raise InvalidAPIKeyError(provider="Unknown", user_id=user_id)

    if provider_name == "gemini":
        return create_gemini_client(api_key), model_name, "gemini"
    if provider_name == "ollama":
        # DB-backed Ollama config: legacy schema stored the host in the
        # api_key column. The multi-provider refactor moved that to
        # config_data.base_url — but legacy rows may still carry a URL
        # in api_key. Use the shared resolver so non-URL placeholders
        # (e.g., "ollama") don't get routed to as if they were hosts.
        extras = await ai_config_service.get_user_provider_extras(user_id) or {}
        host = _resolve_ollama_host(
            extras.get("base_url"), api_key, settings.OLLAMA_HOST
        )
        return create_ollama_client(host, model_name), model_name, "ollama"
    raise UnsupportedProviderError(provider_name)


def _resolve_ollama_host(
    stored_base_url: Optional[str],
    api_key: str,
    env_default: str,
) -> str:
    """Pick the Ollama server URL for a user.

    Priority (first non-empty wins):
      1. ``config_data.base_url`` — the canonical home set via the
         multi-provider ``/ai-config/{provider}`` endpoint.
      2. ``api_key`` column — only when it looks like a URL.
         Backward-compat for legacy Ollama rows that pre-date the
         ``base_url`` field, when the api_key column doubled as the
         host. Without the URL-shape guard, placeholder values like
         ``"ollama"`` (which the UI prompts users to enter when their
         Ollama server has no auth) would silently route requests to
         the literal string ``ollama`` instead of localhost.
      3. ``OLLAMA_HOST`` env — the local-dev default.

    Fix for PR #24 review High: blank-endpoint UX promise ("defaults
    to http://localhost:11434") was broken by the unconditional
    ``api_key`` fallback. With the URL-shape check, blank base_url +
    placeholder api_key now lands on the env default as advertised.

    PR #26-#27 self-review followups:

    - Whitespace-only ``stored_base_url`` is treated as blank (defense
      in depth — the FE trims, but a stray space here would otherwise
      pass through and yield a malformed URL like ``"  /v1"`` from
      BAML's downstream concatenation).
    - URL-shape check is case-insensitive per RFC 3986 (schemes are
      case-insensitive), so ``"HTTP://my-server:11434"`` matches.
    """
    if stored_base_url and stored_base_url.strip():
        return stored_base_url.strip()
    if api_key and api_key.lower().startswith(("http://", "https://")):
        return api_key
    return env_default
