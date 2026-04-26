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
        return _OllamaResponse(response.get("response", ""))


class _OllamaResponse:
    """Mimic the ``response.text`` access pattern of genai responses."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


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
) -> Tuple[baml_py.ClientRegistry, str, str]:
    """Resolve a BAML ClientRegistry for the active provider.

    Mirrors ``get_llm_client_for_user`` but for the BAML extraction
    pipeline. Returns ``(registry, model_name, provider_name)``.

    Provider resolution is identical to the genai path:
        1. ``LLM_PROVIDER=ollama`` env var → DynamicOllama registry
        2. User's stored provider == "gemini" → DynamicGemini registry
        3. User's stored provider == "ollama" → DynamicOllama registry,
           api_key column doubles as the host URL (empty falls back
           to OLLAMA_HOST env)
    """
    from graphora_server.config import get_settings

    settings = get_settings()

    if (settings.LLM_PROVIDER or "").lower() == "ollama":
        registry = create_baml_client_registry(
            api_key="",
            model_name=settings.OLLAMA_MODEL,
            provider="ollama",
            base_url=settings.OLLAMA_HOST,
        )
        return registry, settings.OLLAMA_MODEL, "ollama"

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
        registry = create_baml_client_registry(
            api_key=api_key, model_name=model_name, provider="gemini"
        )
        return registry, model_name, "gemini"
    if provider_name == "ollama":
        host = api_key or settings.OLLAMA_HOST
        registry = create_baml_client_registry(
            api_key="",
            model_name=model_name,
            provider="ollama",
            base_url=host,
        )
        return registry, model_name, "ollama"
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
        # DB-backed Ollama config: the api_key column doubles as the
        # Ollama host URL (no schema migration). Empty value falls back
        # to the env default for local-dev convenience.
        host = api_key or settings.OLLAMA_HOST
        return create_ollama_client(host, model_name), model_name, "ollama"
    raise UnsupportedProviderError(provider_name)
