from typing import Tuple
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
    api_key: str, model_name: str
) -> baml_py.ClientRegistry:
    """
    Create a BAML ClientRegistry with dynamic Gemini client

    Args:
        api_key: Gemini API key
        model_name: Model name to use

    Returns:
        baml_py.ClientRegistry: Configured client registry
    """
    client_registry = baml_py.ClientRegistry()

    # Add Gemini client to registry
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

    # Set as primary client
    client_registry.set_primary("DynamicGemini")

    return client_registry
