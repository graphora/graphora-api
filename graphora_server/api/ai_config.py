from fastapi import APIRouter, Depends, HTTPException
from typing import List
import traceback
from graphora_server.config import get_settings
from graphora_server.schemas.ai_config import (
    AIProvider,
    AIModel,
    GeminiConfigRequest,
    ProviderConfigRequest,
    UserAIConfigDisplay,
)
from graphora_server.services.ai_config_service import ai_config_service
from graphora_server.utils.logger import logger
from graphora_server.auth import get_current_user_id

settings = get_settings()
router = APIRouter(prefix=settings.API_V1_STR, tags=["AI Configuration"])


@router.get("/ai-config", response_model=UserAIConfigDisplay)
async def get_user_ai_config(
    user_id: str = Depends(get_current_user_id),
) -> UserAIConfigDisplay:
    """
    Get user's AI configuration

    Args:
        user_id: User's ID (from header)

    Returns:
        UserAIConfigDisplay: User's AI configuration with masked API key

    Raises:
        HTTPException: 404 if configuration not found
    """
    try:
        user_config = await ai_config_service.get_user_ai_config(user_id)
        if not user_config:
            raise HTTPException(
                status_code=404,
                detail=f"AI configuration not found for user: {user_id}",
            )

        return user_config

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving AI configuration for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/ai-config/gemini", response_model=UserAIConfigDisplay)
async def create_gemini_config(
    config_request: GeminiConfigRequest, user_id: str = Depends(get_current_user_id)
) -> UserAIConfigDisplay:
    """
    Create a new Gemini configuration for a user

    Args:
        config_request: Gemini configuration data
        user_id: User's ID (from header)

    Returns:
        UserAIConfigDisplay: Created configuration with masked API key

    Raises:
        HTTPException: 400 if configuration already exists or validation fails
    """
    try:
        user_config = await ai_config_service.create_gemini_config(
            user_id, config_request
        )
        logger.info(f"Created Gemini configuration for user: {user_id}")
        return user_config

    except ValueError as e:
        logger.warning(
            f"Validation error creating Gemini config for {user_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating Gemini configuration for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/ai-config/gemini", response_model=UserAIConfigDisplay)
async def update_gemini_config(
    config_request: GeminiConfigRequest, user_id: str = Depends(get_current_user_id)
) -> UserAIConfigDisplay:
    """
    Update an existing Gemini configuration for a user

    Args:
        config_request: Updated Gemini configuration data
        user_id: User's ID (from header)

    Returns:
        UserAIConfigDisplay: Updated configuration with masked API key

    Raises:
        HTTPException: 404 if configuration not found, 400 if validation fails
    """
    try:
        user_config = await ai_config_service.update_gemini_config(
            user_id, config_request
        )
        logger.info(f"Updated Gemini configuration for user: {user_id}")
        return user_config

    except ValueError as e:
        logger.warning(
            f"Validation error updating Gemini config for {user_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating Gemini configuration for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post(
    "/ai-config/{provider}",
    response_model=UserAIConfigDisplay,
    summary="Create AI config for any supported provider",
)
async def create_provider_config(
    provider: str,
    config_request: ProviderConfigRequest,
    user_id: str = Depends(get_current_user_id),
) -> UserAIConfigDisplay:
    """Create a user's AI config for ``provider``.

    Supported providers (call `GET /ai-providers` for the live list):
    `gemini`, `openai`, `anthropic`, `ollama`.

    For Ollama / self-hosted endpoints, pass `base_url` in the request
    body (e.g., `"base_url": "http://localhost:11434"`).

    If `default_model_name` is not in the curated catalog for the
    provider, it is auto-registered as a custom model so the UI
    dropdown will surface it next time.

    Returns 409 if the user already has a configuration — use
    `PUT /ai-config/{provider}` to update / switch providers.
    """
    try:
        user_config = await ai_config_service.create_provider_config(
            user_id, provider, config_request
        )
        logger.info(f"Created {provider} configuration for user: {user_id}")
        return user_config
    except ValueError as e:
        msg = str(e)
        status = 409 if "already exists" in msg else 400
        logger.warning(
            f"Validation error creating {provider} config for {user_id}: {msg}"
        )
        raise HTTPException(status_code=status, detail=msg)
    except Exception as e:
        logger.error(f"Error creating {provider} configuration for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put(
    "/ai-config/{provider}",
    response_model=UserAIConfigDisplay,
    summary="Update or switch AI provider for the current user",
)
async def update_provider_config(
    provider: str,
    config_request: ProviderConfigRequest,
    user_id: str = Depends(get_current_user_id),
) -> UserAIConfigDisplay:
    """Update the user's AI config.

    - If the user has no config yet → creates one (upsert behavior).
    - If the existing config matches `provider` → in-place update of
      API key + model + base_url.
    - If the existing config is for a different provider → switches
      to `provider`, repointing the user's active provider config.

    See `POST /ai-config/{provider}` for `default_model_name` and
    `base_url` semantics — they're identical here.
    """
    try:
        user_config = await ai_config_service.update_provider_config(
            user_id, provider, config_request
        )
        logger.info(f"Updated {provider} configuration for user: {user_id}")
        return user_config
    except ValueError as e:
        logger.warning(
            f"Validation error updating {provider} config for {user_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating {provider} configuration for {user_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/ai-providers", response_model=List[AIProvider])
async def get_ai_providers() -> List[AIProvider]:
    """
    Get all available AI providers

    Returns:
        List[AIProvider]: List of available AI providers
    """
    try:
        providers = await ai_config_service.get_providers()
        return providers

    except Exception as e:
        logger.error(f"Error retrieving AI providers: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/ai-models/{provider_name}", response_model=List[AIModel])
async def get_ai_models(provider_name: str) -> List[AIModel]:
    """
    Get all available models for a specific AI provider

    Args:
        provider_name: Name of the AI provider (e.g., 'gemini')

    Returns:
        List[AIModel]: List of available models for the provider
    """
    try:
        models = await ai_config_service.get_models_by_provider(provider_name)
        return models

    except Exception as e:
        logger.error(
            f"Error retrieving AI models for provider {provider_name}: {str(e)}"
        )
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
