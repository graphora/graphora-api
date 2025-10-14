from fastapi import APIRouter, Depends, HTTPException
from typing import List
import traceback
from app.config import get_settings
from app.schemas.ai_config import (
    AIProvider,
    AIModel,
    GeminiConfigRequest,
    UserAIConfigDisplay,
)
from app.services.ai_config_service import ai_config_service
from app.utils.logger import logger
from app.auth import get_current_user_id

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
