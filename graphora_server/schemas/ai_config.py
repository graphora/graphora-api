from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime


class AIProvider(BaseModel):
    """Schema for AI provider information"""

    id: Optional[str] = None
    name: str = Field(..., description="Provider identifier (e.g., 'gemini', 'openai')")
    display_name: str = Field(..., description="Human-readable provider name")
    is_active: bool = Field(default=True, description="Whether the provider is active")


class AIModel(BaseModel):
    """Schema for AI model information"""

    id: Optional[str] = None
    provider_id: str = Field(..., description="Reference to AI provider")
    name: str = Field(..., description="Model identifier (e.g., 'gemini-2.0-flash')")
    display_name: str = Field(..., description="Human-readable model name")
    version: Optional[str] = Field(
        None, description="Model version (e.g., '001', 'latest')"
    )
    is_active: bool = Field(default=True, description="Whether the model is active")


class AIProviderConfig(BaseModel):
    """Schema for AI provider configuration"""

    id: Optional[str] = None
    provider_id: str = Field(..., description="Reference to AI provider")
    api_key: str = Field(..., description="Encrypted API key")
    default_model_id: Optional[str] = Field(None, description="Default model to use")
    config_data: Dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific settings"
    )

    # Related data for convenience
    provider: Optional[AIProvider] = None
    default_model: Optional[AIModel] = None


class UserAIConfig(BaseModel):
    """Schema for user AI configuration"""

    id: Optional[str] = None
    user_id: str = Field(..., description="User identifier")
    active_provider_config_id: str = Field(
        ..., description="Active AI provider configuration"
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Related data for convenience
    active_provider_config: Optional[AIProviderConfig] = None


class GeminiConfigRequest(BaseModel):
    """Schema for creating/updating Gemini configuration.

    Kept for backward compatibility with the legacy
    ``/ai-config/gemini`` endpoints. New callers should use
    ``ProviderConfigRequest`` against ``/ai-config/{provider}``.
    """

    api_key: str = Field(..., description="Gemini API key")
    default_model_name: str = Field(
        ..., description="Default model name (e.g., 'gemini-2.0-flash')"
    )

    @validator("api_key")
    def validate_api_key(cls, v):
        """Validate API key format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("API key cannot be empty")
        if len(v) < 10:  # Basic sanity check
            raise ValueError("API key appears to be too short")
        return v.strip()

    @validator("default_model_name")
    def validate_model_name(cls, v):
        """Validate model name is not empty"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()


class ProviderConfigRequest(BaseModel):
    """Generic provider-config payload — used by ``/ai-config/{provider}``.

    Covers all supported providers (gemini, openai, anthropic, ollama).
    Provider-specific extras land in ``config_data``:

    - ``base_url``: Ollama server URL, or OpenAI custom endpoint
      (Azure / OpenRouter). Ignored for gemini and anthropic.
    """

    api_key: str = Field(
        ...,
        description=(
            "Provider API key. For Ollama with no-auth servers, pass any "
            "non-empty placeholder — the value is stored encrypted but "
            "not used to authenticate the request."
        ),
    )
    default_model_name: str = Field(
        ...,
        description=(
            "Default model identifier. Must be a valid model name for the "
            "provider (e.g., 'gpt-4o-mini', 'claude-sonnet-4-6', "
            "'llama3.3:70b'). If the name is not in the curated catalog, "
            "it is auto-registered as a custom model so the UI dropdown "
            "shows it next time."
        ),
    )
    base_url: Optional[str] = Field(
        None,
        description=(
            "Provider endpoint URL. Required for Ollama (e.g., "
            "'http://localhost:11434'), optional for OpenAI (custom "
            "Azure / OpenRouter endpoints). Ignored for gemini and "
            "anthropic."
        ),
    )

    @validator("api_key")
    def validate_api_key(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("API key cannot be empty")
        return v.strip()

    @validator("default_model_name")
    def validate_model_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()


class UserAIConfigDisplay(BaseModel):
    """Schema for displaying user AI config (with masked API key)"""

    id: Optional[str] = None
    user_id: str
    provider_name: str = Field(..., description="Provider name (e.g., 'gemini')")
    provider_display_name: str = Field(..., description="Human-readable provider name")
    api_key_masked: str = Field(..., description="Masked API key (e.g., 'AIza****')")
    default_model_name: str = Field(..., description="Default model name")
    default_model_display_name: str = Field(
        ..., description="Human-readable model name"
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
