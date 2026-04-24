"""Custom exceptions for AI configuration errors."""

from typing import List, Optional


class AIConfigurationError(Exception):
    """Base exception for AI configuration errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        user_action: str,
        supported_providers: Optional[List[str]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.user_action = user_action
        self.supported_providers = supported_providers or [
            "Google Gemini",
            "OpenAI",
            "Anthropic",
            "Mistral",
        ]
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert exception to a user-friendly dictionary."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "user_action": self.user_action,
            "supported_providers": self.supported_providers,
            "help_url": "/settings/ai-configuration",
        }


class NoAIConfigurationError(AIConfigurationError):
    """Raised when user has not configured any AI provider."""

    def __init__(self, user_id: str):
        super().__init__(
            message="AI provider not configured. Entity extraction requires an AI provider to analyze your documents.",
            error_code="AI_NOT_CONFIGURED",
            user_action="Go to Settings → AI Configuration to add your API key for Google Gemini, OpenAI, or another supported provider.",
        )
        self.user_id = user_id


class InvalidAPIKeyError(AIConfigurationError):
    """Raised when the API key is invalid or has expired."""

    def __init__(self, provider: str, user_id: str):
        super().__init__(
            message=f"The {provider} API key is invalid or has expired.",
            error_code="INVALID_API_KEY",
            user_action=f"Go to Settings → AI Configuration to update your {provider} API key.",
        )
        self.provider = provider
        self.user_id = user_id


class UnsupportedProviderError(AIConfigurationError):
    """Raised when an unsupported AI provider is configured."""

    def __init__(self, provider: str):
        super().__init__(
            message=f"The AI provider '{provider}' is not currently supported.",
            error_code="UNSUPPORTED_PROVIDER",
            user_action="Please configure a supported AI provider: Google Gemini (recommended), OpenAI, Anthropic, or Mistral.",
        )
        self.provider = provider


class AIQuotaExceededError(AIConfigurationError):
    """Raised when the AI provider quota has been exceeded."""

    def __init__(self, provider: str):
        super().__init__(
            message=f"Your {provider} API quota has been exceeded.",
            error_code="QUOTA_EXCEEDED",
            user_action=f"Check your {provider} dashboard for quota limits, or wait for your quota to reset.",
        )
        self.provider = provider


class AIRateLimitError(AIConfigurationError):
    """Raised when rate limited by the AI provider."""

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        retry_msg = f" Please retry after {retry_after} seconds." if retry_after else ""
        super().__init__(
            message=f"Rate limited by {provider}.{retry_msg}",
            error_code="RATE_LIMITED",
            user_action="Wait a moment and try again, or upgrade your API plan for higher rate limits.",
        )
        self.provider = provider
        self.retry_after = retry_after
