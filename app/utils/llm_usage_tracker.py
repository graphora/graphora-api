"""
Utility decorator and context manager for tracking LLM usage.

This module provides convenient ways to automatically track LLM calls
across the application, including Gemini API and BAML library usage.
"""

import functools
from datetime import datetime, timezone
from typing import Optional, Any, Callable, TypeVar, ParamSpec
from contextlib import asynccontextmanager
from app.services.usage_tracking import usage_tracking_service
from app.schemas.usage import LLMUsageRequest, ModelProvider
from app.utils.logger import logger

P = ParamSpec("P")
T = TypeVar("T")


class LLMUsageTracker:
    """Context manager and decorator for tracking LLM usage"""

    def __init__(
        self,
        user_id: str,
        model_provider: ModelProvider,
        model_name: str,
        operation_type: str,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
        operation_context: Optional[str] = None,
    ):
        self.user_id = user_id
        self.model_provider = model_provider
        self.model_name = model_name
        self.operation_type = operation_type
        self.transform_id = transform_id
        self.document_usage_id = document_usage_id
        self.operation_context = operation_context

        # Tracking state
        self.request_timestamp = None
        self.response_timestamp = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.success = True
        self.error_message = None

    async def __aenter__(self):
        """Start tracking LLM usage"""
        self.request_timestamp = datetime.now(timezone.utc)
        logger.debug(
            f"Started tracking LLM usage: {self.model_provider}:{self.model_name}"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, _exc_tb):
        """End tracking and record usage"""
        self.response_timestamp = datetime.now(timezone.utc)

        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val) if exc_val else "Unknown error"

        try:
            await self._record_usage()
        except Exception as e:
            logger.error(f"Failed to record LLM usage: {str(e)}")
            # Don't raise exception to avoid masking original error

    def set_token_usage(self, input_tokens: int, output_tokens: int):
        """Set token usage for this operation"""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def set_usage_from_response(self, response: Any):
        """Extract token usage from API response object"""
        # Handle Gemini response
        if hasattr(response, "usage_metadata"):
            metadata = response.usage_metadata
            if hasattr(metadata, "prompt_token_count"):
                self.input_tokens = metadata.prompt_token_count
            if hasattr(metadata, "candidates_token_count"):
                self.output_tokens = metadata.candidates_token_count
            if hasattr(metadata, "total_token_count"):
                # Use total if individual counts not available
                if self.input_tokens == 0 and self.output_tokens == 0:
                    # Estimate split (typically ~20% output)
                    total = metadata.total_token_count
                    self.output_tokens = int(total * 0.2)
                    self.input_tokens = total - self.output_tokens

        # Handle OpenAI response
        elif hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                self.input_tokens = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                self.output_tokens = usage.completion_tokens

        # Handle Anthropic response
        elif hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "input_tokens"):
                self.input_tokens = usage.input_tokens
            if hasattr(usage, "output_tokens"):
                self.output_tokens = usage.output_tokens

    async def _record_usage(self):
        """Record the usage data"""
        latency_ms = None
        if self.request_timestamp and self.response_timestamp:
            latency_ms = int(
                (self.response_timestamp - self.request_timestamp).total_seconds()
                * 1000
            )

        usage_request = LLMUsageRequest(
            transform_id=self.transform_id,
            document_usage_id=self.document_usage_id,
            model_provider=self.model_provider,
            model_name=self.model_name,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            operation_type=self.operation_type,
            latency_ms=latency_ms,
        )

        await usage_tracking_service.track_llm_usage(
            user_id=self.user_id,
            request=usage_request,
            request_timestamp=self.request_timestamp,
            response_timestamp=self.response_timestamp,
            success=self.success,
            error_message=self.error_message,
        )


def track_llm_usage(
    user_id: str,
    model_provider: ModelProvider,
    model_name: str,
    operation_type: str,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    operation_context: Optional[str] = None,
    extract_tokens_from_response: bool = True,
):
    """
    Decorator to automatically track LLM usage for async functions.

    Usage:
        @track_llm_usage(
            user_id="user123",
            model_provider=ModelProvider.GEMINI,
            model_name="gemini-2.0-flash",
            operation_type="entity_extraction"
        )
        async def extract_entities(text: str) -> dict:
            # Make LLM call
            response = await llm_client.generate(text)
            return response
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with LLMUsageTracker(
                user_id=user_id,
                model_provider=model_provider,
                model_name=model_name,
                operation_type=operation_type,
                transform_id=transform_id,
                document_usage_id=document_usage_id,
                operation_context=operation_context,
            ) as tracker:
                result = await func(*args, **kwargs)

                # Try to extract token usage from result if it looks like an API response
                if extract_tokens_from_response and result is not None:
                    tracker.set_usage_from_response(result)

                return result

        return wrapper

    return decorator


@asynccontextmanager
async def track_llm_call(
    user_id: str,
    model_provider: ModelProvider,
    model_name: str,
    operation_type: str,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    operation_context: Optional[str] = None,
):
    """
    Context manager for tracking LLM usage.

    Usage:
        async with track_llm_call(
            user_id="user123",
            model_provider=ModelProvider.GEMINI,
            model_name="gemini-2.0-flash",
            operation_type="entity_extraction"
        ) as tracker:
            response = await llm_client.generate(text)
            tracker.set_usage_from_response(response)
    """
    tracker = LLMUsageTracker(
        user_id=user_id,
        model_provider=model_provider,
        model_name=model_name,
        operation_type=operation_type,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        operation_context=operation_context,
    )

    async with tracker:
        yield tracker


# Convenience functions for common LLM providers
async def track_gemini_usage(
    user_id: str,
    model_name: str,
    operation_type: str,
    response: Any,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    operation_context: Optional[str] = None,
    request_timestamp: Optional[datetime] = None,
    response_timestamp: Optional[datetime] = None,
):
    """Track Gemini API usage after a call is made"""
    tracker = LLMUsageTracker(
        user_id=user_id,
        model_provider=ModelProvider.GEMINI,
        model_name=model_name,
        operation_type=operation_type,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        operation_context=operation_context,
    )

    tracker.request_timestamp = request_timestamp or datetime.now(timezone.utc)
    tracker.response_timestamp = response_timestamp or datetime.now(timezone.utc)
    tracker.set_usage_from_response(response)

    await tracker._record_usage()


async def track_baml_usage(
    user_id: str,
    model_name: str,
    operation_type: str,
    input_tokens: int,
    output_tokens: int,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    operation_context: Optional[str] = None,
    latency_ms: Optional[int] = None,
    success: bool = True,
    error_message: Optional[str] = None,
):
    """Track BAML library usage"""
    usage_request = LLMUsageRequest(
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        model_provider=ModelProvider.BAML,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        operation_type=operation_type,
        latency_ms=latency_ms,
    )

    await usage_tracking_service.track_llm_usage(
        user_id=user_id,
        request=usage_request,
        success=success,
        error_message=error_message,
    )
