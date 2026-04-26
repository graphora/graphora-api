"""BAML Helper Utilities for Schema Generation"""

import logging
import baml_py

logger = logging.getLogger(__name__)


def create_baml_client(
    api_key: str, model_name: str = "gemini-2.0-flash"
) -> baml_py.ClientRegistry:
    """
    Create and configure a BAML client for user-specific credentials

    Args:
        api_key: User's Google AI API key
        model_name: The Gemini model to use

    Returns:
        baml_py.ClientRegistry: Configured client registry for BAML function calls
    """
    try:
        # Create a new client registry
        client_registry = baml_py.ClientRegistry()

        # Add Gemini client to registry
        client_registry.add_llm_client(
            name="DynamicGemini",
            provider="google-ai",
            options={
                "model": model_name,
                "api_key": api_key,
                "generationConfig": {
                    "temperature": 0.1,  # Low temperature for more deterministic output
                    "topP": 0.9,
                    "topK": 40,
                    "maxOutputTokens": 8192,
                },
            },
        )

        # Set as primary client
        client_registry.set_primary("DynamicGemini")

        logger.info(f"Created BAML client registry with model {model_name}")
        return client_registry

    except Exception as e:
        logger.error(f"Error creating BAML client: {str(e)}")
        raise


def configure_baml_client_for_refinement(
    api_key: str, model_name: str = "gemini-2.0-flash"
) -> baml_py.ClientRegistry:
    """
    Configure BAML client specifically for schema refinement with higher creativity

    Args:
        api_key: User's Google AI API key
        model_name: The Gemini model to use

    Returns:
        baml_py.ClientRegistry: Configured client registry for BAML function calls
    """
    try:
        # Create a new client registry for refinement
        client_registry = baml_py.ClientRegistry()

        # Add Gemini client to registry with higher temperature for creativity
        client_registry.add_llm_client(
            name="DynamicGemini",
            provider="google-ai",
            options={
                "model": model_name,
                "api_key": api_key,
                "generationConfig": {
                    "temperature": 0.3,  # Slightly higher for more creative refinements
                    "topP": 0.9,
                    "topK": 40,
                    "maxOutputTokens": 8192,
                },
            },
        )

        # Set as primary client
        client_registry.set_primary("DynamicGemini")

        logger.info(f"Created BAML refinement client registry with model {model_name}")
        return client_registry

    except Exception as e:
        logger.error(f"Error creating BAML refinement client: {str(e)}")
        raise


def reset_baml_client():
    """Reset BAML client to default configuration"""
    try:
        baml_py.ClientRegistry.reset()
        logger.info("Reset BAML client configuration")
    except Exception as e:
        logger.warning(f"Error resetting BAML client: {str(e)}")
