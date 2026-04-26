from typing import Optional, List
import uuid
from datetime import datetime, timezone
from psycopg.types.json import Json
from graphora_server.config import get_settings
from graphora_server.db import postgres as db
from graphora_server.schemas.ai_config import (
    AIProvider,
    AIModel,
    GeminiConfigRequest,
    UserAIConfigDisplay,
)
from graphora_server.utils.logger import logger
from graphora_server.utils.encryption import encrypt_password, decrypt_password

settings = get_settings()


class AIConfigService:
    """Service for managing AI provider configurations in Postgres."""

    def __init__(self):
        if not (settings.DATABASE_URL or settings.resolved_database_url):
            if not settings.test_mode:
                raise ValueError(
                    "DATABASE_URL must be configured for AI config service"
                )

    async def get_providers(self) -> List[AIProvider]:
        """
        Get all available AI providers

        Returns:
            List[AIProvider]: List of available providers
        """
        try:
            rows = await db.fetch(
                "SELECT id, name, display_name, is_active FROM ai_providers WHERE is_active = TRUE"
            )

            providers = [
                AIProvider(
                    id=str(row["id"]),
                    name=row["name"],
                    display_name=row["display_name"],
                    is_active=row["is_active"],
                )
                for row in rows or []
            ]

            logger.info(f"Retrieved {len(providers)} AI providers")
            return providers

        except Exception as e:
            logger.error(f"Error retrieving AI providers: {str(e)}")
            raise

    async def get_models_by_provider(self, provider_name: str) -> List[AIModel]:
        """
        Get all available models for a specific provider

        Args:
            provider_name: Name of the provider (e.g., 'gemini')

        Returns:
            List[AIModel]: List of available models for the provider
        """
        try:
            rows = await db.fetch(
                """
                SELECT m.*
                FROM ai_models m
                JOIN ai_providers p ON m.provider_id = p.id
                WHERE p.name = %s AND m.is_active = TRUE
                """,
                provider_name,
            )

            models = [self._map_model(row) for row in rows or []]

            logger.info(
                f"Retrieved {len(models)} AI models for provider {provider_name}"
            )
            return models

        except Exception as e:
            logger.error(
                f"Error retrieving AI models for provider {provider_name}: {str(e)}"
            )
            raise

    async def get_all_models(self) -> List[AIModel]:
        """Return every active AI model regardless of provider."""

        rows = await db.fetch(
            "SELECT id, provider_id, name, display_name, version, is_active FROM ai_models WHERE is_active = TRUE"
        )

        return [self._map_model(row) for row in rows or []]

    @staticmethod
    def _map_model(row: dict) -> AIModel:
        return AIModel(
            id=str(row["id"]),
            provider_id=str(row["provider_id"]),
            name=row["name"],
            display_name=row["display_name"],
            version=row["version"],
            is_active=row["is_active"],
        )

    async def get_user_ai_config(self, user_id: str) -> Optional[UserAIConfigDisplay]:
        """
        Get user's AI configuration with masked API key

        Args:
            user_id: User's ID

        Returns:
            UserAIConfigDisplay if found, None otherwise
        """
        try:
            config_data = await db.fetchrow(
                """
                SELECT u.id as user_config_id,
                       u.user_id,
                       u.created_at,
                       u.updated_at,
                       pc.id as provider_config_id,
                       pc.api_key,
                       p.name as provider_name,
                       p.display_name as provider_display_name,
                       m.name as model_name,
                       m.display_name as model_display_name
                FROM user_ai_configs u
                JOIN ai_provider_configs pc ON u.active_provider_config_id = pc.id
                JOIN ai_providers p ON pc.provider_id = p.id
                JOIN ai_models m ON pc.default_model_id = m.id
                WHERE u.user_id = %s
                LIMIT 1
                """,
                user_id,
            )

            if not config_data:
                logger.info(f"No AI configuration found for user: {user_id}")
                return None

            decrypted_key = decrypt_password(config_data["api_key"])
            masked_key = self._mask_api_key(decrypted_key)

            created_at = config_data.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

            updated_at = config_data.get("updated_at")
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

            user_config = UserAIConfigDisplay(
                id=str(config_data["user_config_id"]),
                user_id=str(config_data["user_id"]),
                provider_name=config_data["provider_name"],
                provider_display_name=config_data["provider_display_name"],
                api_key_masked=masked_key,
                default_model_name=config_data["model_name"],
                default_model_display_name=config_data["model_display_name"],
                created_at=created_at,
                updated_at=updated_at,
            )

            logger.info(f"Retrieved AI configuration for user: {user_id}")
            return user_config

        except Exception as e:
            logger.error(f"Error retrieving user AI config for {user_id}: {str(e)}")
            raise

    async def get_user_provider_secret(
        self, user_id: str
    ) -> Optional[tuple[str, str, str]]:
        """Return provider name, decrypted API key, and default model for a user."""

        row = await db.fetchrow(
            """
            SELECT p.name AS provider_name,
                   pc.api_key,
                   m.name AS model_name
            FROM user_ai_configs u
            JOIN ai_provider_configs pc ON u.active_provider_config_id = pc.id
            JOIN ai_providers p ON pc.provider_id = p.id
            JOIN ai_models m ON pc.default_model_id = m.id
            WHERE u.user_id = %s
            LIMIT 1
            """,
            user_id,
        )

        if not row:
            return None

        return (
            row["provider_name"],
            decrypt_password(row["api_key"]),
            row["model_name"],
        )

    async def create_gemini_config(
        self, user_id: str, config_request: GeminiConfigRequest
    ) -> UserAIConfigDisplay:
        """
        Create a new Gemini configuration for a user

        Args:
            user_id: User's ID
            config_request: Gemini configuration data

        Returns:
            UserAIConfigDisplay: Created configuration with masked API key
        """
        try:
            # Check if user already has a configuration
            existing_config = await self.get_user_ai_config(user_id)
            if existing_config:
                raise ValueError(f"AI configuration already exists for user: {user_id}")

            provider_row = await db.fetchrow(
                "SELECT id FROM ai_providers WHERE name = %s",
                "gemini",
            )
            if not provider_row:
                raise ValueError("Gemini provider not found")
            provider_id = provider_row["id"]

            model_row = await db.fetchrow(
                """
                SELECT id
                FROM ai_models
                WHERE provider_id = %s AND name = %s AND is_active = TRUE
                """,
                provider_id,
                config_request.default_model_name,
            )

            if not model_row:
                available_models = await db.fetch(
                    "SELECT name FROM ai_models WHERE provider_id = %s AND is_active = TRUE",
                    provider_id,
                )
                available_names = [model["name"] for model in available_models or []]
                raise ValueError(
                    f"Model '{config_request.default_model_name}' not found for Gemini. Available models: {', '.join(available_names)}"
                )
            model_id = model_row["id"]

            provider_config_id = str(uuid.uuid4())
            user_config_id = str(uuid.uuid4())
            encrypted_key = encrypt_password(config_request.api_key)

            async with db.transaction() as cur:
                await cur.execute(
                    """
                    INSERT INTO ai_provider_configs (
                        id, provider_id, api_key, default_model_id, config_data
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        provider_config_id,
                        provider_id,
                        encrypted_key,
                        model_id,
                        Json({}),
                    ),
                )

                await cur.execute(
                    """
                    INSERT INTO user_ai_configs (
                        id, user_id, active_provider_config_id
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (user_config_id, user_id, provider_config_id),
                )

            logger.info(f"Created Gemini configuration for user: {user_id}")

            # Return the created configuration
            return await self.get_user_ai_config(user_id)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error creating Gemini config for {user_id}: {str(e)}")
            raise Exception(f"Failed to create Gemini configuration: {str(e)}")

    async def update_gemini_config(
        self, user_id: str, config_request: GeminiConfigRequest
    ) -> UserAIConfigDisplay:
        """
        Update an existing Gemini configuration for a user

        Args:
            user_id: User's ID
            config_request: Updated Gemini configuration data

        Returns:
            UserAIConfigDisplay: Updated configuration with masked API key
        """
        try:
            # Get existing configuration
            existing_config = await self.get_user_ai_config(user_id)
            if not existing_config:
                # UI may call update before create (e.g., after DB reset); treat as upsert.
                return await self.create_gemini_config(user_id, config_request)

            user_config_row = await db.fetchrow(
                "SELECT active_provider_config_id FROM user_ai_configs WHERE user_id = %s",
                user_id,
            )

            if not user_config_row:
                raise ValueError(f"User AI configuration not found for user: {user_id}")

            provider_config_id = user_config_row["active_provider_config_id"]

            provider_row = await db.fetchrow(
                "SELECT id FROM ai_providers WHERE name = %s",
                "gemini",
            )
            if not provider_row:
                raise ValueError("Gemini provider not found")
            provider_id = provider_row["id"]

            model_row = await db.fetchrow(
                """
                SELECT id
                FROM ai_models
                WHERE provider_id = %s AND name = %s AND is_active = TRUE
                """,
                provider_id,
                config_request.default_model_name,
            )

            if not model_row:
                available_models = await db.fetch(
                    "SELECT name FROM ai_models WHERE provider_id = %s AND is_active = TRUE",
                    provider_id,
                )
                available_names = [model["name"] for model in available_models or []]
                raise ValueError(
                    f"Model '{config_request.default_model_name}' not found for Gemini. Available models: {', '.join(available_names)}"
                )
            model_id = model_row["id"]

            result = await db.fetchrow(
                """
                UPDATE ai_provider_configs
                SET api_key = %s,
                    default_model_id = %s,
                    updated_at = %s
                WHERE id = %s
                RETURNING id
                """,
                encrypt_password(config_request.api_key),
                model_id,
                datetime.now(timezone.utc),
                provider_config_id,
            )

            if not result:
                raise Exception("Failed to update AI provider configuration")

            logger.info(f"Updated Gemini configuration for user: {user_id}")

            # Return the updated configuration
            return await self.get_user_ai_config(user_id)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error updating Gemini config for {user_id}: {str(e)}")
            raise Exception(f"Failed to update Gemini configuration: {str(e)}")

    def _mask_api_key(self, api_key: str) -> str:
        """
        Mask an API key for display purposes

        Args:
            api_key: The full API key

        Returns:
            str: Masked API key (e.g., 'AIza****')
        """
        if not api_key:
            return "****"

        if len(api_key) <= 8:
            return "****"

        # Show first 4 characters and mask the rest
        return api_key[:4] + "****"


# Global service instance
ai_config_service = AIConfigService()
