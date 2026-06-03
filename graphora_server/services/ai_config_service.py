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
    ProviderConfigRequest,
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

    async def get_user_provider_extras(self, user_id: str) -> Optional[dict]:
        """Return provider-specific extras (``config_data`` JSON) for a user.

        Companion to ``get_user_provider_secret`` — kept separate so the
        existing 3-tuple unpacking in callers doesn't break. Returns
        ``None`` when the user has no config, ``{}`` when they have one
        but no extras were set.

        Currently the only well-known extra is ``base_url`` (used by
        Ollama and optionally by OpenAI for Azure / custom endpoints).
        """
        row = await db.fetchrow(
            """
            SELECT pc.config_data
            FROM user_ai_configs u
            JOIN ai_provider_configs pc ON u.active_provider_config_id = pc.id
            WHERE u.user_id = %s
            LIMIT 1
            """,
            user_id,
        )
        if not row:
            return None
        return dict(row["config_data"] or {})

    # ─── Internal helpers (shared by generic + legacy paths) ─────────

    async def _resolve_provider_id(self, provider_name: str) -> str:
        """Look up provider_id by name. Raises ValueError if unknown."""
        row = await db.fetchrow(
            "SELECT id FROM ai_providers WHERE name = %s AND is_active = TRUE",
            provider_name,
        )
        if not row:
            available = await db.fetch(
                "SELECT name FROM ai_providers WHERE is_active = TRUE"
            )
            names = ", ".join(r["name"] for r in available or [])
            raise ValueError(
                f"Provider '{provider_name}' not found. Available: {names}"
            )
        return row["id"]

    async def _resolve_or_create_model_id(
        self, provider_id: str, model_name: str
    ) -> str:
        """Return model_id for (provider_id, model_name).

        If the model is in the curated catalog, returns its id. If not,
        auto-registers it as ``version='custom'`` and returns the new id
        — supports the UI's free-form model input without forcing users
        to wait for catalog updates.
        """
        row = await db.fetchrow(
            "SELECT id FROM ai_models WHERE provider_id = %s AND name = %s",
            provider_id,
            model_name,
        )
        if row:
            return row["id"]

        new_id = str(uuid.uuid4())
        await db.fetchrow(
            """
            INSERT INTO ai_models (id, provider_id, name, display_name, version, is_active)
            VALUES (%s, %s, %s, %s, 'custom', TRUE)
            ON CONFLICT (provider_id, name) DO UPDATE SET is_active = TRUE
            RETURNING id
            """,
            new_id,
            provider_id,
            model_name,
            model_name,  # display_name = name for custom entries
        )
        logger.info(
            f"Auto-registered custom model '{model_name}' "
            f"under provider_id {provider_id}"
        )
        return new_id

    # ─── Generic create / update — used by /ai-config/{provider} ────

    async def create_provider_config(
        self,
        user_id: str,
        provider_name: str,
        config_request: ProviderConfigRequest,
    ) -> UserAIConfigDisplay:
        """Create the user's first AI provider config.

        Raises ValueError if the user already has a config — callers
        should detect the existing config first and route to
        ``update_provider_config`` (which handles upsert + provider
        switching).
        """
        try:
            existing_config = await self.get_user_ai_config(user_id)
            if existing_config:
                raise ValueError(f"AI configuration already exists for user: {user_id}")

            provider_id = await self._resolve_provider_id(provider_name)
            model_id = await self._resolve_or_create_model_id(
                provider_id, config_request.default_model_name
            )

            provider_config_id = str(uuid.uuid4())
            user_config_id = str(uuid.uuid4())
            encrypted_key = encrypt_password(config_request.api_key)
            config_data = self._build_config_data(config_request)

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
                        Json(config_data),
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

            logger.info(f"Created {provider_name} configuration for user: {user_id}")
            return await self.get_user_ai_config(user_id)

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                f"Error creating {provider_name} config for {user_id}: {str(e)}"
            )
            raise Exception(f"Failed to create {provider_name} configuration: {str(e)}")

    async def update_provider_config(
        self,
        user_id: str,
        provider_name: str,
        config_request: ProviderConfigRequest,
    ) -> UserAIConfigDisplay:
        """Update or upsert the user's AI provider config.

        Behavior:

        - No existing config → falls through to create.
        - Existing config, same provider → in-place update of the
          ``ai_provider_configs`` row.
        - Existing config, different provider → creates a new
          ``ai_provider_configs`` row and repoints
          ``user_ai_configs.active_provider_config_id``. The old row
          stays in place (kept for audit; future GC can clean up
          orphans).
        """
        try:
            existing_config = await self.get_user_ai_config(user_id)
            if not existing_config:
                return await self.create_provider_config(
                    user_id, provider_name, config_request
                )

            provider_id = await self._resolve_provider_id(provider_name)
            model_id = await self._resolve_or_create_model_id(
                provider_id, config_request.default_model_name
            )
            encrypted_key = encrypt_password(config_request.api_key)
            config_data = self._build_config_data(config_request)

            if existing_config.provider_name == provider_name:
                # In-place update of the existing provider config
                user_config_row = await db.fetchrow(
                    "SELECT active_provider_config_id FROM user_ai_configs WHERE user_id = %s",
                    user_id,
                )
                if not user_config_row:
                    raise ValueError(
                        f"User AI configuration not found for user: {user_id}"
                    )
                provider_config_id = user_config_row["active_provider_config_id"]

                result = await db.fetchrow(
                    """
                    UPDATE ai_provider_configs
                    SET api_key = %s,
                        default_model_id = %s,
                        config_data = %s,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING id
                    """,
                    encrypted_key,
                    model_id,
                    Json(config_data),
                    datetime.now(timezone.utc),
                    provider_config_id,
                )
                if not result:
                    raise Exception("Failed to update AI provider configuration")
            else:
                # Provider switch — new ai_provider_configs row, repoint
                new_provider_config_id = str(uuid.uuid4())
                async with db.transaction() as cur:
                    await cur.execute(
                        """
                        INSERT INTO ai_provider_configs (
                            id, provider_id, api_key, default_model_id, config_data
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            new_provider_config_id,
                            provider_id,
                            encrypted_key,
                            model_id,
                            Json(config_data),
                        ),
                    )
                    await cur.execute(
                        """
                        UPDATE user_ai_configs
                        SET active_provider_config_id = %s,
                            updated_at = %s
                        WHERE user_id = %s
                        """,
                        (
                            new_provider_config_id,
                            datetime.now(timezone.utc),
                            user_id,
                        ),
                    )
                logger.info(
                    f"Switched user {user_id} from "
                    f"{existing_config.provider_name} → {provider_name}"
                )

            logger.info(f"Updated {provider_name} configuration for user: {user_id}")
            return await self.get_user_ai_config(user_id)

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                f"Error updating {provider_name} config for {user_id}: {str(e)}"
            )
            raise Exception(f"Failed to update {provider_name} configuration: {str(e)}")

    @staticmethod
    def _build_config_data(config_request: ProviderConfigRequest) -> dict:
        """Extract provider-specific extras into the config_data JSON blob."""
        data: dict = {}
        if config_request.base_url:
            data["base_url"] = config_request.base_url
        return data

    # ─── Legacy Gemini-specific wrappers (kept for backward compat) ───

    async def create_gemini_config(
        self, user_id: str, config_request: GeminiConfigRequest
    ) -> UserAIConfigDisplay:
        """Backward-compat wrapper. Prefer ``create_provider_config``."""
        return await self.create_provider_config(
            user_id,
            "gemini",
            ProviderConfigRequest(
                api_key=config_request.api_key,
                default_model_name=config_request.default_model_name,
            ),
        )

    async def update_gemini_config(
        self, user_id: str, config_request: GeminiConfigRequest
    ) -> UserAIConfigDisplay:
        """Backward-compat wrapper. Prefer ``update_provider_config``."""
        return await self.update_provider_config(
            user_id,
            "gemini",
            ProviderConfigRequest(
                api_key=config_request.api_key,
                default_model_name=config_request.default_model_name,
            ),
        )

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
