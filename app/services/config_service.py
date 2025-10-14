from typing import Optional
import uuid
from datetime import datetime, timezone
from supabase import create_client, Client
from app.config import get_settings
from app.schemas.config import UserConfig, DatabaseConfig, ConfigRequest
from app.utils.logger import logger
from app.utils.encryption import encrypt_password, decrypt_password

settings = get_settings()


class ConfigService:
    """Service for managing user configurations in Supabase"""

    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")

        self.supabase: Client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_KEY
        )

    async def get_user_config(self, user_id: str) -> Optional[UserConfig]:
        """
        Retrieve user configuration by user ID

        Args:
            user_id: User's ID

        Returns:
            UserConfig if found, None otherwise
        """
        try:
            # Query configs table with joins to database_configs
            response = (
                self.supabase.table("configs")
                .select(
                    """
                id,
                user_id,
                created_at,
                updated_at,
                staging_db:staging_db_id(id, name, uri, username, password),
                prod_db:prod_db_id(id, name, uri, username, password)
                """
                )
                .eq("user_id", user_id)
                .execute()
            )

            if not response.data:
                logger.info(f"No configuration found for user: {user_id}")
                return None

            config_data = response.data[0]

            # Convert to UserConfig schema with password decryption
            user_config = UserConfig(
                id=config_data["id"],
                userId=config_data["user_id"],
                stagingDb=DatabaseConfig(
                    id=config_data["staging_db"]["id"],
                    name=config_data["staging_db"]["name"],
                    uri=config_data["staging_db"]["uri"],
                    username=config_data["staging_db"]["username"],
                    password=decrypt_password(config_data["staging_db"]["password"]),
                ),
                prodDb=DatabaseConfig(
                    id=config_data["prod_db"]["id"],
                    name=config_data["prod_db"]["name"],
                    uri=config_data["prod_db"]["uri"],
                    username=config_data["prod_db"]["username"],
                    password=decrypt_password(config_data["prod_db"]["password"]),
                ),
                createdAt=datetime.fromisoformat(
                    config_data["created_at"].replace("Z", "+00:00")
                ),
                updatedAt=datetime.fromisoformat(
                    config_data["updated_at"].replace("Z", "+00:00")
                ),
            )

            logger.info(f"Retrieved configuration for user: {user_id}")
            return user_config

        except Exception as e:
            logger.error(f"Error retrieving user config for {user_id}: {str(e)}")
            raise

    async def create_user_config(
        self, user_id: str, config_request: ConfigRequest
    ) -> UserConfig:
        """
        Create a new user configuration

        Args:
            config_request: Configuration data

        Returns:
            Created UserConfig
        """
        try:
            # Check if user already has a configuration
            existing_config = await self.get_user_config(user_id)
            if existing_config:
                raise ValueError(f"Configuration already exists for user: {user_id}")

            # Create staging database config with encrypted password
            staging_db_id = str(uuid.uuid4())
            staging_db_response = (
                self.supabase.table("database_configs")
                .insert(
                    {
                        "id": staging_db_id,
                        "name": config_request.stagingDb.name,
                        "uri": config_request.stagingDb.uri,
                        "username": config_request.stagingDb.username,
                        "password": encrypt_password(config_request.stagingDb.password),
                    }
                )
                .execute()
            )

            if not staging_db_response.data:
                raise Exception("Failed to create staging database configuration")

            # Create production database config with encrypted password
            prod_db_id = str(uuid.uuid4())
            prod_db_response = (
                self.supabase.table("database_configs")
                .insert(
                    {
                        "id": prod_db_id,
                        "name": config_request.prodDb.name,
                        "uri": config_request.prodDb.uri,
                        "username": config_request.prodDb.username,
                        "password": encrypt_password(config_request.prodDb.password),
                    }
                )
                .execute()
            )

            if not prod_db_response.data:
                raise Exception("Failed to create production database configuration")

            # Create user config
            config_id = str(uuid.uuid4())
            config_response = (
                self.supabase.table("configs")
                .insert(
                    {
                        "id": config_id,
                        "user_id": user_id,
                        "staging_db_id": staging_db_id,
                        "prod_db_id": prod_db_id,
                    }
                )
                .execute()
            )

            if not config_response.data:
                raise Exception("Failed to create user configuration")

            # Return the created configuration
            created_config = await self.get_user_config(user_id)
            if not created_config:
                raise Exception("Failed to retrieve created configuration")

            logger.info(f"Created configuration for user: {user_id}")
            return created_config

        except Exception as e:
            logger.error(f"Error creating user config for {user_id}: {str(e)}")
            raise

    async def update_user_config(
        self, user_id: str, config_request: ConfigRequest
    ) -> UserConfig:
        """
        Update an existing user configuration

        Args:
            config_request: Updated configuration data

        Returns:
            Updated UserConfig
        """
        try:
            # Get existing configuration
            existing_config = await self.get_user_config(user_id)
            if not existing_config:
                raise ValueError(f"No configuration found for user: {user_id}")

            # Update staging database config with encrypted password
            staging_db_response = (
                self.supabase.table("database_configs")
                .update(
                    {
                        "name": config_request.stagingDb.name,
                        "uri": config_request.stagingDb.uri,
                        "username": config_request.stagingDb.username,
                        "password": encrypt_password(config_request.stagingDb.password),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                .eq("id", existing_config.stagingDb.id)
                .execute()
            )

            if not staging_db_response.data:
                raise Exception("Failed to update staging database configuration")

            # Update production database config with encrypted password
            prod_db_response = (
                self.supabase.table("database_configs")
                .update(
                    {
                        "name": config_request.prodDb.name,
                        "uri": config_request.prodDb.uri,
                        "username": config_request.prodDb.username,
                        "password": encrypt_password(config_request.prodDb.password),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                .eq("id", existing_config.prodDb.id)
                .execute()
            )

            if not prod_db_response.data:
                raise Exception("Failed to update production database configuration")

            # Update user config timestamp
            config_response = (
                self.supabase.table("configs")
                .update({"updated_at": datetime.now(timezone.utc).isoformat()})
                .eq("id", existing_config.id)
                .execute()
            )

            if not config_response.data:
                raise Exception("Failed to update user configuration timestamp")

            # Return the updated configuration
            updated_config = await self.get_user_config(user_id)
            if not updated_config:
                raise Exception("Failed to retrieve updated configuration")

            logger.info(f"Updated configuration for user: {user_id}")
            return updated_config

        except Exception as e:
            logger.error(f"Error updating user config for {user_id}: {str(e)}")
            raise

    async def delete_user_config(self, user_id: str) -> bool:
        """
        Delete a user configuration

        Args:
            user_id: User's ID

        Returns:
            True if deleted successfully
        """
        try:
            # Get existing configuration
            existing_config = await self.get_user_config(user_id)
            if not existing_config:
                logger.info(f"No configuration found to delete for user: {user_id}")
                return True

            # Delete user config (this should cascade to database configs if foreign keys are set up properly)
            self.supabase.table("configs").delete().eq(
                "id", existing_config.id
            ).execute()

            # Delete database configs explicitly (in case cascade is not set up)
            self.supabase.table("database_configs").delete().eq(
                "id", existing_config.stagingDb.id
            ).execute()
            self.supabase.table("database_configs").delete().eq(
                "id", existing_config.prodDb.id
            ).execute()

            logger.info(f"Deleted configuration for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting user config for {user_id}: {str(e)}")
            raise


# Global instance
config_service = ConfigService()
