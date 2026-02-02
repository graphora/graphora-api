from typing import Literal, TYPE_CHECKING
from app.services.config_service import config_service
from app.schemas.config import UserConfig, DatabaseConfig
from app.utils.logger import logger
from app.config import settings

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from app.services.graph_service import GraphService

DatabaseEnvironment = Literal["staging", "production"]


def is_memory_storage_enabled() -> bool:
    """Check if in-memory storage is enabled."""
    return settings.STORAGE_TYPE.lower() == "memory"


class UserDatabaseService:
    """Service to manage user-specific database connections"""

    @staticmethod
    async def get_user_config(user_id: str) -> UserConfig:
        """
        Get user configuration by user ID

        Args:
            user_id: User's ID

        Returns:
            UserConfig: User's database configuration

        Raises:
            ValueError: If user configuration not found (unless using memory storage)
        """
        # In memory mode, we don't need database configuration
        if is_memory_storage_enabled():
            logger.info(
                f"Memory storage enabled, skipping database config for user {user_id}"
            )
            # Return a placeholder config for memory mode
            return UserConfig(
                stagingDb=DatabaseConfig(
                    uri="memory://localhost",
                    username="memory",
                    password="memory",
                ),
                prodDb=DatabaseConfig(
                    uri="memory://localhost",
                    username="memory",
                    password="memory",
                ),
            )

        user_config = await config_service.get_user_config(user_id)
        if not user_config:
            raise ValueError(
                f"Database configuration not found for user: {user_id}. Please configure your databases first."
            )

        return user_config

    @staticmethod
    async def get_database_config(
        user_id: str, environment: DatabaseEnvironment
    ) -> DatabaseConfig:
        """
        Get specific database configuration for user

        Args:
            user_id: User's ID
            environment: Database environment (staging or production)

        Returns:
            DatabaseConfig: Database configuration for the specified environment
        """
        user_config = await UserDatabaseService.get_user_config(user_id)

        if environment == "staging":
            return user_config.stagingDb
        elif environment == "production":
            return user_config.prodDb
        else:
            raise ValueError(
                f"Invalid environment: {environment}. Must be 'staging' or 'production'"
            )

    @staticmethod
    async def get_graph_service(
        user_id: str, environment: DatabaseEnvironment
    ) -> "GraphService":
        """
        Get GraphService instance for user's specific database

        Args:
            user_id: User's ID
            environment: Database environment (staging or production)

        Returns:
            GraphService: Configured graph service instance
        """
        db_config = await UserDatabaseService.get_database_config(user_id, environment)

        logger.info(
            f"Creating graph service for user {user_id} on {environment} database: {db_config.uri}"
        )

        from app.services.graph_service import GraphService

        return GraphService(
            uri=db_config.uri, user=db_config.username, password=db_config.password
        )

    @staticmethod
    async def get_staging_graph_service(user_id: str) -> "GraphService":
        """Get staging database GraphService for user"""
        return await UserDatabaseService.get_graph_service(user_id, "staging")

    @staticmethod
    async def get_production_graph_service(user_id: str) -> "GraphService":
        """Get production database GraphService for user"""
        return await UserDatabaseService.get_graph_service(user_id, "production")


# Global instance
user_db_service = UserDatabaseService()
