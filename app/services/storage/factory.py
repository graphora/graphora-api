"""Factory for creating graph storage instances.

This module provides a unified way to create storage instances based on
configuration, supporting both Neo4j and in-memory storage.
"""

import logging
from typing import Optional

from app.config import settings
from app.services.storage.interface import GraphStorageInterface

logger = logging.getLogger(__name__)


class StorageConfig:
    """Configuration for storage creation."""

    def __init__(
        self,
        storage_type: Optional[str] = None,
        user_id: Optional[str] = None,
        # Neo4j-specific settings
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
    ):
        self.storage_type = storage_type or settings.STORAGE_TYPE
        self.user_id = user_id or "default"
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database


async def create_storage(config: StorageConfig) -> GraphStorageInterface:
    """Create a storage instance based on configuration.

    Args:
        config: Storage configuration

    Returns:
        GraphStorageInterface implementation

    Raises:
        ValueError: If storage type is not supported or required config is missing
    """
    storage_type = config.storage_type.lower()

    if storage_type == "memory":
        from app.services.storage.memory import InMemoryStorage

        logger.info(f"Creating in-memory storage for user {config.user_id}")
        return InMemoryStorage(user_id=config.user_id)

    elif storage_type == "neo4j":
        if not all([config.uri, config.username, config.password]):
            raise ValueError(
                "Neo4j storage requires uri, username, and password configuration"
            )

        from app.services.storage.neo4j import Neo4jStorage

        logger.info(f"Creating Neo4j storage for {config.uri}")
        return Neo4jStorage(
            uri=config.uri,
            username=config.username,
            password=config.password,
            database=config.database,
        )

    else:
        raise ValueError(
            f"Unsupported storage type: {storage_type}. "
            f"Supported types: 'neo4j', 'memory'"
        )


async def create_storage_for_user(
    user_id: str,
    use_staging: bool = True,
) -> GraphStorageInterface:
    """Create storage for a user based on their configuration and global settings.

    This is a convenience function that handles the common case of creating
    storage for a user's transform or merge operations.

    Args:
        user_id: User ID for configuration lookup
        use_staging: Whether to use staging DB (True) or production DB (False)

    Returns:
        GraphStorageInterface implementation
    """
    storage_type = settings.STORAGE_TYPE.lower()

    if storage_type == "memory":
        from app.services.storage.memory import InMemoryStorage

        logger.info(f"Using in-memory storage for user {user_id}")
        return InMemoryStorage(user_id=user_id)

    elif storage_type == "neo4j":
        # Get user's database configuration
        from app.services.user_db_service import UserDatabaseService

        user_config = await UserDatabaseService.get_user_config(user_id)

        db_config = user_config.stagingDb if use_staging else user_config.productionDb

        from app.services.storage.neo4j import Neo4jStorage

        logger.info(f"Using Neo4j storage at {db_config.uri} for user {user_id}")
        return Neo4jStorage(
            uri=db_config.uri,
            username=db_config.username,
            password=db_config.password,
            database="neo4j",
        )

    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


def is_memory_storage_enabled() -> bool:
    """Check if in-memory storage is enabled."""
    return settings.STORAGE_TYPE.lower() == "memory"
