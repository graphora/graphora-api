"""Factory for creating graph storage instances.

This module provides a unified way to create storage instances based on
configuration, supporting both Neo4j and in-memory storage.
"""

import logging
from typing import Optional

from graphora_server.config import settings
from graphora_server.services.storage.interface import GraphStorageInterface

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
        from graphora_server.services.storage.memory import InMemoryStorage

        logger.info(f"Creating in-memory storage for user {config.user_id}")
        return InMemoryStorage(user_id=config.user_id)

    elif storage_type == "neo4j":
        if not all([config.uri, config.username, config.password]):
            raise ValueError(
                "Neo4j storage requires uri, username, and password configuration"
            )

        try:
            from graphora_server.services.storage.neo4j import Neo4jStorage
        except ImportError as exc:  # pragma: no cover — exercised without [neo4j]
            raise ImportError(
                "Neo4j storage requires the [neo4j] extra. "
                "Install with: pip install 'graphora-server[neo4j]'"
            ) from exc

        logger.info(f"Creating Neo4j storage for {config.uri}")
        return Neo4jStorage(
            uri=config.uri,
            username=config.username,
            password=config.password,
            database=config.database,
        )

    elif storage_type == "postgres":
        # All real app storage flows enter through
        # ``create_storage_for_user()``, not ``create_storage()`` —
        # tasks.py, api/quality.py, api/dashboard.py,
        # services/quality/tasks.py, services/merge/new_merger.py.
        # Wiring just this entry point would let an operator flip
        # STORAGE_TYPE=postgres, see unit tests pass, then hit a
        # NotImplementedError on the first real extraction request.
        # Both entry points must move together — that lands in
        # slice 3 alongside per-user Postgres config in
        # UserDatabaseService and store_nodes / store_relationships.
        raise NotImplementedError(
            "STORAGE_TYPE='postgres' (Apache AGE) — slice 2 shipped the "
            "adapter method bodies (checkpoint round-trip, agtype parsing) "
            "but factory dispatch is intentionally not wired here because "
            "the app's real storage flows enter through "
            "create_storage_for_user(). Both entry points move together "
            "in slice 3 alongside per-user Postgres config."
        )

    else:
        raise ValueError(
            f"Unsupported storage type: {storage_type}. "
            f"Supported types: 'neo4j', 'postgres', 'memory'."
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

    Raises:
        ValueError: If use_staging=False (production) and prodDb is not configured

    Notes:
        - If use_staging=True and stagingDb is not configured, falls back to in-memory storage
        - If use_staging=False and prodDb is not configured, raises ValueError (production required for merge)
    """
    storage_type = settings.STORAGE_TYPE.lower()

    if storage_type == "memory":
        from graphora_server.services.storage.memory import InMemoryStorage

        logger.info(f"Using in-memory storage for user {user_id}")
        return InMemoryStorage(user_id=user_id)

    elif storage_type == "neo4j":
        # Get user's database configuration
        from graphora_server.services.user_db_service import UserDatabaseService

        user_config = await UserDatabaseService.get_user_config(user_id)

        if use_staging:
            # Staging is optional - fall back to in-memory if not configured
            if user_config.stagingDb is None:
                from graphora_server.services.storage.memory import InMemoryStorage

                logger.info(
                    f"No staging DB configured for user {user_id}, using in-memory storage"
                )
                return InMemoryStorage(user_id=user_id)
            db_config = user_config.stagingDb
        else:
            # Production is required for merge operations
            if user_config.prodDb is None:
                raise ValueError(
                    "Production database is required for this operation. "
                    "Please configure a production database in Settings → Databases."
                )
            db_config = user_config.prodDb

        try:
            from graphora_server.services.storage.neo4j import Neo4jStorage
        except ImportError as exc:  # pragma: no cover — exercised without [neo4j]
            raise ImportError(
                "Neo4j storage requires the [neo4j] extra. "
                "Install with: pip install 'graphora-server[neo4j]'"
            ) from exc

        logger.info(f"Using Neo4j storage at {db_config.uri} for user {user_id}")
        return Neo4jStorage(
            uri=db_config.uri,
            username=db_config.username,
            password=db_config.password,
            database="neo4j",
        )

    elif storage_type == "postgres":
        # Per-user Postgres dispatch (separate stagingDb / prodDb
        # connections per user, like Neo4j) lands in C2-postgres
        # slice 3 alongside store_nodes / store_relationships. For
        # slice 2 the global ``create_storage()`` entry point is
        # the supported way to use the AGE backend.
        raise NotImplementedError(
            "STORAGE_TYPE='postgres' for per-user dispatch is not yet "
            "wired (Gate 5 / C2-postgres slice 3). Use create_storage() "
            "with global POSTGRES_AGE_DSN config for now."
        )

    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


def is_memory_storage_enabled() -> bool:
    """Check if in-memory storage is enabled."""
    return settings.STORAGE_TYPE.lower() == "memory"


async def get_storage_type_for_user(user_id: str, use_staging: bool = True) -> str:
    """Determine what storage type will be used for a user.

    Args:
        user_id: User ID for configuration lookup
        use_staging: Whether checking staging (True) or production (False)

    Returns:
        'neo4j' if Neo4j storage will be used, 'memory' if in-memory
    """
    if settings.STORAGE_TYPE.lower() == "memory":
        return "memory"

    from graphora_server.services.user_db_service import UserDatabaseService

    user_config = await UserDatabaseService.get_user_config(user_id)

    if use_staging and (user_config.stagingDb is None):
        return "memory"

    if not use_staging and (user_config.prodDb is None):
        return "none"  # Production not configured

    return "neo4j"


async def user_has_staging_db(user_id: str) -> bool:
    """Check if user has a staging database configured."""
    from graphora_server.services.user_db_service import UserDatabaseService

    user_config = await UserDatabaseService.get_user_config(user_id)
    return user_config.stagingDb is not None


async def user_has_production_db(user_id: str) -> bool:
    """Check if user has a production database configured."""
    from graphora_server.services.user_db_service import UserDatabaseService

    user_config = await UserDatabaseService.get_user_config(user_id)
    return user_config.prodDb is not None
