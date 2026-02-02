from __future__ import annotations

from typing import Optional
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.db import postgres as db
from app.schemas.config import (
    UserConfig,
    DatabaseConfig,
    DatabaseConfigUpdate,
    ConfigRequest,
    ConfigUpdateRequest,
)
from app.utils.logger import logger
from app.utils.encryption import encrypt_password, decrypt_password


class ConfigService:
    """Service for managing user database configurations."""

    def __init__(self) -> None:
        if not (settings.DATABASE_URL or settings.resolved_database_url):
            if not settings.test_mode:
                raise ValueError("DATABASE_URL must be configured")

    async def get_user_config(self, user_id: str) -> Optional[UserConfig]:
        query = """
            SELECT
                c.id,
                c.user_id,
                c.created_at,
                c.updated_at,
                CASE WHEN s.id IS NOT NULL THEN
                    json_build_object(
                        'id', s.id,
                        'name', s.name,
                        'uri', s.uri,
                        'username', s.username,
                        'password', s.password
                    )
                ELSE NULL END AS staging_db,
                CASE WHEN p.id IS NOT NULL THEN
                    json_build_object(
                        'id', p.id,
                        'name', p.name,
                        'uri', p.uri,
                        'username', p.username,
                        'password', p.password
                    )
                ELSE NULL END AS prod_db
            FROM configs c
            LEFT JOIN database_configs s ON s.id = c.staging_db_id
            LEFT JOIN database_configs p ON p.id = c.prod_db_id
            WHERE c.user_id = %s
            LIMIT 1
        """

        record = await db.fetchrow(query, user_id)
        if not record:
            logger.info("No configuration found for user %s", user_id)
            return None

        return self._map_record_to_user_config(record)

    async def create_user_config(
        self, user_id: str, config_request: ConfigRequest
    ) -> UserConfig:
        existing = await self.get_user_config(user_id)
        if existing:
            raise ValueError(f"Configuration already exists for user: {user_id}")

        # At least one database must be provided
        if not config_request.stagingDb and not config_request.prodDb:
            raise ValueError("At least one database configuration must be provided")

        staging_db_id = str(uuid.uuid4()) if config_request.stagingDb else None
        prod_db_id = str(uuid.uuid4()) if config_request.prodDb else None
        config_id = str(uuid.uuid4())

        async with db.transaction() as cur:
            # Insert staging DB if provided
            if config_request.stagingDb:
                await cur.execute(
                    """
                    INSERT INTO database_configs (id, name, uri, username, password)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        staging_db_id,
                        config_request.stagingDb.name,
                        config_request.stagingDb.uri,
                        config_request.stagingDb.username,
                        encrypt_password(config_request.stagingDb.password),
                    ),
                )

            # Insert prod DB if provided
            if config_request.prodDb:
                await cur.execute(
                    """
                    INSERT INTO database_configs (id, name, uri, username, password)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        prod_db_id,
                        config_request.prodDb.name,
                        config_request.prodDb.uri,
                        config_request.prodDb.username,
                        encrypt_password(config_request.prodDb.password),
                    ),
                )

            await cur.execute(
                """
                INSERT INTO configs (id, user_id, staging_db_id, prod_db_id)
                VALUES (%s, %s, %s, %s)
                """,
                (config_id, user_id, staging_db_id, prod_db_id),
            )

        logger.info("Created configuration for user %s", user_id)
        created = await self.get_user_config(user_id)
        if not created:
            raise RuntimeError("Failed to load configuration after insert")
        return created

    async def update_user_config(
        self, user_id: str, config_request: ConfigUpdateRequest
    ) -> UserConfig:
        existing = await self.get_user_config(user_id)
        if not existing:
            raise ValueError(f"No configuration found for user: {user_id}")

        async with db.transaction() as cur:
            # Handle staging DB update or creation
            if self._should_update(config_request.stagingDb):
                if existing.stagingDb:
                    # Update existing staging DB
                    await cur.execute(
                        """
                        UPDATE database_configs
                        SET name = %s,
                            uri = %s,
                            username = %s,
                            password = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            config_request.stagingDb.name or existing.stagingDb.name,
                            config_request.stagingDb.uri or existing.stagingDb.uri,
                            config_request.stagingDb.username
                            or existing.stagingDb.username,
                            encrypt_password(config_request.stagingDb.password),
                            existing.stagingDb.id,
                        ),
                    )
                else:
                    # Create new staging DB
                    staging_db_id = str(uuid.uuid4())
                    await cur.execute(
                        """
                        INSERT INTO database_configs (id, name, uri, username, password)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            staging_db_id,
                            config_request.stagingDb.name or "neo4j",
                            config_request.stagingDb.uri,
                            config_request.stagingDb.username or "neo4j",
                            encrypt_password(config_request.stagingDb.password),
                        ),
                    )
                    await cur.execute(
                        "UPDATE configs SET staging_db_id = %s WHERE id = %s",
                        (staging_db_id, existing.id),
                    )

            # Handle prod DB update or creation
            if self._should_update(config_request.prodDb):
                if existing.prodDb:
                    # Update existing prod DB
                    await cur.execute(
                        """
                        UPDATE database_configs
                        SET name = %s,
                            uri = %s,
                            username = %s,
                            password = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            config_request.prodDb.name or existing.prodDb.name,
                            config_request.prodDb.uri or existing.prodDb.uri,
                            config_request.prodDb.username or existing.prodDb.username,
                            encrypt_password(config_request.prodDb.password),
                            existing.prodDb.id,
                        ),
                    )
                else:
                    # Create new prod DB
                    prod_db_id = str(uuid.uuid4())
                    await cur.execute(
                        """
                        INSERT INTO database_configs (id, name, uri, username, password)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            prod_db_id,
                            config_request.prodDb.name or "neo4j",
                            config_request.prodDb.uri,
                            config_request.prodDb.username or "neo4j",
                            encrypt_password(config_request.prodDb.password),
                        ),
                    )
                    await cur.execute(
                        "UPDATE configs SET prod_db_id = %s WHERE id = %s",
                        (prod_db_id, existing.id),
                    )

            await cur.execute(
                "UPDATE configs SET updated_at = NOW() WHERE id = %s",
                (existing.id,),
            )

        updated = await self.get_user_config(user_id)
        if not updated:
            raise RuntimeError("Failed to load configuration after update")
        logger.info("Updated configuration for user %s", user_id)
        return updated

    async def delete_user_config(self, user_id: str) -> bool:
        existing = await self.get_user_config(user_id)
        if not existing:
            return False

        async with db.transaction() as cur:
            await cur.execute("DELETE FROM configs WHERE user_id = %s", (user_id,))
            if existing.stagingDb:
                await cur.execute(
                    "DELETE FROM database_configs WHERE id = %s",
                    (existing.stagingDb.id,),
                )
            if existing.prodDb:
                await cur.execute(
                    "DELETE FROM database_configs WHERE id = %s",
                    (existing.prodDb.id,),
                )

        logger.info("Deleted configuration for user %s", user_id)
        return True

    def _map_record_to_user_config(self, record: dict) -> UserConfig:
        staging = record["staging_db"]
        prod = record["prod_db"]

        staging_config = None
        if staging:
            staging_config = DatabaseConfig(
                id=str(staging["id"]),
                name=staging["name"],
                uri=staging["uri"],
                username=staging["username"],
                password=decrypt_password(staging["password"]),
            )

        prod_config = None
        if prod:
            prod_config = DatabaseConfig(
                id=str(prod["id"]),
                name=prod["name"],
                uri=prod["uri"],
                username=prod["username"],
                password=decrypt_password(prod["password"]),
            )

        return UserConfig(
            id=str(record["id"]),
            userId=str(record["user_id"]),
            stagingDb=staging_config,
            prodDb=prod_config,
            createdAt=self._parse_ts(record["created_at"]),
            updatedAt=self._parse_ts(record["updated_at"]),
        )

    @staticmethod
    def _parse_ts(value: Optional[datetime | str]) -> datetime:
        """Normalise DB timestamps returned as either strings or datetime objects."""
        if value is None:
            return datetime.now(timezone.utc)

        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

        if isinstance(value, str):
            normalised = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalised)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    @staticmethod
    def _should_update(db_request: Optional[DatabaseConfigUpdate]) -> bool:
        if not db_request:
            return False
        return bool(
            (db_request.uri and db_request.uri.strip())
            and (db_request.password and db_request.password.strip())
        )


config_service = ConfigService()
