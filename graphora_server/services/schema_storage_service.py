import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from psycopg.types.json import Json

from graphora_server.schemas.schema import (
    StoredSchema,
    CreateSchemaRequest,
    UpdateSchemaRequest,
    SchemaUsageEvent,
)
from graphora_server.config import settings
from graphora_server.db import postgres as db
from graphora_server.services.cache import get_schema_cache

logger = logging.getLogger(__name__)


class SchemaStorageService:
    """Service for storing and managing schemas in Postgres."""

    def __init__(self):
        if not (settings.DATABASE_URL or settings.resolved_database_url):
            if not settings.test_mode:
                raise ValueError("DATABASE_URL must be configured for schema storage")
        self.schemas_table = "generated_schemas"
        self.usage_table = "schema_usage_events"

    async def store_generated_schema(
        self,
        schema_id: str,
        user_id: str,
        schema_content: str,
        context: Dict[str, Any],
        confidence: float,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Store a generated schema"""

        try:
            # Generate title and description from context if not provided
            if not title:
                use_case = context.get("use_case", "Generated Schema")
                domain = context.get("domain", "General")
                title = f"{domain} - {use_case}"[:200]

            if not description:
                description = "Auto-generated schema based on user requirements"
                if context.get("use_case"):
                    description += f" for {context['use_case']}"

            # Extract tags from context
            tags = []
            if context.get("domain"):
                tags.append(context["domain"].lower().replace("/", "_"))
            if context.get("data_complexity"):
                tags.append(context["data_complexity"].lower())
            if context.get("data_volume"):
                tags.append(context["data_volume"].lower().replace(" ", "_"))

            # Store in database
            now = datetime.utcnow()
            row = await db.fetchrow(
                f"""
                INSERT INTO {self.schemas_table} (
                    id, user_id, title, description, content, domain,
                    tags, confidence, context, is_public, usage_count,
                    created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, 0, %s, %s)
                RETURNING id
                """,
                schema_id,
                user_id,
                title,
                description,
                schema_content,
                context.get("domain", "General"),
                tags,
                confidence,
                Json(context),
                now,
                now,
            )

            if row:
                logger.info(f"Stored generated schema {schema_id} for user {user_id}")
                return True

            logger.error(f"Failed to store schema {schema_id}: insert returned no row")
            return False

        except Exception as e:
            logger.error(f"Error storing generated schema {schema_id}: {str(e)}")
            return False

    async def get_schema(
        self, schema_id: str, user_id: str, use_cache: bool = True
    ) -> Optional[StoredSchema]:
        """Get a specific schema by ID"""

        cache = get_schema_cache()

        # Check cache first
        if use_cache:
            cached = await cache.get(schema_id, user_id)
            if cached is not None:
                logger.debug(f"Schema '{schema_id}' loaded from cache")
                return StoredSchema(**cached)

        try:
            row = await db.fetchrow(
                f"""
                SELECT *
                FROM {self.schemas_table}
                WHERE id = %s AND (user_id = %s OR is_public = TRUE)
                LIMIT 1
                """,
                schema_id,
                user_id,
            )

            if row:
                schema = StoredSchema(**row)
                # Cache the schema
                if use_cache:
                    await cache.set(schema_id, user_id, dict(row))
                return schema

            return None

        except Exception as e:
            logger.error(f"Error getting schema {schema_id}: {str(e)}")
            return None

    async def list_user_schemas(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        include_public: bool = True,
    ) -> List[StoredSchema]:
        """List schemas for a user"""

        try:
            rows: List[Dict[str, Any]]
            if include_public:
                rows = await db.fetch(
                    f"""
                    SELECT *
                    FROM {self.schemas_table}
                    WHERE user_id = %s OR is_public = TRUE
                    ORDER BY updated_at DESC
                    OFFSET %s LIMIT %s
                    """,
                    user_id,
                    offset,
                    limit,
                )
            else:
                rows = await db.fetch(
                    f"""
                    SELECT *
                    FROM {self.schemas_table}
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    OFFSET %s LIMIT %s
                    """,
                    user_id,
                    offset,
                    limit,
                )

            return [StoredSchema(**item) for item in rows or []]

        except Exception as e:
            logger.error(f"Error listing schemas for user {user_id}: {str(e)}")
            return []

    async def create_schema(
        self, user_id: str, request: CreateSchemaRequest
    ) -> Optional[StoredSchema]:
        """Create a new schema"""

        try:
            schema_id = str(uuid.uuid4())

            now = datetime.utcnow()
            row = await db.fetchrow(
                f"""
                INSERT INTO {self.schemas_table} (
                    id, user_id, title, description, content,
                    domain, tags, is_public, usage_count,
                    created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, %s, %s)
                RETURNING *
                """,
                schema_id,
                user_id,
                request.title,
                request.description,
                request.content,
                request.domain,
                request.tags,
                request.is_public,
                now,
                now,
            )

            if row:
                return StoredSchema(**row)

            return None

        except Exception as e:
            logger.error(f"Error creating schema for user {user_id}: {str(e)}")
            return None

    async def update_schema(
        self, schema_id: str, user_id: str, request: UpdateSchemaRequest
    ) -> Optional[StoredSchema]:
        """Update an existing schema"""

        try:
            # Build update data
            update_data = {"updated_at": datetime.utcnow().isoformat()}

            if request.title is not None:
                update_data["title"] = request.title
            if request.description is not None:
                update_data["description"] = request.description
            if request.content is not None:
                update_data["content"] = request.content
            if request.domain is not None:
                update_data["domain"] = request.domain
            if request.tags is not None:
                update_data["tags"] = request.tags
            if request.is_public is not None:
                update_data["is_public"] = request.is_public

            row = await db.fetchrow(
                f"""
                UPDATE {self.schemas_table}
                SET title = COALESCE(%s, title),
                    description = COALESCE(%s, description),
                    content = COALESCE(%s, content),
                    domain = COALESCE(%s, domain),
                    tags = COALESCE(%s, tags),
                    is_public = COALESCE(%s, is_public),
                    updated_at = NOW()
                WHERE id = %s AND user_id = %s
                RETURNING *
                """,
                update_data.get("title"),
                update_data.get("description"),
                update_data.get("content"),
                update_data.get("domain"),
                update_data.get("tags"),
                update_data.get("is_public"),
                schema_id,
                user_id,
            )

            if row:
                # Invalidate cache on update
                cache = get_schema_cache()
                await cache.invalidate(schema_id, user_id)
                return StoredSchema(**row)

            return None

        except Exception as e:
            logger.error(f"Error updating schema {schema_id}: {str(e)}")
            return None

    async def delete_schema(self, schema_id: str, user_id: str) -> bool:
        """Delete a schema"""

        try:
            row = await db.fetchrow(
                f"""
                DELETE FROM {self.schemas_table}
                WHERE id = %s AND user_id = %s
                RETURNING id
                """,
                schema_id,
                user_id,
            )

            if row:
                # Invalidate cache on delete
                cache = get_schema_cache()
                await cache.invalidate(schema_id, user_id)
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting schema {schema_id}: {str(e)}")
            return False

    async def update_generated_schema(
        self,
        schema_id: str,
        user_id: str,
        updated_content: str,
        refinement_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a generated schema with refinements"""

        try:
            update_data = {
                "content": updated_content,
                "updated_at": datetime.utcnow().isoformat(),
            }

            if refinement_metadata:
                # Get current context and merge with refinement metadata
                current = await self.get_schema(schema_id, user_id, use_cache=False)
                if current:
                    context = getattr(current, "context", {}) or {}
                    context.update({"refinement": refinement_metadata})
                    update_data["context"] = context

            row = await db.fetchrow(
                f"""
                UPDATE {self.schemas_table}
                SET content = %s,
                    context = COALESCE(%s::jsonb, context),
                    updated_at = NOW()
                WHERE id = %s AND user_id = %s
                RETURNING id
                """,
                updated_content,
                Json(update_data.get("context")) if "context" in update_data else None,
                schema_id,
                user_id,
            )

            if row:
                # Invalidate cache on update
                cache = get_schema_cache()
                await cache.invalidate(schema_id, user_id)
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating generated schema {schema_id}: {str(e)}")
            return False

    async def increment_usage_count(self, schema_id: str) -> bool:
        """Increment the usage count for a schema"""

        try:
            # Use a stored procedure or direct SQL for atomic increment
            row = await db.fetchrow(
                "SELECT increment_schema_usage(%s) AS updated",
                schema_id,
            )

            return bool(row and row.get("updated"))

        except Exception as e:
            logger.error(
                f"Error incrementing usage count for schema {schema_id}: {str(e)}"
            )
            return False

    async def log_usage_event(
        self,
        schema_id: str,
        user_id: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log a schema usage event"""

        try:
            row = await db.fetchrow(
                f"""
                INSERT INTO {self.usage_table} (
                    id, schema_id, user_id, event_type, metadata, timestamp
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                str(uuid.uuid4()),
                schema_id,
                user_id,
                event_type,
                Json(metadata or {}),
                datetime.utcnow(),
            )

            if row:
                await self.increment_usage_count(schema_id)
                return True

            return False

        except Exception as e:
            logger.error(f"Error logging usage event for schema {schema_id}: {str(e)}")
            return False

    async def get_schema_analytics(
        self, schema_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get analytics for a schema"""

        try:
            # Get schema info
            schema = await self.get_schema(schema_id, user_id)
            if not schema:
                return None

            # Get recent usage events
            usage_rows = await db.fetch(
                f"""
                SELECT *
                FROM {self.usage_table}
                WHERE schema_id = %s
                ORDER BY timestamp DESC
                LIMIT 20
                """,
                schema_id,
            )

            usage_events = [SchemaUsageEvent(**event) for event in usage_rows or []]

            # Calculate unique users
            unique_users = len(set(event.user_id for event in usage_events))

            # Calculate popularity score (simple algorithm)
            popularity_score = min(schema.usage_count * 0.1 + unique_users * 0.5, 10.0)

            return {
                "schema_id": schema_id,
                "total_usage": schema.usage_count,
                "unique_users": unique_users,
                "recent_usage": usage_events,
                "popularity_score": popularity_score,
            }

        except Exception as e:
            logger.error(f"Error getting analytics for schema {schema_id}: {str(e)}")
            return None

    async def get_popular_schemas(
        self, domain: Optional[str] = None, limit: int = 10
    ) -> List[StoredSchema]:
        """Get popular public schemas"""

        try:
            params: List[Any] = []
            domain_clause = ""
            if domain:
                domain_clause = " AND domain = %s"
                params.append(domain)

            params.append(limit)

            rows = await db.fetch(
                f"""
                SELECT *
                FROM {self.schemas_table}
                WHERE is_public = TRUE{domain_clause}
                ORDER BY usage_count DESC, updated_at DESC
                LIMIT %s
                """,
                *params,
            )

            return [StoredSchema(**item) for item in rows or []]

        except Exception as e:
            logger.error(f"Error getting popular schemas: {str(e)}")
            return []


# Global service instance
schema_storage_service = SchemaStorageService()
