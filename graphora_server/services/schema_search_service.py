import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from graphora_server.schemas.schema import (
    SchemaSearchResponse,
    SchemaSearchResult,
    StoredSchema,
)
from graphora_server.config import settings
from graphora_server.db import postgres as db

logger = logging.getLogger(__name__)


class SchemaSearchService:
    """Service for searching schemas using vector similarity and text search."""

    def __init__(self):
        if not (settings.DATABASE_URL or settings.resolved_database_url):
            if not settings.test_mode:
                raise ValueError("DATABASE_URL must be configured for schema search")
        self.schemas_table = "generated_schemas"

    async def search_schemas(
        self,
        user_id: str,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.5,
        include_content: bool = False,
    ) -> SchemaSearchResponse:
        """Search schemas using vector similarity and text search"""

        start_time = time.time()

        try:
            # For now, implement text-based search
            # TODO: Implement vector search when embeddings are available
            results = await self._text_based_search(
                user_id=user_id,
                query=query,
                domain=domain,
                limit=limit,
                threshold=threshold,
                include_content=include_content,
            )

            took_ms = int((time.time() - start_time) * 1000)

            return SchemaSearchResponse(
                results=results, total=len(results), query=query, took_ms=took_ms
            )

        except Exception as e:
            logger.error(f"Error searching schemas: {str(e)}")
            took_ms = int((time.time() - start_time) * 1000)

            return SchemaSearchResponse(
                results=[], total=0, query=query, took_ms=took_ms
            )

    async def _text_based_search(
        self,
        user_id: str,
        query: str,
        domain: Optional[str],
        limit: int,
        threshold: float,
        include_content: bool,
    ) -> List[SchemaSearchResult]:
        """Perform text-based search as fallback"""

        try:
            params: List[Any] = [user_id]
            domain_clause = ""
            if domain and domain != "Other":
                domain_clause = " AND domain = %s"
                params.append(domain)

            params.append(limit * 2)

            rows = await db.fetch(
                f"""
                SELECT id, title, description, content, domain, tags,
                       created_at, updated_at, usage_count, user_id, is_public
                FROM {self.schemas_table}
                WHERE (user_id = %s OR is_public = TRUE){domain_clause}
                ORDER BY usage_count DESC, updated_at DESC
                LIMIT %s
                """,
                *params,
            )

            if not rows:
                return []

            # Convert to StoredSchema objects for easier handling
            schemas = []
            for item in rows:
                try:
                    # Ensure required fields have default values if missing
                    schema_data = {
                        "id": item.get("id", ""),
                        "title": item.get("title", "Untitled Schema"),
                        "description": item.get("description", ""),
                        "content": item.get("content", "version: 0.1.0\nentities: {}"),
                        "domain": item.get("domain", "Other"),
                        "tags": item.get("tags", []),
                        "user_id": item.get("user_id", "unknown"),
                        "is_public": item.get("is_public", False),
                        "usage_count": item.get("usage_count", 0),
                        "created_at": item.get("created_at", datetime.utcnow()),
                        "updated_at": item.get("updated_at", datetime.utcnow()),
                    }
                    schemas.append(StoredSchema(**schema_data))
                except Exception as e:
                    logger.warning(
                        f"Skipping invalid schema record {item.get('id', 'unknown')}: {str(e)}"
                    )
                    continue

            # Calculate text similarity scores
            search_results = []
            query_terms = set(query.lower().split())

            for schema in schemas:
                similarity = self._calculate_text_similarity(schema, query_terms)

                if similarity >= threshold:
                    search_result = SchemaSearchResult(
                        id=schema.id,
                        title=schema.title,
                        description=schema.description,
                        content=schema.content if include_content else None,
                        similarity=similarity,
                        domain=schema.domain,
                        tags=schema.tags,
                        created_at=schema.created_at,
                        updated_at=schema.updated_at,
                        usage_count=schema.usage_count,
                        user_id=schema.user_id,
                    )
                    search_results.append(search_result)

            # Sort by similarity score and usage count
            search_results.sort(
                key=lambda x: (x.similarity, x.usage_count), reverse=True
            )

            return search_results[:limit]

        except Exception as e:
            logger.error(f"Error in text-based search: {str(e)}")
            # Log more details for debugging
            logger.error(f"Query: {query}, Domain: {domain}, User: {user_id}")
            return []

    def _calculate_text_similarity(
        self, schema: StoredSchema, query_terms: set
    ) -> float:
        """Calculate text similarity score between schema and query"""

        try:
            # Create searchable text from schema
            searchable_parts = [
                schema.title.lower(),
                schema.description.lower(),
                schema.domain.lower(),
                " ".join(schema.tags).lower(),
            ]

            # Add content if available (sample for performance)
            if hasattr(schema, "content") and schema.content:
                content_sample = schema.content[:500].lower()  # First 500 chars
                searchable_parts.append(content_sample)

            searchable_text = " ".join(searchable_parts)
            text_words = set(searchable_text.split())

            # Calculate similarity metrics
            exact_matches = len(query_terms.intersection(text_words))
            partial_matches = 0

            # Check for partial matches
            for query_term in query_terms:
                for text_word in text_words:
                    if query_term in text_word or text_word in query_term:
                        if (
                            len(query_term) > 2 and len(text_word) > 2
                        ):  # Avoid short word matches
                            partial_matches += 0.5
                            break

            # Calculate base similarity
            if len(query_terms) == 0:
                return 0.0

            similarity = (exact_matches + partial_matches) / len(query_terms)

            # Boost for domain matches
            query_text = " ".join(query_terms).lower()
            if schema.domain.lower() in query_text:
                similarity += 0.2

            # Boost for popular schemas
            if schema.usage_count > 10:
                similarity += 0.1
            elif schema.usage_count > 5:
                similarity += 0.05

            # Cap at 1.0
            return min(similarity, 1.0)

        except Exception as e:
            logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0

    async def search_by_vector_similarity(
        self,
        _user_id: str,
        _query_embedding: List[float],
        _domain: Optional[str] = None,
        _limit: int = 10,
        _threshold: float = 0.7,
    ) -> List[SchemaSearchResult]:
        """Search schemas using vector similarity (requires embeddings)"""

        try:
            # This would use pgvector or similar for vector similarity search
            # For now, return empty results as embeddings are not implemented
            logger.info("Vector similarity search not yet implemented")
            return []

        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            return []

    async def get_popular_schemas_by_domain(
        self, domain: Optional[str] = None, limit: int = 10
    ) -> List[SchemaSearchResult]:
        """Get popular schemas filtered by domain"""

        try:
            params: List[Any] = []
            domain_clause = ""
            if domain and domain != "Other":
                domain_clause = " AND domain = %s"
                params.append(domain)

            params.append(limit)

            rows = await db.fetch(
                f"""
                SELECT id, title, description, domain, tags,
                       created_at, updated_at, usage_count, user_id
                FROM {self.schemas_table}
                WHERE is_public = TRUE{domain_clause}
                ORDER BY usage_count DESC, updated_at DESC
                LIMIT %s
                """,
                *params,
            )

            search_results = []
            for item in rows or []:
                search_result = SchemaSearchResult(
                    id=item["id"],
                    title=item["title"],
                    description=item["description"],
                    content=None,  # Don't include content for popular lists
                    similarity=1.0,  # Not applicable for popular schemas
                    domain=item["domain"],
                    tags=item["tags"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                    usage_count=item["usage_count"],
                    user_id=item["user_id"],
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"Error getting popular schemas by domain: {str(e)}")
            return []

    async def get_related_schemas(
        self, schema_id: str, user_id: str, limit: int = 5
    ) -> List[SchemaSearchResult]:
        """Get schemas related to a given schema"""

        try:
            source_schema = await db.fetchrow(
                f"""
                SELECT title, description, domain, tags
                FROM {self.schemas_table}
                WHERE id = %s
                """,
                schema_id,
            )

            if not source_schema:
                return []

            # Create a search query from the source schema
            query_parts = [source_schema["domain"], source_schema["title"]]
            tags = source_schema.get("tags") or []
            query_parts.extend(tags[:3])  # Add some tags

            search_query = " ".join(query_parts)

            # Search for related schemas
            results = await self.search_schemas(
                user_id=user_id,
                query=search_query,
                domain=source_schema["domain"],
                limit=limit + 1,  # Get one extra in case source schema is included
                threshold=0.3,
            )

            # Filter out the source schema itself
            related_schemas = [
                result for result in results.results if result.id != schema_id
            ]

            return related_schemas[:limit]

        except Exception as e:
            logger.error(f"Error getting related schemas for {schema_id}: {str(e)}")
            return []

    async def suggest_schemas_for_context(
        self, user_id: str, context: Dict[str, Any], limit: int = 5
    ) -> List[SchemaSearchResult]:
        """Suggest schemas based on user context"""

        try:
            # Build search query from context
            query_parts = []

            if context.get("domain"):
                query_parts.append(str(context["domain"]))
            if context.get("use_case"):
                # Extract key terms from use case
                use_case_words = str(context["use_case"]).split()[:5]
                query_parts.extend(use_case_words)
            if context.get("key_entities"):
                # Extract entities
                entities = str(context["key_entities"]).split(",")[:3]
                query_parts.extend([e.strip() for e in entities])

            if not query_parts:
                # Fallback to popular schemas
                return await self.get_popular_schemas_by_domain(limit=limit)

            search_query = " ".join(query_parts)

            # Search for matching schemas
            results = await self.search_schemas(
                user_id=user_id,
                query=search_query,
                domain=context.get("domain"),
                limit=limit,
                threshold=0.4,
            )

            return results.results

        except Exception as e:
            logger.error(f"Error suggesting schemas for context: {str(e)}")
            return []


# Global service instance
schema_search_service = SchemaSearchService()
