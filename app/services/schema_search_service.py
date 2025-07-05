import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from supabase import create_client, Client

from app.schemas.schema import (
    SchemaSearchRequest,
    SchemaSearchResponse, 
    SchemaSearchResult,
    StoredSchema
)
from app.config import settings

logger = logging.getLogger(__name__)


class SchemaSearchService:
    """Service for searching schemas using vector similarity and text search"""
    
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        self.schemas_table = "generated_schemas"
        self.embeddings_table = "schema_embeddings"
    
    async def search_schemas(
        self,
        user_id: str,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.5,
        include_content: bool = False
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
                include_content=include_content
            )
            
            took_ms = int((time.time() - start_time) * 1000)
            
            return SchemaSearchResponse(
                results=results,
                total=len(results),
                query=query,
                took_ms=took_ms
            )
            
        except Exception as e:
            logger.error(f"Error searching schemas: {str(e)}")
            took_ms = int((time.time() - start_time) * 1000)
            
            return SchemaSearchResponse(
                results=[],
                total=0,
                query=query,
                took_ms=took_ms
            )
    
    async def _text_based_search(
        self,
        user_id: str,
        query: str,
        domain: Optional[str],
        limit: int,
        threshold: float,
        include_content: bool
    ) -> List[SchemaSearchResult]:
        """Perform text-based search as fallback"""
        
        try:
            # Build base query
            select_fields = "id, title, description, domain, tags, created_at, updated_at, usage_count, user_id"
            if include_content:
                select_fields += ", content"
            
            # Search in public schemas and user's own schemas
            query_builder = self.supabase.table(self.schemas_table)\
                .select(select_fields)\
                .or_(f"user_id.eq.{user_id},is_public.eq.true")
            
            # Add domain filter if specified
            if domain and domain != "Other":
                query_builder = query_builder.eq("domain", domain)
            
            # Execute query
            result = query_builder.order("usage_count", desc=True)\
                .limit(limit * 2)\
                .execute()
            
            if not result.data:
                return []
            
            # Convert to StoredSchema objects for easier handling
            schemas = []
            for item in result.data:
                schemas.append(StoredSchema(**item))
            
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
                        user_id=schema.user_id
                    )
                    search_results.append(search_result)
            
            # Sort by similarity score and usage count
            search_results.sort(
                key=lambda x: (x.similarity, x.usage_count),
                reverse=True
            )
            
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in text-based search: {str(e)}")
            return []
    
    def _calculate_text_similarity(self, schema: StoredSchema, query_terms: set) -> float:
        """Calculate text similarity score between schema and query"""
        
        try:
            # Create searchable text from schema
            searchable_parts = [
                schema.title.lower(),
                schema.description.lower(),
                schema.domain.lower(),
                ' '.join(schema.tags).lower()
            ]
            
            # Add content if available (sample for performance)
            if hasattr(schema, 'content') and schema.content:
                content_sample = schema.content[:500].lower()  # First 500 chars
                searchable_parts.append(content_sample)
            
            searchable_text = ' '.join(searchable_parts)
            text_words = set(searchable_text.split())
            
            # Calculate similarity metrics
            exact_matches = len(query_terms.intersection(text_words))
            partial_matches = 0
            
            # Check for partial matches
            for query_term in query_terms:
                for text_word in text_words:
                    if query_term in text_word or text_word in query_term:
                        if len(query_term) > 2 and len(text_word) > 2:  # Avoid short word matches
                            partial_matches += 0.5
                            break
            
            # Calculate base similarity
            if len(query_terms) == 0:
                return 0.0
            
            similarity = (exact_matches + partial_matches) / len(query_terms)
            
            # Boost for domain matches
            query_text = ' '.join(query_terms).lower()
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
        user_id: str,
        query_embedding: List[float],
        domain: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.7
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
        self,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> List[SchemaSearchResult]:
        """Get popular schemas filtered by domain"""
        
        try:
            query_builder = self.supabase.table(self.schemas_table)\
                .select("id, title, description, domain, tags, created_at, updated_at, usage_count, user_id")\
                .eq("is_public", True)
            
            if domain and domain != "Other":
                query_builder = query_builder.eq("domain", domain)
            
            result = query_builder.order("usage_count", desc=True)\
                .limit(limit)\
                .execute()
            
            search_results = []
            for item in result.data:
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
                    user_id=item["user_id"]
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error getting popular schemas by domain: {str(e)}")
            return []
    
    async def get_related_schemas(
        self,
        schema_id: str,
        user_id: str,
        limit: int = 5
    ) -> List[SchemaSearchResult]:
        """Get schemas related to a given schema"""
        
        try:
            # Get the source schema
            source_schema = self.supabase.table(self.schemas_table)\
                .select("title, description, domain, tags")\
                .eq("id", schema_id)\
                .single()\
                .execute()
            
            if not source_schema.data:
                return []
            
            # Create a search query from the source schema
            query_parts = [
                source_schema.data["domain"],
                source_schema.data["title"]
            ]
            query_parts.extend(source_schema.data["tags"][:3])  # Add some tags
            
            search_query = ' '.join(query_parts)
            
            # Search for related schemas
            results = await self.search_schemas(
                user_id=user_id,
                query=search_query,
                domain=source_schema.data["domain"],
                limit=limit + 1,  # Get one extra in case source schema is included
                threshold=0.3
            )
            
            # Filter out the source schema itself
            related_schemas = [
                result for result in results.results 
                if result.id != schema_id
            ]
            
            return related_schemas[:limit]
            
        except Exception as e:
            logger.error(f"Error getting related schemas for {schema_id}: {str(e)}")
            return []
    
    async def suggest_schemas_for_context(
        self,
        user_id: str,
        context: Dict[str, Any],
        limit: int = 5
    ) -> List[SchemaSearchResult]:
        """Suggest schemas based on user context"""
        
        try:
            # Build search query from context
            query_parts = []
            
            if context.get('domain'):
                query_parts.append(str(context['domain']))
            if context.get('use_case'):
                # Extract key terms from use case
                use_case_words = str(context['use_case']).split()[:5]
                query_parts.extend(use_case_words)
            if context.get('key_entities'):
                # Extract entities
                entities = str(context['key_entities']).split(',')[:3]
                query_parts.extend([e.strip() for e in entities])
            
            if not query_parts:
                # Fallback to popular schemas
                return await self.get_popular_schemas_by_domain(limit=limit)
            
            search_query = ' '.join(query_parts)
            
            # Search for matching schemas
            results = await self.search_schemas(
                user_id=user_id,
                query=search_query,
                domain=context.get('domain'),
                limit=limit,
                threshold=0.4
            )
            
            return results.results
            
        except Exception as e:
            logger.error(f"Error suggesting schemas for context: {str(e)}")
            return []


# Global service instance
schema_search_service = SchemaSearchService()