import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from supabase import create_client, Client

from app.schemas.schema import (
    StoredSchema, 
    CreateSchemaRequest, 
    UpdateSchemaRequest,
    SchemaUsageEvent
)
from app.config import settings

logger = logging.getLogger(__name__)


class SchemaStorageService:
    """Service for storing and managing schemas in Supabase"""
    
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
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
        description: Optional[str] = None
    ) -> bool:
        """Store a generated schema"""
        
        try:
            # Generate title and description from context if not provided
            if not title:
                use_case = context.get('use_case', 'Generated Schema')
                domain = context.get('domain', 'General')
                title = f"{domain} - {use_case}"[:200]
            
            if not description:
                description = f"Auto-generated schema based on user requirements"
                if context.get('use_case'):
                    description += f" for {context['use_case']}"
            
            # Extract tags from context
            tags = []
            if context.get('domain'):
                tags.append(context['domain'].lower().replace('/', '_'))
            if context.get('data_complexity'):
                tags.append(context['data_complexity'].lower())
            if context.get('data_volume'):
                tags.append(context['data_volume'].lower().replace(' ', '_'))
            
            # Store in database
            data = {
                "id": schema_id,
                "user_id": user_id,
                "title": title,
                "description": description,
                "content": schema_content,
                "domain": context.get('domain', 'General'),
                "tags": tags,
                "confidence": confidence,
                "context": context,
                "is_public": False,
                "usage_count": 0,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table(self.schemas_table).insert(data).execute()
            
            if result.data:
                logger.info(f"Stored generated schema {schema_id} for user {user_id}")
                return True
            else:
                logger.error(f"Failed to store schema {schema_id}: No data returned")
                return False
                
        except Exception as e:
            logger.error(f"Error storing generated schema {schema_id}: {str(e)}")
            return False
    
    async def get_schema(self, schema_id: str, user_id: str) -> Optional[StoredSchema]:
        """Get a specific schema by ID"""
        
        try:
            result = self.supabase.table(self.schemas_table)\
                .select("*")\
                .eq("id", schema_id)\
                .or_(f"user_id.eq.{user_id},is_public.eq.true")\
                .single()\
                .execute()
            
            if result.data:
                return StoredSchema(**result.data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting schema {schema_id}: {str(e)}")
            return None
    
    async def list_user_schemas(
        self, 
        user_id: str, 
        limit: int = 50,
        offset: int = 0,
        include_public: bool = True
    ) -> List[StoredSchema]:
        """List schemas for a user"""
        
        try:
            query = self.supabase.table(self.schemas_table).select("*")
            
            if include_public:
                query = query.or_(f"user_id.eq.{user_id},is_public.eq.true")
            else:
                query = query.eq("user_id", user_id)
            
            result = query.order("updated_at", desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            
            schemas = []
            for item in result.data:
                schemas.append(StoredSchema(**item))
            
            return schemas
            
        except Exception as e:
            logger.error(f"Error listing schemas for user {user_id}: {str(e)}")
            return []
    
    async def create_schema(
        self, 
        user_id: str, 
        request: CreateSchemaRequest
    ) -> Optional[StoredSchema]:
        """Create a new schema"""
        
        try:
            schema_id = str(uuid.uuid4())
            
            data = {
                "id": schema_id,
                "user_id": user_id,
                "title": request.title,
                "description": request.description,
                "content": request.content,
                "domain": request.domain,
                "tags": request.tags,
                "is_public": request.is_public,
                "usage_count": 0,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table(self.schemas_table).insert(data).execute()
            
            if result.data:
                return StoredSchema(**result.data[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating schema for user {user_id}: {str(e)}")
            return None
    
    async def update_schema(
        self, 
        schema_id: str, 
        user_id: str, 
        request: UpdateSchemaRequest
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
            
            result = self.supabase.table(self.schemas_table)\
                .update(update_data)\
                .eq("id", schema_id)\
                .eq("user_id", user_id)\
                .execute()
            
            if result.data:
                return StoredSchema(**result.data[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating schema {schema_id}: {str(e)}")
            return None
    
    async def delete_schema(self, schema_id: str, user_id: str) -> bool:
        """Delete a schema"""
        
        try:
            result = self.supabase.table(self.schemas_table)\
                .delete()\
                .eq("id", schema_id)\
                .eq("user_id", user_id)\
                .execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Error deleting schema {schema_id}: {str(e)}")
            return False
    
    async def update_generated_schema(
        self,
        schema_id: str,
        user_id: str,
        updated_content: str,
        refinement_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a generated schema with refinements"""
        
        try:
            update_data = {
                "content": updated_content,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if refinement_metadata:
                # Get current context and merge with refinement metadata
                current = await self.get_schema(schema_id, user_id)
                if current:
                    context = getattr(current, 'context', {}) or {}
                    context.update({"refinement": refinement_metadata})
                    update_data["context"] = context
            
            result = self.supabase.table(self.schemas_table)\
                .update(update_data)\
                .eq("id", schema_id)\
                .eq("user_id", user_id)\
                .execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Error updating generated schema {schema_id}: {str(e)}")
            return False
    
    async def increment_usage_count(self, schema_id: str) -> bool:
        """Increment the usage count for a schema"""
        
        try:
            # Use a stored procedure or direct SQL for atomic increment
            result = self.supabase.rpc(
                'increment_schema_usage',
                {'schema_id': schema_id}
            ).execute()
            
            # Fallback to read-modify-write if stored procedure doesn't exist
            if not result.data:
                schema = self.supabase.table(self.schemas_table)\
                    .select("usage_count")\
                    .eq("id", schema_id)\
                    .single()\
                    .execute()
                
                if schema.data:
                    new_count = (schema.data.get('usage_count', 0) or 0) + 1
                    result = self.supabase.table(self.schemas_table)\
                        .update({"usage_count": new_count})\
                        .eq("id", schema_id)\
                        .execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Error incrementing usage count for schema {schema_id}: {str(e)}")
            return False
    
    async def log_usage_event(
        self,
        schema_id: str,
        user_id: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log a schema usage event"""
        
        try:
            data = {
                "id": str(uuid.uuid4()),
                "schema_id": schema_id,
                "user_id": user_id,
                "event_type": event_type,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table(self.usage_table).insert(data).execute()
            
            # Also increment usage count
            if result.data:
                await self.increment_usage_count(schema_id)
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Error logging usage event for schema {schema_id}: {str(e)}")
            return False
    
    async def get_schema_analytics(
        self, 
        schema_id: str, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get analytics for a schema"""
        
        try:
            # Get schema info
            schema = await self.get_schema(schema_id, user_id)
            if not schema:
                return None
            
            # Get recent usage events
            usage_result = self.supabase.table(self.usage_table)\
                .select("*")\
                .eq("schema_id", schema_id)\
                .order("timestamp", desc=True)\
                .limit(20)\
                .execute()
            
            usage_events = [SchemaUsageEvent(**event) for event in usage_result.data]
            
            # Calculate unique users
            unique_users = len(set(event.user_id for event in usage_events))
            
            # Calculate popularity score (simple algorithm)
            popularity_score = min(schema.usage_count * 0.1 + unique_users * 0.5, 10.0)
            
            return {
                "schema_id": schema_id,
                "total_usage": schema.usage_count,
                "unique_users": unique_users,
                "recent_usage": usage_events,
                "popularity_score": popularity_score
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics for schema {schema_id}: {str(e)}")
            return None
    
    async def get_popular_schemas(
        self, 
        domain: Optional[str] = None, 
        limit: int = 10
    ) -> List[StoredSchema]:
        """Get popular public schemas"""
        
        try:
            query = self.supabase.table(self.schemas_table)\
                .select("*")\
                .eq("is_public", True)
            
            if domain:
                query = query.eq("domain", domain)
            
            result = query.order("usage_count", desc=True)\
                .limit(limit)\
                .execute()
            
            schemas = []
            for item in result.data:
                schemas.append(StoredSchema(**item))
            
            return schemas
            
        except Exception as e:
            logger.error(f"Error getting popular schemas: {str(e)}")
            return []


# Global service instance
schema_storage_service = SchemaStorageService()