from typing import List, Dict, Any, Optional, Tuple
import redis
import json
import uuid
from datetime import datetime, timedelta
import numpy as np
from app.schemas.conflicts import Conflict, ConflictType
from app.schemas.resolution_history import ResolutionHistoryEntry, ResolutionFilter, PaginationParams
from app.config import settings

class ResolutionHistoryService:
    """Service for tracking and learning from resolution history"""
    
    def __init__(self):
        """Initialize service with Redis connection"""
        self.redis = redis.Redis.from_url(settings.REDIS_URL)
        
    async def store_resolution(
        self,
        conflict: Conflict,
        resolution_id: str,
        applied_by: str,
        merge_id: str,
        success: bool = True,
        feedback: Optional[str] = None,
        effectiveness: Optional[float] = None
    ) -> ResolutionHistoryEntry:
        """Store a resolution for learning"""
        # Get chosen resolution
        chosen_resolution = next(
            (r for r in conflict.resolution_options if r.id == resolution_id),
            None
        )
        
        if not chosen_resolution:
            raise ValueError(f"Resolution {resolution_id} not found in conflict {conflict.id}")
        
        # Extract entity types, property names, and relationship types from context
        entity_types = []
        property_names = []
        relationship_types = []
        
        # Add staging and production entity types
        if conflict.entity_type:
            entity_types.append(conflict.entity_type)
        elif "entity_type" in conflict.context:
            entity_types.append(conflict.context["entity_type"])
            
        # Add property names for property conflicts
        if conflict.conflict_type in [ConflictType.PROPERTY_VALUE, ConflictType.PROPERTY_MISSING]:
            if conflict.property_name:
                property_names.append(conflict.property_name)
            elif "property_name" in conflict.context:
                property_names.append(conflict.context["property_name"])
                
        # Add relationship types for relationship conflicts
        if conflict.conflict_type in [ConflictType.RELATIONSHIP_TYPE, ConflictType.RELATIONSHIP_DIRECTION]:
            if "staging_type" in conflict.context:
                relationship_types.append(conflict.context["staging_type"])
            if "production_type" in conflict.context:
                relationship_types.append(conflict.context["production_type"])
        
        # Create history entry
        entry = ResolutionHistoryEntry(
            id=str(uuid.uuid4()),
            conflict_id=conflict.id,
            merge_id=merge_id,
            conflict_type=conflict.conflict_type,
            severity=conflict.severity,
            context=conflict.context,
            resolution_id=resolution_id,
            resolution_type=chosen_resolution.resolution_type,
            resolution_data=chosen_resolution.resolution_data,
            entity_types=entity_types,
            property_names=property_names,
            relationship_types=relationship_types,
            applied_by=applied_by,
            applied_at=datetime.now(),
            success=success,
            feedback=feedback,
            effectiveness=effectiveness
        )
        
        # Generate vector embedding (simplified for illustration)
        entry.vector_embedding = self._generate_embedding(entry)
        
        # Store in Redis
        self.redis.set(
            f"resolution_history:{entry.id}",
            entry.model_dump_json()
        )
        
        # Add to indexes for quick retrieval
        self._add_to_indexes(entry)
        
        return entry
    
    def _generate_embedding(self, entry: ResolutionHistoryEntry) -> List[float]:
        """
        Generate a vector embedding for similarity search
        This is a simplified version - in production, use a proper embedding model
        """
        # In real implementation, you would use a text embedding model
        # For illustration, we use a simple approach based on features
        
        # Convert enum values to integers
        conflict_type_val = list(ConflictType).index(entry.conflict_type)
        
        # For property conflicts, use property name as a feature
        property_feature = hash(entry.property_names[0]) % 100 if entry.property_names else 0
        
        # For relationship conflicts, use relationship types as features
        relationship_feature = sum(hash(rt) % 100 for rt in entry.relationship_types) if entry.relationship_types else 0
        
        # Combine features into a simple vector
        # In a real implementation, this would be a higher-dimensional vector from a text embedding model
        embedding = [
            conflict_type_val / 10.0,
            property_feature / 100.0,
            relationship_feature / 100.0
        ]
        
        return embedding
    
    def _add_to_indexes(self, entry: ResolutionHistoryEntry):
        """Add entry to various indexes for efficient retrieval"""
        # Add to conflict type index
        self.redis.sadd(
            f"resolution_index:conflict_type:{entry.conflict_type.value}",
            entry.id
        )
        
        # Add to entity type index
        for entity_type in entry.entity_types:
            self.redis.sadd(
                f"resolution_index:entity_type:{entity_type}",
                entry.id
            )
        
        # Add to property name index
        for prop_name in entry.property_names:
            self.redis.sadd(
                f"resolution_index:property_name:{prop_name}",
                entry.id
            )
        
        # Add to relationship type index
        for rel_type in entry.relationship_types:
            self.redis.sadd(
                f"resolution_index:relationship_type:{rel_type}",
                entry.id
            )
            
        # Add to resolution type index
        self.redis.sadd(
            f"resolution_index:resolution_type:{entry.resolution_type}",
            entry.id
        )
        
        # Add to user index
        self.redis.sadd(
            f"resolution_index:user:{entry.applied_by}",
            entry.id
        )
        
        # Add to merge ID index
        self.redis.sadd(
            f"resolution_index:merge_id:{entry.merge_id}",
            entry.id
        )
        
        # Add to date index (by year-month)
        date_key = entry.applied_at.strftime("%Y-%m")
        self.redis.sadd(
            f"resolution_index:date:{date_key}",
            entry.id
        )
        
        # Store vector embedding in a format suitable for similarity search
        # In a production system, you might use a vector database like Qdrant, Pinecone, or Redis with RediSearch
        self.redis.hset(
            "resolution_embeddings",
            entry.id,
            json.dumps(entry.vector_embedding)
        )
    
    async def find_similar_resolutions(
        self,
        conflict: Conflict,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar past resolutions for a given conflict"""
        # Strategy: First filter by conflict type and entity types to reduce search space
        # Then perform vector similarity search on the reduced set
        
        candidates = set()
        
        # Filter by conflict type
        type_matches = self.redis.smembers(
            f"resolution_index:conflict_type:{conflict.conflict_type.value}"
        )
        candidates.update(type_matches)
        
        # Filter by entity type if available
        entity_type = None
        if conflict.entity_type:
            entity_type = conflict.entity_type
        elif "entity_type" in conflict.context:
            entity_type = conflict.context["entity_type"]
            
        if entity_type:
            entity_matches = self.redis.smembers(
                f"resolution_index:entity_type:{entity_type}"
            )
            # If we already have candidates, intersect with entity matches
            if candidates:
                candidates &= entity_matches
            else:
                candidates.update(entity_matches)
        
        # Filter by property name for property conflicts
        if conflict.conflict_type in [ConflictType.PROPERTY_VALUE, ConflictType.PROPERTY_MISSING]:
            prop_name = None
            if conflict.property_name:
                prop_name = conflict.property_name
            elif "property_name" in conflict.context:
                prop_name = conflict.context["property_name"]
                
            if prop_name:
                prop_matches = self.redis.smembers(
                    f"resolution_index:property_name:{prop_name}"
                )
                if candidates:
                    candidates &= prop_matches
                else:
                    candidates.update(prop_matches)
        
        # If no candidates found, try broader search
        if not candidates:
            # Get all entries for this conflict type
            candidates = self.redis.smembers(
                f"resolution_index:conflict_type:{conflict.conflict_type.value}"
            )
        
        # Convert to list of strings
        candidate_ids = [c.decode('utf-8') if isinstance(c, bytes) else c for c in candidates]
        
        if not candidate_ids:
            return []
            
        # Get embeddings for candidates
        candidate_embeddings = {}
        for candidate_id in candidate_ids:
            embedding_json = self.redis.hget("resolution_embeddings", candidate_id)
            if embedding_json:
                candidate_embeddings[candidate_id] = json.loads(embedding_json)
        
        # Generate query embedding
        query_embedding = self._generate_embedding_for_conflict(conflict)
        
        # Calculate similarity scores
        similarity_scores = {}
        for candidate_id, embedding in candidate_embeddings.items():
            similarity_scores[candidate_id] = self._calculate_similarity(
                query_embedding, embedding
            )
        
        # Sort by similarity score
        sorted_candidates = sorted(
            similarity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        # Get full entries for top candidates
        results = []
        for candidate_id, score in sorted_candidates:
            entry_json = self.redis.get(f"resolution_history:{candidate_id}")
            if entry_json:
                entry = ResolutionHistoryEntry.model_validate_json(entry_json)
                results.append({
                    "entry": entry.model_dump(),
                    "similarity_score": score
                })
        
        return results
    
    def _generate_embedding_for_conflict(self, conflict: Conflict) -> List[float]:
        """Generate embedding for a conflict"""
        # Create a temporary history entry to generate embedding
        entity_types = []
        property_names = []
        relationship_types = []
        
        if conflict.entity_type:
            entity_types.append(conflict.entity_type)
        elif "entity_type" in conflict.context:
            entity_types.append(conflict.context["entity_type"])
            
        if conflict.conflict_type in [ConflictType.PROPERTY_VALUE, ConflictType.PROPERTY_MISSING]:
            if conflict.property_name:
                property_names.append(conflict.property_name)
            elif "property_name" in conflict.context:
                property_names.append(conflict.context["property_name"])
                
        if conflict.conflict_type in [ConflictType.RELATIONSHIP_TYPE, ConflictType.RELATIONSHIP_DIRECTION]:
            if "staging_type" in conflict.context:
                relationship_types.append(conflict.context["staging_type"])
            if "production_type" in conflict.context:
                relationship_types.append(conflict.context["production_type"])
        
        temp_entry = ResolutionHistoryEntry(
            id="temp",
            conflict_id=conflict.id,
            merge_id="temp",
            conflict_type=conflict.conflict_type,
            severity=conflict.severity,
            context=conflict.context,
            resolution_id="",
            resolution_type="",
            resolution_data={},
            entity_types=entity_types,
            property_names=property_names,
            relationship_types=relationship_types,
            applied_by="system"
        )
        
        return self._generate_embedding(temp_entry)
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # In real implementation, use a proper vector similarity calculation
        # For simple illustration, we use cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)
    
    async def get_resolution_history(
        self,
        merge_id: Optional[str] = None,
        conflict_type: Optional[ConflictType] = None,
        entity_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "applied_at",
        sort_order: str = "desc"
    ) -> List[ResolutionHistoryEntry]:
        """Get resolution history with optional filtering"""
        # Build filter sets
        filter_sets = []
        
        if merge_id:
            # Get all resolutions for this merge ID
            merge_entry_ids = self.redis.smembers(f"resolution_index:merge_id:{merge_id}")
            if merge_entry_ids:
                filter_sets.append({m.decode('utf-8') if isinstance(m, bytes) else m for m in merge_entry_ids})
            else:
                # If no entries found for this merge ID, return empty list
                return []
        
        if conflict_type:
            type_matches = self.redis.smembers(
                f"resolution_index:conflict_type:{conflict_type.value}"
            )
            filter_sets.append({m.decode('utf-8') if isinstance(m, bytes) else m for m in type_matches})
        
        if entity_type:
            entity_matches = self.redis.smembers(
                f"resolution_index:entity_type:{entity_type}"
            )
            filter_sets.append({m.decode('utf-8') if isinstance(m, bytes) else m for m in entity_matches})
        
        # Intersect all filter sets
        if filter_sets:
            result_ids = set.intersection(*filter_sets)
        else:
            # No filters, get all IDs
            all_entries = []
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor=cursor,
                    match=f"resolution_history:*",
                    count=1000
                )
                all_entries.extend([k.decode('utf-8').split(':')[1] if isinstance(k, bytes) else k.split(':')[1] for k in keys])
                if cursor == 0:
                    break
            result_ids = set(all_entries)
        
        # Get entries
        entries = []
        for entry_id in result_ids:
            entry_json = self.redis.get(f"resolution_history:{entry_id}")
            if entry_json:
                entries.append(
                    ResolutionHistoryEntry.model_validate_json(entry_json)
                )
        
        # Sort entries
        if sort_by == "applied_at":
            entries.sort(key=lambda e: e.applied_at, reverse=(sort_order.lower() == "desc"))
        elif sort_by == "effectiveness" and all(e.effectiveness is not None for e in entries):
            entries.sort(key=lambda e: e.effectiveness or 0.0, reverse=(sort_order.lower() == "desc"))
        elif sort_by == "severity":
            entries.sort(key=lambda e: e.severity.value, reverse=(sort_order.lower() == "desc"))
        
        # Apply pagination
        return entries[offset:offset+limit]
    
    async def get_resolution_count(self, merge_id: Optional[str] = None) -> int:
        """Get count of resolutions matching the filter"""
        if merge_id:
            # Get count of resolutions for this merge ID
            merge_entry_ids = self.redis.smembers(f"resolution_index:merge_id:{merge_id}")
            return len(merge_entry_ids)
        else:
            # Get count of all resolutions
            all_keys = self.redis.keys("resolution_history:*")
            return len(all_keys)
    
    async def filter_resolutions(
        self,
        filter_params: ResolutionFilter,
        pagination_params: PaginationParams
    ) -> Tuple[List[ResolutionHistoryEntry], int]:
        """Filter resolutions by various criteria with pagination and sorting"""
        # Build filter sets
        filter_sets = []
        
        # Filter by conflict type
        if filter_params.conflict_type:
            type_matches = self.redis.smembers(
                f"resolution_index:conflict_type:{filter_params.conflict_type.value}"
            )
            filter_sets.append({m.decode('utf-8') for m in type_matches})
        
        # Filter by resolution type
        if filter_params.resolution_type:
            type_matches = self.redis.smembers(
                f"resolution_index:resolution_type:{filter_params.resolution_type}"
            )
            filter_sets.append({m.decode('utf-8') for m in type_matches})
        
        # Filter by entity type
        if filter_params.entity_type:
            entity_matches = self.redis.smembers(
                f"resolution_index:entity_type:{filter_params.entity_type}"
            )
            filter_sets.append({m.decode('utf-8') for m in entity_matches})
        
        # Filter by property name
        if filter_params.property_name:
            prop_matches = self.redis.smembers(
                f"resolution_index:property_name:{filter_params.property_name}"
            )
            filter_sets.append({m.decode('utf-8') for m in prop_matches})
        
        # Filter by relationship type
        if filter_params.relationship_type:
            rel_matches = self.redis.smembers(
                f"resolution_index:relationship_type:{filter_params.relationship_type}"
            )
            filter_sets.append({m.decode('utf-8') for m in rel_matches})
        
        # Filter by user
        if filter_params.user:
            user_matches = self.redis.smembers(
                f"resolution_index:user:{filter_params.user}"
            )
            filter_sets.append({m.decode('utf-8') for m in user_matches})
        
        # Intersect all filter sets
        if filter_sets:
            result_ids = set.intersection(*filter_sets) if len(filter_sets) > 1 else filter_sets[0]
        else:
            # No filters, get all IDs
            all_entries = []
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor=cursor,
                    match=f"resolution_history:*",
                    count=1000
                )
                all_entries.extend([k.decode('utf-8').split(':')[1] if isinstance(k, bytes) else k.split(':')[1] for k in keys])
                if cursor == 0:
                    break
            result_ids = set(all_entries)
        
        # Get entries
        entries = []
        for entry_id in result_ids:
            entry_json = self.redis.get(f"resolution_history:{entry_id}")
            if entry_json:
                entry = ResolutionHistoryEntry.model_validate_json(entry_json)
                
                # Apply date filters
                if filter_params.start_date and entry.applied_at < filter_params.start_date:
                    continue
                if filter_params.end_date and entry.applied_at > filter_params.end_date:
                    continue
                
                # Apply effectiveness filter
                if filter_params.effectiveness is not None:
                    if entry.effectiveness is None or entry.effectiveness < filter_params.effectiveness:
                        continue
                
                # Apply success filter
                if filter_params.success is not None and entry.success != filter_params.success:
                    continue
                
                entries.append(entry)
        
        # Get total count for pagination
        total_count = len(entries)
        
        # Sort entries
        if pagination_params.sort_by == "applied_at":
            entries.sort(key=lambda e: e.applied_at, reverse=(pagination_params.sort_order.lower() == "desc"))
        elif pagination_params.sort_by == "effectiveness" and all(e.effectiveness is not None for e in entries):
            entries.sort(key=lambda e: e.effectiveness or 0.0, reverse=(pagination_params.sort_order.lower() == "desc"))
        elif pagination_params.sort_by == "severity":
            entries.sort(key=lambda e: e.severity.value, reverse=(pagination_params.sort_order.lower() == "desc"))
        
        # Apply pagination
        paginated_entries = entries[pagination_params.offset:pagination_params.offset+pagination_params.limit]
        
        return paginated_entries, total_count
            
    async def update_resolution_success(
        self,
        resolution_id: str,
        success: bool,
        feedback: Optional[str] = None,
        effectiveness: Optional[float] = None
    ) -> bool:
        """Update success status, feedback, and effectiveness for a resolution"""
        entry_json = self.redis.get(f"resolution_history:{resolution_id}")
        if not entry_json:
            return False
            
        entry = ResolutionHistoryEntry.model_validate_json(entry_json)
        entry.success = success
        if feedback:
            entry.feedback = feedback
        if effectiveness is not None:
            entry.effectiveness = effectiveness
            
        # Update in Redis
        self.redis.set(
            f"resolution_history:{entry.id}",
            entry.model_dump_json()
        )
        
        return True
        
    async def get_resolution_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get statistics about stored resolutions"""
        # Get all entries
        all_entries = []
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(
                cursor=cursor,
                match=f"resolution_history:*",
                count=1000
            )
            
            for key in keys:
                entry_json = self.redis.get(key)
                if entry_json:
                    entry = ResolutionHistoryEntry.model_validate_json(entry_json)
                    
                    # Apply date filters
                    if start_date and entry.applied_at < start_date:
                        continue
                    if end_date and entry.applied_at > end_date:
                        continue
                    
                    all_entries.append(entry)
            
            if cursor == 0:
                break
        
        # Count total entries
        total_count = len(all_entries)
        
        if total_count == 0:
            return {
                "total_resolutions": 0,
                "by_conflict_type": {},
                "by_resolution_type": {},
                "by_entity_type": {},
                "by_user": {},
                "success_count": 0,
                "success_rate": 0.0,
                "average_effectiveness": 0.0,
                "time_distribution": {}
            }
        
        # Count by conflict type
        type_counts = {}
        for entry in all_entries:
            conflict_type = entry.conflict_type.value
            type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
        
        # Count by resolution type
        resolution_type_counts = {}
        for entry in all_entries:
            resolution_type = entry.resolution_type
            resolution_type_counts[resolution_type] = resolution_type_counts.get(resolution_type, 0) + 1
        
        # Count by entity type
        entity_type_counts = {}
        for entry in all_entries:
            for entity_type in entry.entity_types:
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        # Count by user
        user_counts = {}
        for entry in all_entries:
            user = entry.applied_by
            user_counts[user] = user_counts.get(user, 0) + 1
        
        # Count success rate
        success_count = sum(1 for entry in all_entries if entry.success)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        # Calculate average effectiveness
        effectiveness_values = [entry.effectiveness for entry in all_entries if entry.effectiveness is not None]
        average_effectiveness = sum(effectiveness_values) / len(effectiveness_values) if effectiveness_values else 0.0
        
        # Count by time period (month)
        time_distribution = {}
        for entry in all_entries:
            month_key = entry.applied_at.strftime("%Y-%m")
            time_distribution[month_key] = time_distribution.get(month_key, 0) + 1
        
        return {
            "total_resolutions": total_count,
            "by_conflict_type": type_counts,
            "by_resolution_type": resolution_type_counts,
            "by_entity_type": entity_type_counts,
            "by_user": user_counts,
            "success_count": success_count,
            "success_rate": success_rate,
            "average_effectiveness": average_effectiveness,
            "time_distribution": time_distribution
        } 