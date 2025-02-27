"""Service for batch resolution of similar conflicts"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from app.schemas.conflicts import Conflict, ConflictType, ResolutionOption
from app.services.merge.service import MergeService

logger = logging.getLogger(__name__)

class BatchResolver:
    """Resolver for batches of similar conflicts"""
    
    def __init__(self, merge_service: MergeService):
        self.merge_service = merge_service
    
    async def group_similar_conflicts(
        self,
        merge_id: str,
        grouping_strategy: str = "type_and_entity",
        similarity_threshold: float = 0.8,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Conflict]]:
        """
        Group similar conflicts based on specified strategy
        
        Args:
            merge_id: Identifier for the merge process
            grouping_strategy: Strategy for grouping ('type_and_entity', 'property_name', etc.)
            similarity_threshold: Threshold for fuzzy matching (0.0-1.0)
            filters: Optional filters to apply before grouping
            
        Returns:
            Dictionary mapping group keys to lists of conflicts
        """
        # Get conflicts with optional filtering
        conflicts, _ = await self.merge_service.get_conflicts(
            merge_id=merge_id,
            **(filters or {})
        )
        
        # Group conflicts based on strategy
        if grouping_strategy == "type_and_entity":
            return self._group_by_type_and_entity(conflicts)
        elif grouping_strategy == "property_name":
            return self._group_by_property_name(conflicts)
        elif grouping_strategy == "fuzzy_match":
            return await self._group_by_fuzzy_match(conflicts, similarity_threshold)
        else:
            raise ValueError(f"Unknown grouping strategy: {grouping_strategy}")
    
    def _group_by_type_and_entity(self, conflicts: List[Conflict]) -> Dict[str, List[Conflict]]:
        """Group conflicts by type and entity type"""
        groups = {}
        
        for conflict in conflicts:
            # Skip resolved conflicts
            if conflict.resolved:
                continue
                
            # Extract entity type from context or use default
            entity_type = conflict.entity_type or conflict.context.get("entity_type", "unknown")
            
            # Create group key
            group_key = f"{conflict.conflict_type.value}:{entity_type}"
            
            if group_key not in groups:
                groups[group_key] = []
                
            groups[group_key].append(conflict)
        
        return groups
    
    def _group_by_property_name(self, conflicts: List[Conflict]) -> Dict[str, List[Conflict]]:
        """Group property conflicts by property name"""
        groups = {}
        
        for conflict in conflicts:
            # Skip resolved conflicts
            if conflict.resolved:
                continue
                
            # Only process property conflicts
            if conflict.conflict_type not in [ConflictType.PROPERTY_VALUE, ConflictType.PROPERTY_MISSING]:
                continue
                
            # Extract property name
            property_name = conflict.property_name or conflict.context.get("property_name", "unknown")
            
            # Create group key
            group_key = f"{conflict.conflict_type.value}:{property_name}"
            
            if group_key not in groups:
                groups[group_key] = []
                
            groups[group_key].append(conflict)
        
        return groups
    
    async def _group_by_fuzzy_match(
        self, 
        conflicts: List[Conflict],
        similarity_threshold: float
    ) -> Dict[str, List[Conflict]]:
        """Group conflicts by fuzzy matching their properties"""
        groups = {}
        processed_conflicts = set()
        
        for i, conflict1 in enumerate(conflicts):
            # Skip resolved or already processed conflicts
            if conflict1.resolved or conflict1.id in processed_conflicts:
                continue
                
            # Create new group
            group_key = f"group_{i}"
            groups[group_key] = [conflict1]
            processed_conflicts.add(conflict1.id)
            
            # Find similar conflicts
            for conflict2 in conflicts:
                if conflict2.id in processed_conflicts:
                    continue
                    
                # Check similarity
                similarity = self._calculate_conflict_similarity(conflict1, conflict2)
                if similarity >= similarity_threshold:
                    groups[group_key].append(conflict2)
                    processed_conflicts.add(conflict2.id)
        
        return groups
    
    def _calculate_conflict_similarity(self, conflict1: Conflict, conflict2: Conflict) -> float:
        """Calculate similarity between two conflicts (0.0-1.0)"""
        # Must be same conflict type
        if conflict1.conflict_type != conflict2.conflict_type:
            return 0.0
            
        # Must be same severity
        if conflict1.severity != conflict2.severity:
            return 0.0
            
        # Context similarity depends on conflict type
        if conflict1.conflict_type == ConflictType.PROPERTY_VALUE:
            # For property conflicts, check if property name is the same
            prop1 = conflict1.property_name or conflict1.context.get("property_name")
            prop2 = conflict2.property_name or conflict2.context.get("property_name")
            if prop1 == prop2:
                return 0.9  # High similarity for same property name
            return 0.0
            
        elif conflict1.conflict_type == ConflictType.RELATIONSHIP_TYPE:
            # For relationship conflicts, check staging and production types
            staging_type1 = conflict1.context.get("staging_type")
            staging_type2 = conflict2.context.get("staging_type")
            prod_type1 = conflict1.context.get("production_type")
            prod_type2 = conflict2.context.get("production_type")
            
            if staging_type1 == staging_type2 and prod_type1 == prod_type2:
                return 0.9
            elif staging_type1 == staging_type2 or prod_type1 == prod_type2:
                return 0.7
            return 0.3
            
        # Default similarity for other types
        return 0.5
    
    async def apply_batch_resolution(
        self,
        merge_id: str,
        group_key: str,
        resolution_option: ResolutionOption,
        exceptions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Apply a resolution option to all conflicts in a group
        
        Args:
            merge_id: Identifier for the merge process
            group_key: Key identifying the conflict group
            resolution_option: Resolution option to apply
            exceptions: List of conflict IDs to exclude from batch
            
        Returns:
            Summary of the resolution operation
        """
        exceptions = exceptions or []
        
        # Get all conflicts for the group
        all_groups = await self.group_similar_conflicts(merge_id)
        if group_key not in all_groups:
            return {
                "status": "error",
                "message": f"Group {group_key} not found",
                "resolved_count": 0
            }
            
        conflicts = all_groups[group_key]
        
        # Apply resolution to each conflict except exceptions
        resolved_count = 0
        for conflict in conflicts:
            if conflict.id in exceptions:
                continue
                
            # Create a compatible resolution option for this conflict
            conflict_option = self._adapt_resolution_option(resolution_option, conflict)
            
            # Apply resolution
            try:
                await self.merge_service.resolve_conflict(
                    merge_id=merge_id,
                    conflict_id=conflict.id,
                    resolution_id=conflict_option.id
                )
                resolved_count += 1
            except Exception as e:
                logger.error(f"Failed to resolve conflict {conflict.id}: {str(e)}")
        
        return {
            "status": "success",
            "group_key": group_key,
            "resolved_count": resolved_count,
            "total_in_group": len(conflicts),
            "exceptions_count": len(exceptions)
        }
    
    def _adapt_resolution_option(self, option: ResolutionOption, conflict: Conflict) -> ResolutionOption:
        """Adapt a resolution option to a specific conflict"""
        # Create a new option ID for this conflict
        new_option_id = f"{conflict.id}_{option.resolution_type}"
        
        # Copy resolution data and adapt to this conflict
        resolution_data = option.resolution_data.copy() if option.resolution_data else {}
        
        # Adapt based on conflict type
        if conflict.conflict_type == ConflictType.PROPERTY_VALUE:
            # Set property name from conflict context
            property_name = conflict.property_name or conflict.context.get("property_name")
            if property_name:
                resolution_data["property_name"] = property_name
        
        # Create new option
        return ResolutionOption(
            id=new_option_id,
            description=option.description,
            resolution_type=option.resolution_type,
            resolution_data=resolution_data,
            confidence=option.confidence,
            reasoning=option.reasoning,
            requires_review=option.requires_review,
            auto_resolvable=option.auto_resolvable
        ) 