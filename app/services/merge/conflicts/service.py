"""Main merge service for conflict detection and resolution"""
import logging
from typing import List, Dict, Optional, Any
from redis import Redis

from app.schemas.conflicts import Conflict, ConflictFilter, ConflictGroup
from app.schemas.graph import GraphResponse
from app.services.storage.interface import GraphStorageInterface

from app.services.merge.conflicts.detectors.entity_matching import EntityMatchingDetector
from app.services.merge.conflicts.detectors.property import PropertyConflictDetector
from app.services.merge.conflicts.detectors.relationship import RelationshipConflictDetector

logger = logging.getLogger(__name__)

class MergeService:
    """Service for detecting and managing merge conflicts"""
    
    def __init__(
        self,
        storage: GraphStorageInterface,
        redis_client: Optional[Redis] = None
    ):
        """Initialize merge service
        
        Args:
            storage: Graph storage interface
            redis_client: Optional Redis client for caching
        """
        self.storage = storage
        self.redis = redis_client
        
        # Initialize detectors
        self.entity_detector = EntityMatchingDetector(storage)
        self.property_detector = PropertyConflictDetector(storage)
        self.relationship_detector = RelationshipConflictDetector(storage)
    
    async def detect_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]],
        filters: Optional[ConflictFilter] = None
    ) -> List[Conflict]:
        """Detect all conflicts between staging and production
        
        Args:
            staging_graph: Graph containing staging changes
            production_entity_mapping: Mapping of staging IDs to production matches
            filters: Optional filters to apply
            
        Returns:
            List of detected conflicts
        """
        all_conflicts = []
        
        # Detect entity matching conflicts
        entity_conflicts = await self.entity_detector.detect_conflicts(
            staging_graph,
            production_entity_mapping
        )
        all_conflicts.extend(entity_conflicts)
        
        # Detect property conflicts
        property_conflicts = await self.property_detector.detect_conflicts(
            staging_graph,
            production_entity_mapping
        )
        all_conflicts.extend(property_conflicts)
        
        # Detect relationship conflicts
        relationship_conflicts = await self.relationship_detector.detect_conflicts(
            staging_graph,
            production_entity_mapping
        )
        all_conflicts.extend(relationship_conflicts)
        
        # Apply filters if provided
        if filters:
            all_conflicts = self._apply_filters(all_conflicts, filters)
        
        return all_conflicts
    
    def _apply_filters(
        self,
        conflicts: List[Conflict],
        filters: ConflictFilter
    ) -> List[Conflict]:
        """Apply filters to conflicts
        
        Args:
            conflicts: List of conflicts to filter
            filters: Filters to apply
            
        Returns:
            Filtered list of conflicts
        """
        filtered = conflicts
        
        if filters.conflict_types:
            filtered = [
                c for c in filtered
                if c.conflict_type in filters.conflict_types
            ]
            
        if filters.severities:
            filtered = [
                c for c in filtered
                if c.severity in filters.severities
            ]
            
        if filters.entity_types:
            filtered = [
                c for c in filtered
                if any(
                    node.label in filters.entity_types
                    for node in c.staging_nodes + c.production_nodes
                )
            ]
            
        return filtered
    
    async def group_conflicts(
        self,
        conflicts: List[Conflict]
    ) -> List[ConflictGroup]:
        """Group related conflicts together
        
        Args:
            conflicts: List of conflicts to group
            
        Returns:
            List of conflict groups
        """
        # Group conflicts by entity type
        groups = {}
        for conflict in conflicts:
            entity_type = conflict.staging_nodes[0].label if conflict.staging_nodes else "Unknown"
            if entity_type not in groups:
                groups[entity_type] = []
            groups[entity_type].append(conflict)
        
        # Create conflict groups
        conflict_groups = []
        for entity_type, group_conflicts in groups.items():
            # Get common patterns in the group
            pattern = self._get_group_pattern(group_conflicts)
            
            # Check if conflicts can be batch resolved
            batch_resolvable = self._are_conflicts_batch_resolvable(group_conflicts)
            
            # Get recommended strategy if batch resolvable
            recommended_strategy = self._get_recommended_strategy(group_conflicts) if batch_resolvable else None
            
            # Calculate confidence for the recommendation
            confidence = self._calculate_group_confidence(group_conflicts) if recommended_strategy else 0.0
            
            conflict_groups.append(
                ConflictGroup(
                    id=f"group_{entity_type}",
                    entity_type=entity_type,
                    property_name=pattern.get("property_name", ""),
                    value_type=pattern.get("value_type", ""),
                    conflict_ids=[c.id for c in group_conflicts],
                    total_conflicts=len(group_conflicts),
                    pattern=pattern.get("pattern", ""),
                    batch_resolvable=batch_resolvable,
                    recommended_strategy=recommended_strategy,
                    confidence=confidence
                )
            )
        
        return conflict_groups
    
    def _get_group_pattern(self, conflicts: List[Conflict]) -> Dict[str, Any]:
        """Extract common patterns from a group of conflicts"""
        if not conflicts:
            return {}
            
        # Get the first conflict as reference
        ref = conflicts[0]
        
        pattern = {
            "conflict_type": ref.conflict_type,
            "severity": ref.severity
        }
        
        # Check if all conflicts share the same pattern
        for conflict in conflicts[1:]:
            if conflict.conflict_type != pattern["conflict_type"]:
                pattern["conflict_type"] = None
            if conflict.severity != pattern["severity"]:
                pattern["severity"] = None
                
        # Add property info for property conflicts
        if ref.conflict_type == "PROPERTY_CONFLICT":
            property_name = ref.context.get("property_name")
            value_type = ref.context.get("value_type")
            
            # Check if property info is consistent
            for conflict in conflicts[1:]:
                if conflict.context.get("property_name") != property_name:
                    property_name = None
                if conflict.context.get("value_type") != value_type:
                    value_type = None
                    
            if property_name:
                pattern["property_name"] = property_name
            if value_type:
                pattern["value_type"] = value_type
        
        return pattern
    
    def _are_conflicts_batch_resolvable(self, conflicts: List[Conflict]) -> bool:
        """Check if conflicts can be resolved in batch"""
        if not conflicts:
            return False
            
        # Get reference conflict
        ref = conflicts[0]
        
        # Check if all conflicts have same type and resolution options
        return all(
            conflict.conflict_type == ref.conflict_type and
            len(conflict.resolution_options) == len(ref.resolution_options) and
            all(
                opt1.resolution_type == opt2.resolution_type
                for opt1, opt2 in zip(
                    conflict.resolution_options,
                    ref.resolution_options
                )
            )
            for conflict in conflicts[1:]
        )
    
    def _get_recommended_strategy(self, conflicts: List[Conflict]) -> Optional[str]:
        """Get recommended resolution strategy for batch resolution"""
        if not conflicts:
            return None
            
        # Get most common resolution type with highest confidence
        resolution_counts = {}
        resolution_confidences = {}
        
        for conflict in conflicts:
            for option in conflict.resolution_options:
                if option.resolution_type not in resolution_counts:
                    resolution_counts[option.resolution_type] = 0
                    resolution_confidences[option.resolution_type] = 0.0
                    
                resolution_counts[option.resolution_type] += 1
                resolution_confidences[option.resolution_type] += option.confidence
        
        # Get strategy with highest count and average confidence
        max_count = max(resolution_counts.values())
        best_strategies = [
            strategy for strategy, count in resolution_counts.items()
            if count == max_count
        ]
        
        if not best_strategies:
            return None
            
        # Return strategy with highest confidence
        return max(
            best_strategies,
            key=lambda s: resolution_confidences[s] / resolution_counts[s]
        )
    
    def _calculate_group_confidence(self, conflicts: List[Conflict]) -> float:
        """Calculate confidence score for group resolution"""
        if not conflicts:
            return 0.0
            
        # Average the highest confidence resolution option from each conflict
        confidences = [
            max(opt.confidence for opt in conflict.resolution_options)
            for conflict in conflicts
        ]
        
        return sum(confidences) / len(confidences)
