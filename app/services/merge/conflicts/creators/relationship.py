"""Relationship conflict creator"""
from typing import Dict, Any, List

from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionOption,
    ResolutionStrategy
)
from app.schemas.graph import Edge
from app.baml_client.types import RelationshipConflictAnalysis
from ..base import ConflictCreator

class RelationshipConflictCreator(ConflictCreator):
    """Creator for relationship conflicts"""
    
    async def create_conflict(
        self,
        conflict_id: str,
        staging_relationship: Edge,
        production_relationship: Edge,
        analysis: RelationshipConflictAnalysis
    ) -> Conflict:
        """Create a relationship conflict
        
        Args:
            conflict_id: Unique conflict identifier
            staging_relationship: Relationship from staging
            production_relationship: Relationship from production
            analysis: Analysis of relationship conflicts
            
        Returns:
            Relationship conflict with resolution options
        """
        resolution_options = []
        
        # Add resolution options based on conflict type
        if analysis.type_conflict:
            conflict_type = ConflictType.RELATIONSHIP_TYPE
            resolution_options.extend(self._create_type_conflict_options(
                conflict_id,
                staging_relationship,
                production_relationship
            ))
            
        if analysis.direction_conflict:
            conflict_type = ConflictType.RELATIONSHIP_DIRECTION
            resolution_options.extend(self._create_direction_conflict_options(
                conflict_id,
                staging_relationship,
                production_relationship
            ))
            
        if analysis.property_conflicts:
            conflict_type = ConflictType.RELATIONSHIP_PROPERTY
            resolution_options.extend(self._create_property_conflict_options(
                conflict_id,
                staging_relationship,
                production_relationship,
                analysis
            ))
        
        return Conflict(
            id=conflict_id,
            conflict_type=conflict_type,
            severity=ConflictSeverity.CRITICAL,
            staging_ids=[staging_relationship.id],
            production_ids=[production_relationship.id],
            description=analysis.description,
            context={
                "staging_relationship": {
                    "id": staging_relationship.id,
                    "type": staging_relationship.type,
                    "source_id": staging_relationship.source_id,
                    "target_id": staging_relationship.target_id,
                    "properties": staging_relationship.properties
                },
                "production_relationship": {
                    "id": production_relationship.id,
                    "type": production_relationship.type,
                    "source_id": production_relationship.source_id,
                    "target_id": production_relationship.target_id,
                    "properties": production_relationship.properties
                },
                "analysis": {
                    "type_conflict": analysis.type_conflict,
                    "direction_conflict": analysis.direction_conflict,
                    "property_conflicts": analysis.property_conflicts,
                    "impact_analysis": analysis.impact_analysis
                }
            },
            resolution_options=resolution_options
        )
    
    async def create_missing_conflict(
        self,
        conflict_id: str,
        staging_relationship: Edge,
        production_source_id: str,
        production_target_id: str
    ) -> Conflict:
        """Create a missing relationship conflict
        
        Args:
            conflict_id: Unique conflict identifier
            staging_relationship: Missing relationship from staging
            production_source_id: Production source entity ID
            production_target_id: Production target entity ID
            
        Returns:
            Missing relationship conflict
        """
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.RELATIONSHIP_MISSING,
            severity=ConflictSeverity.MAJOR,
            staging_ids=[staging_relationship.id],
            production_ids=[],
            description="Relationship exists in staging but not in production",
            context={
                "staging_relationship": {
                    "id": staging_relationship.id,
                    "type": staging_relationship.type,
                    "source_id": staging_relationship.source_id,
                    "target_id": staging_relationship.target_id,
                    "properties": staging_relationship.properties
                },
                "production_entities": {
                    "source_id": production_source_id,
                    "target_id": production_target_id
                }
            },
            resolution_options=[
                ResolutionOption(
                    id=f"{conflict_id}_create",
                    description="Create relationship in production",
                    resolution_type=ResolutionStrategy.CREATE_NEW,
                    resolution_data={
                        "relationship_type": staging_relationship.type,
                        "source_id": production_source_id,
                        "target_id": production_target_id,
                        "properties": staging_relationship.properties
                    },
                    confidence=0.8
                ),
                ResolutionOption(
                    id=f"{conflict_id}_ignore",
                    description="Ignore missing relationship",
                    resolution_type=ResolutionStrategy.IGNORE,
                    resolution_data={},
                    confidence=0.2
                )
            ]
        )
    
    def _create_type_conflict_options(
        self,
        conflict_id: str,
        staging_rel: Edge,
        prod_rel: Edge
    ) -> List[ResolutionOption]:
        """Create resolution options for type conflicts"""
        return [
            ResolutionOption(
                id=f"{conflict_id}_type_staging",
                description=f"Use staging relationship type: {staging_rel.type}",
                resolution_type=ResolutionStrategy.KEEP_STAGING,
                resolution_data={"type": staging_rel.type},
                confidence=0.6
            ),
            ResolutionOption(
                id=f"{conflict_id}_type_prod",
                description=f"Keep production relationship type: {prod_rel.type}",
                resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                resolution_data={"type": prod_rel.type},
                confidence=0.4
            )
        ]
    
    def _create_direction_conflict_options(
        self,
        conflict_id: str,
        staging_rel: Edge,
        prod_rel: Edge
    ) -> List[ResolutionOption]:
        """Create resolution options for direction conflicts"""
        return [
            ResolutionOption(
                id=f"{conflict_id}_dir_staging",
                description="Use staging relationship direction",
                resolution_type=ResolutionStrategy.KEEP_STAGING,
                resolution_data={
                    "source_id": staging_rel.source_id,
                    "target_id": staging_rel.target_id
                },
                confidence=0.6
            ),
            ResolutionOption(
                id=f"{conflict_id}_dir_prod",
                description="Keep production relationship direction",
                resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                resolution_data={
                    "source_id": prod_rel.source_id,
                    "target_id": prod_rel.target_id
                },
                confidence=0.4
            )
        ]
    
    def _create_property_conflict_options(
        self,
        conflict_id: str,
        staging_rel: Edge,
        prod_rel: Edge,
        analysis: RelationshipConflictAnalysis
    ) -> List[ResolutionOption]:
        """Create resolution options for property conflicts"""
        options = []
        
        for prop_conflict in analysis.property_conflicts:
            # Option to keep staging property
            options.append(
                ResolutionOption(
                    id=f"{conflict_id}_prop_staging_{prop_conflict.property_name}",
                    description=f"Use staging value for {prop_conflict.property_name}",
                    resolution_type=ResolutionStrategy.KEEP_STAGING,
                    resolution_data={
                        "property_name": prop_conflict.property_name,
                        "value": prop_conflict.staging_value
                    },
                    confidence=0.6
                )
            )
            
            # Option to keep production property
            options.append(
                ResolutionOption(
                    id=f"{conflict_id}_prop_prod_{prop_conflict.property_name}",
                    description=f"Keep production value for {prop_conflict.property_name}",
                    resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                    resolution_data={
                        "property_name": prop_conflict.property_name,
                        "value": prop_conflict.production_value
                    },
                    confidence=0.4
                )
            )
            
            # Option to merge if possible
            if prop_conflict.can_merge:
                options.append(
                    ResolutionOption(
                        id=f"{conflict_id}_prop_merge_{prop_conflict.property_name}",
                        description=f"Merge values for {prop_conflict.property_name}",
                        resolution_type=ResolutionStrategy.MERGE_VALUES,
                        resolution_data={
                            "property_name": prop_conflict.property_name,
                            "merged_value": prop_conflict.merged_value
                        },
                        confidence=0.8
                    )
                )
        
        return options
