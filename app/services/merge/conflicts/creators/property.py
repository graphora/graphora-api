"""Property conflict creator"""
from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionOption,
    ResolutionStrategy
)
from app.schemas.graph import Node
from app.baml_client.types import PropertyConflictAnalysis
from ..base import ConflictCreator

class PropertyConflictCreator(ConflictCreator):
    """Creator for property conflicts"""
    
    async def create_conflict(
        self,
        conflict_id: str,
        staging_entity: Node,
        production_entity: Node,
        analysis: PropertyConflictAnalysis
    ) -> Conflict:
        """Create a property conflict
        
        Args:
            conflict_id: Unique conflict identifier
            staging_entity: Entity from staging
            production_entity: Entity from production
            analysis: Analysis of property conflicts
            
        Returns:
            Property conflict with resolution options
        """
        resolution_options = []
        
        # Add resolution options based on analysis
        for prop_conflict in analysis.property_conflicts:
            # Option to keep staging value
            resolution_options.append(
                ResolutionOption(
                    id=f"{conflict_id}_keep_staging_{prop_conflict.property_name}",
                    description=f"Keep staging value for {prop_conflict.property_name}",
                    resolution_type=ResolutionStrategy.KEEP_STAGING,
                    resolution_data={
                        "property_name": prop_conflict.property_name,
                        "value": prop_conflict.staging_value
                    },
                    confidence=prop_conflict.staging_confidence
                )
            )
            
            # Option to keep production value
            resolution_options.append(
                ResolutionOption(
                    id=f"{conflict_id}_keep_prod_{prop_conflict.property_name}",
                    description=f"Keep production value for {prop_conflict.property_name}",
                    resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                    resolution_data={
                        "property_name": prop_conflict.property_name,
                        "value": prop_conflict.production_value
                    },
                    confidence=prop_conflict.production_confidence
                )
            )
            
            # Option to merge values if recommended
            if prop_conflict.can_merge:
                resolution_options.append(
                    ResolutionOption(
                        id=f"{conflict_id}_merge_{prop_conflict.property_name}",
                        description=f"Merge values for {prop_conflict.property_name}",
                        resolution_type=ResolutionStrategy.MERGE_VALUES,
                        resolution_data={
                            "property_name": prop_conflict.property_name,
                            "merged_value": prop_conflict.merged_value
                        },
                        confidence=prop_conflict.merge_confidence
                    )
                )
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=[staging_entity.id],
            production_ids=[production_entity.id],
            description="Property value conflicts detected",
            context={
                "staging_entity": {
                    "id": staging_entity.id,
                    "label": staging_entity.label,
                    "properties": staging_entity.properties
                },
                "production_entity": {
                    "id": production_entity.id,
                    "label": production_entity.label,
                    "properties": production_entity.properties
                },
                "property_conflicts": [
                    {
                        "property_name": pc.property_name,
                        "staging_value": pc.staging_value,
                        "production_value": pc.production_value,
                        "can_merge": pc.can_merge,
                        "merged_value": pc.merged_value,
                        "analysis": pc.analysis
                    }
                    for pc in analysis.property_conflicts
                ]
            },
            resolution_options=resolution_options
        )
