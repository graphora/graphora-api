"""Relationship conflict analyzer"""
import json

from app.baml_client import b
from app.baml_client.types import RelationshipConflictAnalysis
from app.schemas.conflicts import ConflictType
from app.schemas.graph import Edge
from ..base import ConflictAnalyzer

class RelationshipConflictAnalyzer(ConflictAnalyzer):
    """Analyzer for relationship conflicts"""
    
    async def analyze(
        self,
        staging_relationship: Edge,
        production_relationship: Edge
    ) -> RelationshipConflictAnalysis:
        """Analyze conflicts between relationships
        
        Args:
            staging_relationship: Relationship from staging
            production_relationship: Relationship from production
            
        Returns:
            Analysis of relationship conflicts
        """
        # Determine conflict type based on differences
        if staging_relationship.type != production_relationship.type:
            conflict_type = ConflictType.RELATIONSHIP_TYPE
        elif staging_relationship.source_id != production_relationship.source_id or \
             staging_relationship.target_id != production_relationship.target_id:
            conflict_type = ConflictType.RELATIONSHIP_DIRECTION
        else:
            conflict_type = ConflictType.RELATIONSHIP_PROPERTY
            
        # Prepare staging details
        staging_details = {
            "type": staging_relationship.type,
            "source_id": staging_relationship.source_id,
            "target_id": staging_relationship.target_id,
            "properties": staging_relationship.properties
        }
        
        # Prepare production details
        production_details = {
            "type": production_relationship.type,
            "source_id": production_relationship.source_id,
            "target_id": production_relationship.target_id,
            "properties": production_relationship.properties
        }
        
        # Prepare graph context
        graph_context = {
            "staging_source_type": staging_relationship.source_type,
            "staging_target_type": staging_relationship.target_type,
            "production_source_type": production_relationship.source_type,
            "production_target_type": production_relationship.target_type,
            "relationship_type": staging_relationship.type
        }
        
        return b.AnalyzeRelationshipConflict(
            relationship_type=staging_relationship.type,
            staging_details=json.dumps(staging_details),
            production_details=json.dumps(production_details),
            conflict_type=conflict_type,
            graph_context=json.dumps(graph_context)
        )
