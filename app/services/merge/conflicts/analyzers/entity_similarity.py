"""Entity similarity analyzer"""
import json
from typing import Dict, Any

from app.baml_client import b
from app.baml_client.types import EntitySimilarityAnalysis
from app.schemas.graph import Node
from ..base import ConflictAnalyzer

class EntitySimilarityAnalyzer(ConflictAnalyzer):
    """Analyzer for entity similarity"""
    
    async def analyze(
        self,
        staging_entity: Node,
        production_entity: Node
    ) -> EntitySimilarityAnalysis:
        """Analyze similarity between two entities using BAML
        
        Args:
            staging_entity: Entity from staging graph
            production_entity: Entity from production graph
            
        Returns:
            Analysis results including similarity scores and recommendations
        """
        return b.AnalyzeEntitySimilarity(
            entity_type=staging_entity.label,
            staging_properties=json.dumps(staging_entity.properties, indent=2),
            production_properties=json.dumps(production_entity.properties, indent=2),
            domain_context=f"Entity type: {staging_entity.label}"
        )
