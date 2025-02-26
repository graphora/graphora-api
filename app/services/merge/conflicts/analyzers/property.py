"""Property conflict analyzer"""
import json
from typing import Dict, Any, List

from app.baml_client import b
from app.baml_client.types import PropertyConflictAnalysis
from app.schemas.graph import Node
from ..base import ConflictAnalyzer

class PropertyConflictAnalyzer(ConflictAnalyzer):
    """Analyzer for property conflicts"""
    
    async def analyze(
        self,
        staging_entity: Node,
        production_entity: Node
    ) -> PropertyConflictAnalysis:
        """Analyze property conflicts between entities
        
        Args:
            staging_entity: Entity from staging
            production_entity: Entity from production
            
        Returns:
            Analysis of property conflicts
        """
        # Find differing properties
        all_properties = set(staging_entity.properties.keys()) | set(production_entity.properties.keys())
        property_conflicts = []
        
        for prop_name in all_properties:
            staging_value = staging_entity.properties.get(prop_name)
            production_value = production_entity.properties.get(prop_name)
            
            if staging_value != production_value:
                # Determine value type
                if isinstance(staging_value, (int, float)):
                    value_type = "numeric"
                elif isinstance(staging_value, bool):
                    value_type = "boolean"
                elif isinstance(staging_value, list):
                    value_type = "array"
                elif isinstance(staging_value, dict):
                    value_type = "object"
                else:
                    value_type = "string"
                    
                property_conflicts.append({
                    "property_name": prop_name,
                    "staging_value": json.dumps(staging_value) if staging_value is not None else None,
                    "production_value": json.dumps(production_value) if production_value is not None else None,
                    "value_type": value_type
                })
        
        # Get historical resolutions if available
        historical_resolutions = []
        
        # Analyze each property conflict
        analysis_results = []
        for conflict in property_conflicts:
            analysis = b.AnalyzePropertyConflict(
                entity_type=staging_entity.label,
                property_name=conflict["property_name"],
                staging_value=conflict["staging_value"],
                production_value=conflict["production_value"],
                value_type=conflict["value_type"],
                historical_resolutions=json.dumps(historical_resolutions)
            )
            analysis_results.append(analysis)
            
        # Find the most confident analysis result
        if not analysis_results:
            return PropertyConflictAnalysis(
                potential_risks=[],
                recommended_strategy="",
                confidence=0.0,
                explanation="No property conflicts found",
                can_auto_resolve=False
            )
            
        # Get the analysis with highest confidence
        best_analysis = max(analysis_results, key=lambda x: x.confidence)
        
        return PropertyConflictAnalysis(
            potential_risks=best_analysis.potential_risks,
            recommended_strategy=best_analysis.recommended_strategy,
            confidence=best_analysis.confidence,
            explanation=best_analysis.explanation,
            can_auto_resolve=best_analysis.confidence > 0.9
        )
