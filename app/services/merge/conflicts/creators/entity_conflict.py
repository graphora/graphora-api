"""Entity conflict creator"""
from typing import Dict, List

from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionOption,
    ResolutionStrategy
)
from app.schemas.graph import Node
from app.baml_client.types import EntitySimilarityAnalysis
from ..base import ConflictCreator

class EntityConflictCreator(ConflictCreator):
    """Creator for entity matching conflicts"""
    
    async def create_conflict(
        self,
        conflict_id: str,
        staging_entity: Node,
        production_entities: List[Node],
        similarity_scores: Dict[str, float],
        analyses: Dict[str, EntitySimilarityAnalysis],
        similarity_threshold: float
    ) -> Conflict:
        """Create an entity matching conflict with rich context
        
        Args:
            conflict_id: Unique conflict identifier
            staging_entity: Entity from staging graph
            production_entities: Matching entities from production
            similarity_scores: Similarity scores for each production entity
            analyses: Analysis results for each production entity
            similarity_threshold: Threshold for suggesting entity merges
            
        Returns:
            Conflict object with resolution options and context
        """
        resolution_options = []
        
        # Option to match with each production entity
        for prod_entity in production_entities:
            analysis = analyses[prod_entity.id]
            similarity = similarity_scores[prod_entity.id]
            
            resolution_options.append(
                ResolutionOption(
                    id=f"{conflict_id}_match_{prod_entity.id}",
                    description=(
                        f"Match with production entity {prod_entity.id} "
                        f"(similarity: {similarity:.2f})"
                    ),
                    resolution_type=ResolutionStrategy.MERGE_VALUES,
                    resolution_data={
                        "production_id": prod_entity.id,
                        "merge_properties": analysis.matching_properties
                    },
                    confidence=similarity
                )
            )
        
        # Option to merge all production entities if they're similar enough
        avg_similarity = sum(similarity_scores.values()) / len(similarity_scores)
        if avg_similarity > similarity_threshold:
            resolution_options.append(
                ResolutionOption(
                    id=f"{conflict_id}_merge_all",
                    description="Merge all matching production entities",
                    resolution_type=ResolutionStrategy.MERGE_VALUES,
                    resolution_data={
                        "production_ids": [e.id for e in production_entities],
                        "merge_strategy": "combine_properties"
                    },
                    confidence=0.3
                )
            )
        
        # Option to create new entity
        resolution_options.append(
            ResolutionOption(
                id=f"{conflict_id}_create_new",
                description="Create new production entity",
                resolution_type=ResolutionStrategy.CREATE_NEW,
                resolution_data={},
                confidence=0.2
            )
        )
        
        # Build rich context with analysis details
        context = {
            "staging_entity": {
                "id": staging_entity.id,
                "label": staging_entity.label,
                "properties": staging_entity.properties
            },
            "production_entities": [
                {
                    "id": e.id,
                    "label": e.label,
                    "properties": e.properties,
                    "similarity": similarity_scores[e.id],
                    "analysis": {
                        "matching_properties": analyses[e.id].matching_properties,
                        "mismatched_properties": analyses[e.id].mismatched_properties,
                        "semantic_similarity": analyses[e.id].semantic_similarity,
                        "potential_impact": analyses[e.id].potential_merge_impact,
                        "reasoning": analyses[e.id].reasoning
                    }
                }
                for e in production_entities
            ],
            "similarity_scores": similarity_scores,
            "average_similarity": avg_similarity
        }
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.DUPLICATE_ENTITY,
            severity=ConflictSeverity.CRITICAL,
            staging_ids=[staging_entity.id],
            production_ids=[e.id for e in production_entities],
            description=f"Staging entity matches multiple production entities",
            context=context,
            resolution_options=resolution_options
        )
