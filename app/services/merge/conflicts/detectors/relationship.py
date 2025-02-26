"""Relationship conflict detector"""
import asyncio
from typing import Dict, List, Optional, Set

from app.schemas.conflicts import Conflict
from app.schemas.graph import Edge, GraphResponse
from app.services.storage.interface import GraphStorageInterface
from app.services.merge.conflicts.base import ConflictDetector
from app.services.merge.conflicts.analyzers.relationship import RelationshipConflictAnalyzer
from app.services.merge.conflicts.creators.relationship import RelationshipConflictCreator

class RelationshipConflictDetector(ConflictDetector):
    """Detector for relationship conflicts"""
    
    def __init__(self, storage: GraphStorageInterface):
        super().__init__(storage)
        self.analyzer = RelationshipConflictAnalyzer()
        self.creator = RelationshipConflictCreator()
    
    async def detect_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect relationship conflicts between staging and production
        
        Args:
            staging_graph: Graph containing staging entities and relationships
            production_entity_mapping: Mapping of staging IDs to production matches
            
        Returns:
            List of relationship conflicts
        """
        conflicts = []
        
        # Get all relationships for matched entities
        staging_relationships = self._get_entity_relationships(
            staging_graph.edges,
            set(production_entity_mapping.keys())
        )
        
        async def process_relationship(edge: Edge) -> List[Optional[Conflict]]:
            edge_conflicts = []
            
            # Get production matches for source and target
            source_matches = production_entity_mapping.get(edge.source_id, [])
            target_matches = production_entity_mapping.get(edge.target_id, [])
            
            if not source_matches or not target_matches:
                return []
            
            # Check each source-target combination
            for prod_source in source_matches:
                for prod_target in target_matches:
                    # Get production relationships
                    prod_relationships = await self.storage.get_relationships_between(
                        prod_source,
                        prod_target
                    )
                    
                    if not prod_relationships:
                        # Missing relationship conflict
                        conflict = await self.creator.create_missing_conflict(
                            conflict_id=f"rel_missing_{edge.id}_{prod_source}_{prod_target}",
                            staging_relationship=edge,
                            production_source_id=prod_source,
                            production_target_id=prod_target
                        )
                        edge_conflicts.append(conflict)
                        continue
                    
                    # Analyze each production relationship
                    for prod_rel in prod_relationships:
                        analysis = await self.analyzer.analyze(
                            staging_relationship=edge,
                            production_relationship=prod_rel
                        )
                        
                        if analysis.has_conflicts:
                            conflict = await self.creator.create_conflict(
                                conflict_id=f"rel_{edge.id}_{prod_rel.id}",
                                staging_relationship=edge,
                                production_relationship=prod_rel,
                                analysis=analysis
                            )
                            edge_conflicts.append(conflict)
            
            return edge_conflicts
        
        # Process all relationships in parallel
        tasks = [
            process_relationship(edge)
            for edge in staging_relationships
        ]
        
        results = await asyncio.gather(*tasks)
        for edge_conflicts in results:
            conflicts.extend([c for c in edge_conflicts if c])
        
        return conflicts
    
    def _get_entity_relationships(
        self,
        edges: List[Edge],
        entity_ids: Set[str]
    ) -> List[Edge]:
        """Get relationships where both endpoints are in entity_ids"""
        return [
            edge for edge in edges
            if edge.source_id in entity_ids and edge.target_id in entity_ids
        ]
