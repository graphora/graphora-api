"""Entity matching conflict detector"""
import asyncio
from typing import Dict, List, Optional

from app.schemas.conflicts import Conflict
from app.schemas.graph import Edge, GraphResponse, Node
from app.services.storage.interface import GraphStorageInterface
from app.services.merge.conflicts.base import ConflictDetector
from app.services.merge.conflicts.analyzers.entity_similarity import EntitySimilarityAnalyzer
from app.services.merge.conflicts.creators.entity_conflict import EntityConflictCreator

class EntityMatchingDetector(ConflictDetector):
    """Detector for entity matching conflicts"""
    
    def __init__(
        self,
        storage: GraphStorageInterface,
        similarity_threshold: float = 0.7
    ):
        super().__init__(storage)
        self.analyzer = EntitySimilarityAnalyzer()
        self.creator = EntityConflictCreator()
        self.similarity_threshold = similarity_threshold
    
    async def detect_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect conflicts where staging entities match multiple production entities"""
        conflicts = []
        
        async def process_match(staging_id: str, prod_matches: List[str]) -> Optional[Conflict]:
            if len(prod_matches) <= 1:
                return None
                
            staging_entity = next(
                (n for n in staging_graph.nodes if n.id == staging_id),
                None
            )
            if not staging_entity:
                return None
                
            prod_entities = []
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self.storage.get_node_by_id(prod_id))
                    for prod_id in prod_matches
                ]
            prod_entities = [t.result() for t in tasks if t.result()]
            
            if not prod_entities:
                return None
            
            similarity_scores = {}
            analyses = {}
            
            async with asyncio.TaskGroup() as tg:
                tasks = {
                    prod_entity.id: tg.create_task(
                        self.analyzer.analyze(
                            staging_entity=staging_entity,
                            production_entity=prod_entity
                        )
                    )
                    for prod_entity in prod_entities
                }
            
            for prod_id, task in tasks.items():
                analysis = task.result()
                similarity_scores[prod_id] = analysis.similarity_score
                analyses[prod_id] = analysis
            
            return await self.creator.create_conflict(
                conflict_id=f"entity_match_{staging_entity.id}",
                staging_entity=staging_entity,
                production_entities=prod_entities,
                similarity_scores=similarity_scores,
                analyses=analyses,
                similarity_threshold=self.similarity_threshold
            )
        
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(process_match(staging_id, prod_matches))
                for staging_id, prod_matches in production_entity_mapping.items()
            ]
        
        conflicts = [t.result() for t in tasks if t.result()]
        return conflicts
