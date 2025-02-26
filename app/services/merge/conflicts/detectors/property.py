"""Property conflict detector"""
import asyncio
from typing import Dict, List, Optional

from app.schemas.conflicts import Conflict
from app.schemas.graph import GraphResponse
from app.services.storage.interface import GraphStorageInterface
from app.services.merge.conflicts.base import ConflictDetector
from app.services.merge.conflicts.analyzers.property import PropertyConflictAnalyzer
from app.services.merge.conflicts.creators.property import PropertyConflictCreator

class PropertyConflictDetector(ConflictDetector):
    """Detector for property conflicts between staging and production"""
    
    def __init__(self, storage: GraphStorageInterface):
        super().__init__(storage)
        self.analyzer = PropertyConflictAnalyzer()
        self.creator = PropertyConflictCreator()
    
    async def detect_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect property conflicts between staging and production entities
        
        Args:
            staging_graph: Graph containing staging entities
            production_entity_mapping: Mapping of staging IDs to production matches
            
        Returns:
            List of property conflicts
        """
        conflicts = []
        
        async def process_entity(staging_id: str, prod_id: str) -> Optional[Conflict]:
            # Get staging entity
            staging_entity = next(
                (n for n in staging_graph.nodes if n.id == staging_id),
                None
            )
            if not staging_entity:
                return None
                
            # Get production entity
            prod_entity = await self.storage.get_node_by_id(prod_id)
            if not prod_entity:
                return None
            
            # Analyze property differences
            analysis = await self.analyzer.analyze(
                staging_entity=staging_entity,
                production_entity=prod_entity
            )
            
            if not analysis.property_conflicts:
                return None
            
            return await self.creator.create_conflict(
                conflict_id=f"property_{staging_id}_{prod_id}",
                staging_entity=staging_entity,
                production_entity=prod_entity,
                analysis=analysis
            )
        
        # Process all entity pairs in parallel
        tasks = []
        for staging_id, prod_matches in production_entity_mapping.items():
            for prod_id in prod_matches:
                tasks.append(process_entity(staging_id, prod_id))
        
        results = await asyncio.gather(*tasks)
        conflicts = [r for r in results if r]
        
        return conflicts
