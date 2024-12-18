from neo4j import GraphDatabase
from typing import Dict, List, Tuple
from app.config import settings
from app.utils.logger import logger

class GraphService:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
    
    async def create_entity(self, entity):
        with self.driver.session() as session:
            query = """
            CREATE (e:Entity {id: $id, type: $type, value: $value})
            RETURN e
            """
            result = session.run(query, id=entity.id, type=entity.type, value=entity.value)
            return result.single()
    
    async def create_relationship(self, source_id, target_id, relationship_type):
        with self.driver.session() as session:
            query = """
            MATCH (source:Entity {id: $source_id})
            MATCH (target:Entity {id: $target_id})
            CREATE (source)-[r:RELATES {type: $rel_type}]->(target)
            RETURN r
            """
            result = session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                rel_type=relationship_type
            )
            return result.single()
    
    async def create_temp_subgraph(self, document_id: str, entities: List[dict], relationships: List[dict]) -> bool:
        """Create a temporary subgraph for review before merging into main graph"""
        try:
            with self.driver.session() as session:
                # Create temporary entities with a temp label
                for entity in entities:
                    query = """
                    CREATE (e:TempEntity:Entity {
                        id: $id,
                        type: $type,
                        value: $value,
                        document_id: $doc_id,
                        confidence: $confidence
                    })
                    """
                    session.run(
                        query,
                        id=entity["id"],
                        type=entity["type"],
                        value=entity["value"],
                        doc_id=document_id,
                        confidence=entity.get("confidence", 0.0)
                    )
                
                # Create temporary relationships
                for rel in relationships:
                    query = """
                    MATCH (s:TempEntity {id: $source_id, document_id: $doc_id})
                    MATCH (t:TempEntity {id: $target_id, document_id: $doc_id})
                    CREATE (s)-[r:TEMP_RELATES {
                        type: $rel_type,
                        confidence: $confidence,
                        document_id: $doc_id
                    }]->(t)
                    """
                    session.run(
                        query,
                        source_id=rel["source_id"],
                        target_id=rel["target_id"],
                        rel_type=rel["type"],
                        confidence=rel.get("confidence", 0.0),
                        doc_id=document_id
                    )
            
            logger.info(f"Created temporary subgraph for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating temporary subgraph: {str(e)}")
            return False
    
    async def get_temp_subgraph(self, document_id: str) -> Tuple[List[dict], List[dict]]:
        """Retrieve temporary subgraph for review"""
        try:
            with self.driver.session() as session:
                # Get temporary entities
                entity_query = """
                MATCH (e:TempEntity)
                WHERE e.document_id = $doc_id
                RETURN e
                """
                entities = session.run(entity_query, doc_id=document_id)
                
                # Get temporary relationships
                rel_query = """
                MATCH (s:TempEntity)-[r:TEMP_RELATES]->(t:TempEntity)
                WHERE r.document_id = $doc_id
                RETURN s.id as source_id, t.id as target_id, r.type as type, r.confidence as confidence
                """
                relationships = session.run(rel_query, doc_id=document_id)
                
                return (
                    [dict(e["e"]) for e in entities],
                    [dict(r) for r in relationships]
                )
        except Exception as e:
            logger.error(f"Error retrieving temporary subgraph: {str(e)}")
            return [], []
    
    async def incorporate_feedback(self, document_id: str, feedback: Dict) -> bool:
        """Apply feedback and update temporary subgraph"""
        try:
            with self.driver.session() as session:
                if "entity_updates" in feedback:
                    for update in feedback["entity_updates"]:
                        # Prepare updates dictionary with non-None values
                        updates = {}
                        if update.get("type"):
                            updates["type"] = update["type"]
                        if update.get("value"):
                            updates["value"] = update["value"]
                        
                        if updates:  # Only run query if we have updates
                            query = """
                            MATCH (e:TempEntity {id: $id, document_id: $doc_id})
                            SET e += $updates
                            """
                            session.run(
                                query,
                                id=update["id"],
                                doc_id=document_id,
                                updates=updates
                            )
                
                if "relationship_updates" in feedback:
                    for update in feedback["relationship_updates"]:
                        # Prepare updates dictionary with non-None values
                        updates = {}
                        if update.get("type"):
                            updates["type"] = update["type"]
                        
                        if updates:  # Only run query if we have updates
                            query = """
                            MATCH (s:TempEntity {id: $source_id})-[r:TEMP_RELATES]->(t:TempEntity {id: $target_id})
                            WHERE r.document_id = $doc_id
                            SET r += $updates
                            """
                            session.run(
                                query,
                                source_id=update["source_id"],
                                target_id=update["target_id"],
                                doc_id=document_id,
                                updates=updates
                            )
                
                logger.info(f"Successfully incorporated feedback for document {document_id}")
                return True
        except Exception as e:
            logger.error(f"Error incorporating feedback: {str(e)}")
            return False
