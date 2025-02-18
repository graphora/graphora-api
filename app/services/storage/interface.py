from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from neo4j import GraphDatabase
import json

from app.services.storage.models import (
    StorageCheckpoint,
    StorageStage,
    StorageBatchResult,
    DatabaseError,
    TransformationResult
)
from app.config import settings

class GraphStorageInterface(ABC):
    """Abstract interface for graph storage"""
    
    @abstractmethod
    async def store_nodes(
        self,
        nodes: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store nodes in batch"""
        pass
    
    @abstractmethod
    async def store_relationships(
        self,
        relationships: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store relationships in batch"""
        pass
    
    @abstractmethod
    async def get_storage_status(
        self,
        transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status"""
        pass
    
    @abstractmethod
    async def update_checkpoint(
        self,
        transform_id: str,
        last_index: int,
        stage: StorageStage
    ) -> None:
        """Update storage checkpoint"""
        pass
    
    @abstractmethod
    async def get_transformation_data(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships for a transformation"""
        pass

class Neo4jStorage(GraphStorageInterface):
    """Neo4j implementation of graph storage"""
    
    def __init__(
        self,
        uri: str = settings.STAGING_NEO4J_URI,
        username: str = settings.STAGING_NEO4J_USER,
        password: str = settings.STAGING_NEO4J_PASSWORD,
        database: str = settings.STAGING_NEO4J_DATABASE
    ):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(username, password)
            )
            self.database = database
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
                
        except Exception as e:
            raise DatabaseError(f"Failed to connect to Neo4j: {str(e)}")
    
    def _build_node_query(self, node: Dict, transform_id: str) -> Tuple[str, Dict]:
        """Build Cypher query for node creation"""
        labels = [node['type']]
        
        # Extract properties excluding metadata
        properties = {
            k: v for k, v in node.items()
            if k not in ['type', 'provenance'] and v is not None
        }
        
        # Add transform_id and provenance
        properties['transform_id'] = transform_id
        if 'provenance' in node:
            properties['provenance'] = json.dumps(node['provenance'])
        
        return (
            f"MERGE (n:{':'.join(labels)} {{id: $id}}) "
            f"ON CREATE SET n = $properties "
            f"ON MATCH SET n += $properties"
        ), properties
    
    def _build_relationship_query(
        self,
        relationship: Dict,
        transform_id: str
    ) -> Tuple[str, Dict]:
        """Build Cypher query for relationship creation"""
        properties = relationship.get('properties', {})
        properties['transform_id'] = transform_id
        
        return (
            "MATCH (source) WHERE source.id = $source_id "
            "MATCH (target) WHERE target.id = $target_id "
            "MERGE (source)-[r:$relationship_type]->(target) "
            "ON CREATE SET r = $properties "
            "ON MATCH SET r += $properties"
        ), properties
    
    async def store_nodes(
        self,
        nodes: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store nodes in batch"""
        start_time = datetime.now()
        
        try:
            with self.driver.session(database=self.database) as session:
                processed = 0
                warnings = []
                
                # Process each node
                for node in nodes:
                    try:
                        query, properties = self._build_node_query(
                            node,
                            transform_id
                        )
                        session.run(
                            query,
                            id=node['id'],
                            properties=properties
                        )
                        processed += 1
                    except Exception as e:
                        warnings.append(
                            f"Failed to store node {node.get('id')}: {str(e)}"
                        )
                
                processing_time = (
                    datetime.now() - start_time
                ).total_seconds() * 1000
                
                return StorageBatchResult(
                    batch_index=batch_index,
                    items_processed=processed,
                    processing_time_ms=processing_time,
                    success=True,
                    warnings=warnings
                )
                
        except Exception as e:
            processing_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=0,
                processing_time_ms=processing_time,
                success=False,
                error=str(e)
            )
    
    async def store_relationships(
        self,
        relationships: List[Dict],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store relationships in batch"""
        start_time = datetime.now()
        
        try:
            with self.driver.session(database=self.database) as session:
                processed = 0
                warnings = []
                
                # Process each relationship
                for rel in relationships:
                    try:
                        query, properties = self._build_relationship_query(
                            rel,
                            transform_id
                        )
                        session.run(
                            query,
                            source_id=rel['source_id'],
                            target_id=rel['target_id'],
                            relationship_type=rel['relationship_type'],
                            properties=properties
                        )
                        processed += 1
                    except Exception as e:
                        warnings.append(
                            f"Failed to store relationship: {str(e)}"
                        )
                
                processing_time = (
                    datetime.now() - start_time
                ).total_seconds() * 1000
                
                return StorageBatchResult(
                    batch_index=batch_index,
                    items_processed=processed,
                    processing_time_ms=processing_time,
                    success=True,
                    warnings=warnings
                )
                
        except Exception as e:
            processing_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=0,
                processing_time_ms=processing_time,
                success=False,
                error=str(e)
            )
    
    async def get_transformation_data(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships for a transformation"""
        try:
            with self.driver.session(database=self.database) as session:
                # Get nodes
                node_result = session.run(
                    """
                    MATCH (n)
                    WHERE n.transform_id = $transform_id
                    RETURN n
                    """,
                    transform_id=transform_id
                )
                
                nodes = [
                    {
                        **dict(record['n']),
                        'type': next(
                            label for label in record['n'].labels
                            if label != 'Checkpoint'
                        )
                    }
                    for record in node_result
                    if 'Checkpoint' not in record['n'].labels
                ]
                
                # Get relationships
                rel_result = session.run(
                    """
                    MATCH (source)-[r]-(target)
                    WHERE r.transform_id = $transform_id
                    RETURN source, r, target
                    """,
                    transform_id=transform_id
                )
                
                relationships = [
                    {
                        'source_id': record['source']['id'],
                        'target_id': record['target']['id'],
                        'relationship_type': type(record['r']).__name__,
                        'properties': dict(record['r'])
                    }
                    for record in rel_result
                ]
                
                return TransformationResult(
                    transform_id=transform_id,
                    nodes=nodes,
                    relationships=relationships,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get transformation data: {str(e)}"
            )
    
    async def get_storage_status(
        self,
        transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status from database"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (c:Checkpoint {transform_id: $transform_id})
                    RETURN c ORDER BY c.timestamp DESC LIMIT 1
                    """,
                    transform_id=transform_id
                )
                
                record = result.single()
                if record:
                    checkpoint = record['c']
                    return StorageCheckpoint(
                        transform_id=checkpoint['transform_id'],
                        last_processed_index=checkpoint['last_processed_index'],
                        stage=StorageStage(checkpoint['stage']),
                        timestamp=checkpoint['timestamp']
                    )
                
                return None
                
        except Exception as e:
            raise DatabaseError(f"Failed to get storage status: {str(e)}")
    
    async def update_checkpoint(
        self,
        transform_id: str,
        last_index: int,
        stage: StorageStage
    ) -> None:
        """Update storage checkpoint in database"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    """
                    MERGE (c:Checkpoint {transform_id: $transform_id})
                    SET c.last_processed_index = $last_index,
                        c.stage = $stage,
                        c.timestamp = datetime()
                    """,
                    transform_id=transform_id,
                    last_index=last_index,
                    stage=stage
                )
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to update checkpoint: {str(e)}"
            )
    
    def close(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
