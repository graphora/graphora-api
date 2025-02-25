from abc import ABC, abstractmethod
import traceback
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from neo4j import GraphDatabase
import json
import uuid

from app.services.storage.models import (
    StorageBatchResult,
    StorageCheckpoint,
    StorageStage,
    DatabaseError,
    TransformationResult,
    Node,
    Edge
)
from app.config import settings
from app.services.transform.models import BaseNode, RelationshipInstance

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
    
    @abstractmethod
    async def get_nodes_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Node]:
        """Get all nodes with the specified property value"""
        pass
    
    @abstractmethod
    async def get_relationships_between_nodes(
        self,
        node_ids: List[str]
    ) -> List[Edge]:
        """Get all relationships between the specified nodes"""
        pass
        
    @abstractmethod
    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True
    ) -> List[Node]:
        """
        Find nodes with matching property value.
        
        Args:
            label: Node label to filter by
            property_name: Name of the property to match
            property_value: Value to match against
            exact_match: If True, requires exact value match. If False, allows partial matches
            
        Returns:
            List of matching nodes
        """
        pass
        
    @abstractmethod
    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True
    ) -> List[Node]:
        """
        Find nodes with similar properties using fuzzy matching.
        
        Args:
            label: Node label to filter by
            properties: Properties to compare for similarity
            similarity_threshold: Minimum similarity score (0-1) to include in results
            max_results: Maximum number of similar nodes to return
            include_relationships: Whether to include relationship patterns in similarity calculation
            
        Returns:
            List of similar nodes sorted by similarity score (highest first)
        """
        pass
        
    @abstractmethod
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Create a new node"""
        pass
        
    @abstractmethod
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Update an existing node"""
        pass
        
    @abstractmethod
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> Edge:
        """Create a relationship between nodes"""
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
            traceback.print_exc()
            raise DatabaseError(f"Failed to connect to Neo4j: {str(e)}")
    
    def _build_node_query(self, node: BaseNode, transform_id: str) -> Tuple[str, Dict]:
        """Build Cypher query for node creation"""
        labels = [node.type]
        
        # Extract properties excluding metadata
        properties = {}
        if hasattr(node, 'properties') and node.properties:
            properties = {
                k: v for k, v in node.properties.items()
                if v is not None
            }
        
        # Add transform ID and provenance
        properties['transform_id'] = transform_id
        if hasattr(node, 'provenance') and node.provenance:
            properties['provenance'] = json.dumps(node.provenance.model_dump())
        
        return (
            f"CREATE (n:{':'.join(labels)} {{id: $id}}) "
            "SET n += $properties "
            "RETURN n",
            {"id": node.id, "properties": properties}
        )
    
    def _build_relationship_query(
        self,
        relationship: RelationshipInstance,
        transform_id: str
    ) -> Tuple[str, Dict]:
        """Build Cypher query for relationship creation"""
        properties = {}
        if hasattr(relationship, 'properties') and relationship.properties:
            properties = {
                k: v for k, v in relationship.properties.items()
                if v is not None
            }
        
        properties['transform_id'] = transform_id
        if hasattr(relationship, 'provenance') and relationship.provenance:
            properties['provenance'] = json.dumps(relationship.provenance.model_dump())
        
        # Note: relationship type must be directly in query string, not a parameter
        return (
            f"MATCH (source {{id: $source_id}}), (target {{id: $target_id}}) "
            f"MERGE (source)-[r:{relationship.type}]->(target) "
            "SET r += $properties "
            "RETURN r",
            {
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "properties": properties
            }
        )
    
    async def store_nodes(
        self,
        nodes: List[BaseNode],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store a batch of nodes in Neo4j"""
        start_time = datetime.now()
        processed = 0
        warnings = []
        
        try:
            with self.driver.session(database=self.database) as session:
                for node in nodes:
                    try:
                        query, parameters = self._build_node_query(
                            node,
                            transform_id
                        )
                        session.run(
                            query,
                            **parameters
                        )
                        processed += 1
                    except Exception as e:
                        traceback.print_exc()
                        warnings.append(
                            f"Failed to store node {node.id}: {str(e)}"
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
            traceback.print_exc()
            
            return StorageBatchResult(
                batch_index=batch_index,
                items_processed=processed,
                processing_time_ms=processing_time,
                success=False,
                error=str(e)
            )
    
    async def store_relationships(
        self,
        relationships: List[RelationshipInstance],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store a batch of relationships in Neo4j"""
        start_time = datetime.now()
        processed = 0
        warnings = []
        
        try:
            with self.driver.session(database=self.database) as session:
                for rel in relationships:
                    try:
                        query, parameters = self._build_relationship_query(
                            rel,
                            transform_id
                        )
                        session.run(
                            query,
                            **parameters
                        )
                        processed += 1
                    except Exception as e:
                        traceback.print_exc()
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
            traceback.print_exc()
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
            traceback.print_exc()
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
            traceback.print_exc()
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
                    CREATE (c:Checkpoint {transform_id: $transform_id})
                    SET c.last_processed_index = $last_index,
                        c.stage = $stage,
                        c.timestamp = datetime()
                    """,
                    transform_id=transform_id,
                    last_index=last_index,
                    stage=stage
                )
                
        except Exception as e:
            traceback.print_exc()
            raise DatabaseError(
                f"Failed to update checkpoint: {str(e)}"
            )
    
    async def get_nodes_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Node]:
        """Get all nodes with the specified property value"""
        try:
            with self.driver.session(database=self.database) as session:
                query = f"""
                    MATCH (n)
                    WHERE n.{property_name} = $value
                    RETURN n
                """
                result = session.run(query, value=property_value)
                return [self._node_from_record(record['n']) for record in result]
        except Exception as e:
            traceback.print_exc()
            raise DatabaseError(f"Failed to get nodes by property: {str(e)}")
            
    async def get_relationships_between_nodes(
        self,
        node_ids: List[str]
    ) -> List[Edge]:
        """Get all relationships between the specified nodes"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                    MATCH (n)-[r]-(m)
                    WHERE n.id IN $node_ids AND m.id IN $node_ids
                    RETURN r, n, m
                """
                result = session.run(query, node_ids=node_ids)
                return [self._edge_from_record(record) for record in result]
        except Exception as e:
            traceback.print_exc()
            raise DatabaseError(f"Failed to get relationships: {str(e)}")
            
    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True
    ) -> List[Node]:
        """Find nodes with the specified label and property value"""
        try:
            with self.driver.session(database=self.database) as session:
                if exact_match:
                    query = f"""
                        MATCH (n:{label})
                        WHERE n.{property_name} = $value
                        RETURN n
                    """
                else:
                    query = f"""
                        MATCH (n:{label})
                        WHERE n.{property_name} CONTAINS $value
                        RETURN n
                    """
                result = session.run(query, value=property_value)
                return [self._node_from_record(record['n']) for record in result]
        except Exception as e:
            traceback.print_exc()
            raise DatabaseError(f"Failed to find nodes: {str(e)}")
            
    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True
    ) -> List[Node]:
        """Find nodes with similar properties using fuzzy matching"""
        try:
            with self.driver.session(database=self.database) as session:
                # Build dynamic property matching based on type
                property_matches = []
                relationship_matches = []
                params = {}
                
                for idx, (key, value) in enumerate(properties.items()):
                    param_key = f"value{idx}"
                    params[param_key] = value
                    
                    if isinstance(value, str):
                        # Use Levenshtein for string properties
                        property_matches.append(
                            f"apoc.text.levenshteinSimilarity(n.{key}, ${param_key})"
                        )
                    else:
                        # Exact match for non-string properties
                        property_matches.append(
                            f"CASE WHEN n.{key} = ${param_key} THEN 1.0 ELSE 0.0 END"
                        )
                
                # Include relationship patterns if requested
                relationship_score = ""
                if include_relationships and properties.get("id"):
                    relationship_score = """
                    , size([
                        (n)-[r]->(m) WHERE type(r) IN 
                        [(source)-[sr]->() WHERE id(source) = $source_id | type(sr)]
                    ]) * 1.0 / 
                    CASE 
                        WHEN size([(source)-[sr]->() WHERE id(source) = $source_id | type(sr)]) > 0 
                        THEN size([(source)-[sr]->() WHERE id(source) = $source_id | type(sr)])
                        ELSE 1 
                    END as relationship_score
                    """
                    params["source_id"] = properties["id"]
                
                # Calculate weighted similarity score
                query = f"""
                    MATCH (n:{label})
                    WITH n
                    {relationship_score}
                    WITH n,
                         CASE 
                            WHEN size([{', '.join(property_matches)}]) > 0
                            THEN reduce(s = 0.0, x IN [{', '.join(property_matches)}] | s + x) / size([{', '.join(property_matches)}])
                            ELSE 0.0
                         END as property_score
                         {', relationship_score' if include_relationships and properties.get('id') else ''}
                    WITH n, 
                         CASE
                            WHEN $include_relationships AND relationship_score IS NOT NULL
                            THEN (property_score * 0.7 + relationship_score * 0.3)
                            ELSE property_score
                         END as similarity_score
                    WHERE similarity_score >= $threshold
                    RETURN n, similarity_score
                    ORDER BY similarity_score DESC
                    LIMIT $max_results
                """
                
                params.update({
                    "threshold": similarity_threshold,
                    "max_results": max_results,
                    "include_relationships": include_relationships
                })
                
                result = session.run(query, **params)
                return [self._node_from_record(record['n']) for record in result]
                
        except Exception as e:
            traceback.print_exc()
            raise DatabaseError(f"Failed to find similar nodes: {str(e)}")
            
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Create a new node"""
        try:
            with self.driver.session(database=self.database) as session:
                # Ensure node has an ID
                if 'id' not in properties:
                    properties['id'] = str(uuid.uuid4())
                    
                query = f"""
                    CREATE (n:{label} $props)
                    RETURN n
                """
                result = session.run(query, props=properties)
                record = result.single()
                return self._node_from_record(record['n'])
        except Exception as e:
            traceback.print_exc()
            raise DatabaseError(f"Failed to create node: {str(e)}")
            
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Update an existing node"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                    MATCH (n {id: $node_id})
                    SET n += $props
                    RETURN n
                """
                result = session.run(query, node_id=node_id, props=properties)
                record = result.single()
                if not record:
                    raise DatabaseError(f"Node {node_id} not found")
                return self._node_from_record(record['n'])
        except Exception as e:
            traceback.print_exc()
            raise DatabaseError(f"Failed to update node: {str(e)}")
            
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> Edge:
        """Create a relationship between nodes"""
        try:
            with self.driver.session(database=self.database) as session:
                properties = properties or {}
                if 'id' not in properties:
                    properties['id'] = str(uuid.uuid4())
                    
                query = f"""
                    MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
                    CREATE (source)-[r:{rel_type} $props]->(target)
                    RETURN r, source, target
                """
                result = session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    props=properties
                )
                record = result.single()
                if not record:
                    raise DatabaseError("Failed to create relationship")
                return self._edge_from_record(record)
        except Exception as e:
            traceback.print_exc()
            raise DatabaseError(f"Failed to create relationship: {str(e)}")
            
    def _node_from_record(self, record) -> Node:
        """Convert Neo4j node record to Node model"""
        properties = dict(record)
        node_id = properties.pop('id', str(uuid.uuid4()))
        label = list(record.labels)[0] if record.labels else 'Unknown'
        
        return Node(
            id=node_id,
            label=label,
            properties=properties,
            type=label
        )
        
    def _edge_from_record(self, record) -> Edge:
        """Convert Neo4j relationship record to Edge model"""
        rel = record['r']
        properties = dict(rel)
        edge_id = properties.pop('id', str(uuid.uuid4()))
        
        return Edge(
            id=edge_id,
            source=record['source']['id'],
            target=record['target']['id'],
            type=rel.type,
            properties=properties
        )
    
    def close(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
