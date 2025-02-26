"""Neo4j implementation of graph storage"""
from typing import List, Dict, Any, Optional
import logging
import traceback
from contextlib import asynccontextmanager
import asyncio
import uuid
import json
from datetime import datetime
import time

import neo4j
from neo4j import AsyncGraphDatabase, GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, DatabaseError, SessionExpired, TransientError

from .interface import GraphStorageInterface
from .models import Node, Edge, StorageBatchResult, StorageCheckpoint, StorageStage, TransformationResult
from .exceptions import (
    StorageConnectionError,
    StorageAuthError,
    StorageQueryError,
    StorageError
)
from app.services.transform.models import BaseNode, RelationshipInstance

# Configure logger
logger = logging.getLogger(__name__)

class Neo4jStorage(GraphStorageInterface):
    """Neo4j implementation of graph storage"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j", max_retries: int = 3, test_mode: bool = False):
        """Initialize Neo4j connection"""
        try:
            # Create the async driver
            self.driver = AsyncGraphDatabase.driver(
                uri,
                auth=(username, password),
                database=database
            )
            self.database = database
            self.max_retries = max_retries

            if not test_mode:
                # Create a synchronous driver for testing
                sync_driver = GraphDatabase.driver(
                    uri,
                    auth=(username, password),
                    database=database
                )
                try:
                    with sync_driver.session(database=database) as session:
                        session.run("RETURN 1")
                    logger.info(f"Successfully initialized Neo4j connection to {uri}")
                finally:
                    sync_driver.close()

        except AuthError as e:
            logger.error(f"Authentication failed for Neo4j connection: {str(e)}")
            raise StorageAuthError(f"Failed to authenticate: {str(e)}")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise StorageConnectionError(f"Neo4j service unavailable: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error initializing Neo4j connection: {str(e)}")
            raise StorageError(f"Failed to initialize storage: {str(e)}")

    @asynccontextmanager
    async def _get_session(self):
        """Get a Neo4j session with automatic cleanup"""
        session = None
        try:
            session = await self.driver.session()
            yield session
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(f"Failed to create Neo4j session: {str(e)}")
            raise StorageConnectionError(f"Could not create session: {str(e)}")
        except DatabaseError as e:
            logger.error(f"Database error while creating session: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error while creating session: {str(e)}")
            raise StorageError(f"Unexpected error: {str(e)}")
        finally:
            if session:
                try:
                    await session.close()
                except Exception as e:
                    logger.error(f"Error closing session: {str(e)}")

    async def _execute_with_retry(self, operation):
        """Execute Neo4j operation with retry logic"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await operation()
            except (ServiceUnavailable, SessionExpired, TransientError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise StorageError(f"Operation failed after {self.max_retries} retries: {str(e)}")
            except (DatabaseError, StorageConnectionError) as e:
                # Don't retry database or connection errors, just propagate them
                raise StorageError(str(e))
            except Exception as e:
                # Don't retry other errors
                raise StorageError(f"Unexpected error: {str(e)}")

    def _build_node_query(self, node: BaseNode, transform_id: str) -> tuple[str, Dict]:
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
    ) -> tuple[str, Dict]:
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
        """Store nodes in Neo4j"""
        start_time = time.time()
        success = True
        error_message = None
        items_processed = 0
        warnings = []

        # First try to store all nodes
        for node in nodes:
            try:
                async def _execute_query():
                    async with self._get_session() as session:
                        query = self._build_node_query(node, transform_id)
                        await session.run(query)

                await self._execute_with_retry(_execute_query)
                items_processed += 1
            except (StorageError, DatabaseError) as e:
                success = False
                error_message = str(e)
                logger.error(f"Failed to store node {node.id}: {error_message}")
                break

        # Only update checkpoint if at least one node was stored successfully
        if items_processed > 0:
            try:
                checkpoint_result = await self.update_checkpoint(
                    transform_id,
                    batch_index,
                    StorageStage.NODES
                )
                if not checkpoint_result.success:
                    success = False
                    error_message = checkpoint_result.error
                    logger.error(f"Error updating checkpoint: {error_message}")
            except (StorageError, DatabaseError) as e:
                success = False
                error_message = str(e)
                logger.error(f"Failed to update checkpoint: {error_message}")

        if items_processed < len(nodes):
            warnings.append(f"Partial batch failure: {items_processed} of {len(nodes)} nodes stored")

        processing_time_ms = (time.time() - start_time) * 1000
        return StorageBatchResult(
            batch_index=batch_index,
            items_processed=items_processed,
            processing_time_ms=processing_time_ms,
            success=success,
            error=error_message,
            warnings=warnings
        )

    async def store_relationships(
        self,
        relationships: List[RelationshipInstance],
        batch_index: int,
        transform_id: str
    ) -> StorageBatchResult:
        """Store relationships in Neo4j"""
        start_time = time.time()
        success = True
        error_message = None
        items_processed = 0
        warnings = []

        # First try to store all relationships
        for rel in relationships:
            try:
                async def _execute_query():
                    async with self._get_session() as session:
                        query = self._build_relationship_query(rel, transform_id)
                        await session.run(query)

                await self._execute_with_retry(_execute_query)
                items_processed += 1
            except (StorageError, DatabaseError) as e:
                success = False
                error_message = str(e)
                logger.error(f"Failed to store relationship {rel.id}: {error_message}")
                break

        # Only update checkpoint if at least one relationship was stored successfully
        if items_processed > 0:
            try:
                checkpoint_result = await self.update_checkpoint(
                    transform_id,
                    batch_index,
                    StorageStage.RELATIONSHIPS
                )
                if not checkpoint_result.success:
                    success = False
                    error_message = checkpoint_result.error
                    logger.error(f"Error updating checkpoint: {error_message}")
            except (StorageError, DatabaseError) as e:
                success = False
                error_message = str(e)
                logger.error(f"Failed to update checkpoint: {error_message}")

        if items_processed < len(relationships):
            warnings.append(f"Partial batch failure: {items_processed} of {len(relationships)} relationships stored")

        processing_time_ms = (time.time() - start_time) * 1000
        return StorageBatchResult(
            batch_index=batch_index,
            items_processed=items_processed,
            processing_time_ms=processing_time_ms,
            success=success,
            error=error_message,
            warnings=warnings
        )

    async def get_storage_status(
        self,
        transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status"""
        async def _execute_query():
            async with self._get_session() as session:
                query = """
                MATCH (c:Checkpoint {transform_id: $transform_id})
                RETURN c ORDER BY c.timestamp DESC LIMIT 1
                """
                result = await session.run(query, {"transform_id": transform_id})
                records = await result.fetch_all()
                
                if not records:
                    return None
                    
                checkpoint = records[0]["c"]
                return StorageCheckpoint(
                    transform_id=checkpoint["transform_id"],
                    last_processed_index=checkpoint["last_processed_index"],
                    stage=StorageStage(checkpoint["stage"]),
                    timestamp=checkpoint["timestamp"]
                )
        
        return await self._execute_with_retry(_execute_query)

    async def update_checkpoint(
        self,
        transform_id: str,
        last_index: int,
        stage: StorageStage
    ) -> StorageBatchResult:
        """Update storage checkpoint"""
        start_time = time.time()
        success = True
        error_message = None
        items_processed = 0

        async def _execute_query():
            nonlocal success, error_message, items_processed
            async with self._get_session() as session:
                try:
                    query = """
                    MERGE (c:Checkpoint {transform_id: $transform_id})
                    SET c.last_processed_index = $last_index,
                        c.stage = $stage,
                        c.timestamp = datetime()
                    """
                    await session.run(
                        query,
                        transform_id=transform_id,
                        last_index=last_index,
                        stage=stage
                    )
                    items_processed = 1
                except DatabaseError as e:
                    success = False
                    error_message = f"Failed to update checkpoint: {str(e)}"
                    logger.error(error_message)
                    raise

        try:
            await self._execute_with_retry(_execute_query)
        except Exception as e:
            success = False
            error_message = f"Failed to update checkpoint: {str(e)}"
            logger.error(error_message)

        processing_time_ms = (time.time() - start_time) * 1000
        return StorageBatchResult(
            batch_index=last_index,
            items_processed=items_processed,
            processing_time_ms=processing_time_ms,
            success=success,
            error=error_message,
            warnings=[]
        )

    async def get_transformation_data(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships for a transformation"""
        async def _execute_query():
            async with self._get_session() as session:
                # Get nodes
                node_query = """
                MATCH (n)
                WHERE n.transform_id = $transform_id
                RETURN n
                """
                node_result = await session.run(node_query, {"transform_id": transform_id})
                node_records = await node_result.fetch_all()
                
                nodes = [
                    {
                        **dict(record["n"]),
                        "type": next(
                            label for label in record["n"].labels
                            if label != "Checkpoint"
                        )
                    }
                    for record in node_records
                    if "Checkpoint" not in record["n"].labels
                ]
                
                # Get relationships
                rel_query = """
                MATCH (source)-[r]-(target)
                WHERE r.transform_id = $transform_id
                RETURN source, r, target
                """
                rel_result = await session.run(rel_query, {"transform_id": transform_id})
                rel_records = await rel_result.fetch_all()
                
                relationships = [
                    {
                        "source_id": record["source"]["id"],
                        "target_id": record["target"]["id"],
                        "relationship_type": type(record["r"]).__name__,
                        "properties": dict(record["r"])
                    }
                    for record in rel_records
                ]
                
                return TransformationResult(
                    transform_id=transform_id,
                    nodes=nodes,
                    relationships=relationships,
                    timestamp=datetime.now()
                )
        
        return await self._execute_with_retry(_execute_query)

    async def get_nodes_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Node]:
        """Get all nodes with the specified property value"""
        async def _execute_query():
            async with self._get_session() as session:
                query = f"""
                MATCH (n)
                WHERE n.{property_name} = $value
                RETURN n
                """
                result = await session.run(query, {"value": property_value})
                records = await result.fetch_all()
                
                return [
                    Node(
                        id=str(record["n"].id),
                        label=list(record["n"].labels)[0],
                        type=record["n"].get("type", ""),
                        properties=dict(record["n"].items())
                    )
                    for record in records
                ]
        
        return await self._execute_with_retry(_execute_query)

    async def get_relationships_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Edge]:
        """Get all relationships between two nodes"""
        async def _execute_query():
            async with self._get_session() as session:
                if relationship_type:
                    query = f"""
                    MATCH (source {{id: $source_id}})-[r:{relationship_type}]-(target {{id: $target_id}})
                    RETURN r
                    """
                else:
                    query = """
                    MATCH (source {id: $source_id})-[r]-(target {id: $target_id})
                    RETURN r
                    """
                
                result = await session.run(
                    query,
                    {"source_id": source_id, "target_id": target_id}
                )
                records = await result.fetch_all()
                
                return [
                    Edge(
                        id=str(record["r"].id),
                        source=source_id,
                        target=target_id,
                        type=type(record["r"]).__name__,
                        properties=dict(record["r"].items())
                    )
                    for record in records
                ]
        
        return await self._execute_with_retry(_execute_query)

    async def get_relationships_between_nodes(
        self,
        node_ids: List[str]
    ) -> List[Edge]:
        """Get all relationships between a set of nodes"""
        async def _execute_query():
            async with self._get_session() as session:
                query = """
                MATCH (n)-[r]-(m)
                WHERE n.id IN $node_ids AND m.id IN $node_ids
                RETURN DISTINCT r, n, m
                """
                result = await session.run(query, {"node_ids": node_ids})
                records = await result.fetch_all()
                
                return [
                    Edge(
                        id=str(record["r"].id),
                        source=record["n"]["id"],
                        target=record["m"]["id"],
                        type=type(record["r"]).__name__,
                        properties=dict(record["r"].items())
                    )
                    for record in records
                ]
        
        return await self._execute_with_retry(_execute_query)

    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True
    ) -> List[Node]:
        """Find nodes with matching property value"""
        async def _execute_query():
            async with self._get_session() as session:
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
                
                result = await session.run(query, {"value": property_value})
                records = await result.fetch_all()
                
                return [
                    Node(
                        id=str(record["n"].id),
                        label=list(record["n"].labels)[0],
                        type=record["n"].get("type", ""),
                        properties=dict(record["n"].items())
                    )
                    for record in records
                ]
        
        return await self._execute_with_retry(_execute_query)

    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True
    ) -> List[Node]:
        """Find nodes with similar properties using fuzzy matching"""
        async def _execute_query():
            async with self._get_session() as session:
                # Build dynamic property matching based on type
                property_matches = []
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
                
                result = await session.run(query, params)
                records = await result.fetch_all()
                
                return [
                    Node(
                        id=str(record["n"].id),
                        label=list(record["n"].labels)[0],
                        type=record["n"].get("type", ""),
                        properties=dict(record["n"].items())
                    )
                    for record in records
                ]
        
        return await self._execute_with_retry(_execute_query)

    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Create a new node"""
        async def _execute_query():
            async with self._get_session() as session:
                # Ensure node has an ID
                if "id" not in properties:
                    properties["id"] = str(uuid.uuid4())
                
                query = f"""
                CREATE (n:{label} $props)
                RETURN n
                """
                result = await session.run(query, {"props": properties})
                records = await result.fetch_all()
                
                if not records:
                    raise StorageError("Failed to create node")
                
                node_data = records[0]["n"]
                return Node(
                    id=str(node_data.id),
                    label=list(node_data.labels)[0],
                    type=node_data.get("type", ""),
                    properties=dict(node_data.items())
                )
        
        return await self._execute_with_retry(_execute_query)

    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Update an existing node"""
        async def _execute_query():
            async with self._get_session() as session:
                query = """
                MATCH (n {id: $node_id})
                SET n += $props
                RETURN n
                """
                result = await session.run(
                    query,
                    {"node_id": node_id, "props": properties}
                )
                records = await result.fetch_all()
                
                if not records:
                    raise StorageError(f"Node {node_id} not found")
                
                node_data = records[0]["n"]
                return Node(
                    id=str(node_data.id),
                    label=list(node_data.labels)[0],
                    type=node_data.get("type", ""),
                    properties=dict(node_data.items())
                )
        
        return await self._execute_with_retry(_execute_query)

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> Edge:
        """Create a relationship between nodes"""
        async def _execute_query():
            async with self._get_session() as session:
                properties = properties or {}
                if "id" not in properties:
                    properties["id"] = str(uuid.uuid4())
                
                query = f"""
                MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
                CREATE (source)-[r:{rel_type} $props]->(target)
                RETURN r, source, target
                """
                result = await session.run(
                    query,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "props": properties
                    }
                )
                records = await result.fetch_all()
                
                if not records:
                    raise StorageError("Failed to create relationship")
                
                record = records[0]
                return Edge(
                    id=str(record["r"].id),
                    source=source_id,
                    target=target_id,
                    type=rel_type,
                    properties=dict(record["r"].items())
                )
        
        return await self._execute_with_retry(_execute_query)

    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID"""
        async def _execute_query():
            async with self._get_session() as session:
                query = """
                MATCH (n {id: $node_id})
                RETURN n
                """
                result = await session.run(query, {"node_id": node_id})
                records = await result.fetch_all()
                
                if not records:
                    return None
                
                node_data = records[0]["n"]
                return Node(
                    id=str(node_data.id),
                    label=list(node_data.labels)[0],
                    type=node_data.get("type", ""),
                    properties=dict(node_data.items())
                )
        
        return await self._execute_with_retry(_execute_query)

    async def get_edges_between(
        self,
        source_id: str,
        target_id: str
    ) -> List[Edge]:
        """Get all edges between two nodes"""
        async def _execute_query():
            async with self._get_session() as session:
                query = """
                MATCH (source {id: $source_id})-[r]-(target {id: $target_id})
                RETURN r, source, target
                """
                result = await session.run(
                    query,
                    {"source_id": source_id, "target_id": target_id}
                )
                records = await result.fetch_all()
                
                return [
                    Edge(
                        id=str(record["r"].id),
                        source=source_id,
                        target=target_id,
                        type=type(record["r"]).__name__,
                        properties=dict(record["r"].items())
                    )
                    for record in records
                ]
        
        return await self._execute_with_retry(_execute_query)

    async def close(self):
        """Close Neo4j connection"""
        try:
            await self.driver.close()
            logger.info("Neo4j connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
            raise StorageError(f"Failed to close connection: {str(e)}")

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, "driver"):
            asyncio.create_task(self.close())
