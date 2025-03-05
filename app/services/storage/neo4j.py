"""Neo4j implementation of graph storage"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import traceback
from contextlib import asynccontextmanager
import asyncio
import uuid
import json
from datetime import datetime
import time

import traceback
from neo4j import AsyncGraphDatabase, GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, DatabaseError, SessionExpired, TransientError

from .interface import GraphStorageInterface
from .models import Node, Edge, StorageBatchResult, StorageCheckpoint, StorageStage, TransformationResult
from .exceptions import (
    StorageConnectionError,
    StorageAuthError,
    StorageError
)
from app.services.transform.models import BaseNode, RelationshipInstance

# Configure logger
logger = logging.getLogger(__name__)

class Neo4jStorage(GraphStorageInterface):
    """Neo4j implementation of graph storage"""
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        max_retries: int = 3,
        transaction_manager=None
    ):
        """Initialize Neo4j storage
        
        Args:
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            max_retries: Maximum number of retries for operations
            transaction_manager: Optional transaction manager
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_retries = max_retries
        self.transaction_manager = transaction_manager

        try:
            # Create the async driver
            self.driver = AsyncGraphDatabase.driver(
                uri,
                auth=(username, password)
            )

            if not transaction_manager:
                # Create a synchronous driver for testing
                sync_driver = GraphDatabase.driver(
                    uri,
                    auth=(username, password)
                )
                try:
                    with sync_driver.session(database=database) as session:
                        session.run("RETURN 1")
                    logger.info(f"Successfully initialized Neo4j connection to {uri}")
                finally:
                    if sync_driver:
                        try:
                            sync_driver.close()
                        except Exception as e:
                            traceback.print_exc()
                            logger.error(f"Error closing session: {str(e)}")

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
            session = self.driver.session(database=self.database)
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
                    traceback.print_exc()
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
        """Build Cypher query for creating a node with properties"""
        # Extract properties excluding metadata
        print(node)
        if isinstance(node, dict):
            # Handle dictionary input
            node_properties = node.get('properties', {})
            node_type = node.get('type', '')
            node_id = node.get('id', str(uuid.uuid4()))
        else:
            # Handle BaseNode input
            node_properties = node.properties
            node_type = node.type
            node_id = node.id
            
        properties = {
            k: v for k, v in node_properties.items()
            if k not in ['id', 'type', 'transform_id']
        }

        # Add transform ID
        properties['transform_id'] = transform_id

        # Add provenance if present
        if not isinstance(node, dict) and hasattr(node, 'provenance') and node.provenance:
            properties.update(node.provenance)

        # Build labels string
        labels = [node_type] if node_type else []

        # Build query
        query = (
            f"MERGE (n:{':'.join(labels)} {{id: $id}}) "
            "SET n += $properties "
            "RETURN n"
        )

        return query, {
            "id": node_id,
            "properties": properties
        }

    def _build_relationship_query(self, rel: RelationshipInstance) -> Tuple[str, Dict[str, Any]]:
        """Build a Cypher query for creating a relationship"""
        if isinstance(rel, dict):
            source_id = rel.get('source_id') or rel.get('source')
            target_id = rel.get('target_id') or rel.get('target')
            rel_id = rel.get('id', str(uuid.uuid4()))
            rel_type = rel.get('type') or rel.get('relationship_type')
            rel_properties = rel.get('properties', {})
        else:
            source_id = rel.source if hasattr(rel, 'source') else rel.source_id
            target_id = rel.target if hasattr(rel, 'target') else rel.target_id
            rel_id = rel.id
            rel_type = rel.type if hasattr(rel, 'type') else rel.relationship_type
            rel_properties = rel.properties

        # Ensure rel_properties is a dict
        if rel_properties is None:
            rel_properties = {}
            
        # Sanitize properties
        sanitized_properties = {}
        for key, value in rel_properties.items():
            if isinstance(value, (dict, list)):
                sanitized_properties[key] = json.dumps(value)
            elif value is None:
                # Skip None values
                continue
            else:
                sanitized_properties[key] = value
        
        query = """
        MATCH (s), (t)
        WHERE s.id = $source_id AND t.id = $target_id
        CREATE (s)-[r:`{}`]->(t)
        SET r = $properties, r.id = $rel_id
        RETURN r
        """.format(rel_type)
        
        return query, {
            "source_id": source_id,
            "target_id": target_id,
            "rel_id": rel_id,
            "properties": sanitized_properties
        }

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
                        query, params = self._build_node_query(node, transform_id)
                        await session.run(query, params)

                await self._execute_with_retry(_execute_query)
                items_processed += 1
            except (StorageError, DatabaseError) as e:
                traceback.print_exc()
                success = False
                error_message = str(e)
                if isinstance(node, dict):
                    node_id = node.get('id', 'unknown')
                else:
                    node_id = node.id
                logger.error(f"Failed to store node {node_id}: {error_message}")
                warnings.append(f"Failed to store node {node_id}: {error_message}")
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
            stored_rels = set()
            try:
                if rel.id in stored_rels:
                    continue
                async def _execute_query():
                    async with self._get_session() as session:
                        query, params = self._build_relationship_query(rel)
                        await session.run(query, params)
                        stored_rels.add(rel.id)
                
                await self._execute_with_retry(_execute_query)
                items_processed += 1
            except (StorageError, DatabaseError) as e:
                traceback.print_exc()
                success = False
                error_message = str(e)
                if isinstance(rel, dict):
                    rel_id = rel.get('id', 'unknown')
                else:
                    rel_id = rel.id
                logger.error(f"Failed to store relationship {rel_id}: {error_message}")
                warnings.append(f"Failed to store relationship {rel_id}: {error_message}")
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
                MATCH (c:__Checkpoint__ {transform_id: $transform_id})
                RETURN c ORDER BY c.timestamp DESC LIMIT 1
                """
                result = await session.run(query, {"transform_id": transform_id})
                records = []
                async for record in result:
                    records.append(record)
                
                if not records:
                    return None
                    
                checkpoint = records[0]["c"]
                # Convert Neo4j DateTime to Python datetime if needed
                timestamp = checkpoint["timestamp"]
                if hasattr(timestamp, 'to_native'):
                    timestamp = timestamp.to_native()
                elif hasattr(timestamp, 'to_datetime'):
                    timestamp = timestamp.to_datetime()
                else:
                    # Fallback to current time if conversion fails
                    timestamp = datetime.now()
                
                return StorageCheckpoint(
                    transform_id=checkpoint["transform_id"],
                    last_processed_index=checkpoint["last_processed_index"],
                    stage=StorageStage(checkpoint["stage"]),
                    timestamp=timestamp
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
                    MERGE (c:__Checkpoint__ {transform_id: $transform_id})
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
        try:
            # Get nodes
            query = """
            MATCH (n)
            WHERE n.transform_id = $transform_id
            RETURN n, n.id as node_id, labels(n) as labels
            """
            
            nodes = []
            async with self.driver.session() as session:
                result = await session.run(query, transform_id=transform_id)
                async for record in result:
                    node = record["n"]
                    node_dict = dict(node)
                    # Add the id field if it doesn't exist
                    if "id" not in node_dict:
                        node_dict["id"] = node_dict.get("id", str(record["node_id"]))
                    node_dict["type"] = str(record["labels"][0])
                    nodes.append(node_dict)
            
            # Get relationships
            query = """
            MATCH (s)-[r]->(t)
            WHERE s.transform_id = $transform_id OR t.transform_id = $transform_id
            RETURN r, r.id as rel_id, s.id as source_id, t.id as target_id, type(r) as type
            """
            
            relationships = []
            async with self.driver.session() as session:
                result = await session.run(query, transform_id=transform_id)
                async for record in result:
                    rel = record["r"]
                    rel_dict = dict(rel)
                    # Add the id field if it doesn't exist
                    if "id" not in rel_dict:
                        rel_dict["id"] = rel_dict.get("id", str(record["rel_id"]))
                    rel_dict["source"] = str(record["source_id"])
                    rel_dict["target"] = str(record["target_id"])
                    rel_dict["type"] = str(record["type"])
                    relationships.append(rel_dict)
            
            return TransformationResult(
                transform_id=transform_id,
                nodes=nodes,
                relationships=relationships,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error getting transformation data: {str(e)}")
            raise DatabaseError(f"Failed to get transformation data: {str(e)}")

    async def get_production_graph_for_transform(
        self,
        transform_id: str
    ) -> TransformationResult:
        """Get all nodes and relationships from production that were affected by a transform"""
        try:
            # Get nodes that were affected by the transform
            # This includes nodes that were created or updated during the merge
            query = """
            MATCH (n)
            WHERE n.transform_id = $transform_id OR n.affected_by_transform = $transform_id
            RETURN n, n.id as node_id, labels(n) as labels
            """
            
            nodes = []
            async with self.driver.session() as session:
                result = await session.run(query, transform_id=transform_id)
                async for record in result:
                    node = record["n"]
                    node_dict = dict(node)
                    # Add the id field if it doesn't exist
                    if "id" not in node_dict:
                        node_dict["id"] = node_dict.get("id", str(record["node_id"]))
                    node_dict["type"] = str(record["labels"][0])
                    nodes.append(node_dict)
            
            # Get relationships that were affected by the transform
            query = """
            MATCH (s)-[r]->(t)
            WHERE s.transform_id = $transform_id OR t.transform_id = $transform_id
            RETURN r, r.id as rel_id, s.id as source_id, t.id as target_id, type(r) as type
            """
            
            relationships = []
            async with self.driver.session() as session:
                result = await session.run(query, transform_id=transform_id)
                async for record in result:
                    rel = record["r"]
                    rel_dict = dict(rel)
                    # Add the id field if it doesn't exist
                    if "id" not in rel_dict:
                        rel_dict["id"] = rel_dict.get("id", str(record["rel_id"]))
                    rel_dict["source"] = str(record["source_id"])
                    rel_dict["target"] = str(record["target_id"])
                    rel_dict["type"] = str(record["type"])
                    relationships.append(rel_dict)
            
            return TransformationResult(
                transform_id=transform_id,
                nodes=nodes,
                relationships=relationships,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error getting production graph for transform: {str(e)}")
            raise DatabaseError(f"Failed to get production graph for transform: {str(e)}")

    async def get_nodes_by_property(
        self,
        property_name: str,
        property_value: Any
    ) -> List[Node]:
        """Get all nodes with the specified property value"""
        async def _execute_query():
            query = f"""
            MATCH (n)
            WHERE n.{property_name} = $value
            RETURN n
            """
            records = await self._execute_query(query, {"value": property_value})
            return [
                Node(
                    id=dict(record[0].items()).get("id", str(uuid.uuid4())),
                    label=record[0].get("type", list(record[0].labels)[0]),  # Use type property if available, fallback to first label
                    type=record[0].get("type", list(record[0].labels)[0]),  # Use same value for type
                    properties={k: v for k, v in dict(record[0].items()).items() if k != "type"}
                )
                for record in records
            ]
        
        return await self._execute_with_retry(_execute_query)

    async def get_relationships_between(
        self,
        source_id: str,
        target_id: str
    ) -> List[Edge]:
        """Get all relationships between two nodes"""
        query = """
                MATCH (s)-[r]->(t)
                WHERE s.id = $source_id AND t.id = $target_id
                RETURN r
                """
        records = await self._execute_query(query, {
            "source_id": source_id,
            "target_id": target_id
        })

        edges = []
        for record in records:
            source_id = dict(record[0].start_node.items()).get("id", '')
            target_id = dict(record[0].end_node.items()).get("id", '')
            edges.append(Edge(
                id=dict(record[0].items()).get("id", str(uuid.uuid4())),
                source=source_id,
                target=target_id,
                type=record[0].type,
                properties=dict(record[0].items())
            ))
        return edges

    async def get_relationships_between_nodes(
        self,
        node_ids: List[str]
    ) -> List[Edge]:
        """Get all relationships between a set of nodes"""
        query = """
                MATCH (s)-[r]->(t)
                WHERE s.id IN $node_ids OR t.id IN $node_ids
                RETURN r, s as start_node, t as end_node
                """
        records = await self._execute_query(query, {"node_ids": node_ids})

        edges = []
        for record in records:
            source_id = dict(record[0].start_node.items()).get("id", '')
            target_id = dict(record[0].end_node.items()).get("id", '')
            edges.append(Edge(
                id=dict(record[0].items()).get("id", str(uuid.uuid4())),
                source=source_id,
                target=target_id,
                type=record[0].type,
                properties=dict(record[0].items())
            ))
        return edges

    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True
    ) -> List[Node]:
        """Find nodes with matching property value"""
        async def _execute_query():
            if exact_match:
                query = f"""
                MATCH (n:{label})
                WHERE n.{property_name} = $value
                RETURN n
                """
            else:
                query = f"""
                MATCH (n:{label})
                WHERE n.{property_name} =~ $value
                RETURN n
                """
            
            params = {
                "value": property_value if exact_match else f"(?i).*{property_value}.*"
            }
            
            records = await self._execute_query(query, params)
            
            return [
                Node(
                    id=dict(record[0].items()).get("id", ''),
                    label=record[0].get("type", list(record[0].labels)[0]),  # Use type property if available, fallback to first label
                    type=record[0].get("type", list(record[0].labels)[0]),  # Use same value for type
                    properties={k: v for k, v in dict(record[0].items()).items() if k != "type"}
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
        # Build query parameters for each property
        params = {
            "include_relationships": include_relationships,
            "threshold": similarity_threshold,
            "max_results": max_results
        }

        # Add source_id parameter if available
        has_source_id = bool(properties.get("id"))
        if has_source_id:
            params["source_id"] = properties["id"]

        property_conditions = []
        for idx, (key, value) in enumerate(properties.items()):
            if key not in ["id", "chunk_ids", "confidence_score", "transform_id", "affected_by_transform", "extraction_timestamp"]:
                param_key = f"value{idx}"
                params[param_key] = str(value)  # Convert all values to strings
                if type(value) == list:
                    property_conditions.append(f"apoc.text.levenshteinSimilarity(apoc.text.join([x IN n.{key} | toString(x)], ','), ${param_key})")
                elif type(value) == dict:
                    property_conditions.append(f"apoc.text.levenshteinSimilarity(apoc.convert.toJson(props[key]), ${param_key})")
                else:
                    property_conditions.append(f"apoc.text.levenshteinSimilarity(toString(coalesce(n.{key}, '')), ${param_key})")

        # Build final query with conditional relationship score calculation
        query = f"""
        MATCH (n:{label})
        WITH n,
             CASE 
                WHEN size([{', '.join(property_conditions)}]) > 0
                THEN reduce(s = 0.0, x IN [{', '.join(property_conditions)}] | s + x) / size([{', '.join(property_conditions)}])
                ELSE 0.0
             END as property_score
        """

        # Only add relationship score calculation if source_id is present
        if has_source_id and include_relationships:
            query += """
            WITH n, property_score,
                 size([(n)-[r]->() WHERE type(r) IN [(s)-[sr]->() WHERE s.id = $source_id | type(sr)] | r]) * 1.0 /
                 CASE 
                    WHEN size([(s)-[sr]->() WHERE s.id = $source_id | sr]) > 0 
                    THEN size([(s)-[sr]->() WHERE s.id = $source_id | sr])
                    ELSE 1.0
                 END as relationship_score
            WITH n, property_score, relationship_score,
                 property_score * 0.7 + relationship_score * 0.3 as similarity_score
            """
        else:
            query += """
            WITH n, property_score as similarity_score
            """

        query += """
        WHERE similarity_score >= $threshold
        RETURN n, similarity_score
        ORDER BY similarity_score DESC
        LIMIT $max_results
        """
        records = await self._execute_query(query, params)

        return [
            Node(
                id=dict(record[0].items()).get("id", ''),
                label=list(record[0].labels)[0] if record[0].labels else None,
                type=list(record[0].labels)[0] if record[0].labels else None,
                properties=dict(record[0].items())
            )
            for record in records
        ]

    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Node:
        """Create a new node"""
        async def _execute_query():
            query = f"""
            CREATE (n:{label})
            SET n = $properties
            RETURN n
            """
            records = await self._execute_query(query, {
                "properties": properties
            })
            
            node_data = records[0][0]
            return Node(
                id=properties.get("id", str(uuid.uuid4())),
                label=list(node_data.labels)[0],
                type=node_data.get("type", ""),
                properties=dict(node_data.items())
            )
        
        return await self._execute_with_retry(_execute_query)

    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        tx=None
    ) -> Node:
        """Update an existing node
        
        Args:
            node_id: ID of the node to update
            properties: Properties to update
            tx: Optional transaction object
            
        Returns:
            Updated Node object
        """
        print(f"DEBUG: Updating node {node_id} with properties: {properties}")
        async def _execute_query(node_id, properties={}, tx=None):
            query = """
            MATCH (n)
            WHERE n.id = $id
            SET n += $properties
            RETURN n
            """
            
            # If a transaction is provided, use it
            if tx:
                if isinstance(tx, str):
                    # If tx is a transaction ID, we need to get the actual transaction object
                    if self.transaction_manager:
                        tx_data = self.transaction_manager.get_transaction(tx)
                        if tx_data and "tx" in tx_data:
                            tx = tx_data["tx"]
                        else:
                            logger.warning(f"Transaction ID {tx} not found, creating new session")
                            # Fall back to creating a new session
                            async with self._get_session() as session:
                                result = await session.run(query, {
                                    "id": node_id,
                                    "properties": properties
                                })
                                records = []
                                async for record in result:
                                    records.append(record)
                                await result.consume()  # Ensure resources are released
                                return records
                    else:
                        logger.warning("Transaction manager not available, creating new session")
                        # Fall back to creating a new session
                        async with self._get_session() as session:
                            result = await session.run(query, {
                                "id": node_id,
                                "properties": properties
                            })
                            records = []
                            async for record in result:
                                records.append(record)
                            await result.consume()  # Ensure resources are released
                            return records
                records = await tx.run(query, {
                    "id": node_id,
                    "properties": properties
                })
                records = await records.values()
            else:
                # Otherwise use the standard query execution
                records = await self._execute_query(query, {
                    "id": node_id,
                    "properties": properties
                })
            
            if not records:
                raise StorageError(f"Node {node_id} not found")
            
            node_data = records[0][0]
            node_properties = dict(node_data.items())
            print(f"DEBUG: Updated node {node_id}, new properties: {node_properties}")
            return Node(
                id=node_data.get("id", ''),
                label=list(node_data.labels)[0],
                type=node_data.get("type", ""),
                properties=node_properties
            )
        
        # If transaction is provided, execute directly without retry
        if tx:
            return await _execute_query(node_id, properties, tx)
        
        # Otherwise use retry mechanism
        return await self._execute_with_retry(lambda: _execute_query(node_id, properties, tx))

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None,
        tx=None
    ) -> Edge:
        """Create a relationship between nodes
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            rel_type: The type of the relationship
            properties: Optional properties for the relationship
            tx: Optional transaction object or ID to use for this operation
            
        Returns:
            Edge: The created relationship
        """
        # Ensure properties is a dict and not None
        if properties is None:
            properties = {}
            
        # Convert any complex objects in properties to strings
        sanitized_properties = {}
        for key, value in properties.items():
            if isinstance(value, (dict, list)):
                sanitized_properties[key] = json.dumps(value)
            elif value is None:
                # Skip None values as Neo4j doesn't handle them well
                continue
            else:
                sanitized_properties[key] = value
                
        async def _execute_query():
            query = """
            MATCH (s), (t)
            WHERE s.id = $source_id AND t.id = $target_id
            CREATE (s)-[r:`{}`]->(t)
            SET r = $properties
            RETURN r
            """.format(rel_type)
            
            records = await self._execute_query(query, {
                "source_id": source_id,
                "target_id": target_id,
                "properties": sanitized_properties
            }, tx=tx)
            
            if not records:
                raise StorageError(f"Failed to create relationship between {source_id} and {target_id} of type {rel_type}")
            
            rel_data = records[0][0]
            return Edge(
                id=sanitized_properties.get("id", str(uuid.uuid4())),
                source=source_id,
                target=target_id,
                type=rel_data.type,
                properties=dict(rel_data.items())
            )
        
        # If we have a transaction, don't use retry mechanism
        if tx:
            return await _execute_query()
        else:
            return await self._execute_with_retry(_execute_query)

    async def update_relationship(
        self,
        rel_id: str,
        properties: Dict[str, Any]
    ) -> Edge:
        """Update an existing relationship"""
        # Ensure properties is a dict and not None
        if properties is None:
            properties = {}
            
        # Convert any complex objects in properties to strings
        sanitized_properties = {}
        for key, value in properties.items():
            if isinstance(value, (dict, list)):
                sanitized_properties[key] = json.dumps(value)
            elif value is None:
                # Skip None values as Neo4j doesn't handle them well
                continue
            else:
                sanitized_properties[key] = value
                
        async def _execute_query():
            query = """
            MATCH ()-[r]->()
            WHERE r.id = $rel_id
            SET r += $properties
            RETURN r, startNode(r) as source, endNode(r) as target
            """
            
            records = await self._execute_query(query, {
                "rel_id": rel_id,
                "properties": sanitized_properties
            })
            
            if not records:
                raise StorageError(f"Relationship {rel_id} not found")
            
            rel_data = records[0][0]
            source_node = records[0][1]
            target_node = records[0][2]
            
            return Edge(
                id=rel_id,
                source=source_node.get("id", ''),
                target=target_node.get("id", ''),
                type=rel_data.type,
                properties=dict(rel_data.items())
            )
        
        return await self._execute_with_retry(_execute_query)

    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID"""
        # print(f"DEBUG: Getting node by ID: {node_id}")
        query = """
                MATCH (n)
                WHERE n.id = $id
                RETURN n
                """
        records = await self._execute_query(query, {"id": node_id})
        if not records:
            # print(f"DEBUG: No node found with ID: {node_id}")
            return None

        record = records[0]
        labels = list(record[0].labels) if record[0].labels else []
        properties = dict(record[0].items())
        # print(f"DEBUG: Found node with ID: {node_id}, properties: {properties}")
        return Node(
            id=properties.get("id", ''),
            label=labels[0] if labels else None,
            type=labels[0] if labels else None,
            properties=properties
        )

    async def get_edges_between(
        self,
        source_id: str,
        target_id: str
    ) -> List[Edge]:
        """Get all edges between two nodes"""
        query = """
                MATCH (s)-[r]->(t)
                WHERE s.id = $source_id AND t.id = $target_id
                RETURN r
                """
        records = await self._execute_query(query, {
            "source_id": source_id,
            "target_id": target_id
        })

        edges = []
        for record in records:
            edges.append(Edge(
                id=dict(record[0].items()).get("id", ''),
                source=source_id,
                target=target_id,
                type=record[0].type,
                properties=dict(record[0].items())
            ))
        return edges

    async def get_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str
    ) -> Optional[Edge]:
        """Get a specific relationship between two nodes by type
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            rel_type: Type of relationship to find
            
        Returns:
            Edge if found, None otherwise
        """
        query = """
                MATCH (s)-[r:`{}`]->(t)
                WHERE s.id = $source_id AND t.id = $target_id
                RETURN r
                """.format(rel_type)
        
        records = await self._execute_query(query, {
            "source_id": source_id,
            "target_id": target_id
        })

        if not records:
            return None
            
        record = records[0]
        return Edge(
            id=dict(record[0].items()).get("id", ''),
            source=source_id,
            target=target_id,
            type=record[0].type,
            properties=dict(record[0].items())
        )

    async def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship by its ID
        
        Args:
            rel_id: ID of the relationship to delete
            
        Returns:
            True if the relationship was deleted, False otherwise
        """
        query = """
                MATCH ()-[r]->()
                WHERE r.id = $rel_id
                DELETE r
                RETURN count(r) as deleted_count
                """
        records = await self._execute_query(query, {"rel_id": rel_id})
        
        if not records:
            return False
            
        return records[0][0] > 0

    async def close(self):
        """Close the database connection"""
        if hasattr(self, 'driver') and self.driver:
            try:
                await self.driver.close()
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error closing Neo4j driver: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self:
            try:
                await self.close()
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error closing session: {str(e)}")
        
    def __del__(self):
        """Cleanup when object is deleted"""
        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                loop.create_task(self.close())
        except RuntimeError:
            traceback.print_exc()
            # No event loop running, which is fine during interpreter shutdown
            pass

    async def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, tx=None) -> List[Any]:
        """Execute a Cypher query and return results"""
        # Check if we have a transaction manager and an active transaction
        if tx is None and self.transaction_manager:
            tx_data = self.transaction_manager.get_current_transaction()
            if tx_data:
                tx = tx_data["tx"]
        
        if tx:
            # Handle the case where tx is a string (transaction ID)
            if isinstance(tx, str):
                # If tx is a transaction ID, we need to get the actual transaction object
                if self.transaction_manager:
                    tx_data = self.transaction_manager.get_transaction(tx)
                    if tx_data and "tx" in tx_data:
                        tx = tx_data["tx"]
                    else:
                        logger.warning(f"Transaction ID {tx} not found, creating new session")
                        # Fall back to creating a new session
                        async with self._get_session() as session:
                            result = await session.run(query, parameters=params)
                            records = []
                            async for record in result:
                                records.append(record)
                            await result.consume()  # Ensure resources are released
                            return records
                else:
                    logger.warning("Transaction manager not available, creating new session")
                    # Fall back to creating a new session
                    async with self._get_session() as session:
                        result = await session.run(query, parameters=params)
                        records = []
                        async for record in result:
                            records.append(record)
                        await result.consume()  # Ensure resources are released
                        return records
            
            # Use the provided transaction object
            result = await tx.run(query, parameters=params)
            records = []
            async for record in result:
                records.append(record)
            await result.consume()  # Ensure resources are released
            return records
        else:
            # Create a new session
            async with self._get_session() as session:
                result = await session.run(query, parameters=params)
                records = []
                async for record in result:
                    records.append(record)
                await result.consume()  # Ensure resources are released
                return records

    async def clear_all(self) -> None:
        """Delete all nodes and relationships in the database"""
        async def _execute_query():
            async with self._get_session() as session:
                # Delete all relationships first
                await session.run("MATCH ()-[r]-() DELETE r")
                # Then delete all nodes
                await session.run("MATCH (n) DELETE n")
        
        await self._execute_with_retry(_execute_query)
