"""Neo4j implementation of graph storage"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import traceback
from contextlib import asynccontextmanager
import asyncio
import uuid
import json
import ast
import re
from datetime import datetime, timezone
import time
from graphora_server.utils.constants import (
    VALID_FROM,
    VALID_TO,
    TRANSFORM_ID,
    MERGE_ID,
    SYSTEM_PROPERTIES,
    get_full_text_index_name,
)
from neo4j import AsyncGraphDatabase, GraphDatabase
from neo4j.exceptions import (
    ServiceUnavailable,
    AuthError,
    DatabaseError,
    SessionExpired,
    TransientError,
)
from neo4j.time import DateTime as Neo4jDateTime
from .interface import GraphStorageInterface
from .models import StorageBatchResult, StorageCheckpoint, StorageStage
from .exceptions import StorageConnectionError, StorageAuthError, StorageError
from graphora_server.services.transform.models import BaseNode, RelationshipInstance
from graphora_server.schemas.graph import GraphResponse, Node, Edge

# Configure logger
logger = logging.getLogger(__name__)

# Pattern for valid Cypher identifiers (labels, relationship types, property names)
# Allows alphanumeric characters and underscores, must start with letter or underscore
CYPHER_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class CypherInjectionError(Exception):
    """Raised when a potential Cypher injection is detected."""

    pass


def validate_cypher_identifier(
    identifier: str, identifier_type: str = "identifier"
) -> str:
    """
    Validate that a string is safe to use as a Cypher identifier (label, relationship type, etc.).

    Args:
        identifier: The string to validate
        identifier_type: Type of identifier for error messages (e.g., "label", "relationship type")

    Returns:
        The validated identifier

    Raises:
        CypherInjectionError: If the identifier contains invalid characters
    """
    if not identifier:
        raise CypherInjectionError(f"Empty {identifier_type} is not allowed")

    if not CYPHER_IDENTIFIER_PATTERN.match(identifier):
        raise CypherInjectionError(
            f"Invalid {identifier_type} '{identifier}': must contain only alphanumeric "
            f"characters and underscores, and must start with a letter or underscore"
        )

    # Additional check for suspiciously long identifiers
    if len(identifier) > 256:
        raise CypherInjectionError(
            f"Invalid {identifier_type} '{identifier[:50]}...': exceeds maximum length of 256 characters"
        )

    return identifier


def validate_cypher_labels(labels: List[str]) -> List[str]:
    """
    Validate a list of Cypher labels.

    Args:
        labels: List of label strings to validate

    Returns:
        List of validated labels

    Raises:
        CypherInjectionError: If any label contains invalid characters
    """
    return [validate_cypher_identifier(label, "label") for label in labels]


class Neo4jStorage(GraphStorageInterface):
    """Neo4j implementation of graph storage"""

    @property
    def capabilities(self):
        from graphora_server.services.storage.capabilities import (
            NEO4J_CAPABILITIES,
        )

        return NEO4J_CAPABILITIES

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        max_retries: int = 3,
        transaction_manager=None,
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
            self.driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

            if not transaction_manager:
                # Create a synchronous driver for testing
                sync_driver = GraphDatabase.driver(uri, auth=(username, password))
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
        for attempt in range(self.max_retries):
            try:
                return await operation()
            except (ServiceUnavailable, SessionExpired, TransientError) as e:
                if attempt < self.max_retries - 1:
                    delay = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise StorageError(
                    f"Operation failed after {self.max_retries} retries: {str(e)}"
                )
            except (DatabaseError, StorageConnectionError) as e:
                traceback.print_exc()
                # Don't retry database or connection errors, just propagate them
                raise StorageError(str(e))
            except Exception as e:
                traceback.print_exc()
                # Don't retry other errors
                raise StorageError(f"Unexpected error: {str(e)}")

    def _build_node_query(
        self,
        node: BaseNode,
        transform_id: str,
        merge_id: Optional[str] = None,
        merge: bool = True,
    ) -> tuple[str, Dict]:
        """Build Cypher query for creating a node with properties"""
        # Extract properties excluding metadata
        if isinstance(node, dict):
            # Handle dictionary input
            node_properties = node.get("properties", {})
            node_type = node.get("type", "")
            node_id = node.get("id", str(uuid.uuid4()))
        else:
            # Handle BaseNode input
            node_properties = node.properties
            node_type = node.type
            node_id = node.id

        properties = {
            k: v
            for k, v in node_properties.items()
            if k not in ["id", "type", TRANSFORM_ID, MERGE_ID]
        }

        # Add transform ID
        properties[TRANSFORM_ID] = transform_id
        if merge_id:
            properties[MERGE_ID] = merge_id

        # Add provenance if present
        if (
            not isinstance(node, dict)
            and hasattr(node, "provenance")
            and node.provenance
        ):
            properties.update(node.provenance)

        # Build labels string with validation to prevent Cypher injection
        labels = [node_type] if node_type else []
        if labels:
            validated_labels = validate_cypher_labels(labels)
            labels_str = ":".join(validated_labels)
        else:
            labels_str = ""

        # Build query
        if labels_str:
            query = (
                f"{'MERGE' if merge else 'CREATE'} (n:{labels_str} {{id: $id}}) "
                "SET n += $properties "
                "RETURN n"
            )
        else:
            query = (
                f"{'MERGE' if merge else 'CREATE'} (n {{id: $id}}) "
                "SET n += $properties "
                "RETURN n"
            )

        return query, {"id": node_id, "properties": properties}

    async def create_or_replace_ft_index_for_node(
        self, index_name: str, entity_name: str, properties: List[str]
    ) -> None:
        """Create a full text index for a node entity"""
        # Validate identifiers to prevent Cypher injection
        validated_index_name = validate_cypher_identifier(index_name, "index name")
        validated_entity_name = validate_cypher_identifier(entity_name, "entity name")
        validated_properties = [
            validate_cypher_identifier(p, "property name") for p in properties
        ]

        async with self._get_session() as session:
            query = f"DROP INDEX {validated_index_name} IF EXISTS;"
            await session.run(query)
            node_alias = "n"
            prop_names = [f"{node_alias}.{prop}" for prop in validated_properties]
            query = f"CREATE FULLTEXT INDEX {validated_index_name} FOR ({node_alias}:`{validated_entity_name}`) ON EACH [{', '.join(prop_names)}];"
            logger.debug(query)
            await session.run(query)

    async def create_or_replace_ft_index_for_relationship(
        self,
        index_name: str,
        source_name: str,
        rel_name: str,
        target_name: str,
        properties: List[str],
    ) -> None:
        """Create a full text index for a relationship entity"""
        # Validate identifiers to prevent Cypher injection
        validated_index_name = validate_cypher_identifier(index_name, "index name")
        validated_rel_name = validate_cypher_identifier(rel_name, "relationship type")
        validated_properties = [
            validate_cypher_identifier(p, "property name") for p in properties
        ]

        async with self._get_session() as session:
            try:
                # Drop existing index if it exists
                query = f"DROP INDEX {validated_index_name} IF EXISTS;"
                await session.run(query)

                # Create new index with correct syntax
                # Neo4j relationship index syntax requires direction and variable naming
                rel_alias = "r"
                prop_names = [f"{rel_alias}.{prop}" for prop in validated_properties]

                if not prop_names:
                    logger.warning(
                        f"No properties provided for full-text index {validated_index_name}, skipping creation"
                    )
                    return

                # Use correct syntax with direction and properly formatted relationship pattern
                query = f"""
                CREATE FULLTEXT INDEX {validated_index_name}
                FOR ()-[{rel_alias}:`{validated_rel_name}`]->()
                ON EACH [{', '.join(prop_names)}];
                """
                logger.debug(query)
                await session.run(query)
                logger.info(
                    f"Created full-text index {validated_index_name} for relationship {validated_rel_name}"
                )
            except Exception as e:
                logger.error(
                    f"Error creating full-text index for relationship: {str(e)}"
                )
                # Don't raise the exception - we want to continue even if index creation fails

    async def get_all_node_properties(self, entity_name: str) -> List[str]:
        """
        Get all properties of a node entity from the database.

        Args:
            entity_name: Name of the entity to get properties for

        Returns:
            List of property names, excluding system properties
        """
        # Validate entity name to prevent Cypher injection
        validated_entity_name = validate_cypher_identifier(entity_name, "entity name")

        async with self._get_session() as session:
            try:
                query = f"MATCH (n:`{validated_entity_name}`) RETURN keys(n) as props LIMIT 1"
                result = await session.run(query)
                record = await result.single(None)
                if record:
                    # Get all properties excluding system properties
                    all_props = record["props"]
                    return [prop for prop in all_props if prop not in SYSTEM_PROPERTIES]
                return []
            except Exception as e:
                logger.error(
                    f"Error getting node properties for {validated_entity_name}: {str(e)}"
                )
                return []

    async def get_all_relationship_properties(self, rel_name: str) -> List[str]:
        """
        Get all properties of a relationship type from the database.

        Args:
            rel_name: Name of the relationship type to get properties for

        Returns:
            List of property names, excluding system properties
        """
        # Validate relationship name to prevent Cypher injection
        validated_rel_name = validate_cypher_identifier(rel_name, "relationship type")

        async with self._get_session() as session:
            try:
                query = f"MATCH ()-[r:`{validated_rel_name}`]->() RETURN keys(r) as props LIMIT 1"
                result = await session.run(query)
                record = await result.single(None)
                if record:
                    # Get all properties excluding system properties
                    all_props = record["props"]
                    return [prop for prop in all_props if prop not in SYSTEM_PROPERTIES]
                return []
            except Exception as e:
                logger.error(
                    f"Error getting relationship properties for {validated_rel_name}: {str(e)}"
                )
                return []

    async def store_nodes(
        self,
        nodes: List[BaseNode],
        batch_index: int,
        transform_id: str,
        merge_id: Optional[str] = None,
        merge: bool = True,
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
                        query, params = self._build_node_query(
                            node, transform_id, merge_id, merge
                        )
                        await session.run(query, params)

                await self._execute_with_retry(_execute_query)
                items_processed += 1
            except (StorageError, DatabaseError) as e:
                traceback.print_exc()
                success = False
                error_message = str(e)
                if isinstance(node, dict):
                    node_id = node.get("id", "unknown")
                else:
                    node_id = node.id
                logger.error(f"Failed to store node {node_id}: {error_message}")
                warnings.append(f"Failed to store node {node_id}: {error_message}")
                break

        # Only update checkpoint if at least one node was stored successfully
        if items_processed > 0:
            try:
                checkpoint_result = await self.update_checkpoint(
                    transform_id, batch_index, StorageStage.NODES
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
            warnings.append(
                f"Partial batch failure: {items_processed} of {len(nodes)} nodes stored"
            )

        processing_time_ms = (time.time() - start_time) * 1000
        return StorageBatchResult(
            batch_index=batch_index,
            items_processed=items_processed,
            processing_time_ms=processing_time_ms,
            success=success,
            error=error_message,
            warnings=warnings,
        )

    async def store_relationships(
        self,
        relationships: List[RelationshipInstance],
        batch_index: int,
        transform_id: str,
        merge_id: Optional[str] = None,
        merge: bool = True,
    ) -> StorageBatchResult:
        """Store relationships in Neo4j with versioning logic"""
        start_time = time.time()
        success = True
        error_message = None
        items_processed = 0
        warnings = []
        stored_rels = set()

        for rel in relationships:
            if rel.id in stored_rels:
                logger.debug(f"Skipping duplicate relationship ID: {rel.id}")
                continue

            try:

                async def _execute_relationship():
                    async with self._get_session() as session:
                        # Check for existing relationship
                        existing_rel = await self._find_existing_relationship(
                            session, rel
                        )

                        if existing_rel:
                            # Case 1: Existing with no properties beyond valid_from/valid_to/transform_id/merge_id
                            existing_props = {
                                k: v
                                for k, v in existing_rel.get("properties", {}).items()
                                if k
                                not in {VALID_FROM, VALID_TO, TRANSFORM_ID, MERGE_ID}
                            }
                            if not existing_props:
                                logger.debug(
                                    f"Existing relationship {rel.id} has no meaningful properties, skipping"
                                )
                                stored_rels.add(rel.id)
                                return  # Skip adding new relationship

                            # Case 2: Existing with differing properties
                            new_props = {
                                k: v
                                for k, v in rel.properties.items()
                                if k
                                not in {VALID_FROM, VALID_TO, TRANSFORM_ID, MERGE_ID}
                            }
                            if existing_props != new_props:
                                # Version the existing relationship
                                logger.debug(
                                    f"Versioning relationship {rel.id} due to property differences"
                                )
                                await self._close_existing_relationship(
                                    session, existing_rel
                                )
                                # Merge properties and create new version
                                merged_props = {
                                    **existing_props,
                                    **new_props,
                                    VALID_FROM: datetime.now().isoformat(),
                                    VALID_TO: None,
                                    TRANSFORM_ID: transform_id,
                                    MERGE_ID: merge_id if merge_id else None,
                                }
                                query, params = self._build_relationship_query(
                                    rel,
                                    merge=True,
                                    properties=merged_props,
                                    transform_id=transform_id,
                                    merge_id=merge_id,
                                )
                                logger.debug(
                                    f"Creating new version: Query={query}, Params={params}"
                                )
                                await session.run(query, params)
                                stored_rels.add(rel.id)
                            else:
                                logger.debug(
                                    f"Relationship {rel.id} properties unchanged, keeping existing"
                                )
                                stored_rels.add(rel.id)  # No change needed
                        else:
                            # Case 3: No existing relationship
                            logger.debug(
                                f"No existing relationship found for {rel.id}, creating new"
                            )
                            props_with_metadata = {
                                **rel.properties,
                                VALID_FROM: datetime.now().isoformat(),
                                VALID_TO: None,
                                TRANSFORM_ID: transform_id,
                                MERGE_ID: merge_id if merge_id else None,
                            }
                            query, params = self._build_relationship_query(
                                rel,
                                merge=True,
                                properties=props_with_metadata,
                                transform_id=transform_id,
                                merge_id=merge_id,
                            )
                            logger.debug(
                                f"New relationship: Query={query}, Params={params}"
                            )
                            await session.run(query, params)
                            stored_rels.add(rel.id)

                await self._execute_with_retry(_execute_relationship)
                items_processed += 1
                logger.info(f"Stored relationships: {stored_rels}")
            except (StorageError, DatabaseError) as e:
                traceback.print_exc()
                success = False
                error_message = str(e)
                logger.error(f"Failed to store relationship {rel.id}: {error_message}")
                warnings.append(
                    f"Failed to store relationship {rel.id}: {error_message}"
                )
                break
            except Exception as e:
                traceback.print_exc()
                success = False
                error_message = f"Unexpected error: {str(e)}"
                logger.error(
                    f"Unexpected failure for relationship {rel.id}: {error_message}"
                )
                warnings.append(
                    f"Unexpected failure for relationship {rel.id}: {error_message}"
                )
                break

        # Update checkpoint if successful
        if items_processed > 0:
            try:
                checkpoint_result = await self.update_checkpoint(
                    transform_id, batch_index, "RELATIONSHIPS"
                )
                if not checkpoint_result.success:
                    success = False
                    error_message = checkpoint_result.error
                    logger.error(f"Error updating checkpoint: {error_message}")
                    warnings.append(f"Checkpoint update failed: {error_message}")
            except (StorageError, DatabaseError) as e:
                success = False
                error_message = str(e)
                logger.error(f"Failed to update checkpoint: {error_message}")
                warnings.append(f"Checkpoint update failed: {error_message}")

        if items_processed < len(relationships):
            warnings.append(
                f"Partial batch failure: {items_processed} of {len(relationships)} relationships stored"
            )

        processing_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Stored {items_processed} relationships in {processing_time_ms:.2f}ms"
        )
        return StorageBatchResult(
            batch_index=batch_index,
            items_processed=items_processed,
            processing_time_ms=processing_time_ms,
            success=success,
            error=error_message,
            warnings=warnings,
        )

    async def _find_existing_relationship(
        self, session, rel: RelationshipInstance
    ) -> Optional[Dict]:
        """Check for an existing relationship in Neo4j"""
        # Validate relationship type to prevent Cypher injection
        validated_rel_type = validate_cypher_identifier(rel.type, "relationship type")

        query = f"""
        MATCH (s)-[r:`{validated_rel_type}`]->(t)
        WHERE s.id = $source_id AND t.id = $target_id AND r.{VALID_TO} IS NULL
        RETURN r
        """
        result = await session.run(
            query, source_id=rel.source_id, target_id=rel.target_id
        )
        record = await result.single()
        return record["r"] if record else None

    async def _close_existing_relationship(self, session, existing_rel: Dict):
        """Set valid_to on an existing relationship"""
        query = f"""
        MATCH ()-[r]->()
        WHERE r.id = $rel_id
        SET r.{VALID_TO} = $valid_to
        """
        await session.run(
            query,
            rel_id=existing_rel["id"],
            valid_to=datetime.now(timezone.utc).isoformat(),
        )

    def _build_relationship_query(
        self,
        rel: RelationshipInstance,
        merge: bool = True,
        properties: Optional[Dict] = None,
        transform_id: str = None,
        merge_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a Cypher query for creating or versioning a relationship"""
        source_id = rel.source_id
        target_id = rel.target_id
        rel_id = rel.id if merge else str(uuid.uuid4())

        # Validate relationship type to prevent Cypher injection
        validated_rel_type = validate_cypher_identifier(rel.type, "relationship type")

        rel_properties = properties if properties is not None else rel.properties

        # Sanitize properties
        sanitized_properties = {}
        for key, value in rel_properties.items():
            if isinstance(value, (dict, list)):
                sanitized_properties[key] = json.dumps(value)
            elif value is None:
                continue
            else:
                sanitized_properties[key] = value

        # Add versioning properties
        sanitized_properties[VALID_FROM] = (
            datetime.now(timezone.utc).isoformat()
            if not (rel.properties.get(VALID_FROM))
            else rel.properties.get(VALID_FROM)
        )
        sanitized_properties[VALID_TO] = None
        if transform_id:
            sanitized_properties[TRANSFORM_ID] = transform_id
        if merge_id:
            sanitized_properties[MERGE_ID] = merge_id

        query = f"""
        MATCH (s), (t)
        WHERE s.id = $source_id AND t.id = $target_id
        {"MERGE" if merge else "CREATE"} (s)-[r:`{validated_rel_type}`]->(t)
        SET r = $properties, r.id = $rel_id
        RETURN r
        """

        return query, {
            "source_id": source_id,
            "target_id": target_id,
            "rel_id": rel_id,
            "properties": sanitized_properties,
        }

    async def get_storage_status(
        self, transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status"""

        async def _execute_query():
            async with self._get_session() as session:
                query = f"""
                MATCH (c:__Checkpoint__ {{{TRANSFORM_ID}: $transform_id}})
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
                if hasattr(timestamp, "to_native"):
                    timestamp = timestamp.to_native()
                elif hasattr(timestamp, "to_datetime"):
                    timestamp = timestamp.to_datetime()
                else:
                    # Fallback to current time if conversion fails
                    timestamp = datetime.now()

                return StorageCheckpoint(
                    transform_id=checkpoint[TRANSFORM_ID],
                    last_processed_index=checkpoint["last_processed_index"],
                    stage=StorageStage(checkpoint["stage"]),
                    timestamp=timestamp,
                )

        return await self._execute_with_retry(_execute_query)

    async def update_checkpoint(
        self, transform_id: str, last_index: int, stage: StorageStage
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
                        stage=stage,
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
            warnings=[],
        )

    async def get_transformation_data(self, transform_id: str) -> GraphResponse:
        """Get all nodes and relationships for a transformation"""
        try:
            count_query = f"""
            MATCH (n)
            WHERE n.{TRANSFORM_ID} = $transform_id
            WITH count(n) as node_count
            OPTIONAL MATCH (n)-[r]-()
            RETURN node_count, count(DISTINCT r) as edge_count
            """

            async with self.driver.session() as session:
                count_result = await session.run(count_query, transform_id=transform_id)
                count_data = await count_result.single()
                total_nodes = count_data["node_count"]
                total_edges = count_data["edge_count"]

                query = f"""
                MATCH (n)
                WHERE n.{TRANSFORM_ID} = $transform_id
                WITH n ORDER BY n.id
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN 
                    collect(DISTINCT n) as nodes,
                    collect(DISTINCT r) as relationships,
                    collect(DISTINCT m) as connected_nodes
                """

                result = await session.run(query, transform_id=transform_id)
                data = await result.single()

                nodes_list = []
                edges_list = []
                seen_nodes = set()
                seen_edges = set()

                def get_actual_label(node_labels):
                    return list(node_labels)[0]

                def convert_datetime(value):
                    """Convert neo4j.time.DateTime to ISO string"""
                    if isinstance(value, Neo4jDateTime):
                        return value.iso_format()
                    return value

                def extract_properties(entity):
                    """Extract properties, converting Neo4j DateTime to strings"""
                    props = {}
                    entity_dict = dict(entity)
                    for key, value in entity_dict.items():
                        if isinstance(value, Neo4jDateTime):
                            props[key] = value.iso_format()
                        elif isinstance(value, str):
                            try:
                                if value.startswith("[") or value.startswith("{"):
                                    value = ast.literal_eval(value)
                            except (ValueError, SyntaxError):
                                pass
                            props[key] = value
                        else:
                            props[key] = value
                    # Log for debugging
                    logger.debug(f"Extracted properties: {props}")
                    return props

                # Process main nodes
                for node in data["nodes"]:
                    node_id = convert_datetime(node.get("id"))
                    if node_id and node_id not in seen_nodes:
                        actual_label = get_actual_label(node.labels)
                        node_props = extract_properties(node)
                        node_dict = {
                            "id": node_id,
                            "label": actual_label,
                            "properties": node_props,
                            "type": actual_label,
                        }
                        nodes_list.append(node_dict)
                        seen_nodes.add(node_id)
                        # Debug: Check raw node data
                        logger.debug(f"Node: {node_dict}")

                # Process connected nodes
                for node in data["connected_nodes"]:
                    if node is not None:
                        node_id = convert_datetime(node.get("id"))
                        if node_id and node_id not in seen_nodes:
                            actual_label = get_actual_label(node.labels)
                            node_props = extract_properties(node)
                            node_dict = {
                                "id": node_id,
                                "label": actual_label,
                                "properties": node_props,
                                "type": actual_label,
                            }
                            nodes_list.append(node_dict)
                            seen_nodes.add(node_id)
                            logger.debug(f"Connected Node: {node_dict}")

                # Process relationships
                for rel in data["relationships"]:
                    if rel is not None:
                        edge_id = convert_datetime(rel.get("id", str(rel.id)))
                        if edge_id not in seen_edges:
                            source_id = convert_datetime(rel.start_node.get("id"))
                            target_id = convert_datetime(rel.end_node.get("id"))
                            if source_id and target_id:
                                edge_props = extract_properties(rel)
                                edge_dict = {
                                    "id": edge_id,
                                    "source": source_id,
                                    "target": target_id,
                                    "type": str(rel.type),
                                    "properties": edge_props,
                                }
                                edges_list.append(edge_dict)
                                seen_edges.add(edge_id)
                                logger.debug(f"Edge: {edge_dict}")

                # Create GraphResponse and test serialization
                response = GraphResponse(
                    nodes=nodes_list,
                    edges=edges_list,
                    total_nodes=total_nodes,
                    total_edges=total_edges,
                )
                return response

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error retrieving graph data: {str(e)}")
            raise

    async def get_merge_data(self, merge_id: str) -> GraphResponse:
        """Get all nodes and relationships for a merge"""
        try:
            count_query = f"""
            MATCH (n)
            WHERE n.{MERGE_ID} = $merge_id
            WITH count(n) as node_count
            OPTIONAL MATCH (n)-[r]-()
            RETURN node_count, count(DISTINCT r) as edge_count
            """

            async with self.driver.session() as session:
                count_result = await session.run(count_query, merge_id=merge_id)
                count_data = await count_result.single()
                total_nodes = count_data["node_count"]
                total_edges = count_data["edge_count"]

                query = f"""
                MATCH (n)
                WHERE n.{MERGE_ID} = $merge_id
                WITH n ORDER BY n.id
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE r.{VALID_TO} IS NULL 
                RETURN 
                    collect(DISTINCT n) as nodes,
                    collect(DISTINCT r) as relationships,
                    collect(DISTINCT m) as connected_nodes
                """

                result = await session.run(query, merge_id=merge_id)
                data = await result.single()

                nodes_list = []
                edges_list = []
                seen_nodes = set()
                seen_edges = set()

                def get_actual_label(node_labels):
                    return list(node_labels)[0]

                def convert_datetime(value):
                    """Convert neo4j.time.DateTime to ISO string"""
                    if isinstance(value, Neo4jDateTime):
                        return value.iso_format()
                    return value

                def extract_properties(entity):
                    """Extract properties, converting Neo4j DateTime to strings"""
                    props = {}
                    entity_dict = dict(entity)
                    for key, value in entity_dict.items():
                        if isinstance(value, Neo4jDateTime):
                            props[key] = value.iso_format()
                        elif isinstance(value, str):
                            try:
                                if value.startswith("[") or value.startswith("{"):
                                    value = ast.literal_eval(value)
                            except (ValueError, SyntaxError):
                                pass
                            props[key] = value
                        else:
                            props[key] = value
                    # Log for debugging
                    logger.debug(f"Extracted properties: {props}")
                    return props

                # Process main nodes
                for node in data["nodes"]:
                    node_id = convert_datetime(node.get("id"))
                    if node_id and node_id not in seen_nodes:
                        actual_label = get_actual_label(node.labels)
                        node_props = extract_properties(node)
                        node_dict = {
                            "id": node_id,
                            "label": actual_label,
                            "properties": node_props,
                            "type": actual_label,
                        }
                        nodes_list.append(node_dict)
                        seen_nodes.add(node_id)
                        # Debug: Check raw node data
                        logger.debug(f"Node: {node_dict}")

                # Process connected nodes
                for node in data["connected_nodes"]:
                    if node is not None:
                        node_id = convert_datetime(node.get("id"))
                        if node_id and node_id not in seen_nodes:
                            actual_label = get_actual_label(node.labels)
                            node_props = extract_properties(node)
                            node_dict = {
                                "id": node_id,
                                "label": actual_label,
                                "properties": node_props,
                                "type": actual_label,
                            }
                            nodes_list.append(node_dict)
                            seen_nodes.add(node_id)
                            logger.debug(f"Connected Node: {node_dict}")

                # Process relationships
                for rel in data["relationships"]:
                    if rel is not None:
                        edge_id = convert_datetime(rel.get("id", str(rel.id)))
                        if edge_id not in seen_edges:
                            source_id = convert_datetime(rel.start_node.get("id"))
                            target_id = convert_datetime(rel.end_node.get("id"))
                            if source_id and target_id:
                                edge_props = extract_properties(rel)
                                edge_dict = {
                                    "id": edge_id,
                                    "source": source_id,
                                    "target": target_id,
                                    "type": str(rel.type),
                                    "properties": edge_props,
                                }
                                edges_list.append(edge_dict)
                                seen_edges.add(edge_id)
                                logger.debug(f"Edge: {edge_dict}")

                # Create GraphResponse and test serialization
                response = GraphResponse(
                    nodes=nodes_list,
                    edges=edges_list,
                    total_nodes=total_nodes,
                    total_edges=total_edges,
                )
                return response

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error retrieving graph data: {str(e)}")
            raise

    async def get_nodes_by_property(
        self, property_name: str, property_value: Any
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
                    label=record[0].get(
                        "type", list(record[0].labels)[0]
                    ),  # Use type property if available, fallback to first label
                    type=record[0].get(
                        "type", list(record[0].labels)[0]
                    ),  # Use same value for type
                    properties={
                        k: v for k, v in dict(record[0].items()).items() if k != "type"
                    },
                )
                for record in records
            ]

        return await self._execute_with_retry(_execute_query)

    async def get_relationships_between(
        self, source_id: str, target_id: str
    ) -> List[Edge]:
        """Get all relationships between two nodes"""
        query = """
                MATCH (s)-[r]->(t)
                WHERE s.id = $source_id AND t.id = $target_id
                RETURN r
                """
        records = await self._execute_query(
            query, {"source_id": source_id, "target_id": target_id}
        )

        edges = []
        for record in records:
            source_id = dict(record[0].start_node.items()).get("id", "")
            target_id = dict(record[0].end_node.items()).get("id", "")
            edges.append(
                Edge(
                    id=dict(record[0].items()).get("id", str(uuid.uuid4())),
                    source=source_id,
                    target=target_id,
                    type=record[0].type,
                    properties=dict(record[0].items()),
                )
            )
        return edges

    async def get_relationships_between_nodes(self, node_ids: List[str]) -> List[Edge]:
        """Get all relationships between a set of nodes"""
        query = """
                MATCH (s)-[r]->(t)
                WHERE s.id IN $node_ids OR t.id IN $node_ids
                RETURN r, s as start_node, t as end_node
                """
        records = await self._execute_query(query, {"node_ids": node_ids})

        edges = []
        for record in records:
            source_id = dict(record[0].start_node.items()).get("id", "")
            target_id = dict(record[0].end_node.items()).get("id", "")
            edges.append(
                Edge(
                    id=dict(record[0].items()).get("id", str(uuid.uuid4())),
                    source=source_id,
                    target=target_id,
                    type=record[0].type,
                    properties=dict(record[0].items()),
                )
            )
        return edges

    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True,
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
                    id=dict(record[0].items()).get("id", ""),
                    label=record[0].get(
                        "type", list(record[0].labels)[0]
                    ),  # Use type property if available, fallback to first label
                    type=record[0].get(
                        "type", list(record[0].labels)[0]
                    ),  # Use same value for type
                    properties={
                        k: v for k, v in dict(record[0].items()).items() if k != "type"
                    },
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
        include_relationships: bool = True,
    ) -> List[Node]:
        """Find nodes with similar properties using fuzzy matching"""
        # Skip if no properties to search
        if not properties:
            return []

        # First try exact match using get_nodes
        exact_matches = await self.get_nodes(label, properties, max_results)
        if exact_matches and len(exact_matches) >= max_results:
            return exact_matches

        # Then try similarity search
        similarity_conditions = []
        params = {
            "include_relationships": include_relationships,
            "threshold": similarity_threshold,
            "max_results": max_results,
        }

        for idx, (key, value) in enumerate(properties.items()):
            if value is not None and key not in SYSTEM_PROPERTIES:
                param_key = f"value{idx}"
                params[param_key] = str(value).lower()  # Convert all values to strings

                # Create individual similarity conditions for each property
                # (Note: This block MUST be inside the for loop to process all properties)
                if isinstance(value, list):
                    similarity_conditions.append(
                        f"CASE WHEN apoc.text.join([x IN n.{key} | toString(x)], ',') = ${param_key} THEN 1.0 ELSE 0.0 END"
                    )
                elif isinstance(value, dict):
                    similarity_conditions.append(
                        f"CASE WHEN apoc.convert.toJson(n.{key}) = ${param_key} THEN 1.0 ELSE 0.0 END"
                    )
                else:
                    # Text distance similarity
                    similarity_conditions.append(
                        f"CASE WHEN apoc.text.distance(toLower(toString(coalesce(n.{key}, ''))), ${param_key}) < 5 THEN 1.0 ELSE 0.0 END"
                    )
                    # Metaphone similarity (phonetic)
                    similarity_conditions.append(
                        f"CASE WHEN apoc.text.doubleMetaphone(toLower(toString(coalesce(n.{key}, '')))) = apoc.text.doubleMetaphone(${param_key}) THEN 1.0 ELSE 0.0 END"
                    )

        # Build final query with conditional relationship score calculation
        if not similarity_conditions:
            return []

        similarity_expr = " + ".join(similarity_conditions)

        query = f"""
        MATCH (n:{label})
        WITH n, ({similarity_expr}) / {len(similarity_conditions)} as similarity_score
        WHERE similarity_score >= $threshold
        RETURN n, similarity_score
        ORDER BY similarity_score DESC
        LIMIT $max_results
        """
        records = await self._execute_query(query, params)

        # Search Full text index of this node and look for fuzzy matches
        # Use label as the entity_type for full-text search
        ft_results = await self._full_text_search(label, properties, max_results)

        # Combine results from both queries and return unique nodes (by ID)
        similarity_results = [
            Node(
                id=dict(record[0].items()).get("id", ""),
                label=list(record[0].labels)[0] if record[0].labels else label,
                type=list(record[0].labels)[0] if record[0].labels else label,
                properties=dict(record[0].items()),
            )
            for record in records
        ]

        # Combine results, prioritizing exact and similarity matches
        combined_results = []
        seen_ids = set()

        # First add exact matches
        for node in exact_matches:
            if node.id not in seen_ids:
                combined_results.append(node)
                seen_ids.add(node.id)

        # Then add similarity matches
        for node in similarity_results:
            if node.id not in seen_ids:
                combined_results.append(node)
                seen_ids.add(node.id)

        # Finally add full-text matches
        for node in ft_results:
            if node.id not in seen_ids:
                combined_results.append(node)
                seen_ids.add(node.id)

        # Limit to max_results
        logger.debug("************************************************")
        logger.debug(combined_results[:max_results])
        logger.debug("************************************************")
        return combined_results[:max_results]

    async def get_nodes(
        self, label: str, properties: Dict[str, Any], limit: int = 10
    ) -> List[Node]:
        """
        Get nodes with exact property matches

        Args:
            label (str): Node label
            properties (Dict[str, Any]): Properties to match
            limit (int, optional): Maximum number of results. Defaults to 10.

        Returns:
            List[Node]: List of matching nodes
        """
        # Skip system properties and None values
        query_props = {
            k: v
            for k, v in properties.items()
            if k not in SYSTEM_PROPERTIES and v is not None
        }

        if not query_props:
            return []

        # Build WHERE clause for exact matches
        where_clauses = []
        params = {"limit": limit}

        for idx, (key, value) in enumerate(query_props.items()):
            param_name = f"prop{idx}"
            params[param_name] = value
            where_clauses.append(f"n.{key} = ${param_name}")

        where_clause = " AND ".join(where_clauses)

        query = f"""
        MATCH (n:{label})
        WHERE {where_clause}
        RETURN n
        LIMIT $limit
        """

        records = await self._execute_query(query, params)

        return [
            Node(
                id=dict(record[0].items()).get("id", ""),
                label=list(record[0].labels)[0] if record[0].labels else label,
                type=list(record[0].labels)[0] if record[0].labels else label,
                properties=dict(record[0].items()),
            )
            for record in records
        ]

    async def create_node(self, label: str, properties: Dict[str, Any]) -> Node:
        """Create a new node"""

        async def _execute_query():
            query = f"""
            CREATE (n:{label})
            SET n = $properties
            RETURN n
            """
            records = await self._execute_query(query, {"properties": properties})

            node_data = records[0][0]
            return Node(
                id=properties.get("id", str(uuid.uuid4())),
                label=list(node_data.labels)[0],
                type=list(node_data.labels)[0],
                properties=dict(node_data.items()),
            )

        return await self._execute_with_retry(_execute_query)

    async def update_node(
        self, node_id: str, properties: Dict[str, Any], tx=None
    ) -> Node:
        """Update an existing node

        Args:
            node_id: ID of the node to update
            properties: Properties to update
            tx: Optional transaction object

        Returns:
            Updated Node object
        """
        logger.debug(f"DEBUG: Updating node {node_id} with properties: {properties}")

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
                            logger.warning(
                                f"Transaction ID {tx} not found, creating new session"
                            )
                            # Fall back to creating a new session
                            async with self._get_session() as session:
                                result = await session.run(
                                    query, {"id": node_id, "properties": properties}
                                )
                                records = []
                                async for record in result:
                                    records.append(record)
                                await result.consume()  # Ensure resources are released
                                return records
                    else:
                        logger.warning(
                            "Transaction manager not available, creating new session"
                        )
                        # Fall back to creating a new session
                        async with self._get_session() as session:
                            result = await session.run(
                                query, {"id": node_id, "properties": properties}
                            )
                            records = []
                            async for record in result:
                                records.append(record)
                            await result.consume()  # Ensure resources are released
                            return records
                records = await tx.run(query, {"id": node_id, "properties": properties})
                records = await records.values()
            else:
                # Otherwise use the standard query execution
                records = await self._execute_query(
                    query, {"id": node_id, "properties": properties}
                )

            if not records:
                raise StorageError(f"Node {node_id} not found")

            node_data = records[0][0]
            node_properties = dict(node_data.items())
            logger.debug(
                f"DEBUG: Updated node {node_id}, new properties: {node_properties}"
            )
            return Node(
                id=node_data.get("id", ""),
                label=list(node_data.labels)[0],
                type=list(node_data.labels)[0],
                properties=node_properties,
            )

        # If transaction is provided, execute directly without retry
        if tx:
            return await _execute_query(node_id, properties, tx)

        # Otherwise use retry mechanism
        return await self._execute_with_retry(
            lambda: _execute_query(node_id, properties, tx)
        )

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None,
        tx=None,
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
        # Validate relationship type to prevent Cypher injection
        validated_rel_type = validate_cypher_identifier(rel_type, "relationship type")

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
            """.format(
                validated_rel_type
            )

            records = await self._execute_query(
                query,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "properties": sanitized_properties,
                },
                tx=tx,
            )

            if not records:
                raise StorageError(
                    f"Failed to create relationship between {source_id} and {target_id} of type {rel_type}"
                )

            rel_data = records[0][0]
            return Edge(
                id=sanitized_properties.get("id", str(uuid.uuid4())),
                source=source_id,
                target=target_id,
                type=rel_data.type,
                properties=dict(rel_data.items()),
            )

        # If we have a transaction, don't use retry mechanism
        if tx:
            return await _execute_query()
        else:
            return await self._execute_with_retry(_execute_query)

    async def update_relationship(
        self, rel_id: str, properties: Dict[str, Any]
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

            records = await self._execute_query(
                query, {"rel_id": rel_id, "properties": sanitized_properties}
            )

            if not records:
                raise StorageError(f"Relationship {rel_id} not found")

            rel_data = records[0][0]
            source_node = records[0][1]
            target_node = records[0][2]

            return Edge(
                id=rel_id,
                source=source_node.get("id", ""),
                target=target_node.get("id", ""),
                type=rel_data.type,
                properties=dict(rel_data.items()),
            )

        return await self._execute_with_retry(_execute_query)

    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID"""
        # logger.debug(f"DEBUG: Getting node by ID: {node_id}")
        query = """
                MATCH (n)
                WHERE n.id = $id
                RETURN n
                """
        records = await self._execute_query(query, {"id": node_id})
        if not records:
            # logger.debug(f"DEBUG: No node found with ID: {node_id}")
            return None

        record = records[0]
        labels = list(record[0].labels) if record[0].labels else []
        properties = dict(record[0].items())
        # logger.debug(f"DEBUG: Found node with ID: {node_id}, properties: {properties}")
        return Node(
            id=properties.get("id", ""),
            label=labels[0] if labels else None,
            type=labels[0] if labels else None,
            properties=properties,
        )

    async def get_edges_between(self, source_id: str, target_id: str) -> List[Edge]:
        """Get all edges between two nodes"""
        query = """
                MATCH (s)-[r]->(t)
                WHERE s.id = $source_id AND t.id = $target_id
                RETURN r
                """
        records = await self._execute_query(
            query, {"source_id": source_id, "target_id": target_id}
        )

        edges = []
        for record in records:
            edges.append(
                Edge(
                    id=dict(record[0].items()).get("id", ""),
                    source=source_id,
                    target=target_id,
                    type=record[0].type,
                    properties=dict(record[0].items()),
                )
            )
        return edges

    async def get_relationship(
        self, source_id: str, target_id: str, rel_type: str
    ) -> Optional[Edge]:
        """Get a specific relationship between two nodes by type

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            rel_type: Type of relationship to find

        Returns:
            Edge if found, None otherwise
        """
        # Validate relationship type to prevent Cypher injection
        validated_rel_type = validate_cypher_identifier(rel_type, "relationship type")

        query = """
                MATCH (s)-[r:`{}`]->(t)
                WHERE s.id = $source_id AND t.id = $target_id
                RETURN r
                """.format(
            validated_rel_type
        )

        records = await self._execute_query(
            query, {"source_id": source_id, "target_id": target_id}
        )

        if not records:
            return None

        record = records[0]
        return Edge(
            id=dict(record[0].items()).get("id", ""),
            source=source_id,
            target=target_id,
            type=record[0].type,
            properties=dict(record[0].items()),
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

    async def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Public method to execute a Cypher query and return results"""
        return await self._execute_query(query, params)

    async def close(self):
        """Close the database connection"""
        if hasattr(self, "driver") and self.driver:
            try:
                await self.driver.close()
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error closing Neo4j driver: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, _exc_tb):
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
            # Instead of trying to get a running loop, use asyncio.run if needed
            # But since we're in __del__, it's better to just log and not try to close
            # as the connection will be closed when the process ends anyway
            logger.debug("Neo4j connection will be closed on process shutdown")
        except Exception:
            # Silently ignore any errors during cleanup
            pass

    async def _execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None, tx=None
    ) -> List[Any]:
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
                        logger.warning(
                            f"Transaction ID {tx} not found, creating new session"
                        )
                        # Fall back to creating a new session
                        async with self._get_session() as session:
                            result = await session.run(query, parameters=params)
                            records = []
                            async for record in result:
                                records.append(record)
                            await result.consume()  # Ensure resources are released
                            return records
                else:
                    logger.warning(
                        "Transaction manager not available, creating new session"
                    )
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

    async def _full_text_search(
        self, label: str, properties: Dict[str, Any], max_results: int = 10
    ) -> List[Node]:
        """
        Perform a full-text search on a node entity using Neo4j's full-text indexes.

        Args:
            label (str): Type of entity to search
            properties (Dict[str, Any]): Properties to search for
            max_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            List[Node]: List of nodes matching the search criteria
        """
        # Validate label to prevent Cypher injection
        validated_label = validate_cypher_identifier(label, "label")

        # Skip if no properties to search
        if not properties:
            return []

        # Get index name for this entity type
        index_name = get_full_text_index_name(validated_label)

        # First check if the index exists - use a more compatible approach
        try:
            # Try a simple query with the index to see if it exists
            # If it fails, we'll catch the exception
            test_query = """
            CALL db.index.fulltext.queryNodes($index_name, "*")
            YIELD node
            RETURN count(node) as count
            LIMIT 1
            """
            await self._execute_query(test_query, {"index_name": index_name})
            # If we get here, the index exists
            logger.debug(f"Full-text index {index_name} exists.")
        except Exception as e:
            # If we get an error about the index not existing, skip full-text search
            error_str = str(e)
            if (
                "no such fulltext schema index" in error_str.lower()
                or "no procedure" in error_str.lower()
            ):
                logger.debug(
                    f"Full-text index {index_name} does not exist. Skipping full-text search."
                )
                return []
            else:
                # Some other error occurred
                logger.debug(f"Error checking if index exists: {error_str}")
                # Continue anyway - we'll try the search and handle errors there
                pass

        # For full-text search, we'll run separate queries for each property
        # and combine the results, as Neo4j's full-text search works best with
        # individual property searches
        all_results = []
        seen_ids = set()

        for key, value in properties.items():
            if key not in SYSTEM_PROPERTIES and value:
                # Skip complex types
                if isinstance(value, (list, dict)):
                    continue

                # Convert to string and escape special Lucene characters
                value_str = str(value)
                # Escape special Lucene characters: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
                lucene_special_chars = [
                    "+",
                    "-",
                    "&",
                    "|",
                    "!",
                    "(",
                    ")",
                    "{",
                    "}",
                    "[",
                    "]",
                    "^",
                    '"',
                    "~",
                    "*",
                    "?",
                    ":",
                    "\\",
                    "/",
                ]
                for char in lucene_special_chars:
                    value_str = value_str.replace(char, f"\\{char}")

                # Try different search approaches
                search_queries = [
                    # Exact match - use double quotes to handle spaces
                    f'{key}:"{value_str}"',
                    # Prefix match with first word only (only if no special chars)
                    (
                        f"{key}:{value_str.split()[0]}*"
                        if value_str.split() and "*" not in value_str.split()[0]
                        else ""
                    ),
                    # Fuzzy match with first word only (only if no special chars)
                    (
                        f"{key}:{value_str.split()[0]}~0.7"
                        if value_str.split() and "~" not in value_str.split()[0]
                        else ""
                    ),
                ]

                for search_query in search_queries:
                    if not search_query:
                        continue

                    try:
                        # Build query with CALL db.index.fulltext.queryNodes
                        query = f"""
                        CALL db.index.fulltext.queryNodes($index_name, $search_query)
                        YIELD node, score
                        WHERE node:`{validated_label}`
                        RETURN node, score
                        ORDER BY score DESC
                        LIMIT $max_results
                        """

                        records = await self._execute_query(
                            query,
                            {
                                "max_results": max_results,
                                "index_name": index_name,
                                "search_query": search_query,
                            },
                        )

                        # Add results, avoiding duplicates
                        for record in records:
                            node_id = dict(record[0].items()).get("id", "")
                            if node_id and node_id not in seen_ids:
                                seen_ids.add(node_id)
                                node_label = (
                                    list(record[0].labels)[0]
                                    if record[0].labels
                                    else validated_label
                                )
                                all_results.append(
                                    Node(
                                        id=node_id,
                                        label=node_label,
                                        type=node_label,  # Ensure type is always set
                                        properties=dict(record[0].items()),
                                    )
                                )

                                # Stop if we have enough results
                                if len(all_results) >= max_results:
                                    return all_results
                    except Exception as e:
                        # If the index doesn't exist or other error, log and continue
                        logger.debug(
                            f"Full-text search error for {search_query}: {str(e)}"
                        )

        return all_results
