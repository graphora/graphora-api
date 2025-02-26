"""Neo4j implementation of graph storage"""
from typing import List, Dict, Any, Optional
import logging
import traceback
from contextlib import asynccontextmanager
import asyncio

import neo4j
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, DatabaseError

from .interface import GraphStorageInterface
from .models import Node, Edge
from .exceptions import (
    StorageConnectionError,
    StorageAuthError,
    StorageQueryError,
    StorageError
)

# Configure logger
logger = logging.getLogger(__name__)

class Neo4jStorage(GraphStorageInterface):
    """Neo4j implementation of graph storage"""
    
    def __init__(self, uri: str, username: str, password: str, max_retries: int = 3):
        """Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            max_retries: Maximum number of retry attempts for transient errors
            
        Raises:
            StorageConnectionError: If connection cannot be established
            StorageAuthError: If credentials are invalid
        """
        try:
            self.driver = AsyncGraphDatabase.driver(
                uri,
                auth=(username, password),
                max_connection_lifetime=3600  # 1 hour
            )
            self.max_retries = max_retries
            logger.info(f"Successfully initialized Neo4j connection to {uri}")
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
        """Get a Neo4j session with automatic cleanup
        
        Yields:
            Neo4j session object
            
        Raises:
            StorageConnectionError: If session cannot be created
        """
        session = None
        try:
            session = await self.driver.session()
            yield session
        except Exception as e:
            logger.error(f"Failed to create Neo4j session: {str(e)}")
            raise StorageConnectionError(f"Could not create session: {str(e)}")
        finally:
            if session:
                await session.close()

    async def _execute_with_retry(self, operation):
        """Execute Neo4j operation with retry logic
        
        Args:
            operation: Async function to execute
            
        Returns:
            Result of the operation
            
        Raises:
            StorageError: If operation fails after all retries
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await operation()
            except (ServiceUnavailable, DatabaseError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                continue
            except Exception as e:
                logger.error(f"Non-retryable error in Neo4j operation: {str(e)}")
                raise StorageError(f"Operation failed: {str(e)}")
        
        logger.error(f"All retry attempts failed: {str(last_error)}")
        raise StorageError(f"Operation failed after {self.max_retries} retries")
        
    async def get_relationships_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Edge]:
        """Get all relationships between two nodes
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Optional type to filter relationships
            
        Returns:
            List of edges between the nodes
            
        Raises:
            StorageQueryError: If query execution fails
            StorageError: For other unexpected errors
        """
        logger.debug(
            f"Fetching relationships: source={source_id}, target={target_id}, "
            f"type={relationship_type}"
        )
        
        async def _execute_query():
            async with self._get_session() as session:
                # Build Cypher query
                if relationship_type:
                    query = """
                    MATCH (source)-[r:$rel_type]->(target)
                    WHERE ID(source) = $source_id AND ID(target) = $target_id
                    RETURN r
                    """
                    params = {
                        "source_id": source_id,
                        "target_id": target_id,
                        "rel_type": relationship_type
                    }
                else:
                    query = """
                    MATCH (source)-[r]->(target)
                    WHERE ID(source) = $source_id AND ID(target) = $target_id
                    RETURN r
                    """
                    params = {
                        "source_id": source_id,
                        "target_id": target_id
                    }
                
                try:
                    # Execute query
                    result = await session.run(query, params)
                    records = await result.fetch_all()
                    
                    # Convert Neo4j relationships to Edge objects
                    edges = []
                    for record in records:
                        rel = record["r"]
                        edge = Edge(
                            id=str(rel.id),
                            type=rel.type,
                            source_id=source_id,
                            target_id=target_id,
                            properties=dict(rel.items())
                        )
                        edges.append(edge)
                    
                    logger.debug(f"Found {len(edges)} relationships")
                    return edges
                    
                except neo4j.exceptions.CypherSyntaxError as e:
                    logger.error(f"Cypher syntax error: {str(e)}")
                    raise StorageQueryError(f"Invalid query syntax: {str(e)}")
                except Exception as e:
                    logger.error(
                        f"Failed to execute relationship query: {str(e)}\n"
                        f"Query: {query}\nParams: {params}"
                    )
                    raise StorageQueryError(f"Query execution failed: {str(e)}")
        
        return await self._execute_with_retry(_execute_query)
            
    async def close(self):
        """Close Neo4j connection
        
        Raises:
            StorageError: If connection cannot be closed cleanly
        """
        try:
            await self.driver.close()
            logger.info("Neo4j connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
            raise StorageError(f"Failed to close connection: {str(e)}")
