"""Apache AGE implementation of graph storage (Gate 5 / C2-postgres).

AGE is a PostgreSQL extension that adds an openCypher subset on top
of relational tables. The adapter speaks Cypher via the
``ag_catalog.cypher(graph_name, query, params)`` SQL function, with
results returned as ``agtype`` values that we cast/parse on the
Python side.

Design parallels Neo4jStorage so the swap is mechanical for callers:
same async context-manager session pattern, same retry-with-backoff
helper, same exception surface. The differences are confined to the
adapter:

* Cypher is wrapped in ``SELECT * FROM cypher('graphora', $$ ... $$)
  AS (n agtype)`` rather than sent over Bolt.
* AGE supports an openCypher *subset* — full-text indexes use
  Postgres GIN/pg_trgm rather than ``CREATE FULLTEXT INDEX``.
  Methods that depend on Neo4j-specific features document their
  AGE polyfill at the call-site.
* pgvector lives in the same database, so embedding similarity
  search runs in-process via ``<->`` operators on a sibling table
  rather than a remote cosine call.

This file is the foundation slice (connection layer + schema
bootstrap + method skeletons). Method bodies land in subsequent
commits to keep diffs reviewable.
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from graphora_server.schemas.graph import Edge, GraphResponse, Node
from graphora_server.services.storage.exceptions import StorageConnectionError
from graphora_server.services.storage.interface import GraphStorageInterface
from graphora_server.services.storage.models import (
    StorageBatchResult,
    StorageCheckpoint,
    StorageStage,
)
from graphora_server.services.transform.models import BaseNode, RelationshipInstance

logger = logging.getLogger(__name__)

# AGE is strict about identifier shape inside cypher() bodies — same
# rules as Neo4j (alphanumerics + underscore, must start with a letter
# or underscore). Reuse Neo4jStorage's regex-based validator if it
# moves to a shared module; for now we re-validate locally to avoid
# importing neo4j.py when the [neo4j] extra isn't installed.
_CYPHER_IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class CypherInjectionError(Exception):
    """Raised when an identifier interpolated into a cypher() body
    fails the alphanumeric/underscore safety check.

    AGE does not currently parameterize labels or relationship types,
    so they have to be interpolated as strings — which means *we*
    have to validate them or risk a SQL/Cypher injection.
    """


def _validate_identifier(name: str, kind: str = "identifier") -> str:
    if not name:
        raise CypherInjectionError(f"Empty {kind} not allowed")
    if not _CYPHER_IDENT.match(name):
        raise CypherInjectionError(
            f"Invalid {kind} '{name}': alphanumerics/underscores only, "
            f"must start with letter or underscore"
        )
    if len(name) > 256:
        raise CypherInjectionError(
            f"Invalid {kind} '{name[:50]}...': exceeds 256 character limit"
        )
    return name


class PostgresAGEStorage(GraphStorageInterface):
    """Apache AGE implementation of GraphStorageInterface.

    Slice 1 (this commit): connection lifecycle, schema bootstrap,
    cypher() helper. Method bodies stub with NotImplementedError so
    factory dispatch + import paths can be exercised end-to-end
    before we port the 22 storage methods one at a time.
    """

    def __init__(
        self,
        dsn: str,
        graph_name: str = "graphora",
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        """Initialize AGE storage.

        Args:
            dsn: PostgreSQL connection string. Same shape as the
                application database — AGE rides on top of plain
                Postgres, no special protocol.
            graph_name: AGE graph identifier (single-graph-per-database
                model in this adapter). Must satisfy
                ``_validate_identifier``.
            min_pool_size: Minimum connections held in the async pool.
            max_pool_size: Maximum connections — the AGE adapter is
                OLAP-leaning so this can be tighter than typical OLTP.
        """
        self.dsn = dsn
        self.graph_name = _validate_identifier(graph_name, "graph name")
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size

        try:
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:  # pragma: no cover — exercised without [postgres]
            raise ImportError(
                "Apache AGE storage requires the [postgres] extra. "
                "Install with: pip install 'graphora-server[age]'"
            ) from exc

        # Pool is created lazily on first use to keep __init__
        # non-blocking; opening connections happens inside
        # async functions where the event loop exists.
        self._pool: Optional[AsyncConnectionPool] = None
        self._AsyncConnectionPool = AsyncConnectionPool

    async def _ensure_pool(self):
        """Create the connection pool on first use and bootstrap the
        graph schema. Idempotent — repeated calls return the same pool.
        """
        if self._pool is not None:
            return self._pool

        try:
            pool = self._AsyncConnectionPool(
                conninfo=self.dsn,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                open=False,
            )
            await pool.open()
            self._pool = pool
        except Exception as exc:
            logger.error("Failed to open AGE connection pool: %s", exc)
            raise StorageConnectionError(
                f"Could not open AGE connection pool: {exc}"
            ) from exc

        await self._bootstrap_schema()
        return self._pool

    async def _bootstrap_schema(self) -> None:
        """Ensure AGE + pgvector are loaded and the named graph exists.

        Idempotent: ``CREATE EXTENSION IF NOT EXISTS`` and
        ``ag_catalog.create_graph`` is wrapped to no-op on duplicate.
        Run once per process at first use.
        """
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS age")
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                # AGE needs to be loaded into the session before
                # cypher() is callable; SET search_path so unqualified
                # AGE function calls resolve.
                await cur.execute("LOAD 'age'")
                await cur.execute("SET search_path = ag_catalog, '$user', public")
                # Idempotent graph create — AGE raises a duplicate-
                # graph error if it already exists, which we swallow.
                try:
                    await cur.execute("SELECT create_graph(%s)", (self.graph_name,))
                except Exception as exc:
                    msg = str(exc).lower()
                    if "already exists" in msg or "duplicate" in msg:
                        logger.debug(
                            "AGE graph '%s' already exists, skipping create",
                            self.graph_name,
                        )
                    else:
                        raise
                await conn.commit()

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[Any]:
        """Yield a connection from the pool with AGE preloaded."""
        pool = await self._ensure_pool() if self._pool is None else self._pool
        async with pool.connection() as conn:
            # Each connection needs LOAD 'age' + search_path because
            # they're per-session settings, not persistent at the
            # database level. Pool's session_init hook is the right
            # long-term home; for now we set on every checkout.
            async with conn.cursor() as cur:
                await cur.execute("LOAD 'age'")
                await cur.execute("SET search_path = ag_catalog, '$user', public")
            try:
                yield conn
            except Exception:
                await conn.rollback()
                raise

    async def _execute_cypher(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        return_columns: str = "(result agtype)",
    ) -> List[tuple]:
        """Run an AGE Cypher query and return raw rows.

        AGE's ``cypher()`` SQL function takes (graph_name, body,
        jsonb_params). We wrap that in ``SELECT * FROM cypher(...)
        AS <columns>`` because AGE refuses unbound result columns.

        Args:
            query: Cypher body without the surrounding cypher() call.
                Use ``$param_name`` inside the body to reference keys
                from ``params``.
            params: Bind values, serialized to jsonb. AGE only accepts
                JSON-compatible values inside Cypher bodies — pre-
                serialize bytes/datetime callers before invoking.
            return_columns: SQL projection list with agtype typing.
                Default ``(result agtype)`` covers single-value returns;
                override for multi-column projections like
                ``(n agtype, r agtype, m agtype)``.

        Returns:
            List of psycopg row tuples. Each agtype value is a string
            that the caller parses (see ``_parse_agtype``).
        """
        import json

        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                # AGE uses $cypher$ as a custom dollar-quote tag to
                # avoid collision with $$ literals that may appear
                # inside the Cypher body.
                params_json = json.dumps(params or {})
                sql = (
                    f"SELECT * FROM cypher('{self.graph_name}', "
                    f"$cypher${query}$cypher$, %s) AS {return_columns}"
                )
                await cur.execute(sql, (params_json,))
                rows = await cur.fetchall()
                await conn.commit()
                return rows

    @staticmethod
    def _parse_agtype(value: Any) -> Any:
        """Parse an AGE ``agtype`` cell into a Python value.

        AGE returns vertices/edges/paths as strings of the form
        ``{...properties...}::vertex`` (with a trailing type tag).
        Strip the tag and json-decode the body. Scalar types
        (numbers, strings) round-trip through json.loads directly.
        """
        import json

        if value is None:
            return None
        if not isinstance(value, str):
            return value
        # AGE may return either ``{...}::vertex`` (typed) or a bare
        # JSON string. Strip a trailing ``::<type>`` if present.
        body = value
        if "::" in body:
            split_idx = body.rfind("::")
            body = body[:split_idx]
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return value

    # --- GraphStorageInterface methods (stubbed pending slice 2+) ---

    async def create_or_replace_ft_index_for_node(
        self, index_name: str, entity_name: str, properties: List[str]
    ) -> None:
        # AGE has no native CREATE FULLTEXT INDEX; the polyfill is a
        # GIN index on a tsvector expression over the property bag.
        # Lands in the index-management slice.
        raise NotImplementedError(
            "AGE adapter: create_or_replace_ft_index_for_node "
            "(GIN/tsvector polyfill) — slice 5"
        )

    async def create_or_replace_ft_index_for_relationship(
        self,
        index_name: str,
        source_name: str,
        rel_name: str,
        target_name: str,
        properties: List[str],
    ) -> None:
        raise NotImplementedError(
            "AGE adapter: create_or_replace_ft_index_for_relationship " "— slice 5"
        )

    async def store_nodes(
        self,
        nodes: List[BaseNode],
        batch_index: int,
        transform_id: str,
        merge_id: Optional[str] = None,
        merge: bool = True,
    ) -> StorageBatchResult:
        # Bulk node write via UNWIND $batch CREATE/MERGE pattern,
        # avoiding the per-row N+1 footgun the Neo4j adapter has.
        # Slice 2.
        raise NotImplementedError("AGE adapter: store_nodes — slice 2 (write path)")

    async def store_relationships(
        self,
        relationships: List[RelationshipInstance],
        batch_index: int,
        transform_id: str,
        merge_id: Optional[str] = None,
        merge: bool = True,
    ) -> StorageBatchResult:
        raise NotImplementedError(
            "AGE adapter: store_relationships — slice 2 (write path)"
        )

    async def get_storage_status(
        self, transform_id: str
    ) -> Optional[StorageCheckpoint]:
        raise NotImplementedError("AGE adapter: get_storage_status — slice 2")

    async def update_checkpoint(
        self, transform_id: str, last_index: int, stage: StorageStage
    ) -> None:
        raise NotImplementedError("AGE adapter: update_checkpoint — slice 2")

    async def get_transformation_data(self, transform_id: str) -> GraphResponse:
        raise NotImplementedError(
            "AGE adapter: get_transformation_data — slice 3 (read path)"
        )

    async def get_merge_data(self, merge_id: str) -> GraphResponse:
        raise NotImplementedError("AGE adapter: get_merge_data — slice 3 (read path)")

    async def get_all_node_properties(self, entity_name: str) -> List[str]:
        raise NotImplementedError("AGE adapter: get_all_node_properties — slice 3")

    async def get_all_relationship_properties(self, rel_name: str) -> List[str]:
        raise NotImplementedError(
            "AGE adapter: get_all_relationship_properties — slice 3"
        )

    async def get_nodes_by_property(
        self, property_name: str, property_value: Any
    ) -> List[Node]:
        raise NotImplementedError("AGE adapter: get_nodes_by_property — slice 3")

    async def get_relationships_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None,
    ) -> List[Edge]:
        raise NotImplementedError("AGE adapter: get_relationships_between — slice 3")

    async def get_relationships_between_nodes(self, node_ids: List[str]) -> List[Edge]:
        raise NotImplementedError(
            "AGE adapter: get_relationships_between_nodes — slice 3"
        )

    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True,
    ) -> List[Node]:
        raise NotImplementedError("AGE adapter: find_nodes_by_property_value — slice 3")

    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True,
    ) -> List[Node]:
        # Embedding similarity via pgvector <-> operator on a sibling
        # table; property-bag fuzzy match via pg_trgm. Slice 4.
        raise NotImplementedError(
            "AGE adapter: find_similar_nodes (pgvector + pg_trgm) — slice 4"
        )

    async def create_node(self, label: str, properties: Dict[str, Any]) -> Node:
        raise NotImplementedError("AGE adapter: create_node — slice 3")

    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> Node:
        raise NotImplementedError("AGE adapter: update_node — slice 3")

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any] = None,
    ) -> Edge:
        raise NotImplementedError("AGE adapter: create_relationship — slice 3")

    async def update_relationship(
        self, rel_id: str, properties: Dict[str, Any]
    ) -> Edge:
        raise NotImplementedError("AGE adapter: update_relationship — slice 3")

    async def get_relationship(
        self, source_id: str, target_id: str, rel_type: str
    ) -> Optional[Edge]:
        raise NotImplementedError("AGE adapter: get_relationship — slice 3")

    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        raise NotImplementedError("AGE adapter: get_node_by_id — slice 3")

    async def get_edges_between(self, source_id: str, target_id: str) -> List[Edge]:
        raise NotImplementedError("AGE adapter: get_edges_between — slice 3")

    async def close(self) -> None:
        """Close the connection pool.

        Mirrors Neo4jStorage's implicit driver lifecycle — caller
        invokes this from shutdown handlers; not part of
        GraphStorageInterface today but useful in tests.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
