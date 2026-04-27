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
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
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
from graphora_server.utils.constants import MERGE_ID, TRANSFORM_ID

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


def _coerce_for_age(value: Any) -> Any:
    """Coerce a Python value into something AGE's jsonb params accept.

    AGE only accepts JSON-compatible values inside cypher() bodies.
    Datetime → ISO-8601 string, enum → its ``.value``, Pydantic
    model → ``.model_dump()``, dicts/lists → recursed into.
    Everything else passes through; psycopg's json adapter raises
    on truly unsupported types so we don't silently drop.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _coerce_for_age(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _coerce_for_age(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_for_age(v) for v in value]
    return value


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

    # AGE's known agtype type tags. Anything not on this list is
    # not a type tag — even if it follows ``::``. See
    # ``_parse_agtype`` for why this matters.
    _AGE_TYPE_TAGS = ("vertex", "edge", "path")

    @staticmethod
    def _parse_agtype(value: Any) -> Any:
        """Parse an AGE ``agtype`` cell into a Python value.

        AGE returns vertices/edges/paths as strings of the form
        ``{...properties...}::vertex`` (with a trailing type tag).
        Scalars are returned as bare JSON.

        Strategy: try a bare json.loads first — that handles
        scalars (including strings that happen to contain ``::``,
        e.g. ``"a::b"``). If that fails, strip a trailing
        ``::<known-tag>`` and retry. Defensive on malformed input:
        return the raw string rather than raise.

        The earlier implementation stripped on the rightmost
        ``::`` unconditionally, which broke valid scalar strings
        like ``"a::b"`` — flagged in slice-1 review.
        """
        import json

        if value is None:
            return None
        if not isinstance(value, str):
            return value
        # Direct parse first — covers scalars like "a::b" and 42.
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        # Fallback: strip a trailing ::<known-type> tag and retry.
        # We anchor on the known-tag set so an embedded ``::`` in
        # a scalar can't accidentally trigger the strip.
        for tag in PostgresAGEStorage._AGE_TYPE_TAGS:
            suffix = "::" + tag
            if value.endswith(suffix):
                body = value[: -len(suffix)]
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    break  # Tag matched but body is malformed.
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
        """Bulk-write nodes via UNWIND batch grouped by entity type.

        AGE Cypher requires labels to be static (not parameterized),
        so heterogeneous node lists are bucketed by ``node.type``
        and one UNWIND batch query is issued per bucket. This avoids
        the per-row N+1 the Neo4j adapter has (one session per node)
        — even a 1000-node single-type batch is one round trip.

        Identifiers (entity types) are interpolated into the Cypher
        string after passing ``_validate_identifier``; properties go
        through the cypher() jsonb params channel and are coerced
        via ``_coerce_for_age`` to plain JSON values so datetime /
        enum / Pydantic model fields don't blow up the adapter.
        """
        start_time = time.time()
        items_processed = 0
        success = True
        error_message: Optional[str] = None
        warnings: List[str] = []

        # Bucket by validated entity type. Nodes that fail
        # validation get warned and skipped rather than raising —
        # matches Neo4j's resilience for partial-batch failures.
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes:
            try:
                node_id, node_type, props = self._extract_node_writepath_fields(
                    node, transform_id, merge_id
                )
                validated_type = _validate_identifier(node_type, "node type")
            except (CypherInjectionError, ValueError) as exc:
                warnings.append(f"Skipping node: {exc}")
                continue
            by_type.setdefault(validated_type, []).append(
                {"id": node_id, "props": props}
            )

        op = "MERGE" if merge else "CREATE"
        for node_type, batch in by_type.items():
            cypher = (
                f"UNWIND $batch AS row "
                f"{op} (n:{node_type} {{id: row.id}}) "
                f"SET n += row.props "
                f"RETURN n.id AS node_id"
            )
            try:
                await self._execute_cypher(
                    cypher,
                    params={"batch": batch},
                    return_columns="(node_id agtype)",
                )
                items_processed += len(batch)
            except Exception as exc:
                logger.error(
                    "Failed to store batch of %d %s nodes: %s",
                    len(batch),
                    node_type,
                    exc,
                )
                success = False
                error_message = str(exc)
                warnings.append(
                    f"Failed batch of {len(batch)} {node_type} nodes: {exc}"
                )
                break

        if items_processed > 0:
            try:
                checkpoint_result = await self.update_checkpoint(
                    transform_id, batch_index, StorageStage.NODES
                )
                if not checkpoint_result.success:
                    success = False
                    error_message = checkpoint_result.error
                    warnings.append(
                        f"Checkpoint update failed: {checkpoint_result.error}"
                    )
            except Exception as exc:
                success = False
                error_message = f"Failed to update checkpoint: {exc}"
                warnings.append(error_message)

        if items_processed < len(nodes):
            warnings.append(
                f"Partial batch: {items_processed} of {len(nodes)} nodes stored"
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

    @staticmethod
    def _extract_node_writepath_fields(
        node: Any,
        transform_id: str,
        merge_id: Optional[str],
    ) -> tuple:
        """Pull (id, type, props) from a BaseNode or dict input,
        merging in metadata + provenance and coercing to JSON.

        Lifted out of ``store_nodes`` so the value-prep logic can be
        unit-tested without a fake AGE connection.
        """
        if isinstance(node, dict):
            node_id = node.get("id") or str(uuid.uuid4())
            node_type = node.get("type", "")
            raw_props = dict(node.get("properties", {}) or {})
            provenance = None
        else:
            node_id = node.id
            node_type = node.type
            raw_props = dict(node.properties or {})
            provenance = getattr(node, "provenance", None)

        if not node_type:
            raise ValueError(f"Node {node_id} has no type — refusing to write")

        # Strip metadata and rebuild — caller may have left stale
        # transform_id / merge_id values from a prior batch.
        props = {
            k: v
            for k, v in raw_props.items()
            if k not in ("id", "type", TRANSFORM_ID, MERGE_ID)
        }
        props[TRANSFORM_ID] = transform_id
        if merge_id:
            props[MERGE_ID] = merge_id

        # Provenance is optional; when present, flatten its fields
        # onto props so they round-trip through AGE's jsonb path.
        # Skip None-valued fields so the property bag stays compact.
        # Don't clobber LLM-emitted properties — same setdefault
        # semantics as _attach_provenance_properties uses on the
        # extraction side.
        if provenance is not None:
            prov_dict = (
                provenance.model_dump(exclude_none=True)
                if hasattr(provenance, "model_dump")
                else dict(provenance)
            )
            for k, v in prov_dict.items():
                props.setdefault(k, v)

        return node_id, node_type, _coerce_for_age(props)

    async def store_relationships(
        self,
        relationships: List[RelationshipInstance],
        batch_index: int,
        transform_id: str,
        merge_id: Optional[str] = None,
        merge: bool = True,
    ) -> StorageBatchResult:
        """Write relationships one Cypher statement per relationship.

        Slice 3 ships the simple per-row path: AGE's openCypher subset
        doesn't support the same UNWIND-batch shape with mixed
        relationship-type / source-type / target-type triples that
        the node path uses (relationship type, like label, must be
        static in a single Cypher statement). Bucketing by triple
        and batching per bucket is a slice-4 optimization.

        Versioning (valid_from/valid_to bi-temporal updates that the
        Neo4j adapter does on conflict) is also slice-4 work; the
        first cut is MERGE-only — duplicate (source, type, target)
        merges into the same edge with last-write-wins on properties.
        """
        start_time = time.time()
        items_processed = 0
        success = True
        error_message: Optional[str] = None
        warnings: List[str] = []
        seen_rel_ids: set = set()

        op = "MERGE" if merge else "CREATE"

        for rel in relationships:
            if rel.id in seen_rel_ids:
                continue

            try:
                source_type = _validate_identifier(rel.source_type, "source type")
                target_type = _validate_identifier(rel.target_type, "target type")
                rel_type = _validate_identifier(rel.type, "relationship type")
            except CypherInjectionError as exc:
                warnings.append(
                    f"Skipping relationship {rel.id}: invalid identifier ({exc})"
                )
                continue

            # Build property bag the same way as nodes — strip
            # metadata, add transform_id / merge_id, fold provenance,
            # coerce to JSON-safe.
            raw_props = dict(rel.properties or {})
            props = {
                k: v for k, v in raw_props.items() if k not in (TRANSFORM_ID, MERGE_ID)
            }
            props[TRANSFORM_ID] = transform_id
            if merge_id:
                props[MERGE_ID] = merge_id
            if rel.provenance is not None:
                prov_dict = (
                    rel.provenance.model_dump(exclude_none=True)
                    if hasattr(rel.provenance, "model_dump")
                    else dict(rel.provenance)
                )
                for k, v in prov_dict.items():
                    props.setdefault(k, v)
            props = _coerce_for_age(props)

            cypher = (
                f"MATCH (s:{source_type} {{id: $source_id}}) "
                f"MATCH (t:{target_type} {{id: $target_id}}) "
                f"{op} (s)-[r:{rel_type}]->(t) "
                f"SET r += $props "
                f"RETURN r"
            )
            try:
                await self._execute_cypher(
                    cypher,
                    params={
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "props": props,
                    },
                    return_columns="(r agtype)",
                )
                items_processed += 1
                seen_rel_ids.add(rel.id)
            except Exception as exc:
                logger.error(
                    "Failed to store relationship %s (%s -[%s]-> %s): %s",
                    rel.id,
                    rel.source_id,
                    rel.type,
                    rel.target_id,
                    exc,
                )
                success = False
                error_message = str(exc)
                warnings.append(f"Failed relationship {rel.id}: {exc}")
                break

        if items_processed > 0:
            try:
                checkpoint_result = await self.update_checkpoint(
                    transform_id, batch_index, StorageStage.RELATIONSHIPS
                )
                if not checkpoint_result.success:
                    success = False
                    error_message = checkpoint_result.error
                    warnings.append(
                        f"Checkpoint update failed: {checkpoint_result.error}"
                    )
            except Exception as exc:
                success = False
                error_message = f"Failed to update checkpoint: {exc}"
                warnings.append(error_message)

        if items_processed < len(relationships):
            warnings.append(
                f"Partial batch: {items_processed} of "
                f"{len(relationships)} relationships stored"
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

    async def get_storage_status(
        self, transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Read the most recent checkpoint for a transform.

        Stores the checkpoint as a ``:_Checkpoint`` vertex inside the
        same AGE graph (parallels Neo4j's ``:__Checkpoint__`` label,
        renamed because AGE's identifier rules don't allow leading
        double-underscore). One vertex per transform_id, MERGEd on
        the transform_id key.

        Returns None when no checkpoint exists for the transform_id —
        first-write callers rely on that to distinguish "fresh start"
        from "resume in progress".
        """
        rows = await self._execute_cypher(
            """
            MATCH (c:_Checkpoint {transform_id: $transform_id})
            RETURN c
            ORDER BY c.timestamp DESC
            LIMIT 1
            """,
            params={"transform_id": transform_id},
            return_columns="(c agtype)",
        )
        if not rows:
            return None

        parsed = self._parse_agtype(rows[0][0])
        # AGE returns ``{"id": ..., "label": "_Checkpoint",
        # "properties": {...}}`` for vertex agtype values. The
        # property bag is what we built in update_checkpoint.
        props = parsed.get("properties") if isinstance(parsed, dict) else None
        if not props:
            return None

        # Timestamp was written as ISO-8601 from update_checkpoint.
        # Defensive parse — fall back to wall clock if AGE round-
        # tripped the value into something fromisoformat can't read.
        timestamp_raw = props.get("timestamp")
        try:
            timestamp = (
                datetime.fromisoformat(timestamp_raw)
                if isinstance(timestamp_raw, str)
                else datetime.now(timezone.utc)
            )
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        return StorageCheckpoint(
            transform_id=props.get("transform_id", transform_id),
            last_processed_index=int(props.get("last_processed_index", 0)),
            stage=StorageStage(props.get("stage", StorageStage.NODES.value)),
            timestamp=timestamp,
        )

    async def update_checkpoint(
        self, transform_id: str, last_index: int, stage: StorageStage
    ) -> StorageBatchResult:
        """MERGE-write a checkpoint vertex for the transform_id.

        Returns a ``StorageBatchResult`` to match the Neo4j adapter's
        de-facto contract — the abstract interface declares ``-> None``
        but every caller (store_nodes, store_relationships) checks
        ``.success`` on the returned object. Slice 3's write-path
        methods consume this return shape.

        AGE has no Cypher ``datetime()`` builtin, so the timestamp is
        materialized in Python and passed in as an ISO-8601 string.
        """
        start_time = time.time()
        items_processed = 0
        success = True
        error_message: Optional[str] = None

        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            stage_value = stage.value if isinstance(stage, StorageStage) else str(stage)
            await self._execute_cypher(
                """
                MERGE (c:_Checkpoint {transform_id: $transform_id})
                SET c.last_processed_index = $last_index,
                    c.stage = $stage,
                    c.timestamp = $timestamp
                """,
                params={
                    "transform_id": transform_id,
                    "last_index": last_index,
                    "stage": stage_value,
                    "timestamp": timestamp,
                },
            )
            items_processed = 1
        except Exception as exc:
            logger.error(
                "Failed to update AGE checkpoint for transform %s: %s",
                transform_id,
                exc,
            )
            success = False
            error_message = f"Failed to update checkpoint: {exc}"

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
