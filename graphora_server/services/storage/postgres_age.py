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
from graphora_server.utils.constants import MERGE_ID, SYSTEM_PROPERTIES, TRANSFORM_ID

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
        """Ensure AGE is loaded and the named graph exists.

        Idempotent: ``CREATE EXTENSION IF NOT EXISTS`` and
        ``ag_catalog.create_graph`` is wrapped to no-op on duplicate.
        Run once per process at first use.

        ``vector`` (pgvector) is also opportunistically created for
        slice 9+ embedding-similarity work, but we tolerate it being
        unavailable — the apache/age Docker image doesn't ship it
        preinstalled, and slice 6's CONTAINS-based find_similar_nodes
        doesn't need it. When the call fails we log once and keep
        going so the AGE-only path stays viable.
        """
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS age")
                # Wrap the optional pgvector create in a SAVEPOINT so
                # a rollback on its failure doesn't undo the required
                # CREATE EXTENSION age above. ``conn.rollback()`` would
                # discard the entire transaction (including the AGE
                # setup), then LOAD 'age' / create_graph(…) below would
                # run against a database with AGE unregistered.
                await cur.execute("SAVEPOINT vector_ext")
                try:
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await cur.execute("RELEASE SAVEPOINT vector_ext")
                except Exception as exc:
                    await cur.execute("ROLLBACK TO SAVEPOINT vector_ext")
                    # pgvector not installed — fine for slice 6's
                    # CONTAINS scoring path. Future embedding-similarity
                    # work will enforce this dependency at the
                    # callsite that actually needs vectors.
                    # Once-per-instance gate keeps the log readable
                    # across repeated _bootstrap_schema calls.
                    if not getattr(self, "_warned_no_pgvector", False):
                        logger.warning(
                            "pgvector extension not available (%s). "
                            "AGE adapter continues without vector "
                            "support; embedding-similarity features "
                            "will be no-ops until pgvector is "
                            "installed.",
                            exc,
                        )
                        self._warned_no_pgvector = True
                # pg_trgm — same savepoint-isolated optional
                # bootstrap as vector. Used by slice 8's GIN-index
                # polyfill for create_or_replace_ft_index_*.
                # Adapter remembers availability so the index
                # methods can fall back to no-op cleanly when the
                # extension isn't present.
                self._has_pg_trgm = False
                await cur.execute("SAVEPOINT pg_trgm_ext")
                try:
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                    await cur.execute("RELEASE SAVEPOINT pg_trgm_ext")
                    self._has_pg_trgm = True
                except Exception as exc:
                    await cur.execute("ROLLBACK TO SAVEPOINT pg_trgm_ext")
                    if not getattr(self, "_warned_no_pg_trgm", False):
                        logger.warning(
                            "pg_trgm extension not available (%s). "
                            "Full-text index methods will fall back "
                            "to no-op until pg_trgm is installed; "
                            "CONTAINS-based searches still work, "
                            "just without GIN acceleration.",
                            exc,
                        )
                        self._warned_no_pg_trgm = True
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
        """Create a Postgres GIN index over the AGE-managed vertex
        table for ``entity_name`` to accelerate CONTAINS-style
        substring searches (the merge-flow similarity-fallback
        pattern in find_similar_nodes / find_nodes_by_property_value).

        AGE has no native ``CREATE FULLTEXT INDEX``. The polyfill is a
        GIN index using the ``pg_trgm`` ``gin_trgm_ops`` operator
        class on the agtype properties column cast to text:

            CREATE INDEX <name>
              ON <graph>.<entity>
              USING GIN ((properties::text) gin_trgm_ops);

        Idempotent via ``DROP INDEX IF EXISTS`` + ``CREATE INDEX``
        — matches Neo4j's ``DROP INDEX ... IF EXISTS`` + ``CREATE
        FULLTEXT INDEX`` shape so the "or replace" semantics stay
        the same across backends.

        Falls back to a no-op + one warning when ``pg_trgm`` isn't
        installed (the bootstrap_schema savepoint pattern records
        availability in ``self._has_pg_trgm``). CONTAINS searches
        keep working without the index, just without acceleration.

        ``properties`` is ignored for the GIN index because we index
        the whole properties bag — same pattern as the Neo4j adapter
        passing ``ON EACH [props]`` to the fulltext index. AGE's
        agtype-to-text cast linearizes all property values together,
        so a substring match on any property value is accelerated.
        """
        validated_index = _validate_identifier(index_name, "index name")
        validated_entity = _validate_identifier(entity_name, "entity name")
        for prop in properties:
            _validate_identifier(prop, "property name")

        if not getattr(self, "_has_pg_trgm", False):
            if not getattr(self, "_warned_ft_index_node", False):
                logger.warning(
                    "create_or_replace_ft_index_for_node is a no-op "
                    "because pg_trgm is unavailable (see "
                    "_bootstrap_schema warning). Install pg_trgm and "
                    "re-bootstrap to enable GIN index acceleration. "
                    "index=%s entity=%s",
                    index_name,
                    entity_name,
                )
                self._warned_ft_index_node = True
            return

        # AGE creates one table per vertex label under the graph's
        # schema, named ``<graph_name>.<label>`` — use psycopg.sql
        # for safe identifier composition since the index name +
        # schema + label all interpolate.
        from psycopg import sql

        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {schema}.{idx}").format(
                        schema=sql.Identifier(self.graph_name),
                        idx=sql.Identifier(validated_index),
                    )
                )
                await cur.execute(
                    sql.SQL(
                        "CREATE INDEX {idx} ON {schema}.{table} "
                        "USING GIN ((properties::text) gin_trgm_ops)"
                    ).format(
                        idx=sql.Identifier(validated_index),
                        schema=sql.Identifier(self.graph_name),
                        table=sql.Identifier(validated_entity),
                    )
                )
                await conn.commit()
                logger.debug(
                    "Created GIN index %s on %s.%s",
                    validated_index,
                    self.graph_name,
                    validated_entity,
                )

    async def create_or_replace_ft_index_for_relationship(
        self,
        index_name: str,
        source_name: str,
        rel_name: str,
        target_name: str,
        properties: List[str],
    ) -> None:
        """Same shape as the node variant — AGE creates one table
        per edge label. The polyfill is a GIN index over the
        properties column on ``<graph_name>.<rel_name>``.

        ``source_name`` / ``target_name`` are part of the abstract
        contract (Neo4j wires them into the relationship pattern of
        ``CREATE FULLTEXT INDEX FOR ()-[r:TYPE]->()``), but on the
        AGE side every edge with a given label lands in the same
        underlying table regardless of endpoint types — so we
        index the table directly and ignore source/target. Documented
        here so the contract drift is intentional, not a bug.
        """
        validated_index = _validate_identifier(index_name, "index name")
        validated_rel = _validate_identifier(rel_name, "relationship type")
        # source/target validated for defense-in-depth even though
        # they don't shape the SQL — keeps the injection-safety
        # contract uniform across both index methods.
        _validate_identifier(source_name, "source type")
        _validate_identifier(target_name, "target type")
        for prop in properties:
            _validate_identifier(prop, "property name")

        if not getattr(self, "_has_pg_trgm", False):
            if not getattr(self, "_warned_ft_index_rel", False):
                logger.warning(
                    "create_or_replace_ft_index_for_relationship is a "
                    "no-op because pg_trgm is unavailable. index=%s "
                    "rel=%s",
                    index_name,
                    rel_name,
                )
                self._warned_ft_index_rel = True
            return

        from psycopg import sql

        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {schema}.{idx}").format(
                        schema=sql.Identifier(self.graph_name),
                        idx=sql.Identifier(validated_index),
                    )
                )
                await cur.execute(
                    sql.SQL(
                        "CREATE INDEX {idx} ON {schema}.{table} "
                        "USING GIN ((properties::text) gin_trgm_ops)"
                    ).format(
                        idx=sql.Identifier(validated_index),
                        schema=sql.Identifier(self.graph_name),
                        table=sql.Identifier(validated_rel),
                    )
                )
                await conn.commit()
                logger.debug(
                    "Created GIN index %s on %s.%s",
                    validated_index,
                    self.graph_name,
                    validated_rel,
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

        # Bucket by (validated_type, frozenset(prop_keys)) — AGE
        # rejects ``SET n += $param`` (parameter on the RHS isn't a
        # map type), so we have to interpolate property keys into
        # the Cypher SET clause statically. Nodes that share the
        # same type AND key shape go into one UNWIND batch, which
        # preserves bulk-write efficiency for the common case (most
        # nodes of a given type have the same schema). Nodes with
        # an unusual key set fall into their own bucket — a one-row
        # batch is fine; correctness over micro-optimization.
        #
        # All property keys go through ``_validate_identifier``
        # before they reach the Cypher string. Values stay in the
        # params channel as ``row.<key>``, never interpolated.
        by_bucket: Dict[tuple, List[Dict[str, Any]]] = {}
        for node in nodes:
            try:
                node_id, node_type, props = self._extract_node_writepath_fields(
                    node, transform_id, merge_id
                )
                validated_type = _validate_identifier(node_type, "node type")
                # Validate every key that will interpolate; refuse
                # the whole node if any key would be unsafe.
                for k in props.keys():
                    _validate_identifier(k, "property name")
            except (CypherInjectionError, ValueError) as exc:
                warnings.append(f"Skipping node: {exc}")
                continue
            key_shape = frozenset(props.keys())
            bucket_key = (validated_type, key_shape)
            # Build a flat row dict: each property keyed by its own
            # name (no nested "props" sub-map, since AGE can't
            # destructure ``row.props`` into a SET statement).
            row = {"id": node_id, **props}
            by_bucket.setdefault(bucket_key, []).append(row)

        op = "MERGE" if merge else "CREATE"
        for (node_type, key_shape), batch in by_bucket.items():
            sorted_keys = sorted(key_shape)
            if sorted_keys:
                set_clause = "SET " + ", ".join(f"n.{k} = row.{k}" for k in sorted_keys)
            else:
                set_clause = ""
            cypher = (
                f"UNWIND $batch AS row "
                f"{op} (n:{node_type} {{id: row.id}}) "
                f"{set_clause} "
                f"RETURN n.id AS node_id"
            ).strip()
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

        # Checkpoint advancement is OWNED BY THE TASK LAYER, not the
        # adapter — see services/storage/tasks.py.store_knowledge_graph.
        # The earlier internal call here passed ``batch_index`` (batch
        # number, not items_processed) which the task layer
        # interprets as a node-array index, persisting a stale value
        # that conflicts with the task's partial-failure contract:
        # on success=False the task raises before its own checkpoint
        # write, leaving the adapter's bogus value in place and
        # causing duplicate CREATEs on resume (since the task calls
        # store_nodes with merge=False). The Neo4j adapter has the
        # same shape of pre-existing bug; that's out of scope here.

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

            # Validate every property key — they interpolate into
            # the SET clause. Same reason as store_nodes: AGE
            # rejects ``SET r += $param`` with a parameter on the
            # RHS, so the keys go inline (validated) and values
            # flow through the params channel as ``$prop_<key>``
            # to avoid collision with $source_id / $target_id.
            try:
                validated_keys = {
                    k: _validate_identifier(k, "property name") for k in props.keys()
                }
            except CypherInjectionError as exc:
                warnings.append(
                    f"Skipping relationship {rel.id}: invalid property key ({exc})"
                )
                continue

            params: Dict[str, Any] = {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
            }
            if validated_keys:
                set_pieces = []
                for original_key, validated_key in sorted(validated_keys.items()):
                    param_name = f"prop_{validated_key}"
                    set_pieces.append(f"r.{validated_key} = ${param_name}")
                    params[param_name] = props[original_key]
                set_clause = "SET " + ", ".join(set_pieces)
            else:
                set_clause = ""

            cypher = (
                f"MATCH (s:{source_type} {{id: $source_id}}) "
                f"MATCH (t:{target_type} {{id: $target_id}}) "
                f"{op} (s)-[r:{rel_type}]->(t) "
                f"{set_clause} "
                f"RETURN r"
            ).strip()
            try:
                await self._execute_cypher(
                    cypher,
                    params=params,
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

        # Checkpoint advancement is OWNED BY THE TASK LAYER (see
        # the matching comment in store_nodes) — removed the earlier
        # internal update_checkpoint call to prevent the bogus
        # batch_index value from being persisted ahead of the task
        # layer's items_processed-based write.

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
        """Read all nodes + relationships written under one transform_id.

        Used by:
          - storage.tasks.store_knowledge_graph (verification call after
            the write loop, line 232)
          - api.graph.get_graph (HTTP endpoint)
          - api.transform.get_transformation_data (HTTP endpoint)

        AGE Cypher returns vertex / edge agtype values; ``_parse_agtype``
        extracts the property bags. Single round-trip with two MATCH
        clauses — separate node and edge queries are simpler than the
        Neo4j adapter's collect/UNWIND tangle and round-trip-equivalent
        for the size budgets we target.
        """
        # Nodes
        node_rows = await self._execute_cypher(
            f"""
            MATCH (n)
            WHERE n.{TRANSFORM_ID} = $transform_id
            RETURN n
            """,
            params={"transform_id": transform_id},
            return_columns="(n agtype)",
        )

        nodes_list: List[Node] = []
        node_ids_seen: set = set()
        for row in node_rows:
            parsed = self._parse_agtype(row[0])
            if not isinstance(parsed, dict):
                continue
            node = self._vertex_to_node(parsed)
            if node is None or node.id in node_ids_seen:
                continue
            nodes_list.append(node)
            node_ids_seen.add(node.id)

        # Edges scoped to the same transform — match on the
        # __tid metadata stamped at write time. This is more
        # reliable than "any edge touching a node we just read"
        # because it survives merges that add cross-transform edges.
        edge_rows = await self._execute_cypher(
            f"""
            MATCH (s)-[r]->(t)
            WHERE r.{TRANSFORM_ID} = $transform_id
            RETURN r, s.id, t.id
            """,
            params={"transform_id": transform_id},
            return_columns="(r agtype, source_id agtype, target_id agtype)",
        )

        edges_list: List[Edge] = []
        edge_ids_seen: set = set()
        for row in edge_rows:
            edge = self._edge_row_to_edge(row)
            if edge is None or edge.id in edge_ids_seen:
                continue
            edges_list.append(edge)
            edge_ids_seen.add(edge.id)

        return GraphResponse(
            nodes=nodes_list,
            edges=edges_list,
            total_nodes=len(nodes_list),
            total_edges=len(edges_list),
        )

    async def get_merge_data(self, merge_id: str) -> GraphResponse:
        """Same shape as get_transformation_data but keyed by merge_id.

        Used by callers reading a merge result. Implementation mirrors
        get_transformation_data with the metadata key swapped — the
        Neo4j adapter has the same near-duplicate (~200 LoC) flagged
        in CLAUDE.md performance notes; we keep the duplication local
        rather than introduce a shared helper that would obscure the
        Cypher.
        """
        node_rows = await self._execute_cypher(
            f"""
            MATCH (n)
            WHERE n.{MERGE_ID} = $merge_id
            RETURN n
            """,
            params={"merge_id": merge_id},
            return_columns="(n agtype)",
        )
        nodes_list: List[Node] = []
        node_ids_seen: set = set()
        for row in node_rows:
            parsed = self._parse_agtype(row[0])
            if not isinstance(parsed, dict):
                continue
            node = self._vertex_to_node(parsed)
            if node is None or node.id in node_ids_seen:
                continue
            nodes_list.append(node)
            node_ids_seen.add(node.id)

        edge_rows = await self._execute_cypher(
            f"""
            MATCH (s)-[r]->(t)
            WHERE r.{MERGE_ID} = $merge_id
            RETURN r, s.id, t.id
            """,
            params={"merge_id": merge_id},
            return_columns="(r agtype, source_id agtype, target_id agtype)",
        )
        edges_list: List[Edge] = []
        edge_ids_seen: set = set()
        for row in edge_rows:
            edge = self._edge_row_to_edge(row)
            if edge is None or edge.id in edge_ids_seen:
                continue
            edges_list.append(edge)
            edge_ids_seen.add(edge.id)

        return GraphResponse(
            nodes=nodes_list,
            edges=edges_list,
            total_nodes=len(nodes_list),
            total_edges=len(edges_list),
        )

    @staticmethod
    def _vertex_to_node(parsed: Dict[str, Any]) -> Optional[Node]:
        """Map an AGE-parsed vertex dict to the schema's Node.

        ``parsed`` shape (after ``_parse_agtype``):
            {"id": <int>, "label": <type-string>,
             "properties": {...user properties...}}
        The "id" key on the AGE vertex is AGE's internal numeric id,
        not the user-facing UUID we wrote. We pull the user id from
        the property bag.
        """
        props = parsed.get("properties") or {}
        if not isinstance(props, dict):
            return None
        node_id = props.get("id") or props.get(TRANSFORM_ID, "")
        label = parsed.get("label") or props.get("type") or ""
        return Node(
            id=str(node_id),
            label=str(label),
            type=str(label),
            properties={k: v for k, v in props.items() if k != "id"},
        )

    @staticmethod
    def _edge_row_to_edge(row: tuple) -> Optional[Edge]:
        """Map an (edge agtype, source_id agtype, target_id agtype) row
        to the schema's Edge.

        AGE edge agtype shape:
            {"id": <int>, "label": <rel-type>, "start_id": <int>,
             "end_id": <int>, "properties": {...}}
        We pair AGE's internal start/end ids with the user-facing
        source.id / target.id we returned alongside the edge — the
        cypher query projects both so the caller can stitch them.
        """
        if len(row) < 3:
            return None
        edge_parsed = PostgresAGEStorage._parse_agtype(row[0])
        source_id = PostgresAGEStorage._parse_agtype(row[1])
        target_id = PostgresAGEStorage._parse_agtype(row[2])
        if not isinstance(edge_parsed, dict):
            return None
        props = edge_parsed.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        rel_id = props.get("id") or str(edge_parsed.get("id", ""))
        rel_type = edge_parsed.get("label") or props.get("type") or ""
        return Edge(
            id=str(rel_id),
            source=str(source_id) if source_id is not None else "",
            target=str(target_id) if target_id is not None else "",
            type=str(rel_type),
            properties=props,
        )

    async def get_all_node_properties(self, entity_name: str) -> List[str]:
        # Used by ontology-introspection flows. Sample a few nodes of
        # this type and union their property keys — same shape as
        # the Neo4j adapter's implementation.
        validated = _validate_identifier(entity_name, "entity name")
        rows = await self._execute_cypher(
            f"""
            MATCH (n:{validated})
            RETURN keys(n) AS props
            LIMIT 100
            """,
            return_columns="(props agtype)",
        )
        seen: set = set()
        for row in rows:
            parsed = self._parse_agtype(row[0])
            if isinstance(parsed, list):
                for k in parsed:
                    if isinstance(k, str) and not k.startswith("__"):
                        seen.add(k)
        return sorted(seen)

    async def get_all_relationship_properties(self, rel_name: str) -> List[str]:
        validated = _validate_identifier(rel_name, "relationship type")
        rows = await self._execute_cypher(
            f"""
            MATCH ()-[r:{validated}]->()
            RETURN keys(r) AS props
            LIMIT 100
            """,
            return_columns="(props agtype)",
        )
        seen: set = set()
        for row in rows:
            parsed = self._parse_agtype(row[0])
            if isinstance(parsed, list):
                for k in parsed:
                    if isinstance(k, str) and not k.startswith("__"):
                        seen.add(k)
        return sorted(seen)

    async def get_nodes_by_property(
        self, property_name: str, property_value: Any
    ) -> List[Node]:
        """Used by services/merge/new_merger.py (line ~591) to find
        nodes matching a property during merge resolution.

        ``property_name`` is interpolated into the Cypher (validated);
        ``property_value`` goes through the params channel.
        """
        validated = _validate_identifier(property_name, "property name")
        rows = await self._execute_cypher(
            f"""
            MATCH (n)
            WHERE n.{validated} = $value
            RETURN n
            """,
            params={"value": _coerce_for_age(property_value)},
            return_columns="(n agtype)",
        )
        nodes: List[Node] = []
        for row in rows:
            parsed = self._parse_agtype(row[0])
            if isinstance(parsed, dict):
                node = self._vertex_to_node(parsed)
                if node is not None:
                    nodes.append(node)
        return nodes

    async def get_relationships_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None,
    ) -> List[Edge]:
        """Find directed edges from source_id to target_id.

        ``relationship_type`` is optional — when provided, scoped to
        that type (validated for injection-safety since it interpolates
        into Cypher); when None, all types are returned.
        """
        if relationship_type:
            rel_clause = f":{_validate_identifier(relationship_type, 'rel type')}"
        else:
            rel_clause = ""
        rows = await self._execute_cypher(
            f"""
            MATCH (s {{id: $source_id}})-[r{rel_clause}]->(t {{id: $target_id}})
            RETURN r, s.id, t.id
            """,
            params={"source_id": source_id, "target_id": target_id},
            return_columns="(r agtype, source_id agtype, target_id agtype)",
        )
        edges: List[Edge] = []
        for row in rows:
            edge = self._edge_row_to_edge(row)
            if edge is not None:
                edges.append(edge)
        return edges

    async def get_relationships_between_nodes(self, node_ids: List[str]) -> List[Edge]:
        """Used by services/merge/new_merger.py (line ~612) to find
        all edges where either endpoint is in ``node_ids``. AGE's
        openCypher subset supports IN with a list parameter.
        """
        rows = await self._execute_cypher(
            """
            MATCH (s)-[r]->(t)
            WHERE s.id IN $node_ids OR t.id IN $node_ids
            RETURN r, s.id, t.id
            """,
            params={"node_ids": list(node_ids)},
            return_columns="(r agtype, source_id agtype, target_id agtype)",
        )
        edges: List[Edge] = []
        for row in rows:
            edge = self._edge_row_to_edge(row)
            if edge is not None:
                edges.append(edge)
        return edges

    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True,
    ) -> List[Node]:
        """Label-scoped property search; called from the merge flow
        (services/merge/new_merger.py:1016, :1037, :1052) ahead of
        the similarity fallback. Different from get_nodes_by_property
        (already shipped in slice 4): adds a label filter and the
        ``exact_match`` toggle.

        ``exact_match=True``: ``n.<property_name> = $value`` — the
        common case. Both the label and the property name interpolate
        into the Cypher and pass through ``_validate_identifier``.

        ``exact_match=False``: case-insensitive substring match via
        AGE's CONTAINS. The merge flow uses this for "does any
        existing node mention this string?" queries; for now we rely
        on AGE's built-in CONTAINS rather than a Postgres pg_trgm
        index on the underlying entity table. Slice 6 wires the
        index for performance; correctness is the same either way.
        """
        validated_label = _validate_identifier(label, "node label")
        validated_prop = _validate_identifier(property_name, "property name")

        if exact_match:
            cypher = (
                f"MATCH (n:{validated_label}) "
                f"WHERE n.{validated_prop} = $value "
                f"RETURN n"
            )
            params = {"value": _coerce_for_age(property_value)}
        else:
            # AGE CONTAINS is case-sensitive on the value; compare
            # via toLower on both sides to mirror callers' fuzzy
            # expectations. property_value is stringified for the
            # comparison since CONTAINS only operates on strings.
            cypher = (
                f"MATCH (n:{validated_label}) "
                f"WHERE toLower(n.{validated_prop}) CONTAINS toLower($value) "
                f"RETURN n"
            )
            params = {"value": str(property_value)}

        rows = await self._execute_cypher(
            cypher,
            params=params,
            return_columns="(n agtype)",
        )
        nodes: List[Node] = []
        for row in rows:
            parsed = self._parse_agtype(row[0])
            if isinstance(parsed, dict):
                node = self._vertex_to_node(parsed)
                if node is not None:
                    nodes.append(node)
        return nodes

    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True,
    ) -> List[Node]:
        """Property-similarity search using AGE Cypher CONTAINS scoring.

        Called from services/merge/new_merger.py:1068 as the merge
        flow's similarity fallback. The Neo4j adapter does this with
        ``apoc.text.distance`` (Levenshtein) + ``apoc.text.doubleMetaphone``
        (phonetic) — APOC isn't available on AGE, so we score
        candidates by what fraction of property values appear as
        case-insensitive substrings in the corresponding node
        property. That's a cheaper signal than Levenshtein but it's
        the same shape of "fuzzy enough to catch typos and casing
        variants" that the merge flow expects.

        For each candidate node:

            score = (number of property values that match via
                     CONTAINS) / (total non-system properties supplied)

        Candidates with ``score >= similarity_threshold`` are
        returned, ordered by score descending and capped at
        ``max_results``.

        Embedding-based pgvector similarity stays out of scope for
        this slice — it needs upstream changes to where embeddings
        are generated and persisted (a sibling table indexed by node
        id) and isn't a per-adapter concern. ``include_relationships``
        is currently a no-op for the same reason; slice 7+ folds in
        relationship-pattern bonuses if the merge flow needs them.

        Returns ``[]`` rather than raising on empty input — matches
        Neo4j's behaviour and the merge flow's "fallback gives up
        gracefully" expectation.
        """
        # Filter out system / metadata properties so we score on
        # user-meaningful values only. Mirrors Neo4j's same filter
        # (constants.SYSTEM_PROPERTIES is the shared source).
        scoring_props = {
            k: v
            for k, v in (properties or {}).items()
            if v is not None and k not in SYSTEM_PROPERTIES
        }
        if not scoring_props:
            return []

        validated_label = _validate_identifier(label, "node label")

        # Build per-property CASE expressions in Cypher: each
        # contributes 1.0 to the score when the candidate node's
        # value contains the supplied value (case-insensitive).
        # AGE Cypher supports CASE WHEN ... THEN ... ELSE END and
        # CONTAINS; toLower wraps both sides for case-insensitive
        # match. Property name interpolates (validated); the
        # supplied value flows through params as $value<idx>.
        case_expressions: List[str] = []
        params: Dict[str, Any] = {
            "threshold": similarity_threshold,
            "max_results": max_results,
        }
        for idx, (key, value) in enumerate(scoring_props.items()):
            validated_prop = _validate_identifier(key, "property name")
            param_key = f"value{idx}"
            params[param_key] = str(value)
            case_expressions.append(
                f"CASE WHEN toLower(toString(coalesce(n.{validated_prop}, ''))) "
                f"CONTAINS toLower(${param_key}) THEN 1.0 ELSE 0.0 END"
            )

        score_expr = " + ".join(case_expressions)
        denom = float(len(case_expressions))

        # AGE Cypher accepts WITH … RETURN; the score is a numeric
        # agtype that we filter and order on.
        cypher = (
            f"MATCH (n:{validated_label}) "
            f"WITH n, ({score_expr}) / {denom} AS similarity_score "
            f"WHERE similarity_score >= $threshold "
            f"RETURN n, similarity_score "
            f"ORDER BY similarity_score DESC "
            f"LIMIT $max_results"
        )

        rows = await self._execute_cypher(
            cypher,
            params=params,
            return_columns="(n agtype, similarity_score agtype)",
        )

        nodes: List[Node] = []
        seen_ids: set = set()
        for row in rows:
            parsed = self._parse_agtype(row[0])
            if not isinstance(parsed, dict):
                continue
            node = self._vertex_to_node(parsed)
            if node is None or node.id in seen_ids:
                continue
            nodes.append(node)
            seen_ids.add(node.id)
        return nodes

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
        """Find a node by user-facing id property (not AGE's internal
        numeric id). Returns None when the id isn't present."""
        rows = await self._execute_cypher(
            """
            MATCH (n {id: $id})
            RETURN n
            LIMIT 1
            """,
            params={"id": node_id},
            return_columns="(n agtype)",
        )
        if not rows:
            return None
        parsed = self._parse_agtype(rows[0][0])
        if not isinstance(parsed, dict):
            return None
        return self._vertex_to_node(parsed)

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
