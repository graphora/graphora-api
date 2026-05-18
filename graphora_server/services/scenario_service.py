"""B6-scenario slice 1: persisted scenario snapshots.

A scenario is a named, point-in-time snapshot of a transform's
graph — the data primitive behind "what if we re-ran extraction
with this ontology change?" or "what if Alice and Alicia are the
same entity?" workflows. Slice 1 lands the foundation: data
model, dual-backend service (Postgres or in-memory), CRUD
surface for create / list / get / delete. Mutations on a scenario
graph and CoW (diff-from-parent) storage come in slice 2 once
the read API shape proves stable.

The service is intentionally storage-agnostic about the graph
itself: callers pass a materialized :class:`GraphResponse` to
``create_from_transform``. The API layer is responsible for
loading the transform graph (via ``_load_graph_for_diff``) before
calling here — same pattern as ``DiffService`` and
``CorpusScorer`` use. That keeps the service unit-testable
without DB or HTTP plumbing.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from psycopg.types.json import Json

from graphora_server.config import settings
from graphora_server.db import postgres as db
from graphora_server.schemas.graph import GraphResponse
from graphora_server.schemas.graph_changes import SaveGraphRequest

logger = logging.getLogger(__name__)


class ScenarioConflictError(Exception):
    """Raised when create_from_transform would violate the
    (user_id, transform_id, name) uniqueness constraint.

    The API layer maps this to HTTP 409 so the CLI can show "a
    scenario with this name already exists for that transform"
    rather than a 500 stack trace. Surfacing a typed exception
    instead of leaking the raw Postgres UniqueViolation keeps
    the service decoupled from the SQL driver.
    """


class ScenarioNotFoundError(Exception):
    """Raised when a scenario id doesn't resolve for the caller.

    Maps to 404 at the API layer. We don't distinguish "doesn't
    exist" from "belongs to another tenant" — same posture as
    ``/decisions`` and ``/diff`` to avoid leaking cross-tenant
    existence.
    """


class ScenarioMutationError(Exception):
    """Raised when a mutation request would leave the scenario in
    an invalid state.

    Validation cases:
      * A node-delete leaves an edge with a dangling source or
        target endpoint. The caller can fix this by either also
        deleting the edge or by not deleting the node.
      * A node-create with an id that collides with an existing
        node id.
      * An edge-create whose source or target id doesn't resolve
        to a node in the final state (including newly-created
        nodes in the same mutation batch).

    Maps to HTTP 422 at the API layer — the request was
    well-formed (Pydantic validation passed) but its semantic
    contract failed. Distinct from 404 (which means "scenario
    doesn't exist") and 409 (which is for name conflicts).
    """


def _empty_diff() -> Dict[str, Any]:
    """Canonical empty diff shape. Used at create time and as
    the default for the migration's column. Centralized so the
    six keys stay in lockstep across the service, the migration
    default, and tests."""
    return {
        "added_nodes": [],
        "removed_node_ids": [],
        "updated_nodes": [],
        "added_edges": [],
        "removed_edge_ids": [],
        "updated_edges": [],
    }


def _compute_diff(
    *,
    base: Dict[str, Any],
    target_nodes_by_id: Dict[str, Dict[str, Any]],
    target_edges_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the canonical-state diff for (base → target).

    The diff is one-representation-per-(base, target) pair: any
    two callers asking the same question get the same diff
    back, which means diff equality is a meaningful test
    assertion and incremental changes can be diffed against
    each other for review purposes.

    Algorithm: index base by id; for each target item, classify
    as ``added`` (not in base), ``updated`` (in base, different
    content), or unchanged (in base, identical — not emitted).
    For each base id not in target, emit ``removed``.

    Equality is structural via Python's ``==`` on dicts (recursive,
    type-aware). A target node whose properties bag has the
    same keys + values as the base — even if dict ordering
    differs internally — is treated as unchanged. That keeps
    the diff small for property-update-only mutations where
    the unchanged keys still serialize alongside the changed
    ones.
    """
    base_nodes_by_id: Dict[str, Dict[str, Any]] = {
        node["id"]: node for node in (base.get("nodes") or [])
    }
    base_edges_by_id: Dict[str, Dict[str, Any]] = {
        edge["id"]: edge for edge in (base.get("edges") or [])
    }

    added_nodes: List[Dict[str, Any]] = []
    updated_nodes: List[Dict[str, Any]] = []
    removed_node_ids: List[str] = []
    for nid, target in target_nodes_by_id.items():
        base_node = base_nodes_by_id.get(nid)
        if base_node is None:
            added_nodes.append(target)
        elif base_node != target:
            updated_nodes.append(target)
    for nid in base_nodes_by_id:
        if nid not in target_nodes_by_id:
            removed_node_ids.append(nid)

    added_edges: List[Dict[str, Any]] = []
    updated_edges: List[Dict[str, Any]] = []
    removed_edge_ids: List[str] = []
    for eid, target in target_edges_by_id.items():
        base_edge = base_edges_by_id.get(eid)
        if base_edge is None:
            added_edges.append(target)
        elif base_edge != target:
            updated_edges.append(target)
    for eid in base_edges_by_id:
        if eid not in target_edges_by_id:
            removed_edge_ids.append(eid)

    return {
        "added_nodes": added_nodes,
        "removed_node_ids": removed_node_ids,
        "updated_nodes": updated_nodes,
        "added_edges": added_edges,
        "removed_edge_ids": removed_edge_ids,
        "updated_edges": updated_edges,
    }


@dataclass
class Scenario:
    """Service-layer scenario record.

    Mirrors the wire-level :class:`graphora_server.schemas.scenario.Scenario`
    but stores the graph as a plain dict so the service stays
    Pydantic-import-free. The API layer adapts via
    ``GraphResponse.model_validate`` before returning to the
    client.

    Storage layout (B6-scenario slice 2c): the graph lives in
    two columns. ``graph_snapshot`` is the IMMUTABLE base captured
    at create_from_transform time — it never changes after the
    create. ``graph_diff`` accumulates all mutations as a
    canonical-state delta. The "current view" of the scenario is
    ``resolved_graph()`` which applies the diff to the base; the
    API surface uses that helper so callers never see the split.

    CoW win: a mutation writes only the diff (typically small),
    not the full snapshot — eliminating write amplification on
    repeat mutations.
    """

    id: str
    user_id: str
    transform_id: str
    name: str
    description: Optional[str]
    graph_snapshot: Dict[str, Any]
    created_at: str
    parent_scenario_id: Optional[str] = None
    # Slice 2c: canonical-state diff vs ``graph_snapshot``. The
    # default field reads ``_empty_diff()`` so a Scenario built
    # without an explicit diff (e.g., in tests, or in the
    # in-memory create path) starts in the "no mutations yet"
    # state — its resolved_graph() equals graph_snapshot.
    graph_diff: Dict[str, Any] = field(default_factory=_empty_diff)

    def resolved_graph(self) -> Dict[str, Any]:
        """Apply ``graph_diff`` to ``graph_snapshot`` and return
        the materialized view.

        Order matters: remove first, then add, then update.
          * Remove first so an id that appears in both
            ``removed_node_ids`` and ``added_nodes`` ends up
            present (the add wins) — useful for "delete + re-
            create with new properties" patterns the
            apply_mutations diff computer may produce.
          * Add second so newly-introduced ids exist before any
            updates would try to reach them.
          * Update last so it overrides anything the add+remove
            pair left behind.

        Returns a GraphResponse-shaped dict including
        ``total_nodes``, ``total_edges``, and ``metadata``
        passed through from the immutable base. Callers (the
        API layer's ``_to_response``) feed this directly into
        ``GraphResponse.model_validate``.
        """
        base = self.graph_snapshot or {}
        diff = self.graph_diff or {}

        nodes_by_id = {n["id"]: dict(n) for n in (base.get("nodes") or [])}
        edges_by_id = {e["id"]: dict(e) for e in (base.get("edges") or [])}

        for nid in diff.get("removed_node_ids") or []:
            nodes_by_id.pop(nid, None)
        for eid in diff.get("removed_edge_ids") or []:
            edges_by_id.pop(eid, None)
        for node in diff.get("added_nodes") or []:
            nodes_by_id[node["id"]] = dict(node)
        for edge in diff.get("added_edges") or []:
            edges_by_id[edge["id"]] = dict(edge)
        for node in diff.get("updated_nodes") or []:
            nodes_by_id[node["id"]] = dict(node)
        for edge in diff.get("updated_edges") or []:
            edges_by_id[edge["id"]] = dict(edge)

        return {
            "nodes": list(nodes_by_id.values()),
            "edges": list(edges_by_id.values()),
            "total_nodes": len(nodes_by_id),
            "total_edges": len(edges_by_id),
            # Metadata lives on the immutable base — mutations
            # don't touch it. Preserves the same invariant the
            # snapshot-shape reviewer-fix established.
            "metadata": dict(base.get("metadata") or {}),
        }


# Module-level shared dev-mode store. Reviewer-flagged High on
# commit d7a1f6e: the API constructs a fresh ScenarioService per
# request, and the default constructor allocated a fresh empty
# list each time. With no DATABASE_URL configured, POST landed
# data in one instance and GET read from another, so the dev-
# mode CRUD was effectively non-persistent across requests. The
# fix mirrors DisputedPairsService's pattern (commit 26d3e89):
# share one list across instances; tenant filtering at read time
# keeps users isolated even though the underlying list is shared.
_DEFAULT_MEMORY_STORE: List[Scenario] = []


def _reset_default_memory_store_for_tests() -> None:
    """Clear the dev-mode shared memory store. Test fixtures that
    fall through to the production constructor (no
    ``memory_store=`` arg) should call this between scenarios to
    keep tests isolated. Most tests pass ``memory_store=[]``
    explicitly and don't need this — it's only for direct-DB-mock
    tests that construct a default service."""
    _DEFAULT_MEMORY_STORE.clear()


class ScenarioService:
    """Dual-backend store for scenario snapshots.

    Postgres when ``DATABASE_URL`` is configured (production
    path); in-memory list otherwise (zero-config dev path,
    matching the dual-backend convention used by
    :class:`DecisionLogService` and :class:`EntityLedgerService`).

    Tenant scoping is enforced at every read: a request for
    another user's scenario_id raises :class:`ScenarioNotFoundError`,
    mirrored from the same "no leak" posture the rest of the
    Gate-4 surfaces use. There are no legacy NULL-user_id rows
    to accommodate (this is a fresh table from migration 19), so
    the user_id filter is unconditional — simpler than
    DecisionLogService's "optional for legacy" branch.

    Failure posture on the Postgres path: DB exceptions
    propagate. Reviewer-flagged High on commit d7a1f6e — unlike
    DecisionLogService (where decision rows are observability
    and a swallowed insert is acceptable), scenarios are
    user-owned data. A DB outage or missing migration must
    surface as 5xx at the API boundary rather than masquerading
    as "you have no scenarios" (empty list) or "scenario not
    found" (404). The API layer is responsible for translating
    propagated DB exceptions into 5xx — same posture as
    DisputedPairsService (commit 26d3e89). The create path's
    UniqueViolation → ScenarioConflictError mapping is the
    one exception: a typed business error needs typed
    surfacing as 409, and is distinct from "the DB is on fire."
    """

    TABLE_NAME = "scenarios"

    def __init__(
        self,
        memory_store: Optional[List[Scenario]] = None,
    ) -> None:
        self._enabled = bool(settings.DATABASE_URL or settings.resolved_database_url)
        # Reviewer-flagged High on commit d7a1f6e. See the
        # _DEFAULT_MEMORY_STORE comment above for the full
        # rationale: per-request service instances must share the
        # dev-mode list, otherwise POST and GET land on different
        # in-memory dictionaries and the CRUD looks broken.
        # Tests pass ``memory_store=[]`` explicitly for isolation;
        # production constructions fall through to the shared
        # store.
        if memory_store is not None:
            self._memory_store = memory_store
        else:
            self._memory_store = _DEFAULT_MEMORY_STORE

    # Public API -----------------------------------------------------------------

    async def create_from_transform(
        self,
        *,
        user_id: str,
        transform_id: str,
        name: str,
        graph: GraphResponse,
        description: Optional[str] = None,
    ) -> Scenario:
        """Snapshot the supplied transform graph into a new scenario.

        ``graph`` is the materialized graph the caller already
        loaded (typically via the same ``_load_graph_for_diff``
        helper the /diff and /golden/score endpoints use). Keeping
        the load out of this service means the service can be
        unit-tested against any GraphResponse without standing up
        the storage factory.

        Raises:
            ScenarioConflictError: the (user_id, transform_id,
                name) tuple already exists. Pre-flight check
                covers both backends; the DB CHECK is the final
                guarantor for the Postgres path.
        """
        # Pre-flight: surface the conflict before doing any work.
        # On Postgres this is racy with concurrent INSERTs from
        # the same user, but the DB unique constraint is the
        # canonical guarantor — the pre-flight just gives us a
        # clean 409 in the common single-writer case without
        # forcing the API layer to parse Postgres error codes.
        existing = await self._find_by_name(
            user_id=user_id, transform_id=transform_id, name=name
        )
        if existing is not None:
            raise ScenarioConflictError(
                f"Scenario named {name!r} already exists for transform "
                f"{transform_id!r}."
            )

        scenario = Scenario(
            id=str(uuid.uuid4()),
            user_id=user_id,
            transform_id=transform_id,
            name=name,
            description=description,
            graph_snapshot=graph.model_dump(mode="json"),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        if self._enabled:
            try:
                # Slice 2c: include the empty graph_diff so new
                # scenarios start with the canonical "no
                # mutations yet" state. The column has a server-
                # side default but we send it explicitly so the
                # in-memory + postgres paths produce structurally
                # identical Scenario records.
                await db.execute(
                    """
                    INSERT INTO scenarios (
                        id, user_id, transform_id, parent_scenario_id,
                        name, description, graph_snapshot, graph_diff,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    scenario.id,
                    scenario.user_id,
                    scenario.transform_id,
                    scenario.parent_scenario_id,
                    scenario.name,
                    scenario.description,
                    Json(scenario.graph_snapshot),
                    Json(scenario.graph_diff),
                    scenario.created_at,
                )
            except Exception as exc:
                # Most likely cause is the unique constraint
                # (race with another concurrent create); surface
                # the typed conflict so the API still returns 409
                # rather than 500.
                msg = str(exc).lower()
                if "scenarios_name_unique_per_transform" in msg or "unique" in msg:
                    raise ScenarioConflictError(
                        f"Scenario named {name!r} already exists for "
                        f"transform {transform_id!r}."
                    ) from exc
                logger.error("Failed to insert scenario: %s", exc)
                raise
        else:
            self._memory_store.append(scenario)
        return scenario

    async def list_for_user(self, user_id: str) -> List[Scenario]:
        """All scenarios owned by ``user_id``, newest first.

        Returns the full :class:`Scenario` (graph included) for
        simplicity at this layer; the API endpoint projects to
        :class:`ScenarioSummary` before returning to the wire.
        Keeping the projection at the API layer means the
        service doesn't have to expose two shapes.

        DB exceptions propagate per the class docstring: an
        empty list means "no scenarios," not "the DB read
        failed." The API layer surfaces propagated errors as
        5xx so operators can distinguish the two states.
        """
        if self._enabled:
            rows = await db.fetch(
                """
                SELECT id, user_id, transform_id, parent_scenario_id,
                       name, description, graph_snapshot, graph_diff,
                       created_at
                FROM scenarios
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                user_id,
            )
            return self._rows_to_scenarios(rows)
        return sorted(
            (s for s in self._memory_store if s.user_id == user_id),
            key=lambda s: s.created_at,
            reverse=True,
        )

    async def get(self, scenario_id: str, user_id: str) -> Scenario:
        """Fetch a single scenario by id, tenant-scoped.

        Raises:
            ScenarioNotFoundError: id doesn't resolve OR belongs
                to another tenant. Single exception for both
                cases — never leak cross-tenant existence.

        DB exceptions propagate — see class docstring. A
        "DB on fire" outcome must not masquerade as "not
        found"; the API maps the two to different status codes
        (404 vs 5xx), which means the service has to surface
        them as different exceptions too.
        """
        if self._enabled:
            rows = await db.fetch(
                """
                SELECT id, user_id, transform_id, parent_scenario_id,
                       name, description, graph_snapshot, graph_diff,
                       created_at
                FROM scenarios
                WHERE id = %s AND user_id = %s
                """,
                scenario_id,
                user_id,
            )
            scenarios = self._rows_to_scenarios(rows)
            if not scenarios:
                raise ScenarioNotFoundError(f"Scenario {scenario_id!r} not found.")
            return scenarios[0]
        for scenario in self._memory_store:
            if scenario.id == scenario_id and scenario.user_id == user_id:
                return scenario
        raise ScenarioNotFoundError(f"Scenario {scenario_id!r} not found.")

    async def delete(self, scenario_id: str, user_id: str) -> None:
        """Delete a scenario by id, tenant-scoped.

        Idempotency choice: a request to delete a scenario the
        caller doesn't own (or one that never existed) raises
        :class:`ScenarioNotFoundError` rather than 204'ing
        silently. This is the same posture as the get endpoint
        and matches the principle that "tell me about something
        you don't own" returns the same 404 as "tell me about
        nothing." A silent 204 would let an attacker probe for
        existence by timing the response.

        DB exceptions propagate — see class docstring. A
        "DB unreachable" failure must NOT 404, because that
        would tell a malicious client the row also doesn't
        exist (which is information the operator may have, but
        the client should not).
        """
        if self._enabled:
            # RETURNING confirms the row existed AND belonged
            # to the caller. Without it we'd issue a DELETE
            # with no rows affected and have to do a second
            # SELECT to distinguish "didn't exist" from
            # "existed but wrong tenant".
            rows = await db.fetch(
                """
                DELETE FROM scenarios
                WHERE id = %s AND user_id = %s
                RETURNING id
                """,
                scenario_id,
                user_id,
            )
            if not rows:
                raise ScenarioNotFoundError(f"Scenario {scenario_id!r} not found.")
            return
        for index, scenario in enumerate(self._memory_store):
            if scenario.id == scenario_id and scenario.user_id == user_id:
                self._memory_store.pop(index)
                return
        raise ScenarioNotFoundError(f"Scenario {scenario_id!r} not found.")

    async def apply_mutations(
        self,
        scenario_id: str,
        user_id: str,
        changes: SaveGraphRequest,
    ) -> Scenario:
        """Apply node + edge mutations to a scenario's graph.

        Slice 2c (CoW): writes a canonical-state diff vs the
        immutable ``graph_snapshot`` base captured at create
        time. ``graph_snapshot`` itself is never modified —
        mutations only update ``graph_diff``. The slice 2b
        contract is unchanged for callers: this still returns
        the updated Scenario record with the resolved view
        reachable via ``scenario.resolved_graph()``.

        Algorithm:
          1. Load the scenario (raises NotFound on missing /
             cross-tenant).
          2. Get the CURRENT resolved view (base + existing
             diff) — this is the state the caller's mutations
             are described against.
          3. Build the target graph in memory: start from the
             current resolved view, apply node creates →
             updates → deletes, then edge creates → updates →
             deletes. Same per-dimension ordering slice 2b
             established.
          4. Validate the target state (no id collisions on
             creates; no dangling edges after deletes). Same
             rules as slice 2b — failures raise
             ScenarioMutationError → API 422.
          5. Compute the new canonical-state diff between the
             immutable base and the target state. Persist
             only the diff.

        Failure isolation: the dangling-edge / id-collision
        validation happens BEFORE any persist, so a rejected
        mutation leaves the diff untouched (slice 2b's
        atomicity property still holds).
        """
        scenario = await self.get(scenario_id, user_id)
        # Start from the CURRENT view (base + existing diff),
        # not from the immutable base. A mutation describes
        # what the caller wants the END state to look like;
        # they're not aware of the storage split between base
        # and diff.
        current_view = scenario.resolved_graph()

        # Build mutable index of the current graph. Dicts keyed
        # by id make the apply loops O(1) per operation; the
        # cost is one extra pass to repack to lists at the end.
        nodes_by_id: Dict[str, Dict[str, Any]] = {
            node["id"]: dict(node) for node in (current_view.get("nodes") or [])
        }
        edges_by_id: Dict[str, Dict[str, Any]] = {
            edge["id"]: dict(edge) for edge in (current_view.get("edges") or [])
        }

        # ---- Nodes ----------------------------------------------------------
        node_changes = changes.nodes
        if node_changes is not None:
            # Creates first — id-collision detection runs against
            # the EXISTING set (a "create" with an id that
            # already exists is a contract violation, not an
            # implicit update).
            for create in node_changes.created:
                if create.id in nodes_by_id:
                    raise ScenarioMutationError(
                        f"Cannot create node {create.id!r}: a node with "
                        f"that id already exists in the scenario."
                    )
                nodes_by_id[create.id] = {
                    "id": create.id,
                    "label": create.label,
                    "type": create.type,
                    "properties": dict(create.properties or {}),
                }
            # Updates second — merge properties, with None
            # signalling "delete this key" (matches the existing
            # SaveGraphRequest semantics).
            for update in node_changes.updated:
                existing = nodes_by_id.get(update.id)
                if existing is None:
                    raise ScenarioMutationError(
                        f"Cannot update node {update.id!r}: no such node "
                        f"in the scenario."
                    )
                props = dict(existing.get("properties") or {})
                for key, value in update.properties.items():
                    if value is None:
                        props.pop(key, None)
                    else:
                        props[key] = value
                existing["properties"] = props
            # Deletes last — collect the ids being removed for
            # the dangling-edge validation below.
            for delete_id in node_changes.deleted:
                if delete_id not in nodes_by_id:
                    raise ScenarioMutationError(
                        f"Cannot delete node {delete_id!r}: no such node "
                        f"in the scenario."
                    )
                del nodes_by_id[delete_id]

        # ---- Edges ----------------------------------------------------------
        edge_changes = changes.edges
        if edge_changes is not None:
            for create in edge_changes.created:
                if create.id in edges_by_id:
                    raise ScenarioMutationError(
                        f"Cannot create edge {create.id!r}: an edge with "
                        f"that id already exists in the scenario."
                    )
                edges_by_id[create.id] = {
                    "id": create.id,
                    "source": create.source,
                    "target": create.target,
                    "type": create.type,
                    "label": create.label,
                    "properties": dict(create.properties or {}),
                }
            for update in edge_changes.updated:
                existing = edges_by_id.get(update.id)
                if existing is None:
                    raise ScenarioMutationError(
                        f"Cannot update edge {update.id!r}: no such edge "
                        f"in the scenario."
                    )
                props = dict(existing.get("properties") or {})
                for key, value in update.properties.items():
                    if value is None:
                        props.pop(key, None)
                    else:
                        props[key] = value
                existing["properties"] = props
            for delete_id in edge_changes.deleted:
                if delete_id not in edges_by_id:
                    raise ScenarioMutationError(
                        f"Cannot delete edge {delete_id!r}: no such edge "
                        f"in the scenario."
                    )
                del edges_by_id[delete_id]

        # ---- Dangling-edge validation --------------------------------------
        # After all mutations, every surviving edge must point
        # at two surviving nodes. Catches the "delete a node
        # but forget to delete its incident edges" case the
        # API contract rejects. Reporting all offending edges
        # at once (not just the first) helps the caller fix
        # the batch in one shot.
        node_ids = set(nodes_by_id.keys())
        dangling: List[str] = []
        for edge_id, edge in edges_by_id.items():
            if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
                dangling.append(edge_id)
        if dangling:
            raise ScenarioMutationError(
                f"Mutation would leave dangling edges: {dangling!r}. "
                f"Delete these edges explicitly or keep their "
                f"source/target nodes."
            )

        # ---- Compute canonical-state diff vs immutable base ----------------
        # Slice 2c: instead of rewriting graph_snapshot, write
        # a diff that, when applied to the immutable base via
        # ``Scenario.resolved_graph()``, produces the same
        # post-mutation view. Canonical-state form (one
        # representation per (base, target) pair) keeps the
        # diff equality-testable and bounds compaction needs
        # to the slice 2c scope.
        new_diff = _compute_diff(
            base=scenario.graph_snapshot or {"nodes": [], "edges": []},
            target_nodes_by_id=nodes_by_id,
            target_edges_by_id=edges_by_id,
        )

        if self._enabled:
            await db.execute(
                """
                UPDATE scenarios
                SET graph_diff = %s
                WHERE id = %s AND user_id = %s
                """,
                Json(new_diff),
                scenario_id,
                user_id,
            )
            # Re-read to pick up any server-side defaults
            # (timestamps, etc.) and return a consistent
            # record. Costs one extra round-trip but keeps the
            # contract simple ("apply_mutations returns the
            # current scenario").
            return await self.get(scenario_id, user_id)

        # Memory path — mutate the in-memory record's diff in
        # place. graph_snapshot is intentionally NOT touched:
        # the immutability of the base is the load-bearing
        # invariant of the CoW design.
        for stored in self._memory_store:
            if stored.id == scenario_id and stored.user_id == user_id:
                stored.graph_diff = new_diff
                return stored
        # Should be unreachable — get() returned successfully
        # above. Defensive raise to avoid a silent None return.
        raise ScenarioNotFoundError(f"Scenario {scenario_id!r} vanished mid-mutation.")

    # Helpers --------------------------------------------------------------------

    async def _find_by_name(
        self, *, user_id: str, transform_id: str, name: str
    ) -> Optional[Scenario]:
        """Internal: pre-flight conflict check for create_from_transform.

        DB exceptions propagate to the caller. The pre-flight is
        technically best-effort (the DB unique constraint is the
        canonical guarantor), but a degraded DB surfacing during
        pre-flight + then again during INSERT just doubles the
        latency before failure. Let the error propagate so the
        API returns 5xx promptly instead of falling through to a
        guaranteed-to-fail INSERT path.
        """
        if self._enabled:
            rows = await db.fetch(
                """
                SELECT id, user_id, transform_id, parent_scenario_id,
                       name, description, graph_snapshot, graph_diff,
                       created_at
                FROM scenarios
                WHERE user_id = %s AND transform_id = %s AND name = %s
                LIMIT 1
                """,
                user_id,
                transform_id,
                name,
            )
            scenarios = self._rows_to_scenarios(rows)
            return scenarios[0] if scenarios else None
        for scenario in self._memory_store:
            if (
                scenario.user_id == user_id
                and scenario.transform_id == transform_id
                and scenario.name == name
            ):
                return scenario
        return None

    @classmethod
    def _rows_to_scenarios(cls, rows: List[Dict[str, Any]]) -> List[Scenario]:
        """Per-row conversion with isolation: one malformed row
        gets logged + skipped instead of poisoning the whole
        query result (same pattern as DecisionLogService)."""
        scenarios: List[Scenario] = []
        for row in rows:
            try:
                scenarios.append(cls._row_to_scenario(row))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed scenarios row (id=%s): %s",
                    row.get("id") if isinstance(row, dict) else None,
                    exc,
                )
        return scenarios

    @staticmethod
    def _row_to_scenario(row: Dict[str, Any]) -> Scenario:
        created_at = row["created_at"]
        # graph_diff may be missing on rows written before
        # migration 21 ran (the column was added with a default,
        # but a row read mid-deploy or a test fixture might
        # omit it). Fall back to the canonical empty diff so
        # resolved_graph() degrades to "base only" rather than
        # KeyError-ing.
        graph_diff = row.get("graph_diff") or _empty_diff()
        return Scenario(
            id=str(row["id"]),
            user_id=row["user_id"],
            transform_id=row["transform_id"],
            parent_scenario_id=(
                str(row["parent_scenario_id"])
                if row.get("parent_scenario_id") is not None
                else None
            ),
            name=row["name"],
            description=row.get("description"),
            graph_snapshot=row.get("graph_snapshot") or {"nodes": [], "edges": []},
            graph_diff=graph_diff,
            created_at=(
                created_at.isoformat()
                if hasattr(created_at, "isoformat")
                else created_at
            ),
        )
