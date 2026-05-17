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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from psycopg.types.json import Json

from graphora_server.config import settings
from graphora_server.db import postgres as db
from graphora_server.schemas.graph import GraphResponse

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


@dataclass
class Scenario:
    """Service-layer scenario record.

    Mirrors the wire-level :class:`graphora_server.schemas.scenario.Scenario`
    but stores the graph as a plain dict so the service stays
    Pydantic-import-free. The API layer adapts via
    ``GraphResponse.model_validate`` before returning to the
    client.
    """

    id: str
    user_id: str
    transform_id: str
    name: str
    description: Optional[str]
    graph_snapshot: Dict[str, Any]
    created_at: str
    parent_scenario_id: Optional[str] = None


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
                await db.execute(
                    """
                    INSERT INTO scenarios (
                        id, user_id, transform_id, parent_scenario_id,
                        name, description, graph_snapshot, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    scenario.id,
                    scenario.user_id,
                    scenario.transform_id,
                    scenario.parent_scenario_id,
                    scenario.name,
                    scenario.description,
                    Json(scenario.graph_snapshot),
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
                       name, description, graph_snapshot, created_at
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
                       name, description, graph_snapshot, created_at
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
                       name, description, graph_snapshot, created_at
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
            created_at=(
                created_at.isoformat()
                if hasattr(created_at, "isoformat")
                else created_at
            ),
        )
