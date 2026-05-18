"""HTTP client used by the MCP tools.

Kept intentionally small: one class, one httpx.AsyncClient, and
one method per MCP tool. Keeps the surface predictable for tests
(every tool has exactly one network call to mock) and avoids
leaking schema details from the REST API into the MCP layer.

Auth + base URL come from the environment:
    GRAPHORA_API_URL      default ``http://localhost:8000``
    GRAPHORA_AUTH_TOKEN   required unless AUTH_BYPASS is on

``AUTH_BYPASS`` is not read here — bypass is a server-side setting
and requests still travel with whatever token the env provides
(empty is fine when the server bypasses auth).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


DEFAULT_API_URL = "http://localhost:8000"
_API_V1 = "/api/v1"


class GraphoraClientError(RuntimeError):
    """Raised when the API returns a non-2xx response.

    Holds the HTTP status code and response body so tools can
    render a useful message back to the agent without needing to
    re-parse httpx exceptions.
    """

    def __init__(self, status_code: int, body: str, *, url: str = ""):
        self.status_code = status_code
        self.body = body
        self.url = url
        super().__init__(f"Graphora API {status_code} @ {url}: {body[:200]}")


class GraphoraClient:
    """Thin async wrapper around the Graphora REST API.

    Instantiate once per MCP session. The underlying httpx.AsyncClient
    holds the connection pool; call ``aclose()`` on shutdown (or use
    as an async context manager).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        *,
        timeout: float = 60.0,
    ):
        self.base_url = (
            base_url or os.environ.get("GRAPHORA_API_URL") or DEFAULT_API_URL
        ).rstrip("/")
        self.auth_token = auth_token or os.environ.get("GRAPHORA_AUTH_TOKEN", "")
        headers = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    async def __aenter__(self) -> "GraphoraClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- extract_document ----------------------------------------------

    async def upload_file(
        self,
        file_path: str,
        ontology_id: Optional[str] = None,
        *,
        schemaless: bool = False,
    ) -> Dict[str, Any]:
        """POST a local file.

        Routing:
            * ``schemaless=True``   → /transform/schemaless/upload
              (extract with the permissive generic schema; caller
              runs ``finalize_ontology`` after completion to get a
              refined ontology)
            * ``ontology_id``       → /transform/{ontology_id}/upload
              (extract against a specific stored ontology)
            * neither               → /transform/upload
              (auto-schema: pre-extraction text peek + inferred
              ontology drives extraction)
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if schemaless and ontology_id:
            raise ValueError("schemaless=True and ontology_id are mutually exclusive")
        if schemaless:
            endpoint = f"{_API_V1}/transform/schemaless/upload"
        elif ontology_id:
            endpoint = f"{_API_V1}/transform/{ontology_id}/upload"
        else:
            endpoint = f"{_API_V1}/transform/upload"

        with path.open("rb") as fh:
            files = {"files": (path.name, fh, "application/octet-stream")}
            resp = await self._client.post(endpoint, files=files)
        return _ok(resp)

    async def get_inferred_ontology(self, transform_id: str) -> Dict[str, Any]:
        resp = await self._client.get(
            f"{_API_V1}/transform/{transform_id}/inferred-ontology"
        )
        return _ok(resp)

    async def finalize_ontology(self, transform_id: str) -> Dict[str, Any]:
        resp = await self._client.post(
            f"{_API_V1}/transform/{transform_id}/finalize-ontology"
        )
        return _ok(resp)

    async def upload_url(
        self,
        url: str,
        ontology_id: Optional[str] = None,
        *,
        schemaless: bool = False,
    ) -> Dict[str, Any]:
        """Submit a URL for extraction.

        The REST API does not expose a URL upload endpoint today
        (Gate 2 A2 added ``parse_url`` on the parser side but not a
        route), so this fetches the URL content, writes it to a
        temp file, and posts that. Keeps the MCP tool surface
        clean — agents just pass a URL and we handle the detail.
        """
        import tempfile

        try:
            import trafilatura  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "URL extraction needs the [url] extra on graphora-server. "
                "Install with: pip install 'graphora-server[url]'"
            ) from exc

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise GraphoraClientError(0, f"Could not fetch URL: {url}", url=url)
        text = trafilatura.extract(downloaded) or ""
        if not text:
            raise GraphoraClientError(0, f"No extractable text at URL: {url}", url=url)

        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            return await self.upload_file(
                tmp_path, ontology_id=ontology_id, schemaless=schemaless
            )
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    async def get_status(self, transform_id: str) -> Dict[str, Any]:
        resp = await self._client.get(f"{_API_V1}/transform/status/{transform_id}")
        return _ok(resp)

    # ---- query_graph ---------------------------------------------------

    async def get_graph(
        self,
        transform_id: str,
        *,
        limit: int = 1000,
        skip: int = 0,
    ) -> Dict[str, Any]:
        resp = await self._client.get(
            f"{_API_V1}/graph/{transform_id}",
            params={"limit": limit, "skip": skip},
        )
        return _ok(resp)

    async def find_node(
        self,
        transform_id: str,
        node_id: str,
        *,
        page_size: int = 1000,
        max_pages: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Locate a node by id, paginating until found or exhausted.

        The REST API has no "fetch one node" endpoint, so evidence
        lookup has to scan. Pages are the API's natural chunk size
        (1000), and the ceiling of ``max_pages`` covers the server's
        hard cap of 10,000 nodes per graph. Returns the full graph
        slice containing the node so callers can also grab its
        edges — avoids a second roundtrip for get_evidence.
        """
        for page in range(max_pages):
            skip = page * page_size
            data = await self.get_graph(transform_id, limit=page_size, skip=skip)
            nodes = data.get("nodes", []) or []
            if any(n.get("id") == node_id for n in nodes):
                return data
            if len(nodes) < page_size:
                break
        return None

    async def find_edge(
        self,
        transform_id: str,
        edge_id: str,
        *,
        page_size: int = 1000,
        max_pages: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Locate an edge by id, paginating until found or exhausted.

        Mirrors ``find_node``'s pagination contract — same chunk
        size, same ``max_pages`` ceiling. Returns the full graph
        slice containing the edge so callers can pull the source
        and target node summaries from the same payload without a
        second roundtrip.

        Edge ids are stable across pagination pages because the
        predicate-bearing MERGE pattern (commit 797a75c — "Make
        Case 3 atomic via predicate-bearing MERGE") stamps each
        relationship with a UUID at creation. The id survives
        re-extractions of the same transform and is what the
        Decision Log's RELATIONSHIP_ACCEPTED/REJECTED entries
        carry as target_id.

        Returns None when the edge isn't present on any page within
        the ceiling. Same shape contract as ``find_node``.
        """
        for page in range(max_pages):
            skip = page * page_size
            data = await self.get_graph(transform_id, limit=page_size, skip=skip)
            edges = data.get("edges", []) or []
            if any(e.get("id") == edge_id for e in edges):
                return data
            # Page-size short-circuit: if we got fewer than page_size
            # nodes, we've seen the tail of the graph and don't need
            # more pages. We check nodes (not edges) for the
            # termination signal because pagination is keyed on
            # nodes server-side — a page with N nodes ships all
            # edges incident to those nodes, but the next page is
            # ``skip=N``, scoped to the next node window.
            if len(data.get("nodes", []) or []) < page_size:
                break
        return None

    async def get_decisions(
        self,
        transform_id: str,
        node_id: Optional[str] = None,
        edge_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch Decision Log entries for a transform via the API.

        Reviewer-flagged on commit 9ac9bb5 (B0-explain): MCP must
        not touch the DecisionLogService directly — that creates a
        new direct DB dependency from a process that's otherwise
        a pure HTTP client. The /decisions endpoint owns the read
        so MCP stays decoupled.

        Pass ``node_id`` OR ``edge_id`` to narrow to a single
        target's decisions; the endpoint rejects setting both with
        a 400. The schema-decision prefix is always returned for
        causation context regardless of which target kind is
        queried.

        Returns ``{decision_log, alternatives}``. ``alternatives``
        is empty when neither id is supplied (schema-only fetch) or
        when the target had no merge events. The endpoint always
        returns both keys regardless of state, so callers can rely
        on the response shape unconditionally.
        """
        params: Dict[str, Any] = {}
        if node_id is not None:
            params["node_id"] = node_id
        if edge_id is not None:
            params["edge_id"] = edge_id
        resp = await self._client.get(
            f"{_API_V1}/graph/{transform_id}/decisions",
            params=params,
        )
        return _ok(resp)

    async def get_cost_report(self, transform_id: str) -> Dict[str, Any]:
        """B5-obs: per-transform cost / token aggregation.

        Returns the same payload shape as the
        ``/api/v1/graph/{transform_id}/cost`` endpoint:
        ``{transform_id, total_calls, input_tokens, output_tokens,
        total_tokens, estimated_cost_usd, models_used,
        by_operation_type}``. ``estimated_cost_usd`` is a string
        (Decimal-precise) or None when no priced row was found.
        """
        resp = await self._client.get(
            f"{_API_V1}/graph/{transform_id}/cost",
        )
        return _ok(resp)

    async def get_budget_status(self) -> Dict[str, Any]:
        """B5-obs slice 2: authenticated user's current-period
        budget status. Returns the shape from
        ``/api/v1/budgets/me/status``: ``{state, current_spend_usd,
        cap_usd, period_start, period_end}``. ``state`` is one of
        ``unset`` / ``under`` / ``near`` / ``over``; the agent can
        use it to decide whether the next transform will succeed
        before submitting one."""
        resp = await self._client.get(f"{_API_V1}/budgets/me/status")
        return _ok(resp)

    async def diff_transforms(
        self,
        base_transform_id: str,
        compare_transform_id: str,
    ) -> Dict[str, Any]:
        """B3-diff: structured graph-state diff between two
        transforms. Returns the shape from
        ``/api/v1/graph/{base}/diff/{compare}``: ``{summary,
        added_nodes, removed_nodes, changed_nodes, added_edges,
        removed_edges, changed_edges}``.

        Node identity matches across transforms by canonical_id
        (Gate 4 entity resolution), with type:canonical_key as
        fallback."""
        resp = await self._client.get(
            f"{_API_V1}/graph/{base_transform_id}/diff/{compare_transform_id}",
        )
        return _ok(resp)

    async def list_contradictions(
        self,
        transform_id: str,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """B1-prob slice 2a: surface (target, property) pairs
        where the pipeline emitted multiple distinct claimed
        values.

        Returns the wire shape from
        ``/api/v1/graph/{tx}/contradictions``: ``{transform_id,
        min_confidence, contradictions, total_claims_scanned}``.
        Each entry in ``contradictions`` carries
        ``competing_claims`` sorted by confidence DESC (the
        first claim is the 'winning' value the pipeline
        picked) and a ``severity`` count (distinct-value count
        above the confidence floor).

        ``min_confidence`` filters low-confidence noise — set
        to 0.0 (default) to see everything; raise to surface
        only high-confidence disagreements.
        """
        params: Dict[str, Any] = {"min_confidence": min_confidence}
        resp = await self._client.get(
            f"{_API_V1}/graph/{transform_id}/contradictions",
            params=params,
        )
        return _ok(resp)

    async def list_disputed_pairs(
        self,
        transform_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        """B2-active: pending disputed pairs for the authenticated
        user (newest first). When ``transform_id`` is set, filters
        to a single run. The endpoint returns a list of pair
        dicts; this method passes the list through unchanged."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if transform_id is not None:
            params["transform_id"] = transform_id
        resp = await self._client.get(
            f"{_API_V1}/disputed-pairs",
            params=params,
        )
        return _ok(resp)

    async def label_disputed_pair(
        self,
        pair_id: str,
        decision: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """B2-active: apply a label to a disputed pair. Decision
        is one of ``match`` / ``not_match`` / ``skip``. Returns
        the updated pair dict — agents can render the new status
        without re-fetching."""
        body: Dict[str, Any] = {"decision": decision}
        if reason is not None:
            body["reason"] = reason
        resp = await self._client.post(
            f"{_API_V1}/disputed-pairs/{pair_id}/label",
            json=body,
        )
        return _ok(resp)


def _ok(resp: httpx.Response) -> Dict[str, Any]:
    if resp.status_code >= 400:
        raise GraphoraClientError(
            resp.status_code, resp.text, url=str(resp.request.url)
        )
    return resp.json()
