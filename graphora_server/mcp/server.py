"""FastMCP server that exposes Graphora as three agent tools.

Tool docstrings are the agent-facing descriptions — the MCP SDK
auto-generates the JSON Schema the client sees. Keep them written
for an agent: lead with what the tool does, then each argument's
semantics and return shape.

The server talks HTTP to a running Graphora deployment. Agent
clients (Claude Desktop, Cursor) launch this process via stdio;
it stays alive for the lifetime of the conversation.

Tool implementations are factored into module-level
``_tool_impl_*`` async functions so tests can exercise the
business logic without instantiating FastMCP — the [mcp] extra
only needs to be installed to actually *register* the server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from graphora_server.mcp.client import GraphoraClient, GraphoraClientError

_MAX_NODES_RETURNED = 200
_MAX_EDGES_RETURNED = 500


# ---- Tool implementations --------------------------------------------------
# Standalone async functions so unit tests can call them directly without
# pulling in the mcp SDK. FastMCP wraps these via build_server().


async def _tool_impl_extract_document(
    api: GraphoraClient,
    file_path: Optional[str],
    url: Optional[str],
    ontology_id: Optional[str],
    schemaless: bool = False,
) -> Dict[str, Any]:
    if bool(file_path) == bool(url):
        raise ValueError(
            "Provide exactly one of file_path or url (not both, not neither)."
        )
    if schemaless and ontology_id:
        raise ValueError("schemaless=True and ontology_id are mutually exclusive.")
    if file_path:
        return await api.upload_file(
            file_path, ontology_id=ontology_id, schemaless=schemaless
        )
    assert url is not None
    return await api.upload_url(url, ontology_id=ontology_id, schemaless=schemaless)


async def _tool_impl_refine_ontology(
    api: GraphoraClient,
    transform_id: str,
    save: bool,
) -> Dict[str, Any]:
    if save:
        return await api.finalize_ontology(transform_id)
    return await api.get_inferred_ontology(transform_id)


async def _tool_impl_query_graph(
    api: GraphoraClient,
    transform_id: str,
    filter_type: Optional[str],
    limit: int,
) -> Dict[str, Any]:
    node_cap = min(max(1, limit), _MAX_NODES_RETURNED)
    data = await api.get_graph(transform_id, limit=node_cap)

    nodes: List[Dict[str, Any]] = data.get("nodes", []) or []
    edges: List[Dict[str, Any]] = data.get("edges", []) or []

    if filter_type:
        wanted = filter_type.lower()
        nodes = [n for n in nodes if str(n.get("type", "")).lower() == wanted]
        keep_ids = {n["id"] for n in nodes if "id" in n}
        edges = [
            e
            for e in edges
            if e.get("source") in keep_ids and e.get("target") in keep_ids
        ]

    return {
        "transform_id": transform_id,
        "nodes": [_trim_node(n) for n in nodes[:node_cap]],
        "edges": [_trim_edge(e) for e in edges[:_MAX_EDGES_RETURNED]],
        "total_nodes": data.get("total_nodes", len(nodes)),
        "total_edges": data.get("total_edges", len(edges)),
    }


async def _tool_impl_get_evidence(
    api: GraphoraClient,
    transform_id: str,
    node_id: Optional[str] = None,
    edge_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return source-span evidence + decision log for a node OR an
    edge. Exactly one of ``node_id`` / ``edge_id`` must be set.

    The response carries a ``kind`` discriminator (``"node"`` or
    ``"edge"``) so agents can route on the response shape:
      * ``kind=node``: ``{node, incoming_edges, outgoing_edges,
        evidence, decision_log, alternatives}``
      * ``kind=edge``: ``{edge, source_node, target_node, evidence,
        decision_log, alternatives}``

    Reviewer-flagged on Gate-4-wrap: the prior signature only
    accepted ``node_id`` and the corresponding REST/decision-log
    plumbing was node-only. The Gate 4 exit signal "for every edge
    ``graphora explain <edge>`` returns the exact source text"
    needs both paths; this dispatch closes the gap.
    """
    if bool(node_id) == bool(edge_id):
        raise ValueError(
            "Provide exactly one of node_id or edge_id (not both, not neither)."
        )

    if node_id:
        return await _get_node_evidence(api, transform_id, node_id)
    assert edge_id is not None
    return await _get_edge_evidence(api, transform_id, edge_id)


async def _get_node_evidence(
    api: GraphoraClient,
    transform_id: str,
    node_id: str,
) -> Dict[str, Any]:
    # Paginate through the graph rather than hard-capping at 200 — a
    # valid node_id may live on any page of a large extraction, and
    # "not on the first page" must not look like "does not exist".
    data = await api.find_node(transform_id, node_id)
    if data is None:
        return {
            "kind": "node",
            "node": None,
            "incoming_edges": [],
            "outgoing_edges": [],
            "evidence": {},
            # B0-explain (slice 4): the new fields ALWAYS appear in
            # the response shape so consumers can rely on the schema
            # without conditional access. Empty when the node is
            # unknown.
            "decision_log": [],
            "alternatives": [],
        }

    nodes = data.get("nodes", []) or []
    edges = data.get("edges", []) or []

    node = next((n for n in nodes if n.get("id") == node_id), None)
    if node is None:  # pragma: no cover — find_node contract guarantees node presence
        return {
            "kind": "node",
            "node": None,
            "incoming_edges": [],
            "outgoing_edges": [],
            "evidence": {},
            "decision_log": [],
            "alternatives": [],
        }

    incoming = [_trim_edge(e) for e in edges if e.get("target") == node_id]
    outgoing = [_trim_edge(e) for e in edges if e.get("source") == node_id]

    decisions_payload = await _fetch_decisions_safely(
        api, transform_id, kind="node", target_id=node_id
    )

    return {
        "kind": "node",
        "node": _trim_node(node, full_properties=True),
        "incoming_edges": incoming,
        "outgoing_edges": outgoing,
        "evidence": _extract_evidence_fields(node.get("properties", {})),
        "decision_log": decisions_payload.get("decision_log", []),
        "alternatives": decisions_payload.get("alternatives", []),
    }


async def _get_edge_evidence(
    api: GraphoraClient,
    transform_id: str,
    edge_id: str,
) -> Dict[str, Any]:
    """Edge-shaped get_evidence path. Mirrors node path's pagination
    contract but yields edge-specific shape: the edge itself plus
    summaries of its source and target nodes (so the agent doesn't
    need an extra get_evidence(node_id) roundtrip to render context).

    Edge source-span evidence comes from the edge's own properties
    — A1-prov stamps source_chunk_id / source_text / document_name /
    page_number / extraction_confidence on edges at extraction time
    (mirrors the node-side stamping in graph_transformer.py)."""
    data = await api.find_edge(transform_id, edge_id)
    if data is None:
        return {
            "kind": "edge",
            "edge": None,
            "source_node": None,
            "target_node": None,
            "evidence": {},
            "decision_log": [],
            "alternatives": [],
        }

    nodes = data.get("nodes", []) or []
    edges = data.get("edges", []) or []

    edge = next((e for e in edges if e.get("id") == edge_id), None)
    if edge is None:  # pragma: no cover — find_edge contract guarantees presence
        return {
            "kind": "edge",
            "edge": None,
            "source_node": None,
            "target_node": None,
            "evidence": {},
            "decision_log": [],
            "alternatives": [],
        }

    # Look up source/target node summaries from the same graph slice
    # we got back from find_edge — avoids a second roundtrip.
    source_id = edge.get("source")
    target_id = edge.get("target")
    source_node = next((_trim_node(n) for n in nodes if n.get("id") == source_id), None)
    target_node = next((_trim_node(n) for n in nodes if n.get("id") == target_id), None)

    decisions_payload = await _fetch_decisions_safely(
        api, transform_id, kind="edge", target_id=edge_id
    )

    return {
        "kind": "edge",
        "edge": _trim_edge(edge, full_properties=True),
        "source_node": source_node,
        "target_node": target_node,
        "evidence": _extract_evidence_fields(edge.get("properties", {})),
        "decision_log": decisions_payload.get("decision_log", []),
        "alternatives": decisions_payload.get("alternatives", []),
    }


async def _fetch_decisions_safely(
    api: GraphoraClient,
    transform_id: str,
    *,
    kind: str,
    target_id: str,
) -> Dict[str, Any]:
    """Wrap the get_decisions HTTP call with the same fail-open
    pattern get_evidence has used since commit eb22a79 (P3):
    decisions are observability; transient API errors must not
    blank the rest of the evidence response. Catching both
    GraphoraClientError (post-parse) and httpx.HTTPError
    (transport-level) keeps "decision failures never blank source
    evidence" honest.

    Shared between node and edge paths so both inherit the same
    fail-open semantics — and so a future refactor that tightens
    the error handling has a single call site to update."""
    try:
        if kind == "node":
            return await api.get_decisions(transform_id, node_id=target_id)
        return await api.get_decisions(transform_id, edge_id=target_id)
    except (GraphoraClientError, httpx.HTTPError) as exc:
        logger = __import__("logging").getLogger(__name__)
        logger.warning(
            "get_decisions failed for transform=%s %s=%s: %s",
            transform_id,
            kind,
            target_id,
            exc,
        )
        return {"decision_log": [], "alternatives": []}


async def _tool_impl_get_cost_report(
    api: GraphoraClient,
    transform_id: str,
) -> Dict[str, Any]:
    """B5-obs: agent-facing per-transform cost surface.

    Pure HTTP-client passthrough — same architectural shape as
    get_evidence's decisions read (commit eb22a79). The API
    endpoint owns the DB read; MCP forwards the payload.

    Empty/zero state returns the shape the endpoint always emits,
    not a different "not found" structure — callers can render
    "0 calls / no cost recorded" without conditional access.
    Transport errors propagate so the agent sees the real failure
    (this isn't an observability-of-observability path; cost is
    the headline answer the agent asked for)."""
    return await api.get_cost_report(transform_id)


async def _tool_impl_get_budget_status(
    api: GraphoraClient,
) -> Dict[str, Any]:
    """B5-obs slice 2: agent-facing budget status surface.

    Lets the agent pre-flight check before submitting an
    expensive transform. Pure passthrough — same architecture as
    get_cost_report. Transport errors propagate."""
    return await api.get_budget_status()


async def _tool_impl_review_diff(
    api: GraphoraClient,
    base_transform_id: str,
    compare_transform_id: str,
) -> Dict[str, Any]:
    """B3-diff: agent-facing graph-state diff between two transforms.

    Pure passthrough — same architecture as get_cost_report and
    get_budget_status. The API endpoint owns the read + diff
    computation; MCP forwards the payload. Transport errors
    propagate because diff is a primary answer the agent asked
    for, not observability-of-observability."""
    return await api.diff_transforms(base_transform_id, compare_transform_id)


async def _tool_impl_list_contradictions(
    api: GraphoraClient,
    transform_id: str,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """B1-prob slice 2a: agent-facing contradictions surface.

    Pure HTTP passthrough — same architecture as the
    review_diff tool. The API endpoint owns the read +
    contradiction detection; MCP forwards the payload.
    Transport errors propagate because contradictions are a
    primary answer the agent asked for, not observability."""
    return await api.list_contradictions(transform_id, min_confidence=min_confidence)


async def _tool_impl_list_disputed_pairs(
    api: GraphoraClient,
    transform_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """B2-active: list the user's pending disputed-pairs queue.

    Pure HTTP passthrough — same shape as the other agent-facing
    tools. Errors propagate because the queue is a primary
    answer (the agent needs to know WHAT to label before it can
    label anything)."""
    return await api.list_disputed_pairs(
        transform_id=transform_id,
        limit=limit,
        offset=offset,
    )


async def _tool_impl_label_disputed_pair(
    api: GraphoraClient,
    pair_id: str,
    decision: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """B2-active: apply a label to a disputed pair. Decision is
    one of match / not_match / skip. Errors propagate so the
    agent sees real failures (404 for missing pair, 400 for
    invalid decision)."""
    return await api.label_disputed_pair(
        pair_id=pair_id, decision=decision, reason=reason
    )


# ---- FastMCP wiring --------------------------------------------------------


def build_server(client: Optional[GraphoraClient] = None):
    """Construct the FastMCP app with tools bound to ``client``.

    Imports FastMCP lazily so ``graphora-server[mcp]`` is only
    required when actually running the server; tests that exercise
    the tool implementations via ``_tool_impl_*`` don't need it.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover — exercised without [mcp]
        raise ImportError(
            "MCP server requires the [mcp] extra. "
            "Install with: pip install 'graphora-server[mcp]'"
        ) from exc

    mcp = FastMCP("graphora")
    api = client if client is not None else GraphoraClient()

    @mcp.tool()
    async def extract_document(
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        ontology_id: Optional[str] = None,
        schemaless: bool = False,
    ) -> Dict[str, Any]:
        """Extract a knowledge graph from a document or URL.

        Exactly one of ``file_path`` or ``url`` must be provided.

        Args:
            file_path: Absolute path to a local file (.pdf, .docx, .xlsx,
                .pptx, .txt, .md, .html, .csv, .json, .xml).
            url: Web URL whose main article text will be extracted.
            ontology_id: Optional existing ontology to extract against.
                When omitted, Graphora auto-infers a schema from the
                document — use this for zero-config "just extract
                something" flows.
            schemaless: If True, skip pre-extraction schema inference
                entirely. The extractor uses a permissive generic
                ontology so specific types emerge from what was
                actually extracted. Call refine_ontology after the
                transform completes to get a tight refined ontology.
                Mutually exclusive with ontology_id.

        Returns a dict with:
            transform_id (str): Use with query_graph / get_evidence /
                refine_ontology.
            status (str): Initial pipeline status (usually ``pending``).
            document_count (int): Files accepted into the pipeline.
        """
        return await _tool_impl_extract_document(
            api, file_path, url, ontology_id, schemaless=schemaless
        )

    @mcp.tool()
    async def refine_ontology(
        transform_id: str,
        save: bool = False,
    ) -> Dict[str, Any]:
        """Infer a refined ontology from an already-extracted graph.

        Runs post-hoc inference over the nodes and edges produced by
        a completed extraction — useful for turning a schemaless
        extraction into a clean ontology, or for discovering how a
        user-supplied ontology could be tightened based on what
        actually got extracted.

        Args:
            transform_id: A completed transform ID.
            save: If True, persist the refined ontology so it can be
                referenced by id and re-used for future extractions.
                If False (default), just return the YAML inline —
                the caller decides whether to save it separately.

        Returns a dict with:
            transform_id (str)
            ontology_yaml (str): Canonical YAML rendering.
            ontology (dict): Parsed ontology (entities + relationships).
            ontology_id (str, only when save=True): The stored ontology id.
            stats (dict): node_count / edge_count / entity_types /
                relationship_types.
        """
        return await _tool_impl_refine_ontology(api, transform_id, save)

    @mcp.tool()
    async def query_graph(
        transform_id: str,
        filter_type: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Fetch nodes and edges from a completed extraction.

        Args:
            transform_id: The ID returned by extract_document.
            filter_type: If set, only nodes whose ``type`` equals
                this value (case-insensitive) are returned. Edges
                are pruned to those whose endpoints both pass the
                filter. Example: ``"Person"`` or ``"Organization"``.
            limit: Max nodes to return (hard-capped at 200 to keep
                agent context usable).

        Returns a dict with:
            transform_id (str)
            nodes (list): ``{id, label, type, summary}`` — trimmed.
            edges (list): ``{id, source, target, type}``.
            total_nodes (int): Full count in the graph (not the slice).
            total_edges (int): Full count in the graph.
        """
        return await _tool_impl_query_graph(api, transform_id, filter_type, limit)

    @mcp.tool()
    async def get_evidence(
        transform_id: str,
        node_id: Optional[str] = None,
        edge_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the source-document context that supports a node OR
        an edge.

        Surfaces provenance fields stored on the target (source chunk
        text, document name, offsets) plus the Decision Log history —
        useful for an agent that wants to explain *why* an entity or
        relationship is in the graph before citing it.

        Exactly one of ``node_id`` or ``edge_id`` must be provided.
        The response carries a ``kind`` discriminator (``"node"`` or
        ``"edge"``) so callers route on the shape.

        Args:
            transform_id: The extraction that produced the target.
            node_id: ID of the node to inspect. Mutually exclusive
                with edge_id.
            edge_id: ID of the edge (relationship) to inspect.
                Mutually exclusive with node_id.

        Returns a dict whose shape depends on ``kind``:

            kind=node:
                node (dict | None): Full node with properties, or
                    None if unknown.
                incoming_edges (list): Edges where this node is target.
                outgoing_edges (list): Edges where this node is source.
                evidence (dict): Provenance-related properties pulled
                    out of the node (source_chunk_id, source_text,
                    document_name, page_number, etc.).
                decision_log (list): Schema-level decisions for the
                    transform plus per-node decisions (entity merges,
                    LLM disambiguations).
                alternatives (list): Aggregated candidate entities
                    the pipeline considered across all merge events.

            kind=edge:
                edge (dict | None): Full edge with properties.
                source_node (dict | None): Summary of the source node.
                target_node (dict | None): Summary of the target node.
                evidence (dict): Source-span fields stamped on the
                    edge at extraction time.
                decision_log (list): Schema-level decisions plus
                    relationship-accepted/rejected decisions for this
                    edge.
                alternatives (list): Candidate relationships the
                    pipeline considered before settling on this one.
        """
        return await _tool_impl_get_evidence(
            api, transform_id, node_id=node_id, edge_id=edge_id
        )

    @mcp.tool()
    async def get_cost_report(transform_id: str) -> Dict[str, Any]:
        """Per-transform LLM cost / token report.

        Aggregates every LLM call the pipeline made for this
        transform: total invocations, input/output/total tokens,
        estimated cost in USD, distinct provider:model pairs, and
        a per-operation-type breakdown so the agent can answer
        "how much did this extraction cost, and where did the
        budget go?"

        Args:
            transform_id: The extraction whose cost to report.

        Returns a dict with:
            transform_id (str): Echo of the input.
            total_calls (int): Count of LLM invocations.
            input_tokens / output_tokens / total_tokens (int): Sums.
            estimated_cost_usd (str | None): Sum of estimated costs
                in USD (string for Decimal precision). None when no
                row had pricing — distinguishes "cost is zero" from
                "cost is unknown".
            models_used (list[str]): Distinct ``"<provider>:<model>"``.
            by_operation_type (dict[str, dict]): Per-op breakdown,
                same shape as the top-level totals.
        """
        return await _tool_impl_get_cost_report(api, transform_id)

    @mcp.tool()
    async def get_budget_status() -> Dict[str, Any]:
        """Current monthly budget status for the authenticated user.

        Tells the agent whether the next transform will succeed:
          * ``state="under"`` — well under cap, safe to proceed.
          * ``state="near"`` — within 20% of cap; warn the user
            before queueing more work.
          * ``state="over"`` — the next transform-upload will be
            rejected with 402.
          * ``state="unset"`` — no cap configured; spend untracked
            against any limit.

        Returns:
            state (str): one of under / near / over / unset.
            current_spend_usd (str): cumulative LLM spend this
                period (Decimal-precision string).
            cap_usd (str | None): the cap, or null when unset.
            period_start (str): ISO8601 UTC, start of the period.
            period_end (str): ISO8601 UTC, exclusive end of the
                period.
        """
        return await _tool_impl_get_budget_status(api)

    @mcp.tool()
    async def review_diff(
        base_transform_id: str,
        compare_transform_id: str,
    ) -> Dict[str, Any]:
        """Structured graph-state diff between two transforms.

        Answers "what changed between extraction A and extraction
        B?" — useful when an agent re-runs a transform after
        prompt tweaks or ontology changes and wants to inspect
        the delta before accepting / publishing.

        Node identity matches across transforms by canonical_id
        (Gate 4 entity resolution gives this), falling back to
        type:canonical_key. Edge identity uses the same rule on
        each endpoint. Property comparison filters system fields
        (source_chunk_id, validator_score, etc.) so re-extraction
        churn doesn't drown the actual signal.

        Args:
            base_transform_id: The "before" transform.
            compare_transform_id: The "after" transform.

        Returns a dict with:
            base_transform_id / compare_transform_id (str): Echo.
            summary (dict): ``{nodes: {added, removed, changed,
                unchanged}, edges: {...}}`` — the rendering layer
                can show counts without walking the full payload.
            added_nodes / removed_nodes (list): Full node objects
                only present on one side.
            changed_nodes (list): Nodes in both with property
                changes. Each: ``{canonical_id, type, base_id,
                compare_id, property_changes: {<key>: {base,
                compare}}}``.
            added_edges / removed_edges / changed_edges (list):
                Same shape, edge-level.
        """
        return await _tool_impl_review_diff(
            api, base_transform_id, compare_transform_id
        )

    @mcp.tool()
    async def list_contradictions(
        transform_id: str,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """Surface (target, property) pairs where the pipeline
        emitted multiple distinct claimed values.

        Answers "what did the extractor disagree with itself
        about?" — useful for agent-facing review flows that
        want to highlight low-trust regions of a graph. Each
        contradiction carries its competing_claims sorted by
        confidence DESC (the winning value is first; the rest
        are alternatives) and a severity count (distinct-value
        count above the confidence floor).

        Args:
            transform_id: The transform to scan.
            min_confidence: Confidence floor in [0.0, 1.0]. 0.0
                returns every contradiction; raise to filter
                low-confidence noise.

        Returns a dict with:
            transform_id (str): Echo.
            min_confidence (float): The applied floor.
            contradictions (list): Per-(target, property) group
                with ``{target_id, target_kind, property_key,
                competing_claims, severity}``.
            total_claims_scanned (int): Reserved — zero until
                B1-prob slice 2b's pipeline hooks emit claims
                at extraction time.
        """
        return await _tool_impl_list_contradictions(
            api, transform_id, min_confidence=min_confidence
        )

    @mcp.tool()
    async def list_disputed_pairs(
        transform_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Pending entity-resolution disputes for the authenticated
        user, newest first.

        Each pair represents two entities the pipeline wasn't
        confident enough to auto-merge. The agent reviews them
        and labels each as ``match`` (they're the same entity),
        ``not_match`` (they're different), or ``skip`` (defer for
        later). Labels feed back into the entity-resolution
        learner.

        Args:
            transform_id: Optional filter — restrict to a single
                run's pending pairs.
            limit: Page size (default 50, max 200).
            offset: Pagination offset.

        Returns a list of dicts, each with:
            id (str): Pair id, used for the label endpoint.
            transform_id (str): Run that surfaced the pair.
            node_a_id / node_b_id (str): The two candidate nodes.
            entity_type (str): Shared type (Person, Company, etc.)
            node_a_canonical_key / node_b_canonical_key (str|None):
                Canonical keys when ER had them.
            similarity_score (str|None): Stage's confidence
                (Decimal-precision string).
            source_stage (str): Which ER stage flagged this —
                ``property_blocker`` / ``embedding_blocker`` /
                ``splink_blocker`` / ``llm_review``.
            status (str): Always ``pending`` from this endpoint.
            created_at (str): ISO8601.
        """
        return await _tool_impl_list_disputed_pairs(
            api, transform_id=transform_id, limit=limit, offset=offset
        )

    @mcp.tool()
    async def label_disputed_pair(
        pair_id: str,
        decision: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Label a disputed entity-resolution pair.

        Args:
            pair_id: From a prior ``list_disputed_pairs`` result.
            decision: One of ``match`` (the two entities ARE the
                same), ``not_match`` (they're different), or
                ``skip`` (defer this one).
            reason: Optional free-text rationale the operator can
                see in the audit trail.

        Returns the updated pair with:
            status: One of ``labeled_match`` / ``labeled_not_match``
                / ``skipped``.
            labeled_at (str): ISO8601 timestamp.
            labeled_by_user_id (str): Who labeled it.
            label_reason (str|None): Echoed reason.
        """
        return await _tool_impl_label_disputed_pair(
            api, pair_id=pair_id, decision=decision, reason=reason
        )

    return mcp


# ---- Helpers ---------------------------------------------------------------

# Fields commonly populated by the extraction pipeline that
# represent "where this came from". Kept narrow so the evidence
# payload doesn't double as a property dump.
_EVIDENCE_KEYS = {
    "source_chunk",
    "source_chunk_id",
    "source_text",
    "source",
    "document_id",
    "document_name",
    "chunk_offset",
    "page_number",
    "extraction_confidence",
    # B0-prov-extend (Gate 4 entry). The decision-trail fields the
    # Explorer Evidence tab + this tool both consume. Frontend
    # mirror lives in app/src/app/(app)/explorer/[transform_id]/
    # evidence-tab.tsx::EVIDENCE_KEYS.
    "extractor_model",
    "prompt_version",
    "validator_score",
}


def _extract_evidence_fields(props: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in props.items() if k in _EVIDENCE_KEYS}


def _trim_node(
    node: Dict[str, Any], *, full_properties: bool = False
) -> Dict[str, Any]:
    out = {
        "id": node.get("id"),
        "label": node.get("label"),
        "type": node.get("type"),
    }
    props = node.get("properties", {}) or {}
    if full_properties:
        out["properties"] = props
    else:
        summary = props.get("name") or props.get("title")
        if summary:
            out["summary"] = str(summary)[:200]
    return out


def _trim_edge(
    edge: Dict[str, Any], *, full_properties: bool = False
) -> Dict[str, Any]:
    """Edge summary for query_graph results (default) or full edge
    payload for get_evidence's edge path (full_properties=True).

    full_properties=True surfaces the property bag so the source-
    span fields (source_chunk_id, source_text, page_number,
    document_name, extraction_confidence, plus the B0-prov-extend
    fields) flow through. Mirrors _trim_node's contract."""
    out: Dict[str, Any] = {
        "id": edge.get("id"),
        "source": edge.get("source"),
        "target": edge.get("target"),
        "type": edge.get("type"),
    }
    if full_properties:
        out["properties"] = edge.get("properties", {}) or {}
    return out


__all__ = [
    "build_server",
    "GraphoraClient",
    "GraphoraClientError",
    "_tool_impl_extract_document",
    "_tool_impl_query_graph",
    "_tool_impl_get_evidence",
    "_tool_impl_get_cost_report",
    "_tool_impl_get_budget_status",
    "_tool_impl_review_diff",
    "_tool_impl_list_disputed_pairs",
    "_tool_impl_label_disputed_pair",
    "_tool_impl_refine_ontology",
]
