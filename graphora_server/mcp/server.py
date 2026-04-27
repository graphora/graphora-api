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
    node_id: str,
) -> Dict[str, Any]:
    # Paginate through the graph rather than hard-capping at 200 — a
    # valid node_id may live on any page of a large extraction, and
    # "not on the first page" must not look like "does not exist".
    data = await api.find_node(transform_id, node_id)
    if data is None:
        return {
            "node": None,
            "incoming_edges": [],
            "outgoing_edges": [],
            "evidence": {},
        }

    nodes = data.get("nodes", []) or []
    edges = data.get("edges", []) or []

    node = next((n for n in nodes if n.get("id") == node_id), None)
    if node is None:  # pragma: no cover — find_node contract guarantees node presence
        return {
            "node": None,
            "incoming_edges": [],
            "outgoing_edges": [],
            "evidence": {},
        }

    incoming = [_trim_edge(e) for e in edges if e.get("target") == node_id]
    outgoing = [_trim_edge(e) for e in edges if e.get("source") == node_id]

    return {
        "node": _trim_node(node, full_properties=True),
        "incoming_edges": incoming,
        "outgoing_edges": outgoing,
        "evidence": _extract_evidence_fields(node.get("properties", {})),
    }


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
    async def get_evidence(transform_id: str, node_id: str) -> Dict[str, Any]:
        """Return the source-document context that supports a node.

        Surfaces provenance fields stored on the node (source chunk
        text, document name, offsets) plus any relationships the
        node participates in — useful for an agent that wants to
        explain *why* an entity is in the graph before citing it.

        Args:
            transform_id: The extraction that produced the node.
            node_id: The ID of the node to inspect.

        Returns a dict with:
            node (dict | None): Full node with properties, or None
                if the ID is unknown at this transform_id.
            incoming_edges (list): Edges where this node is the target.
            outgoing_edges (list): Edges where this node is the source.
            evidence (dict): Provenance-related properties pulled out of
                the node (e.g. source_chunk, source_text, document_id).
        """
        return await _tool_impl_get_evidence(api, transform_id, node_id)

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


def _trim_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": edge.get("id"),
        "source": edge.get("source"),
        "target": edge.get("target"),
        "type": edge.get("type"),
    }


__all__ = [
    "build_server",
    "GraphoraClient",
    "GraphoraClientError",
    "_tool_impl_extract_document",
    "_tool_impl_query_graph",
    "_tool_impl_get_evidence",
    "_tool_impl_refine_ontology",
]
