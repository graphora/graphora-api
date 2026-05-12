"""B3-diff backend: graph-state comparison between two transforms.

The brief calls out three surfaces that consume this:
  * ``graphora diff`` CLI in graphora-client
  * Evidence Explorer Diff/Review tab in graphora-fe
  * MCP ``review_diff`` tool for agents

All three are pure consumers of the structured payload this service
produces. The agent / CLI / UI rendering layers shape that payload
into their respective surfaces; this service just makes the
comparison.

Identity model (the core design choice — see commit message):
  * Nodes match across transforms by ``canonical_id`` (Gate 4
    entity resolution makes this stable for the same user). When
    ``canonical_id`` is absent, fall back to a synthetic
    ``type:canonical_key`` composite. Nodes with neither signal
    are treated as transform-local — they show up as added/
    removed in every diff, which is the honest answer when ER
    hasn't given us a cross-transform identity.
  * Edges match by ``(source_key, target_key, type)`` using the
    same identity rule on each endpoint.

Property comparison filters SYSTEM_PROPERTIES (the same set
storage/quality use) so source-span / decision-trail / Neo4j
metadata churn doesn't surface as user-visible "changes". Without
that filter, every re-extraction would diff against itself.

Out of scope (intentional):
  * Cross-user diffs (RBAC + admin endpoints land in Gate 6).
  * Streaming for very large diffs (a 100k-node transform diff
    would need pagination; the brief targets the common case of
    re-extractions where deltas are modest).
  * Confidence-shift summary (B0 decision-trail data feeds that;
    revisit when the Decision Log gains relationship-level
    rows).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from graphora_server.schemas.graph import Edge, GraphResponse, Node
from graphora_server.utils.constants import SYSTEM_PROPERTIES

logger = logging.getLogger(__name__)


# Identity-key types --------------------------------------------------------

NodeKey = str  # canonical_id or "type:canonical_key" fallback
EdgeKey = Tuple[str, str, str]  # (source_key, target_key, type)


# Payload dataclasses -------------------------------------------------------


@dataclass
class PropertyChange:
    """Per-property delta within a changed node/edge."""

    base: Any
    compare: Any


@dataclass
class NodeDelta:
    """A node that exists in both transforms but with different
    user-meaningful properties.

    ``base_id`` / ``compare_id`` are the transform-scoped node IDs
    so the rendering layer can deep-link to either side. They
    differ when ER produced a fresh ID on the new transform."""

    canonical_id: Optional[str]
    type: str
    base_id: str
    compare_id: str
    property_changes: Dict[str, PropertyChange] = field(default_factory=dict)


@dataclass
class EdgeDelta:
    """An edge that exists in both transforms with different
    user-meaningful properties."""

    source_key: NodeKey
    target_key: NodeKey
    type: str
    base_id: str
    compare_id: str
    property_changes: Dict[str, PropertyChange] = field(default_factory=dict)


@dataclass
class DiffSummary:
    nodes_added: int = 0
    nodes_removed: int = 0
    nodes_changed: int = 0
    nodes_unchanged: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    edges_changed: int = 0
    edges_unchanged: int = 0


@dataclass
class GraphDiff:
    base_transform_id: str
    compare_transform_id: str
    summary: DiffSummary
    added_nodes: List[Node] = field(default_factory=list)
    removed_nodes: List[Node] = field(default_factory=list)
    changed_nodes: List[NodeDelta] = field(default_factory=list)
    added_edges: List[Edge] = field(default_factory=list)
    removed_edges: List[Edge] = field(default_factory=list)
    changed_edges: List[EdgeDelta] = field(default_factory=list)


# Public API ----------------------------------------------------------------


class DiffService:
    """Compares two transform graphs and returns a structured diff.

    The service is intentionally stateless — each diff call fetches
    both graphs fresh and produces the comparison in memory.
    Caching is the API layer's job (different consumers have
    different freshness requirements)."""

    def diff(
        self,
        base_graph: GraphResponse,
        compare_graph: GraphResponse,
        base_transform_id: str,
        compare_transform_id: str,
    ) -> GraphDiff:
        """Produce the structured diff.

        Callers supply the two graph responses directly rather than
        the service fetching them — that keeps the service
        independent of the user-DB / in-memory backend split and
        makes it trivially unit-testable.

        Diff is symmetric in the sense that ``diff(a, b)`` and
        ``diff(b, a)`` produce mirror-image payloads (added↔removed)
        — but the API surface always names ``base`` first to give
        the rendering layer a stable "what changed FROM base TO
        compare" narrative."""

        base_nodes = {_node_key(n): n for n in base_graph.nodes}
        compare_nodes = {_node_key(n): n for n in compare_graph.nodes}

        base_keys = set(base_nodes.keys())
        compare_keys = set(compare_nodes.keys())

        added_node_keys = compare_keys - base_keys
        removed_node_keys = base_keys - compare_keys
        common_node_keys = base_keys & compare_keys

        added_nodes = [compare_nodes[k] for k in added_node_keys]
        removed_nodes = [base_nodes[k] for k in removed_node_keys]

        changed_nodes: List[NodeDelta] = []
        unchanged_count = 0
        for key in common_node_keys:
            b = base_nodes[key]
            c = compare_nodes[key]
            prop_changes = _property_delta(b.properties, c.properties)
            if not prop_changes:
                unchanged_count += 1
                continue
            changed_nodes.append(
                NodeDelta(
                    # Prefer the compare side's canonical_id (more
                    # likely populated after ER on a re-extraction).
                    canonical_id=_canonical_id_or_none(c) or _canonical_id_or_none(b),
                    type=c.type,
                    base_id=b.id,
                    compare_id=c.id,
                    property_changes=prop_changes,
                )
            )

        # ---- Edges --------------------------------------------------------
        #
        # Edge identity depends on its endpoints' identity keys.
        # If an endpoint is missing from a graph (i.e. the source/
        # target node was added or removed), that edge gets
        # attributed to the same side as the missing endpoint
        # naturally — because the edge's key on that side won't
        # exist on the other side either.

        base_edges = {_edge_key(e, base_nodes): e for e in base_graph.edges}
        compare_edges = {_edge_key(e, compare_nodes): e for e in compare_graph.edges}
        # Drop edges whose endpoints we couldn't key (orphaned
        # source/target ids on the same side). Diffing those is
        # ambiguous; logging the count keeps the issue visible
        # without polluting the payload.
        base_edges = {k: v for k, v in base_edges.items() if k is not None}
        compare_edges = {k: v for k, v in compare_edges.items() if k is not None}

        base_edge_keys = set(base_edges.keys())
        compare_edge_keys = set(compare_edges.keys())

        added_edge_keys = compare_edge_keys - base_edge_keys
        removed_edge_keys = base_edge_keys - compare_edge_keys
        common_edge_keys = base_edge_keys & compare_edge_keys

        added_edges = [compare_edges[k] for k in added_edge_keys]
        removed_edges = [base_edges[k] for k in removed_edge_keys]

        changed_edges: List[EdgeDelta] = []
        unchanged_edge_count = 0
        for key in common_edge_keys:
            b = base_edges[key]
            c = compare_edges[key]
            prop_changes = _property_delta(b.properties, c.properties)
            if not prop_changes:
                unchanged_edge_count += 1
                continue
            src_key, tgt_key, edge_type = key
            changed_edges.append(
                EdgeDelta(
                    source_key=src_key,
                    target_key=tgt_key,
                    type=edge_type,
                    base_id=b.id,
                    compare_id=c.id,
                    property_changes=prop_changes,
                )
            )

        summary = DiffSummary(
            nodes_added=len(added_nodes),
            nodes_removed=len(removed_nodes),
            nodes_changed=len(changed_nodes),
            nodes_unchanged=unchanged_count,
            edges_added=len(added_edges),
            edges_removed=len(removed_edges),
            edges_changed=len(changed_edges),
            edges_unchanged=unchanged_edge_count,
        )

        return GraphDiff(
            base_transform_id=base_transform_id,
            compare_transform_id=compare_transform_id,
            summary=summary,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            changed_nodes=changed_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
            changed_edges=changed_edges,
        )


# Identity-key helpers ------------------------------------------------------


def _canonical_id_or_none(node: Node) -> Optional[str]:
    """Pull canonical_id off either the property bag or the
    typed field if Pydantic put it there. Different storage
    backends populate it in different shapes; tolerate both."""
    if hasattr(node, "canonical_id"):
        val = getattr(node, "canonical_id", None)
        if val:
            return str(val)
    val = node.properties.get("canonical_id") if node.properties else None
    return str(val) if val else None


def _canonical_key_or_none(node: Node) -> Optional[str]:
    """Same shape-tolerance for canonical_key. Used as fallback
    identity when canonical_id is absent."""
    if hasattr(node, "canonical_key"):
        val = getattr(node, "canonical_key", None)
        if val:
            return str(val)
    val = node.properties.get("canonical_key") if node.properties else None
    return str(val) if val else None


def _node_key(node: Node) -> NodeKey:
    """Identity key used for cross-transform matching.

    Preference order:
      1. ``canonical_id`` — Gate 4 entity resolution assigns this;
         stable across runs of the same user when ER works.
      2. ``type:canonical_key`` — present at extraction time
         before ER runs. Stable as long as the canonical-key
         derivation rule doesn't drift.
      3. ``__local__:<id>`` — last resort. Guarantees the node
         shows up on ITS OWN side of the diff (because the other
         side won't share the same id), which is the honest
         answer when we can't tell whether it's "the same node"
         across transforms."""
    cid = _canonical_id_or_none(node)
    if cid:
        return cid
    ckey = _canonical_key_or_none(node)
    if ckey:
        return f"{node.type}:{ckey}"
    return f"__local__:{node.id}"


def _edge_key(
    edge: Edge,
    node_lookup: Dict[NodeKey, Node],
) -> Optional[EdgeKey]:
    """Compose a stable cross-transform identity from the edge's
    endpoint keys + type.

    Returns None when either endpoint can't be located in the
    same-side node map — that's an orphaned edge (source/target id
    pointing at a node that was filtered or simply doesn't exist
    in the graph response). The diff loop drops these because we
    can't sensibly match them across transforms; logging the count
    happens at the call site."""
    # Find each endpoint by id in the same-side node map.
    src_node = next((n for n in node_lookup.values() if n.id == edge.source), None)
    tgt_node = next((n for n in node_lookup.values() if n.id == edge.target), None)
    if src_node is None or tgt_node is None:
        logger.debug(
            "Skipping orphaned edge %s (%s -> %s, type=%s) — endpoint not "
            "in node response",
            edge.id,
            edge.source,
            edge.target,
            edge.type,
        )
        return None
    return (_node_key(src_node), _node_key(tgt_node), edge.type)


def _property_delta(
    base: Dict[str, Any], compare: Dict[str, Any]
) -> Dict[str, PropertyChange]:
    """Per-property diff with SYSTEM_PROPERTIES filtered out.

    The filter is load-bearing: without it, every re-extraction
    re-stamps source_chunk_id / validator_score / __valid_from /
    extraction_timestamp and the diff fires "changed" on every
    node. Mirrors the filter the storage layer uses to decide
    whether to version a relationship, so user-visible "changes"
    here match user-visible "versioning" there."""
    base_filtered = _filter_system(base)
    compare_filtered = _filter_system(compare)
    all_keys = set(base_filtered.keys()) | set(compare_filtered.keys())
    deltas: Dict[str, PropertyChange] = {}
    for key in all_keys:
        b_val = base_filtered.get(key)
        c_val = compare_filtered.get(key)
        if b_val != c_val:
            deltas[key] = PropertyChange(base=b_val, compare=c_val)
    return deltas


def _filter_system(props: Dict[str, Any]) -> Dict[str, Any]:
    if not props:
        return {}
    return {k: v for k, v in props.items() if k not in SYSTEM_PROPERTIES}
