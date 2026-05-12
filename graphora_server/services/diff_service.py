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

        # Reviewer-flagged P1 on commit a75cd73: a single-pass
        # _node_key() build broke fallback matching when one side
        # had canonical_id and the other had only canonical_key.
        # Same logical entity → different keys → false
        # added+removed pair. Fixed with staged matching:
        # canonical_id first, then type:canonical_key on the
        # remainder, then local-id fallback.
        #
        # ``base_match_key`` and ``compare_match_key`` map each
        # side's per-transform node id to a STABLE match identity
        # that survives the side-mismatch (e.g. both alices end
        # up as "Person:alice" even when one has canonical_id and
        # the other doesn't). Edge composition reads from these
        # maps so endpoint identity is the matched identity, not
        # the per-side identity.
        (
            matched_pairs,
            unmatched_base,
            unmatched_compare,
            base_match_key,
            compare_match_key,
        ) = _match_nodes(base_graph.nodes, compare_graph.nodes)

        added_nodes = list(unmatched_compare)
        removed_nodes = list(unmatched_base)

        changed_nodes: List[NodeDelta] = []
        unchanged_count = 0
        for b, c, _match_key in matched_pairs:
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
        # Edge identity uses the MATCHED node identity (not the
        # per-side raw key) so an edge whose endpoint matched
        # across the cid/canonical-key fallback still surfaces as
        # the same edge on both sides.

        base_edges_keyed = _key_edges(base_graph.edges, base_match_key)
        compare_edges_keyed = _key_edges(compare_graph.edges, compare_match_key)
        base_edges = base_edges_keyed
        compare_edges = compare_edges_keyed

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


def _match_nodes(
    base_nodes: List[Node],
    compare_nodes: List[Node],
) -> Tuple[
    List[Tuple[Node, Node, str]],
    List[Node],
    List[Node],
    Dict[str, str],
    Dict[str, str],
]:
    """Three-pass staged matching for cross-transform node identity.

    Reviewer-flagged P1 on commit a75cd73. The previous single-pass
    keyer broke when one side had ``canonical_id`` and the other
    only had ``canonical_key`` — the same entity got two different
    keys and surfaced as a false added+removed pair.

    Pass order (strictest signal first):
      1. ``canonical_id`` — Gate 4 entity resolution. Both sides
         must have it for a match.
      2. ``type:canonical_key`` — extraction-time identity, runs
         only on nodes the first pass didn't claim. Crucially:
         this is the pass that bridges "base lacks canonical_id,
         compare has it" cases — both sides still share
         canonical_key, so the match happens here.
      3. ``__local__:<id>`` — last resort, matches only same-id
         pairs that exist on both sides. Rare in production
         (transforms produce different ids per run) but valuable
         in tests and for the rare same-id retry case.

    Returns:
        matched_pairs: list of (base_node, compare_node, match_key)
        unmatched_base / unmatched_compare: nodes with no partner
        base_match_key / compare_match_key: per-side maps from
            node.id to the stable match key. Edge composition uses
            these so endpoint identity is the *matched* identity,
            not the per-side raw key (load-bearing for the same
            cross-pass scenario as the node fix).
    """
    matched: List[Tuple[Node, Node, str]] = []
    consumed_base: set = set()
    consumed_compare: set = set()
    base_match_key: Dict[str, str] = {}
    compare_match_key: Dict[str, str] = {}

    # ---- Pass 1: canonical_id ------------------------------------------
    base_by_cid: Dict[str, Node] = {}
    for b in base_nodes:
        cid = _canonical_id_or_none(b)
        if cid:
            # If multiple base nodes share a canonical_id (a
            # post-ER mistake), the last one wins — diffing
            # such an ambiguous graph is best-effort.
            base_by_cid[cid] = b

    for c in compare_nodes:
        cid = _canonical_id_or_none(c)
        if cid and cid in base_by_cid:
            b = base_by_cid[cid]
            if b.id in consumed_base:
                # Compare has multiple nodes with the same
                # canonical_id; only the first one matches.
                continue
            matched.append((b, c, cid))
            consumed_base.add(b.id)
            consumed_compare.add(c.id)
            base_match_key[b.id] = cid
            compare_match_key[c.id] = cid

    # ---- Pass 2: type:canonical_key (asymmetric ER signal only) --------
    #
    # Reviewer-flagged P1 on commit a261321. Pre-fix this pass
    # matched any pair sharing canonical_key, including cases
    # where BOTH sides had explicit-but-DIFFERENT canonical_ids
    # — letting the weaker canonical_key signal override ER's
    # verdict. When ER assigned different canonical_ids on each
    # side, it explicitly said "these are different entities";
    # canonical_key alone cannot bridge that decision.
    #
    # Constraint: a pass-2 match is valid iff AT LEAST ONE side
    # lacks canonical_id. Both-with-canonical-id pairs that pass
    # 1 already rejected stay unmatched here too, preserving
    # ER's verdict in the diff payload (they surface as
    # removed + added).
    base_by_ckey: Dict[str, Node] = {}
    for b in base_nodes:
        if b.id in consumed_base:
            continue
        ckey = _canonical_key_or_none(b)
        if ckey:
            base_by_ckey[f"{b.type}:{ckey}"] = b

    for c in compare_nodes:
        if c.id in consumed_compare:
            continue
        ckey = _canonical_key_or_none(c)
        if not ckey:
            continue
        composite = f"{c.type}:{ckey}"
        if composite not in base_by_ckey:
            continue
        b = base_by_ckey[composite]
        if b.id in consumed_base:
            continue

        # Asymmetry constraint: skip when both sides have explicit
        # canonical_ids (pass 1 already rejected them).
        if _canonical_id_or_none(b) and _canonical_id_or_none(c):
            continue

        matched.append((b, c, composite))
        consumed_base.add(b.id)
        consumed_compare.add(c.id)
        base_match_key[b.id] = composite
        compare_match_key[c.id] = composite

    # ---- Pass 3: __local__:<id> fallback -------------------------------
    base_by_local_id = {b.id: b for b in base_nodes if b.id not in consumed_base}
    for c in compare_nodes:
        if c.id in consumed_compare:
            continue
        if c.id in base_by_local_id:
            b = base_by_local_id[c.id]
            local_key = f"__local__:{c.id}"
            matched.append((b, c, local_key))
            consumed_base.add(b.id)
            consumed_compare.add(c.id)
            base_match_key[b.id] = local_key
            compare_match_key[c.id] = local_key

    # ---- Unmatched + their individual match keys -----------------------
    unmatched_base = [b for b in base_nodes if b.id not in consumed_base]
    unmatched_compare = [c for c in compare_nodes if c.id not in consumed_compare]

    # Unmatched nodes still need a key for edge composition. Use
    # their best individual key — won't collide with anything on
    # the other side because that side didn't have it (otherwise
    # they would have matched).
    for b in unmatched_base:
        base_match_key[b.id] = _node_key(b)
    for c in unmatched_compare:
        compare_match_key[c.id] = _node_key(c)

    return matched, unmatched_base, unmatched_compare, base_match_key, compare_match_key


def _key_edges(
    edges: List[Edge],
    match_key_by_node_id: Dict[str, str],
) -> Dict[EdgeKey, Edge]:
    """Build a {(source_match_key, target_match_key, type) -> Edge}
    map. Drops orphaned edges (endpoint not in the same-side
    match-key map) — they can't be sensibly matched across
    transforms."""
    keyed: Dict[EdgeKey, Edge] = {}
    for edge in edges:
        src_key = match_key_by_node_id.get(edge.source)
        tgt_key = match_key_by_node_id.get(edge.target)
        if src_key is None or tgt_key is None:
            logger.debug(
                "Skipping orphaned edge %s (%s -> %s, type=%s) — endpoint not "
                "in match-key map",
                edge.id,
                edge.source,
                edge.target,
                edge.type,
            )
            continue
        keyed[(src_key, tgt_key, edge.type)] = edge
    return keyed


def _edge_key(
    edge: Edge,
    node_lookup: Dict[NodeKey, Node],
) -> Optional[EdgeKey]:
    """Single-side edge key. Retained for unit-test callers that
    exercise the per-edge identity rule directly; production
    diffing goes through _match_nodes + _key_edges so endpoint
    identity uses the matched identity across the staged passes."""
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


# Properties the diff service treats as identity / metadata
# signal rather than user-meaningful content. Filtering these
# from _property_delta keeps "ER added a canonical_id" or
# "canonical_key got re-derived" from surfacing as a property
# change — those are identity signals consumed by the staged
# matcher, not user-visible facts about the entity.
#
# Kept LOCAL to the diff service rather than added to the
# global SYSTEM_PROPERTIES list because the global list has
# wide blast radius (similarity scoring, ontology validation,
# storage versioning) and the diff service is the only surface
# that needs canonical_id / canonical_key filtered out of
# property comparison.
_DIFF_EXCLUDED_PROPERTIES = set(SYSTEM_PROPERTIES) | {
    "canonical_id",
    "canonical_key",
    "canonical_properties",
}


def _property_delta(
    base: Dict[str, Any], compare: Dict[str, Any]
) -> Dict[str, PropertyChange]:
    """Per-property diff with system + identity fields filtered out.

    The filter is load-bearing: without it, every re-extraction
    re-stamps source_chunk_id / validator_score / __valid_from /
    extraction_timestamp and the diff fires "changed" on every
    node. Also strips identity signals (canonical_id,
    canonical_key) — those are how the staged matcher pairs
    nodes across sides, not user-meaningful facts about the
    entity. See _DIFF_EXCLUDED_PROPERTIES."""
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
    return {k: v for k, v in props.items() if k not in _DIFF_EXCLUDED_PROPERTIES}
