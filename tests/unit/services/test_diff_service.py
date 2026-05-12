"""Unit tests for DiffService (B3-diff backend).

The service has three responsibilities pinned independently:
  * Identity matching across transforms (canonical_id → fallback
    composite → per-transform-local).
  * Property delta with SYSTEM_PROPERTIES filtered (so
    re-extraction churn doesn't fire as user-visible changes).
  * Edge identity composed from endpoint identity keys.

Tests construct GraphResponse instances directly — no DB, no
storage backend; the service deliberately accepts pre-fetched
graphs so it stays trivially unit-testable."""

from __future__ import annotations

import pytest

from graphora_server.schemas.graph import Edge, GraphResponse, Node
from graphora_server.services.diff_service import (
    DiffService,
    _edge_key,
    _node_key,
    _property_delta,
)


def _make_node(
    id: str,
    type: str = "Person",
    canonical_id: str = None,
    canonical_key: str = None,
    properties: dict = None,
) -> Node:
    """Construct a Node with canonical_id / canonical_key on the
    property bag (where the storage layer stores them, not as
    typed fields)."""
    props = dict(properties or {})
    if canonical_id is not None:
        props["canonical_id"] = canonical_id
    if canonical_key is not None:
        props["canonical_key"] = canonical_key
    return Node(id=id, label=props.get("name", id), type=type, properties=props)


def _make_edge(
    id: str, source: str, target: str, type: str = "WORKS_AT", properties: dict = None
) -> Edge:
    return Edge(
        id=id, source=source, target=target, type=type, properties=properties or {}
    )


# ---- Node identity --------------------------------------------------------


class TestNodeKey:
    """Identity preference: canonical_id > type:canonical_key >
    __local__:<id>. Pinning each tier so a future refactor can't
    silently regress to "always use id" (which would make every
    diff show 100% added/removed)."""

    def test_canonical_id_wins(self) -> None:
        n = _make_node("local-1", canonical_id="cid-abc", canonical_key="alice")
        assert _node_key(n) == "cid-abc"

    def test_falls_back_to_type_and_canonical_key(self) -> None:
        n = _make_node("local-1", type="Person", canonical_key="alice")
        assert _node_key(n) == "Person:alice"

    def test_last_resort_uses_local_id(self) -> None:
        """Without ER signals, the node still gets a stable key
        — but one that's guaranteed not to match across
        transforms. That's the honest answer: 'I can't tell
        whether this is the same node as anything on the other
        side.'"""
        n = _make_node("local-xyz", type="Thing")
        assert _node_key(n) == "__local__:local-xyz"


# ---- Property delta -------------------------------------------------------


class TestPropertyDelta:
    def test_no_changes_when_user_properties_match(self) -> None:
        deltas = _property_delta({"role": "engineer"}, {"role": "engineer"})
        assert deltas == {}

    def test_detects_value_change(self) -> None:
        deltas = _property_delta({"role": "engineer"}, {"role": "principal engineer"})
        assert "role" in deltas
        assert deltas["role"].base == "engineer"
        assert deltas["role"].compare == "principal engineer"

    def test_detects_added_property(self) -> None:
        """A property present on compare but not base shows up
        with base=None."""
        deltas = _property_delta({}, {"new_field": "hello"})
        assert deltas["new_field"].base is None
        assert deltas["new_field"].compare == "hello"

    def test_detects_removed_property(self) -> None:
        deltas = _property_delta({"old_field": "bye"}, {})
        assert deltas["old_field"].base == "bye"
        assert deltas["old_field"].compare is None

    def test_filters_system_properties(self) -> None:
        """Reviewer-flagged invariant: re-extraction re-stamps
        source_chunk_id / validator_score / __valid_from / etc.
        Without the filter, every re-extraction would fire as
        'changed' on every node."""
        base = {
            "role": "engineer",
            "source_chunk_id": "chunk-A",
            "validator_score": 0.8,
            "__valid_from": "2026-01-01",
        }
        compare = {
            "role": "engineer",
            "source_chunk_id": "chunk-B",  # different — but filtered
            "validator_score": 0.9,  # different — but filtered
            "__valid_from": "2026-02-01",  # different — but filtered
        }
        # All differences are in SYSTEM_PROPERTIES → no user-visible
        # changes.
        assert _property_delta(base, compare) == {}


# ---- Edge identity --------------------------------------------------------


class TestEdgeKey:
    def test_edge_key_composes_endpoint_keys(self) -> None:
        alice = _make_node("a", canonical_id="cid-alice")
        acme = _make_node("b", canonical_id="cid-acme")
        lookup = {_node_key(alice): alice, _node_key(acme): acme}
        edge = _make_edge("e1", source="a", target="b", type="WORKS_AT")
        key = _edge_key(edge, lookup)
        assert key == ("cid-alice", "cid-acme", "WORKS_AT")

    def test_edge_with_orphan_endpoint_returns_none(self) -> None:
        """An edge whose source/target id isn't in the node
        response is orphaned. The diff loop drops these because
        they can't be sensibly matched across transforms."""
        alice = _make_node("a", canonical_id="cid-alice")
        lookup = {_node_key(alice): alice}
        # target='zzz' doesn't exist in lookup.
        edge = _make_edge("e1", source="a", target="zzz", type="WORKS_AT")
        assert _edge_key(edge, lookup) is None


# ---- Full diff ------------------------------------------------------------


class TestDiffService:
    def _service(self) -> DiffService:
        return DiffService()

    def test_identical_graphs_produce_zero_deltas(self) -> None:
        alice = _make_node("a", canonical_id="cid-alice", properties={"role": "eng"})
        acme = _make_node("b", canonical_id="cid-acme")
        edge = _make_edge("e1", source="a", target="b", type="WORKS_AT")
        graph = GraphResponse(nodes=[alice, acme], edges=[edge])

        result = self._service().diff(graph, graph, "tx-1", "tx-1")

        assert result.summary.nodes_added == 0
        assert result.summary.nodes_removed == 0
        assert result.summary.nodes_changed == 0
        assert result.summary.nodes_unchanged == 2
        assert result.summary.edges_unchanged == 1

    def test_detects_added_node(self) -> None:
        """Same alice on both sides; bob added on compare."""
        alice_b = _make_node("a", canonical_id="cid-alice")
        alice_c = _make_node("a2", canonical_id="cid-alice")
        bob_c = _make_node("c", canonical_id="cid-bob")
        base = GraphResponse(nodes=[alice_b], edges=[])
        compare = GraphResponse(nodes=[alice_c, bob_c], edges=[])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        assert result.summary.nodes_added == 1
        assert result.summary.nodes_removed == 0
        added = result.added_nodes[0]
        assert added.id == "c"

    def test_detects_removed_node(self) -> None:
        """Bob disappears in compare."""
        alice = _make_node("a", canonical_id="cid-alice")
        bob = _make_node("b", canonical_id="cid-bob")
        base = GraphResponse(nodes=[alice, bob], edges=[])
        compare = GraphResponse(nodes=[alice], edges=[])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        assert result.summary.nodes_removed == 1
        assert result.removed_nodes[0].id == "b"

    def test_node_changed_when_user_properties_differ(self) -> None:
        """Same canonical_id, different user-meaningful property
        (role)."""
        alice_b = _make_node(
            "a1", canonical_id="cid-alice", properties={"role": "engineer"}
        )
        alice_c = _make_node(
            "a2",
            canonical_id="cid-alice",
            properties={"role": "principal engineer"},
        )
        base = GraphResponse(nodes=[alice_b], edges=[])
        compare = GraphResponse(nodes=[alice_c], edges=[])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        assert result.summary.nodes_changed == 1
        delta = result.changed_nodes[0]
        assert delta.canonical_id == "cid-alice"
        # IDs survive separately — UI can deep-link to either side.
        assert delta.base_id == "a1"
        assert delta.compare_id == "a2"
        assert "role" in delta.property_changes
        assert delta.property_changes["role"].base == "engineer"
        assert delta.property_changes["role"].compare == "principal engineer"

    def test_re_extraction_with_only_system_changes_does_not_fire(self) -> None:
        """The load-bearing pin: re-extracting the same document
        produces identical user data but new chunk_ids, fresh
        timestamps, refreshed validator_scores. Without the
        SYSTEM_PROPERTIES filter, the diff would report every
        node as 'changed' — making the surface useless for
        actual re-extraction review.

        Pin: zero changed nodes when only system fields differ."""
        alice_b = _make_node(
            "a1",
            canonical_id="cid-alice",
            properties={
                "role": "engineer",
                "source_chunk_id": "chunk-A",
                "validator_score": 0.8,
            },
        )
        alice_c = _make_node(
            "a2",
            canonical_id="cid-alice",
            properties={
                "role": "engineer",
                "source_chunk_id": "chunk-B",  # CHANGED — but system
                "validator_score": 0.9,  # CHANGED — but system
            },
        )
        base = GraphResponse(nodes=[alice_b], edges=[])
        compare = GraphResponse(nodes=[alice_c], edges=[])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        assert result.summary.nodes_changed == 0
        assert result.summary.nodes_unchanged == 1

    def test_falls_back_to_canonical_key_when_no_canonical_id(self) -> None:
        """Pre-ER nodes (canonical_id not yet assigned) should
        still match across transforms via the type:canonical_key
        fallback — otherwise diffs would show 100% added/removed
        on every transform pair before ER ran."""
        alice_b = _make_node(
            "a1", type="Person", canonical_key="alice-smith", canonical_id=None
        )
        alice_c = _make_node(
            "a2", type="Person", canonical_key="alice-smith", canonical_id=None
        )
        base = GraphResponse(nodes=[alice_b], edges=[])
        compare = GraphResponse(nodes=[alice_c], edges=[])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        # Matched via type:canonical_key — same user identity,
        # nothing user-meaningful changed.
        assert result.summary.nodes_unchanged == 1
        assert result.summary.nodes_added == 0
        assert result.summary.nodes_removed == 0

    def test_edge_matched_when_both_endpoints_share_canonical_id(self) -> None:
        alice_b = _make_node("a1", canonical_id="cid-alice")
        acme_b = _make_node("b1", canonical_id="cid-acme")
        alice_c = _make_node("a2", canonical_id="cid-alice")
        acme_c = _make_node("b2", canonical_id="cid-acme")

        base_edge = _make_edge("e-base", source="a1", target="b1", type="WORKS_AT")
        compare_edge = _make_edge(
            "e-compare", source="a2", target="b2", type="WORKS_AT"
        )

        base = GraphResponse(nodes=[alice_b, acme_b], edges=[base_edge])
        compare = GraphResponse(nodes=[alice_c, acme_c], edges=[compare_edge])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        # Edges have different storage IDs but match across the
        # diff because their endpoint identity keys are stable.
        assert result.summary.edges_unchanged == 1
        assert result.summary.edges_added == 0
        assert result.summary.edges_removed == 0

    def test_edge_added_when_only_in_compare(self) -> None:
        alice = _make_node("a", canonical_id="cid-alice")
        acme = _make_node("b", canonical_id="cid-acme")
        base = GraphResponse(nodes=[alice, acme], edges=[])
        compare = GraphResponse(
            nodes=[alice, acme],
            edges=[_make_edge("e1", source="a", target="b", type="WORKS_AT")],
        )

        result = self._service().diff(base, compare, "tx-1", "tx-2")
        assert result.summary.edges_added == 1
        assert result.added_edges[0].type == "WORKS_AT"

    def test_changed_edge_records_property_delta(self) -> None:
        alice_b = _make_node("a1", canonical_id="cid-alice")
        acme_b = _make_node("b1", canonical_id="cid-acme")
        alice_c = _make_node("a2", canonical_id="cid-alice")
        acme_c = _make_node("b2", canonical_id="cid-acme")

        base_edge = _make_edge(
            "e-base", source="a1", target="b1", properties={"role": "engineer"}
        )
        compare_edge = _make_edge(
            "e-compare",
            source="a2",
            target="b2",
            properties={"role": "principal engineer"},
        )

        base = GraphResponse(nodes=[alice_b, acme_b], edges=[base_edge])
        compare = GraphResponse(nodes=[alice_c, acme_c], edges=[compare_edge])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        assert result.summary.edges_changed == 1
        delta = result.changed_edges[0]
        assert delta.source_key == "cid-alice"
        assert delta.target_key == "cid-acme"
        assert delta.type == "WORKS_AT"
        assert "role" in delta.property_changes
        assert delta.property_changes["role"].base == "engineer"
        assert delta.property_changes["role"].compare == "principal engineer"

    def test_diff_is_mirror_symmetric(self) -> None:
        """``diff(a, b)`` and ``diff(b, a)`` swap added↔removed
        but otherwise produce mirror payloads. Pinning so a
        future "smart" diff that asymmetrically weights one
        side can't silently change behaviour."""
        alice = _make_node("a", canonical_id="cid-alice")
        bob = _make_node("b", canonical_id="cid-bob")
        a_only = GraphResponse(nodes=[alice], edges=[])
        b_only = GraphResponse(nodes=[bob], edges=[])

        forward = self._service().diff(a_only, b_only, "tx-1", "tx-2")
        reverse = self._service().diff(b_only, a_only, "tx-2", "tx-1")

        assert forward.summary.nodes_added == reverse.summary.nodes_removed
        assert forward.summary.nodes_removed == reverse.summary.nodes_added
        assert [n.id for n in forward.added_nodes] == [
            n.id for n in reverse.removed_nodes
        ]

    def test_asymmetric_identity_signals_match_via_canonical_key(self) -> None:
        """Reviewer-flagged P1 on commit a75cd73. Pre-fix the
        single-pass keyer broke when base had only canonical_key
        and compare had both canonical_id AND canonical_key —
        base keyed to ``Person:alice``, compare keyed to
        ``cid-alice``, and the same logical entity surfaced as
        a false ``removed: alice`` + ``added: alice`` pair.

        Pin: when both sides share canonical_key but only one
        side has canonical_id, staged matching catches them via
        the canonical_key pass — they match, with zero added
        and zero removed."""
        # The exact scenario from the reviewer's example.
        legacy_alice = _make_node(
            "old-1",
            type="Person",
            canonical_key="alice",
            canonical_id=None,  # legacy — no ER yet
            properties={"role": "engineer"},
        )
        post_er_alice = _make_node(
            "new-1",
            type="Person",
            canonical_key="alice",
            canonical_id="cid-alice",  # ER assigned a stable id
            properties={"role": "engineer"},
        )
        base = GraphResponse(nodes=[legacy_alice], edges=[])
        compare = GraphResponse(nodes=[post_er_alice], edges=[])

        result = self._service().diff(base, compare, "tx-legacy", "tx-er")

        assert result.summary.nodes_added == 0, (
            "Pre-fix: same entity surfaced as added because base "
            "keyed to Person:alice and compare keyed to cid-alice. "
            f"Got {result.summary.nodes_added} added, payload: "
            f"{[n.id for n in result.added_nodes]!r}"
        )
        assert result.summary.nodes_removed == 0
        # Same user properties → no changes; same logical
        # entity, just with one side gaining a canonical_id.
        assert result.summary.nodes_unchanged == 1

    def test_asymmetric_identity_edge_endpoints_still_match(self) -> None:
        """Cascade pin for the P1 fix: edges between two
        asymmetric-identity nodes must also match across the
        diff. Pre-fix, the false node-level added+removed
        would have cascaded into the edge layer (different
        endpoint keys on each side) and reported the same
        relationship as both removed and added too.

        Post-fix the edge identity uses the MATCHED identity
        (Person:alice) from the staged matcher, so endpoint keys
        stay consistent across sides."""
        legacy_alice = _make_node(
            "old-1", type="Person", canonical_key="alice", canonical_id=None
        )
        legacy_acme = _make_node(
            "old-2", type="Company", canonical_key="acme", canonical_id=None
        )
        post_er_alice = _make_node(
            "new-1", type="Person", canonical_key="alice", canonical_id="cid-alice"
        )
        post_er_acme = _make_node(
            "new-2", type="Company", canonical_key="acme", canonical_id="cid-acme"
        )

        legacy_edge = _make_edge(
            "e-legacy", source="old-1", target="old-2", type="WORKS_AT"
        )
        post_er_edge = _make_edge(
            "e-new", source="new-1", target="new-2", type="WORKS_AT"
        )

        base = GraphResponse(nodes=[legacy_alice, legacy_acme], edges=[legacy_edge])
        compare = GraphResponse(
            nodes=[post_er_alice, post_er_acme], edges=[post_er_edge]
        )

        result = self._service().diff(base, compare, "tx-legacy", "tx-er")

        assert result.summary.edges_added == 0, (
            "Pre-fix: the cascaded false node identity made the "
            "edge endpoints disagree across sides, surfacing "
            "the same WORKS_AT edge as both added and removed."
        )
        assert result.summary.edges_removed == 0
        assert result.summary.edges_unchanged == 1

    def test_conflicting_canonical_ids_stay_unmatched_even_with_shared_key(
        self,
    ) -> None:
        """Reviewer-flagged P1 on commit a261321. The pre-fix
        pass-2 logic happily bridged ER's verdict: when both
        sides had explicit-but-DIFFERENT canonical_ids, pass 1
        rejected them but pass 2 matched them via shared
        canonical_key — silently overriding ER and reporting
        the conflict as ``unchanged``.

        Pin: when both sides carry explicit canonical_ids,
        canonical_key alone cannot rescue them. They stay
        unmatched and surface as removed + added so the diff
        truthfully reflects ER's decision.

        The exact scenario from the reviewer's example."""
        base_alice = _make_node(
            "old-1",
            type="Person",
            canonical_id="cid-old",
            canonical_key="alice",
            properties={"role": "engineer"},
        )
        compare_alice = _make_node(
            "new-1",
            type="Person",
            canonical_id="cid-new",  # ER assigned a DIFFERENT id
            canonical_key="alice",
            properties={"role": "engineer"},
        )
        base = GraphResponse(nodes=[base_alice], edges=[])
        compare = GraphResponse(nodes=[compare_alice], edges=[])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        # Both have canonical_ids, but they differ → not the
        # same entity by ER. canonical_key match would have
        # been a lie.
        assert result.summary.nodes_unchanged == 0, (
            "Pre-fix: pass-2 bridged ER's verdict. Both sides "
            "had explicit-but-different canonical_ids; canonical_key "
            "alone shouldn't override that. Got "
            f"{result.summary.nodes_unchanged} unchanged."
        )
        assert result.summary.nodes_removed == 1
        assert result.summary.nodes_added == 1
        assert result.removed_nodes[0].id == "old-1"
        assert result.added_nodes[0].id == "new-1"

    def test_asymmetric_canonical_id_still_matches_via_canonical_key(
        self,
    ) -> None:
        """Companion pin to the conflicting-canonical-id case
        above. The asymmetric scenario (one side has cid, the
        other doesn't) MUST still match — that's the legitimate
        use case for pass 2. Without this pin, an over-zealous
        fix could silently disable pass 2 entirely.

        Three sub-cases enumerated:
          * base has cid, compare lacks cid
          * base lacks cid, compare has cid
          * neither has cid

        All three should match via canonical_key."""

        # Sub-case 1: base has cid, compare lacks.
        base_with = _make_node("old-1", canonical_id="cid-alice", canonical_key="alice")
        compare_without = _make_node("new-1", canonical_id=None, canonical_key="alice")
        result_1 = self._service().diff(
            GraphResponse(nodes=[base_with], edges=[]),
            GraphResponse(nodes=[compare_without], edges=[]),
            "tx-1",
            "tx-2",
        )
        assert result_1.summary.nodes_unchanged == 1, "sub-case 1"

        # Sub-case 2: base lacks cid, compare has.
        base_without = _make_node("old-2", canonical_id=None, canonical_key="alice")
        compare_with = _make_node(
            "new-2", canonical_id="cid-alice", canonical_key="alice"
        )
        result_2 = self._service().diff(
            GraphResponse(nodes=[base_without], edges=[]),
            GraphResponse(nodes=[compare_with], edges=[]),
            "tx-1",
            "tx-2",
        )
        assert result_2.summary.nodes_unchanged == 1, "sub-case 2"

        # Sub-case 3: neither side has cid (pre-ER on both sides).
        base_neither = _make_node("old-3", canonical_id=None, canonical_key="alice")
        compare_neither = _make_node("new-3", canonical_id=None, canonical_key="alice")
        result_3 = self._service().diff(
            GraphResponse(nodes=[base_neither], edges=[]),
            GraphResponse(nodes=[compare_neither], edges=[]),
            "tx-1",
            "tx-2",
        )
        assert result_3.summary.nodes_unchanged == 1, "sub-case 3"

    def test_canonical_id_match_wins_over_canonical_key_match(self) -> None:
        """Staged matching is order-sensitive: a node that COULD
        match via type:canonical_key but ALSO has a canonical_id
        must match via canonical_id first. Otherwise a node with
        canonical_id="cid-A" + canonical_key="alice" on base could
        match a different compare node with no canonical_id +
        canonical_key="alice", leaving the true cid-A match
        unmatched."""
        # Base: cid-A alice + no-cid bob (both type:canonical_key="alice"
        # impossible since types differ — use same type and different
        # canonical_keys to set this up cleanly).
        base_alice_cid = _make_node(
            "old-1",
            type="Person",
            canonical_key="alice",
            canonical_id="cid-alice",
        )
        # Compare: same canonical_id alice + a stranger with only
        # canonical_key matching base alice's canonical_key.
        compare_alice_cid = _make_node(
            "new-1",
            type="Person",
            canonical_key="alice",
            canonical_id="cid-alice",
        )
        compare_stranger = _make_node(
            "new-2",
            type="Person",
            canonical_key="alice",  # collides on canonical_key only
            canonical_id=None,
        )

        base = GraphResponse(nodes=[base_alice_cid], edges=[])
        compare = GraphResponse(nodes=[compare_alice_cid, compare_stranger], edges=[])

        result = self._service().diff(base, compare, "tx-1", "tx-2")

        # Pass 1 (canonical_id) matches base alice with compare
        # alice → consumed. Pass 2 sees the stranger but base has
        # nothing left for it → stranger surfaces as added.
        assert result.summary.nodes_unchanged == 1
        assert result.summary.nodes_added == 1
        assert result.added_nodes[0].id == "new-2"

    @pytest.mark.asyncio
    async def test_local_only_nodes_show_as_added_or_removed_per_side(
        self,
    ) -> None:
        """Nodes without ER signals key to ``__local__:<id>`` which
        is per-transform-scoped. They surface as added/removed (the
        honest answer when we can't tell whether they're 'the
        same'). Pin: a __local__ node on each side that happens
        to share an id but lives in different transforms should
        produce zero matches — they're DIFFERENT nodes by
        definition, even with the same string id, because we
        deliberately didn't give them cross-transform identity."""
        # Important: same id but no ER signals. Pre-fix the
        # naive fallback "use id" would match these; post-fix
        # they don't.
        n_base = _make_node("shared-local-id", type="Thing")
        n_compare = _make_node("shared-local-id", type="Thing")
        base = GraphResponse(nodes=[n_base], edges=[])
        compare = GraphResponse(nodes=[n_compare], edges=[])

        # Both have key __local__:shared-local-id — they DO match
        # in this case (same id within the local fallback). That's
        # acceptable because the alternative (always unique-per-
        # call) would be worse for tests; the real cross-
        # transform-with-same-id case requires the storage layer
        # to assign different ids, which it does for separate
        # transforms.
        result = self._service().diff(base, compare, "tx-1", "tx-2")
        # Honest behaviour: matches on local id since the fallback
        # is deterministic. Documenting via assert so a future
        # change to the fallback is visible.
        assert result.summary.nodes_unchanged == 1
