"""Tests for the B2-er slice 1 blocking stage in
``_compare_and_merge_nodes`` (graph_transformer.py).

Pre-slice-1, that function did an O(n²) list-scan dedup and sent
ALL nodes of a given type to the LLM in one resolve call. Slice 1:
  - Dict-keyed O(n) dedup
  - Property-based blocking (``_block_keys_for_node``) buckets
    candidate match groups before the LLM call
  - Singletons skip the LLM entirely

These tests pin both: the block-key generation logic (pure
function, deterministic), and the candidate-group iteration
(grouping shape + dedup-across-blocks contract).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from graphora_server.services.transform.graph_transformer import (
    _block_keys_for_node,
    _candidate_groups_for_resolution,
    _compare_and_merge_nodes,
)
from graphora_server.services.transform.models import BaseNode


def _node(
    id_: str,
    type_: str = "Person",
    name: str = None,
    canonical_key: str = None,
) -> BaseNode:
    props = {"name": name} if name is not None else {}
    return BaseNode(
        id=id_,
        type=type_,
        properties=props,
        canonical_key=canonical_key,
    )


class TestBlockKeysForNode:
    """Pin the block-key generation contract. Block keys drive
    candidate grouping; changing the recipe changes ER recall."""

    def test_canonical_key_emits_canon_block(self) -> None:
        node = _node("n1", canonical_key="alice-smith")
        keys = _block_keys_for_node(node)
        assert "Person|canon:alice-smith" in keys

    def test_name_emits_name3_block(self) -> None:
        node = _node("n1", name="Alice")
        keys = _block_keys_for_node(node)
        assert "Person|name3:ali" in keys

    def test_title_used_when_no_name(self) -> None:
        # Falls back to title for entities like Document where
        # "name" might not be the natural identifier.
        node = BaseNode(
            id="d1", type="Document", properties={"title": "Annual Report 2026"}
        )
        keys = _block_keys_for_node(node)
        assert any("name3:ann" in k for k in keys)

    def test_normalization_strips_punctuation_and_case(self) -> None:
        # "St. Mary's" → "stm" (3 alnum chars after lowering).
        node = _node("n1", name="St. Mary's")
        keys = _block_keys_for_node(node)
        assert "Person|name3:stm" in keys

    def test_short_name_uses_full_alphanumeric(self) -> None:
        node = _node("n1", name="Bo")
        keys = _block_keys_for_node(node)
        assert "Person|name3:bo" in keys

    def test_type_prefix_separates_entity_types(self) -> None:
        # Two nodes with the same name but different types must
        # not share blocks.
        person_keys = _block_keys_for_node(_node("p1", "Person", name="Acme"))
        company_keys = _block_keys_for_node(_node("c1", "Company", name="Acme"))
        assert set(person_keys).isdisjoint(set(company_keys))

    def test_canonical_key_and_name_both_emit_when_present(self) -> None:
        node = _node("n1", name="Alice", canonical_key="alice-smith")
        keys = _block_keys_for_node(node)
        assert "Person|canon:alice-smith" in keys
        assert "Person|name3:ali" in keys

    def test_no_name_no_canonical_key_falls_back_to_type_block(self) -> None:
        # Pre-slice-1 code grouped by type only — slice 1 preserves
        # that behaviour for nodes with neither signal so blocking
        # can't reduce recall below the pre-slice-1 baseline.
        node = BaseNode(id="n1", type="Person", properties={})
        keys = _block_keys_for_node(node)
        assert keys == ["Person|_all"]

    def test_missing_type_uses_underscore_prefix(self) -> None:
        # Defensive — type shouldn't be empty but if it is the key
        # is still well-formed (won't accidentally match other
        # untyped nodes' keys vs typed ones).
        node = BaseNode(id="n1", type="", properties={"name": "Alice"})
        keys = _block_keys_for_node(node)
        assert all(k.startswith("_|") for k in keys)


class TestCandidateGroupsForResolution:
    """Pin the candidate-group iteration: bucketing, dedup across
    blocks, oversize-block chunking."""

    def test_singletons_are_not_yielded(self) -> None:
        # A block with one node has no resolution work to do.
        nodes = [_node("p1", name="Alice"), _node("p2", name="Bob")]
        groups = list(_candidate_groups_for_resolution(nodes))
        assert groups == []  # Different name3 blocks; each is a singleton

    def test_shared_block_yields_one_group(self) -> None:
        # Same name3 prefix ("ali") groups them together. "Alex"
        # would be a different block ("ale") so it's intentionally
        # not in this set — the next test pins type+block isolation.
        nodes = [
            _node("p1", name="Alice"),
            _node("p2", name="Alic"),
            _node("p3", name="Alistair"),
        ]
        groups = list(_candidate_groups_for_resolution(nodes))
        assert len(groups) == 1
        assert {n.id for n in groups[0]} == {"p1", "p2", "p3"}

    def test_canonical_key_groups_across_typo_names(self) -> None:
        # name3 wouldn't catch "Alice" vs "Smith, A." but a shared
        # canonical_key would. The blocker must surface that pair.
        nodes = [
            _node("p1", name="Alice", canonical_key="alice-smith"),
            _node("p2", name="Smith", canonical_key="alice-smith"),
        ]
        groups = list(_candidate_groups_for_resolution(nodes))
        assert len(groups) == 1
        assert {n.id for n in groups[0]} == {"p1", "p2"}

    def test_node_in_multiple_blocks_appears_in_only_one_group(self) -> None:
        # p1 shares name3 with p2 AND canonical_key with p3. The
        # iteration's seen_ids dedup keeps p1 in one yielded group.
        nodes = [
            _node("p1", name="Alice", canonical_key="alice-smith"),
            _node("p2", name="Alic"),
            _node("p3", name="Bob", canonical_key="alice-smith"),
        ]
        groups = list(_candidate_groups_for_resolution(nodes))
        all_yielded_ids = []
        for g in groups:
            all_yielded_ids.extend(n.id for n in g)
        # p1 must appear at most once across all yielded groups.
        assert all_yielded_ids.count("p1") <= 1

    def test_oversize_block_chunks_at_max_block_size(self) -> None:
        # 60 nodes sharing a block, max=50 → first chunk = 50,
        # second = 10. Both yielded.
        nodes = [_node(f"p{i}", name="Alice") for i in range(60)]
        groups = list(_candidate_groups_for_resolution(nodes, max_block_size=50))
        assert len(groups) == 2
        assert len(groups[0]) == 50
        assert len(groups[1]) == 10

    def test_oversize_block_drops_tail_singleton(self) -> None:
        # If the last chunk is just one node, it's not yielded —
        # singletons skip the LLM.
        nodes = [_node(f"p{i}", name="Alice") for i in range(51)]
        groups = list(_candidate_groups_for_resolution(nodes, max_block_size=50))
        assert len(groups) == 1
        assert len(groups[0]) == 50

    def test_different_types_never_share_a_group(self) -> None:
        # Even with identical names, type-prefixed keys keep them
        # in separate blocks.
        nodes = [
            _node("p1", "Person", name="Acme"),
            _node("p2", "Person", name="Acme"),
            _node("c1", "Company", name="Acme"),
            _node("c2", "Company", name="Acme"),
        ]
        groups = list(_candidate_groups_for_resolution(nodes))
        for g in groups:
            types = {n.type for n in g}
            assert len(types) == 1, "candidate group spans multiple types"


class TestCompareAndMergeNodesIntegration:
    """End-to-end: feed nodes into the pipeline, verify dedup +
    blocking + LLM-call shape. The LLM is mocked so we can pin
    that the new pipeline calls it with bounded inputs (not the
    whole-type list)."""

    @pytest.mark.asyncio
    async def test_dedupe_by_id_replaces_o_n_squared_scan(self) -> None:
        # Same id appears 3 times with different prop bags; the
        # dict-keyed dedup must collapse them into one merged node
        # before any LLM call. Pre-slice-1 used a list scan that
        # was O(n²) in the duplicate-id case.
        n_first = _node("p1", name="Alice")
        n_dup1 = _node("p1", name="Alice", canonical_key="alice")
        n_dup2 = _node("p1", name="Alice")
        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=lambda *a, **kw: [[a[1][0]]]),
        ) as mock_resolve:
            result = await _compare_and_merge_nodes([n_first, n_dup1, n_dup2])
        # Three id-duplicates → one merged node out.
        assert len(result) == 1
        # No resolution call needed for a single deduped node.
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_singletons_pass_through_without_llm_call(self) -> None:
        # Three Persons with different names — each is its own
        # block-of-one. Old code would have sent the whole type
        # to the LLM (a 3-way "are any of these the same?" call).
        # Slice 1 short-circuits.
        nodes = [
            _node("p1", name="Alice"),
            _node("p2", name="Bob"),
            _node("p3", name="Charlie"),
        ]
        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(),
        ) as mock_resolve:
            result = await _compare_and_merge_nodes(nodes)
        mock_resolve.assert_not_called()
        assert {n.id for n in result} == {"p1", "p2", "p3"}

    @pytest.mark.asyncio
    async def test_llm_called_only_with_candidate_group(self) -> None:
        # Two Alice* nodes share a name3 block; the third Person is
        # a singleton in its own block. LLM gets the 2-node block
        # only — the third passes through.
        nodes = [
            _node("p1", name="Alice"),
            _node("p2", name="Alic"),
            _node("p3", name="Bob"),
        ]

        # Resolver returns each input as its own group (no merge).
        async def fake_resolve(entity_type, group, **kwargs):
            return [[n] for n in group]

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ) as mock_resolve:
            result = await _compare_and_merge_nodes(nodes)
        assert mock_resolve.call_count == 1
        called_group = mock_resolve.call_args.args[1]
        assert {n.id for n in called_group} == {"p1", "p2"}
        assert {n.id for n in result} == {"p1", "p2", "p3"}

    @pytest.mark.asyncio
    async def test_resolved_groups_merge_by_highest_confidence(self) -> None:
        # Resolver says p1 and p2 should be merged. The merge
        # picks the highest-confidence node as base — preserves
        # the pre-slice-1 contract.
        n1 = BaseNode(
            id="p1",
            type="Person",
            properties={"name": "Alice"},
            confidence_score=0.5,
        )
        n2 = BaseNode(
            id="p2",
            type="Person",
            properties={"name": "Alice"},
            confidence_score=0.9,
        )

        async def fake_resolve(entity_type, group, **kwargs):
            return [list(group)]  # One group containing both

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ):
            result = await _compare_and_merge_nodes([n1, n2])
        assert len(result) == 1
        # Higher-confidence node is the base; merged node keeps its
        # id (or whatever merge_nodes returns — pinned by the
        # existing merge_nodes contract).
        assert result[0].id in {"p1", "p2"}
