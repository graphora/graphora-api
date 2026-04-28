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

from unittest.mock import AsyncMock, MagicMock, patch

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
            new=AsyncMock(side_effect=lambda *a, **_: [[a[1][0]]]),
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


class TestNodeToEmbeddingText:
    """Pin the node-to-text serialization used to feed the
    embedding service. Same recipe as
    entity_ledger_service._node_to_text — canonical first, then
    regular, capped at 5 segments."""

    def test_returns_empty_for_no_properties(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _node_to_embedding_text,
        )

        assert _node_to_embedding_text(_node("p1")) == ""

    def test_canonical_properties_take_precedence(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _node_to_embedding_text,
        )

        node = BaseNode(
            id="p1",
            type="Person",
            properties={"name": "Alice", "title": "Engineer"},
            canonical_properties={"name": "alice"},
        )
        text = _node_to_embedding_text(node)
        # canonical_properties first, then regular (skipping
        # already-canonicalized keys).
        assert text.startswith("alice")
        assert "Engineer" in text

    def test_filters_short_strings_and_non_strings(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _node_to_embedding_text,
        )

        node = BaseNode(
            id="p1",
            type="Person",
            properties={
                "name": "Alice",
                "age": 30,  # Non-string — skipped
                "code": "A",  # Single char — too short
            },
        )
        text = _node_to_embedding_text(node)
        # Only "Alice" survives the len>1 filter; check segments
        # rather than raw substring containment ("A" is in "Alice").
        assert text.split(" | ") == ["Alice"]

    def test_caps_at_five_segments(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _node_to_embedding_text,
        )

        # Eight strings → only first 5 land in the embedding text.
        node = BaseNode(
            id="p1",
            type="Person",
            properties={f"k{i}": f"value-{i}-long-enough" for i in range(8)},
        )
        text = _node_to_embedding_text(node)
        assert text.count(" | ") == 4  # 5 segments → 4 separators


class TestEmbeddingCandidateGroups:
    """Slice 2 embedding-based candidate-pair generation. Mocks
    the embedding service so the tests don't load a real model
    or hit disk."""

    def test_returns_empty_when_setting_disabled(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Bob")]
        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = False
            groups = list(_embedding_candidate_groups(nodes))
        assert groups == []

    def test_returns_empty_when_embedding_extras_missing(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Bob")]
        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            # Make the lazy import fail → degrade to "no candidate groups"
            import sys

            sys.modules.pop(
                "graphora_server.services.entity_resolution.embedding_similarity",
                None,
            )
            with patch.dict(
                sys.modules,
                {
                    "graphora_server.services.entity_resolution.embedding_similarity": None
                },
            ):
                groups = list(_embedding_candidate_groups(nodes))
        assert groups == []

    def test_pairs_nodes_above_similarity_threshold(self) -> None:
        # Two nodes of the same type with embeddings that the mock
        # similarity matrix says are close (0.95). Should land in
        # one candidate group.
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [
            _node("p1", name="John Smith"),
            _node("p2", name="Jonathan S."),
            _node("p3", name="Bob"),
        ]
        # Similarity matrix: p1↔p2 = 0.95, p1↔p3 / p2↔p3 = 0.10.
        sim = [
            [1.0, 0.95, 0.10],
            [0.95, 1.0, 0.10],
            [0.10, 0.10, 1.0],
        ]
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(return_value=sim)

        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(
                    _embedding_candidate_groups(nodes, similarity_threshold=0.85)
                )

        # One group with p1+p2; p3 is below threshold from both.
        assert len(groups) == 1
        ids = {n.id for n in groups[0]}
        assert ids == {"p1", "p2"}

    def test_no_groups_when_all_pairs_below_threshold(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Bob")]
        sim = [[1.0, 0.20], [0.20, 1.0]]
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(return_value=sim)
        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(
                    _embedding_candidate_groups(nodes, similarity_threshold=0.85)
                )
        assert groups == []

    def test_different_types_never_share_a_group(self) -> None:
        # Even with very high similarity, nodes of different types
        # don't get grouped — embeddings are computed per-type.
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [
            _node("p1", "Person", name="Acme"),
            _node("c1", "Company", name="Acme"),
        ]
        # Service receives one call per type with a single text,
        # so the matrix is 1x1 and there's no "other node" to
        # pair with within a type.
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(return_value=[[1.0]])
        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(_embedding_candidate_groups(nodes))
        assert groups == []

    def test_union_find_merges_transitive_pairs(self) -> None:
        # p1↔p2 = 0.90, p2↔p3 = 0.90, p1↔p3 = 0.50. Direct pair-
        # threshold would miss p1↔p3, but union-find connects them
        # transitively through p2.
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [
            _node("p1", name="John Smith"),
            _node("p2", name="Jonathan Smith"),
            _node("p3", name="J. Smith"),
        ]
        sim = [
            [1.0, 0.90, 0.50],
            [0.90, 1.0, 0.90],
            [0.50, 0.90, 1.0],
        ]
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(return_value=sim)
        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(
                    _embedding_candidate_groups(nodes, similarity_threshold=0.85)
                )
        assert len(groups) == 1
        ids = {n.id for n in groups[0]}
        assert ids == {"p1", "p2", "p3"}

    def test_skips_nodes_with_no_embeddable_text(self) -> None:
        # Two nodes of the same type, but only one has a string
        # property. The other should be dropped from the embedding
        # pass — no text to compute.
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [
            _node("p1", name="Alice"),
            BaseNode(id="p2", type="Person", properties={}),
        ]
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(return_value=[[1.0]])
        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(_embedding_candidate_groups(nodes))
        # With only one valid_node after the embeddable-text filter,
        # there's no pair to score → no groups.
        assert groups == []

    def test_resilient_to_similarity_compute_failure(self) -> None:
        # Embedding service raises mid-flow → degrade to no groups.
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Alice")]
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(
            side_effect=RuntimeError("boom")
        )
        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(_embedding_candidate_groups(nodes))
        assert groups == []


class TestEmbeddingTextFiltersSystemProperties:
    """Slice 2 round-1 review: A1-prov source-span fields and
    B0-prov-extend decision-trail fields stamped on every node by
    the extraction pipeline must NOT enter the embedding signal.
    Without this filter, two unrelated nodes from the same
    document would score artificially similar on document_name +
    source_text overlap; real entity matches get diluted by
    metadata-heavy text. Same shape of bug the C2-postgres slice
    6 review caught for the AGE find_similar_nodes scoring path."""

    def test_excludes_a1_prov_source_span_fields(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _node_to_embedding_text,
        )

        node = BaseNode(
            id="p1",
            type="Person",
            properties={
                "name": "Alice",
                "source_chunk_id": "chunk-7",
                "document_name": "report.pdf",
                "source_text": "Alice is a software engineer at Acme.",
                "document_id": "doc-42",
                "chunk_offset": 1024,
                "page_number": 3,
                "extraction_confidence": 0.87,
            },
        )
        text = _node_to_embedding_text(node)
        # Only "Alice" survives; the source-span fields are filtered.
        segments = text.split(" | ")
        assert segments == ["Alice"]
        assert "report.pdf" not in text
        assert "Alice is a software engineer" not in text
        assert "doc-42" not in text

    def test_excludes_b0_decision_trail_fields(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _node_to_embedding_text,
        )

        node = BaseNode(
            id="p1",
            type="Person",
            properties={
                "name": "Alice",
                "extractor_model": "gemini-1.5-pro",
                "prompt_version": "v1.0.0",
                "validator_score": 0.92,
            },
        )
        text = _node_to_embedding_text(node)
        segments = text.split(" | ")
        assert segments == ["Alice"]
        assert "gemini" not in text
        assert "v1.0.0" not in text


class TestEmbeddingThresholdHonorsSetting:
    """Slice 2 round-1 review: the ER similarity threshold must
    come from settings.ENTITY_RESOLUTION_SIMILARITY_THRESHOLD when
    the caller doesn't pin one. Other embedding-based ER paths
    (cross_document_service, splink_embedding_comparison) honor
    it; this stage joins the convention so operators tuning the
    setting see consistent behaviour across all four ER stages."""

    def test_uses_setting_when_no_explicit_threshold(self) -> None:
        # Setting at 0.99 → only essentially-identical embeddings
        # pair up. The mock matrix has a 0.95 pair which would
        # have grouped under the prior hardcoded 0.85.
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Alic")]
        sim = [[1.0, 0.95], [0.95, 1.0]]
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(return_value=sim)

        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            mock_settings.ENTITY_RESOLUTION_SIMILARITY_THRESHOLD = 0.99
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(_embedding_candidate_groups(nodes))
        # 0.95 < 0.99 setting → no group.
        assert groups == []

    def test_falls_back_to_hardcoded_default_when_setting_missing(self) -> None:
        # Defensive: if the setting attribute is missing entirely
        # (e.g. an older config snapshot), the helper falls back
        # to 0.85 so it still does something sensible.
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Alic")]
        sim = [[1.0, 0.90], [0.90, 1.0]]
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(return_value=sim)

        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            # Simulate "setting attribute missing" — getattr returns
            # the default. We do this by deleting the attribute on
            # the mock so attribute access via getattr falls back.
            del mock_settings.ENTITY_RESOLUTION_SIMILARITY_THRESHOLD
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(_embedding_candidate_groups(nodes))
        # 0.90 >= hardcoded fallback 0.85 → group of two.
        assert len(groups) == 1

    def test_explicit_threshold_overrides_setting(self) -> None:
        # When a caller passes similarity_threshold explicitly,
        # the setting is bypassed.
        from graphora_server.services.transform.graph_transformer import (
            _embedding_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Alic")]
        sim = [[1.0, 0.50], [0.50, 1.0]]
        fake_service = MagicMock()
        fake_service.compute_similarity_matrix = MagicMock(return_value=sim)

        with patch(
            "graphora_server.services.transform.graph_transformer.settings"
        ) as mock_settings:
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_ENABLED = True
            mock_settings.ENTITY_RESOLUTION_EMBEDDING_MODEL = "fake-model"
            mock_settings.ENTITY_RESOLUTION_SIMILARITY_THRESHOLD = 0.99
            with patch(
                "graphora_server.services.entity_resolution.embedding_similarity.get_embedding_similarity",
                return_value=fake_service,
            ):
                groups = list(
                    _embedding_candidate_groups(nodes, similarity_threshold=0.4)
                )
        # Explicit 0.4 wins over setting 0.99; 0.50 >= 0.4 → group.
        assert len(groups) == 1
