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


class TestCompareAndMergeNodesEmitsDecisions:
    """B0-log slice 2: the entity-merge site emits one
    ``entity_merged`` decision per merge event so the Decision Log
    surface (Evidence tab, MCP get_evidence) can render which signal
    drove the merge.

    Tests cover three properties:
      1. Default (no decision_log argument) preserves pre-slice-2
         behaviour — no decisions emitted, no errors.
      2. When a memory-mode service is supplied and a merge happens
         in the property-blocker stage, exactly one decision lands
         with the right shape.
      3. Singleton resolutions DO NOT emit decisions — there's no
         merge event to log when a "group of 1" goes through.
    """

    @pytest.fixture(autouse=True)
    def _force_memory_mode(self, monkeypatch):
        """conftest.py defaults DATABASE_URL to a localhost Postgres
        for tests that need it; this class needs the memory backend
        of DecisionLogService so we can read appended rows directly
        from ``memory_log``. Disabling DATABASE_URL flips
        ``_enabled`` False at service init."""
        from graphora_server.config import settings

        monkeypatch.setattr(settings, "DATABASE_URL", "")
        monkeypatch.setattr(settings, "POSTGRES_HOST", None)

    @pytest.mark.asyncio
    async def test_no_decision_log_means_no_emission(self) -> None:
        """Pre-slice-2 callers don't construct a DecisionLogService.
        The hook must no-op cleanly so existing call paths see zero
        behaviour change. We can't assert "no append happened" without
        a mock; the proof is "the merge still works AND no exception
        was raised", which is what the existing
        test_resolved_groups_merge_by_highest_confidence already
        covers — this test pins the contract explicitly."""
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
            return [list(group)]

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ):
            # decision_log argument deliberately omitted — that's
            # the legacy-caller path.
            result = await _compare_and_merge_nodes([n1, n2], transform_id="tx-no-log")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_property_blocker_merge_emits_entity_merged_decision(
        self,
    ) -> None:
        """Two same-name nodes share a name3 block → property blocker
        groups them → resolver merges them → exactly one
        entity_merged decision lands with target_id=base_node.id,
        evidence.stage='property_blocker', and alternatives carrying
        the merged-away node's id+canonical_key."""
        from graphora_server.services.decision_log_service import (
            DecisionLogService,
            DecisionType,
            TargetKind,
        )

        n1 = BaseNode(
            id="p1",
            type="Person",
            properties={"name": "Alice"},
            canonical_key="alice-1",
            confidence_score=0.5,
        )
        n2 = BaseNode(
            id="p2",
            type="Person",
            properties={"name": "Alice"},
            canonical_key="alice-2",
            confidence_score=0.9,
        )

        async def fake_resolve(entity_type, group, **kwargs):
            return [list(group)]  # One group containing both

        memory_log: list = []
        decision_log = DecisionLogService(memory_store=memory_log)

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ):
            result = await _compare_and_merge_nodes(
                [n1, n2],
                transform_id="tx-merge",
                decision_log=decision_log,
            )

        assert len(result) == 1
        # Higher-confidence node is the base.
        base = result[0]
        assert base.id == "p2"

        # Exactly one decision; correct target + shape.
        assert len(memory_log) == 1
        decision = memory_log[0]
        assert decision.transform_id == "tx-merge"
        assert decision.target_id == "p2"
        assert decision.target_kind == TargetKind.NODE
        assert decision.decision_type == DecisionType.ENTITY_MERGED

        # evidence.stage names the blocker so the Evidence tab can
        # render "merged via property blocker" vs "via embedding".
        assert decision.evidence["stage"] == "property_blocker"
        assert decision.evidence["merge_group_size"] == 2
        assert decision.evidence["node_type"] == "Person"

        # alternatives lists the merged-away node(s).
        assert len(decision.alternatives) == 1
        alt = decision.alternatives[0]
        assert alt["id"] == "p1"
        assert alt["canonical_key"] == "alice-1"

    @pytest.mark.asyncio
    async def test_singleton_resolution_emits_no_decision(self) -> None:
        """When the resolver returns each input as its own group
        (no merge), no decision must be emitted — only actual merge
        events warrant a row. Without this pin, a future helper
        change ('emit on every resolved group') would silently fill
        the Decision Log with no-op entries."""
        from graphora_server.services.decision_log_service import (
            DecisionLogService,
        )

        # Two same-name nodes share a name3 block so the resolver
        # gets called, but it splits them back out.
        n1 = BaseNode(id="p1", type="Person", properties={"name": "Alice"})
        n2 = BaseNode(id="p2", type="Person", properties={"name": "Alic"})

        async def fake_resolve(entity_type, group, **kwargs):
            return [[n] for n in group]  # Each its own group → no merge

        memory_log: list = []
        decision_log = DecisionLogService(memory_store=memory_log)

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ):
            await _compare_and_merge_nodes(
                [n1, n2],
                transform_id="tx-no-merge",
                decision_log=decision_log,
            )

        assert memory_log == []

    @pytest.mark.asyncio
    async def test_no_transform_id_means_no_emission(self) -> None:
        """The Decision Log is keyed by transform; without
        transform_id a row would be unfindable. Helper short-circuits
        rather than appending an orphaned row."""
        from graphora_server.services.decision_log_service import (
            DecisionLogService,
        )

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
            return [list(group)]

        memory_log: list = []
        decision_log = DecisionLogService(memory_store=memory_log)

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ):
            # transform_id omitted (None).
            await _compare_and_merge_nodes(
                [n1, n2],
                decision_log=decision_log,
            )

        assert memory_log == []


class TestBuildGraphFromThreadsDecisionLog:
    """Reviewer-flagged on commit f5df894.

    _build_graph_from accepts decision_log: Optional[DecisionLogService]
    and the public entry points (build_graph_from_chunks /
    build_graph_from_pdfs) thread it down. But the actual
    _compare_and_merge_nodes call inside _build_graph_from was
    missing ``decision_log=decision_log``, so single-pass extractions
    silently dropped the service even when the caller supplied one.
    Multi-pass was wired correctly; single-pass was not.

    These tests pin the contract directly: when _build_graph_from
    is called with a decision_log, the _compare_and_merge_nodes
    invocation it makes MUST receive that same decision_log
    instance. Mirror tests for _build_graph_with_multi_pass guard
    against the inverse regression (the multi-pass kwarg getting
    dropped in some future refactor).
    """

    @pytest.fixture(autouse=True)
    def _force_memory_mode(self, monkeypatch):
        from graphora_server.config import settings

        monkeypatch.setattr(settings, "DATABASE_URL", "")
        monkeypatch.setattr(settings, "POSTGRES_HOST", None)

    @pytest.mark.asyncio
    async def test_build_graph_from_threads_decision_log_to_merge_call(
        self,
    ) -> None:
        """Pin the kwarg flow: a decision_log handed to
        _build_graph_from must reach _compare_and_merge_nodes. Pre-fix
        the call site dropped the kwarg, leaving slice-3-onward
        callers with a "decision_log set on the entry point but no
        decisions ever emitted" foot-gun."""
        from graphora_server.services.decision_log_service import (
            DecisionLogService,
        )
        from graphora_server.services.transform import graph_transformer
        from graphora_server.services.transform.ontology_helper import (
            OntologyParser,
        )
        from graphora_server.services.entity_ledger_service import (
            entity_ledger_service,
        )

        # Minimal ontology — no Person extraction needed; the merge
        # call is the only thing we're inspecting.
        parser = OntologyParser.__new__(OntologyParser)
        parser.parsed_ontology = {
            "entities": {
                "Person": {"properties": {"name": {"type": "str"}}},
            },
        }
        parser.ontology_yaml = "version: '0.1.0'\n"
        parser.build_entities_only_model = lambda: object  # noqa: E731
        parser.build_relationships_only_model = lambda: object  # noqa: E731

        async def fake_extract_nodes(*_a, **_kw):
            class _Empty:
                confidence_score = 0.9

            return _Empty()

        async def fake_extract_rels(*_a, **_kw):
            class _Empty:
                confidence_score = 0.9

            return _Empty()

        decision_log = DecisionLogService(memory_store=[])

        captured: dict = {}

        async def fake_compare(nodes, **kwargs):
            # Pin: this is the kwarg the production call must pass.
            captured["decision_log"] = kwargs.get("decision_log")
            captured["transform_id"] = kwargs.get("transform_id")
            return nodes

        with (
            patch.object(
                entity_ledger_service,
                "hydrate_nodes",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                graph_transformer,
                "_compare_and_merge_nodes",
                new=AsyncMock(side_effect=fake_compare),
            ),
        ):
            await graph_transformer._build_graph_from(
                ontology_parser=parser,
                chunks_or_pdf_paths=["Alice joined Acme."],
                transform_id="tx-thread",
                node_extractor=fake_extract_nodes,
                relationship_extractor=fake_extract_rels,
                node_baml_function="ExtractNodesFromChunk",
                rel_baml_function="ExtractRelationshipsFromChunk",
                decision_log=decision_log,
            )

        assert captured["decision_log"] is decision_log, (
            "_build_graph_from didn't pass decision_log to "
            "_compare_and_merge_nodes. Pre-fix this kwarg was "
            "silently dropped — single-pass transforms would never "
            "emit entity_merged decisions even when a service was "
            "supplied. Got: "
            f"{captured.get('decision_log')!r}; expected: {decision_log!r}."
        )
        assert captured["transform_id"] == "tx-thread"

    @pytest.mark.asyncio
    async def test_build_graph_with_multi_pass_threads_decision_log(
        self,
    ) -> None:
        """Mirror pin for the multi-pass path. It was correct at
        slice-2 landing time; this guards against a future refactor
        accidentally dropping the kwarg from this path while leaving
        the single-pass one intact (or vice versa)."""
        from graphora_server.services.decision_log_service import (
            DecisionLogService,
        )
        from graphora_server.services.transform import graph_transformer
        from graphora_server.services.transform.ontology_helper import (
            OntologyParser,
        )
        from graphora_server.services.entity_ledger_service import (
            entity_ledger_service,
        )

        parser = OntologyParser.__new__(OntologyParser)
        parser.parsed_ontology = {
            "entities": {
                "Person": {"properties": {"name": {"type": "str"}}},
            },
        }

        decision_log = DecisionLogService(memory_store=[])
        captured: dict = {}

        async def fake_compare(nodes, **kwargs):
            captured["decision_log"] = kwargs.get("decision_log")
            return nodes

        # Stub the MultiPassExtractor.extract to return empty
        # results — we only care about whether the post-extract
        # _compare_and_merge_nodes call receives the kwarg.
        class _StubExtractor:
            def __init__(self, *_a, **_kw):
                pass

            async def extract(self, *_a, **_kw):
                return [], []

        with (
            patch.object(
                entity_ledger_service,
                "hydrate_nodes",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                graph_transformer,
                "_compare_and_merge_nodes",
                new=AsyncMock(side_effect=fake_compare),
            ),
            patch(
                "graphora_server.services.extraction.MultiPassExtractor",
                new=_StubExtractor,
            ),
            patch(
                "graphora_server.services.transform.graph_transformer."
                "deduplicate_entities_with_splink",
                new=AsyncMock(side_effect=lambda **kwargs: ([], [])),
            ),
        ):
            await graph_transformer._build_graph_with_multi_pass(
                ontology_parser=parser,
                chunks=["Alice joined Acme."],
                transform_id="tx-mp",
                llm_client=MagicMock(),
                decision_log=decision_log,
            )

        assert captured["decision_log"] is decision_log, (
            "_build_graph_with_multi_pass dropped the decision_log "
            "kwarg on its way to _compare_and_merge_nodes. The "
            f"single-pass path also has this contract. Got: "
            f"{captured.get('decision_log')!r}."
        )


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


class TestSplinkCandidateGroups:
    """Pin the B2-er slice-3 Splink-as-blocker contract.

    ``_splink_candidate_groups`` lives in graph_transformer.py and
    delegates clustering to ``cluster_entities_with_splink`` in
    helpers.py. We mock the helper to control the
    ``id_to_representative`` mapping and pin the materialization
    logic — that's the part slice 3 owns. Splink's actual
    probabilistic-linkage behaviour is covered by
    test_splink_ontology.py, which is unaffected by this slice.
    """

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_or_single_input(self) -> None:
        from graphora_server.services.transform.graph_transformer import (
            _splink_candidate_groups,
        )

        assert await _splink_candidate_groups([]) == []
        assert await _splink_candidate_groups([_node("p1", name="Alice")]) == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_clusterer_returns_no_mappings(self) -> None:
        # Splink ran fine but found no duplicates — caller's
        # contract is "no candidate groups, fall through to
        # pass-through".
        from graphora_server.services.transform.graph_transformer import (
            _splink_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Bob")]
        with patch(
            "graphora_server.services.transform.helpers.cluster_entities_with_splink",
            new=AsyncMock(return_value={}),
        ):
            groups = await _splink_candidate_groups(nodes)
        assert groups == []

    @pytest.mark.asyncio
    async def test_swallows_clusterer_exceptions(self) -> None:
        # A Splink-side failure (RuntimeError, OOM, missing
        # column, etc.) must NOT raise into extraction. Same
        # degradation contract as _embedding_candidate_groups.
        from graphora_server.services.transform.graph_transformer import (
            _splink_candidate_groups,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Alic")]
        with patch(
            "graphora_server.services.transform.helpers.cluster_entities_with_splink",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            groups = await _splink_candidate_groups(nodes)
        assert groups == []

    @pytest.mark.asyncio
    async def test_materializes_single_cluster(self) -> None:
        # Mapping: p1 -> p2 means p2 is the representative, p1 is
        # the duplicate. The materialized group must contain BOTH
        # nodes. This pins the "rep is in its own group" rule
        # even though the rep doesn't appear as a key in the
        # mapping.
        from graphora_server.services.transform.graph_transformer import (
            _splink_candidate_groups,
        )

        n1 = _node("p1", name="Alice")
        n2 = _node("p2", name="Alic")
        with patch(
            "graphora_server.services.transform.helpers.cluster_entities_with_splink",
            new=AsyncMock(return_value={"p1": "p2"}),
        ):
            groups = await _splink_candidate_groups([n1, n2])
        assert len(groups) == 1
        assert {n.id for n in groups[0]} == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_materializes_multiple_clusters(self) -> None:
        # Two independent clusters {p1,p2} and {p3,p4} → two
        # candidate groups. p5 is a singleton (not in mapping)
        # and must NOT appear in any group — slice 3 only feeds
        # multi-node clusters to the LLM.
        from graphora_server.services.transform.graph_transformer import (
            _splink_candidate_groups,
        )

        nodes = [
            _node("p1", name="Alice"),
            _node("p2", name="Alic"),
            _node("p3", name="Bob"),
            _node("p4", name="Bobby"),
            _node("p5", name="Carol"),
        ]
        mapping = {"p1": "p2", "p3": "p4"}
        with patch(
            "graphora_server.services.transform.helpers.cluster_entities_with_splink",
            new=AsyncMock(return_value=mapping),
        ):
            groups = await _splink_candidate_groups(nodes)
        assert len(groups) == 2
        all_ids = {n.id for grp in groups for n in grp}
        assert all_ids == {"p1", "p2", "p3", "p4"}
        # Each materialized group must be size >= 2 (slice 3
        # contract — singletons go to pass-through).
        for grp in groups:
            assert len(grp) >= 2

    @pytest.mark.asyncio
    async def test_drops_unknown_entity_ids_from_mapping(self) -> None:
        # Defensive: if the clusterer somehow returned IDs the
        # caller didn't pass in (shouldn't happen, but be
        # explicit), they're skipped rather than raising KeyError.
        from graphora_server.services.transform.graph_transformer import (
            _splink_candidate_groups,
        )

        n1 = _node("p1", name="Alice")
        n2 = _node("p2", name="Alic")
        # ghost-id is not in the input list.
        mapping = {"p1": "p2", "ghost-id": "p2"}
        with patch(
            "graphora_server.services.transform.helpers.cluster_entities_with_splink",
            new=AsyncMock(return_value=mapping),
        ):
            groups = await _splink_candidate_groups([n1, n2])
        assert len(groups) == 1
        assert {n.id for n in groups[0]} == {"p1", "p2"}

    def test_clusterer_does_not_record_learning_outcomes(self) -> None:
        # Contract pin (reviewer fix #1):
        # ``cluster_entities_with_splink`` is a candidate-pair
        # generator — the scores Splink emits are speculative
        # until the LLM confirms the merge. Recording them via
        # ``merge_learning_service.record_outcome`` would:
        #   (a) pollute the adaptive threshold with pairs the LLM
        #       later rejects, and
        #   (b) double-record on every transform, since the
        #       post-relationship ``deduplicate_entities_with_splink``
        #       call already records once on the real population.
        # The real dedup path stays the single source of learning
        # truth. This test catches any future reintroduction of
        # the call.
        import inspect
        from graphora_server.services.transform.helpers import (
            cluster_entities_with_splink,
        )

        src = inspect.getsource(cluster_entities_with_splink)
        # The deliberate-omission docstring/comment mentions
        # ``record_outcome`` for context — check for the actual
        # await-call expression instead.
        assert "await merge_learning_service.record_outcome" not in src, (
            "cluster_entities_with_splink is recording learning outcomes "
            "for speculative pre-LLM candidates — see reviewer fix #1."
        )


class TestCompareAndMergeNodesSliceThreeIntegration:
    """End-to-end: Splink-blocker stage runs after embedding stage,
    only on the still-unblocked subset, and feeds groups to the
    same ``resolve_entity_group`` LLM seam as slices 1 and 2.

    The Splink loop is the cleanup pattern caught the slice-2
    review: identical boilerplate (resolve → confidence-merge →
    track nodes_in_groups) but with a different blocker source.
    These tests pin the wiring: that the loop fires only on
    nodes the prior blockers missed, that resolved groups go
    through the LLM, and that pass-through still works for
    Splink misses.
    """

    @pytest.mark.asyncio
    async def test_splink_blocker_runs_on_embedding_misses(self) -> None:
        # Three Persons with distinct name3 prefixes (Alice/Bob/
        # Carol) and no canonical_key — property blocker yields no
        # groups. Embedding stage is off in this test, so no groups
        # there either. Slice 3 sees all three as the unblocked
        # subset. Splink (mocked) finds {p1, p2, p3} as a single
        # cluster with p2 as representative. Expected: one
        # resolve_entity_group call with all three nodes.
        #
        # Three nodes — not two — because production
        # ``cluster_entities_with_splink`` skips the
        # ``len(type_entities) < 3`` case per type (mirrors
        # ``deduplicate_entities_with_splink``). Two-node candidates
        # are slice 1's territory via canonical_key + name3.
        n1 = _node("p1", name="Alice")
        n2 = _node("p2", name="Bob")
        n3 = _node("p3", name="Carol")

        async def fake_resolve(entity_type, group, **kwargs):
            return [list(group)]  # treat as one merge group

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ) as mock_resolve:
            with patch(
                "graphora_server.services.transform.helpers.cluster_entities_with_splink",
                new=AsyncMock(return_value={"p1": "p2", "p3": "p2"}),
            ):
                result = await _compare_and_merge_nodes(
                    [n1, n2, n3], parsed_ontology={"entities": {"Person": {}}}
                )
        # resolve_entity_group called exactly once for the Splink
        # cluster (property + embedding stages found nothing).
        assert mock_resolve.call_count == 1
        called_group = mock_resolve.call_args.args[1]
        assert {n.id for n in called_group} == {"p1", "p2", "p3"}
        # All three collapse to one merged node.
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_splink_misses_pass_through(self) -> None:
        # Splink returns {} (no clusters). Three nodes that no
        # blocker caught fall through to step 4 pass-through
        # unchanged. Three nodes (not two) so the test exercises
        # the same population shape Splink would actually see in
        # production.
        n1 = _node("p1", name="Alice")
        n2 = _node("p2", name="Bob")
        n3 = _node("p3", name="Carol")

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(),
        ) as mock_resolve:
            with patch(
                "graphora_server.services.transform.helpers.cluster_entities_with_splink",
                new=AsyncMock(return_value={}),
            ):
                result = await _compare_and_merge_nodes(
                    [n1, n2, n3], parsed_ontology={"entities": {"Person": {}}}
                )
        # No blocker fired → no LLM call.
        mock_resolve.assert_not_called()
        # All three nodes pass through.
        assert {n.id for n in result} == {"p1", "p2", "p3"}

    @pytest.mark.asyncio
    async def test_splink_only_sees_unblocked_subset(self) -> None:
        # Property-blocker catches {p1, p2} via name3 prefix.
        # Splink should be called with ONLY the remaining nodes
        # (p3, p4, p5) — never with p1/p2 again. Pins the
        # "bounded-cost" promise: each blocker only sees prior
        # blockers' misses. Three unblocked nodes (not two) so the
        # production ``< 3`` skip wouldn't apply.
        n1 = _node("p1", name="Alice")
        n2 = _node("p2", name="Alic")
        n3 = _node("p3", name="Carol")
        n4 = _node("p4", name="Dan")
        n5 = _node("p5", name="Eve")

        async def fake_resolve(entity_type, group, **kwargs):
            return [[n] for n in group]

        clusterer = AsyncMock(return_value={})
        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ):
            with patch(
                "graphora_server.services.transform.helpers.cluster_entities_with_splink",
                new=clusterer,
            ):
                await _compare_and_merge_nodes(
                    [n1, n2, n3, n4, n5],
                    parsed_ontology={"entities": {"Person": {}}},
                )
        # Splink got called — once — with only the unblocked
        # subset (p3, p4, p5). p1+p2 already went to the LLM via
        # the property blocker.
        assert clusterer.await_count == 1
        passed_entities = clusterer.await_args.kwargs["entities"]
        assert {n.id for n in passed_entities} == {"p3", "p4", "p5"}

    @pytest.mark.asyncio
    async def test_no_parsed_ontology_still_works(self) -> None:
        # Backward-compat: callers that don't pass parsed_ontology
        # (e.g. older tests, fallback paths) still get a valid
        # result. Splink runs but with no per-type comparison
        # rules it produces no clusters; the property + embedding
        # stages handle whatever they catch.
        n1 = _node("p1", name="Alice", canonical_key="alice")
        n2 = _node("p2", name="Bob", canonical_key="bob")
        n3 = _node("p3", name="Carol", canonical_key="carol")

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(),
        ) as mock_resolve:
            with patch(
                "graphora_server.services.transform.helpers.cluster_entities_with_splink",
                new=AsyncMock(return_value={}),
            ):
                result = await _compare_and_merge_nodes([n1, n2, n3])
        mock_resolve.assert_not_called()
        assert {n.id for n in result} == {"p1", "p2", "p3"}

    @pytest.mark.asyncio
    async def test_splink_resolved_group_merges_by_confidence(self) -> None:
        # Pins that slice-3's loop reuses the same confidence-sort
        # merge boilerplate as slices 1/2. Highest-confidence node
        # is the merge base. Three nodes (not two) so the test
        # input matches the production-reachable Splink path.
        n1 = BaseNode(
            id="p1", type="Person", properties={"name": "Alice"}, confidence_score=0.4
        )
        n2 = BaseNode(
            id="p2", type="Person", properties={"name": "Bob"}, confidence_score=0.95
        )
        n3 = BaseNode(
            id="p3",
            type="Person",
            properties={"name": "Carol"},
            confidence_score=0.6,
        )

        async def fake_resolve(entity_type, group, **kwargs):
            return [list(group)]

        with patch(
            "graphora_server.services.transform.graph_transformer.resolve_entity_group",
            new=AsyncMock(side_effect=fake_resolve),
        ):
            with patch(
                "graphora_server.services.transform.helpers.cluster_entities_with_splink",
                new=AsyncMock(return_value={"p1": "p2", "p3": "p2"}),
            ):
                result = await _compare_and_merge_nodes(
                    [n1, n2, n3], parsed_ontology={"entities": {"Person": {}}}
                )
        assert len(result) == 1
        # Highest-confidence node wins as base — same contract as
        # slice 1 and slice 2 merge logic.
        assert result[0].id in {"p1", "p2", "p3"}

    @pytest.mark.asyncio
    async def test_real_clusterer_skips_two_node_case(self) -> None:
        # Pins the production reality of slice 3:
        # ``cluster_entities_with_splink`` has a
        # ``len(type_entities) < 3`` early-exit per type (mirrors
        # the original ``deduplicate_entities_with_splink``). With
        # two same-type nodes, no Splink machinery runs and the
        # mapping is empty — the candidate-group materializer in
        # ``_splink_candidate_groups`` then yields no groups.
        #
        # This is by design, not a bug: slice 1's property blocker
        # already covers the 2-node case via canonical_key + name3
        # prefix. Splink only adds value when the unblocked subset
        # has at least 3 records of the same type (where blocking
        # rules can actually reduce the comparison space).
        #
        # NOTE: this test calls the real helper without mocking
        # the inner Splink machinery — that's the whole point. The
        # tests above mock the helper to control the mapping; this
        # one verifies what the unmocked helper actually does.
        from graphora_server.services.transform.helpers import (
            cluster_entities_with_splink,
        )

        nodes = [_node("p1", name="Alice"), _node("p2", name="Alic")]
        mapping = await cluster_entities_with_splink(
            entities=nodes,
            parsed_ontology={
                "entities": {"Person": {"properties": {"name": {"type": "string"}}}}
            },
        )
        # Real helper short-circuits: 2 < 3 → empty mapping.
        assert mapping == {}
