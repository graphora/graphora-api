"""Tests for the orphan-relationship re-extraction pass
(``_orphan_relationship_pass`` in graph_transformer.py).

The pass runs after Step 4 (relationship dedup + Splink) and
before Step 5 (prune). For each node that ended up with no
incoming/outgoing edge it re-calls the relationship extractor
with a *focused* context — orphan(s) plus same-chunk candidates
— scoped to the chunk that produced the orphan. New edges are
deduped against the existing set and appended.

Why this matters: without it, the prune-guard's
"keep nodes that carry ontology props" branch surfaces every
orphan to the UI even when the LLM's relationship pass simply
missed the edge. The re-extraction pass plugs that gap before
the user ever sees it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphora_server.services.transform.graph_transformer import (
    _build_orphan_focused_context,
    _orphan_relationship_pass,
)
from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
)


# --------------------------------------------------------------
# Test helpers
# --------------------------------------------------------------


_ONTOLOGY = {
    "entities": {
        "Company": {
            "properties": {"name": {"type": "string", "required": True}},
            "relationships": {"HAS_BUSINESS": {"target": "Business"}},
        },
        "Business": {
            "properties": {"name": {"type": "string", "required": True}},
        },
    },
}


class _FakeChunkMetadata:
    """Minimal stand-in for ChunkMetadata. The pass only reads
    ``chunk_id``; everything else stays None so we don't accidentally
    couple this test to provenance plumbing."""

    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        self.source_file = None
        self.page_number = None
        self.start_position = None


def _make_node(
    id_: str,
    name: str,
    type_: str = "Company",
    chunk_ids: list[str] | None = None,
    original_extraction_ids: list[str] | None = None,
) -> BaseNode:
    return BaseNode(
        id=id_,
        type=type_,
        properties={"name": name},
        original_extraction_ids=list(original_extraction_ids or []),
        provenance=NodeProvenance(chunk_ids=list(chunk_ids or [])),
    )


def _make_relationship(
    source: BaseNode, target: BaseNode, type_: str = "HAS_BUSINESS"
) -> RelationshipInstance:
    return RelationshipInstance(
        type=type_,
        source_id=source.id,
        target_id=target.id,
        source_type=source.type,
        target_type=target.type,
        properties={},
    )


class _RelItem(BaseModel):
    source_id: str
    target_id: str


class _RelResult(BaseModel):
    """Mirrors transform_as_relationships' field-naming convention:
    ``<Source>_<Type>_<Target>`` mapping to a list of source/target id
    pairs. Confidence is required because the helper reads it
    directly when stamping confidence on emitted edges."""

    Company_HAS_BUSINESS_Business: list = []
    confidence_score: float = 1.0


def _fake_ontology_parser() -> MagicMock:
    parser = MagicMock()
    parser.ontology_yaml = "yaml-stub"
    return parser


# --------------------------------------------------------------
# _build_orphan_focused_context
# --------------------------------------------------------------


class TestBuildOrphanFocusedContext:
    def test_lists_orphans_under_dedicated_header(self) -> None:
        orphan = _make_node("u1", "Apple", chunk_ids=["c1"])
        context = _build_orphan_focused_context([orphan], [orphan])
        assert "Entities needing relationships:" in context
        assert "u1" in context
        assert "Apple" in context

    def test_separates_candidates_section_when_present(self) -> None:
        orphan = _make_node("u1", "Apple", type_="Company")
        candidate = _make_node("u2", "Apple Stores", type_="Business")
        context = _build_orphan_focused_context([orphan], [orphan, candidate])
        assert "Entities needing relationships:" in context
        assert "Other entities from the same chunk" in context
        assert "u2" in context

    def test_omits_candidate_section_when_only_orphans(self) -> None:
        orphan = _make_node("u1", "Apple")
        context = _build_orphan_focused_context([orphan], [orphan])
        assert "Other entities from the same chunk" not in context

    def test_filters_system_properties_via_format_helper(self) -> None:
        # source_text is in SYSTEM_PROPERTIES; if the helper leaks it,
        # the prompt-bloat regression that triggered fix 69f3cd7
        # would silently come back.
        orphan = BaseNode(
            id="u1",
            type="Company",
            properties={
                "name": "Apple",
                "source_text": "x" * 500,
            },
            provenance=NodeProvenance(chunk_ids=["c1"]),
        )
        context = _build_orphan_focused_context([orphan], [orphan])
        assert "source_text" not in context
        assert "x" * 50 not in context


# --------------------------------------------------------------
# _orphan_relationship_pass
# --------------------------------------------------------------


class TestOrphanRelationshipPass:
    @pytest.mark.asyncio
    async def test_no_orphans_short_circuits_extractor(self) -> None:
        """Every node already has an edge → the LLM is not called.
        Pin this so cost stays bounded on docs that don't need the
        second pass."""
        a = _make_node("a", "Apple", chunk_ids=["c1"])
        b = _make_node("b", "Apple Stores", type_="Business", chunk_ids=["c1"])
        rels = [_make_relationship(a, b)]

        extractor = AsyncMock()

        result = await _orphan_relationship_pass(
            nodes=[a, b],
            relationships=rels,
            chunks_or_pdf_paths=["chunk-text-1"],
            chunk_metadatas=[_FakeChunkMetadata("c1")],
            relationship_extractor=extractor,
            relationships_only_ontology=_RelResult,
            ontology_parser=_fake_ontology_parser(),
            parsed_ontology=_ONTOLOGY,
            treat_chunks_as_text=True,
            user_id="u",
            transform_id="t",
            document_usage_id=None,
            extractor_model=None,
            rel_baml_function=None,
        )

        assert result == rels
        extractor.assert_not_called()

    @pytest.mark.asyncio
    async def test_orphan_triggers_focused_extractor_call(self) -> None:
        """An orphan with a known chunk_id causes one extractor call,
        scoped to that chunk, with a context naming the orphan."""
        orphan = _make_node("a", "Apple", chunk_ids=["c1"])
        partner = _make_node("b", "Apple Stores", type_="Business", chunk_ids=["c1"])

        extractor = AsyncMock(return_value=_RelResult())

        result = await _orphan_relationship_pass(
            nodes=[orphan, partner],
            relationships=[],
            chunks_or_pdf_paths=["chunk-text-1"],
            chunk_metadatas=[_FakeChunkMetadata("c1")],
            relationship_extractor=extractor,
            relationships_only_ontology=_RelResult,
            ontology_parser=_fake_ontology_parser(),
            parsed_ontology=_ONTOLOGY,
            treat_chunks_as_text=True,
            user_id="u",
            transform_id="t",
            document_usage_id=None,
            extractor_model=None,
            rel_baml_function=None,
        )

        assert extractor.await_count == 1
        # Both args and kwargs supported — _timed_call forwards
        # positionally for the chunk and as kwargs for the rest.
        call = extractor.await_args
        assert call.args[0] == "chunk-text-1"
        assert "Apple" in call.kwargs["context"]
        assert "Entities needing relationships:" in call.kwargs["context"]
        assert result == []

    @pytest.mark.asyncio
    async def test_new_edge_from_extractor_appended(self) -> None:
        orphan = _make_node(
            "a-uuid", "Apple", chunk_ids=["c1"], original_extraction_ids=["company_0"]
        )
        partner = _make_node(
            "b-uuid",
            "Apple Stores",
            type_="Business",
            chunk_ids=["c1"],
            original_extraction_ids=["business_0"],
        )

        # The model emits positional ids; helpers' fallback resolves
        # them via original_extraction_ids.
        rel_result = _RelResult(
            Company_HAS_BUSINESS_Business=[
                _RelItem(source_id="company_0", target_id="business_0")
            ],
        )
        extractor = AsyncMock(return_value=rel_result)

        result = await _orphan_relationship_pass(
            nodes=[orphan, partner],
            relationships=[],
            chunks_or_pdf_paths=["chunk-text-1"],
            chunk_metadatas=[_FakeChunkMetadata("c1")],
            relationship_extractor=extractor,
            relationships_only_ontology=_RelResult,
            ontology_parser=_fake_ontology_parser(),
            parsed_ontology=_ONTOLOGY,
            treat_chunks_as_text=True,
            user_id="u",
            transform_id="t",
            document_usage_id=None,
            extractor_model=None,
            rel_baml_function=None,
        )

        assert len(result) == 1
        assert result[0].source_id == "a-uuid"
        assert result[0].target_id == "b-uuid"
        assert result[0].type == "HAS_BUSINESS"

    @pytest.mark.asyncio
    async def test_duplicate_edge_filtered_against_existing(self) -> None:
        """If the model re-emits an edge that already exists in the
        passed-in relationships set, we drop it. Pin this so a noisy
        re-extraction can't bloat the final graph."""
        orphan_a = _make_node(
            "a", "Apple", chunk_ids=["c1"], original_extraction_ids=["company_0"]
        )
        orphan_b = _make_node(
            "b",
            "Apple Stores",
            type_="Business",
            chunk_ids=["c1"],
            original_extraction_ids=["business_0"],
        )
        existing = [_make_relationship(orphan_a, orphan_b)]
        # Both nodes already connected — but pretend the existing
        # edge somehow didn't register them as connected. Force the
        # pass to run by passing them as orphans-by-construction:
        # we lie to the orphan detection by removing the existing
        # edge from the set used to detect orphans. Instead, just
        # set up: existing has rel a→b, and the extractor will
        # re-emit a→b. Since a and b are already connected, the
        # detection will skip them — that's the wrong test.
        #
        # Rebuild: make a third node c that's an orphan and have
        # the extractor try to emit an a→b edge as a "new" edge.
        # The duplicate check should drop it.
        orphan_c = _make_node(
            "c", "Cupertino", chunk_ids=["c1"], original_extraction_ids=["company_1"]
        )
        # Existing edge: a→b
        existing = [_make_relationship(orphan_a, orphan_b)]
        # Extractor emits a→b again (already exists) and nothing else.
        rel_result = _RelResult(
            Company_HAS_BUSINESS_Business=[
                _RelItem(source_id="company_0", target_id="business_0"),
            ],
        )
        extractor = AsyncMock(return_value=rel_result)

        result = await _orphan_relationship_pass(
            nodes=[orphan_a, orphan_b, orphan_c],
            relationships=existing,
            chunks_or_pdf_paths=["chunk-text-1"],
            chunk_metadatas=[_FakeChunkMetadata("c1")],
            relationship_extractor=extractor,
            relationships_only_ontology=_RelResult,
            ontology_parser=_fake_ontology_parser(),
            parsed_ontology=_ONTOLOGY,
            treat_chunks_as_text=True,
            user_id="u",
            transform_id="t",
            document_usage_id=None,
            extractor_model=None,
            rel_baml_function=None,
        )

        # Original a→b stays, no duplicate appended.
        assert len(result) == 1
        assert result[0].source_id == "a"
        assert result[0].target_id == "b"

    @pytest.mark.asyncio
    async def test_orphan_with_unmappable_chunk_skipped(self) -> None:
        """Orphan whose chunk_id doesn't appear in chunk_metadatas
        cannot be re-extracted (we have no chunk to send). Pin the
        skip so the loop doesn't crash on stale provenance."""
        orphan = _make_node("a", "Apple", chunk_ids=["unknown-chunk"])
        extractor = AsyncMock()

        result = await _orphan_relationship_pass(
            nodes=[orphan],
            relationships=[],
            chunks_or_pdf_paths=["chunk-text-1"],
            chunk_metadatas=[_FakeChunkMetadata("c1")],
            relationship_extractor=extractor,
            relationships_only_ontology=_RelResult,
            ontology_parser=_fake_ontology_parser(),
            parsed_ontology=_ONTOLOGY,
            treat_chunks_as_text=True,
            user_id="u",
            transform_id="t",
            document_usage_id=None,
            extractor_model=None,
            rel_baml_function=None,
        )

        extractor.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_extractor_failure_keeps_existing_relationships(self) -> None:
        """If the LLM call raises (rate limit, timeout, parse error),
        we log and continue — the original relationship list comes
        back intact rather than the whole pipeline tipping over."""
        orphan_a = _make_node(
            "a", "Apple", chunk_ids=["c1"], original_extraction_ids=["company_0"]
        )
        orphan_b = _make_node(
            "b",
            "Apple Stores",
            type_="Business",
            chunk_ids=["c1"],
            original_extraction_ids=["business_0"],
        )
        extractor = AsyncMock(side_effect=RuntimeError("boom"))

        result = await _orphan_relationship_pass(
            nodes=[orphan_a, orphan_b],
            relationships=[],
            chunks_or_pdf_paths=["chunk-text-1"],
            chunk_metadatas=[_FakeChunkMetadata("c1")],
            relationship_extractor=extractor,
            relationships_only_ontology=_RelResult,
            ontology_parser=_fake_ontology_parser(),
            parsed_ontology=_ONTOLOGY,
            treat_chunks_as_text=True,
            user_id="u",
            transform_id="t",
            document_usage_id=None,
            extractor_model=None,
            rel_baml_function=None,
        )

        # Extractor was invoked, but the failure didn't propagate.
        assert extractor.await_count == 1
        assert result == []

    @pytest.mark.asyncio
    async def test_two_orphans_share_chunk_only_one_call(self) -> None:
        """Two orphans from the same chunk should result in ONE
        extractor call, not two. Pin the per-chunk grouping so
        cost stays bounded as orphan count grows."""
        orphan_a = _make_node(
            "a", "Apple", chunk_ids=["c1"], original_extraction_ids=["company_0"]
        )
        orphan_b = _make_node(
            "b", "Microsoft", chunk_ids=["c1"], original_extraction_ids=["company_1"]
        )
        extractor = AsyncMock(return_value=_RelResult())

        await _orphan_relationship_pass(
            nodes=[orphan_a, orphan_b],
            relationships=[],
            chunks_or_pdf_paths=["chunk-text-1"],
            chunk_metadatas=[_FakeChunkMetadata("c1")],
            relationship_extractor=extractor,
            relationships_only_ontology=_RelResult,
            ontology_parser=_fake_ontology_parser(),
            parsed_ontology=_ONTOLOGY,
            treat_chunks_as_text=True,
            user_id="u",
            transform_id="t",
            document_usage_id=None,
            extractor_model=None,
            rel_baml_function=None,
        )

        assert extractor.await_count == 1
        # Both orphans named in the focused context.
        ctx = extractor.await_args.kwargs["context"]
        assert "Apple" in ctx and "Microsoft" in ctx
