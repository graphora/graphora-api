"""A1-prov regression tests.

Covers the source-span provenance contract that links the chunker
output → extraction output → node/edge properties consumed by the
Explorer Evidence tab and the MCP ``get_evidence`` tool.

The test names follow the brief in
``work/Graphora/a1-prov-implementation.md`` so a reviewer can map
1:1 between spec and verification.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


from graphora_server.services.chunking.models import ChunkMetadata
from graphora_server.services.chunking.tasks import _page_number_from_path
from graphora_server.services.transform.helpers import (
    _attach_provenance_properties,
    transform_as_nodes,
)
from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
)


# ---- Test 1 + 2: schema round-trip ----------------------------------------


class TestSchemaRoundTrip:
    def test_chunk_metadata_round_trips_new_fields(self) -> None:
        cm = ChunkMetadata(
            transform_id="t1",
            chunk_id="c1",
            source_file="paper.pdf",
            page_number=3,
        )
        # Pydantic round-trip via dict.
        restored = ChunkMetadata(**cm.model_dump())
        assert restored.source_file == "paper.pdf"
        assert restored.page_number == 3

    def test_node_provenance_round_trips_new_fields(self) -> None:
        prov = NodeProvenance(
            chunk_ids=["c1"],
            confidence_score=0.91,
            source_file="paper.pdf",
            page_number=3,
            char_offset=1024,
        )
        restored = NodeProvenance(**prov.model_dump())
        assert restored.source_file == "paper.pdf"
        assert restored.page_number == 3
        assert restored.char_offset == 1024


# ---- Test 3 + 4: page-number filename parser ------------------------------


class TestPageNumberParser:
    def test_pdf_split_filename_extracts_page(self) -> None:
        assert _page_number_from_path(Path("page_abc123_3.pdf")) == 3
        assert _page_number_from_path(Path("/tmp/x/page_def456_27.pdf")) == 27
        # Single-digit and multi-digit pages.
        assert _page_number_from_path(Path("page_a_1.pdf")) == 1
        assert _page_number_from_path(Path("page_a_999.pdf")) == 999

    def test_non_page_filenames_return_none(self) -> None:
        assert _page_number_from_path(Path("doc.txt")) is None
        assert _page_number_from_path(Path("report.pdf")) is None
        # Pattern with prefix-uppercase (regex is case-insensitive on
        # the prefix anchor only — confirmed via re.IGNORECASE flag).
        assert _page_number_from_path(Path("Page_abc_3.pdf")) == 3
        # Missing trailing -<n>.
        assert _page_number_from_path(Path("page_abc.pdf")) is None
        # Wrong extension.
        assert _page_number_from_path(Path("page_abc_3.txt")) is None

    def test_real_uuid4_split_pdf_filenames_parse(self) -> None:
        """Regression: split_pdf uses ``str(uuid.uuid4())`` which
        produces hyphenated uuids like ``123e4567-e89b-12d3-a456-
        426614174000``. The regex must accept hyphens — without this
        the entire PDF-binary provenance path silently falls back to
        page_number=None on every real extraction.
        """
        import uuid as _uuid

        for trailing_page in (1, 27, 100, 999):
            real_uuid = str(_uuid.uuid4())
            filename = f"page_{real_uuid}_{trailing_page}.pdf"
            assert (
                _page_number_from_path(Path(filename)) == trailing_page
            ), f"failed to parse {filename}"


# ---- Test 5: helper happy path --------------------------------------------


class TestAttachProvenancePropertiesHappyPath:
    def test_full_metadata_writes_every_evidence_key(self) -> None:
        node = BaseNode(
            id="n1",
            type="Person",
            properties={"name": "Alice"},
            provenance=NodeProvenance(chunk_ids=["c1"], confidence_score=0.95),
        )
        cm = ChunkMetadata(
            transform_id="t1",
            chunk_id="c1",
            chunk_index=0,
            source_file="paper.pdf",
            page_number=3,
            start_position=512,
        )
        _attach_provenance_properties(
            node,
            chunk_metadata=cm,
            chunk_text="Alice joined Acme in 2019.",
            document_id="doc-7",
        )
        props = node.properties
        assert props["source_chunk_id"] == "c1"
        assert props["document_name"] == "paper.pdf"
        assert props["page_number"] == 3
        assert props["chunk_offset"] == 512
        assert props["source_text"] == "Alice joined Acme in 2019."
        assert props["document_id"] == "doc-7"
        assert props["extraction_confidence"] == 0.95


# ---- Test 6: graceful no-op when nothing is supplied ----------------------


class TestAttachProvenancePropertiesPartial:
    def test_no_metadata_only_confidence_written(self) -> None:
        node = BaseNode(
            id="n1",
            type="Person",
            properties={"name": "Alice"},
            provenance=NodeProvenance(chunk_ids=["c1"], confidence_score=0.95),
        )
        _attach_provenance_properties(node)
        # Only the confidence_score from existing provenance should
        # survive — nothing else.
        assert "extraction_confidence" in node.properties
        assert "source_chunk_id" not in node.properties
        assert "document_name" not in node.properties
        assert "source_text" not in node.properties

    def test_no_provenance_object_no_crash(self) -> None:
        node = BaseNode(
            id="n1",
            type="Person",
            properties={"name": "Alice"},
            provenance=None,
        )
        _attach_provenance_properties(node)
        assert "extraction_confidence" not in node.properties

    def test_chunk_offset_zero_is_skipped(self) -> None:
        """start_position defaults to 0; don't pollute every node
        with an uninformative chunk_offset: 0."""
        node = BaseNode(id="n", type="T", properties={})
        cm = ChunkMetadata(
            transform_id="t", chunk_id="c", source_file="f.txt", start_position=0
        )
        _attach_provenance_properties(node, chunk_metadata=cm)
        assert "chunk_offset" not in node.properties

    def test_document_id_auto_derives_from_source_file(self) -> None:
        """Regression for the document_id-never-populated finding.

        Callers don't have to thread document_id explicitly — the
        helper falls back to chunk_metadata.source_file, which is the
        same value the upload site (api/transform.py) treats as the
        per-file identifier (safe_filename).
        """
        node = BaseNode(id="n", type="T", properties={})
        cm = ChunkMetadata(
            transform_id="t",
            chunk_id="c",
            source_file="paper.pdf",
            start_position=0,
        )
        _attach_provenance_properties(
            node, chunk_metadata=cm, chunk_text=None, document_id=None
        )
        assert node.properties["document_id"] == "paper.pdf"
        assert node.properties["document_name"] == "paper.pdf"

    def test_explicit_document_id_wins_over_auto_derived(self) -> None:
        node = BaseNode(id="n", type="T", properties={})
        cm = ChunkMetadata(transform_id="t", chunk_id="c", source_file="paper.pdf")
        _attach_provenance_properties(node, chunk_metadata=cm, document_id="doc-7")
        assert node.properties["document_id"] == "doc-7"
        assert node.properties["document_name"] == "paper.pdf"


# ---- Test 7: setdefault semantics — never clobber LLM values --------------


class TestSetdefaultSemantics:
    def test_existing_property_value_is_not_overwritten(self) -> None:
        node = BaseNode(
            id="n",
            type="Person",
            properties={
                "name": "Alice",
                "document_id": "set-by-llm",
                "page_number": 99,
            },
            provenance=NodeProvenance(chunk_ids=["c"], confidence_score=0.9),
        )
        cm = ChunkMetadata(
            transform_id="t",
            chunk_id="c",
            source_file="paper.pdf",
            page_number=3,
            start_position=10,
        )
        _attach_provenance_properties(
            node,
            chunk_metadata=cm,
            chunk_text="hello",
            document_id="set-by-system",
        )
        # LLM-emitted values win.
        assert node.properties["document_id"] == "set-by-llm"
        assert node.properties["page_number"] == 99
        # Untouched-by-LLM keys filled in.
        assert node.properties["document_name"] == "paper.pdf"
        assert node.properties["source_text"] == "hello"


# ---- Test 8: end-to-end through transform_as_nodes / _relationships -------


class _StubEntityResult:
    """Mimic the BAML entity-extraction shape ``transform_as_nodes`` walks.

    The function iterates ``dir()`` for ``*_list`` attributes, strips
    ``_list``, and uses the result as the entity_type lookup key in
    the ontology. Field name must exactly match the ontology entity
    case (``Person_list`` → looks up ``Person``).
    """

    def __init__(self):
        self.Person_list = [_StubEntity(name="Alice")]
        self.confidence_score = 0.92


class _StubEntity:
    def __init__(self, name: str):
        self.name = name


class TestTransformAsNodesIntegrates:
    """End-to-end: stub a BAML result + chunk metadata → assert the
    resulting BaseNode carries every _EVIDENCE_KEYS field."""

    def test_node_emerges_with_full_provenance_properties(self) -> None:
        cm = ChunkMetadata(
            transform_id="tx-1",
            chunk_id="chunk-42",
            chunk_index=0,
            source_file="paper.pdf",
            page_number=3,
            start_position=1024,
        )
        ontology = {
            "entities": {
                "Person": {
                    "properties": {
                        "name": {"type": "str", "required": True},
                    }
                }
            }
        }

        nodes = transform_as_nodes(
            ontology,
            _StubEntityResult(),
            transform_id="tx-1",
            chunk_metadata=cm,
            chunk_text="Alice joined Acme in 2019.",
            document_id="doc-7",
        )

        assert len(nodes) == 1
        node = nodes[0]
        # NodeProvenance carries the structured copy.
        assert node.provenance is not None
        assert node.provenance.source_file == "paper.pdf"
        assert node.provenance.page_number == 3
        assert node.provenance.char_offset == 1024
        assert node.provenance.chunk_ids == ["chunk-42"]
        # Properties carry the flat _EVIDENCE_KEYS surface the FE +
        # MCP read.
        props = node.properties
        assert props["source_chunk_id"] == "chunk-42"
        assert props["document_name"] == "paper.pdf"
        assert props["page_number"] == 3
        assert props["chunk_offset"] == 1024
        assert props["source_text"] == "Alice joined Acme in 2019."
        assert props["document_id"] == "doc-7"
        assert props["extraction_confidence"] == 0.92


# ---- Coverage gaps surfaced by the second review --------------------------


class TestPdfBinaryPathProvenance:
    """Production-contract regression for the PDF-binary path
    (Gemini multimodal, default for PDFs).

    The contract this suite asserts — matches the brief's
    desired-state table:
      * source_chunk_id   ← cm.chunk_id (split filename)
      * document_name     ← cm.source_file (original PDF filename)
      * source_text       ← cm.source_text (excerpt extracted by
                             DocumentParser at split time)
      * extraction_confidence ← LLM confidence_score
      * page_number       INTENTIONALLY ABSENT — split filename's
                          trailing integer is the chunk's last-page
                          index, not the per-fact page; emitting it
                          as page_number would be wrong provenance.
                          Per-page citation is Gate 4 work.
      * chunk_offset      INTENTIONALLY ABSENT — no character-level
                          offset is meaningful for a binary PDF
                          chunk.

    The end-to-end test below exercises the contract via
    _build_graph_from(treat_chunks_as_text=False), which is the path
    flows.py routes binary PDFs through.
    """

    @pytest.mark.asyncio
    async def test_pdf_binary_path_writes_source_text_from_chunk_metadata(
        self,
    ) -> None:
        """Closes the brief's contract gap: binary PDF nodes need
        source_text. flows.py pre-extracts text per split via
        DocumentParser, stores on ChunkMetadata.source_text, and the
        helper picks it up even though the chunk_text arg is None
        (the binary path passes treat_chunks_as_text=False)."""
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
                "Person": {"properties": {"name": {"type": "str", "required": True}}}
            }
        }
        parser.ontology_yaml = "version: '0.1.0'\n"
        parser.build_entities_only_model = lambda: object  # noqa: E731
        parser.build_relationships_only_model = lambda: object  # noqa: E731

        async def fake_extract_nodes(*_a, **_kw):
            return _StubEntityResult()

        async def fake_extract_rels(*_a, **_kw):
            class _Empty:
                confidence_score = 0.9

            return _Empty()

        cm = ChunkMetadata(
            transform_id="tx",
            chunk_id="page_abc-def_3.pdf",
            source_file="report.pdf",
            source_text="Alice joined Acme in 2019. (extracted from page 1.)",
        )

        with patch.object(
            entity_ledger_service, "hydrate_nodes", new=AsyncMock(return_value=None)
        ):
            graph = await graph_transformer._build_graph_from(
                ontology_parser=parser,
                chunks_or_pdf_paths=["/tmp/pdf/page_abc-def_3.pdf"],
                transform_id="tx",
                node_extractor=fake_extract_nodes,
                relationship_extractor=fake_extract_rels,
                chunk_metadatas=[cm],
                treat_chunks_as_text=False,
            )

        node = graph.nodes[0]
        # source_text DOES populate on the binary path now — pulled
        # from ChunkMetadata.source_text since the helper found it
        # there. Path strings would still NOT show up because the
        # treat_chunks_as_text=False gate prevents the path from
        # being passed as chunk_text.
        assert (
            node.properties["source_text"]
            == "Alice joined Acme in 2019. (extracted from page 1.)"
        )
        assert node.properties["source_chunk_id"] == "page_abc-def_3.pdf"
        assert node.properties["document_name"] == "report.pdf"
        # document_id auto-derives from source_file when not passed.
        assert node.properties["document_id"] == "report.pdf"
        # extraction_confidence flows through from the LLM stub's
        # confidence_score (0.92 from _StubEntityResult).
        assert node.properties["extraction_confidence"] == 0.92
        # page_number INTENTIONALLY absent — see class docstring.
        assert "page_number" not in node.properties
        # chunk_offset INTENTIONALLY absent — no character offset
        # for binary PDF chunks.
        assert "chunk_offset" not in node.properties


class TestOllamaTextSidecarPath:
    """Regression for finding #3 — Ollama-extracted PDF previously
    cited the .txt sidecar as document_name, not the original .pdf."""

    def test_chunk_metadata_source_file_can_be_overridden_post_chunk(
        self,
    ) -> None:
        """flows.py overwrites source_file after chunk_document
        returns, when a PDF-to-text sidecar mapping exists. This
        test exercises the override mechanism on the data model
        without re-running the full flow."""
        cm = ChunkMetadata(
            transform_id="tx",
            chunk_id="c1",
            source_file="report.txt",
        )
        # Simulate what flows.py does after chunking the sidecar.
        cm.source_file = "report.pdf"
        assert cm.source_file == "report.pdf"

        # And the helper picks up the override correctly.
        node = BaseNode(id="n", type="T", properties={})
        _attach_provenance_properties(node, chunk_metadata=cm)
        assert node.properties["document_name"] == "report.pdf"
        assert node.properties["document_id"] == "report.pdf"


class TestRefinementPassProvenance:
    """The multi-pass refinement (gap re-extraction) used to call
    transform_as_nodes/_relationships without chunk_metadata, so any
    node introduced during refinement fell out of the contract.
    """

    @pytest.mark.asyncio
    async def test_refinement_pass_threads_chunk_metadata(self) -> None:
        from graphora_server.services.extraction.multi_pass_extractor import (
            MultiPassExtractor,
            MultiPassConfig,
        )
        from graphora_server.services.extraction.models import (
            ExtractionGap,
            GapType,
        )

        # Minimal stub ontology parser.
        class _Parser:
            parsed_ontology = {
                "entities": {
                    "Person": {
                        "properties": {"name": {"type": "str", "required": True}}
                    }
                }
            }
            ontology_yaml = "v: 1\n"

            def build_entities_only_model(self):
                return object

            def build_relationships_only_model(self):
                return object

        # Stub LLM client whose every call returns one Person.
        class _StubLLM:
            async def extract_nodes_from_chunk(self, *_a, **_kw):
                return _StubEntityResult()

            async def extract_relationships_from_chunk(self, *_a, **_kw):
                class _Empty:
                    confidence_score = 0.9

                return _Empty()

        ext = MultiPassExtractor(
            ontology_parser=_Parser(),
            llm_client=_StubLLM(),
            config=MultiPassConfig(
                max_passes=2,
                gap_severity_threshold=0.5,
                enable_parallel_refinement=False,
            ),
        )

        gap = ExtractionGap(
            gap_type=GapType.LOW_CONFIDENCE,
            description="forced",
            severity=1.0,
            chunk_indices=[0],
        )
        cm = ChunkMetadata(
            transform_id="tx",
            chunk_id="c0",
            source_file="paper.pdf",
            page_number=2,
        )

        new_nodes, _new_rels = await ext._extract_for_gaps(
            gaps=[gap],
            chunks=["chunk text body"],
            existing_nodes=[],
            existing_relationships=[],
            transform_id="tx",
            user_id=None,
            chunk_metadatas=[cm],
        )
        assert new_nodes, "expected refinement to produce at least one node"
        node = new_nodes[0]
        assert node.properties.get("source_chunk_id") == "c0"
        assert node.properties.get("document_name") == "paper.pdf"
        assert node.properties.get("page_number") == 2
        assert node.properties.get("source_text") == "chunk text body"
