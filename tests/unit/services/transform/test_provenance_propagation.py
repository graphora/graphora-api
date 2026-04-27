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
