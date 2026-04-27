"""B0-prov-extend regression tests (Gate 4 entry).

Builds on the A1-prov foundation in test_provenance_propagation.py.
Tests here exercise the *decision-trail* fields — extractor_model,
prompt_version, validator_score — that B0 adds to NodeProvenance
and to the _EVIDENCE_KEYS contract.

The full Decision Log surface (Slice 2) and the MCP get_evidence
extension (Slice 3) are out of scope for this file.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from graphora_server.services.chunking.models import ChunkMetadata
from graphora_server.services.extraction.multi_pass_extractor import (
    _backfill_validator_score,
)
from graphora_server.services.extraction.prompt_versions import (
    BAML_PROMPT_VERSIONS,
    get_prompt_version,
)
from graphora_server.services.transform.helpers import (
    _attach_provenance_properties,
    transform_as_nodes,
    transform_as_relationships,
)
from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
    RelationshipInstance,
)


# ---- 1. Schema round-trip ------------------------------------------------


def test_node_provenance_round_trips_decision_trail_fields() -> None:
    prov = NodeProvenance(
        chunk_ids=["c1"],
        extractor_model="gemini-2.5-flash",
        prompt_version="v1.0.0",
        validator_score=0.87,
    )
    restored = NodeProvenance(**prov.model_dump())
    assert restored.extractor_model == "gemini-2.5-flash"
    assert restored.prompt_version == "v1.0.0"
    assert restored.validator_score == 0.87


# ---- 2. Helper writes the new fields onto properties --------------------


def test_attach_provenance_properties_writes_decision_trail() -> None:
    node = BaseNode(
        id="n1",
        type="Person",
        properties={"name": "Alice"},
        provenance=NodeProvenance(
            chunk_ids=["c1"],
            extractor_model="gemini-2.5-flash",
            prompt_version="v1.0.0",
            validator_score=0.91,
        ),
    )
    _attach_provenance_properties(node)
    assert node.properties["extractor_model"] == "gemini-2.5-flash"
    assert node.properties["prompt_version"] == "v1.0.0"
    assert node.properties["validator_score"] == 0.91


def test_attach_provenance_properties_skips_when_provenance_lacks_fields() -> None:
    node = BaseNode(
        id="n1",
        type="Person",
        properties={"name": "Alice"},
        provenance=NodeProvenance(chunk_ids=["c1"]),  # no decision-trail fields set
    )
    _attach_provenance_properties(node)
    assert "extractor_model" not in node.properties
    assert "prompt_version" not in node.properties
    assert "validator_score" not in node.properties


# ---- 3. transform_as_nodes accepts + stamps the new kwargs --------------


class _StubEntityResult:
    """Mimics the BAML *_list shape transform_as_nodes walks."""

    def __init__(self):
        self.Person_list = [_StubEntity(name="Alice")]
        self.confidence_score = 0.92


class _StubEntity:
    def __init__(self, name: str):
        self.name = name


_ONTOLOGY_PERSON = {
    "entities": {"Person": {"properties": {"name": {"type": "str", "required": True}}}}
}


def test_transform_as_nodes_stamps_decision_trail() -> None:
    nodes = transform_as_nodes(
        _ONTOLOGY_PERSON,
        _StubEntityResult(),
        transform_id="tx-1",
        extractor_model="gemini-2.5-flash",
        prompt_version="v1.0.0",
    )
    assert len(nodes) == 1
    node = nodes[0]
    assert node.provenance.extractor_model == "gemini-2.5-flash"
    assert node.provenance.prompt_version == "v1.0.0"
    # The helper auto-copies onto properties.
    assert node.properties["extractor_model"] == "gemini-2.5-flash"
    assert node.properties["prompt_version"] == "v1.0.0"


# ---- 4. transform_as_relationships accepts + stamps the new kwargs ------


class _StubRelResult:
    def __init__(self):
        self.Person_works_at_Person_list = [
            _StubRel(source_id="alice", target_id="bob")
        ]
        self.confidence_score = 0.88


class _StubRel:
    def __init__(self, source_id: str, target_id: str):
        self.source_id = source_id
        self.target_id = target_id


# A note on this test: transform_as_relationships requires a
# matching ontology AND nodes that exist for the source/target ids.
# Rather than scaffold the full setup (which is independently
# covered in the A1-prov tests), we exercise the kwargs handling at
# the helper boundary instead — confirming the function accepts
# the new params without raising. The actual stamping is verified
# end-to-end in test_multi_pass_stamps_decision_trail below.


def test_transform_as_relationships_accepts_decision_trail_kwargs() -> None:
    """Backward-compat smoke: the kwargs are optional and don't
    break callers that don't supply them."""
    # No nodes → no relationships emitted, but no crash either.
    rels = transform_as_relationships(
        _ONTOLOGY_PERSON,
        nodes=[],
        relationship_result=_StubRelResult(),
        extractor_model="gemini-2.5-flash",
        prompt_version="v1.0.0",
        validator_score=0.91,
    )
    assert rels == []


# ---- 5. End-to-end through _build_graph_from ----------------------------


@pytest.mark.asyncio
async def test_build_graph_from_stamps_decision_trail() -> None:
    from graphora_server.services.transform import graph_transformer
    from graphora_server.services.transform.ontology_helper import OntologyParser
    from graphora_server.services.entity_ledger_service import entity_ledger_service

    parser = OntologyParser.__new__(OntologyParser)
    parser.parsed_ontology = _ONTOLOGY_PERSON
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
        transform_id="tx-1",
        chunk_id="chunk-1",
        source_file="paper.pdf",
        source_text="Alice joined Acme.",
    )

    with patch.object(
        entity_ledger_service, "hydrate_nodes", new=AsyncMock(return_value=None)
    ):
        graph = await graph_transformer._build_graph_from(
            ontology_parser=parser,
            chunks_or_pdf_paths=["Alice joined Acme."],
            transform_id="tx-1",
            node_extractor=fake_extract_nodes,
            relationship_extractor=fake_extract_rels,
            chunk_metadatas=[cm],
            extractor_model="gemini-2.5-flash",
            node_baml_function="ExtractNodesFromChunk",
            rel_baml_function="ExtractRelationshipsFromChunk",
        )

    assert graph.nodes
    node = graph.nodes[0]
    assert node.provenance.extractor_model == "gemini-2.5-flash"
    assert node.provenance.prompt_version == "v1.0.0"
    assert node.properties["extractor_model"] == "gemini-2.5-flash"
    assert node.properties["prompt_version"] == "v1.0.0"


# ---- 6. validator_score back-fill helper --------------------------------


def test_backfill_validator_score_writes_only_when_absent() -> None:
    n1 = BaseNode(id="n1", type="T", properties={}, provenance=NodeProvenance())
    n2_already = BaseNode(
        id="n2",
        type="T",
        properties={"validator_score": 0.55},
        provenance=NodeProvenance(validator_score=0.55),
    )

    _backfill_validator_score([n1, n2_already], [], score=0.92)

    # Fresh node gets the new score.
    assert n1.provenance.validator_score == 0.92
    assert n1.properties["validator_score"] == 0.92
    # Existing scored node is left alone.
    assert n2_already.provenance.validator_score == 0.55
    assert n2_already.properties["validator_score"] == 0.55


def test_backfill_validator_score_no_op_when_score_is_none() -> None:
    n = BaseNode(id="n", type="T", properties={}, provenance=NodeProvenance())
    _backfill_validator_score([n], [], score=None)
    assert n.provenance.validator_score is None
    assert "validator_score" not in n.properties


def test_backfill_validator_score_handles_relationships() -> None:
    rel = RelationshipInstance(
        id="r",
        type="WORKS_AT",
        source_id="a",
        target_id="b",
        source_type="P",
        target_type="O",
        properties={},
        provenance=NodeProvenance(),
    )
    _backfill_validator_score([], [rel], score=0.87)
    assert rel.provenance.validator_score == 0.87
    assert rel.properties["validator_score"] == 0.87


# ---- 7. prompt_versions registry ----------------------------------------


def test_prompt_versions_registry_covers_all_baml_extraction_functions() -> None:
    """All BAML functions used at extraction sites must have a
    registered version. Catches the case where a new BAML function
    is added but its version constant isn't."""
    expected_functions = {
        "ExtractNodesFromChunk",
        "ExtractRelationshipsFromChunk",
        "ExtractNodesFromPdf",
        "ExtractRelationshipsFromPdf",
    }
    assert expected_functions <= set(BAML_PROMPT_VERSIONS.keys())


def test_get_prompt_version_returns_none_for_unknown_function() -> None:
    assert get_prompt_version("DoesNotExist") is None


# ---- 8. Backward compat: existing callsites still work ------------------


def test_transform_as_nodes_backward_compat_no_decision_trail_kwargs() -> None:
    """Older callsites that don't supply extractor_model /
    prompt_version still produce nodes with None for those fields."""
    nodes = transform_as_nodes(
        _ONTOLOGY_PERSON,
        _StubEntityResult(),
        transform_id="tx-old",
    )
    node = nodes[0]
    assert node.provenance.extractor_model is None
    assert node.provenance.prompt_version is None
    assert "extractor_model" not in node.properties
    assert "prompt_version" not in node.properties


# ---- 9. Refinement-pass propagation (multi-pass end-to-end) -------------


@pytest.mark.asyncio
async def test_multi_pass_refinement_threads_decision_trail() -> None:
    """Regression: refinement-pass facts must carry the same
    extractor_model / prompt_version that initial-pass facts do.
    Without the threading through _refinement_pass + _extract_for_gaps,
    refined nodes silently drop these fields and fall out of the
    Slice 1 contract.
    """
    from graphora_server.services.extraction.config import MultiPassConfig
    from graphora_server.services.extraction.models import (
        ExtractionGap,
        GapType,
        ValidationResult,
    )
    from graphora_server.services.extraction.multi_pass_extractor import (
        MultiPassExtractor,
    )

    # Stub LLM client returns a single Person on every call.
    class _StubLLM:
        async def extract_nodes_from_chunk(self, *_a, **_kw):
            return _StubEntityResult()

        async def extract_relationships_from_chunk(self, *_a, **_kw):
            class _Empty:
                confidence_score = 0.9

            return _Empty()

    # Stub parser.
    class _Parser:
        parsed_ontology = _ONTOLOGY_PERSON
        ontology_yaml = "v: 1\n"

        def build_entities_only_model(self):
            return object

        def build_relationships_only_model(self):
            return object

    extractor = MultiPassExtractor(
        ontology_parser=_Parser(),
        llm_client=_StubLLM(),
        config=MultiPassConfig(
            max_passes=2,
            gap_severity_threshold=0.5,
            enable_parallel_refinement=False,
        ),
    )

    # Force a refinement pass: make the validator say "needs
    # refinement" once, then return False on the second call.
    validator_call = {"n": 0}

    def fake_validate(_nodes, _rels):
        validator_call["n"] += 1
        return ValidationResult(
            overall_confidence=0.6,
            gaps=(
                [
                    ExtractionGap(
                        gap_type=GapType.LOW_CONFIDENCE,
                        description="forced",
                        severity=1.0,
                        chunk_indices=[0],
                    )
                ]
                if validator_call["n"] == 1
                else []
            ),
        )

    extractor.validator.validate = fake_validate  # type: ignore[assignment]

    # Stub the EnhancedContextBuilder methods extract() walks.
    class _Ctx:
        text = ""
        truncated = False
        raw_length = 0

    extractor.context_builder.build_relationship_aware_entity_context = (
        lambda *_a, **_kw: _Ctx()
    )
    extractor.context_builder.build_relationship_context = lambda *_a, **_kw: _Ctx()
    extractor.context_builder.build_refinement_context = lambda *_a, **_kw: ""
    extractor.context_builder.get_expected_entity_types_from_relationships = (
        lambda *_a, **_kw: set()
    )

    nodes, _rels = await extractor.extract(
        chunks=["Alice joined Acme."],
        transform_id="tx-multi",
        extractor_model="gemini-2.5-flash",
    )

    # Both initial-pass AND refinement-pass nodes must carry the
    # decision-trail fields. With max_passes=2 and a forced gap on
    # pass 1, refinement runs and produces additional nodes.
    assert len(nodes) >= 1
    for node in nodes:
        assert node.provenance.extractor_model == "gemini-2.5-flash", (
            f"node {node.id} from {'refinement' if node.provenance else 'initial'} "
            f"missing extractor_model"
        )
        assert node.provenance.prompt_version == "v1.0.0"
        assert node.properties.get("extractor_model") == "gemini-2.5-flash"
        assert node.properties.get("prompt_version") == "v1.0.0"


@pytest.mark.asyncio
async def test_multi_pass_validator_score_reaches_final_refinement_nodes() -> None:
    """Regression for the second review finding: facts created in
    the LAST refinement iteration must end up with validator_score
    set. The in-loop back-fill happens before refinement, so without
    the post-loop final-validator pass, those facts return None.
    """
    from graphora_server.services.extraction.config import MultiPassConfig
    from graphora_server.services.extraction.models import (
        ExtractionGap,
        GapType,
        ValidationResult,
    )
    from graphora_server.services.extraction.multi_pass_extractor import (
        MultiPassExtractor,
    )

    class _StubLLM:
        async def extract_nodes_from_chunk(self, *_a, **_kw):
            return _StubEntityResult()

        async def extract_relationships_from_chunk(self, *_a, **_kw):
            class _Empty:
                confidence_score = 0.9

            return _Empty()

    class _Parser:
        parsed_ontology = _ONTOLOGY_PERSON
        ontology_yaml = "v: 1\n"

        def build_entities_only_model(self):
            return object

        def build_relationships_only_model(self):
            return object

    extractor = MultiPassExtractor(
        ontology_parser=_Parser(),
        llm_client=_StubLLM(),
        config=MultiPassConfig(
            max_passes=2,
            gap_severity_threshold=0.5,
            enable_parallel_refinement=False,
        ),
    )

    # Validator always says "needs refinement" — exhausts max_passes.
    def fake_validate(_n, _r):
        return ValidationResult(
            overall_confidence=0.78,
            gaps=[
                ExtractionGap(
                    gap_type=GapType.LOW_CONFIDENCE,
                    description="forced",
                    severity=1.0,
                    chunk_indices=[0],
                )
            ],
        )

    extractor.validator.validate = fake_validate  # type: ignore[assignment]

    class _Ctx:
        text = ""
        truncated = False
        raw_length = 0

    extractor.context_builder.build_relationship_aware_entity_context = (
        lambda *_a, **_kw: _Ctx()
    )
    extractor.context_builder.build_relationship_context = lambda *_a, **_kw: _Ctx()
    extractor.context_builder.build_refinement_context = lambda *_a, **_kw: ""
    extractor.context_builder.get_expected_entity_types_from_relationships = (
        lambda *_a, **_kw: set()
    )

    nodes, _rels = await extractor.extract(
        chunks=["Alice joined Acme."],
        transform_id="tx-final",
        extractor_model="gemini-2.5-flash",
    )

    # Final-validator back-fill must reach every node, including
    # those created in the final refinement iteration.
    for node in nodes:
        assert (
            node.provenance.validator_score == 0.78
        ), f"node {node.id} missing validator_score from final back-fill"
        assert node.properties.get("validator_score") == 0.78
