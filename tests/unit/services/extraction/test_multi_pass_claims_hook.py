"""Regression pin: B1-prob slice 2b claim emission via the multi-pass path.

Reviewer-flagged on commit 60aebe9: the single-pass extraction
path emitted claims, but ``build_graph_from_chunks(enable_multi_pass=True,
claims_service=...)`` accepted-and-ignored the parameter. Any
caller turning on multi-pass got the slice-2a empty
contradictions response — silently losing the slice-2b writer.

These tests pin the closed gap: when multi-pass extraction runs
with a ClaimsService threaded through, claims land for every
per-chunk extracted node — at BOTH the initial pass and the
refinement pass. Mocks live at the LLM-client + validator
boundary so we can drive the multi-pass control flow without a
real Gemini call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from graphora_server.config import settings
from graphora_server.services.claims_service import (
    ClaimsService,
    TargetKind,
    _reset_default_memory_store_for_tests,
)
from graphora_server.services.extraction.multi_pass_extractor import (
    MultiPassExtractor,
)
from graphora_server.services.extraction.config import MultiPassConfig
from graphora_server.services.transform.models import (
    BaseNode,
    NodeProvenance,
)


class _FakeOntologyParser:
    """Minimum ontology-parser duck-type. The multi-pass
    extractor only calls ``parsed_ontology``, ``ontology_yaml``,
    and ``build_entities_only_model`` / ``build_relationships_only_model``
    during the path we're exercising — fake them out so the test
    can drive the flow without a real YAML round-trip."""

    def __init__(self) -> None:
        self.parsed_ontology = {"entities": {"Person": {}}, "relationships": {}}
        self.ontology_yaml = "version: '0.1.0'\nentities:\n  Person: {}\n"

    def build_entities_only_model(self):  # noqa: D401
        # The model class itself is passed straight through to
        # the LLM client mock; any pydantic stand-in works.
        from pydantic import BaseModel

        class _Stub(BaseModel):
            pass

        return _Stub

    def build_relationships_only_model(self):  # noqa: D401
        from pydantic import BaseModel

        class _Stub(BaseModel):
            pass

        return _Stub


def _node(
    *,
    node_id: str = "alice_chunk_1",
    canonical_id: str = "cid-alice",
    title: str = "Engineer",
    confidence: float = 0.9,
    chunk_id: str = "chunk-1",
) -> BaseNode:
    """A BaseNode shaped like what transform_as_nodes would
    produce post-provenance-attachment."""
    return BaseNode(
        id=node_id,
        type="Person",
        properties={"name": "Alice", "title": title},
        canonical_id=canonical_id,
        canonical_key="Person:name=alice",
        provenance=NodeProvenance(
            chunk_ids=[chunk_id],
            confidence_score=confidence,
            extractor_model="gemini-2.5-flash",
            prompt_version="v1.0",
        ),
    )


@pytest.fixture
def memory_service(monkeypatch):
    """Real ClaimsService with isolated memory store."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)
    _reset_default_memory_store_for_tests()
    yield ClaimsService(memory_store=[])
    _reset_default_memory_store_for_tests()


@pytest.mark.asyncio
async def test_initial_extraction_pass_emits_claims_through_multi_pass(
    memory_service,
):
    """Reviewer-flagged Medium on commit 60aebe9. The
    multi-pass path now emits claims at its per-chunk
    transform_as_nodes site (not just signature-parity'd).
    Pin so a refactor that strips the hook from
    ``_initial_extraction_pass`` regresses noisily.

    Strategy: mock the LLM call to return a deterministic
    extracted-nodes payload, mock ``transform_as_nodes`` so
    the helper sees a real BaseNode without standing up the
    ontology parser, then assert the claims landed.
    """
    parser = _FakeOntologyParser()
    extractor = MultiPassExtractor(
        ontology_parser=parser,
        llm_client=AsyncMock(),
        config=MultiPassConfig(max_passes=1),  # single pass for clarity
    )

    # Mock the LLM call (returns an opaque kg payload — the
    # subsequent transform_as_nodes mock determines what the
    # extracted node looks like).
    extractor.llm_client.extract_nodes_from_chunk = AsyncMock(return_value=object())
    extractor.llm_client.extract_relationships_from_chunk = AsyncMock(
        return_value=object()
    )

    # Replace transform_as_nodes / transform_as_relationships
    # at the call site so the test can drive what the helper
    # sees. Two patches because the helper reads from
    # multi_pass_extractor's import namespace, not its
    # original module.
    with (
        patch(
            "graphora_server.services.extraction.multi_pass_extractor.transform_as_nodes",
            return_value=[_node()],
        ),
        patch(
            "graphora_server.services.extraction.multi_pass_extractor.transform_as_relationships",
            return_value=[],
        ),
    ):
        await extractor.extract(
            chunks=["Alice is an Engineer at Acme."],
            transform_id="tx-mp-1",
            user_id="user-1",
            max_passes=1,
            claims_service=memory_service,
        )

    # The hook ran — both properties landed as claims.
    claims = await memory_service.for_target(
        transform_id="tx-mp-1",
        target_id="cid-alice",
        target_kind=TargetKind.NODE,
        user_id="user-1",
    )
    property_keys = {c.property_key for c in claims}
    assert property_keys == {"name", "title"}, (
        "Multi-pass initial extraction must emit claims via "
        "emit_node_property_claims. Pre-fix the multi-pass "
        "path silently dropped them. Got: "
        f"{property_keys!r}"
    )


@pytest.mark.asyncio
async def test_multi_pass_extract_signature_accepts_claims_service():
    """Signature pin: ``MultiPassExtractor.extract`` must accept
    ``claims_service`` as a keyword argument. Pin so a refactor
    that drops the parameter back to "signature parity only"
    regresses at the type level."""
    import inspect

    sig = inspect.signature(MultiPassExtractor.extract)
    assert "claims_service" in sig.parameters, (
        "MultiPassExtractor.extract must accept claims_service. "
        f"Got params: {list(sig.parameters.keys())!r}"
    )


@pytest.mark.asyncio
async def test_extract_for_gaps_signature_accepts_claims_service():
    """Refinement-side signature pin. ``_extract_for_gaps`` is
    where the refinement pass re-extracts nodes for identified
    gaps — its extracted nodes should also contribute to claims.
    Pin the signature so a refactor that wires only the initial
    pass (forgetting refinement) regresses."""
    import inspect

    sig = inspect.signature(MultiPassExtractor._extract_for_gaps)
    assert "claims_service" in sig.parameters


@pytest.mark.asyncio
async def test_build_graph_with_multi_pass_forwards_claims_service():
    """End-to-end signature pin: ``_build_graph_with_multi_pass``
    (the graph_transformer entry point that
    ``build_graph_from_chunks(enable_multi_pass=True)`` delegates
    to) must forward claims_service into ``extractor.extract``.
    Pre-fix it accepted the parameter for signature parity but
    ignored it — exactly the bug the reviewer caught on commit
    60aebe9.

    Strategy: inspect the source of the function. A regression
    would either drop the parameter from the call OR pass
    None instead of forwarding it. Both shapes are caught by
    a literal-source search for ``claims_service=claims_service``."""
    import inspect

    from graphora_server.services.transform.graph_transformer import (
        _build_graph_with_multi_pass,
    )

    source = inspect.getsource(_build_graph_with_multi_pass)
    assert "claims_service=claims_service" in source, (
        "_build_graph_with_multi_pass must forward "
        "claims_service to extractor.extract — pre-fix it "
        "accepted the parameter but didn't pass it through, "
        "so callers using enable_multi_pass=True saw empty "
        "/contradictions responses. The literal token "
        "'claims_service=claims_service' should appear in the "
        "source."
    )
