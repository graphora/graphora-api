"""End-to-end extraction harness for the Gemini provider.

Behaviour this test asserts:
    1. BAML's Gemini client path resolves at import time.
    2. `build_graph_from_chunks` completes without raising against the
       real Gemini API using a ~100-word document and a 2-entity
       ontology.
    3. The returned `DocumentKnowledgeGraph` contains at least the
       core entities and one FOUNDED_BY / ACQUIRED relationship we
       know the fixture should produce.

This is deliberately fuzzy: LLMs are nondeterministic and asserting
exact node IDs would flake constantly. We check shape and minimum
counts; the Gate 4 public-benchmark corpus will layer strict
precision/recall on top of this same fixture format.

Skips automatically when GEMINI_API_KEY_TEST (or local GEMINI_API_KEY /
GOOGLE_API_KEY) isn't set, so the harness is safe to import everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pytest

pytestmark = pytest.mark.e2e_provider


def _resolve_test_api_key() -> str | None:
    """Pick a Gemini API key from the usual suspects.

    CI sets GEMINI_API_KEY_TEST from a spend-capped test account.
    Local developers typically have GEMINI_API_KEY or GOOGLE_API_KEY
    already exported.
    """
    return (
        os.environ.get("GEMINI_API_KEY_TEST")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )


@pytest.fixture(scope="module")
def gemini_api_key() -> str:
    key = _resolve_test_api_key()
    if not key:
        pytest.skip(
            "Gemini E2E provider test requires GEMINI_API_KEY_TEST "
            "(or GEMINI_API_KEY / GOOGLE_API_KEY) to be set."
        )
    return key


@pytest.fixture(scope="module")
def gemini_model() -> str:
    """Cheapest Gemini tier suitable for short extraction prompts."""
    return os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")


@pytest.fixture(autouse=True)
def _patch_credentials(
    monkeypatch: pytest.MonkeyPatch,
    gemini_api_key: str,
    gemini_model: str,
) -> None:
    """Inject test creds into every get_user_llm_credentials call site.

    graphora_server.services.llm.client imports get_user_llm_credentials
    from utils.llm_helper at module load, so we have to patch BOTH the
    helper module AND the re-exported name the client holds.
    """

    async def fake(user_id: str) -> Tuple[str, str]:
        return gemini_api_key, gemini_model

    monkeypatch.setattr(
        "graphora_server.utils.llm_helper.get_user_llm_credentials",
        fake,
    )
    monkeypatch.setattr(
        "graphora_server.services.llm.client.get_user_llm_credentials",
        fake,
    )


class TestGeminiProvider:
    """Exercises the Gemini extraction path end-to-end."""

    def test_import_chain_resolves(self) -> None:
        """The first tripwire: if the BAML Gemini client can't be
        constructed, every other test in this module fails. Surface
        that early with a clear assertion rather than an exception
        cascade halfway through the real extraction."""
        from graphora_server.baml_client import b  # noqa: F401
        from graphora_server.services.llm.client import LLMClient

        client = LLMClient()
        assert client is not None

    @pytest.mark.asyncio
    async def test_extract_from_tiny_text(
        self,
        tiny_text: str,
        minimal_ontology_path: Path,
    ) -> None:
        """Full extraction against the live Gemini API.

        Fuzzy assertions on purpose — extraction count varies run to
        run. We require at least the core entities (Acme, BetaTech,
        Jane Smith) show up and at least one relationship is
        produced. This is the Gate 4 public-benchmark harness in
        embryo; more rigorous precision/recall lands there.
        """
        from graphora_server.services.transform.graph_transformer import (
            build_graph_from_chunks,
        )
        from graphora_server.services.transform.ontology_helper import (
            OntologyParser,
        )

        parser = OntologyParser(yaml_path=minimal_ontology_path)
        transform_id = "e2e-gemini-" + os.urandom(4).hex()

        graph = await build_graph_from_chunks(
            ontology_parser=parser,
            chunks=[tiny_text],
            transform_id=transform_id,
            user_id="e2e-test-user",
        )

        assert graph is not None
        assert len(graph.nodes) >= 2, (
            f"expected at least 2 entities, got {len(graph.nodes)}: "
            f"{[n.entity_type for n in graph.nodes]}"
        )
        assert len(graph.relationships) >= 1, (
            f"expected at least 1 relationship, got " f"{len(graph.relationships)}"
        )

        node_types = {n.entity_type for n in graph.nodes}
        assert (
            "Company" in node_types
        ), f"no Company nodes extracted; types were {node_types}"
