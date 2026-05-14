"""B2-active slice B: tasks.py disputed-pairs wiring contract.

Mirrors test_tasks_decision_log_wiring.py for the disputed-pairs
queue. The graph-builder-internal hook in slice B is wired behind a
``disputed_pairs_service`` parameter that defaults to None. With
None the helper returns early — which is what every legacy
construct_knowledge_graph caller would do until tasks.py is updated.

This module pins that tasks.py CONSTRUCTS a per-transform
``DisputedPairsService`` and threads it through to
``build_graph_from_chunks`` / ``build_graph_from_pdfs``. Without
this pin, the slice-B hook is dead code in production — a future
refactor that "cleans up an unused parameter" wouldn't be caught
by any other test.

Per-call instance (not module-global) so dev-mode runs don't share
state across transforms via the helper's enqueue path — same design
choice as the Decision Log.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from graphora_server.services.disputed_pairs_service import (
    DisputedPairsService,
)


@pytest.fixture(autouse=True)
def _force_memory_mode(monkeypatch):
    """conftest defaults DATABASE_URL for tests; switch to memory
    mode so the constructed DisputedPairsService doesn't try to
    talk to a real DB during init."""
    from graphora_server.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)


@pytest.fixture
def stub_graph():
    """Minimal DocumentKnowledgeGraph the build_graph_from_* mocks
    return so construct_knowledge_graph's downstream metric code
    doesn't blow up on .nodes / .relationships / .metrics reads."""
    from datetime import datetime, timezone
    from graphora_server.services.transform.models import (
        DocumentKnowledgeGraph,
        ExtractionMetrics,
    )

    return DocumentKnowledgeGraph(
        nodes=[],
        relationships=[],
        metrics=ExtractionMetrics(
            start_time=datetime.now(timezone.utc),
            merged_nodes=0,
        ),
    )


@pytest.mark.asyncio
async def test_chunks_path_passes_disputed_pairs_service(stub_graph):
    """Pin: construct_knowledge_graph constructs a
    DisputedPairsService and forwards it as
    ``disputed_pairs_service=<instance>`` to build_graph_from_chunks.

    Pre-slice-B this kwarg was missing and every transform ran with
    disputed_pairs_service=None — the gray-zone hook lived inside
    _compare_and_merge_nodes but the service was always None so
    the helper was a no-op in production.
    """
    from graphora_server.services.transform import tasks

    captured: dict = {}

    async def fake_build_chunks(**kwargs):
        captured["disputed_pairs_service"] = kwargs.get("disputed_pairs_service")
        captured["transform_id"] = kwargs.get("transform_id")
        return stub_graph

    with (
        patch.object(
            tasks,
            "build_graph_from_chunks",
            new=AsyncMock(side_effect=fake_build_chunks),
        ),
        patch.object(
            tasks,
            "build_graph_from_pdfs",
            new=AsyncMock(),
        ),
        patch.object(
            tasks,
            "OntologyParser",
            new=lambda *_a, **_kw: object(),
        ),
        patch.object(
            tasks,
            "get_run_logger",
            new=lambda: __import__("logging").getLogger("test"),
        ),
    ):
        await tasks.construct_knowledge_graph.fn(
            ontology_path="ignored",
            transform_id="tx-chunks-1",
            chunks=["Alice joined Acme."],
            pdf_paths=[],
            user_id=None,
        )

    assert captured["transform_id"] == "tx-chunks-1"
    assert isinstance(captured["disputed_pairs_service"], DisputedPairsService), (
        "construct_knowledge_graph didn't pass a DisputedPairsService "
        "instance to build_graph_from_chunks. Without this wiring, "
        "slice B's gray-zone hook emits nothing in production — the "
        "disputed-pairs queue stays empty. "
        f"Got: {captured.get('disputed_pairs_service')!r}"
    )


@pytest.mark.asyncio
async def test_pdfs_path_passes_disputed_pairs_service(stub_graph):
    """Mirror pin for the PDF-binary path. Same contract: an
    instance must be threaded, not None."""
    from graphora_server.services.transform import tasks

    captured: dict = {}

    async def fake_build_pdfs(**kwargs):
        captured["disputed_pairs_service"] = kwargs.get("disputed_pairs_service")
        captured["transform_id"] = kwargs.get("transform_id")
        return stub_graph

    with (
        patch.object(
            tasks,
            "build_graph_from_chunks",
            new=AsyncMock(),
        ),
        patch.object(
            tasks,
            "build_graph_from_pdfs",
            new=AsyncMock(side_effect=fake_build_pdfs),
        ),
        patch.object(
            tasks,
            "OntologyParser",
            new=lambda *_a, **_kw: object(),
        ),
        patch.object(
            tasks,
            "get_run_logger",
            new=lambda: __import__("logging").getLogger("test"),
        ),
    ):
        await tasks.construct_knowledge_graph.fn(
            ontology_path="ignored",
            transform_id="tx-pdfs-1",
            chunks=[],
            pdf_paths=["/tmp/page_1.pdf"],
            user_id=None,
        )

    assert captured["transform_id"] == "tx-pdfs-1"
    assert isinstance(captured["disputed_pairs_service"], DisputedPairsService), (
        "construct_knowledge_graph didn't pass a DisputedPairsService "
        "instance to build_graph_from_pdfs. Without this wiring, "
        "slice B's gray-zone hook emits nothing on the PDF path."
    )


@pytest.mark.asyncio
async def test_each_transform_gets_its_own_disputed_pairs_service(stub_graph):
    """Per-call instance, not module-global. Two transforms in the
    same process must NOT share a DisputedPairsService instance.

    The in-memory backend uses a SHARED module-level store
    (_DEFAULT_MEMORY_STORE) so cross-instance reads work in dev
    mode, but the SERVICE instance itself stays per-transform to
    keep the construct_knowledge_graph signature clean and avoid
    accidental state leaks through any future per-instance fields
    (e.g., per-transform write batching)."""
    from graphora_server.services.transform import tasks

    seen_services: list = []

    async def fake_build_chunks(**kwargs):
        seen_services.append(kwargs.get("disputed_pairs_service"))
        return stub_graph

    with (
        patch.object(
            tasks,
            "build_graph_from_chunks",
            new=AsyncMock(side_effect=fake_build_chunks),
        ),
        patch.object(
            tasks,
            "OntologyParser",
            new=lambda *_a, **_kw: object(),
        ),
        patch.object(
            tasks,
            "get_run_logger",
            new=lambda: __import__("logging").getLogger("test"),
        ),
    ):
        await tasks.construct_knowledge_graph.fn(
            ontology_path="ignored",
            transform_id="tx-a",
            chunks=["chunk-a"],
            pdf_paths=[],
            user_id=None,
        )
        await tasks.construct_knowledge_graph.fn(
            ontology_path="ignored",
            transform_id="tx-b",
            chunks=["chunk-b"],
            pdf_paths=[],
            user_id=None,
        )

    assert len(seen_services) == 2
    assert isinstance(seen_services[0], DisputedPairsService)
    assert isinstance(seen_services[1], DisputedPairsService)
    assert seen_services[0] is not seen_services[1], (
        "Two transforms shared the same DisputedPairsService "
        "instance. The design choice is per-call construction so "
        "any future per-instance state (write batching, request "
        "context) doesn't leak across transforms."
    )
