"""B0-log slice 3b: tasks.py wiring contract.

Slice 2 hooked the entity-merge sites in _compare_and_merge_nodes
behind a ``decision_log: Optional[DecisionLogService]`` parameter
that defaulted to None. With None, no decisions are emitted — which
is what every production call path was doing because tasks.py never
constructed the service.

Slice 3b makes ``construct_knowledge_graph`` (the Prefect task that
fronts every transform run) construct one ``DecisionLogService`` per
transform and thread it through ``build_graph_from_chunks`` /
``build_graph_from_pdfs``. Per-call instance, not module-global —
no cross-transform state leak, garbage-collected when the task ends.

Tests pin the wiring directly: when ``construct_knowledge_graph``
runs, the public extractor it calls MUST receive a
``DecisionLogService`` (not None). Without this pin, slice 2's hooks
are dead code in production — a future refactor that "cleans up an
unused parameter" wouldn't be caught by any other test.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from graphora_server.services.decision_log_service import DecisionLogService


@pytest.fixture(autouse=True)
def _force_memory_mode(monkeypatch):
    """conftest defaults DATABASE_URL for tests; switch to memory
    mode so the constructed DecisionLogService doesn't try to talk
    to a real DB during init or via append calls inside the hooked
    merge logic."""
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
async def test_chunks_path_passes_decision_log_to_build_graph_from_chunks(
    stub_graph,
):
    """Pin: construct_knowledge_graph constructs a DecisionLogService
    and forwards it as ``decision_log=<instance>`` to
    build_graph_from_chunks. Pre-slice-3b this kwarg was missing and
    every transform ran with decision_log=None — slice 2's hooks
    were dead code in production."""
    from graphora_server.services.transform import tasks

    captured: dict = {}

    async def fake_build_chunks(**kwargs):
        captured["decision_log"] = kwargs.get("decision_log")
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
            new=AsyncMock(),  # Should not be called on the chunks path
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
        # ``.fn`` is the underlying coroutine, bypassing Prefect's
        # task-runtime requirements so the test runs as a plain
        # async unit test.
        await tasks.construct_knowledge_graph.fn(
            ontology_path="ignored",
            transform_id="tx-chunks-1",
            chunks=["Alice joined Acme."],
            pdf_paths=[],
            user_id=None,
        )

    assert captured["transform_id"] == "tx-chunks-1"
    assert isinstance(captured["decision_log"], DecisionLogService), (
        "construct_knowledge_graph didn't pass a DecisionLogService "
        "instance to build_graph_from_chunks. Without this wiring "
        "slice 2's entity-merge hooks emit nothing in production. "
        f"Got: {captured.get('decision_log')!r}"
    )


@pytest.mark.asyncio
async def test_pdfs_path_passes_decision_log_to_build_graph_from_pdfs(
    stub_graph,
):
    """Mirror pin for the PDF-binary path. Same contract: an instance
    must be threaded, not None."""
    from graphora_server.services.transform import tasks

    captured: dict = {}

    async def fake_build_pdfs(**kwargs):
        captured["decision_log"] = kwargs.get("decision_log")
        captured["transform_id"] = kwargs.get("transform_id")
        return stub_graph

    with (
        patch.object(
            tasks,
            "build_graph_from_chunks",
            new=AsyncMock(),  # Should not be called on the pdfs path
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
    assert isinstance(captured["decision_log"], DecisionLogService), (
        "construct_knowledge_graph didn't pass a DecisionLogService "
        "instance to build_graph_from_pdfs. Without this wiring "
        "slice 2's entity-merge hooks emit nothing on the PDF path. "
        f"Got: {captured.get('decision_log')!r}"
    )


@pytest.mark.asyncio
async def test_each_transform_gets_its_own_decision_log_instance(stub_graph):
    """Per-call instance, not module-global. Two transforms in the
    same process must NOT share a DecisionLogService — the entity_ledger
    is module-global because it's a thin wrapper around a connection
    pool, but the Decision Log holds per-transform memory state in
    dev mode and a shared instance would conflate decisions across
    runs."""
    from graphora_server.services.transform import tasks

    seen_logs: list = []

    async def fake_build_chunks(**kwargs):
        seen_logs.append(kwargs.get("decision_log"))
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

    assert len(seen_logs) == 2
    # Both must be services; both must be DIFFERENT instances.
    assert isinstance(seen_logs[0], DecisionLogService)
    assert isinstance(seen_logs[1], DecisionLogService)
    assert seen_logs[0] is not seen_logs[1], (
        "Two transforms shared the same DecisionLogService instance. "
        "Pre-slice-3b this would have been a module-global; the "
        "design choice is per-call construction so memory_store "
        "doesn't leak across transforms in dev mode."
    )
