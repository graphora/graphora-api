"""Regression tests for the partial-failure handling in
``store_knowledge_graph`` (services/storage/tasks.py).

The slice-3 review caught that ``StorageBatchResult.success`` was
ignored: a partial-failure batch (success=False, items_processed <
batch size) used to advance the checkpoint past the actual write
head, which is a silent resume / data-loss bug. The fix raises on
success=False and writes the checkpoint at items_processed instead
of the assumed batch tail.

These tests use mocked storage so the regression is pinned without
spinning up a real backend.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphora_server.services.storage.models import (
    StorageBatchResult,
    StorageError,
    StorageStage,
)
from graphora_server.services.storage.tasks import store_knowledge_graph
from graphora_server.services.transform.models import (
    BaseNode,
    DocumentKnowledgeGraph,
    RelationshipInstance,
)


def _node(name: str) -> BaseNode:
    return BaseNode(type="Person", properties={"name": name})


def _rel(rel_id_suffix: str) -> RelationshipInstance:
    return RelationshipInstance(
        type="WORKS_AT",
        source_id=f"s{rel_id_suffix}",
        target_id=f"t{rel_id_suffix}",
        source_type="Person",
        target_type="Company",
    )


def _make_storage_mock(
    *,
    store_nodes_result: StorageBatchResult,
    store_rel_result: StorageBatchResult = None,
):
    storage = MagicMock()
    storage.get_storage_status = AsyncMock(return_value=None)
    storage.store_nodes = AsyncMock(return_value=store_nodes_result)
    storage.store_relationships = AsyncMock(
        return_value=store_rel_result
        or StorageBatchResult(
            batch_index=0,
            items_processed=0,
            processing_time_ms=0.1,
            success=True,
        )
    )
    storage.update_checkpoint = AsyncMock()
    storage.get_transformation_data = AsyncMock(
        return_value=MagicMock(total_nodes=0, total_edges=0, nodes=[], edges=[])
    )
    storage.close = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_store_knowledge_graph_raises_on_node_batch_failure(monkeypatch):
    """A failed node batch must abort the task with StorageError —
    the previous behaviour silently advanced the checkpoint past the
    failed batch and continued to relationships, masking the data
    loss until resume."""

    failing = StorageBatchResult(
        batch_index=0,
        items_processed=2,
        processing_time_ms=1.0,
        success=False,
        error="bucket of Company nodes blew up",
        warnings=["Partial batch: 2 of 5 nodes stored"],
    )
    storage = _make_storage_mock(store_nodes_result=failing)

    async def fake_create(user_id, use_staging=True):
        return storage

    monkeypatch.setattr(
        "graphora_server.services.storage.tasks.create_storage_for_user",
        fake_create,
    )
    # Bypass Prefect's run-context requirement — get_run_logger()
    # raises MissingContextError outside a flow/task run, so swap
    # it for a stdlib logger.
    import logging as _logging

    monkeypatch.setattr(
        "graphora_server.services.storage.tasks.get_run_logger",
        lambda: _logging.getLogger("test"),
    )

    graph = DocumentKnowledgeGraph(
        nodes=[_node("a"), _node("b"), _node("c"), _node("d"), _node("e")],
        relationships=[],
    )
    with pytest.raises(StorageError, match="reported failure"):
        await store_knowledge_graph.fn(
            transform_id="t-42",
            graph=graph,
            user_id="u1",
            checkpoint_size=5,
        )

    # store_relationships must NOT be called once nodes failed.
    storage.store_relationships.assert_not_called()


@pytest.mark.asyncio
async def test_store_knowledge_graph_raises_on_relationship_batch_failure(
    monkeypatch,
):
    """Relationships fail the same way nodes do — partial failure
    must abort the task. Pinned separately because the relationship
    loop has its own copy of the checkpoint-advance code."""

    storage = _make_storage_mock(
        store_nodes_result=StorageBatchResult(
            batch_index=0,
            items_processed=1,
            processing_time_ms=0.5,
            success=True,
        ),
        store_rel_result=StorageBatchResult(
            batch_index=0,
            items_processed=0,
            processing_time_ms=0.5,
            success=False,
            error="rel batch raised mid-flight",
            warnings=["Partial batch: 0 of 1 relationships stored"],
        ),
    )

    async def fake_create(user_id, use_staging=True):
        return storage

    monkeypatch.setattr(
        "graphora_server.services.storage.tasks.create_storage_for_user",
        fake_create,
    )
    # Bypass Prefect's run-context requirement — get_run_logger()
    # raises MissingContextError outside a flow/task run, so swap
    # it for a stdlib logger.
    import logging as _logging

    monkeypatch.setattr(
        "graphora_server.services.storage.tasks.get_run_logger",
        lambda: _logging.getLogger("test"),
    )

    graph = DocumentKnowledgeGraph(
        nodes=[_node("a")],
        relationships=[_rel("1")],
    )
    with pytest.raises(StorageError, match="reported failure"):
        await store_knowledge_graph.fn(
            transform_id="t-42",
            graph=graph,
            user_id="u1",
            checkpoint_size=5,
        )


@pytest.mark.asyncio
async def test_store_knowledge_graph_advances_checkpoint_at_items_processed(
    monkeypatch,
):
    """When a batch succeeds with items_processed < len(batch), the
    checkpoint must advance to items_processed, not to the assumed
    batch tail. Pin the math so a future tweak doesn't reintroduce
    the off-by-batch resume bug.

    This test only exercises the success path with the new index
    math — the failure path is covered by the two tests above.
    """
    # 3 nodes in a single batch, all succeed. The checkpoint should
    # land at index 3 (items_processed) rather than 0 * 5 + 3 = 3 in
    # this case — but the math becomes load-bearing once a batch is
    # partial-success. Keep this test as the simple-case anchor.
    storage = _make_storage_mock(
        store_nodes_result=StorageBatchResult(
            batch_index=0,
            items_processed=3,
            processing_time_ms=0.5,
            success=True,
        )
    )

    async def fake_create(user_id, use_staging=True):
        return storage

    monkeypatch.setattr(
        "graphora_server.services.storage.tasks.create_storage_for_user",
        fake_create,
    )
    # Bypass Prefect's run-context requirement — get_run_logger()
    # raises MissingContextError outside a flow/task run, so swap
    # it for a stdlib logger.
    import logging as _logging

    monkeypatch.setattr(
        "graphora_server.services.storage.tasks.get_run_logger",
        lambda: _logging.getLogger("test"),
    )

    graph = DocumentKnowledgeGraph(
        nodes=[_node("a"), _node("b"), _node("c")],
        relationships=[],
    )
    await store_knowledge_graph.fn(
        transform_id="t-42",
        graph=graph,
        user_id="u1",
        checkpoint_size=5,
    )

    # Checkpoint calls: one per batch (1) + one between stages (1).
    # The first one is the per-batch checkpoint at items_processed.
    nodes_stage_calls = [
        c
        for c in storage.update_checkpoint.call_args_list
        if c.args[2] == StorageStage.NODES
    ]
    # First call is the per-batch advance.
    first_call = nodes_stage_calls[0]
    advanced_index = first_call.args[1]
    assert advanced_index == 3  # batch_idx (0) * 5 + items_processed (3)
