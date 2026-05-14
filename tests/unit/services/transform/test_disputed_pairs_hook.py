"""B2-active slice B: gray-zone disputed-pair hook contract.

Slice A built the disputed-pairs queue + API surface, but with no
producer wired in — the queue was dead in production. Slice B
wires a write hook inside ``_compare_and_merge_nodes`` so that when
a blocker (property / embedding / Splink) flags a 2-node candidate
and the LLM resolver returns them as singletons (no merge), the
pair is enqueued for human/agent review.

The "all-singletons after 2-node candidate" predicate is the
clearest "blocker said yes, LLM said no" signal — that's what
slice B captures. Larger candidate groups can produce combinatorial
disputed pairs and are deferred to a future slice; this test
module pins the helper's predicates so a future refactor that
loosens or tightens them is forced to update the tests
intentionally.

Failures during enqueue are logged-and-swallowed because the
disputed-pairs queue is OBSERVABILITY (the merge pipeline already
committed to the LLM verdict). Tests pin that swallowing — a
future refactor that "cleans up" the broad except would silently
turn pipeline observability into a pipeline-blocker.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphora_server.services.disputed_pairs_service import (
    DisputedPair,
    SourceStage,
)
from graphora_server.services.transform.graph_transformer import (
    _enqueue_disputed_pair_if_unresolved,
)
from tests.factories.node_factory import NodeFactory


@pytest.fixture(autouse=True)
def _reset_factory():
    NodeFactory.reset_counter()


@pytest.fixture
def two_person_candidate():
    """A typical gray-zone candidate: two Person nodes that a
    blocker grouped (e.g., same canonical_key 'alice') but might
    or might not be the same entity."""
    return [
        NodeFactory.create_person(name="Alice Smith", node_id="n-1"),
        NodeFactory.create_person(name="Alice Smyth", node_id="n-2"),
    ]


@pytest.fixture
def all_singletons(two_person_candidate):
    """LLM returned both nodes as singletons — no merge happened.
    This is the 'blocker said yes, LLM said no' signal worth
    enqueuing."""
    return [[two_person_candidate[0]], [two_person_candidate[1]]]


@pytest.mark.asyncio
async def test_legacy_caller_with_none_service_is_noop(
    two_person_candidate, all_singletons
):
    """The hook MUST be a no-op when ``disputed_pairs_service`` is
    None. Without this, every existing test/production call path
    that hasn't been updated would start failing — the hook is
    additive, not mandatory."""
    # Should not raise. We can't directly observe "did nothing"
    # without a service, but reaching the function-level early
    # return is the contract: no AttributeError, no exception.
    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=None,
        user_id="user-1",
        transform_id="tx-1",
        candidate_group=two_person_candidate,
        resolved_groups=all_singletons,
        source_stage=SourceStage.PROPERTY_BLOCKER,
    )


@pytest.mark.asyncio
async def test_missing_user_id_short_circuits(two_person_candidate, all_singletons):
    """Tenant-scoping pin: the disputed-pairs table has a NOT NULL
    user_id constraint. Without a tenant we cannot enqueue —
    short-circuit BEFORE touching the service. Mirrors the safe-default
    pattern used in _emit_entity_merged_decision."""
    service = AsyncMock()
    service.enqueue = AsyncMock()

    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=service,
        user_id=None,
        transform_id="tx-1",
        candidate_group=two_person_candidate,
        resolved_groups=all_singletons,
        source_stage=SourceStage.PROPERTY_BLOCKER,
    )

    service.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_transform_id_short_circuits(
    two_person_candidate, all_singletons
):
    """The transform_id is the foreign-key/correlation handle that
    lets the per-transform review surface fetch the queue for one
    run. Missing it makes the row orphaned — short-circuit."""
    service = AsyncMock()
    service.enqueue = AsyncMock()

    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=service,
        user_id="user-1",
        transform_id=None,
        candidate_group=two_person_candidate,
        resolved_groups=all_singletons,
        source_stage=SourceStage.PROPERTY_BLOCKER,
    )

    service.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_node_all_singletons_triggers_enqueue(
    two_person_candidate, all_singletons
):
    """Happy path: 2-node candidate + all-singletons resolution =
    enqueue. The dispatched DisputedPair carries user_id,
    transform_id, both node ids/canonical_keys, the entity_type
    from node_a, and the caller-supplied source_stage.

    similarity_score is None on the wire: each blocker stage uses
    a different scoring system (canonical_key match vs cosine vs
    m/u), so picking one stage's number would surface as misleading
    for cross-stage queues. Pin via assertion."""
    service = AsyncMock()
    service.enqueue = AsyncMock()

    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=service,
        user_id="user-1",
        transform_id="tx-abc",
        candidate_group=two_person_candidate,
        resolved_groups=all_singletons,
        source_stage=SourceStage.EMBEDDING_BLOCKER,
    )

    service.enqueue.assert_awaited_once()
    pair: DisputedPair = service.enqueue.await_args.args[0]
    assert pair.user_id == "user-1"
    assert pair.transform_id == "tx-abc"
    assert pair.node_a_id == "n-1"
    assert pair.node_b_id == "n-2"
    assert pair.entity_type == "Person"
    assert pair.source_stage == SourceStage.EMBEDDING_BLOCKER
    assert pair.similarity_score is None, (
        "similarity_score must be None — each blocker stage uses "
        "a different scoring system, picking one stage's number "
        "across the cross-stage queue would mislead reviewers."
    )
    assert pair.node_a_canonical_key is not None
    assert pair.node_b_canonical_key is not None


@pytest.mark.asyncio
async def test_three_node_candidate_does_not_trigger():
    """Slice B keeps the trigger tight: only 2-node candidate groups
    enqueue. Larger groups can produce combinatorial disputed
    pairs (3 nodes → 3 pairs) and the UX/agent surface for
    multi-pair groups isn't designed in slice B. Pin so a future
    'just enqueue everything' refactor surfaces the design decision
    explicitly."""
    nodes = [
        NodeFactory.create_person(name="Alice", node_id="n-1"),
        NodeFactory.create_person(name="Alicia", node_id="n-2"),
        NodeFactory.create_person(name="Aly", node_id="n-3"),
    ]
    resolved_all_singletons = [[nodes[0]], [nodes[1]], [nodes[2]]]
    service = AsyncMock()
    service.enqueue = AsyncMock()

    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=service,
        user_id="user-1",
        transform_id="tx-1",
        candidate_group=nodes,
        resolved_groups=resolved_all_singletons,
        source_stage=SourceStage.SPLINK_BLOCKER,
    )

    service.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_node_merged_pair_does_not_trigger(two_person_candidate):
    """When the LLM resolver MERGED the 2-node candidate (returned
    one group of two), the blocker and LLM agreed — no dispute,
    no enqueue. Pin the 'all-singletons' predicate so a future
    refactor that drops the singleton check doesn't start
    enqueueing every blocker hit (which would drown reviewers)."""
    merged_group = [two_person_candidate]  # both nodes in one group
    service = AsyncMock()
    service.enqueue = AsyncMock()

    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=service,
        user_id="user-1",
        transform_id="tx-1",
        candidate_group=two_person_candidate,
        resolved_groups=merged_group,
        source_stage=SourceStage.PROPERTY_BLOCKER,
    )

    service.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_resolution_does_not_trigger():
    """Hypothetical: 2-node candidate, LLM returns one merge group
    of size 1 + one of size 0 (degenerate but defensive). Not all
    singletons → no enqueue. Pin so the resolver contract can
    surface weird edge cases without polluting the queue."""
    nodes = [
        NodeFactory.create_person(name="Alice", node_id="n-1"),
        NodeFactory.create_person(name="Alicia", node_id="n-2"),
    ]
    # Two groups but the second has 2 nodes (i.e. the second group
    # is a merge). This is "merged group + singleton" — not all
    # singletons, no dispute.
    weird_resolution = [
        [nodes[0]],
        [nodes[1], NodeFactory.create_person(name="Phantom", node_id="n-3")],
    ]
    service = AsyncMock()
    service.enqueue = AsyncMock()

    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=service,
        user_id="user-1",
        transform_id="tx-1",
        candidate_group=nodes,
        resolved_groups=weird_resolution,
        source_stage=SourceStage.PROPERTY_BLOCKER,
    )

    service.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_failure_is_swallowed(
    two_person_candidate, all_singletons, caplog
):
    """The disputed-pairs queue is OBSERVABILITY for ER (the merge
    pipeline already committed to the LLM's verdict). A failure
    in enqueue MUST NOT propagate and abort the transform — that
    would turn observability into a correctness blocker. Tests
    pin the swallowing so a future refactor that 'cleans up the
    broad except' surfaces the design intent.

    Mirror of the same observability-vs-correctness split applied
    to DecisionLogService.append earlier (B0-log slice 2)."""
    service = MagicMock()
    service.enqueue = AsyncMock(side_effect=RuntimeError("database down"))

    # Should not raise — caller is inside the merge pipeline.
    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=service,
        user_id="user-1",
        transform_id="tx-1",
        candidate_group=two_person_candidate,
        resolved_groups=all_singletons,
        source_stage=SourceStage.PROPERTY_BLOCKER,
    )

    # The failure should be visible in logs (observability lives in
    # the warning, not the swallowing) — so reviewers can see when
    # the queue is dropping rows.
    assert any(
        "Failed to enqueue disputed pair" in rec.message for rec in caplog.records
    ), "Swallowed enqueue failures must still be logged as warnings"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        SourceStage.PROPERTY_BLOCKER,
        SourceStage.EMBEDDING_BLOCKER,
        SourceStage.SPLINK_BLOCKER,
    ],
)
async def test_source_stage_is_threaded_unchanged(
    two_person_candidate, all_singletons, stage
):
    """The caller-supplied source_stage is preserved verbatim on
    the enqueued pair. The hook lives in 3 different sites
    (property/embedding/splink blocker loops) and the stage tag
    is how reviewers filter the queue by which blocker raised
    each pair — losing the tag would collapse the diagnostic
    surface."""
    service = AsyncMock()
    service.enqueue = AsyncMock()

    await _enqueue_disputed_pair_if_unresolved(
        disputed_pairs_service=service,
        user_id="user-1",
        transform_id="tx-1",
        candidate_group=two_person_candidate,
        resolved_groups=all_singletons,
        source_stage=stage,
    )

    pair: DisputedPair = service.enqueue.await_args.args[0]
    assert pair.source_stage == stage
