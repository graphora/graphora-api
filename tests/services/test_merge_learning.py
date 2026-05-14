import pytest

from graphora_server.services.merge.learning import MergeLearningService


@pytest.mark.asyncio
async def test_learning_returns_default_threshold_when_empty():
    service = MergeLearningService()
    assert await service.get_threshold("user-1", "Company", 0.95) == 0.95


@pytest.mark.asyncio
async def test_learning_adapts_threshold_downwards():
    service = MergeLearningService(alpha=0.5, margin=0.05)
    await service.record_outcome("user-2", "Company", [0.9, 0.88])

    adaptive = await service.get_threshold("user-2", "Company", 0.95)
    assert pytest.approx(adaptive, rel=1e-3) == 0.83


@pytest.mark.asyncio
async def test_learning_respects_default_ceiling():
    service = MergeLearningService(alpha=0.3)
    await service.record_outcome(None, "Company", [0.99, 0.97])

    adaptive = await service.get_threshold(None, "Company", 0.95)
    assert adaptive == 0.95


# ============================================================
# B2-active slice C (transition-aware after P2 fix on 72381b4):
# apply_pair_label is now keyed on a (old_status, new_status)
# transition rather than a single decision. The threshold delta
# = contribution(new) - contribution(old) makes:
#   * idempotent re-labels (same status both sides) no-ops
#   * match↔not_match swings carry the full 2x nudge magnitude
#   * match→skip undoes the prior contribution
# These tests pin those semantics.
# ============================================================


# Status string values — match what the disputed_pairs_service
# Status enum emits via .value. We deliberately use raw strings
# here to keep this unit test independent of disputed_pairs
# imports (apply_pair_label accepts Any and matches by .value).
_PENDING = "pending"
_LABELED_MATCH = "labeled_match"
_LABELED_NOT_MATCH = "labeled_not_match"
_SKIPPED = "skipped"


@pytest.mark.asyncio
async def test_apply_label_pending_to_match_bootstrap_from_empty():
    """First-ever label on a (user, type) slot. No prior stats,
    so seed = 1.0 (perfect-match prior) + delta. Pending → match
    delta = match_nudge (=-0.05 default) → seed = 0.95.

    Pin so a refactor that drops the bootstrap branch (which
    would crash on stats=None) is caught immediately."""
    service = MergeLearningService(label_match_nudge=-0.05)

    result = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_MATCH,
    )

    assert result == (1.0, 0.95)
    snapshot = service.snapshot()
    assert (snapshot[("user-1", "Company")].ema_low_score) == 0.95


@pytest.mark.asyncio
async def test_apply_label_pending_to_not_match_bootstrap_clamps_at_one():
    """Pending → not_match on empty stats: seed = 1.0 + nudge
    → clamped at 1.0 (we can't exceed perfect-match prior). Pin
    the ceiling clamp so a refactor that drops it doesn't
    quietly let the threshold drift > 1.0."""
    service = MergeLearningService(label_not_match_nudge=0.05)

    result = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_NOT_MATCH,
    )

    assert result == (1.0, 1.0)


@pytest.mark.asyncio
async def test_apply_label_pending_to_skipped_is_noop():
    """SKIP carries no directional signal. Transitioning from
    pending → skipped applies no delta and leaves stats
    untouched."""
    service = MergeLearningService()
    result = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_SKIPPED,
    )

    assert result is None
    assert service.snapshot() == {}


@pytest.mark.asyncio
async def test_apply_label_same_status_is_idempotent():
    """Reviewer-flagged P2 (commit 72381b4): a client retry or
    double-submit of the SAME match label must not move the
    threshold twice. apply_pair_label must be idempotent on
    same-status transitions.

    This is the central pin for the transition-aware redesign.
    Pre-fix: each label call moved the threshold by a full
    nudge regardless of whether the row's status actually
    changed."""
    service = MergeLearningService(label_match_nudge=-0.10)
    # First label: pending → labeled_match.
    first = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_MATCH,
    )
    assert first == (1.0, 0.90)

    # Double-submit: labeled_match → labeled_match (idempotent).
    # Must be a no-op — delta = 0.
    second = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_LABELED_MATCH,
        new_status=_LABELED_MATCH,
    )
    assert second is None, (
        "Same-status transition wasn't a no-op. A double-submit "
        "of the same label is now applying its nudge twice — "
        "exactly the bug the transition redesign fixes."
    )
    snapshot = service.snapshot()
    assert (
        pytest.approx(snapshot[("user-1", "Company")].ema_low_score, rel=1e-9) == 0.90
    ), "ema_low_score moved on the no-op re-label"


@pytest.mark.asyncio
async def test_apply_label_match_to_not_match_full_swing():
    """User changes their mind: match → not_match. The delta
    is contribution(not_match) - contribution(match) =
    +nudge - (-nudge) = 2*|nudge|. The threshold must reflect
    BOTH undoing the prior match nudge AND applying the
    not_match nudge in a single transition."""
    service = MergeLearningService(
        label_match_nudge=-0.10,
        label_not_match_nudge=+0.10,
    )
    # Seed via pending → match.
    await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_MATCH,
    )
    # ema = 0.90.

    swing = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_LABELED_MATCH,
        new_status=_LABELED_NOT_MATCH,
    )

    assert swing is not None
    old, new = swing
    assert pytest.approx(old, rel=1e-9) == 0.90
    # delta = +0.10 - (-0.10) = +0.20. new = 0.90 + 0.20 = 1.10
    # → clamped at 1.0.
    assert pytest.approx(new, rel=1e-9) == 1.0


@pytest.mark.asyncio
async def test_apply_label_match_to_skip_undoes_prior_contribution():
    """User downgrades a match label to skip (they're no longer
    confident). The delta is contribution(skip)=0 minus
    contribution(match)=-nudge → +nudge. Effect: the prior
    match nudge is undone and the threshold returns toward
    neutral."""
    service = MergeLearningService(label_match_nudge=-0.10)
    # Seed via pending → match.
    await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_MATCH,
    )
    # ema = 0.90.

    undo = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_LABELED_MATCH,
        new_status=_SKIPPED,
    )

    assert undo is not None
    old, new = undo
    assert pytest.approx(old, rel=1e-9) == 0.90
    # delta = 0 - (-0.10) = +0.10. new = 0.90 + 0.10 = 1.00.
    assert pytest.approx(new, rel=1e-9) == 1.00


@pytest.mark.asyncio
async def test_apply_label_match_clamps_at_floor_across_transitions():
    """Repeated match contributions can't drive the threshold
    below the floor. Pin so a refactor that drops the floor
    clamp doesn't let repeated matches go to 0 (accept
    everything)."""
    service = MergeLearningService(floor=0.70, label_match_nudge=-0.10)
    # Manually seed near the floor.
    await service.record_outcome("user-1", "Company", [0.75])

    # First time labeling: pending → match. delta = -0.10.
    # new = 0.75 + (-0.10) = 0.65 → clamped at floor 0.70.
    result = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_MATCH,
    )
    old, new = result
    assert pytest.approx(old, rel=1e-9) == 0.75
    assert pytest.approx(new, rel=1e-9) == 0.70


@pytest.mark.asyncio
async def test_apply_label_isolates_user_and_entity_type():
    """A label for (user-1, Company) must not affect
    (user-1, Person) or (user-2, Company). The threshold is
    per (user_id, entity_type) tuple — collapsing either axis
    would let one tenant's labeling decisions bleed into
    another's, or one entity type's calibration pollute
    another's."""
    service = MergeLearningService(label_match_nudge=-0.05)

    await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_MATCH,
    )

    snapshot = service.snapshot()
    assert ("user-1", "Company") in snapshot
    assert ("user-1", "Person") not in snapshot
    assert ("user-2", "Company") not in snapshot


@pytest.mark.asyncio
async def test_apply_label_unknown_status_treated_as_zero_contribution():
    """A garbage status (typo / stale enum after schema drift)
    contributes zero. Pending → unknown is a no-op (0 - 0 = 0).
    Unknown → labeled_match applies a full match nudge (the
    transition reads as "from neutral to labeled_match")."""
    service = MergeLearningService(label_match_nudge=-0.05)

    # pending → garbage: both contribute 0; delta = 0.
    result = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status="garbage_status",
    )
    assert result is None
    assert service.snapshot() == {}

    # garbage → labeled_match: garbage contributes 0, match
    # contributes nudge. delta = nudge. Bootstrap seed = 0.95.
    bootstrap = await service.apply_pair_label(
        "user-1",
        "Company",
        old_status="garbage_status",
        new_status=_LABELED_MATCH,
    )
    assert bootstrap == (1.0, 0.95)


@pytest.mark.asyncio
async def test_apply_label_threshold_change_propagates_to_get_threshold():
    """End-to-end at the service surface: apply transitions, then
    call get_threshold. Pin the contract that the feedback loop
    closes via the public read API.

    This is the exit-signal-level pin for slice C: "weights
    have updated" verifiable by reading get_threshold."""
    service = MergeLearningService(floor=0.70, margin=0.05, label_match_nudge=-0.10)

    default_threshold = 0.95
    before = await service.get_threshold("user-1", "Company", default_threshold)
    assert before == default_threshold

    # pending → match. ema = 1.0 + (-0.10) = 0.90.
    await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_MATCH,
    )
    # Another distinct pair: pending → match on top of existing
    # stats. ema = 0.90 + (-0.10) = 0.80.
    await service.apply_pair_label(
        "user-1",
        "Company",
        old_status=_PENDING,
        new_status=_LABELED_MATCH,
    )

    after = await service.get_threshold("user-1", "Company", default_threshold)
    assert after < default_threshold, (
        f"Threshold didn't budge after match labels: {after}. The "
        "feedback loop is broken — labels must influence "
        "get_threshold output."
    )
    assert pytest.approx(after, abs=1e-9) == 0.75
