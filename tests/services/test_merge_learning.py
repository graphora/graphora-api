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
# B2-active slice C: apply_pair_label feedback loop
#
# The disputed-pairs queue closes the loop on entity resolution:
# users label ambiguous pairs as match / not_match / skip and
# the service nudges the per-(user, entity_type) threshold so
# future ER runs adapt to the user's judgment. These tests pin
# the directional semantics (match → more permissive,
# not_match → more strict, skip → no-op) and the bootstrap
# behavior on first label.
# ============================================================


class _FakeDecision:
    """Local stand-in for graphora_server.services.disputed_pairs_service.Decision.

    apply_pair_label accepts ``Any`` and matches on
    ``getattr(decision, "value", decision)`` to avoid a circular
    import between merge.learning and disputed_pairs_service.
    These tests use a tiny stand-in so the unit-level test
    doesn't pull the disputed-pairs module in transitively."""

    def __init__(self, value: str) -> None:
        self.value = value


@pytest.mark.asyncio
async def test_apply_label_match_bootstrap_from_empty():
    """First-ever label on a (user, type) slot. No prior stats,
    so the seed is 1.0 (perfect-match prior) + match_nudge.
    Returns (1.0, seed) so callers can log the bootstrap.

    Pin so a refactor that drops the bootstrap branch (which
    would crash on stats=None) is caught immediately."""
    service = MergeLearningService(label_match_nudge=-0.05)

    result = await service.apply_pair_label("user-1", "Company", _FakeDecision("match"))

    assert result == (1.0, 0.95)
    snapshot = service.snapshot()
    assert (snapshot[("user-1", "Company")].ema_low_score) == 0.95


@pytest.mark.asyncio
async def test_apply_label_not_match_bootstrap_clamps_at_one():
    """NOT_MATCH on empty stats: seed = 1.0 + (+nudge) → clamped
    at 1.0 (we can't exceed perfect-match prior). Pin the
    ceiling clamp so a refactor that drops it doesn't quietly
    let the threshold drift > 1.0."""
    service = MergeLearningService(label_not_match_nudge=0.05)

    result = await service.apply_pair_label(
        "user-1", "Company", _FakeDecision("not_match")
    )

    assert result == (1.0, 1.0), (
        "not_match nudge on empty stats should clamp at 1.0 — "
        "an ema_low_score > 1.0 would make every future "
        "comparison fall below the (already-clamped) threshold "
        "and effectively turn ER off for this type."
    )


@pytest.mark.asyncio
async def test_apply_label_skip_is_noop():
    """SKIP is a valid review outcome (operators may defer pairs
    they don't have enough context for) but carries no
    directional signal. apply_pair_label MUST NOT mutate stats
    on SKIP — otherwise deferring a pair would silently shift
    the threshold."""
    service = MergeLearningService()
    result = await service.apply_pair_label("user-1", "Company", _FakeDecision("skip"))

    assert result is None
    assert service.snapshot() == {}


@pytest.mark.asyncio
async def test_apply_label_match_lowers_existing_threshold():
    """With prior stats, a match nudge LOWERS ema_low_score by
    the nudge magnitude. Reviewer-visible direction: user
    confirms the blocker grouped the pair correctly → be more
    permissive (i.e. lower the EMA so the threshold-from-EMA
    formula yields a lower threshold)."""
    service = MergeLearningService(label_match_nudge=-0.10)
    await service.record_outcome("user-1", "Company", [0.85])
    # record_outcome seeds ema_low_score=0.85

    result = await service.apply_pair_label("user-1", "Company", _FakeDecision("match"))

    assert result is not None
    old, new = result
    assert pytest.approx(old, rel=1e-9) == 0.85
    assert pytest.approx(new, rel=1e-9) == 0.75


@pytest.mark.asyncio
async def test_apply_label_match_clamps_at_floor():
    """Match labels accumulate until the floor stops them. Pin
    so a refactor that drops the floor clamp doesn't let
    repeated matches drive the threshold to 0 (which would
    accept every pair regardless of similarity)."""
    service = MergeLearningService(floor=0.70, label_match_nudge=-0.10)
    # Pre-seed at slightly above floor so one nudge takes us
    # to the floor.
    await service.record_outcome("user-1", "Company", [0.75])

    result = await service.apply_pair_label("user-1", "Company", _FakeDecision("match"))
    old, new = result
    assert pytest.approx(old, rel=1e-9) == 0.75
    assert pytest.approx(new, rel=1e-9) == 0.70

    # A second match should NOT push below the floor.
    second = await service.apply_pair_label("user-1", "Company", _FakeDecision("match"))
    assert pytest.approx(second[1], rel=1e-9) == 0.70


@pytest.mark.asyncio
async def test_apply_label_not_match_raises_existing_threshold():
    """NOT_MATCH on existing stats: ema_low_score rises by the
    nudge. User rejected a previously-grouped pair, so future
    runs should require a higher similarity to merge."""
    service = MergeLearningService(label_not_match_nudge=+0.05)
    await service.record_outcome("user-1", "Company", [0.85])

    old_new = await service.apply_pair_label(
        "user-1", "Company", _FakeDecision("not_match")
    )

    old, new = old_new
    assert pytest.approx(old, rel=1e-9) == 0.85
    assert pytest.approx(new, rel=1e-9) == 0.90


@pytest.mark.asyncio
async def test_apply_label_isolates_user_and_entity_type():
    """A label for (user-1, Company) must not affect
    (user-1, Person) or (user-2, Company). The threshold is
    per (user_id, entity_type) tuple — collapsing either axis
    would let one tenant's labeling decisions bleed into
    another's, or one entity type's calibration pollute
    another's."""
    service = MergeLearningService(label_match_nudge=-0.05)

    await service.apply_pair_label("user-1", "Company", _FakeDecision("match"))

    snapshot = service.snapshot()
    assert ("user-1", "Company") in snapshot
    assert ("user-1", "Person") not in snapshot
    assert ("user-2", "Company") not in snapshot


@pytest.mark.asyncio
async def test_apply_label_unknown_decision_is_noop():
    """A garbage decision (typo / stale enum after schema drift)
    is treated as a no-op rather than crashing or applying a
    surprising direction. The closed-set enum on the wire
    already prevents this from real callers — this test
    pins the defensive branch for the programmatic path."""
    service = MergeLearningService()
    result = await service.apply_pair_label(
        "user-1", "Company", _FakeDecision("unknown_value")
    )
    assert result is None
    assert service.snapshot() == {}


@pytest.mark.asyncio
async def test_apply_label_threshold_change_propagates_to_get_threshold():
    """End-to-end at the service surface: apply a match label,
    then call get_threshold and verify the returned threshold
    is below the default. Pins the contract that the feedback
    loop closes — labels alter what get_threshold returns.

    This is the exit-signal-level pin for slice C: "weights
    have updated" verifiable by reading get_threshold."""
    service = MergeLearningService(floor=0.70, margin=0.05, label_match_nudge=-0.10)

    default_threshold = 0.95
    before = await service.get_threshold("user-1", "Company", default_threshold)
    assert before == default_threshold

    # Bootstrap with a single match label.
    await service.apply_pair_label("user-1", "Company", _FakeDecision("match"))
    # ema_low_score = 1.0 + (-0.10) = 0.90. With margin=0.05,
    # default - margin = 0.90, so ema_low_score >= default-margin
    # → adaptive sticks at default. Apply another match.
    await service.apply_pair_label("user-1", "Company", _FakeDecision("match"))
    # ema_low_score now 0.80. adaptive = 0.80 - 0.05 = 0.75.

    after = await service.get_threshold("user-1", "Company", default_threshold)
    assert after < default_threshold, (
        f"Threshold didn't budge after match labels: {after}. The "
        "feedback loop is broken — labels must influence "
        "get_threshold output."
    )
    assert pytest.approx(after, abs=1e-9) == 0.75
