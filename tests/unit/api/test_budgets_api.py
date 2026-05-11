"""Unit tests for the budgets API.

Three concerns pinned:
  * Tenant scoping: every route reads/writes via auth.user_id —
    same pattern as /decisions and /cost.
  * Wire shapes: Decimal serializes to a string so JSON precision
    survives across the boundary (matches /cost's
    estimated_cost_usd convention).
  * Preflight at transform-upload: blocked users get 402, allowed
    users see the normal flow (or its downstream error, but NOT
    a 402).
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from graphora_server.auth import AuthContext, get_current_auth, get_current_user_id
from graphora_server.main import app
from graphora_server.services.budget_service import (
    Budget,
    BudgetCheckResult,
    BudgetState,
    BudgetStatus,
)


@pytest.fixture
def test_client():
    """Auth bypass — fix user_id so we can assert it gets threaded
    through every read/write."""

    def fake_auth():
        return AuthContext(user_id="test-user-1", token="t", claims={})

    def fake_user_id():
        return "test-user-1"

    app.dependency_overrides[get_current_auth] = fake_auth
    app.dependency_overrides[get_current_user_id] = fake_user_id
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_auth, None)
        app.dependency_overrides.pop(get_current_user_id, None)


# ---- GET /me ---------------------------------------------------------------


def test_get_my_budget_returns_null_when_unset(test_client):
    """null is a valid response, not 404 — UI renders the
    'add budget' CTA instead of a not-found error page."""
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.get_budget = AsyncMock(return_value=None)
        mock_class.return_value = mock

        response = test_client.get("/api/v1/budgets/me")

    assert response.status_code == 200
    assert response.json() is None
    mock.get_budget.assert_awaited_once_with("test-user-1")


def test_get_my_budget_returns_string_decimal(test_client):
    """monthly_cap_usd serializes as a string for Decimal
    precision — same convention as /cost's estimated_cost_usd."""
    budget = Budget(
        user_id="test-user-1",
        monthly_cap_usd=Decimal("25.5000"),
        created_at="2026-05-01T00:00:00+00:00",
        updated_at="2026-05-10T00:00:00+00:00",
    )
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.get_budget = AsyncMock(return_value=budget)
        mock_class.return_value = mock

        response = test_client.get("/api/v1/budgets/me")

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "test-user-1"
    assert body["monthly_cap_usd"] == "25.5000"


# ---- PUT /me ---------------------------------------------------------------


def test_put_my_budget_persists_via_service(test_client):
    budget = Budget(
        user_id="test-user-1",
        monthly_cap_usd=Decimal("100"),
        created_at="2026-05-11T00:00:00+00:00",
        updated_at="2026-05-11T00:00:00+00:00",
    )
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.set_budget = AsyncMock(return_value=budget)
        mock_class.return_value = mock

        response = test_client.put(
            "/api/v1/budgets/me",
            json={"monthly_cap_usd": "100"},
        )

    assert response.status_code == 200
    mock.set_budget.assert_awaited_once_with("test-user-1", Decimal("100"))
    assert response.json()["monthly_cap_usd"] == "100"


def test_put_my_budget_rejects_negative(test_client):
    """Pydantic ge=0 catches this at the schema layer before the
    service is touched — cheaper than the service-level ValueError
    and gives a structured 422 the UI can render field-by-field."""
    response = test_client.put(
        "/api/v1/budgets/me",
        json={"monthly_cap_usd": "-1"},
    )
    assert response.status_code == 422


def test_put_my_budget_503_in_dev_mode_to_avoid_silent_loss(test_client):
    """Dev mode (no DATABASE_URL) returns None from set_budget.
    The endpoint surfaces a 503 so operators don't silently lose
    their write — the alternative (returning 200 with a stale
    body) would be confusing."""
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.set_budget = AsyncMock(return_value=None)
        mock_class.return_value = mock

        response = test_client.put(
            "/api/v1/budgets/me",
            json={"monthly_cap_usd": "100"},
        )

    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


# ---- DELETE /me ------------------------------------------------------------


def test_delete_my_budget_idempotent(test_client):
    """Deleting a non-existent budget returns deleted=False, not
    404. Idempotency matters — clients can re-issue DELETE
    without branching on prior state."""
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.delete_budget = AsyncMock(return_value=False)
        mock_class.return_value = mock

        response = test_client.delete("/api/v1/budgets/me")

    assert response.status_code == 200
    assert response.json() == {"deleted": False}


# ---- GET /me/status --------------------------------------------------------


def test_get_status_under(test_client):
    """The default happy-path: $5 of $100, UNDER. The state
    string is what the UI/agent renders; pinning it explicitly
    keeps the enum and the JSON wire contract in sync."""
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.get_status = AsyncMock(
            return_value=BudgetStatus(
                state=BudgetState.UNDER,
                current_spend_usd=Decimal("5.00"),
                cap_usd=Decimal("100"),
                period_start="2026-05-01T00:00:00+00:00",
                period_end="2026-06-01T00:00:00+00:00",
            )
        )
        mock_class.return_value = mock

        response = test_client.get("/api/v1/budgets/me/status")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "under"
    assert body["current_spend_usd"] == "5.00"
    assert body["cap_usd"] == "100"


def test_get_status_unset_has_null_cap(test_client):
    """When no budget is set, cap_usd is null but state is
    'unset' (not the same as 'under' — a future code path could
    branch on this to show the CTA)."""
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.get_status = AsyncMock(
            return_value=BudgetStatus(
                state=BudgetState.UNSET,
                current_spend_usd=Decimal("0"),
                cap_usd=None,
                period_start="2026-05-01T00:00:00+00:00",
                period_end="2026-06-01T00:00:00+00:00",
            )
        )
        mock_class.return_value = mock

        response = test_client.get("/api/v1/budgets/me/status")

    body = response.json()
    assert body["state"] == "unset"
    assert body["cap_usd"] is None


# ---- Preflight enforcement ------------------------------------------------


@pytest.fixture
def _empty_upload(tmp_path):
    """A 1-byte file the transform-upload endpoints accept. The
    preflight runs before any file processing, so the file
    contents don't matter — but the multipart form needs SOMETHING
    to send."""
    f = tmp_path / "a.txt"
    f.write_text("x")
    return f


def test_preflight_blocks_with_402_when_over_budget(test_client, _empty_upload):
    """Pin the enforcement contract: a user over their cap gets
    HTTP 402 from the upload endpoint, NOT a 200 + downstream
    error in the task. The 402 body carries the structured
    reason / spend / cap so the UI can render a useful
    "buy more budget" CTA without parsing the message."""
    blocked = BudgetCheckResult(
        allowed=False,
        state=BudgetState.OVER,
        current_spend_usd=Decimal("150"),
        cap_usd=Decimal("100"),
        reason="Monthly budget exceeded: spent $150 of $100 cap.",
    )
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.check_can_proceed = AsyncMock(return_value=blocked)
        mock_class.return_value = mock

        with open(_empty_upload, "rb") as f:
            response = test_client.post(
                "/api/v1/transform/some-ontology/upload",
                files={"files": ("a.txt", f, "text/plain")},
            )

    assert response.status_code == 402
    body = response.json()["detail"]
    assert body["error"] == "budget_exceeded"
    assert body["state"] == "over"
    assert body["current_spend_usd"] == "150"
    assert body["cap_usd"] == "100"
    # The check ran before any audit row was created — pin that
    # check_can_proceed was the FIRST service interaction.
    assert mock.check_can_proceed.await_count == 1


def test_preflight_lets_under_budget_through(
    test_client, _empty_upload, monkeypatch, tmp_path
):
    """Under-budget users should NOT get 402. They may still hit
    downstream errors (the underlying transform may fail for other
    reasons), but the budget gate doesn't fire. Pin: not 402.

    Mock the audit-service downstream call so the test doesn't
    open a psycopg pool against a missing dev DB — without this
    mock the test slows by ~30s waiting for the pool init.

    Reviewer-flagged on commit 5b78f85 (P2.2): also monkeypatch
    UPLOAD_DIR to a tmpdir. Pre-fix this test hit the real default
    upload directory (~/.graphora/uploads/...), which fails with
    PermissionError on CI workers and pollutes the dev environment
    elsewhere. The 402-cleanup test already does this; mirror it
    here for parity."""
    from graphora_server.config import settings as app_settings

    monkeypatch.setattr(app_settings, "UPLOAD_DIR", str(tmp_path / "uploads"))

    allowed = BudgetCheckResult(
        allowed=True,
        state=BudgetState.UNDER,
        current_spend_usd=Decimal("5"),
        cap_usd=Decimal("100"),
        reason=None,
    )
    with (
        patch("graphora_server.api.budgets.BudgetService") as mock_class,
        patch(
            "graphora_server.api.transform.audit_service.log_operation_start",
            new=AsyncMock(return_value="audit-1"),
        ),
        patch(
            "graphora_server.api.transform.audit_service.log_operation_failure",
            new=AsyncMock(return_value=True),
        ),
    ):
        mock = AsyncMock()
        mock.check_can_proceed = AsyncMock(return_value=allowed)
        mock_class.return_value = mock

        with open(_empty_upload, "rb") as f:
            response = test_client.post(
                "/api/v1/transform/some-ontology/upload",
                files={"files": ("a.txt", f, "text/plain")},
            )

    # The downstream code may 4xx/5xx for other reasons (no DB
    # in unit-test mode, ontology not found, etc.) — the only
    # thing this test guards against is the budget gate.
    assert response.status_code != 402


def test_preflight_runs_on_auto_schema_endpoint(test_client, _empty_upload):
    """The same enforcement gate covers the auto-schema endpoint.
    Three transform-upload routes share the same helper — if
    a future fourth route forgets to call it, the user gets
    free spending. Pin via the auto-schema route specifically."""
    blocked = BudgetCheckResult(
        allowed=False,
        state=BudgetState.OVER,
        current_spend_usd=Decimal("1"),
        cap_usd=Decimal("0"),
        reason="cap=0 blocks all spend",
    )
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.check_can_proceed = AsyncMock(return_value=blocked)
        mock_class.return_value = mock

        with open(_empty_upload, "rb") as f:
            response = test_client.post(
                "/api/v1/transform/upload",
                files={"files": ("a.txt", f, "text/plain")},
            )

    assert response.status_code == 402


def test_preflight_runs_on_schemaless_endpoint(test_client, _empty_upload):
    """And on the schemaless endpoint — the third upload route."""
    blocked = BudgetCheckResult(
        allowed=False,
        state=BudgetState.OVER,
        current_spend_usd=Decimal("1"),
        cap_usd=Decimal("0"),
        reason="cap=0 blocks all spend",
    )
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.check_can_proceed = AsyncMock(return_value=blocked)
        mock_class.return_value = mock

        with open(_empty_upload, "rb") as f:
            response = test_client.post(
                "/api/v1/transform/schemaless/upload",
                files={"files": ("a.txt", f, "text/plain")},
            )

    assert response.status_code == 402


# ---- Fail-closed on degraded DB (P1 reviewer fix) -------------------------


def test_preflight_returns_503_when_budget_db_read_fails(test_client, _empty_upload):
    """Reviewer-flagged P1 on commit 535f56d. Pre-fix the service
    swallowed DB errors and returned None → state UNSET →
    allowed=True. A degraded Postgres with a configured
    DATABASE_URL silently disabled every budget cap.

    Post-fix: BudgetReadError propagates out of check_can_proceed,
    the helper catches it, and the API responds with 503. This
    pin asserts the user sees the failure rather than slipping
    through."""
    from graphora_server.services.budget_service import BudgetReadError

    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.check_can_proceed = AsyncMock(
            side_effect=BudgetReadError("postgres unreachable"),
        )
        mock_class.return_value = mock

        with open(_empty_upload, "rb") as f:
            response = test_client.post(
                "/api/v1/transform/some-ontology/upload",
                files={"files": ("a.txt", f, "text/plain")},
            )

    assert response.status_code == 503
    body = response.json()["detail"]
    assert body["error"] == "budget_db_unavailable"


def test_ontology_upload_does_not_delete_upload_root_on_402(
    test_client, _empty_upload, monkeypatch, tmp_path
):
    """Reviewer-flagged P1 on commit 535f56d. The ontology-supplied
    upload endpoint set ``temp_dir = Path(settings.UPLOAD_DIR)``
    early, then deleted ``temp_dir`` in the broad except. A 402
    fell into the broad except (no HTTPException guard) and
    nuked the entire upload root, taking out every other
    transform's working directory.

    Pin: after a 402 from preflight, the upload root still
    exists. Auto-schema and schemaless already had the guard;
    this fires loud on the ontology endpoint specifically."""
    from graphora_server.config import settings as app_settings

    # Point UPLOAD_DIR at a tmpdir with a sibling-transform
    # marker file in it. Pre-fix the rmtree wipes the marker;
    # post-fix it survives.
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    sibling_marker = upload_dir / "another_transform_dir"
    sibling_marker.mkdir()
    monkeypatch.setattr(app_settings, "UPLOAD_DIR", str(upload_dir))

    blocked = BudgetCheckResult(
        allowed=False,
        state=BudgetState.OVER,
        current_spend_usd=Decimal("150"),
        cap_usd=Decimal("100"),
        reason="Monthly budget exceeded",
    )
    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.check_can_proceed = AsyncMock(return_value=blocked)
        mock_class.return_value = mock

        with open(_empty_upload, "rb") as f:
            response = test_client.post(
                "/api/v1/transform/some-ontology/upload",
                files={"files": ("a.txt", f, "text/plain")},
            )

    assert response.status_code == 402
    # Pin the fix: the upload root and the sibling marker survive.
    assert upload_dir.exists(), (
        "UPLOAD_DIR was deleted on 402 — the broad except in "
        "the ontology endpoint caught HTTPException and ran "
        "shutil.rmtree on the root. Add `except HTTPException: "
        "raise` like the other two upload endpoints."
    )
    assert sibling_marker.exists(), (
        "A sibling transform's working dir was collateral-damaged "
        "by the cleanup. Same root cause as the upload_dir "
        "assertion above."
    )


def test_get_my_budget_returns_503_on_db_read_error(test_client):
    """Mirror pin for the read endpoint. When the DB is configured
    but degraded, the user must see 503 not "no budget set" —
    otherwise operators believe they have no cap and behave
    accordingly."""
    from graphora_server.services.budget_service import BudgetReadError

    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.get_budget = AsyncMock(side_effect=BudgetReadError("DB down"))
        mock_class.return_value = mock

        response = test_client.get("/api/v1/budgets/me")

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "budget_db_unavailable"


def test_get_status_returns_503_on_db_read_error(test_client):
    """Same fail-closed contract on the status endpoint. The
    dashboard renders the state label — silently reporting
    "unset" when the DB is degraded would mislead operators."""
    from graphora_server.services.budget_service import BudgetReadError

    with patch("graphora_server.api.budgets.BudgetService") as mock_class:
        mock = AsyncMock()
        mock.get_status = AsyncMock(side_effect=BudgetReadError("DB down"))
        mock_class.return_value = mock

        response = test_client.get("/api/v1/budgets/me/status")

    assert response.status_code == 503
