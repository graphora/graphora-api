"""Unit tests for Audit API endpoints.

Phase 3: API Layer Tests - Audit Endpoints
London School TDD with mocked dependencies.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.auth import get_current_user_id
from app.services.audit_service import OperationType


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def test_client():
    """Create test client with auth override."""
    app.dependency_overrides[get_current_user_id] = lambda: "test-user-123"
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_user_id, None)


@pytest.fixture
def mock_audit_service():
    """Mock audit service."""
    with patch("app.api.audit.audit_service") as mock:
        mock.get_audit_summary = AsyncMock()
        mock.get_user_audit_trail = AsyncMock()
        yield mock


@pytest.fixture
def mock_db():
    """Mock database module."""
    with patch("app.api.audit.db") as mock:
        mock.fetch = AsyncMock()
        yield mock


@pytest.fixture
def sample_audit_summary():
    """Create sample audit summary."""
    return {
        "total_operations": 100,
        "operations_by_type": {
            "transform_started": 30,
            "transform_completed": 28,
            "merge_started": 20,
            "merge_completed": 18,
        },
        "success_rate": 0.92,
        "recent_operations": [
            {
                "id": "op-1",
                "operation_type": "transform_completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
            }
        ],
    }


@pytest.fixture
def sample_audit_trail():
    """Create sample audit trail records."""
    return [
        {
            "id": "audit-1",
            "user_id": "test-user-123",
            "operation_type": "transform_started",
            "operation_id": "transform-abc",
            "resource_name": "Transform abc",
            "status": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": 1500,
        },
        {
            "id": "audit-2",
            "user_id": "test-user-123",
            "operation_type": "transform_completed",
            "operation_id": "transform-abc",
            "resource_name": "Transform abc",
            "status": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": 30000,
        },
    ]


# ============================================================
# Audit Summary Endpoint Tests
# ============================================================


class TestAuditSummaryEndpoint:
    """Test GET /audit/summary endpoint."""

    def test_get_summary_should_return_user_summary(
        self, test_client, mock_audit_service, sample_audit_summary
    ):
        """Should return audit summary for current user."""
        mock_audit_service.get_audit_summary.return_value = sample_audit_summary

        response = test_client.get("/api/v1/audit/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["total_operations"] == 100
        assert data["success_rate"] == 0.92
        mock_audit_service.get_audit_summary.assert_awaited_once_with("test-user-123")

    def test_get_summary_should_handle_empty_summary(
        self, test_client, mock_audit_service
    ):
        """Should handle empty audit summary."""
        mock_audit_service.get_audit_summary.return_value = {
            "total_operations": 0,
            "operations_by_type": {},
            "success_rate": 1.0,
            "recent_operations": [],
        }

        response = test_client.get("/api/v1/audit/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["total_operations"] == 0

    def test_get_summary_should_return_500_on_error(
        self, test_client, mock_audit_service
    ):
        """Should return 500 on internal errors."""
        mock_audit_service.get_audit_summary.side_effect = Exception("Database error")

        response = test_client.get("/api/v1/audit/summary")

        assert response.status_code == 500
        assert "Error fetching audit summary" in response.json()["detail"]


# ============================================================
# Audit Trail Endpoint Tests
# ============================================================


class TestAuditTrailEndpoint:
    """Test GET /audit/trail endpoint."""

    def test_get_trail_should_return_records(
        self, test_client, mock_audit_service, sample_audit_trail
    ):
        """Should return audit trail records."""
        mock_audit_service.get_user_audit_trail.return_value = sample_audit_trail

        response = test_client.get("/api/v1/audit/trail")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["records"]) == 2
        assert data["offset"] == 0
        assert data["limit"] == 50

    def test_get_trail_should_filter_by_operation_type(
        self, test_client, mock_audit_service
    ):
        """Should filter by operation type."""
        mock_audit_service.get_user_audit_trail.return_value = []

        response = test_client.get(
            "/api/v1/audit/trail",
            params={"operation_type": "transform_started"},
        )

        assert response.status_code == 200
        mock_audit_service.get_user_audit_trail.assert_awaited_once()
        call_kwargs = mock_audit_service.get_user_audit_trail.call_args[1]
        assert call_kwargs["operation_type"] == OperationType.TRANSFORM_STARTED

    def test_get_trail_should_reject_invalid_operation_type(
        self, test_client, mock_audit_service
    ):
        """Should reject invalid operation types."""
        response = test_client.get(
            "/api/v1/audit/trail",
            params={"operation_type": "invalid_operation"},
        )

        assert response.status_code == 400
        assert "Invalid operation type" in response.json()["detail"]

    def test_get_trail_should_respect_pagination(
        self, test_client, mock_audit_service, sample_audit_trail
    ):
        """Should respect pagination parameters."""
        mock_audit_service.get_user_audit_trail.return_value = sample_audit_trail[:1]

        response = test_client.get(
            "/api/v1/audit/trail",
            params={"limit": 1, "offset": 1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 1
        assert data["limit"] == 1

        call_kwargs = mock_audit_service.get_user_audit_trail.call_args[1]
        assert call_kwargs["limit"] == 1
        assert call_kwargs["offset"] == 1

    def test_get_trail_should_use_default_pagination(
        self, test_client, mock_audit_service, sample_audit_trail
    ):
        """Should use default pagination values."""
        mock_audit_service.get_user_audit_trail.return_value = sample_audit_trail

        response = test_client.get("/api/v1/audit/trail")

        assert response.status_code == 200

        call_kwargs = mock_audit_service.get_user_audit_trail.call_args[1]
        assert call_kwargs["limit"] == 50
        assert call_kwargs["offset"] == 0

    def test_get_trail_should_return_500_on_error(
        self, test_client, mock_audit_service
    ):
        """Should return 500 on internal errors."""
        mock_audit_service.get_user_audit_trail.side_effect = Exception(
            "Database error"
        )

        response = test_client.get("/api/v1/audit/trail")

        assert response.status_code == 500
        assert "Error fetching audit trail" in response.json()["detail"]


# ============================================================
# Conflicts Summary Endpoint Tests
# ============================================================


class TestConflictsSummaryEndpoint:
    """Test GET /audit/conflicts endpoint."""

    def test_get_conflicts_should_return_empty_for_no_merges(
        self, test_client, mock_db
    ):
        """Should return empty summary when user has no merges."""
        mock_db.fetch.return_value = []

        response = test_client.get("/api/v1/audit/conflicts")

        assert response.status_code == 200
        data = response.json()
        assert data["total_conflicts"] == 0
        assert data["conflicts_by_merge"] == []
        assert data["recent_conflicts"] == []

    def test_get_conflicts_should_aggregate_by_merge(self, test_client, mock_db):
        """Should aggregate conflicts by merge ID."""
        # First query returns user's merges
        mock_db.fetch.side_effect = [
            [{"operation_id": "merge-1"}, {"operation_id": "merge-2"}],
            # Second query returns conflicts
            [
                {
                    "merge_id": "merge-1",
                    "node_type": "Company",
                    "need_human_review": True,
                    "previous_props": {},
                    "changed_props": {},
                },
                {
                    "merge_id": "merge-1",
                    "node_type": "Company",
                    "need_human_review": True,
                    "previous_props": {},
                    "changed_props": {},
                },
                {
                    "merge_id": "merge-2",
                    "node_type": "Person",
                    "need_human_review": True,
                    "previous_props": {},
                    "changed_props": {},
                },
            ],
        ]

        response = test_client.get("/api/v1/audit/conflicts")

        assert response.status_code == 200
        data = response.json()
        assert data["total_conflicts"] == 3

        # Check aggregation
        merge_1_summary = next(
            (m for m in data["conflicts_by_merge"] if m["merge_id"] == "merge-1"),
            None,
        )
        assert merge_1_summary is not None
        assert merge_1_summary["total_conflicts"] == 2
        assert merge_1_summary["by_type"]["Company"] == 2

    def test_get_conflicts_should_limit_recent_conflicts(self, test_client, mock_db):
        """Should return only 10 most recent conflicts."""
        # First query returns user's merges
        mock_db.fetch.side_effect = [
            [{"operation_id": "merge-1"}],
            # Second query returns many conflicts
            [
                {
                    "merge_id": "merge-1",
                    "node_type": "Company",
                    "need_human_review": True,
                    "previous_props": {},
                    "changed_props": {},
                }
                for _ in range(15)
            ],
        ]

        response = test_client.get("/api/v1/audit/conflicts")

        assert response.status_code == 200
        data = response.json()
        assert len(data["recent_conflicts"]) == 10

    def test_get_conflicts_should_return_500_on_error(self, test_client, mock_db):
        """Should return 500 on internal errors."""
        mock_db.fetch.side_effect = Exception("Database error")

        response = test_client.get("/api/v1/audit/conflicts")

        assert response.status_code == 500
        assert "Error fetching conflicts summary" in response.json()["detail"]


# ============================================================
# Authentication Tests
# ============================================================


class TestAuditAuthenticationRequirements:
    """Test that audit endpoints require authentication."""

    def test_summary_endpoint_requires_auth(self, mock_audit_service):
        """Summary endpoint should require authentication."""
        client = TestClient(app)

        response = client.get("/api/v1/audit/summary")

        # Without auth, should fail
        assert response.status_code in [401, 403, 422]

    def test_trail_endpoint_requires_auth(self, mock_audit_service):
        """Trail endpoint should require authentication."""
        client = TestClient(app)

        response = client.get("/api/v1/audit/trail")

        assert response.status_code in [401, 403, 422]

    def test_conflicts_endpoint_requires_auth(self, mock_db):
        """Conflicts endpoint should require authentication."""
        client = TestClient(app)

        response = client.get("/api/v1/audit/conflicts")

        assert response.status_code in [401, 403, 422]


# ============================================================
# Response Format Tests
# ============================================================


class TestAuditResponseFormats:
    """Test audit API response formats."""

    def test_trail_response_should_include_pagination_metadata(
        self, test_client, mock_audit_service, sample_audit_trail
    ):
        """Trail response should include pagination metadata."""
        mock_audit_service.get_user_audit_trail.return_value = sample_audit_trail

        response = test_client.get("/api/v1/audit/trail")

        assert response.status_code == 200
        data = response.json()

        # Required pagination fields
        assert "records" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data

    def test_conflicts_response_should_include_structure(self, test_client, mock_db):
        """Conflicts response should include expected structure."""
        mock_db.fetch.return_value = []

        response = test_client.get("/api/v1/audit/conflicts")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "total_conflicts" in data
        assert "conflicts_by_merge" in data
        assert "recent_conflicts" in data


# ============================================================
# Operation Type Validation Tests
# ============================================================


class TestOperationTypeValidation:
    """Test operation type parameter validation."""

    def test_should_accept_valid_operation_types(self, test_client, mock_audit_service):
        """Should accept all valid operation types."""
        mock_audit_service.get_user_audit_trail.return_value = []

        valid_types = [
            "transform_started",
            "transform_completed",
            "merge_started",
            "merge_completed",
        ]

        for op_type in valid_types:
            response = test_client.get(
                "/api/v1/audit/trail",
                params={"operation_type": op_type},
            )
            assert response.status_code == 200, f"Failed for operation type: {op_type}"

    def test_should_list_valid_types_in_error_message(
        self, test_client, mock_audit_service
    ):
        """Error message should list valid operation types."""
        response = test_client.get(
            "/api/v1/audit/trail",
            params={"operation_type": "invalid"},
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Valid types:" in detail
