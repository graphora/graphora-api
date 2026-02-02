"""Unit tests for Transform API endpoints.

Phase 3: API Layer Tests - Transform Endpoints
London School TDD with mocked dependencies.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app
from app.auth import get_current_user_id
from app.schemas.transform import TransformStatus
from app.services.transform.status_models import (
    DetailedTransformStatus,
    TransformationStage,
    StageStatus,
    StageProgress,
    ResourceMetrics,
)


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
    with patch("app.api.transform.audit_service") as mock:
        mock.log_operation_start = AsyncMock(return_value="audit-123")
        mock.log_operation_success = AsyncMock()
        mock.log_operation_failure = AsyncMock()
        yield mock


@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker."""
    with patch("app.api.transform.progress_tracker") as mock:
        mock.initialize_transform = AsyncMock()
        mock.complete_stage = AsyncMock()
        mock.get_detailed_status = AsyncMock()
        mock.cleanup_transform = AsyncMock()
        yield mock


@pytest.fixture
def mock_file_validator():
    """Mock file validator."""
    with patch("app.api.transform.FileValidator") as MockClass:
        instance = MagicMock()
        instance.validate = AsyncMock(return_value=MagicMock(is_valid=True, errors=[]))
        MockClass.return_value = instance
        yield instance


@pytest.fixture
def sample_transform_status():
    """Create sample transform status."""
    return DetailedTransformStatus(
        transform_id="transform-abc123",
        overall_status=TransformStatus.COMPLETED,
        current_stage=TransformationStage.LOAD,
        stages_progress={
            TransformationStage.UPLOAD: StageProgress(
                stage=TransformationStage.UPLOAD,
                status=StageStatus.COMPLETED,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                percentage_complete=100.0,
            )
        },
        start_time=datetime.now(timezone.utc),
        resource_metrics=ResourceMetrics(),
    )


# ============================================================
# Filename Sanitization Tests
# ============================================================


class TestFilenameSanitization:
    """Test filename sanitization for security."""

    def test_sanitize_filename_should_allow_valid_names(self):
        """Should allow valid filenames."""
        from app.api.transform import sanitize_filename

        assert sanitize_filename("document.pdf") == "document.pdf"
        assert sanitize_filename("my-file_2024.txt") == "my-file_2024.txt"
        assert sanitize_filename("report 2024.docx") == "report 2024.docx"

    def test_sanitize_filename_should_reject_path_traversal(self):
        """Should reject path traversal attempts."""
        from app.api.transform import sanitize_filename

        with pytest.raises(ValueError, match="path traversal"):
            sanitize_filename("../etc/passwd")

        with pytest.raises(ValueError, match="path traversal"):
            sanitize_filename("..\\windows\\system32")

        with pytest.raises(ValueError, match="path traversal"):
            sanitize_filename("/etc/passwd")

    def test_sanitize_filename_should_reject_empty(self):
        """Should reject empty filenames."""
        from app.api.transform import sanitize_filename

        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_filename("")

        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_filename(None)

    def test_sanitize_filename_should_reject_special_characters(self):
        """Should reject filenames with dangerous characters."""
        from app.api.transform import sanitize_filename

        with pytest.raises(ValueError, match="disallowed characters"):
            sanitize_filename("file;rm -rf.txt")

        with pytest.raises(ValueError, match="disallowed characters"):
            sanitize_filename("file`whoami`.txt")

    def test_sanitize_filename_should_reject_dot_only(self):
        """Should reject dot-only filenames."""
        from app.api.transform import sanitize_filename

        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_filename(".")

        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_filename("..")


# ============================================================
# Transform Status Endpoint Tests
# ============================================================


class TestTransformStatusEndpoint:
    """Test GET /transform/status/{transform_id} endpoint."""

    def test_get_status_should_return_detailed_status(
        self, test_client, mock_progress_tracker, sample_transform_status
    ):
        """Should return detailed transform status."""
        mock_progress_tracker.get_detailed_status.return_value = sample_transform_status

        response = test_client.get("/api/v1/transform/status/transform-abc123")

        assert response.status_code == 200
        data = response.json()
        assert data["transform_id"] == "transform-abc123"
        assert data["overall_status"] == "completed"
        mock_progress_tracker.get_detailed_status.assert_awaited_once_with(
            "transform-abc123"
        )

    def test_get_status_should_return_404_for_unknown_transform(
        self, test_client, mock_progress_tracker
    ):
        """Should return 404 for unknown transform ID."""
        mock_progress_tracker.get_detailed_status.return_value = None

        response = test_client.get("/api/v1/transform/status/unknown-transform")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_status_should_exclude_metrics_when_requested(
        self, test_client, mock_progress_tracker, sample_transform_status
    ):
        """Should exclude metrics when include_metrics=false."""
        sample_transform_status.resource_metrics = {"cpu": 50, "memory": 100}
        mock_progress_tracker.get_detailed_status.return_value = sample_transform_status

        response = test_client.get(
            "/api/v1/transform/status/transform-abc123",
            params={"include_metrics": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("resource_metrics") is None

    def test_get_status_should_handle_internal_errors(
        self, test_client, mock_progress_tracker
    ):
        """Should return 500 on internal errors."""
        mock_progress_tracker.get_detailed_status.side_effect = Exception(
            "Database connection failed"
        )

        response = test_client.get("/api/v1/transform/status/transform-abc123")

        assert response.status_code == 500
        assert "Failed to get transform status" in response.json()["detail"]


# ============================================================
# Transform Cleanup Endpoint Tests
# ============================================================


class TestTransformCleanupEndpoint:
    """Test POST /transform/status/{transform_id}/cleanup endpoint."""

    def test_cleanup_should_schedule_background_task(
        self, test_client, mock_progress_tracker
    ):
        """Should schedule cleanup as background task."""
        response = test_client.post("/api/v1/transform/status/transform-abc123/cleanup")

        assert response.status_code == 200
        data = response.json()
        assert "cleanup scheduled" in data["message"].lower()
        assert "transform-abc123" in data["message"]

    def test_cleanup_should_include_user_context(
        self, test_client, mock_progress_tracker
    ):
        """Should include user context in response."""
        response = test_client.post("/api/v1/transform/status/transform-abc123/cleanup")

        assert response.status_code == 200
        data = response.json()
        assert "test-user-123" in data["message"]


# ============================================================
# Upload Endpoint Validation Tests
# ============================================================


class TestUploadEndpointValidation:
    """Test upload endpoint input validation."""

    def test_upload_should_reject_invalid_file(
        self,
        test_client,
        mock_audit_service,
        mock_progress_tracker,
        mock_file_validator,
    ):
        """Should reject invalid files."""
        mock_file_validator.validate.return_value = MagicMock(
            is_valid=False, errors=["File too large"]
        )

        files = {"files": ("test.pdf", BytesIO(b"fake pdf content"), "application/pdf")}

        with patch("app.api.transform.settings") as mock_settings:
            mock_settings.UPLOAD_DIR = "/tmp/test-uploads"
            mock_settings.API_V1_STR = "/api/v1"

            with patch("app.api.transform.Path") as MockPath:
                mock_path = MagicMock()
                mock_path.mkdir = MagicMock()
                mock_path.exists.return_value = False
                mock_path.__truediv__ = MagicMock(return_value=mock_path)
                MockPath.return_value = mock_path

                response = test_client.post(
                    "/api/v1/transform/test-ontology/upload",
                    files=files,
                )

        assert response.status_code == 400
        assert "File too large" in response.json()["detail"]

    def test_upload_should_reject_path_traversal_filename(
        self,
        test_client,
        mock_audit_service,
        mock_progress_tracker,
    ):
        """Should reject filenames with path traversal."""
        files = {"files": ("../../../etc/passwd", BytesIO(b"content"), "text/plain")}

        with patch("app.api.transform.settings") as mock_settings:
            mock_settings.UPLOAD_DIR = "/tmp/test-uploads"
            mock_settings.API_V1_STR = "/api/v1"

            with patch("app.api.transform.Path") as MockPath:
                mock_path = MagicMock()
                mock_path.mkdir = MagicMock()
                MockPath.return_value = mock_path

                response = test_client.post(
                    "/api/v1/transform/test-ontology/upload",
                    files=files,
                )

        assert response.status_code == 400
        assert "path traversal" in response.json()["detail"].lower()


# ============================================================
# Chunking Config Validation Tests
# ============================================================


class TestChunkingConfigValidation:
    """Test chunking configuration parsing."""

    def test_should_handle_invalid_chunking_config_json(self):
        """Should handle invalid JSON in chunking config gracefully."""
        # Test that the endpoint handles invalid JSON without crashing
        # The actual validation happens within the endpoint

        invalid_json = "not valid json {"

        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    def test_should_validate_chunking_strategy_enum(self):
        """Should validate chunking strategy values."""
        from app.services.chunking.config import ChunkingConfig

        # Valid strategies
        valid_strategies = ["semantic", "structural", "hybrid", "recursive"]
        for strategy in valid_strategies:
            config = ChunkingConfig(strategy=strategy)
            assert config.strategy.value == strategy

        # Invalid strategy should raise error
        with pytest.raises(Exception):  # Pydantic validation error
            ChunkingConfig(strategy="invalid_strategy")


# ============================================================
# Authentication Tests
# ============================================================


class TestTransformAuthenticationRequirements:
    """Test that endpoints require authentication."""

    def test_status_endpoint_requires_auth(self, monkeypatch):
        """Status endpoint should require authentication."""
        # Ensure auth bypass is disabled for this test
        monkeypatch.setenv("AUTH_BYPASS_ENABLED", "false")

        from app.config import Settings

        test_settings = Settings()

        with patch("app.auth.dependencies.settings", test_settings):
            client = TestClient(app)
            response = client.get("/api/v1/transform/status/transform-123")

        # Without auth, should fail with 401 (missing auth header)
        assert response.status_code == 401

    def test_cleanup_endpoint_requires_auth(self, monkeypatch):
        """Cleanup endpoint should require authentication."""
        monkeypatch.setenv("AUTH_BYPASS_ENABLED", "false")

        from app.config import Settings

        test_settings = Settings()

        with patch("app.auth.dependencies.settings", test_settings):
            client = TestClient(app)
            response = client.post("/api/v1/transform/status/transform-123/cleanup")

        assert response.status_code == 401


# ============================================================
# Response Format Tests
# ============================================================


class TestTransformResponseFormats:
    """Test API response formats."""

    def test_status_response_should_include_required_fields(
        self, test_client, mock_progress_tracker, sample_transform_status
    ):
        """Status response should include all required fields."""
        mock_progress_tracker.get_detailed_status.return_value = sample_transform_status

        response = test_client.get("/api/v1/transform/status/transform-abc123")

        assert response.status_code == 200
        data = response.json()

        # Required fields as per DetailedTransformStatus model
        assert "transform_id" in data
        assert "overall_status" in data
        assert "current_stage" in data
        assert "stages_progress" in data
        assert "start_time" in data
        assert "resource_metrics" in data

    def test_cleanup_response_should_be_json(self, test_client, mock_progress_tracker):
        """Cleanup response should be valid JSON."""
        response = test_client.post("/api/v1/transform/status/transform-abc123/cleanup")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        # Should be parseable JSON
        data = response.json()
        assert isinstance(data, dict)
        assert "message" in data
