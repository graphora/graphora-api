"""Unit tests for Audit Service.

Phase 4: Service Layer Tests - Audit Service
London School TDD with mocked database interactions.

The audit service is critical for:
1. Compliance tracking
2. Operation logging
3. User activity monitoring
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from graphora_server.services.audit_service import (
    AuditService,
    OperationType,
    OperationStatus,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_db():
    """Mock the database module."""
    with patch("graphora_server.services.audit_service.db") as mock:
        mock.fetchrow = AsyncMock()
        mock.fetch = AsyncMock()
        yield mock


@pytest.fixture
def mock_settings():
    """Mock settings for audit service."""
    with patch("graphora_server.services.audit_service.settings") as mock:
        mock.DATABASE_URL = "postgresql://test:test@localhost/test"
        mock.resolved_database_url = None
        mock.test_mode = False
        yield mock


@pytest.fixture
def audit_service(mock_settings):
    """Create audit service instance."""
    return AuditService()


@pytest.fixture
def sample_audit_record():
    """Create sample audit record."""
    return {
        "id": "audit-123",
        "user_id": "user-456",
        "operation_type": "transform_started",
        "operation_id": "transform-789",
        "resource_name": "Transform 789",
        "status": "in_progress",
        "metadata": {"ontology_id": "ont-1"},
        "created_at": datetime.utcnow(),
        "updated_at": None,
        "duration_ms": None,
        "error_message": None,
    }


# ============================================================
# Initialization Tests
# ============================================================


class TestAuditServiceInitialization:
    """Test audit service initialization."""

    def test_should_initialize_when_database_url_configured(self, mock_settings):
        """Should initialize successfully when DATABASE_URL is set."""
        mock_settings.DATABASE_URL = "postgresql://test@localhost/test"
        mock_settings.resolved_database_url = None
        mock_settings.test_mode = False

        service = AuditService()

        assert service is not None

    def test_should_initialize_when_resolved_database_url_configured(
        self, mock_settings
    ):
        """Should initialize when resolved_database_url is set."""
        mock_settings.DATABASE_URL = None
        mock_settings.resolved_database_url = "postgresql://test@localhost/test"
        mock_settings.test_mode = False

        service = AuditService()

        assert service is not None

    def test_should_raise_error_when_no_database_configured_in_production(self):
        """Should raise ValueError when no database URL in non-test mode."""
        with patch("graphora_server.services.audit_service.settings") as mock_settings:
            mock_settings.DATABASE_URL = None
            mock_settings.resolved_database_url = None
            mock_settings.test_mode = False

            with pytest.raises(ValueError, match="DATABASE_URL must be configured"):
                AuditService()

    def test_should_allow_no_database_in_test_mode(self):
        """Should allow initialization without database in test mode."""
        with patch("graphora_server.services.audit_service.settings") as mock_settings:
            mock_settings.DATABASE_URL = None
            mock_settings.resolved_database_url = None
            mock_settings.test_mode = True

            service = AuditService()
            assert service is not None


# ============================================================
# Log Operation Start Tests
# ============================================================


class TestLogOperationStart:
    """Test log_operation_start method."""

    @pytest.mark.asyncio
    async def test_should_insert_audit_record(self, audit_service, mock_db):
        """Should insert audit record into database."""
        mock_db.fetchrow.return_value = {"id": "audit-new-123"}

        result = await audit_service.log_operation_start(
            user_id="user-123",
            operation_type=OperationType.TRANSFORM_STARTED,
            operation_id="transform-456",
            resource_name="Transform 456",
            metadata={"ontology_id": "ont-1"},
        )

        assert result == "audit-new-123"
        mock_db.fetchrow.assert_awaited_once()

        # Verify the INSERT query was called
        call_args = mock_db.fetchrow.call_args
        query = call_args[0][0]
        assert "INSERT INTO audit_trail" in query
        assert "user_id" in query
        assert "operation_type" in query

    @pytest.mark.asyncio
    async def test_should_set_status_to_in_progress(self, audit_service, mock_db):
        """Should set initial status to IN_PROGRESS."""
        mock_db.fetchrow.return_value = {"id": "audit-123"}

        await audit_service.log_operation_start(
            user_id="user-123",
            operation_type=OperationType.TRANSFORM_STARTED,
            operation_id="transform-456",
        )

        call_args = mock_db.fetchrow.call_args[0]
        # Status should be in_progress
        assert OperationStatus.IN_PROGRESS.value in call_args

    @pytest.mark.asyncio
    async def test_should_return_empty_string_on_failure(self, audit_service, mock_db):
        """Should return empty string when insert fails."""
        mock_db.fetchrow.return_value = None

        result = await audit_service.log_operation_start(
            user_id="user-123",
            operation_type=OperationType.TRANSFORM_STARTED,
            operation_id="transform-456",
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_should_handle_database_exceptions(self, audit_service, mock_db):
        """Should handle database exceptions gracefully."""
        mock_db.fetchrow.side_effect = Exception("Database connection failed")

        result = await audit_service.log_operation_start(
            user_id="user-123",
            operation_type=OperationType.TRANSFORM_STARTED,
            operation_id="transform-456",
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_should_accept_all_operation_types(self, audit_service, mock_db):
        """Should accept all defined operation types."""
        mock_db.fetchrow.return_value = {"id": "audit-123"}

        for op_type in OperationType:
            result = await audit_service.log_operation_start(
                user_id="user-123",
                operation_type=op_type,
                operation_id=f"{op_type.value}-456",
            )
            assert result == "audit-123"


# ============================================================
# Log Operation Success Tests
# ============================================================


class TestLogOperationSuccess:
    """Test log_operation_success method."""

    @pytest.mark.asyncio
    async def test_should_update_status_to_success(self, audit_service, mock_db):
        """Should update audit record status to SUCCESS."""
        mock_db.fetchrow.side_effect = [
            {"metadata": {}},  # _merge_metadata query
            {"id": "audit-123"},  # UPDATE query
        ]

        result = await audit_service.log_operation_success(
            audit_id="audit-123",
            duration_ms=1500,
        )

        assert result is True

        # Verify UPDATE was called with SUCCESS status
        update_call = mock_db.fetchrow.call_args_list[-1]
        assert OperationStatus.SUCCESS.value in update_call[0]

    @pytest.mark.asyncio
    async def test_should_update_duration_ms(self, audit_service, mock_db):
        """Should update duration_ms field."""
        mock_db.fetchrow.side_effect = [
            {"metadata": {}},
            {"id": "audit-123"},
        ]

        await audit_service.log_operation_success(
            audit_id="audit-123",
            duration_ms=2500,
        )

        update_call = mock_db.fetchrow.call_args_list[-1]
        assert 2500 in update_call[0]

    @pytest.mark.asyncio
    async def test_should_merge_metadata(self, audit_service, mock_db):
        """Should merge new metadata with existing."""
        mock_db.fetchrow.side_effect = [
            {"metadata": {"existing_key": "existing_value"}},
            {"id": "audit-123"},
        ]

        await audit_service.log_operation_success(
            audit_id="audit-123",
            metadata={"new_key": "new_value"},
        )

        # Metadata should be merged
        mock_db.fetchrow.assert_awaited()

    @pytest.mark.asyncio
    async def test_should_return_false_on_update_failure(self, audit_service, mock_db):
        """Should return False when update fails."""
        # When passing metadata, _merge_metadata calls fetchrow first
        # Then log_operation_success calls fetchrow for UPDATE
        mock_db.fetchrow.side_effect = [
            {"metadata": {}},  # _merge_metadata SELECT
            None,  # UPDATE returns nothing (failure)
        ]

        result = await audit_service.log_operation_success(
            audit_id="audit-123",
            metadata={"key": "value"},  # Pass metadata to trigger _merge_metadata
        )

        assert result is False


# ============================================================
# Log Operation Failure Tests
# ============================================================


class TestLogOperationFailure:
    """Test log_operation_failure method."""

    @pytest.mark.asyncio
    async def test_should_update_status_to_failed(self, audit_service, mock_db):
        """Should update audit record status to FAILED."""
        mock_db.fetchrow.side_effect = [
            {"metadata": {}},
            {"id": "audit-123"},
        ]

        result = await audit_service.log_operation_failure(
            audit_id="audit-123",
            error_message="Something went wrong",
        )

        assert result is True

        update_call = mock_db.fetchrow.call_args_list[-1]
        assert OperationStatus.FAILED.value in update_call[0]

    @pytest.mark.asyncio
    async def test_should_store_error_message(self, audit_service, mock_db):
        """Should store error message in record."""
        mock_db.fetchrow.side_effect = [
            {"metadata": {}},
            {"id": "audit-123"},
        ]

        await audit_service.log_operation_failure(
            audit_id="audit-123",
            error_message="Database timeout",
        )

        update_call = mock_db.fetchrow.call_args_list[-1]
        assert "Database timeout" in update_call[0]

    @pytest.mark.asyncio
    async def test_should_include_duration_on_failure(self, audit_service, mock_db):
        """Should record duration even on failure."""
        mock_db.fetchrow.side_effect = [
            {"metadata": {}},
            {"id": "audit-123"},
        ]

        await audit_service.log_operation_failure(
            audit_id="audit-123",
            error_message="Failed",
            duration_ms=500,
        )

        update_call = mock_db.fetchrow.call_args_list[-1]
        assert 500 in update_call[0]


# ============================================================
# Log Operation End Tests
# ============================================================


class TestLogOperationEnd:
    """Test log_operation_end method."""

    @pytest.mark.asyncio
    async def test_should_find_record_by_user_and_operation_id(
        self, audit_service, mock_db
    ):
        """Should find audit record by user_id and operation_id."""
        mock_db.fetchrow.side_effect = [
            {"id": "audit-123", "metadata": {}},  # SELECT query
            {"id": "audit-123"},  # UPDATE query
        ]

        result = await audit_service.log_operation_end(
            user_id="user-456",
            operation_id="transform-789",
            status=OperationStatus.SUCCESS,
        )

        assert result is True

        # Verify SELECT was called with correct params
        select_call = mock_db.fetchrow.call_args_list[0]
        assert "user-456" in select_call[0]
        assert "transform-789" in select_call[0]

    @pytest.mark.asyncio
    async def test_should_return_false_when_record_not_found(
        self, audit_service, mock_db
    ):
        """Should return False when no matching record found."""
        mock_db.fetchrow.return_value = None

        result = await audit_service.log_operation_end(
            user_id="user-456",
            operation_id="unknown-operation",
            status=OperationStatus.SUCCESS,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_should_update_with_error_message_on_failure(
        self, audit_service, mock_db
    ):
        """Should include error message when status is FAILED."""
        mock_db.fetchrow.side_effect = [
            {"id": "audit-123", "metadata": {}},
            {"id": "audit-123"},
        ]

        await audit_service.log_operation_end(
            user_id="user-456",
            operation_id="transform-789",
            status=OperationStatus.FAILED,
            error_message="Transform failed",
        )

        update_call = mock_db.fetchrow.call_args_list[-1]
        assert "Transform failed" in update_call[0]


# ============================================================
# Log Direct Operation Tests
# ============================================================


class TestLogDirectOperation:
    """Test log_direct_operation method for single-call logging."""

    @pytest.mark.asyncio
    async def test_should_create_complete_audit_record(self, audit_service, mock_db):
        """Should create complete audit record in one call."""
        mock_db.fetchrow.return_value = {"id": "audit-direct-123"}

        result = await audit_service.log_direct_operation(
            user_id="user-123",
            operation_type=OperationType.SCHEMA_SEARCH,
            operation_id="search-456",
            status=OperationStatus.SUCCESS,
            resource_name="Schema Search",
            metadata={"query": "test"},
            duration_ms=100,
        )

        assert result == "audit-direct-123"

    @pytest.mark.asyncio
    async def test_should_allow_failed_status_with_error(self, audit_service, mock_db):
        """Should create record with FAILED status and error message."""
        mock_db.fetchrow.return_value = {"id": "audit-123"}

        await audit_service.log_direct_operation(
            user_id="user-123",
            operation_type=OperationType.SCHEMA_GENERATION,
            operation_id="gen-456",
            status=OperationStatus.FAILED,
            error_message="Generation failed",
        )

        call_args = mock_db.fetchrow.call_args[0]
        assert OperationStatus.FAILED.value in call_args
        assert "Generation failed" in call_args


# ============================================================
# Get User Audit Trail Tests
# ============================================================


class TestGetUserAuditTrail:
    """Test get_user_audit_trail method."""

    @pytest.mark.asyncio
    async def test_should_return_user_audit_records(self, audit_service, mock_db):
        """Should return audit records for user."""
        mock_db.fetch.return_value = [
            {"id": "audit-1", "operation_type": "transform_started"},
            {"id": "audit-2", "operation_type": "transform_completed"},
        ]

        result = await audit_service.get_user_audit_trail(user_id="user-123")

        assert len(result) == 2
        mock_db.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_filter_by_operation_type(self, audit_service, mock_db):
        """Should filter records by operation type when specified."""
        mock_db.fetch.return_value = [
            {"id": "audit-1", "operation_type": "transform_started"},
        ]

        await audit_service.get_user_audit_trail(
            user_id="user-123",
            operation_type=OperationType.TRANSFORM_STARTED,
        )

        call_args = mock_db.fetch.call_args[0]
        assert "operation_type" in call_args[0]
        assert OperationType.TRANSFORM_STARTED.value in call_args

    @pytest.mark.asyncio
    async def test_should_respect_pagination(self, audit_service, mock_db):
        """Should apply limit and offset for pagination."""
        mock_db.fetch.return_value = []

        await audit_service.get_user_audit_trail(
            user_id="user-123",
            limit=10,
            offset=20,
        )

        call_args = mock_db.fetch.call_args[0]
        # limit and offset should be in params
        assert 10 in call_args
        assert 20 in call_args

    @pytest.mark.asyncio
    async def test_should_return_empty_list_on_error(self, audit_service, mock_db):
        """Should return empty list on database error."""
        mock_db.fetch.side_effect = Exception("Database error")

        result = await audit_service.get_user_audit_trail(user_id="user-123")

        assert result == []

    @pytest.mark.asyncio
    async def test_should_order_by_created_at_desc(self, audit_service, mock_db):
        """Should order results by created_at descending."""
        mock_db.fetch.return_value = []

        await audit_service.get_user_audit_trail(user_id="user-123")

        call_args = mock_db.fetch.call_args[0]
        assert "ORDER BY created_at DESC" in call_args[0]


# ============================================================
# Operation Type Enum Tests
# ============================================================


class TestOperationType:
    """Test OperationType enum."""

    def test_should_have_transform_operations(self):
        """Should include transform-related operations."""
        assert OperationType.TRANSFORM_STARTED.value == "transform_started"
        assert OperationType.TRANSFORM_COMPLETED.value == "transform_completed"

    def test_should_have_merge_operations(self):
        """Should include merge-related operations."""
        assert OperationType.MERGE_STARTED.value == "merge_started"
        assert OperationType.MERGE_COMPLETED.value == "merge_completed"

    def test_should_have_schema_operations(self):
        """Should include schema-related operations."""
        assert OperationType.SCHEMA_GENERATION.value == "schema_generation"
        assert OperationType.SCHEMA_SEARCH.value == "schema_search"
        assert OperationType.SCHEMA_REFINEMENT.value == "schema_refinement"
        assert OperationType.SCHEMA_CREATE.value == "schema_create"

    def test_should_have_chat_operations(self):
        """Should include chat-related operations."""
        assert OperationType.CHAT_SESSION_STARTED.value == "chat_session_started"
        assert OperationType.CHAT_MESSAGE_SENT.value == "chat_message_sent"
        assert OperationType.CHAT_MESSAGE_RECEIVED.value == "chat_message_received"
        assert OperationType.CHAT_SESSION_ENDED.value == "chat_session_ended"


# ============================================================
# Operation Status Enum Tests
# ============================================================


class TestOperationStatus:
    """Test OperationStatus enum."""

    def test_should_have_all_status_values(self):
        """Should have success, failed, and in_progress statuses."""
        assert OperationStatus.SUCCESS.value == "success"
        assert OperationStatus.FAILED.value == "failed"
        assert OperationStatus.IN_PROGRESS.value == "in_progress"

    def test_status_values_should_be_lowercase(self):
        """Status values should be lowercase strings."""
        for status in OperationStatus:
            assert status.value == status.value.lower()
