"""Unit tests for AIConfigService — focused on PR #26 review followups.

Most of the AIConfigService surface is exercised via the BAML routing
tests in ``tests/unit/utils/test_llm_helper_ollama.py`` and the e2e
provider tests. This file pins the small but high-stakes invariants
that those don't cover.
"""

import pytest
from unittest.mock import AsyncMock, patch

from graphora_server.services.ai_config_service import AIConfigService


@pytest.fixture
def mock_db():
    """Mock the postgres ``db`` module imported by ai_config_service."""
    with patch("graphora_server.services.ai_config_service.db") as mock:
        mock.fetchrow = AsyncMock()
        mock.fetch = AsyncMock()
        yield mock


@pytest.fixture
def mock_settings():
    """Mock settings so the service constructor doesn't refuse to init."""
    with patch("graphora_server.services.ai_config_service.settings") as mock:
        mock.DATABASE_URL = "postgresql://test:test@localhost/test"
        mock.resolved_database_url = None
        mock.test_mode = False
        yield mock


@pytest.fixture
def service(mock_settings):
    return AIConfigService()


class TestGetModelsByProvider:
    """PR #26 fixed a cross-tenant model-name leak by excluding
    ``version='custom'`` rows from this method's response. These tests
    pin the SQL filter so a future refactor can't silently regress it."""

    @pytest.mark.asyncio
    async def test_sql_filters_out_custom_rows(self, service, mock_db) -> None:
        """The query MUST exclude ``version='custom'`` rows so user-
        added private model names don't bleed cross-tenant."""
        mock_db.fetch.return_value = []

        await service.get_models_by_provider("openai")

        # Verify the SQL was called with the version filter — the
        # ``IS DISTINCT FROM`` form handles NULL versions correctly
        # (a row with version=NULL should still appear; only literal
        # ``'custom'`` is excluded).
        mock_db.fetch.assert_awaited_once()
        sql = mock_db.fetch.call_args[0][0]
        assert "IS DISTINCT FROM 'custom'" in sql, (
            "get_models_by_provider must exclude version='custom' rows "
            "from the public catalog (PR #26 review High #2)"
        )

    @pytest.mark.asyncio
    async def test_returns_rows_db_supplies(self, service, mock_db) -> None:
        """The service is a thin wrapper over the SQL — whatever rows
        the DB returns are mapped to AIModel. The filtering is in the
        SQL, not in Python."""
        mock_db.fetch.return_value = [
            {
                "id": "m1",
                "provider_id": "p1",
                "name": "gpt-5.5",
                "display_name": "GPT-5.5",
                "version": "latest",
                "is_active": True,
            },
        ]

        models = await service.get_models_by_provider("openai")

        assert len(models) == 1
        assert models[0].name == "gpt-5.5"


class TestResolveOrCreateModelId:
    """#29: strict resolution that rejects deprecated rows.

    Pre-#29 behavior: the method returned any matching row regardless
    of ``is_active``, which let users silently re-enable deprecated
    models (e.g., ``gpt-4-turbo`` after migration 23 deactivated it).
    The failure surfaced only at extraction time when OpenAI rejected
    the model. These tests pin the new fail-fast contract.
    """

    @pytest.mark.asyncio
    async def test_active_row_returns_id(self, service, mock_db) -> None:
        """Active match (curated or user-custom) → returns its id."""
        mock_db.fetchrow.return_value = {"id": "model-id-active"}

        result = await service._resolve_or_create_model_id("p1", "gpt-5.5")

        assert result == "model-id-active"
        # Only one query — the first SELECT for active rows hit.
        assert mock_db.fetchrow.await_count == 1

    @pytest.mark.asyncio
    async def test_deactivated_row_raises_with_helpful_message(
        self, service, mock_db
    ) -> None:
        """Inactive match (deprecated curated row) → ValueError that
        names the model and points users at the current catalog."""
        # First call (active lookup) returns None, second call
        # (deactivated lookup) returns a row.
        mock_db.fetchrow.side_effect = [
            None,  # no active match
            {"id": "deprecated-id", "version": "previous"},  # deactivated row exists
        ]

        with pytest.raises(ValueError) as exc_info:
            await service._resolve_or_create_model_id("p1", "gpt-4-turbo")

        msg = str(exc_info.value)
        assert "gpt-4-turbo" in msg
        assert "deprecated" in msg
        assert "catalog" in msg

    @pytest.mark.asyncio
    async def test_unknown_name_auto_registers_as_custom(
        self, service, mock_db
    ) -> None:
        """Genuinely-new name (no row at all) → INSERT as custom."""
        mock_db.fetchrow.side_effect = [
            None,  # no active match
            None,  # no deactivated match
            {"id": "newly-inserted-id"},  # INSERT RETURNING
        ]

        result = await service._resolve_or_create_model_id(
            "p1", "gpt-experimental-2026"
        )

        assert result is not None
        # Three queries: active lookup, deactivated lookup, INSERT
        assert mock_db.fetchrow.await_count == 3
        # Last call was the INSERT
        last_sql = mock_db.fetchrow.await_args_list[-1][0][0]
        assert "INSERT INTO ai_models" in last_sql
        assert "'custom'" in last_sql


class TestResolveExistingModelId:
    """The lenient companion to ``_resolve_or_create_model_id``.

    Used by ``update_provider_config`` when the user is keeping the
    same model — preserves backward compat so they can rotate api_key
    without being forced to migrate from a since-deactivated model.
    """

    @pytest.mark.asyncio
    async def test_returns_id_for_active_row(self, service, mock_db) -> None:
        mock_db.fetchrow.return_value = {"id": "active-model-id"}
        result = await service._resolve_existing_model_id("p1", "gpt-5.5")
        assert result == "active-model-id"

    @pytest.mark.asyncio
    async def test_returns_id_for_deactivated_row(self, service, mock_db) -> None:
        """Key contract — unlike the strict variant, this returns the
        id of a deactivated row instead of raising."""
        mock_db.fetchrow.return_value = {"id": "deprecated-model-id"}
        result = await service._resolve_existing_model_id("p1", "gpt-4-turbo")
        assert result == "deprecated-model-id"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_row_exists(self, service, mock_db) -> None:
        mock_db.fetchrow.return_value = None
        result = await service._resolve_existing_model_id("p1", "never-seen-name")
        assert result is None
