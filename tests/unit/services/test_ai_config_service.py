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
