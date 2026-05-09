"""Unit tests for UsageTrackingService.get_transform_cost_report.

B5-obs slice 1: per-extraction cost / token aggregation. The
UsageTrackingService class has many methods; these tests cover the
B5-obs addition only — pre-existing methods are well-exercised
through dashboard.py integration paths.

Pinned contracts:
  * SQL filters on BOTH transform_id AND user_id — a malicious
    request for another user's transform_id returns the zero-aggregate.
  * estimated_cost_usd is None when no priced row was found,
    string-Decimal otherwise (distinguishes "free" from "unpriced").
  * by_operation_type partitions the totals so callers can see
    "extraction cost X, schema_inference cost Y".
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from graphora_server.config import settings
from graphora_server.services.usage_tracking import UsageTrackingService


@pytest.fixture(autouse=True)
def _ensure_test_mode(monkeypatch):
    """UsageTrackingService.__init__ raises when DATABASE_URL isn't
    set unless test_mode is True. Set both so we can construct the
    service in test."""
    monkeypatch.setattr(settings, "test_mode", True)


@pytest.fixture
def service():
    return UsageTrackingService()


@pytest.mark.asyncio
async def test_cost_report_filters_by_transform_id_and_user_id(service):
    """Pin the SQL contract: WHERE clause includes BOTH
    ``transform_id = %s`` AND ``user_id = %s``. Without the user_id
    filter, any caller who knows another user's transform_id could
    fetch their cost."""
    with patch(
        "graphora_server.services.usage_tracking.db.fetch",
        new=AsyncMock(return_value=[]),
    ) as mock_fetch:
        await service.get_transform_cost_report(transform_id="tx-1", user_id="user-1")

    query = mock_fetch.await_args.args[0]
    assert "transform_id = %s" in query
    assert "user_id = %s" in query
    assert mock_fetch.await_args.args[1:] == ("tx-1", "user-1")


@pytest.mark.asyncio
async def test_cost_report_aggregates_tokens_and_cost(service):
    """Two LLM rows for the same transform → totals sum; cost
    sums precisely (Decimal); models_used is sorted unique pairs."""
    rows = [
        {
            "model_provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "operation_type": "extraction",
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "estimated_cost_usd": Decimal("0.0050"),
        },
        {
            "model_provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "operation_type": "extraction",
            "input_tokens": 500,
            "output_tokens": 100,
            "total_tokens": 600,
            "estimated_cost_usd": Decimal("0.0025"),
        },
    ]
    with patch(
        "graphora_server.services.usage_tracking.db.fetch",
        new=AsyncMock(return_value=rows),
    ):
        result = await service.get_transform_cost_report(
            transform_id="tx-1", user_id="user-1"
        )

    assert result["transform_id"] == "tx-1"
    assert result["total_calls"] == 2
    assert result["input_tokens"] == 1500
    assert result["output_tokens"] == 300
    assert result["total_tokens"] == 1800
    # String-precise sum: 0.0050 + 0.0025 = 0.0075. Pin the string
    # representation to catch float-conversion regressions.
    assert result["estimated_cost_usd"] == "0.0075"
    assert result["models_used"] == ["gemini:gemini-2.5-flash"]


@pytest.mark.asyncio
async def test_cost_report_distinguishes_unpriced_from_zero(service):
    """When NO row has a non-NULL estimated_cost_usd (e.g. the
    model wasn't in the pricing table), the top-level field
    returns None — not "0". This lets the agent distinguish
    "the LLM ran but we couldn't price it" from "no LLM ran"."""
    rows = [
        {
            "model_provider": "ollama",
            "model_name": "llama3.2",
            "operation_type": "extraction",
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "estimated_cost_usd": None,  # No pricing row
        },
    ]
    with patch(
        "graphora_server.services.usage_tracking.db.fetch",
        new=AsyncMock(return_value=rows),
    ):
        result = await service.get_transform_cost_report(
            transform_id="tx-1", user_id="user-1"
        )

    assert result["total_calls"] == 1
    assert result["total_tokens"] == 1200
    # The crucial distinction.
    assert result["estimated_cost_usd"] is None


@pytest.mark.asyncio
async def test_cost_report_zero_calls_when_no_rows(service):
    """No llm_usage rows → zero-aggregate. None for cost (nothing
    priced), empty list for models_used, empty dict for
    by_operation_type."""
    with patch(
        "graphora_server.services.usage_tracking.db.fetch",
        new=AsyncMock(return_value=[]),
    ):
        result = await service.get_transform_cost_report(
            transform_id="tx-empty", user_id="user-1"
        )

    assert result == {
        "transform_id": "tx-empty",
        "total_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": None,
        "models_used": [],
        "by_operation_type": {},
    }


@pytest.mark.asyncio
async def test_cost_report_partitions_by_operation_type(service):
    """schema_inference call + 2 extraction calls → by_operation_type
    has both buckets with correct sub-totals. Lets the agent see
    where the budget went without re-aggregating client-side."""
    rows = [
        {
            "model_provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "operation_type": "schema_inference",
            "input_tokens": 500,
            "output_tokens": 100,
            "total_tokens": 600,
            "estimated_cost_usd": Decimal("0.0030"),
        },
        {
            "model_provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "operation_type": "extraction",
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "estimated_cost_usd": Decimal("0.0050"),
        },
        {
            "model_provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "operation_type": "extraction",
            "input_tokens": 800,
            "output_tokens": 150,
            "total_tokens": 950,
            "estimated_cost_usd": Decimal("0.0040"),
        },
    ]
    with patch(
        "graphora_server.services.usage_tracking.db.fetch",
        new=AsyncMock(return_value=rows),
    ):
        result = await service.get_transform_cost_report(
            transform_id="tx-1", user_id="user-1"
        )

    by_op = result["by_operation_type"]
    assert set(by_op.keys()) == {"schema_inference", "extraction"}
    assert by_op["schema_inference"]["calls"] == 1
    assert by_op["schema_inference"]["total_tokens"] == 600
    assert by_op["schema_inference"]["estimated_cost_usd"] == "0.0030"

    assert by_op["extraction"]["calls"] == 2
    assert by_op["extraction"]["total_tokens"] == 1200 + 950
    assert by_op["extraction"]["estimated_cost_usd"] == "0.0090"


@pytest.mark.asyncio
async def test_cost_report_per_op_distinguishes_unpriced(service):
    """Same unpriced/None-vs-string contract at the per-op level:
    if every call in a bucket is unpriced, that bucket's
    estimated_cost_usd is None (not "0")."""
    rows = [
        {
            "model_provider": "ollama",
            "model_name": "llama3.2",
            "operation_type": "extraction",
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "estimated_cost_usd": None,
        },
    ]
    with patch(
        "graphora_server.services.usage_tracking.db.fetch",
        new=AsyncMock(return_value=rows),
    ):
        result = await service.get_transform_cost_report(
            transform_id="tx-1", user_id="user-1"
        )

    assert result["by_operation_type"]["extraction"]["estimated_cost_usd"] is None
    assert result["by_operation_type"]["extraction"]["calls"] == 1
