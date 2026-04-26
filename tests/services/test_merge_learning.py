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
