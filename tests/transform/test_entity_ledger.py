import pytest

from app.config import settings
from app.services.entity_ledger_service import EntityLedgerService
from app.services.transform.models import BaseNode
from app.services.transform.helpers import _make_canonical_node_id


@pytest.fixture(autouse=True)
def disable_database(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)


@pytest.mark.asyncio
async def test_entity_ledger_memory_store_roundtrip():
    service = EntityLedgerService(memory_store={})

    canonical_key = "Company:name=acme"
    canonical_id = _make_canonical_node_id(canonical_key)

    node = BaseNode(
        id="local-1",
        type="Company",
        properties={"name": "Acme"},
        canonical_properties={"name": "acme"},
        canonical_key=canonical_key,
        canonical_id=canonical_id,
        confidence_score=0.9,
    )

    await service.hydrate_nodes("user-1", [node])  # nothing stored yet
    assert node.canonical_id == canonical_id

    await service.record_nodes("user-1", [node])

    node2 = BaseNode(
        id="local-2",
        type="Company",
        properties={"name": "ACME"},
        canonical_properties={"name": "acme"},
        canonical_key=canonical_key,
    )

    await service.hydrate_nodes("user-1", [node2])
    assert node2.canonical_id == canonical_id


@pytest.mark.asyncio
async def test_entity_ledger_ignore_missing_user():
    service = EntityLedgerService(memory_store={})

    node = BaseNode(type="Company")
    await service.hydrate_nodes(None, [node])
    await service.record_nodes(None, [node])
    # no exceptions

