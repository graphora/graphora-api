import pytest

from app.config import settings
from app.services.entity_ledger_service import EntityLedgerService
from app.services.transform.models import BaseNode
from app.services.transform.helpers import _make_canonical_node_id


@pytest.fixture(autouse=True)
def disable_supabase(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "")
    monkeypatch.setattr(settings, "SUPABASE_KEY", "")


@pytest.mark.asyncio
async def test_entity_ledger_memory_store_roundtrip():
    service = EntityLedgerService(supabase_client=None, memory_store={})

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
    service = EntityLedgerService(supabase_client=None, memory_store={})

    node = BaseNode(type="Company")
    await service.hydrate_nodes(None, [node])
    await service.record_nodes(None, [node])
    # no exceptions


class _DummyTable:
    def __init__(self) -> None:
        self.calls = []

    def upsert(self, data, on_conflict=None):  # type: ignore[override]
        self.calls.append((data, on_conflict))
        return self

    def execute(self):  # pragma: no cover - simple stub
        class _Response:
            data = []

        return _Response()


class _DummySupabaseClient:
    def __init__(self, table: _DummyTable) -> None:
        self._table = table

    def table(self, name: str):  # type: ignore[override]
        assert name == EntityLedgerService.TABLE_NAME
        return self._table


@pytest.mark.asyncio
async def test_entity_ledger_supabase_upsert_uses_composite_constraint(monkeypatch):
    dummy_table = _DummyTable()
    dummy_client = _DummySupabaseClient(dummy_table)
    service = EntityLedgerService(supabase_client=dummy_client, memory_store=None)

    node = BaseNode(
        type="Company",
        canonical_key="Company:name=acme",
        canonical_id="canonical-id",
        canonical_properties={"name": "acme"},
    )

    await service.record_nodes("user-1", [node])

    assert dummy_table.calls, "Expected upsert to be invoked"
    _, on_conflict = dummy_table.calls[0]
    assert on_conflict == "user_id,entity_type,canonical_key"
