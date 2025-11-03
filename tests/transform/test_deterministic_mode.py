from typing import List
from types import SimpleNamespace
import sys


class _SplinkStub(SimpleNamespace):
    def __init__(self):
        comparison_library = SimpleNamespace()
        super().__init__(
            block_on=lambda *args, **kwargs: None,
            DuckDBAPI=lambda *args, **kwargs: None,
            Linker=lambda *args, **kwargs: None,
            SettingsCreator=lambda *args, **kwargs: None,
            comparison_library=comparison_library,
        )


_splink_stub = _SplinkStub()
sys.modules.setdefault("splink", _splink_stub)
sys.modules.setdefault("splink.comparison_library", _splink_stub.comparison_library)

import pytest
from pydantic import BaseModel

from app.config import settings
from app.services.transform.helpers import (
    transform_as_nodes,
    _generate_node_key,
    _make_deterministic_node_id,
)
from app.services.transform.graph_transformer import (
    _build_nodes_context,
    _build_relationships_context,
)
from app.services.transform.models import BaseNode, RelationshipInstance


class CompanyModel(BaseModel):
    name: str
    ticker: str


class ExtractionResult(BaseModel):
    Company_list: List[CompanyModel]
    confidence_score: float = 0.9


@pytest.fixture
def sample_ontology() -> dict:
    return {
        "entities": {
            "Company": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "ticker": {"type": "string", "required": False},
                }
            }
        }
    }


@pytest.fixture
def extraction_result() -> ExtractionResult:
    return ExtractionResult(
        Company_list=[CompanyModel(name="Acme Corp", ticker="ACM")],
        confidence_score=0.9,
    )


@pytest.mark.parametrize("deterministic", [True, False])
def test_transform_as_nodes_id_strategy(sample_ontology, extraction_result, deterministic, monkeypatch):
    monkeypatch.setattr(settings, "DETERMINISTIC_MODE", deterministic)

    first = transform_as_nodes(sample_ontology, extraction_result, transform_id="tx-1")
    second = transform_as_nodes(sample_ontology, extraction_result, transform_id="tx-1")

    if deterministic:
        assert first[0].id == second[0].id
        other = transform_as_nodes(sample_ontology, extraction_result, transform_id="tx-2")
        assert first[0].id != other[0].id
    else:
        assert first[0].id != second[0].id


@pytest.mark.asyncio
async def test_nodes_context_is_stable(monkeypatch):
    monkeypatch.setattr(settings, "DETERMINISTIC_MODE", True)

    node_a = BaseNode(id="b", type="Company", properties={"name": "Beta"})
    node_b = BaseNode(id="a", type="Company", properties={"name": "Alpha"})

    context_one = await _build_nodes_context([node_a, node_b])
    context_two = await _build_nodes_context([node_b, node_a])

    assert context_one == context_two
    assert context_one.splitlines()[0].startswith("Node Type: Company, Id: a")


@pytest.mark.asyncio
async def test_relationship_context_is_stable(monkeypatch):
    monkeypatch.setattr(settings, "DETERMINISTIC_MODE", True)

    nodes = [
        BaseNode(id="1", type="Company", properties={"name": "Acme"}),
        BaseNode(id="2", type="Person", properties={"name": "Jane"}),
        BaseNode(id="3", type="Office", properties={"name": "HQ"}),
    ]

    rel_primary = RelationshipInstance(
        id="rel-b",
        type="EMPLOYS",
        source_id="1",
        target_id="2",
        source_type="Company",
        target_type="Person",
        properties={"role": "CEO"},
    )
    rel_secondary = RelationshipInstance(
        id="rel-a",
        type="OWNS",
        source_id="1",
        target_id="2",
        source_type="Company",
        target_type="Person",
        properties={"shares": "51%"},
    )

    context_one = await _build_relationships_context(nodes, [rel_primary, rel_secondary])
    context_two = await _build_relationships_context(nodes, [rel_secondary, rel_primary])

    assert context_one == context_two
    lines = [line for line in context_one.splitlines() if line and not line.startswith("These")]
    assert lines[0].startswith("(Company")
    assert "These Nodes without any relationships:" in context_one
    assert "Office" in context_one


def test_generate_node_key_fallback_to_raw_hash():
    parsed = {"entities": {"Item": {"properties": {}}}}
    key_a = _generate_node_key(
        parsed,
        "Item",
        properties={},
        raw_properties={"raw_text": "First"},
    )
    key_b = _generate_node_key(
        parsed,
        "Item",
        properties={},
        raw_properties={"raw_text": "Second"},
    )

    assert key_a != key_b
    assert key_a.startswith("Item:raw=")


def test_deterministic_node_id_varies_with_transform():
    node_key = "Item:raw=digest"
    first = _make_deterministic_node_id("tx-1", "Item", node_key)
    second = _make_deterministic_node_id("tx-1", "Item", node_key)
    other = _make_deterministic_node_id("tx-2", "Item", node_key)

    assert first == second
    assert first != other
