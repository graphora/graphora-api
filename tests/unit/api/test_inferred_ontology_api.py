"""Unit tests for the inferred-ontology endpoint.

Targets ``GET /api/v1/transform/{transform_id}/inferred-ontology``.
Mocks the graph storage and the LLM client so the endpoint can
exercise its happy path, empty-graph guard, and LLM-failure guard
without a database or network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from graphora_server.auth import get_current_user_id
from graphora_server.main import app
from graphora_server.schemas.graph import Edge, GraphResponse, Node


@pytest.fixture
def test_client():
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_user_id, None)


def _graph_with_two_types() -> GraphResponse:
    return GraphResponse(
        nodes=[
            Node(
                id="n1",
                label="Alice",
                type="Person",
                properties={"name": "Alice"},
            ),
            Node(
                id="n2",
                label="Acme",
                type="Organization",
                properties={"name": "Acme"},
            ),
        ],
        edges=[Edge(id="e1", source="n1", target="n2", type="WORKS_AT")],
        total_nodes=2,
        total_edges=1,
    )


def _mock_llm_response(text: str) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.text = text
    client.models.generate_content.return_value = resp
    return client


class TestInferredOntologyEndpoint:
    def test_happy_path_returns_yaml_and_stats(self, test_client: TestClient) -> None:
        mock_client = _mock_llm_response(
            """version: "0.1.0"
entities:
  Person:
    description: "A person"
    properties:
      name:
        type: str
        required: true
  Organization:
    description: "A company"
    properties:
      name:
        type: str
relationships:
  WORKS_AT:
    source: Person
    target: Organization
    properties: {}
"""
        )

        # Patch the whole path the endpoint walks through: memory-storage
        # check, graph retrieval, LLM creds + client.
        with (
            patch(
                "graphora_server.services.user_db_service.is_memory_storage_enabled",
                return_value=True,
            ),
            patch(
                "graphora_server.services.storage.factory.user_has_staging_db",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "graphora_server.services.storage.memory.InMemoryStorage."
                "get_transformation_data",
                new_callable=AsyncMock,
                return_value=_graph_with_two_types(),
            ),
            patch(
                "graphora_server.services.schema_postprocess.get_user_llm_credentials",
                new_callable=AsyncMock,
                return_value=("k", "m"),
            ),
            patch(
                "graphora_server.services.schema_postprocess.create_gemini_client",
                return_value=mock_client,
            ),
        ):
            resp = test_client.get("/api/v1/transform/tx-1/inferred-ontology")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["transform_id"] == "tx-1"
        assert "Person" in body["ontology"]["entities"]
        assert "Organization" in body["ontology"]["entities"]
        assert "WORKS_AT" in body["ontology"]["relationships"]
        assert body["stats"]["node_count"] == 2
        assert body["stats"]["edge_count"] == 1
        assert body["stats"]["entity_types"] == 2
        assert body["stats"]["relationship_types"] == 1
        assert "Person:" in body["ontology_yaml"]

    def test_empty_graph_returns_404(self, test_client: TestClient) -> None:
        empty_graph = GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)
        with (
            patch(
                "graphora_server.services.user_db_service.is_memory_storage_enabled",
                return_value=True,
            ),
            patch(
                "graphora_server.services.storage.factory.user_has_staging_db",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "graphora_server.services.storage.memory.InMemoryStorage."
                "get_transformation_data",
                new_callable=AsyncMock,
                return_value=empty_graph,
            ),
        ):
            resp = test_client.get("/api/v1/transform/tx-empty/inferred-ontology")

        assert resp.status_code == 404
        body = resp.json()
        assert "no extracted nodes" in body["detail"]

    def test_bad_llm_output_returns_400(self, test_client: TestClient) -> None:
        # LLM returns empty text → service raises ValueError → endpoint 400s.
        mock_client = _mock_llm_response("")
        with (
            patch(
                "graphora_server.services.user_db_service.is_memory_storage_enabled",
                return_value=True,
            ),
            patch(
                "graphora_server.services.storage.factory.user_has_staging_db",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "graphora_server.services.storage.memory.InMemoryStorage."
                "get_transformation_data",
                new_callable=AsyncMock,
                return_value=_graph_with_two_types(),
            ),
            patch(
                "graphora_server.services.schema_postprocess.get_user_llm_credentials",
                new_callable=AsyncMock,
                return_value=("k", "m"),
            ),
            patch(
                "graphora_server.services.schema_postprocess.create_gemini_client",
                return_value=mock_client,
            ),
        ):
            resp = test_client.get("/api/v1/transform/tx-1/inferred-ontology")

        assert resp.status_code == 400
        assert "Empty response" in resp.json()["detail"]
