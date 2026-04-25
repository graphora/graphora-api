"""Unit tests for the finalize-ontology endpoint.

Targets ``POST /api/v1/transform/{transform_id}/finalize-ontology``.
The companion to the inferred-ontology endpoint — same inference,
but persists the result so it can be referenced by id.

Covers the in-memory happy path, the staging-DB resource-cleanup
contract, and the empty-graph guard.
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


_VALID_LLM_YAML = """version: "0.1.0"
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


class TestFinalizeOntologyEndpoint:
    def test_happy_path_persists_and_returns_id(self, test_client: TestClient) -> None:
        mock_client = _mock_llm_response(_VALID_LLM_YAML)
        mock_store = AsyncMock(return_value=True)

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
            patch(
                "graphora_server.services.ontology_storage_service."
                "ontology_storage_service.store_ontology",
                new=mock_store,
            ),
        ):
            resp = test_client.post("/api/v1/transform/tx-1/finalize-ontology")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["transform_id"] == "tx-1"
        assert body["ontology_id"].startswith("auto_refined_")
        assert "Person" in body["ontology"]["entities"]
        # store_ontology was called with the new id and the rendered yaml.
        mock_store.assert_awaited_once()
        kwargs = mock_store.await_args.kwargs
        assert kwargs["user_id"] == "user-1"
        assert kwargs["ontology_id"].startswith("auto_refined_")
        assert "Person:" in kwargs["yaml_content"]

    def test_empty_graph_returns_404(self, test_client: TestClient) -> None:
        empty = GraphResponse(nodes=[], edges=[], total_nodes=0, total_edges=0)
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
                return_value=empty,
            ),
        ):
            resp = test_client.post("/api/v1/transform/tx-empty/finalize-ontology")

        assert resp.status_code == 404
        assert "empty graph" in resp.json()["detail"].lower()

    def test_staging_path_closes_graph_service(self, test_client: TestClient) -> None:
        """Regression — finalize must close staging graph_service."""
        mock_client = _mock_llm_response(_VALID_LLM_YAML)

        mock_graph_service = MagicMock()
        mock_graph_service.get_graph_by_transform_id.return_value = (
            _graph_with_two_types()
        )
        mock_graph_service.close = MagicMock()

        with (
            patch(
                "graphora_server.services.user_db_service.is_memory_storage_enabled",
                return_value=False,
            ),
            patch(
                "graphora_server.services.storage.factory.user_has_staging_db",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                __import__(
                    "graphora_server.api.transform", fromlist=["UserDatabaseService"]
                ).UserDatabaseService,
                "get_staging_graph_service",
                new=AsyncMock(return_value=mock_graph_service),
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
            patch(
                "graphora_server.services.ontology_storage_service."
                "ontology_storage_service.store_ontology",
                new=AsyncMock(return_value=True),
            ),
        ):
            resp = test_client.post("/api/v1/transform/tx-1/finalize-ontology")

        assert resp.status_code == 200, resp.text
        mock_graph_service.close.assert_called_once()
