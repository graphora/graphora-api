"""Unit tests for the MCP tool implementations.

Tests target ``_tool_impl_*`` async functions with a FakeAPIClient
that duck-types GraphoraClient. This keeps the test surface free
of the mcp SDK (FastMCP wiring is covered separately when [mcp]
is present) and free of httpx (the client itself has its own
tests against respx).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from graphora_server.mcp.server import (
    _tool_impl_extract_document,
    _tool_impl_get_evidence,
    _tool_impl_query_graph,
    _tool_impl_refine_ontology,
)


class FakeAPIClient:
    """Duck-typed stand-in for GraphoraClient.

    Records calls so tests can assert the right endpoint was used
    and returns whatever the test pre-seeded.
    """

    def __init__(
        self,
        *,
        upload_file_return: Optional[Dict[str, Any]] = None,
        upload_url_return: Optional[Dict[str, Any]] = None,
        graph_return: Optional[Dict[str, Any]] = None,
    ):
        self.upload_file_return = upload_file_return or {
            "transform_id": "tx_file",
            "status": "pending",
            "document_count": 1,
        }
        self.upload_url_return = upload_url_return or {
            "transform_id": "tx_url",
            "status": "pending",
            "document_count": 1,
        }
        self.graph_return = graph_return or {
            "nodes": [],
            "edges": [],
            "total_nodes": 0,
            "total_edges": 0,
        }
        self.calls: List[tuple] = []

    async def upload_file(
        self,
        file_path: str,
        ontology_id: Optional[str] = None,
        *,
        schemaless: bool = False,
    ) -> Dict[str, Any]:
        self.calls.append(("upload_file", file_path, ontology_id, schemaless))
        return self.upload_file_return

    async def upload_url(
        self,
        url: str,
        ontology_id: Optional[str] = None,
        *,
        schemaless: bool = False,
    ) -> Dict[str, Any]:
        self.calls.append(("upload_url", url, ontology_id, schemaless))
        return self.upload_url_return

    async def get_inferred_ontology(self, transform_id: str) -> Dict[str, Any]:
        self.calls.append(("get_inferred_ontology", transform_id))
        return {
            "transform_id": transform_id,
            "ontology_yaml": "version: '0.1.0'\n",
            "ontology": {"entities": {"Person": {}}, "relationships": {}},
            "stats": {"node_count": 1, "edge_count": 0},
        }

    async def finalize_ontology(self, transform_id: str) -> Dict[str, Any]:
        self.calls.append(("finalize_ontology", transform_id))
        return {
            "transform_id": transform_id,
            "ontology_id": "auto_refined_abc",
            "ontology_yaml": "version: '0.1.0'\n",
            "ontology": {"entities": {"Person": {}}, "relationships": {}},
            "stats": {"node_count": 1, "edge_count": 0},
        }

    async def get_graph(
        self, transform_id: str, *, limit: int = 1000, skip: int = 0
    ) -> Dict[str, Any]:
        self.calls.append(("get_graph", transform_id, limit, skip))
        return self.graph_return

    async def find_node(
        self,
        transform_id: str,
        node_id: str,
        *,
        page_size: int = 1000,
        max_pages: int = 10,
    ) -> Optional[Dict[str, Any]]:
        # Walk the same data get_graph would hand out but pretend
        # it lives across pages — lets us test get_evidence hits
        # the pagination path without a real HTTP client.
        self.calls.append(("find_node", transform_id, node_id))
        nodes = self.graph_return.get("nodes", []) or []
        if any(n.get("id") == node_id for n in nodes):
            return self.graph_return
        return None


# ---- extract_document ------------------------------------------------------


class TestExtractDocument:
    @pytest.mark.asyncio
    async def test_uploads_file_when_file_path_given(self) -> None:
        api = FakeAPIClient()
        result = await _tool_impl_extract_document(
            api, file_path="/tmp/a.pdf", url=None, ontology_id=None
        )
        assert result["transform_id"] == "tx_file"
        assert api.calls == [("upload_file", "/tmp/a.pdf", None, False)]

    @pytest.mark.asyncio
    async def test_uploads_url_when_url_given(self) -> None:
        api = FakeAPIClient()
        result = await _tool_impl_extract_document(
            api, file_path=None, url="https://example.com", ontology_id=None
        )
        assert result["transform_id"] == "tx_url"
        assert api.calls == [("upload_url", "https://example.com", None, False)]

    @pytest.mark.asyncio
    async def test_passes_ontology_id_through(self) -> None:
        api = FakeAPIClient()
        await _tool_impl_extract_document(
            api, file_path="/tmp/a.pdf", url=None, ontology_id="ont-42"
        )
        assert api.calls == [("upload_file", "/tmp/a.pdf", "ont-42", False)]

    @pytest.mark.asyncio
    async def test_rejects_both_inputs(self) -> None:
        api = FakeAPIClient()
        with pytest.raises(ValueError, match="Provide exactly one"):
            await _tool_impl_extract_document(
                api,
                file_path="/tmp/a.pdf",
                url="https://example.com",
                ontology_id=None,
            )

    @pytest.mark.asyncio
    async def test_rejects_neither_input(self) -> None:
        api = FakeAPIClient()
        with pytest.raises(ValueError, match="Provide exactly one"):
            await _tool_impl_extract_document(
                api, file_path=None, url=None, ontology_id=None
            )


# ---- query_graph -----------------------------------------------------------


class TestQueryGraph:
    def _graph(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": "n1",
                    "label": "Alice",
                    "type": "Person",
                    "properties": {"name": "Alice"},
                },
                {
                    "id": "n2",
                    "label": "Acme",
                    "type": "Organization",
                    "properties": {"name": "Acme Co."},
                },
                {
                    "id": "n3",
                    "label": "Bob",
                    "type": "Person",
                    "properties": {"name": "Bob"},
                },
            ],
            "edges": [
                {"id": "e1", "source": "n1", "target": "n2", "type": "WORKS_AT"},
                {"id": "e2", "source": "n1", "target": "n3", "type": "KNOWS"},
                {"id": "e3", "source": "n3", "target": "n2", "type": "WORKS_AT"},
            ],
            "total_nodes": 3,
            "total_edges": 3,
        }

    @pytest.mark.asyncio
    async def test_returns_trimmed_nodes_and_edges(self) -> None:
        api = FakeAPIClient(graph_return=self._graph())
        result = await _tool_impl_query_graph(
            api, transform_id="tx1", filter_type=None, limit=50
        )
        assert result["transform_id"] == "tx1"
        assert len(result["nodes"]) == 3
        # Nodes should carry id/label/type/summary — not raw properties.
        first = result["nodes"][0]
        assert set(first.keys()) == {"id", "label", "type", "summary"}
        assert first["summary"] == "Alice"

    @pytest.mark.asyncio
    async def test_filter_type_is_case_insensitive(self) -> None:
        api = FakeAPIClient(graph_return=self._graph())
        result = await _tool_impl_query_graph(
            api, transform_id="tx1", filter_type="person", limit=50
        )
        assert [n["id"] for n in result["nodes"]] == ["n1", "n3"]
        # Edges pruned to those whose endpoints both survive the filter.
        assert [e["id"] for e in result["edges"]] == ["e2"]

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max(self) -> None:
        # limit over 200 should not crash; client should be called with
        # the capped value.
        api = FakeAPIClient(graph_return=self._graph())
        await _tool_impl_query_graph(
            api, transform_id="tx1", filter_type=None, limit=10_000
        )
        # last call should have limit=200
        _, _, limit_seen, _ = api.calls[-1]
        assert limit_seen == 200


# ---- get_evidence ----------------------------------------------------------


class TestGetEvidence:
    def _graph(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": "n1",
                    "label": "Alice",
                    "type": "Person",
                    "properties": {
                        "name": "Alice",
                        "source_chunk_id": "chunk-42",
                        "source_text": "Alice joined Acme in 2019.",
                        "document_id": "doc-7",
                        "some_other_prop": "ignored",
                    },
                },
                {"id": "n2", "label": "Acme", "type": "Organization", "properties": {}},
            ],
            "edges": [
                {"id": "e1", "source": "n1", "target": "n2", "type": "WORKS_AT"},
                {"id": "e2", "source": "n2", "target": "n1", "type": "EMPLOYS"},
            ],
            "total_nodes": 2,
            "total_edges": 2,
        }

    @pytest.mark.asyncio
    async def test_returns_node_with_full_properties(self) -> None:
        api = FakeAPIClient(graph_return=self._graph())
        result = await _tool_impl_get_evidence(api, "tx1", "n1")
        assert result["node"]["properties"]["name"] == "Alice"
        assert result["node"]["properties"]["some_other_prop"] == "ignored"

    @pytest.mark.asyncio
    async def test_extracts_evidence_keys_only(self) -> None:
        api = FakeAPIClient(graph_return=self._graph())
        result = await _tool_impl_get_evidence(api, "tx1", "n1")
        evidence = result["evidence"]
        assert set(evidence.keys()) == {"source_chunk_id", "source_text", "document_id"}
        assert "some_other_prop" not in evidence

    @pytest.mark.asyncio
    async def test_splits_incoming_and_outgoing_edges(self) -> None:
        api = FakeAPIClient(graph_return=self._graph())
        result = await _tool_impl_get_evidence(api, "tx1", "n1")
        assert [e["id"] for e in result["outgoing_edges"]] == ["e1"]
        assert [e["id"] for e in result["incoming_edges"]] == ["e2"]

    @pytest.mark.asyncio
    async def test_unknown_node_returns_null_node(self) -> None:
        api = FakeAPIClient(graph_return=self._graph())
        result = await _tool_impl_get_evidence(api, "tx1", "no-such-node")
        assert result["node"] is None
        assert result["incoming_edges"] == []
        assert result["outgoing_edges"] == []
        assert result["evidence"] == {}


# ---- schemaless + refine_ontology -----------------------------------------


class TestSchemalessFlag:
    @pytest.mark.asyncio
    async def test_schemaless_passed_to_client_on_file(self) -> None:
        api = FakeAPIClient()
        await _tool_impl_extract_document(
            api, file_path="/tmp/a.pdf", url=None, ontology_id=None, schemaless=True
        )
        assert api.calls == [("upload_file", "/tmp/a.pdf", None, True)]

    @pytest.mark.asyncio
    async def test_schemaless_passed_to_client_on_url(self) -> None:
        api = FakeAPIClient()
        await _tool_impl_extract_document(
            api,
            file_path=None,
            url="https://example.com",
            ontology_id=None,
            schemaless=True,
        )
        assert api.calls == [("upload_url", "https://example.com", None, True)]

    @pytest.mark.asyncio
    async def test_schemaless_and_ontology_id_are_exclusive(self) -> None:
        api = FakeAPIClient()
        with pytest.raises(ValueError, match="mutually exclusive"):
            await _tool_impl_extract_document(
                api,
                file_path="/tmp/a.pdf",
                url=None,
                ontology_id="ont-1",
                schemaless=True,
            )


class TestRefineOntology:
    @pytest.mark.asyncio
    async def test_save_false_hits_inferred_endpoint(self) -> None:
        api = FakeAPIClient()
        result = await _tool_impl_refine_ontology(api, "tx-1", save=False)
        assert result["transform_id"] == "tx-1"
        assert "ontology_id" not in result
        assert api.calls == [("get_inferred_ontology", "tx-1")]

    @pytest.mark.asyncio
    async def test_save_true_hits_finalize_endpoint(self) -> None:
        api = FakeAPIClient()
        result = await _tool_impl_refine_ontology(api, "tx-1", save=True)
        assert result["ontology_id"] == "auto_refined_abc"
        assert api.calls == [("finalize_ontology", "tx-1")]
