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
        decisions_return: Optional[Dict[str, Any]] = None,
        decisions_error: Optional[Exception] = None,
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
        self.decisions_return = decisions_return or {
            "decision_log": [],
            "alternatives": [],
        }
        self.decisions_error = decisions_error
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

    async def get_decisions(
        self,
        transform_id: str,
        node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Mirrors GraphoraClient.get_decisions: returns whatever the
        # test pre-seeded, raises whatever the test injected. The
        # MCP tool catches GraphoraClientError specifically and
        # degrades to empty arrays — pinned by
        # test_get_decisions_failure_degrades_to_empty.
        self.calls.append(("get_decisions", transform_id, node_id))
        if self.decisions_error is not None:
            raise self.decisions_error
        return self.decisions_return


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
    @pytest.fixture(autouse=True)
    def _force_memory_mode(self, monkeypatch):
        """B0-explain (slice 4) made ``_tool_impl_get_evidence``
        construct a ``DecisionLogService`` whenever the caller
        doesn't inject one. ``tests/conftest.py`` sets a default
        ``DATABASE_URL`` for tests that need it, which would flip
        the service into Postgres mode and open a psycopg pool on
        every for_target / for_transform call (each test pays
        seconds of pool init against an unreachable DB).

        Force memory mode at the class boundary so all tests in
        this file run cheaply. Same pattern as
        TestCreateAutoSchemaOntology (commit 82eaaba) and
        TestCompareAndMergeNodesEmitsDecisions."""
        from graphora_server.config import settings

        monkeypatch.setattr(settings, "DATABASE_URL", "")
        monkeypatch.setattr(settings, "POSTGRES_HOST", None)

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
        # B0-explain: even on the not-found path, the new fields
        # appear with empty lists so consumers can rely on the
        # response schema without conditional access.
        assert result["decision_log"] == []
        assert result["alternatives"] == []

    @pytest.mark.asyncio
    async def test_known_node_with_no_decisions_returns_empty_lists(self) -> None:
        """B0-explain forward-compat: when the Decision Log has
        nothing for this node and no schema-level decisions for
        the transform, decision_log + alternatives are empty —
        but the keys are still present in the response. Pre-fix
        Gate 3 callers' contract (node + evidence + edges) is
        unchanged; the new fields just add alongside."""
        api = FakeAPIClient(graph_return=self._graph())
        result = await _tool_impl_get_evidence(api, "tx1", "n1")
        # Pre-existing contract preserved.
        assert result["node"]["properties"]["name"] == "Alice"
        assert "evidence" in result
        # New fields: present, empty.
        assert result["decision_log"] == []
        assert result["alternatives"] == []

    @pytest.mark.asyncio
    async def test_get_evidence_reads_decisions_via_api_not_local_service(
        self,
    ) -> None:
        """Reviewer-flagged on commit 9ac9bb5 (P1): the MCP server
        is a pure HTTP client. Reading decisions from a local
        DecisionLogService inside the MCP process either silently
        falls into an empty in-memory store (when the operator only
        configured GRAPHORA_API_URL) or opens a new direct DB
        dependency / secret surface (when DATABASE_URL is set
        locally). Both are wrong.

        Pin the contract: MCP MUST invoke api.get_decisions; the
        FakeAPIClient records that call. If MCP regresses to
        constructing a local service, this assertion fires."""
        api = FakeAPIClient(
            graph_return=self._graph(),
            decisions_return={
                "decision_log": [
                    {
                        "id": "d1",
                        "transform_id": "tx1",
                        "target_id": "n1",
                        "target_kind": "node",
                        "decision_type": "entity_merged",
                        "reason": "merge n1",
                        "evidence": {"stage": "property_blocker"},
                        "alternatives": [{"id": "n1-alias"}],
                        "created_at": "2026-05-09T00:00:00+00:00",
                    }
                ],
                "alternatives": [{"id": "n1-alias"}],
            },
        )
        result = await _tool_impl_get_evidence(api, "tx1", "n1")

        # The api.get_decisions call must happen — pin that the
        # MCP tool delegated to the HTTP boundary rather than going
        # to a local DB.
        decision_calls = [c for c in api.calls if c[0] == "get_decisions"]
        assert decision_calls == [("get_decisions", "tx1", "n1")], (
            f"_tool_impl_get_evidence must read decisions through "
            f"api.get_decisions, not a local DecisionLogService. "
            f"Calls seen: {api.calls}"
        )

        # Response surfaces what the API returned, untouched.
        assert result["decision_log"] == api.decisions_return["decision_log"]
        assert result["alternatives"] == api.decisions_return["alternatives"]

    @pytest.mark.asyncio
    async def test_get_decisions_failure_degrades_to_empty_arrays(self) -> None:
        """A transient API error fetching decisions must not blank
        the rest of the evidence response. The agent still gets
        node + edges + source-span evidence; decision_log and
        alternatives degrade to empty arrays. Decision Log is
        observability — losing it is regrettable but extracting
        ``why is this fact here?`` from the source span alone is
        still useful."""
        from graphora_server.mcp.client import GraphoraClientError

        api = FakeAPIClient(
            graph_return=self._graph(),
            decisions_error=GraphoraClientError(503, "Service Unavailable"),
        )
        result = await _tool_impl_get_evidence(api, "tx1", "n1")

        # Pre-existing contract preserved — the rest of the
        # evidence response survives a decisions API outage.
        assert result["node"]["properties"]["name"] == "Alice"
        assert result["evidence"]["source_chunk_id"] == "chunk-42"
        assert len(result["incoming_edges"]) + len(result["outgoing_edges"]) >= 1

        # Decision-related keys present, empty.
        assert result["decision_log"] == []
        assert result["alternatives"] == []


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
