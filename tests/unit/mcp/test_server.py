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
    _tool_impl_get_budget_status,
    _tool_impl_get_cost_report,
    _tool_impl_get_evidence,
    _tool_impl_label_disputed_pair,
    _tool_impl_list_disputed_pairs,
    _tool_impl_query_graph,
    _tool_impl_refine_ontology,
    _tool_impl_review_diff,
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
        cost_report_return: Optional[Dict[str, Any]] = None,
        cost_report_error: Optional[Exception] = None,
        budget_status_return: Optional[Dict[str, Any]] = None,
        budget_status_error: Optional[Exception] = None,
        diff_return: Optional[Dict[str, Any]] = None,
        diff_error: Optional[Exception] = None,
        list_disputed_pairs_return: Optional[List[Dict[str, Any]]] = None,
        list_disputed_pairs_error: Optional[Exception] = None,
        label_disputed_pair_return: Optional[Dict[str, Any]] = None,
        label_disputed_pair_error: Optional[Exception] = None,
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
        self.cost_report_return = cost_report_return or {
            "transform_id": "tx_default",
            "total_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": None,
            "models_used": [],
            "by_operation_type": {},
        }
        self.cost_report_error = cost_report_error
        self.budget_status_return = budget_status_return or {
            "state": "unset",
            "current_spend_usd": "0",
            "cap_usd": None,
            "period_start": "2026-05-01T00:00:00+00:00",
            "period_end": "2026-06-01T00:00:00+00:00",
        }
        self.budget_status_error = budget_status_error
        self.diff_return = diff_return or {
            "base_transform_id": "tx-base",
            "compare_transform_id": "tx-cmp",
            "summary": {
                "nodes": {"added": 0, "removed": 0, "changed": 0, "unchanged": 0},
                "edges": {"added": 0, "removed": 0, "changed": 0, "unchanged": 0},
            },
            "added_nodes": [],
            "removed_nodes": [],
            "changed_nodes": [],
            "added_edges": [],
            "removed_edges": [],
            "changed_edges": [],
        }
        self.diff_error = diff_error
        self.list_disputed_pairs_return = (
            list_disputed_pairs_return if list_disputed_pairs_return is not None else []
        )
        self.list_disputed_pairs_error = list_disputed_pairs_error
        self.label_disputed_pair_return = label_disputed_pair_return or {
            "id": "p-1",
            "status": "labeled_match",
        }
        self.label_disputed_pair_error = label_disputed_pair_error
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

    async def get_cost_report(self, transform_id: str) -> Dict[str, Any]:
        self.calls.append(("get_cost_report", transform_id))
        if self.cost_report_error is not None:
            raise self.cost_report_error
        return self.cost_report_return

    async def get_budget_status(self) -> Dict[str, Any]:
        self.calls.append(("get_budget_status",))
        if self.budget_status_error is not None:
            raise self.budget_status_error
        return self.budget_status_return

    async def diff_transforms(
        self,
        base_transform_id: str,
        compare_transform_id: str,
    ) -> Dict[str, Any]:
        self.calls.append(("diff_transforms", base_transform_id, compare_transform_id))
        if self.diff_error is not None:
            raise self.diff_error
        return self.diff_return

    async def list_disputed_pairs(
        self,
        transform_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        self.calls.append(("list_disputed_pairs", transform_id, limit, offset))
        if self.list_disputed_pairs_error is not None:
            raise self.list_disputed_pairs_error
        return self.list_disputed_pairs_return

    async def label_disputed_pair(
        self,
        pair_id: str,
        decision: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.calls.append(("label_disputed_pair", pair_id, decision, reason))
        if self.label_disputed_pair_error is not None:
            raise self.label_disputed_pair_error
        return self.label_disputed_pair_return


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

    @pytest.mark.asyncio
    async def test_get_decisions_transport_error_degrades_to_empty_arrays(
        self,
    ) -> None:
        """Reviewer-flagged on commit eb22a79 (P3). The earlier
        try/except caught only ``GraphoraClientError`` (raised
        AFTER the HTTP response is parsed). httpx transport errors
        (timeout, connect reset, DNS failure) raise
        ``httpx.HTTPError`` BEFORE _ok() runs and would have
        propagated, blanking the entire evidence response.

        Pin: a transport-layer failure on the decisions call is
        treated the same as a 5xx — the rest of the evidence
        response survives, decision_log + alternatives degrade
        to empty arrays. Without this catch, network flake on the
        decisions endpoint takes down the whole get_evidence tool
        for the agent."""
        import httpx

        api = FakeAPIClient(
            graph_return=self._graph(),
            decisions_error=httpx.ConnectTimeout("connect timed out"),
        )
        result = await _tool_impl_get_evidence(api, "tx1", "n1")

        # Source-span evidence + edges still present.
        assert result["node"]["properties"]["name"] == "Alice"
        assert result["evidence"]["source_chunk_id"] == "chunk-42"
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


# ---- get_cost_report ------------------------------------------------------


class TestGetCostReport:
    """B5-obs: agent-facing cost surface. Tool is a pure passthrough
    over the /cost endpoint — same architectural pattern as
    get_evidence's decision read (commit eb22a79).
    """

    @pytest.mark.asyncio
    async def test_passthrough_returns_endpoint_payload_unchanged(self) -> None:
        """Pin the passthrough contract: whatever the API returns,
        the tool surfaces. No client-side aggregation, no shape
        massaging — the API endpoint is the single source of truth
        for the cost report shape."""
        payload = {
            "transform_id": "tx-1",
            "total_calls": 12,
            "input_tokens": 4500,
            "output_tokens": 800,
            "total_tokens": 5300,
            "estimated_cost_usd": "0.0234",
            "models_used": ["gemini:gemini-2.5-flash"],
            "by_operation_type": {
                "schema_inference": {
                    "calls": 1,
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "total_tokens": 600,
                    "estimated_cost_usd": "0.0030",
                },
                "extraction": {
                    "calls": 11,
                    "input_tokens": 4000,
                    "output_tokens": 700,
                    "total_tokens": 4700,
                    "estimated_cost_usd": "0.0204",
                },
            },
        }
        api = FakeAPIClient(cost_report_return=payload)
        result = await _tool_impl_get_cost_report(api, "tx-1")

        assert result == payload
        assert api.calls == [("get_cost_report", "tx-1")]

    @pytest.mark.asyncio
    async def test_zero_call_transform_returns_zero_aggregate(self) -> None:
        """Empty transform → zero counts, models_used empty,
        estimated_cost_usd None (NOT "0" — the None vs "0"
        distinction lets callers tell "no LLM was invoked" apart
        from "the LLM ran but the model wasn't priced").
        Default FakeAPIClient.cost_report_return matches this
        shape."""
        api = FakeAPIClient()
        result = await _tool_impl_get_cost_report(api, "tx-empty")

        assert result["total_calls"] == 0
        assert result["estimated_cost_usd"] is None
        assert result["models_used"] == []
        assert result["by_operation_type"] == {}

    @pytest.mark.asyncio
    async def test_transport_error_propagates(self) -> None:
        """Cost is the headline answer the agent asked for — unlike
        decisions on get_evidence (observability OF observability),
        a cost-fetch failure has no fallback. Pin: errors propagate
        so the agent sees the real failure rather than silently
        getting a zero-aggregate."""
        from graphora_server.mcp.client import GraphoraClientError

        api = FakeAPIClient(
            cost_report_error=GraphoraClientError(503, "Service Unavailable"),
        )
        with pytest.raises(GraphoraClientError):
            await _tool_impl_get_cost_report(api, "tx-1")


# ---- get_budget_status -----------------------------------------------------


class TestGetBudgetStatus:
    """B5-obs slice 2: agent-facing budget status surface.
    Pure passthrough — same architectural pattern as
    get_cost_report. The agent uses this to predict whether the
    next transform will succeed."""

    @pytest.mark.asyncio
    async def test_passthrough_returns_endpoint_payload_unchanged(self) -> None:
        """Whatever the API returns, the tool surfaces verbatim.
        Pinning the wire shape so a refactor that "helpfully"
        coerces fields can't slip through."""
        payload = {
            "state": "near",
            "current_spend_usd": "85.0000",
            "cap_usd": "100.0000",
            "period_start": "2026-05-01T00:00:00+00:00",
            "period_end": "2026-06-01T00:00:00+00:00",
        }
        api = FakeAPIClient(budget_status_return=payload)
        result = await _tool_impl_get_budget_status(api)
        assert result == payload
        assert api.calls == [("get_budget_status",)]

    @pytest.mark.asyncio
    async def test_unset_state_returns_null_cap(self) -> None:
        """The default FakeAPIClient payload mirrors the
        unset-budget shape: state=unset, cap_usd=null. The agent
        layer renders this as 'no budget set' rather than
        '$0/$0'."""
        api = FakeAPIClient()
        result = await _tool_impl_get_budget_status(api)
        assert result["state"] == "unset"
        assert result["cap_usd"] is None

    @pytest.mark.asyncio
    async def test_transport_error_propagates(self) -> None:
        """Budget status is a primary signal for the agent's
        decision to submit work — propagate failures rather than
        silently report 'unset' (which would let the agent submit
        and then 402)."""
        from graphora_server.mcp.client import GraphoraClientError

        api = FakeAPIClient(
            budget_status_error=GraphoraClientError(503, "Service Unavailable"),
        )
        with pytest.raises(GraphoraClientError):
            await _tool_impl_get_budget_status(api)


# ---- review_diff -----------------------------------------------------------


class TestReviewDiff:
    """B3-diff backend: agent-facing graph-state diff surface.
    Pure passthrough — same architectural pattern as
    get_cost_report and get_budget_status."""

    @pytest.mark.asyncio
    async def test_passthrough_returns_endpoint_payload_unchanged(self) -> None:
        """Whatever the API returns, the tool surfaces verbatim.
        Pinning the wire shape so a "helpful" client-side
        re-aggregation can't slip into the tool."""
        payload = {
            "base_transform_id": "tx-1",
            "compare_transform_id": "tx-2",
            "summary": {
                "nodes": {"added": 5, "removed": 2, "changed": 1, "unchanged": 100},
                "edges": {"added": 3, "removed": 0, "changed": 1, "unchanged": 50},
            },
            "added_nodes": [{"id": "x", "type": "Person", "properties": {}}],
            "removed_nodes": [],
            "changed_nodes": [
                {
                    "canonical_id": "cid-alice",
                    "type": "Person",
                    "base_id": "a1",
                    "compare_id": "a2",
                    "property_changes": {
                        "role": {"base": "engineer", "compare": "principal"}
                    },
                }
            ],
            "added_edges": [],
            "removed_edges": [],
            "changed_edges": [],
        }
        api = FakeAPIClient(diff_return=payload)
        result = await _tool_impl_review_diff(api, "tx-1", "tx-2")

        assert result == payload
        assert api.calls == [("diff_transforms", "tx-1", "tx-2")]

    @pytest.mark.asyncio
    async def test_empty_diff_returns_zero_aggregate(self) -> None:
        """Identical transforms (or empty graphs) return the zero-
        aggregate shape, not a 'not found' structure — callers
        can render 'no differences' without conditional access."""
        api = FakeAPIClient()
        result = await _tool_impl_review_diff(api, "tx-x", "tx-y")

        assert result["summary"]["nodes"]["added"] == 0
        assert result["summary"]["nodes"]["removed"] == 0
        assert result["added_nodes"] == []
        assert result["changed_nodes"] == []

    @pytest.mark.asyncio
    async def test_transport_error_propagates(self) -> None:
        """Diff is a primary answer the agent asked for — same
        as cost report and budget status. Errors propagate rather
        than silently report 'no differences'."""
        from graphora_server.mcp.client import GraphoraClientError

        api = FakeAPIClient(
            diff_error=GraphoraClientError(500, "boom"),
        )
        with pytest.raises(GraphoraClientError):
            await _tool_impl_review_diff(api, "tx-1", "tx-2")


# ---- list_disputed_pairs / label_disputed_pair ----------------------------


class TestDisputedPairsTools:
    """B2-active backend slice A: agent-facing read+label surface
    for the disputed-pairs queue. Pure passthrough — same
    architectural pattern as get_cost_report and review_diff."""

    @pytest.mark.asyncio
    async def test_list_passthrough_returns_endpoint_payload_unchanged(
        self,
    ) -> None:
        payload = [
            {
                "id": "p-1",
                "user_id": "user-1",
                "transform_id": "tx-1",
                "node_a_id": "alice-1",
                "node_b_id": "alice-2",
                "entity_type": "Person",
                "similarity_score": "0.85",
                "source_stage": "embedding_blocker",
                "status": "pending",
                "created_at": "2026-05-14T00:00:00+00:00",
            },
        ]
        api = FakeAPIClient(list_disputed_pairs_return=payload)
        result = await _tool_impl_list_disputed_pairs(api)
        assert result == payload
        assert api.calls == [("list_disputed_pairs", None, 50, 0)]

    @pytest.mark.asyncio
    async def test_list_threads_transform_id_filter(self) -> None:
        api = FakeAPIClient(list_disputed_pairs_return=[])
        await _tool_impl_list_disputed_pairs(api, transform_id="tx-1")
        assert api.calls == [("list_disputed_pairs", "tx-1", 50, 0)]

    @pytest.mark.asyncio
    async def test_list_transport_error_propagates(self) -> None:
        """The queue is a primary answer the agent asked for —
        errors propagate so agents see real failures rather than
        an empty queue that masks a degraded backend."""
        from graphora_server.mcp.client import GraphoraClientError

        api = FakeAPIClient(
            list_disputed_pairs_error=GraphoraClientError(500, "boom"),
        )
        with pytest.raises(GraphoraClientError):
            await _tool_impl_list_disputed_pairs(api)

    @pytest.mark.asyncio
    async def test_label_passthrough_returns_endpoint_payload_unchanged(
        self,
    ) -> None:
        """Tool forwards pair_id + decision + reason to the
        client unchanged. Response from the endpoint surfaces
        verbatim so the agent can render the new status without
        re-fetching."""
        payload = {
            "id": "p-1",
            "status": "labeled_match",
            "labeled_at": "2026-05-14T01:00:00+00:00",
            "labeled_by_user_id": "user-1",
            "label_reason": "same person",
        }
        api = FakeAPIClient(label_disputed_pair_return=payload)
        result = await _tool_impl_label_disputed_pair(
            api, pair_id="p-1", decision="match", reason="same person"
        )
        assert result == payload
        assert api.calls == [("label_disputed_pair", "p-1", "match", "same person")]

    @pytest.mark.asyncio
    async def test_label_without_reason_passes_none(self) -> None:
        """The reason kwarg is optional — pin that omitting it
        passes None to the client rather than an empty string
        (which the API might reject or render confusingly)."""
        api = FakeAPIClient()
        await _tool_impl_label_disputed_pair(api, "p-1", "match")
        assert api.calls == [("label_disputed_pair", "p-1", "match", None)]

    @pytest.mark.asyncio
    async def test_label_transport_error_propagates(self) -> None:
        """Labels are writes — errors must propagate so the
        agent knows the label didn't land. Silent success would
        be much worse than a propagated 404 / 5xx."""
        from graphora_server.mcp.client import GraphoraClientError

        api = FakeAPIClient(
            label_disputed_pair_error=GraphoraClientError(404, "not found"),
        )
        with pytest.raises(GraphoraClientError):
            await _tool_impl_label_disputed_pair(api, "p-bad", "match")
