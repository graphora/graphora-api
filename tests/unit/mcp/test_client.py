"""Unit tests for the Graphora HTTP client used by the MCP server.

Uses httpx's MockTransport so we can exercise URL construction,
auth header behaviour, and error translation without a network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import httpx
import pytest

from graphora_server.mcp.client import (
    DEFAULT_API_URL,
    GraphoraClient,
    GraphoraClientError,
)


def _make_client(handler, base_url: str = DEFAULT_API_URL, token: str = "test-token"):
    """Construct a GraphoraClient wired to a MockTransport."""
    transport = httpx.MockTransport(handler)
    client = GraphoraClient(base_url=base_url, auth_token=token)
    client._client = httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        transport=transport,
    )
    return client


# ---- auth header ----------------------------------------------------------


class TestAuthHeader:
    @pytest.mark.asyncio
    async def test_bearer_token_sent(self, tmp_path: Path) -> None:
        seen_headers: Dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.update(request.headers)
            return httpx.Response(200, json={"transform_id": "tx1"})

        # Write a real file so upload_file's existence check passes.
        f = tmp_path / "doc.txt"
        f.write_text("hello")

        client = _make_client(handler)
        await client.upload_file(str(f))
        assert seen_headers.get("authorization") == "Bearer test-token"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_no_auth_header_when_token_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GRAPHORA_AUTH_TOKEN", raising=False)
        # Build a real client (no token) and verify the header is absent.
        client = GraphoraClient(base_url=DEFAULT_API_URL, auth_token=None)
        assert "Authorization" not in client._client.headers
        await client.aclose()


# ---- upload_file routing --------------------------------------------------


class TestUploadFile:
    @pytest.mark.asyncio
    async def test_auto_schema_endpoint_when_no_ontology(self, tmp_path: Path) -> None:
        seen_paths = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "transform_id": "tx1",
                    "status": "pending",
                    "document_count": 1,
                },
            )

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-")

        client = _make_client(handler)
        result = await client.upload_file(str(f))
        assert result["transform_id"] == "tx1"
        assert seen_paths == ["/api/v1/transform/upload"]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_ontology_endpoint_when_id_given(self, tmp_path: Path) -> None:
        seen_paths = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            return httpx.Response(
                200,
                json={"transform_id": "tx2", "status": "pending", "document_count": 1},
            )

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-")

        client = _make_client(handler)
        await client.upload_file(str(f), ontology_id="ont-9")
        assert seen_paths == ["/api/v1/transform/ont-9/upload"]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_raises_file_not_found(self) -> None:
        client = _make_client(lambda r: httpx.Response(200, json={}))
        with pytest.raises(FileNotFoundError):
            await client.upload_file("/no/such/file.pdf")
        await client.aclose()


# ---- get_graph / get_status -----------------------------------------------


class TestGraphEndpoints:
    @pytest.mark.asyncio
    async def test_get_graph_passes_pagination(self) -> None:
        seen_params: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.update(dict(request.url.params))
            return httpx.Response(
                200,
                json={
                    "nodes": [],
                    "edges": [],
                    "total_nodes": 0,
                    "total_edges": 0,
                },
            )

        client = _make_client(handler)
        await client.get_graph("tx1", limit=25, skip=10)
        assert seen_params == {"limit": "25", "skip": "10"}
        await client.aclose()

    @pytest.mark.asyncio
    async def test_find_node_finds_on_first_page(self) -> None:
        # Short-circuit on the first page when the node is there.
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(
                200,
                json={
                    "nodes": [{"id": "n42", "label": "x", "type": "T"}],
                    "edges": [],
                    "total_nodes": 1,
                    "total_edges": 0,
                },
            )

        client = _make_client(handler)
        result = await client.find_node("tx1", "n42")
        assert result is not None
        assert call_count["n"] == 1
        await client.aclose()

    @pytest.mark.asyncio
    async def test_find_node_paginates_until_found(self) -> None:
        """Regression test for the 200-node cap bug.

        A node living beyond the first page must still be found.
        Simulates a graph where page 1 is full (1000 nodes, none
        matching) and page 2 holds the target.
        """
        pages_served: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            skip = int(request.url.params.get("skip", "0"))
            pages_served.append(skip)
            if skip == 0:
                # Full first page, target not present.
                nodes = [
                    {"id": f"n{i}", "label": "x", "type": "T"} for i in range(1000)
                ]
            elif skip == 1000:
                nodes = [{"id": "target", "label": "hit", "type": "T"}]
            else:
                nodes = []
            return httpx.Response(
                200,
                json={
                    "nodes": nodes,
                    "edges": [],
                    "total_nodes": 1001,
                    "total_edges": 0,
                },
            )

        client = _make_client(handler)
        result = await client.find_node("tx1", "target", page_size=1000)
        assert result is not None
        assert any(n["id"] == "target" for n in result["nodes"])
        assert pages_served == [0, 1000]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_find_node_returns_none_when_exhausted(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # Always return an empty page — find_node should stop.
            return httpx.Response(
                200,
                json={"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0},
            )

        client = _make_client(handler)
        result = await client.find_node("tx1", "missing", page_size=1000)
        assert result is None
        await client.aclose()

    @pytest.mark.asyncio
    async def test_get_status_hits_status_endpoint(self) -> None:
        seen_paths = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            return httpx.Response(200, json={"status": "completed"})

        client = _make_client(handler)
        result = await client.get_status("tx1")
        assert seen_paths == ["/api/v1/transform/status/tx1"]
        assert result == {"status": "completed"}
        await client.aclose()


# ---- error translation ----------------------------------------------------


class TestErrors:
    @pytest.mark.asyncio
    async def test_4xx_raises_graphora_client_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        client = _make_client(handler)
        with pytest.raises(GraphoraClientError) as exc_info:
            await client.get_graph("nope")
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.body
        await client.aclose()

    @pytest.mark.asyncio
    async def test_5xx_raises_graphora_client_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server blew up")

        client = _make_client(handler)
        with pytest.raises(GraphoraClientError) as exc_info:
            await client.get_graph("tx1")
        assert exc_info.value.status_code == 500
        await client.aclose()
