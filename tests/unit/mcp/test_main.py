"""Tests for the graphora-mcp console script entry point.

Covers the error path when the [mcp] extra is missing — a
base-install user running `graphora-mcp` must see the friendly
install-hint rather than a raw ModuleNotFoundError from httpx/mcp.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


class TestConsoleScriptErrorMessages:
    def test_exits_with_install_hint_when_server_import_fails(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Simulate ModuleNotFoundError on the server import path.

        The entry point must:
        1. Return a non-zero exit code
        2. Print the [mcp] install hint to stderr
        """
        from graphora_server.mcp import __main__ as mcp_main

        # Patch the lazy import to blow up the way a bare install would.
        real_import = (
            __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
        )

        def fake_import(name: str, *args: Any, **kwargs: Any):
            if name == "graphora_server.mcp.server":
                raise ModuleNotFoundError("No module named 'httpx'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            rc = mcp_main.main([])

        captured = capsys.readouterr()
        assert rc == 1
        assert "graphora-mcp requires the [mcp] extra" in captured.err
        assert "pip install 'graphora-server[mcp]'" in captured.err

    def test_exits_with_install_hint_when_build_server_raises(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Simulate FastMCP missing — build_server's own guard fires."""
        from graphora_server.mcp import __main__ as mcp_main

        def boom(*_args, **_kwargs):
            raise ImportError(
                "MCP server requires the [mcp] extra. "
                "Install with: pip install 'graphora-server[mcp]'"
            )

        with patch.object(mcp_main, "main", wraps=mcp_main.main):
            with patch("graphora_server.mcp.server.build_server", side_effect=boom):
                rc = mcp_main.main([])

        captured = capsys.readouterr()
        assert rc == 1
        assert "MCP server requires the [mcp] extra" in captured.err
