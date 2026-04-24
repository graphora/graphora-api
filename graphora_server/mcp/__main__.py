"""Console entry point for the `graphora-mcp` command.

Runs the FastMCP server over stdio — the transport agent clients
(Claude Desktop, Cursor, etc.) expect when they launch a local
MCP process.

Config via env:
    GRAPHORA_API_URL      (default: http://localhost:8000)
    GRAPHORA_AUTH_TOKEN   (required unless the server has auth bypass on)
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence


_INSTALL_HINT = (
    "graphora-mcp requires the [mcp] extra. "
    "Install with: pip install 'graphora-server[mcp]'"
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # The script is registered unconditionally by
    # [project.scripts] (so `graphora-mcp --help` is discoverable
    # on any install) but its dependencies — httpx, mcp SDK,
    # trafilatura — only ship with [mcp]. Deferring every import
    # into main() lets us translate a bare ModuleNotFoundError
    # into the same "install [mcp]" message users see everywhere
    # else in the codebase.
    try:
        from graphora_server.mcp.server import build_server
    except ImportError as exc:
        print(f"{_INSTALL_HINT}\nUnderlying error: {exc}", file=sys.stderr)
        return 1

    try:
        server = build_server()
    except ImportError as exc:
        # build_server itself imports FastMCP lazily and re-raises
        # ImportError with its own install-hint. Surface it as-is.
        print(str(exc), file=sys.stderr)
        return 1

    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
