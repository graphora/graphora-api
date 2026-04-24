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


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Avoid eager FastMCP import at module load — gives `--help`
    # a chance to run (and a cleaner error) without pulling the
    # whole mcp SDK when [mcp] isn't installed.
    from graphora_server.mcp.server import build_server

    server = build_server()
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
