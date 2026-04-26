"""MCP (Model Context Protocol) server for Graphora.

Exposes the extraction API as MCP tools for agent clients. Talks
HTTP to a running Graphora deployment via ``GRAPHORA_API_URL`` +
``GRAPHORA_AUTH_TOKEN`` — does not import the service modules
directly so the MCP process starts instantly without needing a
database.

Installed with the ``[mcp]`` extra. Run via the ``graphora-mcp``
console script or ``python -m graphora_server.mcp``.
"""
