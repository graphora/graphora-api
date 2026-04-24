"""Console entry point for the graphora-server package.

Registered as the `graphora-server` command by pyproject.toml's
[project.scripts] block. Also runs via `python -m graphora_server`.

Keep the subcommand surface narrow and additive: new verbs can be
introduced without renaming the entry point or changing existing
invocations.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover — exercised without [server]
        raise ImportError(
            "Running `graphora-server serve` requires the [server] extra. "
            "Install with: pip install 'graphora-server[server]'"
        ) from exc

    from graphora_server.config import settings
    from graphora_server.utils.logger import logger

    host = args.host or "0.0.0.0"
    port = args.port or settings.API_PORT
    log_level = (args.log_level or settings.LOG_LEVEL).lower()

    logger.info(
        "Starting uvicorn server on %s:%d (reload=%s, log_level=%s)",
        host,
        port,
        args.reload,
        log_level,
    )
    uvicorn.run(
        "graphora_server.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=log_level,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphora-server",
        description="Graphora server — extraction pipeline and HTTP API.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    serve = sub.add_parser(
        "serve",
        help="Run the HTTP server (uvicorn).",
    )
    serve.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0)")
    serve.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: settings.API_PORT)",
    )
    serve.add_argument(
        "--log-level",
        default=None,
        help="Uvicorn log level (default: settings.LOG_LEVEL)",
    )
    serve.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        # Default to `serve` with no flags for parity with previous
        # `python -m app.main` behaviour. Anyone who needs custom
        # flags should invoke `graphora-server serve --reload ...`.
        args = parser.parse_args(["serve"])

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
