"""Utility to generate the OpenAPI schema without requiring external services."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def stub_dependencies() -> None:
    import types

    # Minimal Neo4j stubs
    neo4j_stub = types.ModuleType("neo4j")

    class GraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):  # pragma: no cover - simple stub
            raise RuntimeError("GraphDatabase driver stubbed for OpenAPI generation")

    class AsyncGraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):  # pragma: no cover - simple stub
            raise RuntimeError("AsyncGraphDatabase driver stubbed for OpenAPI generation")

    neo4j_stub.GraphDatabase = GraphDatabase
    neo4j_stub.AsyncGraphDatabase = AsyncGraphDatabase

    exceptions_stub = types.ModuleType("neo4j.exceptions")

    class _BaseNeo4jError(Exception):
        pass

    for name in [
        "ServiceUnavailable",
        "AuthError",
        "DatabaseError",
        "SessionExpired",
        "TransientError",
    ]:
        exceptions_stub.__dict__[name] = type(name, (_BaseNeo4jError,), {})

    neo4j_time_stub = types.ModuleType("neo4j.time")

    class DateTime:
        pass

    neo4j_time_stub.DateTime = DateTime

    sys.modules["neo4j"] = neo4j_stub
    sys.modules["neo4j.exceptions"] = exceptions_stub
    sys.modules["neo4j.time"] = neo4j_time_stub

    # Prefect stubs
    prefect_stub = types.ModuleType("prefect")
    prefect_stub.flow = lambda *args, **kwargs: (lambda func: func)
    prefect_stub.task = lambda *args, **kwargs: (lambda func: func)

    class _Logger:
        def info(self, *args, **kwargs):
            return None

    prefect_stub.get_run_logger = lambda: _Logger()
    prefect_stub.futures = types.SimpleNamespace()
    prefect_stub.context = types.SimpleNamespace()

    class _PrefectClient:
        async def __aenter__(self):  # pragma: no cover - simple stub
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover
            return False

        async def read_flow_run(self, *args, **kwargs):  # pragma: no cover
            return types.SimpleNamespace(
                state=types.SimpleNamespace(type=types.SimpleNamespace(value="COMPLETED"))
            )

    prefect_stub.get_client = lambda: _PrefectClient()
    sys.modules["prefect"] = prefect_stub

    # Prefect client schemas stubs
    prefect_client_stub = types.ModuleType("prefect.client")
    sys.modules["prefect.client"] = prefect_client_stub

    prefect_client_schemas_stub = types.ModuleType("prefect.client.schemas")
    sys.modules["prefect.client.schemas"] = prefect_client_schemas_stub

    prefect_client_schemas_objects_stub = types.ModuleType("prefect.client.schemas.objects")

    class FlowRun:
        """Stub for Prefect FlowRun"""
        pass

    prefect_client_schemas_objects_stub.FlowRun = FlowRun
    sys.modules["prefect.client.schemas.objects"] = prefect_client_schemas_objects_stub

    prefect_filesystems_stub = types.ModuleType("prefect.filesystems")

    class LocalFileSystem:
        def __init__(self, *args, **kwargs):
            pass

    prefect_filesystems_stub.LocalFileSystem = LocalFileSystem
    sys.modules["prefect.filesystems"] = prefect_filesystems_stub

    redis_stub = types.ModuleType("redis")

    class _RedisClient:
        def set(self, *args, **kwargs):  # pragma: no cover - simple stub
            return None

        def get(self, *args, **kwargs):  # pragma: no cover - simple stub
            return None

        def delete(self, *args, **kwargs):  # pragma: no cover - simple stub
            return None

    redis_stub.from_url = lambda *args, **kwargs: _RedisClient()
    sys.modules["redis"] = redis_stub


def write_openapi_snapshot(output_path: Path) -> None:
    stub_dependencies()

    from app.main import app

    schema = app.openapi()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=True))


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/snapshots/openapi.json")
    write_openapi_snapshot(output)
    print(f"OpenAPI schema written to {output}")
