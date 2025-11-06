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
            raise RuntimeError(
                "AsyncGraphDatabase driver stubbed for OpenAPI generation"
            )

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
                state=types.SimpleNamespace(
                    type=types.SimpleNamespace(value="COMPLETED")
                )
            )

    prefect_stub.get_client = lambda: _PrefectClient()
    sys.modules["prefect"] = prefect_stub

    # Prefect client schemas stubs
    prefect_client_stub = types.ModuleType("prefect.client")
    sys.modules["prefect.client"] = prefect_client_stub

    prefect_client_schemas_stub = types.ModuleType("prefect.client.schemas")
    sys.modules["prefect.client.schemas"] = prefect_client_schemas_stub

    prefect_client_schemas_objects_stub = types.ModuleType(
        "prefect.client.schemas.objects"
    )

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
        def __init__(self):  # pragma: no cover
            self._store = {}

        @classmethod
        def from_url(cls, *args, **kwargs):  # pragma: no cover
            return cls()

        async def set(self, *args, **kwargs):  # pragma: no cover
            key, value = args[0], args[1]
            self._store[key] = value
            return True

        async def get(self, key):  # pragma: no cover
            return self._store.get(key)

        async def delete(self, key):  # pragma: no cover
            self._store.pop(key, None)
            return True

    redis_asyncio_stub = types.ModuleType("redis.asyncio")
    redis_asyncio_stub.Redis = _RedisClient

    redis_stub.from_url = _RedisClient.from_url  # Backwards compatibility
    redis_stub.asyncio = redis_asyncio_stub
    sys.modules["redis"] = redis_stub
    sys.modules["redis.asyncio"] = redis_asyncio_stub

    # LangChain semantic chunking stubs
    langchain_experimental_stub = types.ModuleType("langchain_experimental")
    text_splitter_stub = types.ModuleType("langchain_experimental.text_splitter")

    class SemanticChunker:
        """Lightweight stub mimicking LangChain's SemanticChunker"""

        def __init__(
            self, *args, **kwargs
        ):  # pragma: no cover - initialization is trivial
            self._config = kwargs

        def create_documents(self, texts):  # pragma: no cover - simple passthrough
            return [types.SimpleNamespace(page_content=text) for text in texts]

    text_splitter_stub.SemanticChunker = SemanticChunker
    langchain_experimental_stub.text_splitter = text_splitter_stub
    sys.modules["langchain_experimental"] = langchain_experimental_stub
    sys.modules["langchain_experimental.text_splitter"] = text_splitter_stub

    # LangChain HuggingFace embeddings stub
    langchain_hf_stub = types.ModuleType("langchain_huggingface")
    hf_embeddings_stub = types.ModuleType("langchain_huggingface.embeddings")

    class HuggingFaceEmbeddings:
        """Stub for HuggingFaceEmbeddings used during schema generation."""

        def __init__(
            self, *args, **kwargs
        ):  # pragma: no cover - initialization is trivial
            self._config = kwargs

        def embed_documents(self, texts):  # pragma: no cover - deterministic stub
            return [0.0 for _ in texts]

    hf_embeddings_stub.HuggingFaceEmbeddings = HuggingFaceEmbeddings
    langchain_hf_stub.embeddings = hf_embeddings_stub
    sys.modules["langchain_huggingface"] = langchain_hf_stub
    sys.modules["langchain_huggingface.embeddings"] = hf_embeddings_stub

    # LangChain recursive text splitter stub
    text_splitters_stub = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        """Stub matching the interface used in hybrid chunker."""

        def __init__(self, *args, **kwargs):  # pragma: no cover - no-op setup
            self._config = kwargs

        def create_documents(self, texts):  # pragma: no cover - deterministic stub
            return [types.SimpleNamespace(page_content=text) for text in texts]

    text_splitters_stub.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain_text_splitters"] = text_splitters_stub

    # Splink stubs to avoid heavy data science dependencies
    splink_stub = types.ModuleType("splink")

    def _noop(*args, **kwargs):  # pragma: no cover - simple placeholder
        return None

    class _SplinkPlaceholder:
        def __init__(self, *args, **kwargs):  # pragma: no cover - placeholder init
            self._config = kwargs

        def __call__(self, *args, **kwargs):  # pragma: no cover
            return None

    splink_stub.block_on = _noop
    splink_stub.DuckDBAPI = _SplinkPlaceholder
    splink_stub.Linker = _SplinkPlaceholder
    splink_stub.SettingsCreator = _SplinkPlaceholder

    comparison_library_stub = types.ModuleType("splink.comparison_library")
    sys.modules["splink"] = splink_stub
    sys.modules["splink.comparison_library"] = comparison_library_stub

    # Pandas stub to prevent NumPy initialization in sandbox
    pandas_stub = types.ModuleType("pandas")

    class _PandasDataFrame:  # pragma: no cover - placeholder for type hints
        def __init__(self, *args, **kwargs):
            self._data = kwargs

    pandas_stub.DataFrame = _PandasDataFrame
    pandas_stub.Series = _PandasDataFrame
    pandas_stub.concat = _noop
    sys.modules["pandas"] = pandas_stub


def write_openapi_snapshot(output_path: Path) -> None:
    stub_dependencies()

    from app.main import app

    schema = app.openapi()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=True))


if __name__ == "__main__":
    output = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/snapshots/openapi.json")
    )
    write_openapi_snapshot(output)
    print(f"OpenAPI schema written to {output}")
