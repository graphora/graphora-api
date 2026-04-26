import os
import sys
import types
from pathlib import Path

import pytest


def _configure_test_environment() -> None:
    os.environ.setdefault("TEST_MODE", "true")
    os.environ.setdefault(
        "DATABASE_URL",
        os.environ.get(
            "DATABASE_URL",
            "postgresql://graphora:graphora@localhost:5432/graphora",
        ),
    )


def _ensure_project_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_settings():
    from graphora_server.config import settings as app_settings

    return app_settings


_configure_test_environment()
_ensure_project_on_path()
settings = _load_settings()


def _install_neo4j_stub() -> None:
    """Install lightweight neo4j stubs to avoid loading native deps during tests."""

    if (
        os.environ.get("GRAPHORA_TEST_REAL_NEO4J") == "1"
    ):  # Allow opt-out when real driver present
        return

    if "neo4j" in sys.modules:
        return

    neo4j_stub = types.ModuleType("neo4j")

    class GraphDatabase:
        @staticmethod
        def driver(
            *args, **kwargs
        ):  # pragma: no cover - consistent failure for unexpected usage
            raise RuntimeError("Neo4j GraphDatabase driver is stubbed for tests")

    class AsyncGraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):  # pragma: no cover
            raise RuntimeError("Neo4j AsyncGraphDatabase driver is stubbed for tests")

    neo4j_stub.GraphDatabase = GraphDatabase
    neo4j_stub.AsyncGraphDatabase = AsyncGraphDatabase

    exceptions_stub = types.ModuleType("neo4j.exceptions")
    for name in [
        "ServiceUnavailable",
        "AuthError",
        "DatabaseError",
        "SessionExpired",
        "TransientError",
    ]:
        exceptions_stub.__dict__[name] = type(name, (Exception,), {})

    time_stub = types.ModuleType("neo4j.time")

    class DateTime:  # pragma: no cover - placeholder
        pass

    time_stub.DateTime = DateTime

    optional_stub = types.ModuleType("neo4j._optional_deps")

    sys.modules["neo4j"] = neo4j_stub
    sys.modules["neo4j.exceptions"] = exceptions_stub
    sys.modules["neo4j.time"] = time_stub
    sys.modules["neo4j._optional_deps"] = optional_stub


_install_neo4j_stub()
settings.test_mode = True


def _install_langchain_and_splink_stubs() -> None:
    """Install lightweight langchain/splink stubs for tests without native deps.

    The E2E provider harness (tests/e2e/providers/**) needs the real
    modules — it exercises the live BAML extraction + splink ER path.
    Set GRAPHORA_TEST_REAL_DEPS=1 to skip ALL stubbing (langchain,
    splink, redis, pandas). The provider-e2e workflow sets this.

    GRAPHORA_TEST_REAL_LANGCHAIN=1 is retained as a legacy/narrower
    opt-out that only skips the langchain subset.
    """

    if (
        os.environ.get("GRAPHORA_TEST_REAL_DEPS") == "1"
        or os.environ.get("GRAPHORA_TEST_REAL_LANGCHAIN") == "1"
    ):
        return

    # LangChain semantic chunker stub
    if "langchain_experimental.text_splitter" not in sys.modules:
        langchain_experimental_stub = types.ModuleType("langchain_experimental")
        text_splitter_stub = types.ModuleType("langchain_experimental.text_splitter")

        class SemanticChunker:
            def __init__(self, *args, **kwargs):  # pragma: no cover
                self._config = kwargs

            def create_documents(self, texts):  # pragma: no cover
                return [types.SimpleNamespace(page_content=text) for text in texts]

        text_splitter_stub.SemanticChunker = SemanticChunker
        langchain_experimental_stub.text_splitter = text_splitter_stub
        sys.modules["langchain_experimental"] = langchain_experimental_stub
        sys.modules["langchain_experimental.text_splitter"] = text_splitter_stub

    # LangChain HF embeddings stub
    if "langchain_huggingface.embeddings" not in sys.modules:
        langchain_hf_stub = types.ModuleType("langchain_huggingface")
        hf_embeddings_stub = types.ModuleType("langchain_huggingface.embeddings")

        class HuggingFaceEmbeddings:
            def __init__(self, *args, **kwargs):  # pragma: no cover
                self._config = kwargs

            def embed_documents(self, texts):  # pragma: no cover
                return [0.0 for _ in texts]

        hf_embeddings_stub.HuggingFaceEmbeddings = HuggingFaceEmbeddings
        langchain_hf_stub.embeddings = hf_embeddings_stub
        sys.modules["langchain_huggingface"] = langchain_hf_stub
        sys.modules["langchain_huggingface.embeddings"] = hf_embeddings_stub

    # LangChain recursive text splitter stub
    if "langchain_text_splitters" not in sys.modules:
        text_splitters_stub = types.ModuleType("langchain_text_splitters")

        class RecursiveCharacterTextSplitter:
            def __init__(self, *args, **kwargs):  # pragma: no cover
                self._config = kwargs

            def create_documents(self, texts):  # pragma: no cover
                return [types.SimpleNamespace(page_content=text) for text in texts]

        text_splitters_stub.RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter
        )
        sys.modules["langchain_text_splitters"] = text_splitters_stub

    # Splink stub
    if "splink" not in sys.modules:
        splink_stub = types.ModuleType("splink")

        def _noop(*args, **kwargs):  # pragma: no cover
            return None

        class _Placeholder:
            def __init__(self, *args, **kwargs):  # pragma: no cover
                self._config = kwargs

            def __call__(self, *args, **kwargs):  # pragma: no cover
                return None

        splink_stub.block_on = _noop
        splink_stub.DuckDBAPI = _Placeholder
        splink_stub.Linker = _Placeholder
        splink_stub.SettingsCreator = _Placeholder

        comparison_library_stub = types.ModuleType("splink.comparison_library")

        sys.modules["splink"] = splink_stub
        sys.modules["splink.comparison_library"] = comparison_library_stub

    # Redis stub to avoid native client dependency
    if "redis.asyncio" not in sys.modules:
        redis_stub = sys.modules.get("redis")
        if redis_stub is None:
            redis_stub = types.ModuleType("redis")
            sys.modules["redis"] = redis_stub

        redis_asyncio_stub = types.ModuleType("redis.asyncio")

        class _StubAsyncRedis:  # pragma: no cover - lightweight test double
            def __init__(self):
                self._store = {}

            @classmethod
            def from_url(cls, *args, **kwargs):
                return cls()

            async def get(self, key):
                return self._store.get(key)

            async def set(self, key, value, ex=None):
                del ex
                self._store[key] = value
                return True

            async def delete(self, key):
                self._store.pop(key, None)
                return True

        redis_asyncio_stub.Redis = _StubAsyncRedis
        redis_stub.from_url = (
            _StubAsyncRedis.from_url
        )  # Compatibility with sync helpers
        redis_stub.asyncio = redis_asyncio_stub
        sys.modules["redis.asyncio"] = redis_asyncio_stub

    # Pandas stub to avoid numpy on sandbox
    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")

        class _MiniSeries:
            def __init__(self, data):  # pragma: no cover
                self._data = list(data)
                if all(isinstance(v, str) or v is None for v in self._data):
                    self.dtype = "object"
                elif all(isinstance(v, (int, float)) or v is None for v in self._data):
                    self.dtype = "number"
                else:
                    self.dtype = "mixed"

            def notna(self):  # pragma: no cover
                return _MiniSeries([value is not None for value in self._data])

            def sum(self):  # pragma: no cover
                return sum(self._data)

            def __iter__(self):  # pragma: no cover
                return iter(self._data)

            def __len__(self):  # pragma: no cover
                return len(self._data)

            def __getitem__(self, idx):  # pragma: no cover
                return self._data[idx]

            def dropna(self):  # pragma: no cover
                return _MiniSeries([value for value in self._data if value is not None])

        class _AtIndexer:
            def __init__(self, frame):  # pragma: no cover
                self._frame = frame

            def __setitem__(self, key, value):  # pragma: no cover
                row, column = key
                self._frame._set_value(row, column, value)

        class _LocIndexer:
            def __init__(self, frame):  # pragma: no cover
                self._frame = frame

            def __getitem__(self, key):  # pragma: no cover
                if isinstance(key, tuple) and len(key) == 2:
                    row, column = key
                    return self._frame._get_value(row, column)
                raise KeyError("MiniDataFrame.loc expects (row, column)")

        class DataFrame:  # pragma: no cover
            def __init__(self, data=None):
                rows = data or []
                self._rows = [dict(row) for row in rows]
                columns = set()
                for row in self._rows:
                    columns.update(row.keys())
                self._columns = list(columns)
                for row in self._rows:
                    for col in self._columns:
                        row.setdefault(col, None)
                self.at = _AtIndexer(self)
                self.loc = _LocIndexer(self)

            @property
            def columns(self):
                return list(self._columns)

            def _ensure_column(self, column):
                if column not in self._columns:
                    self._columns.append(column)
                    for row in self._rows:
                        row[column] = None

            def __setitem__(self, column, value):
                self._ensure_column(column)
                if isinstance(value, list):
                    for row, val in zip(self._rows, value):
                        row[column] = val
                else:
                    for row in self._rows:
                        row[column] = value

            def __getitem__(self, column):
                if column not in self._columns:
                    raise KeyError(column)
                return _MiniSeries(row[column] for row in self._rows)

            def get(self, column, default=None):
                try:
                    return self.__getitem__(column)
                except KeyError:
                    return default

            def _set_value(self, row_idx, column, value):
                self._ensure_column(column)
                while len(self._rows) <= row_idx:
                    self._rows.append({col: None for col in self._columns})
                self._rows[row_idx][column] = value

            def _get_value(self, row_idx, column):
                if row_idx >= len(self._rows):
                    raise IndexError(row_idx)
                return self._rows[row_idx].get(column)

            def __len__(self):
                return len(self._rows)

            @property
            def shape(self):
                return (len(self._rows), len(self._columns))

            @property
            def index(self):
                return range(len(self._rows))

        pandas_stub.DataFrame = DataFrame
        pandas_stub.Series = _MiniSeries
        pandas_stub.concat = lambda *args, **kwargs: DataFrame()
        sys.modules["pandas"] = pandas_stub


_install_langchain_and_splink_stubs()


@pytest.fixture(autouse=True)
def _reset_merge_learning_service():
    from graphora_server.services.merge.learning import merge_learning_service

    merge_learning_service.reset()
    yield
    merge_learning_service.reset()


# Provide placeholder Supabase settings so service singletons configure without
# raising during import in CI environments.
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")
