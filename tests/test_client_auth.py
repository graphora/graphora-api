import os
import sys
import warnings
from pathlib import Path

import pytest

# Ensure project root is on sys.path for in-place testing without installation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

graphora_client_module = pytest.importorskip(
    "graphora.client",
    reason="graphora client package not available for import",
)
GraphoraClient = graphora_client_module.GraphoraClient


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("GRAPHORA_AUTH_TOKEN", "GRAPHORA_API_KEY", "GRAPHORA_USER_ID"):
        monkeypatch.delenv(key, raising=False)


def test_headers_use_explicit_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)

    client = GraphoraClient(base_url="https://example", auth_token="token-123")

    assert client.headers["Authorization"] == "Bearer token-123"
    assert "user-id" not in client.headers


def test_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("GRAPHORA_AUTH_TOKEN", "env-token")

    client = GraphoraClient(base_url="https://example")

    assert client.headers["Authorization"] == "Bearer env-token"


def test_api_key_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("GRAPHORA_API_KEY", "legacy-token")
    monkeypatch.setenv("GRAPHORA_AUTH_TOKEN", "env-token")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GraphoraClient(base_url="https://example")

    assert any("deprecated" in str(item.message).lower() for item in caught)
