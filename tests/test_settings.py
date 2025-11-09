"""Unit tests for app configuration helpers."""

from app.config import Settings


def test_settings_split_cors_origins_from_env(monkeypatch):
    """Comma-separated strings should become a list of trimmed origins."""

    monkeypatch.setenv(
        "CORS_ORIGINS",
        "http://localhost:3000, http://127.0.0.1:5173 ,",
    )

    settings = Settings(_env_file=None)

    assert settings.CORS_ORIGINS == [
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]


def test_settings_normalizes_log_level(monkeypatch):
    """Log level strings are upper-cased so logging accepts them."""

    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = Settings(_env_file=None)

    assert settings.LOG_LEVEL == "DEBUG"
