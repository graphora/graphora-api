"""Unit tests for app configuration helpers."""

from graphora_server.config import Settings


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


def test_cross_document_entity_resolution_default_is_on(monkeypatch):
    """Slice 4: ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED defaults to
    True. Slice 2 shipped the feature as opt-in (default False); slice
    4 promotes the embedding-similarity hydrate path to default-on.

    This pin guards both directions:
      * If a future change reverts the default to False, this fails
        loud — operators upgrading would silently lose cross-document
        linking, which is the regression we don't want.
      * The env-var opt-out still works, so a single-tenant deployment
        with a known-clean dataset can still suppress the second-stage
        similarity lookup without code changes."""
    # Clear any inherited env so we read the in-code default.
    monkeypatch.delenv("ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED", raising=False)

    settings = Settings(_env_file=None)
    assert settings.ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED is True, (
        "Cross-document entity resolution default flipped back to "
        "False. Slice 4 promoted this to default-on; reverting needs "
        "an explicit migration plan, not a silent default change."
    )

    # Operator opt-out via env is still honoured.
    monkeypatch.setenv("ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED", "false")
    opted_out = Settings(_env_file=None)
    assert opted_out.ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED is False
