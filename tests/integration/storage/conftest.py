"""Shared fixtures for the C2-postgres integration suite.

Slice 7 bootstrap: an Apache AGE container brought up via
testcontainers so the adapter can be exercised against real
psycopg + cypher() round trips, complementing the mocked unit
tests in tests/unit/services/storage/test_postgres_age.py.

Skip behaviour:
  - Module ``import testcontainers`` failure → skipped (e.g. the
    extra didn't get installed).
  - Docker daemon unreachable → skipped (CI without Docker, dev
    laptop with Docker stopped).
  - The ``apache/age`` image not pullable → skipped with the pull
    error surfaced to the operator.

Tests fail (don't skip) on errors AFTER the container is up — that's
the suite's job to catch.
"""

from __future__ import annotations

import os

import pytest

# Importing testcontainers at module load so the skip kicks in
# before fixtures bind. ``contextmanager`` lets us swallow startup
# errors and convert them into pytest.skip cleanly.
try:  # pragma: no cover — ImportError path is the skip
    from testcontainers.postgres import PostgresContainer
except ImportError as exc:  # pragma: no cover
    PostgresContainer = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


# Pin the AGE image. apache/age:latest ships PostgreSQL + AGE
# preinstalled; pgvector isn't included but the adapter tolerates
# that (slice 6 made the pgvector CREATE EXTENSION warn-and-continue
# rather than fail).
#
# AGE's published tags are oddly named ("release_PG16_1.6.0",
# "dev_snapshot_PG17") and not all PG-versioned tags get pushed —
# ":latest" is the most reliable working choice. Override with
# AGE_TEST_IMAGE for local experiments against a specific PG/AGE
# combination.
_AGE_IMAGE = os.environ.get("AGE_TEST_IMAGE", "apache/age:latest")


@pytest.fixture(scope="session")
def age_container():
    """Session-scoped Apache AGE container.

    One container shared across the whole integration suite — startup
    is ~10s on a warm machine and we don't want each test paying it.
    Tests that need isolation use a unique transform_id rather than
    a fresh database.

    Common skip causes:
      - testcontainers extra not installed
      - Docker daemon unreachable. On macOS Docker Desktop the
        socket is at ~/.docker/run/docker.sock rather than
        /var/run/docker.sock; if testcontainers can't find it,
        export ``DOCKER_HOST=unix://$HOME/.docker/run/docker.sock``
        before running pytest.
      - apache/age image not pullable (network / registry)
    """
    if PostgresContainer is None:
        pytest.skip(f"testcontainers not installed: {_IMPORT_ERROR}")

    # Build the container object first — its constructor doesn't
    # touch Docker. Then start() is the real network/Docker call.
    # Wrap both behind one skip path so any infra failure produces
    # a clean skip instead of an error noise-flood.
    try:
        container = PostgresContainer(
            image=_AGE_IMAGE,
            username="age",
            password="age",
            dbname="age",
        )
        container.start()
    except Exception as exc:  # pragma: no cover — Docker-not-available path
        import logging
        import traceback as _tb

        # pytest truncates the skip reason; log the full traceback
        # at WARNING so operators can debug "why did the integration
        # suite skip" without having to re-run with --tb.
        logging.getLogger(__name__).warning(
            "AGE container startup failed:\n%s", _tb.format_exc()
        )
        pytest.skip(
            f"Could not start AGE container ({_AGE_IMAGE}): "
            f"{type(exc).__name__}: {exc}. "
            "Skipping integration suite. On macOS Docker Desktop, try "
            "DOCKER_HOST=unix://$HOME/.docker/run/docker.sock."
        )

    try:
        yield container
    finally:
        try:
            container.stop()
        except Exception:  # pragma: no cover — best-effort cleanup
            pass


@pytest.fixture
def age_dsn(age_container) -> str:
    """psycopg-compatible DSN for the running container."""
    # testcontainers exposes get_connection_url() shaped for
    # SQLAlchemy (postgresql+psycopg://); strip the dialect for
    # the bare libpq DSN psycopg wants.
    url = age_container.get_connection_url()
    return url.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )


@pytest.fixture
async def age_storage(age_dsn):
    """A fresh PostgresAGEStorage backed by the shared container.

    The graph name is randomized per test so tests don't trample
    each other's data. Connection pool closes at fixture teardown.
    """
    import uuid

    from graphora_server.services.storage.postgres_age import PostgresAGEStorage

    graph_name = f"t_{uuid.uuid4().hex[:12]}"
    storage = PostgresAGEStorage(dsn=age_dsn, graph_name=graph_name)
    try:
        yield storage
    finally:
        await storage.close()
