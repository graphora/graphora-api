"""Shared fixtures for the Neo4j integration suite.

A real Neo4j container brought up via testcontainers so the
adapter's Cypher patterns + transaction shape can be exercised
against a real driver, complementing the mocked unit tests in
``tests/unit/services/test_graph_service_count_query.py``.

Two-layer skip behaviour (both fixture-level — module-level
``pytest.skip`` crashes pytest's config phase):

  1. ``GRAPHORA_TEST_REAL_NEO4J != "1"`` → skipped. The root
     ``tests/conftest.py`` installs a Neo4j-driver stub for the
     unit suite (real driver = native deps + slow imports we
     don't need on every unit run); the env var bypasses that
     stub. ``make test-integration`` sets it automatically.
  2. Docker daemon unreachable / image not pullable / testcontainers
     failure → skipped with the underlying error surfaced.

Tests fail (don't skip) on errors AFTER the container is up — the
suite's job is to catch real adapter bugs, not infra flakiness.
"""

from __future__ import annotations

import os
import sys

import pytest

_GATING_ENV_VAR = "GRAPHORA_TEST_REAL_NEO4J"

# Pin the Neo4j image to a 5.x release. testcontainers' default
# (``neo4j:latest``) drifts and can pull a major version that
# breaks driver compatibility — pinning here keeps CI deterministic.
# Override with ``NEO4J_TEST_IMAGE`` for local experiments.
_NEO4J_IMAGE = os.environ.get("NEO4J_TEST_IMAGE", "neo4j:5.20-community")

# Import testcontainers lazily inside the fixture (not at module
# load) — the testcontainers.neo4j module imports the real neo4j
# Python driver at module load, which crashes against the unit-suite
# pandas stub before our fixture-level env-var skips can run.
# Reviewer-flagged on the integration-test infra commit (ce22727):
# `GRAPHORA_TEST_REAL_NEO4J=1` alone (without GRAPHORA_TEST_REAL_DEPS)
# would hit this pandas crash instead of getting a clean skip
# message.


def _ensure_real_neo4j_driver() -> None:
    """Drop the unit-suite Neo4j stub (installed by root conftest)
    and force a fresh import of the real driver.

    Must run BEFORE any code path imports Neo4jStorage — the storage
    module captures ``AsyncGraphDatabase`` at module-load time, so a
    late removal won't help if Neo4jStorage already bound to the
    stubbed reference. We aggressively purge any neo4j path from
    sys.modules; the next ``import`` reaches for the real driver."""
    for mod_name in list(sys.modules.keys()):
        if mod_name == "neo4j" or mod_name.startswith("neo4j."):
            del sys.modules[mod_name]


@pytest.fixture(scope="session")
def neo4j_container():
    """Session-scoped Neo4j container.

    One container shared across the whole integration suite — startup
    is ~15s on a warm machine and we don't want each test paying it.
    Tests that need isolation use a unique transform_id rather than
    a fresh database; the ``neo4j_storage`` fixture wipes nodes/
    edges before each test for a clean slate.
    """
    if os.environ.get(_GATING_ENV_VAR) != "1":
        pytest.skip(
            f"Set {_GATING_ENV_VAR}=1 to run the Neo4j integration "
            "suite (the unit-suite stub bypasses the real driver "
            "otherwise). `make test-integration` sets this "
            "automatically."
        )
    if os.environ.get("GRAPHORA_TEST_REAL_DEPS") != "1":
        # The neo4j Python driver depends on pandas at import time
        # (uses pd.NA in its packstream codec). The root conftest
        # ships a minimal pandas stub for fast unit tests; that stub
        # doesn't implement pd.NA, so importing the real neo4j driver
        # crashes with AttributeError. Real Neo4j integration needs
        # the real pandas — gated by GRAPHORA_TEST_REAL_DEPS=1.
        pytest.skip(
            "Set GRAPHORA_TEST_REAL_DEPS=1 in addition to "
            f"{_GATING_ENV_VAR}=1: the real neo4j driver imports "
            "pandas (pd.NA in packstream codec) which the unit-"
            "suite pandas stub doesn't implement. "
            "`make test-integration` sets both."
        )
    # Purge the stub BEFORE importing testcontainers.neo4j — that
    # package imports the real neo4j driver at module load and would
    # otherwise crash on the pandas-stub's missing pd.NA. The env
    # gates above guaranteed we have the real pandas + intent to use
    # the real driver.
    _ensure_real_neo4j_driver()

    try:
        from testcontainers.neo4j import Neo4jContainer
    except ImportError as exc:
        pytest.skip(f"testcontainers[neo4j] not installed: {exc}")

    try:
        container = Neo4jContainer(image=_NEO4J_IMAGE)
        container.start()
    except Exception as exc:  # pragma: no cover — Docker-not-available path
        import logging
        import traceback as _tb

        logging.getLogger(__name__).warning(
            "Neo4j container startup failed:\n%s", _tb.format_exc()
        )
        pytest.skip(
            f"Could not start Neo4j container ({_NEO4J_IMAGE}): "
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
def neo4j_uri(neo4j_container) -> str:
    """The bolt URI of the running container."""
    return neo4j_container.get_connection_url()


@pytest.fixture
def neo4j_credentials(neo4j_container) -> tuple[str, str]:
    """(username, password) for the running container.

    testcontainers exposes the values used at start time on the
    container instance. We read them rather than hard-coding so
    fixture-level overrides keep working without test changes."""
    user = neo4j_container.username if hasattr(neo4j_container, "username") else "neo4j"
    password = (
        neo4j_container.password if hasattr(neo4j_container, "password") else "password"
    )
    return user, password


@pytest.fixture
async def neo4j_storage(neo4j_uri, neo4j_credentials):
    """A fresh Neo4jStorage backed by the shared container.

    Each test gets a clean graph state — the fixture wipes nodes
    and relationships before yielding so prior tests don't pollute.
    Cheaper than a fresh container per test (15s startup) and
    enforces test isolation without overhead. Driver closes at
    fixture teardown.
    """
    # Belt-and-braces: even though the session-scoped container
    # fixture purged the stub, fixture re-evaluation in some pytest
    # configurations could re-trigger an import of the stubbed
    # neo4j. Re-purge here as a no-op safety net.
    _ensure_real_neo4j_driver()

    from graphora_server.services.storage.neo4j import Neo4jStorage

    user, password = neo4j_credentials
    storage = Neo4jStorage(uri=neo4j_uri, username=user, password=password)
    # Wipe before each test. After-each would also work but
    # before-each makes manual inspection of a failed run easier —
    # the state when you connect post-fail reflects the test, not
    # next-test cleanup.
    async with storage._get_session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    try:
        yield storage
    finally:
        try:
            await storage.driver.close()
        except Exception:  # pragma: no cover — best-effort cleanup
            pass
