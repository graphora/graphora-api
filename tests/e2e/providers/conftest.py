"""Shared fixtures for provider E2E harness tests.

Scope:
    The harness runs full extraction (BAML LLM call → ontology-typed
    graph → node/edge count assertions) against a real LLM endpoint.
    It is NOT a unit test and is NOT part of `make test`; it runs via
    the dedicated `.github/workflows/provider-e2e.yml` matrix, gated
    on path filters (BAML sources, llm client, pyproject).

Cost discipline:
    Fixture text is ~100 words, target model is the cheapest tier
    available per provider (Gemini Flash Lite, GPT-4o Mini,
    Claude Haiku 4.5). Each provider's full test class should cost
    well under $0.01 per run.

Adding a new provider test module:
    1. Create `tests/e2e/providers/test_<provider>.py` using
       test_gemini.py as the template.
    2. Pick a cheap model per the comment above.
    3. Add the provider to the matrix in
       `.github/workflows/provider-e2e.yml`.
    4. Add the test secret (e.g. `OPENAI_API_KEY_TEST`) in the
       repo settings with a spend cap.

Today only Gemini has a working BAML client path (see
graphora_server/utils/baml_helper.py). OpenAI, Anthropic, and
Vertex harnesses are deferred until server-side BAML wiring lands
for those providers (Goal 9 in the product strategy).
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tiny_txt_path() -> Path:
    """Path to the small text fixture (~100 words)."""
    return FIXTURES / "tiny.txt"


@pytest.fixture(scope="module")
def tiny_text(tiny_txt_path: Path) -> str:
    """Raw text content of the tiny fixture."""
    return tiny_txt_path.read_text()


@pytest.fixture(scope="module")
def minimal_ontology_path() -> Path:
    """Path to the minimal ontology YAML fixture."""
    return FIXTURES / "minimal.ontology.yaml"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e_provider: provider-specific end-to-end test. Requires a real "
        "API key; excluded from `make test` and skipped locally unless the "
        "provider's test env var is set.",
    )
