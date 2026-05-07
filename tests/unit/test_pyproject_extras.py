"""Pin the convenience-extras shape in ``pyproject.toml``.

The ``[all]`` extra is documented as 'a runnable server with all
features' — operators run ``pip install graphora-server[all]`` and
expect every optional code path to be active. Pre-fix, ``[all]``
was missing ``pdf-llm`` (the layout-aware PDF backend that gates
Evidence-tab source_text), so an [all] install silently disabled
the new Evidence-tab capability. The reviewer caught this on
commit 920b8f9.

This test pins the inclusion contract: every optional-extra except
``dev``, ``test``, and ``all`` itself MUST appear inside ``[all]``.
A future contributor who adds a new ``[X]`` extra and forgets to
list it under ``[all]`` will fail this test loudly.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# extras intentionally excluded from [all] — these aren't
# 'features the runnable server uses', they're tooling.
_EXCLUDED_FROM_ALL = {"dev", "test", "all"}


def _load_pyproject() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    pyproject_path = project_root / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        return tomllib.load(fh)


def _parse_all_extras(all_value: list) -> set[str]:
    """``all`` is declared as a single self-reference like
    ``"graphora-server[a,b,c]"``. Extract the comma-separated names
    inside the brackets."""
    assert len(all_value) == 1, (
        f"expected the [all] extra to be a single self-reference; " f"got {all_value}"
    )
    spec = all_value[0]
    open_bracket = spec.index("[")
    close_bracket = spec.index("]", open_bracket)
    inside = spec[open_bracket + 1 : close_bracket]
    return {name.strip() for name in inside.split(",") if name.strip()}


def test_all_extra_includes_every_feature_extra() -> None:
    pyproject = _load_pyproject()
    extras = pyproject["project"]["optional-dependencies"]
    declared_extras = set(extras.keys())
    feature_extras = declared_extras - _EXCLUDED_FROM_ALL

    all_listed = _parse_all_extras(extras["all"])

    missing = feature_extras - all_listed
    assert not missing, (
        f"The [all] extra is documented as 'all features' but is "
        f"missing: {sorted(missing)}. Add them to the [all] entry "
        f"in pyproject.toml or, if intentionally excluded, expand "
        f"_EXCLUDED_FROM_ALL in this test with a comment explaining "
        f"why."
    )


def test_pdf_llm_specifically_listed_in_all() -> None:
    """Regression pin for the reviewer's P3 finding on commit
    920b8f9: ``[pdf-llm]`` was added as a feature extra but not
    rolled into ``[all]``, so an `[all]` install couldn't surface
    Evidence-tab source_text. This test is intentionally redundant
    with the generic check above — it's specifically for grep-ability
    if pdf-llm ever gets dropped from [all] and the symptom is
    'Evidence tab silently empty on [all] installs'."""
    pyproject = _load_pyproject()
    all_listed = _parse_all_extras(pyproject["project"]["optional-dependencies"]["all"])
    assert "pdf-llm" in all_listed
