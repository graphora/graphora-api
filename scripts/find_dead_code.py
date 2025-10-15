#!/usr/bin/env python3
"""Run Vulture on the project to surface dead code.

Usage:
    uv run python scripts/find_dead_code.py

Installs/runs Vulture in the current environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["app", "tests", "scripts"]

VULTURE_CMD = [
    sys.executable,
    "-m",
    "vulture",
    *TARGETS,
    "--min-confidence",
    "80",
    "--exclude",
    "app/baml_client",
]


def main() -> int:
    print("Running Vulture dead-code scan...", flush=True)
    result = subprocess.run(VULTURE_CMD, cwd=ROOT)
    if result.returncode == 0:
        print("No dead code detected at the configured threshold.")
    else:
        print(
            "Vulture reported potential dead code. Review the output above and "
            "decide whether to remove or ignore the findings.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
