#!/usr/bin/env python3
"""Static check for a Cypher anti-pattern that has bitten this
codebase three times in two review rounds (commits e96c15c,
16c8ecb, b1edc7c).

The pattern:

    MATCH (n) WHERE n.<scope> = $param
    WITH count(n) as node_count
    OPTIONAL MATCH (n)-[r]-(...)        <-- BUG
    RETURN node_count, count(r) ...

After ``WITH count(n) as node_count`` the variable ``n`` is gone
from scope (``WITH`` only forwards the projections it names).
Cypher then silently *rebinds* ``n`` in the next ``OPTIONAL MATCH``
to a fresh anonymous node — so the relationship match runs across
the whole database, not the rows we thought we filtered. The bug is
quiet: the query parses, returns numbers, and only the count is
wrong. We've rediscovered it three times by users seeing
cross-transform totals on the FE.

This check scans every Python source under ``graphora_server/`` for
string literals that look like Cypher and reports any place where:

  * a ``WITH`` clause aggregates a variable away (count/collect/etc.)
    without re-projecting it explicitly, AND
  * a subsequent ``(OPTIONAL )MATCH (<same-name>)`` rebinds the name

If the script exits non-zero, the build fails; review the listed
files and either preserve the variable (``WITH n, count(...)``) or
use an explicit anonymous match (``MATCH ()-[r]->()``) with a
relationship-side filter.

Usage:
    uv run python scripts/check_cypher_patterns.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS: Sequence[str] = ("graphora_server",)

# Aggregation functions that consume their argument and produce a
# scalar/collection. After a `WITH <agg>(<var>) as <alias>`, the
# inner <var> is no longer in scope unless explicitly re-projected.
_AGGREGATION_FUNCS = ("count", "collect", "sum", "avg", "min", "max")
_AGG_PIECE = "|".join(_AGGREGATION_FUNCS)

# Cypher clause keywords that terminate a WITH "block". ORDER BY,
# SKIP, LIMIT, and DISTINCT are intentionally NOT in this list —
# they're modifiers of the WITH itself and don't introduce a new
# clause.
_NEXT_CLAUSE_KEYWORDS = (
    "OPTIONAL",
    "MATCH",
    "WHERE",
    "RETURN",
    "UNWIND",
    "CALL",
    "SET",
    "REMOVE",
    "DELETE",
    "CREATE",
    "MERGE",
    "FOREACH",
    "WITH",
    "UNION",
)
_NEXT_CLAUSE_PIECE = "|".join(_NEXT_CLAUSE_KEYWORDS)

_WITH_BODY = re.compile(
    rf"\bWITH\s+(?P<body>.+?)(?=\b(?:{_NEXT_CLAUSE_PIECE})\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_AGG_OF_VAR = re.compile(
    rf"\b(?:{_AGG_PIECE})\s*\(\s*(?:DISTINCT\s+)?(?P<var>\w+)\s*\)",
    re.IGNORECASE,
)


def _vars_carried_forward(with_body: str) -> set[str]:
    """Return the set of identifiers in scope AFTER this WITH clause.

    A WITH clause's projections survive into subsequent clauses; the
    rest is dropped. For each comma-separated projection we take the
    alias (when ``... as <alias>`` is present) or the leading bare
    identifier. Aggregations are stripped first so their inner
    variable doesn't pollute the bare-name scan."""
    cleaned = _AGG_OF_VAR.sub("__AGG__", with_body)
    forward: set[str] = set()
    for projection in cleaned.split(","):
        projection = projection.strip()
        if not projection:
            continue
        # Trim trailing ORDER BY / SKIP / LIMIT modifiers — they
        # belong to the last projection but don't add to scope.
        projection = re.split(
            r"\b(?:ORDER\s+BY|SKIP|LIMIT)\b",
            projection,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        as_match = re.search(r"\bas\s+(\w+)", projection, re.IGNORECASE)
        if as_match:
            forward.add(as_match.group(1))
            continue
        bare_match = re.match(r"\s*(\w+)", projection)
        if bare_match and bare_match.group(1) != "__AGG__":
            forward.add(bare_match.group(1))
    return forward


def find_leaks_in_query(query: str) -> List[Tuple[int, str, str]]:
    """Scan a Cypher-ish string for the leaky-aggregation pattern.

    Returns a list of (line_offset_within_string, var_name, snippet)
    triples — one per detected issue."""
    issues: List[Tuple[int, str, str]] = []
    for with_match in _WITH_BODY.finditer(query):
        body = with_match.group("body")
        forward = _vars_carried_forward(body)
        # All aggregated variables in this WITH.
        agg_vars = {m.group("var") for m in _AGG_OF_VAR.finditer(body)}
        dropped_vars = agg_vars - forward
        if not dropped_vars:
            continue
        rest_of_query = query[with_match.end() :]
        for var in dropped_vars:
            rebind = re.search(
                rf"\b(?:OPTIONAL\s+)?MATCH\s*\(\s*{re.escape(var)}\b",
                rest_of_query,
                re.IGNORECASE,
            )
            if not rebind:
                continue
            line_offset = query[: with_match.start()].count("\n")
            snippet = with_match.group(0)[:120].replace("\n", " ").strip()
            issues.append((line_offset, var, snippet))
    return issues


def _string_literal_value(node: ast.AST) -> str | None:
    """Reconstruct an approximate literal text for a constant or
    f-string AST node. F-string interpolations become a placeholder
    that is unlikely to collide with Cypher identifiers, so the
    surrounding pattern check still works (e.g. ``n.{TID}`` becomes
    ``n.__INTERP__``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("__INTERP__")
        return "".join(parts)
    return None


def _looks_like_cypher(text: str) -> bool:
    upper = text.upper()
    return "MATCH" in upper or "WITH" in upper


def _iter_string_nodes(tree: ast.AST):
    """Yield every string-bearing AST node, skipping inner Constants
    that are children of a JoinedStr (those get reconstructed into
    the f-string's combined text, so walking into them would
    double-count). ``ast.walk`` has no such filter, so we do our
    own traversal."""
    stack: List[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.JoinedStr):
            yield node
            # Don't descend — its Constant children are already
            # captured in the reconstructed string.
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node
        for child in ast.iter_child_nodes(node):
            stack.append(child)


def scan_file(path: Path) -> List[Tuple[Path, int, str, str]]:
    findings: List[Tuple[Path, int, str, str]] = []
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return findings
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return findings

    for node in _iter_string_nodes(tree):
        text = _string_literal_value(node)
        if text is None or not _looks_like_cypher(text):
            continue
        for offset, var, snippet in find_leaks_in_query(text):
            findings.append((path, getattr(node, "lineno", 0) + offset, var, snippet))
    return findings


def main(argv: Sequence[str]) -> int:
    targets = list(argv) if argv else list(DEFAULT_TARGETS)
    print("Running Cypher anti-pattern check...", flush=True)

    all_findings: List[Tuple[Path, int, str, str]] = []
    for target in targets:
        target_path = ROOT / target
        if not target_path.exists():
            print(f"Error: target not found: {target_path}", file=sys.stderr)
            return 2
        for py_path in target_path.rglob("*.py"):
            all_findings.extend(scan_file(py_path))

    if not all_findings:
        print("Cypher anti-pattern check: no issues found.")
        return 0

    print(
        "\nCypher anti-pattern detected.\n"
        "`WITH <agg>(<var>) as ...` followed by `MATCH (<var>)` "
        "silently rebinds <var> across the whole database — see\n"
        "scripts/check_cypher_patterns.py docstring for the fix "
        "patterns.\n",
        file=sys.stderr,
    )
    for path, line, var, snippet in all_findings:
        rel = path.relative_to(ROOT) if path.is_absolute() else path
        print(f"  {rel}:{line} — variable `{var}` rebound after aggregation")
        print(f"    {snippet}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
