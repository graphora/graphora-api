"""Unit tests for the Cypher anti-pattern detector
(``scripts/check_cypher_patterns.py``).

The detector exists because the same `WITH count(n) as ...` /
subsequent `MATCH (n)` shape silently scoped queries across the
whole DB and bit production code three times in two review rounds
(commits e96c15c, 16c8ecb, b1edc7c). These tests pin the
detection logic so it stays sensitive to the historical bug
without flagging the safe patterns the codebase actually uses.

If a true positive ever stops firing or a true negative starts
firing, a future Cypher refactor can quietly slip back into the
buggy shape — that's exactly what the script is designed to
prevent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# scripts/ isn't a package; load the module by file path so we can
# exercise its functions directly from a test.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "check_cypher_patterns.py"
)
_spec = importlib.util.spec_from_file_location("check_cypher_patterns", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
check_cypher_patterns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_cypher_patterns)


class TestVarsCarriedForward:
    """Pin the helper that decides which identifiers survive a WITH.
    A wrong answer here makes every other check unreliable."""

    def test_bare_identifier_carries_forward(self) -> None:
        assert "n" in check_cypher_patterns._vars_carried_forward("n")

    def test_alias_overrides_expression(self) -> None:
        forward = check_cypher_patterns._vars_carried_forward("n.foo as bar")
        assert "bar" in forward
        assert "n" not in forward

    def test_aggregation_drops_inner_variable(self) -> None:
        forward = check_cypher_patterns._vars_carried_forward("count(n) as c")
        assert "c" in forward
        assert "n" not in forward

    def test_mixed_projections_keeps_named_variable(self) -> None:
        forward = check_cypher_patterns._vars_carried_forward("n, count(other) as c")
        assert {"n", "c"} <= forward

    def test_order_by_modifier_does_not_add_to_scope(self) -> None:
        forward = check_cypher_patterns._vars_carried_forward("n ORDER BY n.id")
        assert forward == {"n"}

    def test_distinct_in_aggregation_still_drops_var(self) -> None:
        forward = check_cypher_patterns._vars_carried_forward(
            "count(DISTINCT r) as edge_count"
        )
        assert "edge_count" in forward
        assert "r" not in forward


class TestFindLeaksInQueryTruePositives:
    """The shapes that historically broke production. Each test is
    a regression pin against a real bug we shipped — keep these
    failing-when-broken even after refactors."""

    def test_historical_count_n_then_optional_match_n(self) -> None:
        """The exact shape from the original transform-id leak
        (commit e96c15c). If this stops firing, the whole reason
        the script exists is gone."""
        query = """
        MATCH (n)
        WHERE n.__tid = $transform_id
        WITH count(n) as node_count
        OPTIONAL MATCH (n)-[r]-()
        RETURN node_count, count(DISTINCT r) as edge_count
        """
        leaks = check_cypher_patterns.find_leaks_in_query(query)
        assert len(leaks) == 1
        _, var, _ = leaks[0]
        assert var == "n"

    def test_count_then_subsequent_required_match(self) -> None:
        """Same pattern but with a non-OPTIONAL MATCH — also leaks."""
        query = """
        MATCH (n) WHERE n.tag = $tag
        WITH count(n) as c
        MATCH (n)-[r]->(m)
        RETURN c, r
        """
        leaks = check_cypher_patterns.find_leaks_in_query(query)
        assert len(leaks) == 1

    def test_collect_then_match_also_flags(self) -> None:
        """Aggregations other than count have the same scoping
        problem; pin that the detector covers them, not just count."""
        query = """
        MATCH (n)
        WITH collect(n) as ns
        MATCH (n)-[r]-()
        RETURN ns, r
        """
        leaks = check_cypher_patterns.find_leaks_in_query(query)
        assert len(leaks) == 1


class TestFindLeaksInQueryTrueNegatives:
    """The shapes the codebase actually uses. False positives here
    would block legitimate queries — be conservative about adding
    new ones, and make sure they reflect real query patterns."""

    def test_with_n_carried_forward_is_safe(self) -> None:
        """`WITH n, count(other) as c` keeps n in scope — the next
        MATCH is a legitimate continuation, not a rebind."""
        query = """
        MATCH (n)
        WITH n, count(other) as c
        MATCH (n)-[r]-()
        RETURN c, r
        """
        assert check_cypher_patterns.find_leaks_in_query(query) == []

    def test_anonymous_endpoint_match_is_safe(self) -> None:
        """The fix pattern: `OPTIONAL MATCH ()-[r]->()` doesn't
        reference `n` at all, so the dropped scope is harmless."""
        query = """
        MATCH (n) WHERE n.__tid = $transform_id
        WITH count(n) as node_count
        OPTIONAL MATCH ()-[r]->()
        WHERE r.__tid = $transform_id
        RETURN node_count, count(r) as edge_count
        """
        assert check_cypher_patterns.find_leaks_in_query(query) == []

    def test_with_n_order_by_and_subsequent_match_is_safe(self) -> None:
        """The paginated data-query shape (graph_service.py:111).
        WITH carries n forward; ORDER BY/SKIP/LIMIT are modifiers."""
        query = """
        MATCH (n) WHERE n.__tid = $transform_id
        WITH n ORDER BY n.id SKIP $skip LIMIT $limit
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE r.__tid = $transform_id
        RETURN collect(n) as nodes
        """
        assert check_cypher_patterns.find_leaks_in_query(query) == []

    def test_aggregation_without_subsequent_rebind_is_safe(self) -> None:
        """If `n` is dropped but never re-referenced, no rebind can
        happen — don't flag."""
        query = """
        MATCH (n) WHERE n.tag = $tag
        WITH count(n) as c
        RETURN c
        """
        assert check_cypher_patterns.find_leaks_in_query(query) == []

    def test_different_variable_name_after_rebind_is_safe(self) -> None:
        """Aggregation drops `n`, but the next MATCH uses `m` —
        no scope confusion, no flag."""
        query = """
        MATCH (n)
        WITH count(n) as c
        MATCH (m)-[r]-()
        RETURN c, m
        """
        assert check_cypher_patterns.find_leaks_in_query(query) == []


class TestScanFile:
    """Exercise the file-level scan via real fixture content; this is
    the surface CI calls. The tests use tmp_path so they don't depend
    on the project's current state (which the maintained queries
    already pass)."""

    def test_clean_file_yields_no_findings(self, tmp_path: Path) -> None:
        sample = tmp_path / "clean.py"
        sample.write_text(
            "def q():\n"
            '    return """\n'
            "    MATCH (n) WHERE n.x = 1\n"
            "    WITH n, count(other) as c\n"
            "    MATCH (n)-[r]-()\n"
            "    RETURN c\n"
            '    """\n'
        )
        assert check_cypher_patterns.scan_file(sample) == []

    def test_file_with_buggy_query_is_reported(self, tmp_path: Path) -> None:
        sample = tmp_path / "buggy.py"
        sample.write_text(
            "def q():\n"
            '    return """\n'
            "    MATCH (n)\n"
            "    WITH count(n) as c\n"
            "    OPTIONAL MATCH (n)-[r]-()\n"
            "    RETURN c, r\n"
            '    """\n'
        )
        findings = check_cypher_patterns.scan_file(sample)
        assert len(findings) == 1
        path, _, var, _ = findings[0]
        assert path == sample
        assert var == "n"

    def test_fstring_with_interpolation_still_scanned(self, tmp_path: Path) -> None:
        """f-string property accesses like `n.{TID}` shouldn't break
        scanning — the placeholder substitution leaves the bare
        identifier intact."""
        sample = tmp_path / "fstring.py"
        sample.write_text(
            'TID = "__tid"\n'
            "def q():\n"
            '    return f"""\n'
            "    MATCH (n) WHERE n.{TID} = $p\n"
            "    WITH count(n) as c\n"
            "    OPTIONAL MATCH (n)-[r]-()\n"
            "    RETURN c\n"
            '    """\n'
        )
        findings = check_cypher_patterns.scan_file(sample)
        assert len(findings) == 1


class TestEntryPoint:
    """The exit-code contract is what CI keys on; pin it."""

    def test_zero_exit_when_target_clean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clean = tmp_path / "graphora_server"
        clean.mkdir()
        (clean / "ok.py").write_text(
            "def q():\n" '    return """MATCH (n) RETURN n"""\n'
        )
        monkeypatch.setattr(check_cypher_patterns, "ROOT", tmp_path)
        assert check_cypher_patterns.main(["graphora_server"]) == 0

    def test_nonzero_exit_when_target_has_leak(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        leaky = tmp_path / "graphora_server"
        leaky.mkdir()
        (leaky / "buggy.py").write_text(
            "def q():\n"
            '    return """\n'
            "    MATCH (n)\n"
            "    WITH count(n) as c\n"
            "    OPTIONAL MATCH (n)-[r]-()\n"
            "    RETURN c\n"
            '    """\n'
        )
        monkeypatch.setattr(check_cypher_patterns, "ROOT", tmp_path)
        assert check_cypher_patterns.main(["graphora_server"]) == 1
