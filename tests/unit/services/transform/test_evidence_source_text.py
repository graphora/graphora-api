"""Pin the binary-PDF Evidence-tab source_text gate
(``flows._resolve_evidence_source_text``).

Reviewer-flagged on commit f1596f3: parse_pdf_layout_only now
returns full-document Markdown for the binary-PDF path, but
helpers._attach_provenance_properties truncates source_text to
the first 1000 chars before stamping it on every node from a
split. For a 100-page split that's pages 1-2 only — entities
extracted from page 60 would show wrong evidence text.

The gate's contract: only populate source_text when the parsed
Markdown is small enough that the truncation isn't misleading.
The threshold is set to 2x the truncation limit (so the first
1000 chars covers >= 50% of actual content).

Tests cover the full behaviour matrix:
  * parse returns None        → returns None
  * parse returns short text  → returns the text
  * parse returns long text   → returns None (gate fires)
  * parse raises              → returns None (defensive catch)
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock

import pytest

from graphora_server.services.transform.flows import (
    _resolve_evidence_source_text,
)


class _ParserStub:
    """Minimal stand-in for DocumentParser exposing only the method
    the helper calls. Avoids needing to construct a real
    DocumentParser (which lazy-imports pymupdf4llm)."""

    def __init__(
        self, return_value: Optional[str] = None, exc: Exception | None = None
    ):
        self._return_value = return_value
        self._exc = exc
        self.calls: list[str] = []

    async def parse_pdf_layout_only(self, file_path: str) -> Optional[str]:
        self.calls.append(file_path)
        if self._exc is not None:
            raise self._exc
        return self._return_value


@pytest.mark.asyncio
async def test_returns_none_when_parser_returns_none() -> None:
    """pymupdf4llm not installed (or the file isn't a PDF the
    backend can read) -> parse_pdf_layout_only returns None ->
    source_text stays None. This is the inherited 920b8f9
    contract; pin so the new gate doesn't accidentally synthesize
    text from a None parse."""
    parser = _ParserStub(return_value=None)
    result = await _resolve_evidence_source_text(parser, "/tmp/x.pdf", max_chars=1000)
    assert result is None
    assert parser.calls == ["/tmp/x.pdf"]


@pytest.mark.asyncio
async def test_returns_text_when_within_max_chars() -> None:
    """Small parses (≤ max_chars) survive the gate — the first-
    1000-chars truncation downstream covers >= half of the actual
    content, which is honest as a sample. Pin the happy path so a
    refactor can't accidentally make the gate too strict and
    silently disable Evidence-tab text on small documents."""
    short_text = "# Title\n\nSome short markdown content."
    parser = _ParserStub(return_value=short_text)
    result = await _resolve_evidence_source_text(parser, "/tmp/x.pdf", max_chars=1000)
    assert result == short_text


@pytest.mark.asyncio
async def test_returns_none_when_text_exceeds_max_chars() -> None:
    """The reviewer's actual finding: a 100-page split's Markdown
    parse is much larger than the truncation budget. Capturing
    the first 1000 chars means later-page entities show wrong
    evidence. Pin the gate fires — source_text=None instead.

    Operators look up the original split file via source_chunk_id
    when they need to read the actual content."""
    long_text = "# Big doc\n\n" + ("Lorem ipsum " * 1000)  # well above 1000 chars
    parser = _ParserStub(return_value=long_text)
    result = await _resolve_evidence_source_text(parser, "/tmp/x.pdf", max_chars=1000)
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_on_parser_exception() -> None:
    """Defensive catch: a pymupdf4llm runtime error on a specific
    file (corrupt, encrypted, weird format) must not propagate up
    and break the transform. Pin: any exception -> None,
    transform continues with source_text=None."""
    parser = _ParserStub(exc=RuntimeError("pymupdf4llm choked on this PDF"))
    result = await _resolve_evidence_source_text(parser, "/tmp/x.pdf", max_chars=1000)
    assert result is None


@pytest.mark.asyncio
async def test_threshold_boundary_inclusive() -> None:
    """A parse exactly equal to max_chars passes the gate. The
    threshold is inclusive at the upper bound — pin this so a
    refactor doesn't accidentally swap <= to <."""
    text = "x" * 500
    parser = _ParserStub(return_value=text)
    result = await _resolve_evidence_source_text(parser, "/tmp/x.pdf", max_chars=500)
    assert result == text


@pytest.mark.asyncio
async def test_uses_async_parser_correctly() -> None:
    """Sanity: parse_pdf_layout_only is awaited (not just called).
    A regression that returns the coroutine instead of awaiting
    would surface as a coroutine-typed result here."""
    parser = AsyncMock()
    parser.parse_pdf_layout_only = AsyncMock(return_value="short text")
    result = await _resolve_evidence_source_text(parser, "/tmp/x.pdf", max_chars=1000)
    assert result == "short text"
    parser.parse_pdf_layout_only.assert_awaited_once_with("/tmp/x.pdf")
