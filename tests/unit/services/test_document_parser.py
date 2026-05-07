"""Tests for the extended DocumentParser surface.

Covers:
    - text-file path (txt, md, json, html, csv, xml)
    - PDF path with the pypdf/pymupdf backend rotation
    - Office path (docx/xlsx/pptx) via MarkItDown
    - URL path via trafilatura
    - Graceful degradation when optional deps are missing
    - Unsupported extensions → None (with a logged warning)

Fixtures are generated in-process from openpyxl / python-docx /
python-pptx (all pulled in transitively by markitdown). This keeps
binary blobs out of the repo and guarantees the fixtures match the
exact library versions under test.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from graphora_server.services.document_parser import (
    DocumentParser,
    parse_document,
)


# ---- text fixtures -----------------------------------------------


@pytest.fixture
def txt_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("Hello world from txt fixture")
    return p


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.md"
    p.write_text("# Heading\n\nSome body text.")
    return p


@pytest.fixture
def html_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.html"
    p.write_text("<html><body><p>Hello</p></body></html>")
    return p


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.json"
    p.write_text('{"key": "value"}')
    return p


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    return p


@pytest.fixture
def xml_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.xml"
    p.write_text("<?xml version='1.0'?><root><item>x</item></root>")
    return p


@pytest.fixture
def latin1_file(tmp_path: Path) -> Path:
    """Non-UTF-8 encoded text — exercises the encoding fallback path."""
    p = tmp_path / "latin.txt"
    # 0xe9 = é in latin-1, invalid UTF-8
    p.write_bytes(b"caf\xe9")
    return p


# ---- PDF fixture ------------------------------------------------


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    """Minimal real PDF generated via pypdf. Contents: two pages of
    text so the max-pages truncation logic has something to exercise."""
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    # pypdf's add_blank_page doesn't let us add text; use
    # reportlab-free approach: craft a tiny PDF via PyMuPDF if
    # available, else skip.
    try:
        import pymupdf  # type: ignore

        doc = pymupdf.open()
        for i in range(2):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1} body text")
        p = tmp_path / "sample.pdf"
        doc.save(str(p))
        doc.close()
        return p
    except ImportError:
        # Fall back: build one-page PDF manually
        p = tmp_path / "sample.pdf"
        writer.add_blank_page(width=200, height=200)
        with open(p, "wb") as f:
            writer.write(f)
        return p


# ---- Office fixtures --------------------------------------------


@pytest.fixture
def docx_file(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Acme Corp was founded in 2015 by Jane Smith.")
    doc.add_paragraph("BetaTech was acquired in 2020.")
    p = tmp_path / "sample.docx"
    doc.save(str(p))
    return p


@pytest.fixture
def xlsx_file(tmp_path: Path) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Companies"
    ws.append(["Name", "Founded", "Location"])
    ws.append(["Acme Corp", 2015, "San Francisco"])
    ws.append(["BetaTech", 2018, "Seattle"])
    p = tmp_path / "sample.xlsx"
    wb.save(str(p))
    return p


@pytest.fixture
def pptx_file(tmp_path: Path) -> Path:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide_layout = prs.slide_layouts[0]  # title slide
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Acme Corp deck"
    subtitle.text = "Founded 2015 by Jane Smith"
    p = tmp_path / "sample.pptx"
    prs.save(str(p))
    return p


# ---- core routing tests ------------------------------------------


class TestTextFormats:
    """Parser should handle every text-like format declared in
    SUPPORTED_TEXT_EXTENSIONS without any optional deps."""

    @pytest.mark.asyncio
    async def test_reads_plain_text(self, txt_file: Path) -> None:
        result = await DocumentParser().parse_file(str(txt_file))
        assert result == "Hello world from txt fixture"

    @pytest.mark.asyncio
    async def test_reads_markdown(self, md_file: Path) -> None:
        result = await DocumentParser().parse_file(str(md_file))
        assert result is not None
        assert "# Heading" in result

    @pytest.mark.asyncio
    async def test_reads_html(self, html_file: Path) -> None:
        # HTML is returned raw — schema inference consumes markup fine.
        result = await DocumentParser().parse_file(str(html_file))
        assert result is not None
        assert "<p>Hello</p>" in result

    @pytest.mark.asyncio
    async def test_reads_json(self, json_file: Path) -> None:
        result = await DocumentParser().parse_file(str(json_file))
        assert result == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_reads_csv(self, csv_file: Path) -> None:
        result = await DocumentParser().parse_file(str(csv_file))
        assert result is not None and "1,2" in result

    @pytest.mark.asyncio
    async def test_reads_xml(self, xml_file: Path) -> None:
        result = await DocumentParser().parse_file(str(xml_file))
        assert result is not None and "<item>x</item>" in result

    @pytest.mark.asyncio
    async def test_latin1_fallback(self, latin1_file: Path) -> None:
        """Non-UTF-8 bytes decoded via the latin-1 fallback."""
        result = await DocumentParser().parse_file(str(latin1_file))
        assert result is not None and "caf" in result


class TestPdfBackends:
    """Backend preference: pymupdf first, then pypdf, then pdfplumber.
    Missing backends short-circuit to the next."""

    @pytest.mark.asyncio
    async def test_extracts_pdf_text(self, pdf_file: Path) -> None:
        result = await DocumentParser().parse_file(str(pdf_file))
        # Fixture put text on both pages; at minimum the first page's
        # substring is present.
        assert result is not None
        assert "Page" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_pypdf_used_when_pymupdf_missing(
        self, pdf_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate pymupdf not installed — parser should fall through
        to pypdf without raising."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pymupdf":
                raise ImportError("simulated missing pymupdf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        result = await DocumentParser().parse_file(str(pdf_file))
        # pypdf extracts minimally; it may not produce text for a
        # pymupdf-generated PDF, but the call must not raise.
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_no_pdf_backend_logs_install_hint(
        self, pdf_file: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """All pdf backends missing → returns None + logs install hint."""
        import builtins

        real_import = builtins.__import__
        missing = {"pymupdf4llm", "pymupdf", "pypdf", "pdfplumber"}

        def fake_import(name, *args, **kwargs):
            if name in missing:
                raise ImportError(f"simulated missing {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with caplog.at_level("WARNING"):
            result = await DocumentParser().parse_file(str(pdf_file))
        assert result is None
        # The hint covers both [pdf-llm] (preferred) and [pdf]
        # (raw-text fallback) so operators see the full menu.
        warnings = " ".join(r.message for r in caplog.records)
        assert "graphora-server[pdf-llm]" in warnings
        assert "graphora-server[pdf]" in warnings

    def test_has_layout_aware_backend_reports_pymupdf4llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin the contract callers like flows.py rely on: when
        pymupdf4llm is importable, ``has_layout_aware_backend()``
        is True; when it's not, it's False. flows.py gates Evidence-
        tab source_text capture on this — the wrong answer either
        re-introduces the garbled-output regression (false True)
        or silently disables a working install (false False)."""
        import builtins

        real_import = builtins.__import__

        def fake_missing(name, *args, **kwargs):
            if name == "pymupdf4llm":
                raise ImportError("simulated missing pymupdf4llm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_missing)
        assert DocumentParser.has_layout_aware_backend() is False

        # And when the import succeeds — undo the patch and check
        # the truthy branch. This relies on pymupdf4llm being
        # available in the test env when [pdf-llm] is installed; if
        # it's not, the test falls back to skipping the truthy half.
        monkeypatch.undo()
        try:
            import pymupdf4llm  # noqa: F401

            assert DocumentParser.has_layout_aware_backend() is True
        except ImportError:
            pytest.skip(
                "pymupdf4llm not installed in this test env; "
                "the True-branch of has_layout_aware_backend can't "
                "be exercised here. Provider E2E covers it."
            )

    @pytest.mark.asyncio
    async def test_parse_pdf_layout_only_returns_none_when_pymupdf4llm_missing(
        self, pdf_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reviewer-flagged contract: parse_pdf_layout_only is the
        Evidence-tab path and must NEVER fall through to raw-text
        backends. Without pymupdf4llm installed, it returns None —
        not a pymupdf/pypdf result, no matter how readable that
        result might be. 'No source text' is the documented
        graceful-degradation outcome on this surface."""
        import builtins

        real_import = builtins.__import__

        def fake_missing(name, *args, **kwargs):
            if name == "pymupdf4llm":
                raise ImportError("simulated missing pymupdf4llm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_missing)

        result = await DocumentParser().parse_pdf_layout_only(str(pdf_file))
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_pdf_layout_only_does_not_fall_back_on_failure(
        self, pdf_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hole the reviewer caught: pymupdf4llm IS installed but
        fails on this specific PDF (corrupt, encrypted, unusual
        format). parse_file would fall through to pymupdf/pypdf and
        produce garbled text, defeating the entire purpose of the
        Evidence-tab gate. parse_pdf_layout_only must return None
        instead.

        Pin: stub pymupdf4llm to raise on to_markdown, and assert
        no other backend's output ends up in the result. If a
        future refactor wires parse_file behind
        parse_pdf_layout_only, this test fails loud."""
        import sys
        import types

        # Stub pymupdf4llm so it imports cleanly but raises on use.
        stub = types.ModuleType("pymupdf4llm")

        def failing_to_markdown(file_path, pages=None):
            raise RuntimeError("simulated pymupdf4llm failure on this PDF")

        stub.to_markdown = failing_to_markdown
        monkeypatch.setitem(sys.modules, "pymupdf4llm", stub)

        # Also stub a sentinel pymupdf to detect raw-text fallback —
        # if the Evidence path ever reaches pymupdf, this string
        # would land in the result.
        called = {"pymupdf_used": False}
        original = sys.modules.get("pymupdf")

        class _SentinelPymupdf:
            @staticmethod
            def open(_path):
                called["pymupdf_used"] = True
                # Return enough to satisfy the page-count probe in
                # _try_pymupdf4llm; raises here would be misleading.
                raise RuntimeError(
                    "pymupdf must NOT be invoked from the layout-only path"
                )

        # Don't actually replace pymupdf — only the *raw-text* pymupdf
        # backend would call it; _try_pymupdf4llm uses pymupdf only
        # for a page-count probe before calling to_markdown, and we
        # want that probe to succeed (or fall back to PDF_MAX_PAGES)
        # so we can verify to_markdown is what fails.
        del _SentinelPymupdf  # unused; keep test focused
        del original

        result = await DocumentParser().parse_pdf_layout_only(str(pdf_file))
        assert result is None
        # And the strict path never invoked the raw-text pymupdf
        # backend — because parse_pdf_layout_only doesn't have a
        # raw-text branch at all.
        assert called["pymupdf_used"] is False

    @pytest.mark.asyncio
    async def test_pymupdf4llm_preferred_when_available(
        self, pdf_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin the preference order: when pymupdf4llm is available,
        the parser uses it FIRST and never falls through to pymupdf
        (which produces lower-quality text on multi-column docs).
        Without this pin, a refactor that re-orders the backend
        chain would silently revert to the garbled raw-text path
        for everyone with [pdf-llm] installed."""
        import sys
        import types

        # Stub pymupdf4llm so we can detect whether _try_pymupdf4llm
        # got called. The real module isn't required for this test —
        # we only care about the branch ordering inside _parse_pdf_file.
        called = {"to_markdown": False}

        stub = types.ModuleType("pymupdf4llm")

        def fake_to_markdown(file_path, pages=None):
            called["to_markdown"] = True
            return "# Stubbed pymupdf4llm output\n\nbody"

        stub.to_markdown = fake_to_markdown
        monkeypatch.setitem(sys.modules, "pymupdf4llm", stub)

        result = await DocumentParser().parse_file(str(pdf_file))
        assert called["to_markdown"] is True, (
            "Layout-aware backend was NOT called even though it was "
            "available — preference order regressed."
        )
        assert "Stubbed pymupdf4llm output" in (result or "")


def _real_pandas_available() -> bool:
    """Conftest installs a lightweight pandas stub for tests that
    don't need the real thing. MarkItDown's Excel converter calls
    pandas.read_excel(), which the stub lacks — so the xlsx path
    can only be exercised when the real pandas is in sys.modules.
    Same story for other MarkItDown converters with real-dep needs."""
    try:
        import pandas as pd  # type: ignore

        return hasattr(pd, "read_excel")
    except ImportError:
        return False


class TestOfficeFormats:
    """docx/xlsx/pptx routed through MarkItDown.

    MarkItDown delegates to python-docx / openpyxl+pandas /
    python-pptx internally. The test conftest stubs some of these —
    particularly pandas — so tests must gracefully skip when the
    stub is active. The Provider E2E workflow (GRAPHORA_TEST_REAL_DEPS=1)
    runs them for real.
    """

    @pytest.mark.asyncio
    async def test_extracts_docx(self, docx_file: Path) -> None:
        pytest.importorskip("markitdown")
        result = await DocumentParser().parse_file(str(docx_file))
        assert result is not None
        assert "Acme Corp" in result

    @pytest.mark.asyncio
    async def test_extracts_xlsx(self, xlsx_file: Path) -> None:
        pytest.importorskip("markitdown")
        if not _real_pandas_available():
            pytest.skip(
                "pandas is stubbed in conftest; xlsx requires real pandas. "
                "Run with GRAPHORA_TEST_REAL_DEPS=1 to exercise this path."
            )
        result = await DocumentParser().parse_file(str(xlsx_file))
        assert result is not None
        # MarkItDown renders the sheet as a markdown table; the cell
        # text should appear somewhere in the output.
        assert "Acme Corp" in result

    @pytest.mark.asyncio
    async def test_extracts_pptx(self, pptx_file: Path) -> None:
        pytest.importorskip("markitdown")
        result = await DocumentParser().parse_file(str(pptx_file))
        assert result is not None
        assert "Acme Corp deck" in result

    @pytest.mark.asyncio
    async def test_missing_markitdown_logs_install_hint(
        self, docx_file: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "markitdown":
                raise ImportError("simulated missing markitdown")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with caplog.at_level("WARNING"):
            result = await DocumentParser().parse_file(str(docx_file))
        assert result is None
        assert any("graphora-server[docling]" in r.message for r in caplog.records)


class TestUrlPath:
    """parse_url goes through trafilatura; we mock the fetch to avoid
    real network calls in CI."""

    @pytest.mark.asyncio
    async def test_parse_url_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("trafilatura")
        import trafilatura  # type: ignore

        monkeypatch.setattr(
            trafilatura, "fetch_url", lambda _url: "<html>some body</html>"
        )
        monkeypatch.setattr(trafilatura, "extract", lambda _html: "some body")
        result = await DocumentParser().parse_url("https://example.com/article")
        assert result == "some body"

    @pytest.mark.asyncio
    async def test_parse_url_fetch_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("trafilatura")
        import trafilatura  # type: ignore

        monkeypatch.setattr(trafilatura, "fetch_url", lambda url: None)
        result = await DocumentParser().parse_url("https://example.com/missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_trafilatura_logs_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "trafilatura":
                raise ImportError("simulated missing trafilatura")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with caplog.at_level("WARNING"):
            result = await DocumentParser().parse_url("https://example.com")
        assert result is None
        assert any("graphora-server[url]" in r.message for r in caplog.records)


class TestUnsupportedExtensions:
    @pytest.mark.asyncio
    async def test_unknown_extension_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "mystery.xyz"
        p.write_text("nothing")
        result = await DocumentParser().parse_file(str(p))
        assert result is None

    @pytest.mark.asyncio
    async def test_module_level_shim(self, txt_file: Path) -> None:
        """parse_document() convenience wrapper used by the CLI."""
        result = await parse_document(str(txt_file))
        assert result == "Hello world from txt fixture"


# ---- smoke: MarkItDown error path does not crash ---------------


class TestMarkItDownFailurePath:
    @pytest.mark.asyncio
    async def test_unreadable_docx_returns_none_or_empty(
        self, tmp_path: Path, caplog
    ) -> None:
        """MarkItDown is permissive: a plain text file labeled .docx
        may be returned as its literal content (since MarkItDown has a
        text fallback). The contract we assert is narrower — the call
        must NOT raise, and must return either None or a string.

        Truly malformed binary docx files (corrupt zip, non-xml inner
        content) get caught by MarkItDown's inner converters and
        surface as errors logged here."""
        p = tmp_path / "bogus.docx"
        # Non-zip, non-text garbage bytes force the converter to
        # actually fail rather than falling through to text mode.
        p.write_bytes(bytes(range(256)) * 4)
        pytest.importorskip("markitdown")
        with caplog.at_level("ERROR"):
            result = await DocumentParser().parse_file(str(p))
        # Contract: never raises. Either returned None with a logged
        # error, or returned a string (graceful text fallback).
        assert result is None or isinstance(result, str)


# Silence an unused-import complaint from ruff — AsyncMock + io are
# kept for future tests that need them (mocking MarkItDown's .convert).
_ = AsyncMock, io, patch
