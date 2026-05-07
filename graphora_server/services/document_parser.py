"""Document parser for text extraction.

Primary use: schema inference — grab a representative text sample
from an uploaded document so the LLM can propose an ontology. The
main extraction path (flows.py + chunking) handles files directly
via pypdf and text splitters; DocumentParser is the "tell me what's
in this document without running the pipeline" shortcut.

Supported surfaces:

- Plain text (`.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`) —
  base install, no extras.
- PDF (`.pdf`) — tries backends in this order:
    1. `pymupdf4llm` (preferred, layout-aware Markdown). Install
       with `graphora-server[pdf-llm]` — handles multi-column,
       tables, and heading hierarchy correctly. Used by the
       Evidence-tab source_text capture in flows.py.
    2. `pymupdf` (raw text, no layout awareness). `[pdf]` extra.
    3. `pypdf` (pure-Python fallback). Also `[pdf]`.
    4. `pdfplumber` (legacy fallback). Client-extras.
  ``has_layout_aware_backend()`` reports whether (1) is available
  so callers like flows.py can decide whether the output quality
  is acceptable for their use case.
- Office (`.docx`, `.xlsx`, `.pptx`) — via Microsoft's MarkItDown.
  Install with `graphora-server[docling]`.
- URL — `parse_url()` method, via trafilatura. Install with
  `graphora-server[url]`.

Every optional backend is lazy-imported. Missing extras surface a
clear install hint in the error log rather than a cryptic
ModuleNotFoundError.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import aiofiles

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parser for extracting text from documents and URLs."""

    SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html"}
    SUPPORTED_PDF_EXTENSION = ".pdf"
    # MarkItDown-backed Office formats. Separate from the text set
    # because they follow a different extraction path.
    SUPPORTED_OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx"}

    # PDF sample ceiling — schema inference only needs the opening
    # context, and large PDFs would bloat the LLM prompt. Applied
    # uniformly across backends so the output is comparable.
    PDF_MAX_PAGES = 5

    async def parse_file(self, file_path: str) -> Optional[str]:
        """Extract text content from a file.

        Routes by file extension. Returns None (with a logged warning)
        when the extension is unsupported or the backend isn't
        installed — callers decide how to handle missing samples
        (schema inference falls back to the first chunk's raw text).

        Args:
            file_path: Path to the file.

        Returns:
            Extracted text content or None on failure.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            if suffix in self.SUPPORTED_TEXT_EXTENSIONS:
                return await self._parse_text_file(file_path)
            elif suffix == self.SUPPORTED_PDF_EXTENSION:
                return await self._parse_pdf_file(file_path)
            elif suffix in self.SUPPORTED_OFFICE_EXTENSIONS:
                return await self._parse_office_file(file_path)
            else:
                logger.warning(f"Unsupported file type for text extraction: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return None

    async def _parse_text_file(self, file_path: str) -> Optional[str]:
        """Parse a text-based file with encoding fallback."""
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                return await f.read()
        except UnicodeDecodeError:
            try:
                async with aiofiles.open(file_path, "r", encoding="latin-1") as f:
                    return await f.read()
            except Exception as e:
                logger.error(f"Failed to read text file with fallback encoding: {e}")
                return None

    @classmethod
    def has_layout_aware_backend(cls) -> bool:
        """Whether the layout-aware PDF backend (pymupdf4llm) is
        importable. Callers like flows.py gate Evidence-tab source_text
        capture on this — without the layout-aware backend, raw-text
        backends (pymupdf/pypdf) garble multi-column 10K-style PDFs
        and the wrong text is worse than no text."""
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError:
            return False
        return True

    async def parse_pdf_layout_only(self, file_path: str) -> Optional[str]:
        """Layout-aware PDF parse with NO raw-text fallback.

        Reviewer-flagged hole on the original Evidence-tab fix
        (commit 920b8f9): flows.py was calling ``parse_file`` after
        gating on ``has_layout_aware_backend``, but parse_file's PDF
        chain falls through to pymupdf/pypdf/pdfplumber when
        pymupdf4llm fails on a specific document — so a corrupt or
        unusual PDF still produced garbled raw-text output and
        landed in source_text, exactly what beafa92 was designed
        to prevent.

        This method is the strict-quality variant: pymupdf4llm
        succeeds → return its Markdown; pymupdf4llm not installed
        OR fails on this file → return None. The raw-text backends
        are unreachable from here.

        Schema inference still uses ``parse_file`` (with the
        full fallback chain) because there 'rough text' is better
        than 'no text' — the LLM tolerates lossy input. Evidence
        tab needs the opposite trade-off: 'no text' is better than
        'wrong text' because users read it directly.
        """
        return self._try_pymupdf4llm(file_path)

    async def _parse_pdf_file(self, file_path: str) -> Optional[str]:
        """Extract text from a PDF file.

        Preference order:
            1. pymupdf4llm (preferred, layout-aware Markdown — handles
               multi-column / tables / heading hierarchy). Install
               with `graphora-server[pdf-llm]`.
            2. pymupdf (raw text, no layout awareness).
            3. pypdf (pure-Python fallback).
            4. pdfplumber (legacy fallback).

        (2) and (3) ship in `graphora-server[pdf]`. If none of (1-4)
        are installed, a clear install-hint is logged and None is
        returned.
        """
        text = self._try_pymupdf4llm(file_path)
        if text is not None:
            return text
        text = self._try_pymupdf(file_path)
        if text is not None:
            return text
        text = self._try_pypdf(file_path)
        if text is not None:
            return text
        text = self._try_pdfplumber(file_path)
        if text is not None:
            return text

        logger.warning(
            "No PDF backend available. "
            "Install with: pip install 'graphora-server[pdf-llm]' "
            "(layout-aware) or 'graphora-server[pdf]' (raw-text)"
        )
        return None

    def _try_pymupdf4llm(self, file_path: str) -> Optional[str]:
        """Extract via pymupdf4llm. Returns Markdown that respects
        column boundaries, tables, and heading hierarchy — the layout
        cues are exactly what pymupdf/pypdf raw-text extraction
        loses. Returns None if not installed.

        We cap pages at PDF_MAX_PAGES for parity with the other
        backends (schema-inference doesn't need more), but the
        cap is per-call: flows.py's Evidence-tab capture passes
        single-split files and consumes the full output."""
        try:
            import pymupdf4llm  # type: ignore
        except ImportError:
            return None

        try:
            # to_markdown's ``pages`` arg takes a list of page
            # indices when set. Without it, all pages are processed.
            try:
                import pymupdf  # type: ignore

                doc = pymupdf.open(file_path)
                try:
                    page_count = doc.page_count
                finally:
                    doc.close()
            except Exception:
                page_count = self.PDF_MAX_PAGES

            max_pages = min(self.PDF_MAX_PAGES, page_count)
            return pymupdf4llm.to_markdown(
                file_path,
                pages=list(range(max_pages)),
            )
        except Exception as e:
            logger.warning("pymupdf4llm failed on %s: %s; falling back", file_path, e)
            return None

    def _try_pymupdf(self, file_path: str) -> Optional[str]:
        """Extract via PyMuPDF (preferred). Returns None if not installed."""
        try:
            import pymupdf  # type: ignore
        except ImportError:
            return None

        try:
            doc = pymupdf.open(file_path)
            try:
                max_pages = min(self.PDF_MAX_PAGES, doc.page_count)
                parts = []
                for i in range(max_pages):
                    page_text = doc[i].get_text()
                    if page_text:
                        parts.append(page_text)
                return "\n\n".join(parts) if parts else None
            finally:
                doc.close()
        except Exception as e:
            logger.warning(f"pymupdf failed on {file_path}: {e}; falling back")
            return None

    def _try_pypdf(self, file_path: str) -> Optional[str]:
        """Extract via pypdf. Returns None if not installed."""
        try:
            from pypdf import PdfReader
        except ImportError:
            return None

        try:
            reader = PdfReader(file_path)
            max_pages = min(self.PDF_MAX_PAGES, len(reader.pages))
            parts = []
            for i in range(max_pages):
                page_text = reader.pages[i].extract_text()
                if page_text:
                    parts.append(page_text)
            return "\n\n".join(parts) if parts else None
        except Exception as e:
            logger.warning(f"pypdf failed on {file_path}: {e}; falling back")
            return None

    def _try_pdfplumber(self, file_path: str) -> Optional[str]:
        """Extract via pdfplumber (legacy client-extras fallback)."""
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            return None

        try:
            parts = []
            with pdfplumber.open(file_path) as pdf:
                max_pages = min(self.PDF_MAX_PAGES, len(pdf.pages))
                for i in range(max_pages):
                    page_text = pdf.pages[i].extract_text()
                    if page_text:
                        parts.append(page_text)
            return "\n\n".join(parts) if parts else None
        except Exception as e:
            logger.warning(f"pdfplumber failed on {file_path}: {e}")
            return None

    async def _parse_office_file(self, file_path: str) -> Optional[str]:
        """Extract text from docx/xlsx/pptx via MarkItDown.

        MarkItDown returns Markdown; we hand the Markdown straight to
        schema inference (the LLM is fine with Markdown as context,
        and the formatting is a useful hint for identifying sections
        and tables).
        """
        try:
            from markitdown import MarkItDown  # type: ignore
        except ImportError:
            logger.warning(
                "Office parsing requires the [docling] extra. "
                "Install with: pip install 'graphora-server[docling]'"
            )
            return None

        try:
            md = MarkItDown()
            result = md.convert(file_path)
            return result.text_content if result else None
        except Exception as e:
            logger.error(f"MarkItDown failed on {file_path}: {e}")
            return None

    async def parse_url(self, url: str) -> Optional[str]:
        """Extract main article text from a URL via trafilatura.

        Strips navigation, ads, comments, and other boilerplate —
        closer to what a human would read than a raw HTML dump.
        Returns None when trafilatura isn't installed or the fetch
        fails.

        Install with: pip install 'graphora-server[url]'.
        """
        try:
            import trafilatura  # type: ignore
        except ImportError:
            logger.warning(
                "URL parsing requires the [url] extra. "
                "Install with: pip install 'graphora-server[url]'"
            )
            return None

        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                logger.warning(f"trafilatura could not fetch {url}")
                return None
            text = trafilatura.extract(downloaded)
            return text if text else None
        except Exception as e:
            logger.error(f"trafilatura failed on {url}: {e}")
            return None


# Module-level helper to match the callsite pattern used elsewhere
# (e.g. graphora-client/cli/commands/schema.py imports `parse_document`
# as a standalone function).
async def parse_document(file_path: str) -> Optional[str]:
    """Module-level shim around DocumentParser.parse_file."""
    return await DocumentParser().parse_file(file_path)
