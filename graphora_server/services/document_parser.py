"""Simple document parser for text extraction.

Used primarily for schema inference - extracts sample text from documents
without full chunking or processing.
"""

import logging
from pathlib import Path
from typing import Optional

import aiofiles

logger = logging.getLogger(__name__)


class DocumentParser:
    """Simple parser for extracting text from documents."""

    SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html"}
    SUPPORTED_PDF_EXTENSION = ".pdf"

    async def parse_file(self, file_path: str) -> Optional[str]:
        """Extract text content from a file.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content or None if extraction fails
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            if suffix in self.SUPPORTED_TEXT_EXTENSIONS:
                return await self._parse_text_file(file_path)
            elif suffix == self.SUPPORTED_PDF_EXTENSION:
                return await self._parse_pdf_file(file_path)
            else:
                logger.warning(f"Unsupported file type for text extraction: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return None

    async def _parse_text_file(self, file_path: str) -> Optional[str]:
        """Parse a text-based file.

        Args:
            file_path: Path to text file

        Returns:
            File content as string
        """
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                async with aiofiles.open(file_path, "r", encoding="latin-1") as f:
                    content = await f.read()
                return content
            except Exception as e:
                logger.error(f"Failed to read text file with fallback encoding: {e}")
                return None

    async def _parse_pdf_file(self, file_path: str) -> Optional[str]:
        """Extract text from a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        try:
            # Try pypdf first (commonly available)
            try:
                from pypdf import PdfReader

                reader = PdfReader(file_path)
                text_parts = []
                # Only read first few pages for schema inference
                max_pages = min(5, len(reader.pages))
                for i in range(max_pages):
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        text_parts.append(page_text)
                return "\n\n".join(text_parts)
            except ImportError:
                pass

            # Try pdfplumber as alternative
            try:
                import pdfplumber

                text_parts = []
                with pdfplumber.open(file_path) as pdf:
                    max_pages = min(5, len(pdf.pages))
                    for i in range(max_pages):
                        page_text = pdf.pages[i].extract_text()
                        if page_text:
                            text_parts.append(page_text)
                return "\n\n".join(text_parts)
            except ImportError:
                pass

            logger.warning(
                "No PDF library available (pypdf or pdfplumber). "
                "Cannot extract text from PDF for schema inference."
            )
            return None

        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return None
