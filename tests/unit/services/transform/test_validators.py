"""Unit tests for Transform Validators.

Phase 5: Transform Service Tests - File Validators
Tests for file upload validation and security checks.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from io import BytesIO

from app.services.transform.validators import FileValidator


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def file_validator():
    """Create FileValidator instance."""
    return FileValidator()


@pytest.fixture
def mock_upload_file():
    """Create a factory for mock upload files."""
    def _create(
        filename: str = "document.pdf",
        content: bytes = b"%PDF-1.4 mock pdf content",
        content_type: str = "application/pdf",
    ):
        mock_file = MagicMock()
        mock_file.filename = filename
        mock_file.content_type = content_type
        mock_file._file = content
        mock_file.read = AsyncMock(return_value=content)
        mock_file.seek = AsyncMock()
        return mock_file
    return _create


# ============================================================
# Filename Validation Tests
# ============================================================


class TestFilenameValidation:
    """Test filename validation for security."""

    def test_should_accept_valid_filenames(self, file_validator):
        """Should accept valid filenames."""
        valid_filenames = [
            "document.pdf",
            "my-file_2024.txt",
            "Report 2024.docx",
            "file.md",
            "document.markdown",
        ]

        for filename in valid_filenames:
            errors = file_validator._validate_filename(filename)
            assert errors == [], f"Expected no errors for '{filename}', got {errors}"

    def test_should_reject_path_traversal_attempts(self, file_validator):
        """Should reject filenames with path traversal sequences."""
        errors = file_validator._validate_filename("../etc/passwd")
        assert len(errors) > 0
        assert any("path traversal" in e.lower() or "directory" in e.lower() for e in errors)

    def test_should_reject_directory_separators(self, file_validator):
        """Should reject filenames with directory separators."""
        # Forward slash
        errors = file_validator._validate_filename("path/to/file.txt")
        assert len(errors) > 0

        # Backslash
        errors = file_validator._validate_filename("path\\to\\file.txt")
        assert len(errors) > 0

    def test_should_reject_empty_filenames(self, file_validator):
        """Should reject empty filenames."""
        errors = file_validator._validate_filename("")
        assert len(errors) > 0
        assert any("empty" in e.lower() for e in errors)

    def test_should_reject_none_filename(self, file_validator):
        """Should reject None filename."""
        errors = file_validator._validate_filename(None)
        assert len(errors) > 0

    def test_should_reject_long_filenames(self, file_validator):
        """Should reject filenames exceeding max length."""
        long_filename = "a" * 300 + ".pdf"
        errors = file_validator._validate_filename(long_filename)
        assert len(errors) > 0
        assert any("length" in e.lower() or "exceeds" in e.lower() for e in errors)

    def test_should_reject_unsafe_characters(self, file_validator):
        """Should reject filenames with unsafe characters."""
        unsafe_filenames = [
            "file;rm -rf.txt",
            "file`whoami`.txt",
            "file$(cat /etc/passwd).txt",
            "file|cat.txt",
            "file>output.txt",
            "file<input.txt",
        ]

        for filename in unsafe_filenames:
            errors = file_validator._validate_filename(filename)
            assert len(errors) > 0, f"Expected errors for unsafe filename '{filename}'"

    def test_should_reject_dot_filenames(self, file_validator):
        """Should reject . and .. as filenames."""
        errors_dot = file_validator._validate_filename(".")
        assert len(errors_dot) > 0

        errors_dotdot = file_validator._validate_filename("..")
        assert len(errors_dotdot) > 0

    def test_should_reject_hidden_files(self, file_validator):
        """Should reject hidden files starting with dot."""
        errors = file_validator._validate_filename(".hidden_file.txt")
        assert len(errors) > 0
        assert any("hidden" in e.lower() for e in errors)


# ============================================================
# Extension Validation Tests
# ============================================================


class TestExtensionValidation:
    """Test file extension validation."""

    def test_should_accept_matching_pdf_extension(self, file_validator):
        """Should accept .pdf extension for PDF MIME type."""
        errors = file_validator._validate_extension_matches_mime(
            "document.pdf", "application/pdf"
        )
        assert errors == []

    def test_should_accept_matching_txt_extension(self, file_validator):
        """Should accept .txt extension for text/plain MIME type."""
        errors = file_validator._validate_extension_matches_mime(
            "document.txt", "text/plain"
        )
        assert errors == []

    def test_should_accept_matching_md_extension(self, file_validator):
        """Should accept .md extension for text/markdown MIME type."""
        errors = file_validator._validate_extension_matches_mime(
            "README.md", "text/markdown"
        )
        assert errors == []

    def test_should_accept_matching_docx_extension(self, file_validator):
        """Should accept .docx extension for Word MIME type."""
        errors = file_validator._validate_extension_matches_mime(
            "document.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert errors == []

    def test_should_reject_mismatched_extension(self, file_validator):
        """Should reject when extension doesn't match MIME type."""
        errors = file_validator._validate_extension_matches_mime(
            "document.txt", "application/pdf"
        )
        assert len(errors) > 0
        assert any("does not match" in e.lower() for e in errors)

    def test_should_skip_validation_for_unknown_mime(self, file_validator):
        """Should skip extension check for unknown MIME types."""
        errors = file_validator._validate_extension_matches_mime(
            "file.xyz", "application/unknown"
        )
        assert errors == []

    def test_should_handle_case_insensitive_extensions(self, file_validator):
        """Should handle extensions case-insensitively."""
        errors = file_validator._validate_extension_matches_mime(
            "DOCUMENT.PDF", "application/pdf"
        )
        assert errors == []


# ============================================================
# Full Validation Tests (with mocking)
# ============================================================


class TestFullFileValidation:
    """Test full file validation with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_should_validate_valid_pdf(self, file_validator, mock_upload_file):
        """Should accept valid PDF file."""
        mock_file = mock_upload_file(
            filename="document.pdf",
            content=b"%PDF-1.4 mock pdf content" * 100,
        )

        with patch("app.services.transform.validators.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "application/pdf"

            result = await file_validator.validate(mock_file)

            assert result.is_valid is True
            assert result.errors == []

    @pytest.mark.asyncio
    async def test_should_validate_valid_txt(self, file_validator, mock_upload_file):
        """Should accept valid text file."""
        mock_file = mock_upload_file(
            filename="document.txt",
            content=b"This is plain text content",
        )

        with patch("app.services.transform.validators.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "text/plain"

            result = await file_validator.validate(mock_file)

            assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_should_reject_disallowed_mime_type(
        self, file_validator, mock_upload_file
    ):
        """Should reject files with disallowed MIME types."""
        mock_file = mock_upload_file(
            filename="script.exe",
            content=b"MZ" + b"\x00" * 100,  # PE header
        )

        with patch("app.services.transform.validators.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "application/x-executable"

            result = await file_validator.validate(mock_file)

            assert result.is_valid is False
            assert any("not allowed" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_should_reject_oversized_file(self, file_validator, mock_upload_file):
        """Should reject files exceeding max size."""
        # Create content larger than 100MB limit
        large_content = b"x" * (101 * 1024 * 1024)
        mock_file = mock_upload_file(
            filename="large.pdf",
            content=large_content,
        )

        with patch("app.services.transform.validators.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "application/pdf"

            result = await file_validator.validate(mock_file)

            assert result.is_valid is False
            assert any("exceeds" in e.lower() or "size" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_should_reject_extension_mismatch(
        self, file_validator, mock_upload_file
    ):
        """Should reject when extension doesn't match detected MIME type."""
        mock_file = mock_upload_file(
            filename="actually-pdf.txt",  # Wrong extension
            content=b"%PDF-1.4 mock pdf content",
        )

        with patch("app.services.transform.validators.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "application/pdf"

            result = await file_validator.validate(mock_file)

            assert result.is_valid is False
            assert any("does not match" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_should_handle_validation_exceptions(
        self, file_validator, mock_upload_file
    ):
        """Should handle exceptions during validation gracefully."""
        mock_file = mock_upload_file(filename="document.pdf")

        with patch("app.services.transform.validators.magic") as mock_magic:
            mock_magic.from_buffer.side_effect = Exception("Magic library error")

            result = await file_validator.validate(mock_file)

            assert result.is_valid is False
            assert any("validation failed" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_should_combine_multiple_errors(
        self, file_validator, mock_upload_file
    ):
        """Should combine multiple validation errors."""
        # Invalid filename AND wrong MIME type
        mock_file = mock_upload_file(
            filename="../../../etc/passwd",  # Path traversal
            content=b"malicious content",
        )

        with patch("app.services.transform.validators.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "application/x-executable"

            result = await file_validator.validate(mock_file)

            assert result.is_valid is False
            # Should have multiple errors
            assert len(result.errors) >= 2


# ============================================================
# Allowed Types Tests
# ============================================================


class TestAllowedTypes:
    """Test allowed MIME types and extensions."""

    def test_should_have_pdf_in_allowed_types(self, file_validator):
        """Should allow PDF files."""
        assert "application/pdf" in file_validator.ALLOWED_MIME_TYPES

    def test_should_have_plain_text_in_allowed_types(self, file_validator):
        """Should allow plain text files."""
        assert "text/plain" in file_validator.ALLOWED_MIME_TYPES

    def test_should_have_markdown_in_allowed_types(self, file_validator):
        """Should allow markdown files."""
        assert "text/markdown" in file_validator.ALLOWED_MIME_TYPES

    def test_should_have_docx_in_allowed_types(self, file_validator):
        """Should allow DOCX files."""
        docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert docx_mime in file_validator.ALLOWED_MIME_TYPES

    def test_should_not_allow_executable_types(self, file_validator):
        """Should not allow executable file types."""
        dangerous_types = [
            "application/x-executable",
            "application/x-msdos-program",
            "application/x-sh",
            "application/javascript",
        ]
        for mime_type in dangerous_types:
            assert mime_type not in file_validator.ALLOWED_MIME_TYPES


# ============================================================
# Configuration Tests
# ============================================================


class TestValidatorConfiguration:
    """Test validator configuration values."""

    def test_max_file_size_should_be_100mb(self, file_validator):
        """Max file size should be 100MB."""
        expected_size = 100 * 1024 * 1024  # 100MB in bytes
        assert file_validator.MAX_FILE_SIZE == expected_size

    def test_max_filename_length_should_be_255(self, file_validator):
        """Max filename length should be 255 characters."""
        assert file_validator.MAX_FILENAME_LENGTH == 255

    def test_should_have_extension_mappings_for_all_allowed_types(self, file_validator):
        """Should have extension mappings for all allowed MIME types."""
        for mime_type in file_validator.ALLOWED_MIME_TYPES:
            assert mime_type in file_validator.MIME_TO_EXTENSIONS
            assert len(file_validator.MIME_TO_EXTENSIONS[mime_type]) > 0
