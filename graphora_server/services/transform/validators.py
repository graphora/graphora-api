from fastapi import UploadFile
from graphora_server.schemas.transform import ValidationResult
import magic
import os
import re
from typing import List, Set, Dict


class FileValidator:
    """Validator for uploaded files"""

    # Allowed MIME types
    ALLOWED_MIME_TYPES: Set[str] = {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    # Maximum file size (100MB)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024

    # Maximum filename length
    MAX_FILENAME_LENGTH: int = 255

    # Allowed filename pattern (alphanumeric, dots, hyphens, underscores, spaces)
    SAFE_FILENAME_PATTERN: re.Pattern = re.compile(r"^[\w\-. ]+$")

    # Expected extensions for each MIME type
    MIME_TO_EXTENSIONS: Dict[str, Set[str]] = {
        "application/pdf": {".pdf"},
        "text/plain": {".txt", ".text"},
        "text/markdown": {".md", ".markdown"},
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
            ".docx"
        },
    }

    def _validate_filename(self, filename: str) -> List[str]:
        """
        Validate a filename for security issues.

        Args:
            filename: The filename to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not filename:
            errors.append("Filename cannot be empty")
            return errors

        # Check for path traversal attempts
        if ".." in filename:
            errors.append("Filename contains path traversal sequence '..'")

        # Check if filename contains directory separators
        if "/" in filename or "\\" in filename:
            errors.append("Filename contains directory separator characters")

        # Verify basename matches original (no path components)
        basename = os.path.basename(filename)
        if basename != filename:
            errors.append("Filename contains path components")

        # Check filename length
        if len(filename) > self.MAX_FILENAME_LENGTH:
            errors.append(
                f"Filename exceeds maximum length of {self.MAX_FILENAME_LENGTH} characters"
            )

        # Check for safe characters
        if not self.SAFE_FILENAME_PATTERN.match(filename):
            errors.append(
                "Filename contains disallowed characters. "
                "Only alphanumeric characters, dots, hyphens, underscores, and spaces are allowed"
            )

        # Check for dangerous filenames
        if filename in (".", ".."):
            errors.append("Filename cannot be '.' or '..'")

        # Check for hidden files (starting with .)
        if filename.startswith("."):
            errors.append("Hidden files (starting with '.') are not allowed")

        return errors

    def _validate_extension_matches_mime(
        self, filename: str, mime_type: str
    ) -> List[str]:
        """
        Validate that the file extension matches the detected MIME type.

        Args:
            filename: The filename to check
            mime_type: The detected MIME type

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if mime_type not in self.MIME_TO_EXTENSIONS:
            return errors  # Skip check for unknown MIME types

        extension = os.path.splitext(filename)[1].lower()
        expected_extensions = self.MIME_TO_EXTENSIONS[mime_type]

        if extension not in expected_extensions:
            errors.append(
                f"File extension '{extension}' does not match detected type '{mime_type}'. "
                f"Expected extensions: {expected_extensions}"
            )

        return errors

    async def validate(self, file: UploadFile) -> ValidationResult:
        """
        Validate an uploaded file

        Args:
            file: FastAPI UploadFile object

        Returns:
            ValidationResult with validation status and any errors
        """
        errors: List[str] = []

        try:
            # Validate filename first (security check)
            filename_errors = self._validate_filename(file.filename)
            errors.extend(filename_errors)

            # Check file size
            content = file._file if hasattr(file, "_file") else await file.read()
            if not hasattr(file, "_file"):
                await file.seek(0)

            file_size = len(content)
            if file_size > self.MAX_FILE_SIZE:
                errors.append(
                    f"File size {file_size} bytes exceeds maximum of {self.MAX_FILE_SIZE} bytes"
                )

            # Check MIME type
            mime = magic.from_buffer(content[0:2048], mime=True)
            if mime not in self.ALLOWED_MIME_TYPES:
                errors.append(
                    f"File type {mime} not allowed. Allowed types: {self.ALLOWED_MIME_TYPES}"
                )

            # Validate extension matches MIME type (prevent extension spoofing)
            if file.filename and mime in self.ALLOWED_MIME_TYPES:
                extension_errors = self._validate_extension_matches_mime(
                    file.filename, mime
                )
                errors.extend(extension_errors)

            return ValidationResult(is_valid=len(errors) == 0, errors=errors)

        except Exception as e:
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors)
