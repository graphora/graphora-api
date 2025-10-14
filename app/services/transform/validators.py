from fastapi import UploadFile
from app.schemas.transform import ValidationResult
import magic
from typing import List, Set


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

            return ValidationResult(is_valid=len(errors) == 0, errors=errors)

        except Exception as e:
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors)
