import magic
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from typing import Set, List
from app.schemas.transform import ValidationResult, DocumentType

class FileValidator:
    ALLOWED_EXTENSIONS: Set[str] = {'.txt', '.pdf', '.docx', '.md'}
    MAX_FILE_SIZE: int = 52_428_800  # 50MB in bytes
    
    async def validate(self, file: UploadFile) -> ValidationResult:
        """
        Validate uploaded file against size and type restrictions
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            ValidationResult with validation status and any errors
        """
        errors: List[str] = []
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            errors.append(f"File extension {file_ext} not allowed. Allowed extensions: {self.ALLOWED_EXTENSIONS}")
        
        # Validate file size
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        if len(content) > self.MAX_FILE_SIZE:
            errors.append(f"File size exceeds maximum allowed size of {self.MAX_FILE_SIZE/1024/1024:.1f}MB")
        
        # Verify file content matches extension
        mime_type = magic.from_buffer(content, mime=True)
        content_type_valid = self._validate_content_type(mime_type, file_ext)
        
        if not content_type_valid:
            errors.append(f"File content does not match extension {file_ext}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors if errors else None
        )
    
    def _validate_content_type(self, mime_type: str, extension: str) -> bool:
        """Validate that file content matches its extension"""
        mime_ext_map = {
            '.txt': ['text/plain'],
            '.pdf': ['application/pdf'],
            # '.docx': [
            #     'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            # ],
            '.md': ['text/plain', 'text/markdown']
        }
        
        return mime_type in mime_ext_map.get(extension, [])
