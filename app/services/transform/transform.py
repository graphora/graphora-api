import os
import aiofiles
from fastapi import UploadFile
from typing import List
import magic
from pathlib import Path

from app.schemas.transform import FileValidationError
from app.config import settings

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed MIME types and their extensions
ALLOWED_MIME_TYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'text/plain': '.txt'
}

async def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file type and size
    
    Args:
        file: FastAPI UploadFile object
        
    Raises:
        FileValidationError: If file validation fails
    """
    # Check file size
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise FileValidationError(f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE/1024/1024}MB")
        
        # Reset file pointer for later use
        await file.seek(0)
        
        # Check file type using python-magic
        mime_type = magic.from_buffer(content, mime=True)
        if mime_type not in ALLOWED_MIME_TYPES:
            raise FileValidationError(
                f"File {file.filename} has invalid type. Allowed types: PDF, DOCX, TXT"
            )
    except Exception as e:
        raise FileValidationError(f"Error validating file {file.filename}: {str(e)}")

async def save_files(transform_id: str, files: List[UploadFile]) -> List[str]:
    """
    Save uploaded files to storage
    
    Args:
        transform_id: Unique transformation ID
        files: List of FastAPI UploadFile objects
        
    Returns:
        List of saved file paths
        
    Raises:
        FileValidationError: If file saving fails
    """
    saved_files = []
    upload_dir = Path(settings.UPLOAD_DIR) / transform_id
    
    try:
        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in files:
            # Create safe filename
            file_path = upload_dir / file.filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
                saved_files.append(str(file_path))
                
        return saved_files
        
    except Exception as e:
        # Clean up any saved files on error
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass
        raise FileValidationError(f"Error saving files: {str(e)}")

async def initialize_processing(transform_id: str, ontology_id: str, file_paths: List[str]) -> None:
    """
    Initialize document processing
    
    Args:
        transform_id: Unique transformation ID
        ontology_id: Ontology ID to use for transformation
        file_paths: List of saved file paths
    """
    from datetime import datetime
    o = []
    def run_pipeline(data, ontology, form10k):
        output = []
        for text in data:
            staging_id = f"Staging_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            state_input = {"text": text, "ontology_str": ontology, "ontology_obj": form10k}
            result = app.invoke(state_input)
            output.append((staging_id, result))
            o = output
            sanitise_and_ingest(ontology, result['metadata'], result['domain_graphs'], Neo4jStagingManager(staging_id, is_staging=True))
        return output
