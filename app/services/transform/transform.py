import os
import aiofiles
from fastapi import UploadFile
from typing import List
from pathlib import Path
from app.schemas.transform import FileValidationError
from app.config import settings
from app.api.ontology import ontology_cache
from app.services.transform.langraph import app
from app.services.local_merge.ingestion_helpers import Neo4jStagingManager 
from app.services.local_merge.sanitise_ingest import sanitise_and_ingest

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed file extensions and their MIME types
ALLOWED_EXTENSIONS = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain'
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
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise FileValidationError(
                f"File {file.filename} has invalid extension. Allowed types: PDF, DOCX, TXT"
            )
        
        # Verify content type from file header matches extension
        content_type = file.content_type
        if content_type != ALLOWED_EXTENSIONS[file_ext]:
            raise FileValidationError(
                f"File {file.filename} content type {content_type} does not match its extension"
            )
            
    except FileValidationError:
        raise
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

async def initialize_processing(
    transform_id: str, ontology_id: str, file_paths: List[str]) -> None:
    """
    Initialize document processing
    
    Args:
        transform_id: Unique transformation ID
        ontology_id: Ontology ID to use for transformation
        file_paths: List of saved file paths
    """
    output = []
    for path in file_paths:
      with open(path, 'r') as f:
          text = f.read()
      staging_id = f"Staging_{transform_id}"
      ontology_str, ontology = ontology_cache[ontology_id]
      state_input = {"text": text, "ontology_str": ontology_str, "ontology_obj": ontology}
      result = app().invoke(state_input)
      output.append((staging_id, result))
      sanitise_and_ingest(ontology_str, result['metadata'], result['domain_graphs'], 
                          Neo4jStagingManager(staging_id, is_staging=True))
    return output
