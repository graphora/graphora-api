import os
import aiofiles
from fastapi import UploadFile
from typing import List
from pathlib import Path
from app.schemas.transform import FileValidationError
from app.config import settings
from app.services.transform.langraph import app
from app.services.local_merge.ingestion_helpers import Neo4jStagingManager 
from app.services.local_merge.sanitise_ingest import sanitise_and_ingest
import logging
from app.services.job_manager import get_job_manager
from fastapi import FastAPI
from app.api.ontology import ontology_cache

logger = logging.getLogger(__name__)

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed file types
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx'}

async def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file
    
    Args:
        file: FastAPI UploadFile object
        
    Raises:
        FileValidationError: If file is invalid
    """
    try:
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset position
        
        if size > MAX_FILE_SIZE:
            raise FileValidationError(
                f"File {file.filename} is too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
            )
            
        # Check extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise FileValidationError(
                f"File type {ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
            
    except Exception as e:
        if not isinstance(e, FileValidationError):
            raise FileValidationError(f"Error validating file {file.filename}: {str(e)}")
        raise

async def save_files(transform_id: str, files: List[UploadFile]) -> List[str]:
    """
    Save uploaded files
    
    Args:
        transform_id: Unique transformation ID
        files: List of FastAPI UploadFile objects
        
    Returns:
        List of saved file paths
        
    Raises:
        FileValidationError: If files cannot be saved
    """
    try:
        # Create upload directory
        upload_dir = Path(settings.UPLOAD_DIR) / transform_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for file in files:
            # Save file
            file_path = upload_dir / file.filename
            try:
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)
                saved_files.append(str(file_path))
            except Exception as e:
                # Clean up on error
                for path in saved_files:
                    try:
                        Path(path).unlink()
                    except:
                        pass
                raise FileValidationError(f"Error saving file {file.filename}: {str(e)}")
                
        return saved_files
        
    except Exception as e:
        # Clean up directory on error
        try:
            if upload_dir.exists():
                for file in upload_dir.iterdir():
                    file.unlink()
                upload_dir.rmdir()
        except:
            pass
        raise FileValidationError(f"Error saving files: {str(e)}")

async def initialize_processing(transform_id: str, ontology_id: str, file_paths: List[str], app_instance: FastAPI) -> str:
    """
    Initialize document processing
    
    Args:
        transform_id: Unique transformation ID
        ontology_id: Ontology ID to use for transformation
        file_paths: List of saved file paths
        app_instance: FastAPI app instance for job management
        
    Returns:
        transform_id: The transformation ID
    """
    try:
        output = []
        total_files = len(file_paths)
        job_manager = get_job_manager(app_instance)
        
        for idx, path in enumerate(file_paths, 1):
            async with aiofiles.open(path, 'r') as f:
                text = await f.read()
            staging_id = f"Staging_{transform_id}"
            ontology_str, ontology = ontology_cache[ontology_id]
            state_input = {
              "transform_id": transform_id,
              "text": text, 
              "ontology_str": ontology_str, 
              "ontology_obj": ontology,
              "app": app_instance
            }
            
            # Process file
            chain = app()
            try:
                result = await chain.ainvoke(state_input)
                await job_manager.update_progress(transform_id, 50.0)
                
                if result.get('metadata') and result.get('domain_graphs'):
                    output.append((staging_id, result))
                    
                    # Ingest results
                    await sanitise_and_ingest(ontology_str, result['metadata'], result['domain_graphs'], 
                                    Neo4jStagingManager(staging_id, is_staging=True),
                                    transform_id,
                                    app_instance)
                else:
                    logger.warning(f"Skipping file {path} due to missing metadata or domain graphs")
                    
            except Exception as e:
                logger.error(f"Error processing file {path}: {str(e)}")
                continue
            
            # Update progress
            progress = (idx / total_files) * 100
            await job_manager.update_progress(transform_id, progress)
            
        return transform_id
        
    except Exception as e:
        logger.error(f"Processing failed for transform_id {transform_id}: {str(e)}")
        raise
