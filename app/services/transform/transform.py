import traceback
import aiofiles
from fastapi import UploadFile, HTTPException
from typing import List
from pathlib import Path
from app.schemas.transform import FileValidationError, KnowledgeGraph, ChunkMetadata
from app.config import settings
from app.services.transform.langraph import create_processing_chain
from app.services.local_merge.ingestion_helpers import Neo4jStagingManager 
from app.services.local_merge.sanitise_ingest import sanitise_and_ingest
import logging
from app.services.job_manager import get_job_manager
from fastapi import FastAPI
from app.services.ontology_validator import parse_and_validate_yaml
import json

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

async def initialize_processing(transform_id: str, 
                                ontology_id: str, 
                                file_paths: List[str], 
                                app_instance: FastAPI) -> None:
    """Initialize and run document processing"""
    
    job_manager = get_job_manager(app_instance)
    
    try:
        # Get and validate ontology
        try:
            with open(f"{settings.ONTOLOGY_DIR}/{ontology_id}.yaml", 'r') as f:
                ontology_yaml = f.read()
            ontology_dict = parse_and_validate_yaml(ontology_yaml)
            ontology_str = ontology_yaml
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Ontology {ontology_id} not found or invalid: {str(e)}"
            )
        
        # Create chain
        chain = await create_processing_chain()
        
        total_files = len(file_paths)
        for idx, path in enumerate(file_paths, 1):
            logger.info(f"Processing file {path}")
            
            # Read file
            with open(path, 'r') as f:
                text = f.read()
                
            # Process file
            try:
                # Pass all required parameters for job status updates
                state_input = {
                    "transform_id": transform_id,
                    "text": text, 
                    "ontology_str": ontology_str,
                    "ontology_obj": ontology_dict,
                    "app": app_instance
                }
                
                result = await chain.ainvoke(state_input)
                
                await sanitise_and_ingest(
                    ontology_str, 
                    result['metadata'],
                    result['domain_graphs'],
                    Neo4jStagingManager(f"Staging_{transform_id}", is_staging=True),
                    transform_id,
                    app_instance
                )
                    
            except Exception as e:
              traceback.print_exc()
              logger.error(f"Error processing file {path}: {str(e)}")
              continue
            
            # Update progress for this file
            progress = (idx / total_files) * 100
            await job_manager.update_progress(transform_id, progress)
            
        await job_manager.complete_job(transform_id)
    except Exception as e:
        logger.error(f"Error in initialize_processing: {str(e)}")
        await job_manager.fail_job(transform_id, str(e))
        raise
