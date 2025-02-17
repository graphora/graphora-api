import json
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from typing import Optional
from app.schemas.transform import StorageLocation, DocumentMetadata

class DocumentStorage:
    def __init__(self, base_path: Path):
        """
        Initialize document storage with base path
        
        Directory structure:
        base_path/
          ├── uploads/
          │   └── {transform_id}/
          │       ├── original/
          │       ├── processed/
          │       └── metadata.json
        """
        self.base_path = Path(base_path)
        self.uploads_dir = self.base_path / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_document(
        self,
        file: UploadFile,
        transform_id: str,
        metadata: DocumentMetadata
    ) -> StorageLocation:
        """
        Save uploaded document and its metadata
        
        Args:
            file: FastAPI UploadFile object
            transform_id: Unique transform ID
            metadata: Document metadata
            
        Returns:
            StorageLocation with paths to saved files
        """
        # Create transform directory structure
        transform_dir = self.uploads_dir / transform_id
        original_dir = transform_dir / "original"
        processed_dir = transform_dir / "processed"
        
        for dir_path in [transform_dir, original_dir, processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save original file
        original_path = original_dir / file.filename
        async with aiofiles.open(original_path, 'wb') as f:
            content = await file.read()
            await file.seek(0)
            await f.write(content)
        
        # Save metadata
        metadata_path = transform_dir / "metadata.json"
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(metadata.model_dump_json(indent=2))
        
        return StorageLocation(
            transform_id=transform_id,
            original_path=str(original_path),
            processed_path=None,  # Will be set during processing
            metadata_path=str(metadata_path)
        )
