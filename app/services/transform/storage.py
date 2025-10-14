from pathlib import Path
import aiofiles
from typing import Optional, Union
from app.schemas.transform import StorageLocation, DocumentMetadata
from prefect.filesystems import LocalFileSystem
from app.config import settings
import shutil


class DocumentStorage:
    def __init__(self, base_path: Union[str, Path]):
        """Initialize document storage with base path"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_document(
        self, file: Union[str, Path], transform_id: str, metadata: DocumentMetadata
    ) -> StorageLocation:
        """
        Save document and metadata to storage

        Args:
            file: File path as string or Path object
            transform_id: Unique transform ID
            metadata: Document metadata

        Returns:
            StorageLocation with paths to stored files
        """
        try:
            # Create transform directory
            transform_dir = self.base_path / transform_id
            transform_dir.mkdir(parents=True, exist_ok=True)

            # Get source file path
            source_path = Path(file).resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")

            # Create destination path
            dest_path = (transform_dir / source_path.name).resolve()

            # Only copy if source and destination are different
            if source_path != dest_path:
                shutil.copy2(source_path, dest_path)

            # Save metadata
            metadata_path = transform_dir / f"{source_path.name}.metadata.json"
            async with aiofiles.open(metadata_path, "w") as f:
                await f.write(metadata.model_dump_json())

            return StorageLocation(
                transform_id=transform_id,
                original_path=str(dest_path),
                processed_path=None,  # Will be set during processing
                metadata_path=str(metadata_path),
            )

        except Exception as e:
            # Clean up on error
            if "transform_dir" in locals() and transform_dir.exists():
                shutil.rmtree(transform_dir)
            raise Exception(f"Failed to save document: {str(e)}")

    async def get_document(
        self, transform_id: str, filename: str
    ) -> Optional[StorageLocation]:
        """
        Get document and metadata from storage

        Args:
            transform_id: Unique transform ID
            filename: Name of document file

        Returns:
            StorageLocation if found, None otherwise
        """
        transform_dir = self.base_path / transform_id
        if not transform_dir.exists():
            return None

        doc_path = transform_dir / filename
        metadata_path = transform_dir / f"{filename}.metadata.json"

        if not doc_path.exists() or not metadata_path.exists():
            return None

        return StorageLocation(
            transform_id=transform_id,
            original_path=str(doc_path),
            processed_path=None,
            metadata_path=str(metadata_path),
        )


def get_flow_storage():
    """Get local filesystem storage for Prefect flows"""
    storage_path = Path(settings.UPLOAD_DIR) / "prefect" / "flows"
    storage_path.mkdir(parents=True, exist_ok=True)

    return LocalFileSystem(
        basepath=str(storage_path),
    )
