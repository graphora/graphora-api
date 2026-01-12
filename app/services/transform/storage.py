from pathlib import Path
import aiofiles
import os
from typing import Optional, Union
from app.schemas.transform import StorageLocation, DocumentMetadata
from prefect.filesystems import LocalFileSystem
from app.config import settings
import shutil


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""

    pass


class DocumentStorage:
    def __init__(self, base_path: Union[str, Path]):
        """Initialize document storage with base path"""
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _validate_path_containment(self, path: Path, base_dir: Path) -> Path:
        """
        Validate that a path is contained within the base directory.

        Args:
            path: The path to validate
            base_dir: The base directory that should contain the path

        Returns:
            The resolved path if valid

        Raises:
            PathTraversalError: If the path escapes the base directory
        """
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()

        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise PathTraversalError(
                f"Path traversal detected: '{path}' escapes base directory"
            )

        return resolved_path

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to prevent path traversal.

        Args:
            filename: The filename to sanitize

        Returns:
            A safe filename

        Raises:
            PathTraversalError: If the filename contains path traversal attempts
        """
        if not filename:
            raise PathTraversalError("Filename cannot be empty")

        # Get just the basename
        safe_name = os.path.basename(filename)

        # Check for traversal attempts
        if ".." in filename or filename != safe_name:
            raise PathTraversalError(
                f"Path traversal detected in filename: '{filename}'"
            )

        if not safe_name or safe_name in (".", ".."):
            raise PathTraversalError(f"Invalid filename: '{filename}'")

        return safe_name

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

        Raises:
            PathTraversalError: If path traversal is detected
            FileNotFoundError: If source file doesn't exist
        """
        try:
            # Sanitize transform_id to prevent path traversal
            safe_transform_id = self._sanitize_filename(transform_id)

            # Create transform directory
            transform_dir = self.base_path / safe_transform_id
            transform_dir.mkdir(parents=True, exist_ok=True)

            # Validate transform directory is within base path
            self._validate_path_containment(transform_dir, self.base_path)

            # Get source file path
            source_path = Path(file).resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")

            # Sanitize the filename from source path
            safe_filename = self._sanitize_filename(source_path.name)

            # Create destination path and validate containment
            dest_path = transform_dir / safe_filename
            dest_path = self._validate_path_containment(dest_path, transform_dir)

            # Only copy if source and destination are different
            if source_path != dest_path:
                shutil.copy2(source_path, dest_path)

            # Save metadata with validated path
            metadata_path = transform_dir / f"{safe_filename}.metadata.json"
            metadata_path = self._validate_path_containment(
                metadata_path, transform_dir
            )
            async with aiofiles.open(metadata_path, "w") as f:
                await f.write(metadata.model_dump_json())

            return StorageLocation(
                transform_id=safe_transform_id,
                original_path=str(dest_path),
                processed_path=None,  # Will be set during processing
                metadata_path=str(metadata_path),
            )

        except PathTraversalError:
            # Re-raise path traversal errors without cleanup
            raise
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

        Raises:
            PathTraversalError: If path traversal is detected
        """
        # Sanitize inputs to prevent path traversal
        safe_transform_id = self._sanitize_filename(transform_id)
        safe_filename = self._sanitize_filename(filename)

        transform_dir = self.base_path / safe_transform_id

        # Validate transform directory is within base path
        self._validate_path_containment(transform_dir, self.base_path)

        if not transform_dir.exists():
            return None

        # Build and validate document path
        doc_path = transform_dir / safe_filename
        doc_path = self._validate_path_containment(doc_path, transform_dir)

        # Build and validate metadata path
        metadata_path = transform_dir / f"{safe_filename}.metadata.json"
        metadata_path = self._validate_path_containment(metadata_path, transform_dir)

        if not doc_path.exists() or not metadata_path.exists():
            return None

        return StorageLocation(
            transform_id=safe_transform_id,
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
