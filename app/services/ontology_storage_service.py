"""Ontology Storage Service for storing ontologies in Postgres."""

import logging
import uuid
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.config import settings
from app.db import postgres as db

logger = logging.getLogger(__name__)


def _stringify_row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    record = dict(row)
    for key, value in list(record.items()):
        if isinstance(value, uuid.UUID):
            record[key] = str(value)
    return record


class OntologyStorageService:
    """Service for storing and retrieving ontologies from Postgres."""

    def __init__(self):
        if not (settings.DATABASE_URL or settings.resolved_database_url):
            if not settings.test_mode:
                raise ValueError("DATABASE_URL must be configured for ontology storage")

    def _increment_version(self, version: str) -> str:
        """Increment the minor version number

        Args:
            version: Current version string (e.g., "1.2.3")

        Returns:
            str: Incremented version string (e.g., "1.3.0")
        """
        try:
            parts = version.split(".")
            if len(parts) >= 2:
                major = int(parts[0])
                minor = int(parts[1])
                return f"{major}.{minor + 1}.0"
            else:
                return "1.1.0"
        except (ValueError, IndexError):
            return "1.1.0"

    async def _store_version_backup(self, ontology_record: Dict[str, Any]) -> bool:
        """Store a backup of the ontology version before updating

        Args:
            ontology_record: The current ontology record to backup

        Returns:
            bool: True if backup was successful, False otherwise
        """
        try:
            # Create a backup record in ontology_versions table (if it exists)
            # For now, we'll just log the version change
            logger.info(
                f"Backing up ontology {ontology_record.get('id')} version {ontology_record.get('version')}"
            )

            # TODO: If you want full version history, create an 'ontology_versions' table
            # and store the backup there. For now, we'll just update in place with version increment.

            return True
        except Exception as e:
            logger.error(f"Error storing version backup: {str(e)}")
            return False

    async def store_ontology(
        self,
        user_id: str,
        ontology_id: str,
        yaml_content: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_update: bool = False,
    ) -> bool:
        """Store ontology in Supabase

        Args:
            user_id: User ID
            ontology_id: Unique ontology ID
            yaml_content: YAML content of the ontology
            name: Optional name for the ontology
            description: Optional description
            is_update: If True, create a new version and deactivate old versions

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Parse YAML to extract version and metadata
            ontology_data = yaml.safe_load(yaml_content)
            yaml_version = ontology_data.get("version", "1.0.0")

            # Extract name from ontology if not provided
            if not name:
                name = f"Ontology {ontology_id[:8]}"

            existing = await db.fetchrow(
                """
                SELECT *
                FROM ontologies
                WHERE id = %s AND user_id = %s AND is_active = TRUE
                LIMIT 1
                """,
                ontology_id,
                user_id,
            )

            if existing and is_update:
                # This is an update to an existing ontology
                # Get the current version and increment it
                current_version = existing.get("version", "1.0.0")
                new_version = self._increment_version(current_version)

                # Store the previous version as a backup (optional - for history tracking)
                await self._store_version_backup(existing)

                # Update the existing record with new version and content
                row = await db.fetchrow(
                    """
                    UPDATE ontologies
                    SET name = %s,
                        file_name = %s,
                        description = %s,
                        yaml_content = %s,
                        version = %s,
                        updated_at = NOW()
                    WHERE id = %s AND user_id = %s AND is_active = TRUE
                    RETURNING id
                    """,
                    name,
                    f"{ontology_id}.yaml",
                    description or existing.get("description"),
                    yaml_content,
                    new_version,
                    ontology_id,
                    user_id,
                )

            elif existing:
                # This is a regular update (not versioned) - preserve version
                version = existing.get("version", yaml_version)

                row = await db.fetchrow(
                    """
                    UPDATE ontologies
                    SET name = %s,
                        file_name = %s,
                        description = %s,
                        yaml_content = %s,
                        version = %s,
                        updated_at = NOW()
                    WHERE id = %s AND user_id = %s AND is_active = TRUE
                    RETURNING id
                    """,
                    name,
                    f"{ontology_id}.yaml",
                    description or existing.get("description"),
                    yaml_content,
                    version,
                    ontology_id,
                    user_id,
                )
            else:
                # This is a new ontology
                version = yaml_version

                row = await db.fetchrow(
                    """
                    INSERT INTO ontologies (
                        id,
                        user_id,
                        name,
                        file_name,
                        description,
                        yaml_content,
                        version,
                        is_active,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, TRUE, NOW(), NOW()
                    )
                    RETURNING id
                    """,
                    ontology_id,
                    user_id,
                    name,
                    f"{ontology_id}.yaml",
                    description,
                    yaml_content,
                    version,
                )

            if row:
                logger.info(
                    f"Successfully stored ontology {ontology_id} for user {user_id} (update: {is_update})"
                )
                return True
            else:
                logger.error(
                    f"Failed to store ontology {ontology_id} for user {user_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Error storing ontology: {str(e)}")
            return False

    async def get_ontology(
        self, user_id: str, ontology_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get ontology by ID for a specific user

        Args:
            user_id: User ID
            ontology_id: Ontology ID

        Returns:
            Dict containing ontology data or None if not found
        """
        try:
            row = await db.fetchrow(
                """
                SELECT *
                FROM ontologies
                WHERE id = %s AND user_id = %s AND is_active = TRUE
                LIMIT 1
                """,
                ontology_id,
                user_id,
            )

            record = _stringify_row(row)
            if record:
                return record
            else:
                logger.warning(f"Ontology {ontology_id} not found for user {user_id}")
                return None

        except Exception as e:
            logger.error(f"Error fetching ontology: {str(e)}")
            return None

    async def list_ontologies(self, user_id: str) -> List[Dict[str, Any]]:
        """List all active ontologies for a user

        Args:
            user_id: User ID

        Returns:
            List of ontology records
        """
        try:
            rows = await db.fetch(
                """
                SELECT id, name, file_name, description, version, created_at, updated_at
                FROM ontologies
                WHERE user_id = %s AND is_active = TRUE
                ORDER BY updated_at DESC
                """,
                user_id,
            )

            return [_stringify_row(row) for row in rows or []]

        except Exception as e:
            logger.error(f"Error listing ontologies: {str(e)}")
            return []

    async def delete_ontology(self, user_id: str, ontology_id: str) -> bool:
        """Soft delete an ontology (mark as inactive)

        Args:
            user_id: User ID
            ontology_id: Ontology ID

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            row = await db.fetchrow(
                """
                UPDATE ontologies
                SET is_active = FALSE, updated_at = NOW()
                WHERE id = %s AND user_id = %s AND is_active = TRUE
                RETURNING id
                """,
                ontology_id,
                user_id,
            )

            if row:
                logger.info(
                    f"Successfully deleted ontology {ontology_id} for user {user_id}"
                )
                return True
            else:
                logger.error(
                    f"Failed to delete ontology {ontology_id} for user {user_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Error deleting ontology: {str(e)}")
            return False

    async def get_ontology_versions(
        self, user_id: str, ontology_id: str
    ) -> List[Dict[str, Any]]:
        """Get version information for an ontology

        Args:
            user_id: User ID
            ontology_id: Ontology ID

        Returns:
            List containing the current version info (single item for now)
        """
        try:
            # Since we're now updating in place, we only have the current version
            rows = await db.fetch(
                """
                SELECT id, name, version, created_at, updated_at, is_active
                FROM ontologies
                WHERE id = %s AND user_id = %s AND is_active = TRUE
                """,
                ontology_id,
                user_id,
            )

            return [_stringify_row(row) for row in rows or []]

        except Exception as e:
            logger.error(f"Error fetching ontology version info: {str(e)}")
            return []

    async def create_file_backup(self, user_id: str, ontology_id: str) -> Optional[str]:
        """Create a file backup of the ontology (for backward compatibility)

        Args:
            user_id: User ID
            ontology_id: Ontology ID

        Returns:
            str: File path if successful, None otherwise
        """
        try:
            ontology = await self.get_ontology(user_id, ontology_id)
            if not ontology:
                return None

            # Ensure ontology directory exists
            ontology_dir = Path(settings.ONTOLOGY_DIR).expanduser()
            ontology_dir.mkdir(parents=True, exist_ok=True)

            # Write to file
            file_path = ontology_dir / f"{ontology_id}.yaml"
            with open(file_path, "w") as f:
                f.write(ontology["yaml_content"])

            logger.info(
                f"Created file backup for ontology {ontology_id} at {file_path}"
            )
            return str(file_path)

        except Exception as e:
            logger.error(f"Error creating file backup: {str(e)}")
            return None


# Create global instance
ontology_storage_service = OntologyStorageService()
