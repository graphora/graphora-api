"""Service for handling ontology operations"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from graphora_server.config import settings
from graphora_server.db import postgres as db
from graphora_server.services.cache import get_ontology_cache

logger = logging.getLogger(__name__)


async def load_ontology(
    ontology_id: str, user_id: Optional[str] = None, use_cache: bool = True
) -> Dict[str, Any]:
    """Load ontology definition from file or database with fallback

    Args:
        ontology_id: ID of the ontology to load
        user_id: User ID for database fallback (optional)
        use_cache: Whether to use caching (default True)

    Returns:
        Dictionary containing the ontology definition

    Raises:
        ValueError: If ontology not found in file or database
    """
    cache = get_ontology_cache()

    # Check cache first
    if use_cache:
        cached = await cache.get(ontology_id, user_id)
        if cached is not None:
            logger.debug(f"Ontology '{ontology_id}' loaded from cache")
            return cached

    try:
        # First try to load from local file
        ontology_path = Path(settings.ontology_dir).expanduser() / f"{ontology_id}.yaml"

        if ontology_path.exists():
            try:
                with open(ontology_path, "r") as f:
                    content = f.read()
                    # Validate that it's valid YAML content
                    ontology_data = yaml.safe_load(content)
                    logger.info(
                        f"Successfully loaded ontology '{ontology_id}' from local file"
                    )
                    # Cache the loaded ontology
                    if use_cache:
                        await cache.set(ontology_id, ontology_data, user_id)
                    return ontology_data
            except (IOError, yaml.YAMLError) as e:
                logger.warning(
                    f"Failed to read local ontology file {ontology_path}: {e}"
                )
                # Continue to database fallback if local file is corrupted

        # If file doesn't exist or is corrupted and we have user_id, try database
        if user_id:
            logger.info(
                f"Attempting to load ontology '{ontology_id}' from database for user {user_id}"
            )
            db_content = await _load_from_database(ontology_id, user_id)
            if db_content:
                logger.info(
                    f"Successfully loaded ontology '{ontology_id}' from database"
                )
                ontology_data = yaml.safe_load(db_content)
                # Cache the loaded ontology
                if use_cache:
                    await cache.set(ontology_id, ontology_data, user_id)
                return ontology_data
            else:
                logger.warning(
                    f"Ontology '{ontology_id}' not found in database for user {user_id}"
                )

        # If all else fails, provide helpful error message
        error_msg = f"Ontology '{ontology_id}' not found"
        if ontology_path.exists():
            error_msg += " (local file exists but is corrupted)"
        else:
            error_msg += " (local file not found)"

        if user_id:
            error_msg += f" and not found in database for user {user_id}"
        else:
            error_msg += " and no user_id provided for database fallback"

        raise ValueError(error_msg)

    except Exception as e:
        if isinstance(e, ValueError):
            raise  # Re-raise ValueError as-is
        logger.error(f"Failed to load ontology '{ontology_id}': {str(e)}")
        raise ValueError(f"Failed to load ontology '{ontology_id}': {str(e)}")


async def _load_from_database(ontology_id: str, user_id: str) -> Optional[str]:
    """Load ontology content from database"""
    try:
        record = await db.fetchrow(
            """
            SELECT yaml_content
            FROM ontologies
            WHERE id = %s AND user_id = %s AND is_active = TRUE
            """,
            ontology_id,
            user_id,
        )

        if record and record.get("yaml_content"):
            yaml_content = record["yaml_content"]
            # Validate that the retrieved content is valid YAML
            try:
                yaml.safe_load(yaml_content)
                return yaml_content
            except yaml.YAMLError as e:
                logger.warning(
                    f"Ontology '{ontology_id}' from database contains invalid YAML: {e}"
                )
                return None
        else:
            logger.info(
                f"No active ontology found with id '{ontology_id}' for user '{user_id}'"
            )
            return None

    except Exception as e:
        logger.error(f"Error loading ontology '{ontology_id}' from database: {e}")
        return None


async def invalidate_ontology_cache(
    ontology_id: str, user_id: Optional[str] = None
) -> bool:
    """Invalidate cached ontology when it's updated.

    Args:
        ontology_id: ID of the ontology to invalidate
        user_id: User ID for user-specific ontologies

    Returns:
        True if cache entry was removed, False if not found
    """
    cache = get_ontology_cache()
    return await cache.invalidate(ontology_id, user_id)
