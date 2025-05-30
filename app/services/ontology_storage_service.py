"""Ontology Storage Service for storing ontologies in Supabase"""
import logging
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger(__name__)

class OntologyStorageService:
    """Service for storing and retrieving ontologies from Supabase"""
    
    def __init__(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")
        
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
    
    async def store_ontology(
        self,
        user_id: str,
        ontology_id: str,
        yaml_content: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """Store ontology in Supabase
        
        Args:
            user_id: User ID
            ontology_id: Unique ontology ID
            yaml_content: YAML content of the ontology
            name: Optional name for the ontology
            description: Optional description
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Parse YAML to extract version and metadata
            ontology_data = yaml.safe_load(yaml_content)
            yaml_version = ontology_data.get('version', '1.0.0')
            
            # Extract name from ontology if not provided
            if not name:
                name = f"Ontology {ontology_id[:8]}"
            
            # Check if ontology already exists for this user
            existing = self.supabase.table("ontologies").select("id, version").eq("id", ontology_id).eq("user_id", user_id).execute()
            
            # Determine which version to use
            if existing.data and len(existing.data) > 0:
                # For existing ontologies, preserve the current version instead of using YAML version
                # This prevents version increments when editing YAML content
                version = existing.data[0].get('version', yaml_version)
            else:
                # For new ontologies, use the version from YAML
                version = yaml_version
            
            ontology_record = {
                "id": ontology_id,
                "user_id": user_id,
                "name": name,
                "file_name": f"{ontology_id}.yaml",
                "description": description,
                "yaml_content": yaml_content,
                "version": version,  # Use preserved or YAML version as determined above
                "is_active": True,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if existing.data and len(existing.data) > 0:
                # Update existing ontology
                result = self.supabase.table("ontologies").update(ontology_record).eq("id", ontology_id).eq("user_id", user_id).execute()
            else:
                # Insert new ontology
                ontology_record["created_at"] = datetime.utcnow().isoformat()
                result = self.supabase.table("ontologies").insert(ontology_record).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully stored ontology {ontology_id} for user {user_id}")
                return True
            else:
                logger.error(f"Failed to store ontology {ontology_id} for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing ontology: {str(e)}")
            return False
    
    async def get_ontology(self, user_id: str, ontology_id: str) -> Optional[Dict[str, Any]]:
        """Get ontology by ID for a specific user
        
        Args:
            user_id: User ID
            ontology_id: Ontology ID
            
        Returns:
            Dict containing ontology data or None if not found
        """
        try:
            result = self.supabase.table("ontologies").select("*").eq("id", ontology_id).eq("user_id", user_id).eq("is_active", True).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
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
            result = self.supabase.table("ontologies").select(
                "id, name, file_name, description, version, created_at, updated_at"
            ).eq("user_id", user_id).eq("is_active", True).order("updated_at", desc=True).execute()
            
            return result.data or []
            
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
            result = self.supabase.table("ontologies").update({
                "is_active": False,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", ontology_id).eq("user_id", user_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully deleted ontology {ontology_id} for user {user_id}")
                return True
            else:
                logger.error(f"Failed to delete ontology {ontology_id} for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting ontology: {str(e)}")
            return False
    
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
            with open(file_path, 'w') as f:
                f.write(ontology['yaml_content'])
            
            logger.info(f"Created file backup for ontology {ontology_id} at {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error creating file backup: {str(e)}")
            return None

# Create global instance
ontology_storage_service = OntologyStorageService() 