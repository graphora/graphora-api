"""Service for handling ontology operations"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

async def load_ontology(ontology_id: str) -> Dict[str, Any]:
    """Load ontology definition from file
    
    Args:
        ontology_id: ID of the ontology to load
        
    Returns:
        Dictionary containing the ontology definition
        
    Raises:
        ValueError: If ontology file not found or invalid
    """
    try:
        ontology_path = Path(settings.ontology_dir).expanduser() / f"{ontology_id}.yaml"
        if not ontology_path.exists():
            raise ValueError(f"Ontology not found at {ontology_path}")
            
        with open(ontology_path, 'r') as f:
            return yaml.safe_load(f)
            
    except Exception as e:
        logger.error(f"Failed to load ontology: {str(e)}")
        raise 