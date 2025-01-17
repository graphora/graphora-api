from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
import secrets
import base64
import hashlib

class OntologyCache:
    def __init__(self):
        self._cache: Dict[str, tuple[dict, datetime]] = {}
        self._ttl = timedelta(hours=3)  # Cache entries expire after 3 hour
    
    def store(self, session_id: str, ontology: dict):
        """Store ontology with timestamp"""
        self._cache[session_id] = (ontology, datetime.now())
        self._cleanup()
    
    def get(self, session_id: str) -> Optional[dict]:
        """Retrieve ontology if exists and not expired"""
        if session_id not in self._cache:
            return None
        
        ontology, timestamp = self._cache[session_id]
        if datetime.now() - timestamp > self._ttl:
            del self._cache[session_id]
            return None
            
        return ontology
    
    def _cleanup(self):
        """Remove expired entries"""
        current_time = datetime.now()
        expired = [
            k for k, (_, t) in self._cache.items() 
            if current_time - t > self._ttl
        ]
        for k in expired:
            del self._cache[k]

def generate_session_id(method: str = "uuid") -> str:
    """
    Generate a unique session ID using various methods.
    
    Args:
        method: The method to use for generation
               - "uuid": UUID4-based (default)
               - "timestamp": Timestamp-based
               - "secure": Cryptographically secure
               - "short": Shorter, URL-friendly
    
    Returns:
        str: A unique session identifier
    """
    
    match method:
        case "uuid":
            # UUID4-based (default) - 32 characters
            # Example: "550e8400-e29b-41d4-a716-446655440000"
            return str(uuid.uuid4())
            
        case "timestamp":
            # Timestamp + random suffix - 24 characters
            # Example: "20241219_1234567_abcdef"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = secrets.token_hex(3)  # 6 characters
            return f"{timestamp}_{random_suffix}"
            
        case "secure":
            # Cryptographically secure - 64 characters
            # Example: "a1b2c3d4e5f6..."
            random_bytes = secrets.token_bytes(32)
            return hashlib.sha256(random_bytes).hexdigest()
            
        case "short":
            # Short, URL-friendly - 16 characters
            # Example: "xK7m_p9q-r5s_t2w"
            # Uses base64 but replaces + and / with _ and - for URL safety
            random_bytes = secrets.token_bytes(12)
            b64 = base64.urlsafe_b64encode(random_bytes).decode('utf-8')
            return b64.rstrip('=')  # Remove padding characters
            
        case _:
            raise ValueError(f"Unknown session ID generation method: {method}")