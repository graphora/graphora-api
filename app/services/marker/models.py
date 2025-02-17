from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class ConversionMetadata(BaseModel):
    pages: int
    conversion_time: float
    file_size: int
    conversion_timestamp: datetime
    settings: Dict[str, Any]

class MarkerResponse(BaseModel):
    status: str
    markdown_content: List[str]  # One entry per page if paginated
    conversion_metadata: ConversionMetadata

class ConversionResult(BaseModel):
    transform_id: str
    original_path: str
    markdown_paths: List[str]
    metadata: ConversionMetadata
    status: str
    error: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    latency: float
    error_rate: float
    last_check: datetime
