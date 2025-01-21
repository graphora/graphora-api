from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class JobStatus(BaseModel):
    status: Literal['processing', 'completed', 'failed']
    progress: float = Field(
        default=0.0,
        description="Progress percentage from 0 to 100",
        ge=0.0,
        le=100.0
    )
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

class JobStatusResponse(BaseModel):
    """API response model for job status endpoint"""
    id: str
    status: Literal['processing', 'completed', 'failed']
    progress: float = Field(
        default=0.0,
        description="Progress percentage from 0 to 100",
        ge=0.0,
        le=100.0
    )
