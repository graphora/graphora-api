from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

from app.schemas.graph import GraphResponse

class MergeEventType(str, Enum):
    PROGRESS = "PROGRESS"
    QUESTION = "QUESTION"
    ERROR = "ERROR"

class MergeOption(BaseModel):
    """Option for a merge question"""
    id: str = Field(..., description="Unique identifier for this option")
    label: str = Field(..., description="Display label for the option")
    description: Optional[str] = Field(None, description="Optional detailed description")

class MergeQuestionData(BaseModel):
    """Data for a merge question event"""
    questionId: str = Field(..., description="Unique identifier for this question")
    content: str = Field(..., description="Question text/content")
    options: List[MergeOption] = Field(..., description="Available options for this question")
    previewGraphData: Optional[GraphResponse] = Field(None, description="Optional preview of changes")

class MergeProgressData(BaseModel):
    """Data for a merge progress event"""
    progress: float = Field(..., description="Progress percentage (0-100)", ge=0, le=100)
    currentStep: str = Field(..., description="Current step description")
    graphData: Optional[GraphResponse] = Field(None, description="Optional intermediate graph state")

class MergeEvent(BaseModel):
    """Base model for merge events"""
    type: MergeEventType
    payload: Dict[str, Any] = Field(..., description="Event payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MergeAnswer(BaseModel):
    """Model for merge question answers"""
    questionId: str = Field(..., description="ID of the question being answered")
    optionId: str = Field(..., description="ID of the selected option")
