from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class DbNode(BaseModel):
    id: str
    labels: List[str]
    properties: Dict

    model_config = {
        "arbitrary_types_allowed": True
    }

class DbEdge(BaseModel):
    id: str
    type: str
    properties: Dict
    source_id: str
    target_id: str

    model_config = {
        "arbitrary_types_allowed": True
    }

class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

class ReviewItem(BaseModel):
    """Model for review queue items"""
    id: str  # Review item ID
    staging_node: DbNode
    prod_node_id: Optional[str]
    issues: List[str]
    status: ReviewStatus
    confidence: float
    changes: List[Dict]
    content: str
    assigned_to: Optional[str] = None
    reviewer_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {
        "arbitrary_types_allowed": True
    }

class ResolutionStatus(str, Enum):
    NEW = "new"
    PENDING = "pending"
    RESOLVED = "resolved"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"

class NodeResolutionResult(BaseModel):
    status: ResolutionStatus
    staging_node: DbNode
    prod_node_id: Optional[str]
    confidence: float
    issues: List[str]
    review_notes: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True
    }
    
class ERState(BaseModel):
    """State object for the ER pipeline"""
    staging_nodes: List[DbNode]
    processed_nodes: List[NodeResolutionResult] = Field(default_factory=list)
    review_queue: List[ReviewItem] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True
    }

    @field_validator("processed_nodes", mode="before")
    @classmethod
    def validate_processed_nodes(cls, v):
        if isinstance(v, list):
            return [item.model_dump() if isinstance(item, NodeResolutionResult) else item for item in v]
        return v

    @field_validator("review_queue", mode="before")
    @classmethod
    def validate_review_queue(cls, v):
        if isinstance(v, list):
            return [item.model_dump() if isinstance(item, ReviewItem) else item for item in v]
        return v