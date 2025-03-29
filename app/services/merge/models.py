"""Models for merge operations"""
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import uuid
from app.schemas.graph import Node, Edge

class MergeStatus(str, Enum):
    """Status of a merge operation"""
    STARTED = "started"
    AUTO_RESOLVE = "auto_resolve"
    HUMAN_REVIEW = "human_review"
    READY_TO_MERGE = "ready_to_merge"
    MERGE_IN_PROGRESS = "merge_in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MergeInitResponse(BaseModel):
    """Response for merge initialization"""
    merge_id: str
    status: MergeStatus
    start_time: datetime

class MatchStrategy(str, Enum):
    """Available strategies for entity matching"""
    EXACT_NAME = "exact_name"
    PROPERTY_SIMILARITY = "property_similarity"

class MergeInitResponse(BaseModel):
    """Response for merge initialization"""
    merge_id: str
    status: MergeStatus
    start_time: datetime

class GraphResponse(BaseModel):
    """Response model for graph extraction"""
    nodes: List[Node]
    edges: List[Edge]
    total_nodes: int
    total_edges: int
    extraction_time_ms: Optional[float] = None

class EntityMatch(BaseModel):
    """Model for entity matching results"""
    staging_id: str
    production_matches: List[Node]
    best_match: Optional[Node] = None
    match_confidence: float
    match_strategy: str
    metadata: Dict[str, Any] = {}

class EntityMappingResult(BaseModel):
    """Result of entity mapping process"""
    matches: Dict[str, EntityMatch]
    total_entities: int
    matched_entities: int
    mapping_time_ms: float
    metadata: Dict[str, Any] = {}

class ChangeLog(BaseModel):
    """Model for change logs"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prop_changes: Dict[str, Tuple[Any, Any]]
    staging_node: Node
    prod_node: Node
