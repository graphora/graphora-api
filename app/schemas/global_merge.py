from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Set
from enum import Enum
from datetime import datetime

class DbNode(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]

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
    RESOLVED = "resolved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"

class ConflictType(str, Enum):
    MULTIPLE_MATCHES = "multiple_matches"
    LOW_CONFIDENCE = "low_confidence"
    PROPERTY_CONFLICT = "property_conflict"
    RELATIONSHIP_CONFLICT = "relationship_conflict"

class ConflictResolutionSuggestion(BaseModel):
    suggestion_type: str  # e.g., "merge", "create_new", "update_existing"
    description: str
    confidence: float
    affected_properties: List[str] = []

class NodeConflict(BaseModel):
    conflict_type: ConflictType
    staging_node_id: str
    prod_node_ids: List[str]
    description: str
    suggestions: List[ConflictResolutionSuggestion]
    properties_affected: Dict[str, Dict[str, Any]] = {}  # property -> {staging: val, prod: val}

class NodeResolutionResult(BaseModel):
    status: ResolutionStatus
    staging_node: DbNode
    prod_node_id: Optional[str]
    confidence: float
    issues: List[str] = []
    conflicts: List[NodeConflict] = []
    suggested_resolution: Optional[ConflictResolutionSuggestion]

    model_config = {
        "arbitrary_types_allowed": True
    }

class ERState(BaseModel):
    """State for the Entity Resolution pipeline"""
    staging_nodes: List[DbNode]
    processed_nodes: List[NodeResolutionResult] = []
    review_queue: List[ReviewItem] = Field(default_factory=list)
    errors: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Track changes for visualization
    new_nodes: List[DbNode] = []
    updated_nodes: List[Dict] = []  # Contains both staging and prod versions
    conflicts: List[NodeConflict] = []
    
    def add_conflict(self, conflict: NodeConflict):
        """Add a conflict and update the node's status"""
        self.conflicts.append(conflict)
        # Update the corresponding node's status
        for node in self.processed_nodes:
            if node.staging_node.id == conflict.staging_node_id:
                node.status = ResolutionStatus.NEEDS_REVIEW
                node.conflicts.append(conflict)
    
    def get_visualization_data(self) -> Dict:
        """Get data formatted for graph visualization"""
        nodes = []
        edges = []
        
        # Add all nodes
        for result in self.processed_nodes:
            node_data = {
                "id": result.staging_node.id,
                "labels": result.staging_node.labels,
                "properties": result.staging_node.properties,
                "status": result.status,
                "type": "staging"
            }
            
            if result.conflicts:
                node_data["conflicts"] = [
                    {
                        "type": c.conflict_type,
                        "description": c.description,
                        "suggestions": [s.dict() for s in c.suggestions]
                    }
                    for c in result.conflicts
                ]
            
            nodes.append(node_data)
            
            # Add production nodes if there are conflicts
            if result.prod_node_id:
                nodes.append({
                    "id": result.prod_node_id,
                    "type": "production",
                    "status": "existing"
                })
                
                # Add edge to show the relationship
                edges.append({
                    "source": result.staging_node.id,
                    "target": result.prod_node_id,
                    "type": "potential_match",
                    "confidence": result.confidence
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "conflicts": [conflict.dict() for conflict in self.conflicts]
        }

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