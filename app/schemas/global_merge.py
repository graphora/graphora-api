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
        node_map = {}  # Track processed nodes
        
        # Add staging nodes
        for node in self.staging_nodes:
            # Find resolution result for this node
            resolution = next((r for r in self.processed_nodes if r.staging_node.id == node.id), None)
            
            # Determine node status
            status = "new"  # Default for staging nodes
            if resolution:
                if resolution.prod_node_id:
                    status = "both"  # Node exists in both staging and prod
                if resolution.status == ResolutionStatus.NEEDS_REVIEW:
                    status = "conflict"
            
            # Add node with required format
            node_data = {
                "id": node.id,
                "labels": node.labels,
                "properties": {
                    **node.properties,
                    "__status": status,
                    "__type": node.labels[0] if node.labels else None
                },
                "status": status
            }
            
            # Add conflicts if any
            if resolution and resolution.conflicts:
                node_data["properties"]["__conflicts"] = [
                    {
                        "type": c.conflict_type,
                        "description": c.description,
                        "properties": c.properties_affected
                    } for c in resolution.conflicts
                ]
            
            nodes.append(node_data)
            node_map[node.id] = True
            
            # Add relationships from node properties
            for prop_name, prop_value in node.properties.items():
                # Skip non-relationship properties
                if prop_name in ['name', 'type', 'labels', 'id', 'properties', 'relationships']:
                    continue
                    
                # Handle both single ID and list of IDs
                if isinstance(prop_value, str):
                    edges.append({
                        "source": node.id,
                        "target": prop_value,
                        "type": f"HAS_{prop_name.upper()}",
                        "properties": {"__status": "new"},
                        "status": "new"
                    })
                elif isinstance(prop_value, list):
                    for target_id in prop_value:
                        if isinstance(target_id, str):
                            edges.append({
                                "source": node.id,
                                "target": target_id,
                                "type": f"HAS_{prop_name.upper()}",
                                "properties": {"__status": "new"},
                                "status": "new"
                            })
        
        # Add production nodes that are involved in resolutions
        for result in self.processed_nodes:
            if result.prod_node_id and result.prod_node_id not in node_map:
                # Get node data from updated_nodes if available
                prod_node = None
                for update in self.updated_nodes:
                    if update.get("prod", {}).get("id") == result.prod_node_id:
                        prod_node = update["prod"]
                        break
                
                if prod_node:
                    nodes.append({
                        "id": prod_node["id"],
                        "labels": prod_node.get("labels", []),
                        "properties": {
                            **prod_node.get("properties", {}),
                            "__status": "unchanged",
                            "__type": prod_node.get("labels", [""])[0]
                        },
                        "status": "unchanged"
                    })
                    node_map[prod_node["id"]] = True
        
        # Add edges between staging and production nodes (resolution edges)
        for result in self.processed_nodes:
            if result.prod_node_id:
                edge_status = "both" if result.status == ResolutionStatus.RESOLVED else "new"
                edges.append({
                    "source": result.staging_node.id,
                    "target": result.prod_node_id,
                    "type": "resolution",
                    "properties": {
                        "__status": edge_status
                    },
                    "status": edge_status
                })
        
        # Add explicit relationships from staging nodes
        for node in self.staging_nodes:
            if "relationships" in node.properties:
                for rel in node.properties["relationships"]:
                    if rel["source_id"] in node_map and rel["target_id"] in node_map:
                        edges.append({
                            "source": rel["source_id"],
                            "target": rel["target_id"],
                            "type": rel["type"],
                            "properties": {
                                **rel.get("properties", {}),
                                "__status": "new"
                            },
                            "status": "new"
                        })
        
        # Add existing relationships from production
        for update in self.updated_nodes:
            if "relationships" in update.get("prod", {}):
                for rel in update["prod"]["relationships"]:
                    if rel["source_id"] in node_map and rel["target_id"] in node_map:
                        edges.append({
                            "source": rel["source_id"],
                            "target": rel["target_id"],
                            "type": rel["type"],
                            "properties": {
                                **rel.get("properties", {}),
                                "__status": "unchanged"
                            },
                            "status": "unchanged"
                        })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "conflicts": [
                {
                    "node_id": result.staging_node.id,
                    "conflicts": [
                        {
                            "type": c.conflict_type,
                            "description": c.description,
                            "properties": c.properties_affected,
                            "suggestions": [s.dict() for s in c.suggestions]
                        } for c in result.conflicts
                    ]
                } for result in self.processed_nodes if result.conflicts
            ]
        }

    def get_visualization_data_from_state(self) -> Dict:
        """Get data for visualizing the merge state"""
        nodes = []
        edges = []
        node_map = {}  # Keep track of node IDs we've added
        
        # Add staging nodes
        for node in self.staging_nodes:
            node_data = {
                "id": node.id,
                "labels": node.labels,
                "properties": node.properties,
                "type": "staging",
                "status": "new"  # Default status
            }
            
            # Update status based on resolution
            for result in self.processed_nodes:
                if result.staging_node.id == node.id:
                    node_data["status"] = result.status.value
                    if result.prod_node_id:
                        # Add edge to production node
                        edges.append({
                            "source": node.id,
                            "target": result.prod_node_id,
                            "type": "resolution",
                            "status": result.status.value
                        })
                    break
            
            nodes.append(node_data)
            node_map[node.id] = True
        
        # Add production nodes that are involved in resolutions
        for result in self.processed_nodes:
            if result.prod_node_id and result.prod_node_id not in node_map:
                # Get node data from updated_nodes if available
                prod_node = None
                for update in self.updated_nodes:
                    if update["prod"]["id"] == result.prod_node_id:
                        prod_node = update["prod"]
                        break
                
                if prod_node:
                    nodes.append({
                        "id": prod_node["id"],
                        "labels": prod_node["labels"],
                        "properties": prod_node["properties"],
                        "type": "production",
                        "status": "existing"
                    })
                    node_map[prod_node["id"]] = True
        
        # Add conflict data
        conflict_data = []
        for conflict in self.conflicts:
            conflict_data.append({
                "id": conflict.staging_node_id,
                "type": conflict.conflict_type.value,
                "description": conflict.description,
                "prod_nodes": conflict.prod_node_ids,
                "properties_affected": conflict.properties_affected,
                "suggestions": [
                    {
                        "type": s.suggestion_type,
                        "description": s.description,
                        "confidence": s.confidence,
                        "properties": s.affected_properties
                    } for s in conflict.suggestions
                ]
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "conflicts": conflict_data
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