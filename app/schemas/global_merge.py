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
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SKIP = "skip"

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
    staging_edges: List[DbEdge] = []
    
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
        """Get data formatted for visualization showing final merge state"""
        nodes = []
        edges = []
        node_map = {}  # Track all nodes by ID
        
        # Process all nodes to determine their final state
        for result in self.processed_nodes:
            staging_node = result.staging_node
            node_type = staging_node.labels[0] if staging_node.labels else None
            
            # Node status based on resolution
            if result.status == ResolutionStatus.CREATE:
                status = "created"
            elif result.status == ResolutionStatus.UPDATE:
                status = "modified"
            elif result.status == ResolutionStatus.DELETE:
                status = "deleted"
            elif result.status == ResolutionStatus.SKIP:
                status = "unchanged"
            else:
                status = "needs_review"
                
            # Create node data
            node_data = {
                "id": staging_node.id,
                "labels": staging_node.labels,
                "properties": {
                    **staging_node.properties,
                    "__status": status,
                    "__type": node_type
                },
                "status": status
            }
            
            # Add conflicts if present
            if result.conflicts:
                node_data["properties"]["__conflicts"] = [
                    {
                        "type": c.conflict_type,
                        "description": c.description,
                        "properties": c.properties_affected,
                        "suggestions": [s.dict() for s in c.suggestions]
                    } for c in result.conflicts
                ]
            
            nodes.append(node_data)
            node_map[staging_node.id] = node_data
            
            # Add production node reference if it exists
            if result.prod_node_id:
                # Find production node
                prod_node = None
                for update in self.updated_nodes:
                    if isinstance(update, dict) and update.get("prod", {}).get("id") == result.prod_node_id:
                        prod_node = update["prod"]
                        break
                    elif isinstance(update, DbNode) and update.id == result.prod_node_id:
                        prod_node = update
                        break
                
                if prod_node and result.prod_node_id not in node_map:
                    if isinstance(prod_node, dict):
                        prod_type = prod_node.get("labels", [None])[0]
                        prod_props = prod_node.get("properties", {})
                    else:
                        prod_type = prod_node.labels[0] if prod_node.labels else None
                        prod_props = prod_node.properties
                    
                    prod_node_data = {
                        "id": result.prod_node_id,
                        "labels": [prod_type] if prod_type else [],
                        "properties": {
                            **(prod_props or {}),
                            "__status": "unchanged",
                            "__type": prod_type
                        },
                        "status": "unchanged"
                    }
                    nodes.append(prod_node_data)
                    node_map[result.prod_node_id] = prod_node_data
        
        # Add edges from staging graph
        for edge in self.staging_edges:
            # Only include edges where both nodes exist
            if edge.source_id in node_map and edge.target_id in node_map:
                # Get source node's status
                source_node = next((r for r in self.processed_nodes if r.staging_node.id == edge.source_id), None)
                edge_status = source_node.status.value if source_node else "needs_review"
                
                edges.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.type,
                    "properties": {
                        **edge.properties,
                        "__status": edge_status
                    },
                    "status": edge_status
                })
        
        # Add property edges for created/modified nodes
        for result in self.processed_nodes:
            if result.status in [ResolutionStatus.CREATE, ResolutionStatus.UPDATE]:
                node = result.staging_node
                for prop_name, prop_value in node.properties.items():
                    if prop_name.startswith('_') or not prop_value:  # Skip internal/empty properties
                        continue
                        
                    if isinstance(prop_value, str):
                        # Create property node
                        prop_node_id = f"{node.id}_{prop_name}"
                        if prop_node_id not in node_map:
                            prop_node = {
                                "id": prop_node_id,
                                "labels": ["Property"],
                                "properties": {
                                    "name": prop_name,
                                    "value": prop_value,
                                    "__status": result.status.value,
                                    "__type": "Property"
                                },
                                "status": result.status.value
                            }
                            nodes.append(prop_node)
                            node_map[prop_node_id] = prop_node
                        
                        # Create edge to property node
                        edges.append({
                            "source": node.id,
                            "target": prop_node_id,
                            "type": f"HAS_{prop_name.upper()}",
                            "properties": {"__status": result.status.value},
                            "status": result.status.value
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
                }
                for result in self.processed_nodes 
                if result.conflicts
            ]
        }

    def get_visualization_data_from_state(self) -> Dict:
        """Get data for visualizing the merge state"""
        nodes = []
        edges = []
        node_map = {}  # Keep track of node IDs we've added
        
        # Add staging nodes
        for node in self.staging_nodes:
            node_type = node.labels[0] if node.labels else None
            node_data = {
                "id": node.id,
                "labels": [node_type] if node_type else [],
                "properties": node.properties,
                "type": "staging",
                "status": "new"  # Default status
            }
            
            # Update status based on resolution
            for result in self.processed_nodes:
                if result.staging_node.id == node.id:
                    node_data["status"] = result.status.value
                    node_data["properties"]["__status"] = result.status.value
                    
                    # Add conflicts if present
                    if result.conflicts:
                        node_data["properties"]["__conflicts"] = [
                            {
                                "type": c.conflict_type,
                                "description": c.description,
                                "properties": c.properties_affected,
                                "suggestions": [s.dict() for s in c.suggestions]
                            } for c in result.conflicts
                        ]
            
            nodes.append(node_data)
            node_map[node.id] = True
        
        # Add production nodes that are involved in resolutions
        for result in self.processed_nodes:
            if result.prod_node_id and result.prod_node_id not in node_map:
                prod_node = next((n for n in self.updated_nodes if n["id"] == result.prod_node_id), None)
                if prod_node:
                    nodes.append({
                        "id": prod_node["id"],
                        "labels": [prod_node["labels"][0]] if prod_node["labels"] else [],
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