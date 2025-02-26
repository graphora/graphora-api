from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class StorageStage(str, Enum):
    """Storage process stages"""
    NODES = "nodes"
    RELATIONSHIPS = "relationships"
    COMPLETED = "completed"
    FAILED = "failed"

class Node(BaseModel):
    """Model representing a graph node"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any]

class Edge(BaseModel):
    """Model representing a graph relationship"""
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

class StorageCheckpoint(BaseModel):
    """Storage process checkpoint"""
    transform_id: str
    last_processed_index: int
    stage: StorageStage
    timestamp: datetime
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StorageBatchResult(BaseModel):
    """Result of a single batch storage operation"""
    batch_index: int
    items_processed: int
    processing_time_ms: float
    success: bool
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

class StorageMetrics(BaseModel):
    """Metrics for storage operations"""
    nodes_processed: int = 0
    relationships_processed: int = 0
    storage_time_ms: float = 0.0
    retries: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    batch_timings: List[float] = Field(default_factory=list)
    avg_batch_time: float = 0.0
    success_rate: float = 0.0
    peak_memory_mb: float = 0.0
    checkpoint_count: int = 0
    
    def add_batch_timing(self, timing_ms: float):
        """Add batch timing and update average"""
        self.batch_timings.append(timing_ms)
        self.avg_batch_time = sum(self.batch_timings) / len(self.batch_timings)
    
    def add_error(self, error: str, batch_index: int, stage: StorageStage):
        """Add error with context"""
        self.errors.append({
            "error": str(error),
            "batch_index": batch_index,
            "stage": stage,
            "timestamp": datetime.now().isoformat()
        })
    
    def update_success_rate(self):
        """Update success rate based on errors"""
        total_operations = (
            self.nodes_processed + self.relationships_processed
        )
        if total_operations > 0:
            self.success_rate = 1 - (len(self.errors) / total_operations)

class TransformationResult(BaseModel):
    """Result of retrieving transformation data"""
    transform_id: str
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    timestamp: datetime
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def relationship_count(self) -> int:
        return len(self.relationships)
    
    @property
    def node_types(self) -> List[str]:
        return list(set(node['type'] for node in self.nodes))
    
    @property
    def relationship_types(self) -> List[str]:
        return list(set(
            rel['relationship_type'] for rel in self.relationships
        ))
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type"""
        return [
            node for node in self.nodes
            if node['type'] == node_type
        ]
    
    def get_relationships_by_type(
        self,
        relationship_type: str
    ) -> List[Dict[str, Any]]:
        """Get all relationships of a specific type"""
        return [
            rel for rel in self.relationships
            if rel['relationship_type'] == relationship_type
        ]

class StorageResult(BaseModel):
    """Final result of storage operation"""
    transform_id: str
    nodes_stored: int
    relationships_stored: int
    start_time: datetime
    end_time: Optional[datetime] = None
    status: StorageStage
    metrics: StorageMetrics
    checkpoints: List[StorageCheckpoint] = Field(default_factory=list)
    
    def finalize(self, status: StorageStage = StorageStage.COMPLETED):
        """Finalize storage result"""
        self.end_time = datetime.now()
        self.status = status
        self.metrics.update_success_rate()

class StorageError(Exception):
    """Base exception for storage errors"""
    pass

class StorageConnectionError(StorageError):
    """Error in establishing database connection"""
    pass

class CheckpointError(StorageError):
    """Error in checkpoint operations"""
    pass

class DatabaseError(StorageError):
    """Error in database operations"""
    pass
