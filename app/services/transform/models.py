from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, create_model, validator
from enum import Enum

class PropertyType(str, Enum):
    """Supported property types in ontology"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"
    REFERENCE = "reference"

class PropertyDefinition(BaseModel):
    """Definition of a property in the ontology"""
    type: PropertyType
    required: bool = False
    description: Optional[str] = None
    default: Optional[Any] = None
    items_type: Optional[PropertyType] = None  # For list types
    reference_to: Optional[str] = None  # For reference types
    nested_properties: Optional[Dict[str, 'PropertyDefinition']] = None  # For object types

class EntityDefinition(BaseModel):
    """Definition of an entity in the ontology"""
    name: str
    description: Optional[str] = None
    properties: Dict[str, PropertyDefinition]
    relationships: Optional[Dict[str, 'RelationshipDefinition']] = None

class RelationshipDefinition(BaseModel):
    """Definition of a relationship between entities"""
    target_entity: str
    relationship_type: str
    cardinality: str  # one-to-one, one-to-many, many-to-many
    properties: Optional[Dict[str, PropertyDefinition]] = None

class OntologyDefinition(BaseModel):
    """Complete ontology definition"""
    version: str
    entities: Dict[str, EntityDefinition]
    metadata: Optional[Dict[str, Any]] = None

class ExtractionConfidence(BaseModel):
    """Confidence scores for extracted information"""
    overall_score: float = Field(ge=0.0, le=1.0)
    property_scores: Dict[str, float] = Field(default_factory=dict)
    extraction_method: str
    llm_model: str
    timestamp: datetime

class NodeProvenance(BaseModel):
    """Tracking information about node origins"""
    chunk_ids: List[str] = Field(default_factory=list)
    extraction_confidence: ExtractionConfidence
    last_modified: datetime
    merge_history: List[Dict[str, Any]] = Field(default_factory=list)

class BaseNode(BaseModel):
    """Base class for all generated entity nodes"""
    id: str
    type: str
    provenance: NodeProvenance
    
    class Config:
        arbitrary_types_allowed = True

class RelationshipInstance(BaseModel):
    """Instance of a relationship between nodes"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Optional[Dict[str, Any]] = None
    provenance: NodeProvenance

class ExtractionMetrics(BaseModel):
    """Metrics for the extraction process"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    total_nodes: int = 0
    total_relationships: int = 0
    extraction_times: List[float] = Field(default_factory=list)
    llm_token_usage: Dict[str, int] = Field(default_factory=dict)
    merge_operations: int = 0
    entity_resolution_stats: Dict[str, int] = Field(default_factory=dict)
    peak_memory_mb: float = 0.0
    
    def add_failed_chunk(self, chunk_id: str, error: str):
        """Add a failed chunk to the metrics"""
        self.failed_chunks.append({
            'chunk_id': chunk_id,
            'error': error,
            'timestamp': datetime.now()
        })
    
    def track_extraction_time(self, duration_ms: float):
        """Track extraction time for a chunk"""
        self.extraction_times.append(duration_ms)
    
    def update_token_usage(self, model: str, tokens: int):
        """Update token usage for LLM models"""
        self.llm_token_usage[model] = (
            self.llm_token_usage.get(model, 0) + tokens
        )
    
    def track_merge_operation(self, entity_type: str):
        """Track merge operation for entity type"""
        self.merge_operations += 1
        self.entity_resolution_stats[entity_type] = (
            self.entity_resolution_stats.get(entity_type, 0) + 1
        )

class KnowledgeGraph:
    """Main knowledge graph structure"""
    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.relationships: List[RelationshipInstance] = []
        self.metrics = ExtractionMetrics(start_time=datetime.now())
    
    def add_node(self, node: BaseNode) -> None:
        """Add or merge a node into the graph"""
        if node.id in self.nodes:
            # Implement merge strategy
            existing = self.nodes[node.id]
            self.nodes[node.id] = self._merge_nodes(existing, node)
            self.metrics.track_merge_operation(node.type)
        else:
            self.nodes[node.id] = node
            self.metrics.total_nodes += 1
    
    def add_relationship(self, relationship: RelationshipInstance) -> None:
        """Add a relationship to the graph"""
        # Validate that nodes exist
        if (relationship.source_id in self.nodes and 
            relationship.target_id in self.nodes):
            self.relationships.append(relationship)
            self.metrics.total_relationships += 1
    
    def _merge_nodes(self, existing: BaseNode, new: BaseNode) -> BaseNode:
        """Merge two nodes, preserving history"""
        # Create merge history entry
        merge_entry = {
            'timestamp': datetime.now(),
            'original_id': existing.id,
            'merged_id': new.id,
            'confidence_scores': {
                'original': existing.provenance.extraction_confidence.overall_score,
                'merged': new.provenance.extraction_confidence.overall_score
            }
        }
        
        # Merge provenance
        merged_provenance = NodeProvenance(
            chunk_ids=list(set(
                existing.provenance.chunk_ids + 
                new.provenance.chunk_ids
            )),
            extraction_confidence=(
                new.provenance.extraction_confidence
                if new.provenance.extraction_confidence.overall_score >
                existing.provenance.extraction_confidence.overall_score
                else existing.provenance.extraction_confidence
            ),
            last_modified=datetime.now(),
            merge_history=existing.provenance.merge_history + [merge_entry]
        )
        
        # Create merged node
        merged = existing.model_copy(update={
            'provenance': merged_provenance
        })
        
        return merged

    def get_node_by_id(self, node_id: str) -> Optional[BaseNode]:
        """Retrieve a node by its ID"""
        return self.nodes.get(node_id)
    
    def get_relationships_for_node(
        self,
        node_id: str
    ) -> List[RelationshipInstance]:
        """Get all relationships for a node"""
        return [
            r for r in self.relationships
            if r.source_id == node_id or r.target_id == node_id
        ]
    
    def finalize(self) -> None:
        """Finalize the graph and complete metrics"""
        self.metrics.end_time = datetime.now()
        self.metrics.peak_memory_mb = (
            psutil.Process().memory_info().rss / 1024 / 1024
        )
