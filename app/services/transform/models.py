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
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_nodes: int = 0
    new_nodes: int = 0
    merged_nodes: int = 0
    total_relationships: int = 0
    total_tokens: int = 0
    chunk_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    def track_extraction(
        self,
        chunk_id: str,
        duration_ms: float,
        llm_token_usage: Dict[str, int],
        entity_count: int
    ):
        """Track metrics for chunk extraction"""
        self.processed_chunks += 1
        self.total_tokens += llm_token_usage.get('total', 0)
        self.chunk_metrics[chunk_id] = {
            'duration_ms': duration_ms,
            'token_usage': llm_token_usage,
            'entity_count': entity_count
        }
    
    def record_failure(self, chunk_id: str, error: str):
        """Record chunk extraction failure"""
        self.failed_chunks += 1
        self.chunk_metrics[chunk_id] = {
            'error': error,
            'status': 'failed'
        }
    
    def record_node_stats(self, new_nodes: int, merged_nodes: int, total_relationships: int):
        """Record node and relationship statistics"""
        self.new_nodes = new_nodes
        self.merged_nodes = merged_nodes
        self.total_nodes = new_nodes + merged_nodes
        self.total_relationships = total_relationships

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
            self.metrics.record_node_stats(0, 1, 0)
        else:
            self.nodes[node.id] = node
            self.metrics.record_node_stats(1, 0, 0)
    
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

class OntologyBasedExtractionModels:
    class PropertyType(str, Enum):
        """Types of properties in ontology"""
        STRING = "string"
        INTEGER = "integer"
        FLOAT = "float"
        DATETIME = "datetime"
        BOOLEAN = "boolean"
        LIST = "list"
        OBJECT = "object"
        REFERENCE = "reference"

    class PropertyDefinition(BaseModel):
        """Definition of a property in ontology"""
        type: PropertyType
        description: Optional[str] = None
        required: bool = False
        items_type: Optional[PropertyType] = None  # For list properties
        reference_to: Optional[str] = None  # For reference properties
        default: Optional[Any] = None

    class EntityDefinition(BaseModel):
        """Definition of an entity in ontology"""
        description: str
        properties: Dict[str, PropertyDefinition]
        relationships: Optional[Dict[str, List[str]]] = None

    class OntologyDefinition(BaseModel):
        """Complete ontology definition"""
        version: str
        entities: Dict[str, EntityDefinition]
        
    class NodeProvenance(BaseModel):
        """Provenance information for a node"""
        chunk_ids: List[str] = Field(default_factory=list)
        confidence: float = 0.0
        extraction_time: datetime = Field(default_factory=datetime.utcnow)
        
    class BaseNode(BaseModel):
        """Base class for all nodes"""
        id: str
        type: str
        provenance: NodeProvenance = Field(default_factory=NodeProvenance)

    class RelationshipInstance(BaseModel):
        """Instance of a relationship between nodes"""
        source_id: str
        target_id: str
        type: str
        properties: Dict[str, Any] = Field(default_factory=dict)
        provenance: NodeProvenance = Field(default_factory=NodeProvenance)

    class ExtractionMetrics(BaseModel):
        """Metrics for extraction process"""
        total_chunks: int = 0
        successful_chunks: int = 0
        failed_chunks: Dict[str, str] = Field(default_factory=dict)
        total_nodes: int = 0
        total_relationships: int = 0
        extraction_time_ms: float = 0.0
        
        def add_failed_chunk(self, chunk_id: str, error: str):
            """Add failed chunk with error"""
            self.failed_chunks[chunk_id] = error
            
        def track_extraction_time(self, duration_ms: float):
            """Track extraction time"""
            self.extraction_time_ms += duration_ms

    class KnowledgeGraph(BaseModel):
        """Knowledge graph with nodes and relationships"""
        nodes: Dict[str, BaseNode] = Field(default_factory=dict)
        relationships: List[RelationshipInstance] = Field(default_factory=list)
        metrics: ExtractionMetrics = Field(default_factory=ExtractionMetrics)
        
        def add_node(self, node: BaseNode):
            """Add node to graph"""
            self.nodes[node.id] = node
            self.metrics.total_nodes += 1
            
        def add_relationship(self, rel: RelationshipInstance):
            """Add relationship to graph"""
            self.relationships.append(rel)
            self.metrics.total_relationships += 1
            
        def merge(self, other: 'KnowledgeGraph'):
            """Merge another graph into this one"""
            for node in other.nodes.values():
                self.add_node(node)
                
            for rel in other.relationships:
                self.add_relationship(rel)
                
            # Merge metrics
            self.metrics.total_chunks += other.metrics.total_chunks
            self.metrics.successful_chunks += other.metrics.successful_chunks
            self.metrics.failed_chunks.update(other.metrics.failed_chunks)
            self.metrics.extraction_time_ms += other.metrics.extraction_time_ms
            
        def finalize(self):
            """Finalize graph after construction"""
            # Additional finalization steps can be added here
            pass
