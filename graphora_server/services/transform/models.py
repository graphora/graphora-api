from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from pydantic import BaseModel, Field, create_model, ConfigDict
from enum import Enum
import uuid


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
    nested_properties: Optional[Dict[str, "PropertyDefinition"]] = (
        None  # For object types
    )
    model_config = ConfigDict(extra="ignore")


class EntityDefinition(BaseModel):
    """Definition of an entity in the ontology"""

    name: str
    description: Optional[str] = None
    properties: Dict[str, PropertyDefinition]
    relationships: Optional[Dict[str, "RelationshipDefinition"]] = None
    model_config = ConfigDict(extra="ignore")


class RelationshipDefinition(BaseModel):
    """Definition of a relationship between entities"""

    target_entity: str
    relationship_type: str
    cardinality: str  # one-to-one, one-to-many, many-to-many
    properties: Optional[Dict[str, PropertyDefinition]] = None
    model_config = ConfigDict(extra="ignore")


class OntologyDefinition(BaseModel):
    """Complete ontology definition"""

    version: str
    entities: Dict[str, EntityDefinition]
    metadata: Optional[Dict[str, Any]] = None


class NodeProvenance(BaseModel):
    """Information about where a node came from.

    Source-span fields (``source_file``, ``page_number``,
    ``char_offset``) and decision-trail fields (``extractor_model``,
    ``prompt_version``, ``validator_score``) populate the contract
    documented in ``graphora_server/mcp/server.py::_EVIDENCE_KEYS``
    — the same set the Explorer Evidence tab consumes. All Optional
    + None-default so older payloads (cached responses, in-flight
    extractions) stay valid through the schema bump.

    A1-prov added the source-span fields. B0-prov-extend (Gate 4
    entry) adds the decision-trail fields so a user inspecting an
    extracted fact can see *which model + which prompt version*
    produced it, and what the validator scored — the foundation for
    the full Decision Log landing in slice 2.
    """

    chunk_ids: List[str] = Field(default_factory=list)
    extraction_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    confidence_score: Optional[float] = None
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    char_offset: Optional[int] = None
    # B0-prov-extend (Gate 4 entry). All Optional — single-pass
    # extraction leaves validator_score None; non-instrumented
    # callers leave the others None too. Graceful degrade
    # everywhere.
    extractor_model: Optional[str] = None
    prompt_version: Optional[str] = None
    validator_score: Optional[float] = None


class BaseNode(BaseModel):
    """Base class for all nodes in the knowledge graph"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    canonical_properties: Dict[str, Any] = Field(default_factory=dict)
    canonical_key: Optional[str] = None
    canonical_id: Optional[str] = None
    provenance: Optional[NodeProvenance] = None
    confidence_score: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RelationshipInstance(BaseModel):
    """A relationship between two nodes"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[NodeProvenance] = None
    confidence_score: Optional[float] = None


class ExtractionMetrics(BaseModel):
    """Metrics for the extraction process"""

    start_time: datetime
    end_time: Optional[datetime] = None
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    new_nodes: int = 0
    merged_nodes: int = 0
    total_nodes: int = 0
    total_relationships: int = 0
    invalid_relationships: int = 0
    total_tokens: int = 0
    chunk_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def track_extraction(
        self,
        chunk_id: str,
        duration_ms: float,
        llm_token_usage: Dict[str, int],
        entity_count: int,
    ) -> None:
        """Track metrics for a single chunk extraction"""
        self.processed_chunks += 1

        # Track tokens
        total_tokens = llm_token_usage.get("total", 0)
        self.total_tokens += total_tokens

        # Store chunk metrics
        self.chunk_metrics[chunk_id] = {
            "duration_ms": duration_ms,
            "token_usage": llm_token_usage,
            "entity_count": entity_count,
            "success": True,
        }

    def record_failure(self, chunk_id: str, error: str) -> None:
        """Record a chunk processing failure"""
        self.failed_chunks += 1
        self.chunk_metrics[chunk_id] = {"success": False, "error": error}

    def record_node_stats(
        self, new_nodes: int, merged_nodes: int, relationships: int
    ) -> None:
        """Update node and relationship statistics"""
        self.new_nodes = new_nodes
        self.merged_nodes = merged_nodes
        self.total_relationships = relationships

    def finalize(self) -> None:
        """Mark extraction as complete and calculate final stats"""
        self.end_time = datetime.now()


class KnowledgeGraph(BaseModel):
    """Generic Knowledge Graph for storing extracted information"""

    extraction_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    tokens_used: Optional[int] = None
    confidence_score: Optional[float] = None
    metrics: Optional[ExtractionMetrics] = None

    # These will be dynamically populated based on the ontology

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DocumentKnowledgeGraph(KnowledgeGraph):
    """Generic Knowledge Graph for storing extracted information"""

    nodes: List[BaseNode] = Field(default_factory=list)
    relationships: List[RelationshipInstance] = Field(default_factory=list)


class OntologyBasedExtractionModels:
    class PropertyType(str, Enum):
        """Property types supported in the ontology"""

        STRING = "string"
        INTEGER = "integer"
        FLOAT = "float"
        BOOLEAN = "boolean"
        LIST = "list"
        OBJECT = "object"
        REFERENCE = "reference"

    @classmethod
    def create_node_class(cls, entity_def: EntityDefinition) -> Type[BaseModel]:
        """Create a Pydantic model class for an entity"""
        properties = {
            "id": (str, ...),
            "type": (str, entity_def.name),
            "properties": (Dict[str, Any], Field(default_factory=dict)),
            "provenance": (
                Dict[str, Any],
                Field(
                    default_factory=lambda: {
                        "chunk_ids": [],
                        "extraction_timestamp": "",
                        "confidence_score": 0.0,
                    }
                ),
            ),
        }

        model_name = f"{entity_def.name}Node"
        return create_model(
            model_name, **properties, __config__=ConfigDict(extra="ignore")
        )

    @classmethod
    def create_relationship_class(
        cls, rel_def: RelationshipDefinition
    ) -> Type[BaseModel]:
        """Create a Pydantic model class for a relationship"""
        properties = {
            "source_id": (str, ...),
            "target_id": (str, ...),
            "type": (str, rel_def.relationship_type),
            "properties": (Dict[str, Any], Field(default_factory=dict)),
            "provenance": (
                Dict[str, Any],
                Field(
                    default_factory=lambda: {
                        "chunk_ids": [],
                        "extraction_timestamp": "",
                        "confidence_score": 0.0,
                    }
                ),
            ),
        }

        model_name = f"{rel_def.relationship_type}Relationship"
        return create_model(
            model_name, **properties, __config__=ConfigDict(extra="ignore")
        )
