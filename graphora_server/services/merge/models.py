"""Models for merge operations"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import uuid
from graphora_server.schemas.graph import Node, Edge
from graphora_server.baml_client.types import ResolutionStrategy


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
    NOT_FOUND = "not_found"


class MergeInitResponse(BaseModel):
    """Response for merge initialization"""

    merge_id: str
    status: MergeStatus
    start_time: datetime


class MatchStrategy(str, Enum):
    """Available strategies for entity matching"""

    EXACT_NAME = "exact_name"
    UNIQUE_PROPERTY = "unique_property"
    PROPERTY_SIMILARITY = "property_similarity"


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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    need_human_review: bool = False
    match_confidence: Optional[float] = None
    match_strategy: Optional[str] = None

    def to_record(
        self,
        merge_id: str,
        *,
        need_human_review: Optional[bool] = None,
    ) -> "ChangeLogRecord":
        """Serialise the in-memory change log into a persistence record."""

        review_flag = (
            self.need_human_review if need_human_review is None else need_human_review
        )
        if need_human_review is not None:
            self.need_human_review = review_flag

        previous_props = self.prod_node.properties
        changed_props = {prop: values[0] for prop, values in self.prop_changes.items()}

        return ChangeLogRecord(
            id=self.id,
            merge_id=merge_id,
            node_id=self.staging_node.id,
            prod_node_id=self.prod_node.id,
            node_type=self.staging_node.type,
            previous_props=previous_props,
            changed_props=changed_props,
            need_human_review=review_flag,
            created_at=self.created_at,
            match_confidence=self.match_confidence,
            match_strategy=self.match_strategy,
        )


class ChangeLogRecord(BaseModel):
    """Serializable representation of a change log for persistence."""

    id: str
    merge_id: str
    node_id: str
    prod_node_id: str
    node_type: str
    previous_props: Dict[str, Any]
    changed_props: Dict[str, Any]
    need_human_review: bool
    created_at: Optional[datetime] = None
    match_confidence: Optional[float] = None
    match_strategy: Optional[str] = None

    @classmethod
    def from_row(cls, payload: Dict[str, Any]) -> "ChangeLogRecord":
        """Create a record instance from a database row."""

        return cls.model_validate(payload)

    def to_row(self) -> Dict[str, Any]:
        """Return a JSON-serialisable payload for database inserts/updates."""

        return self.model_dump(
            mode="json",
            exclude_none=True,
        )


class ChangeLogResolution(BaseModel):
    """Structured conflict resolution record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    merge_id: str
    ontology_id: str
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    previous_props: Dict[str, Any]
    changed_props: Dict[str, Any]
    resolved_props: Dict[str, Any]
    resolution: ResolutionStrategy
    learning_comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_row(cls, payload: Dict[str, Any]) -> "ChangeLogResolution":
        return cls.model_validate(payload)

    def to_row(self) -> Dict[str, Any]:
        # Include fields that exist in the resolutions table schema
        # Note: merge_id and node_id are not in the database schema
        return self.model_dump(
            mode="json",
            include={
                "id",
                "ontology_id",
                "node_type",
                "previous_props",
                "changed_props",
                "resolved_props",
                "resolution",
                "learning_comment",
                "created_at",
            },
        )


class MergePerformanceMetrics(BaseModel):
    """Performance telemetry collected during a merge run."""

    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    node_batch_durations_ms: List[float] = Field(default_factory=list)
    relationship_batch_durations_ms: List[float] = Field(default_factory=list)
    node_batches: int = 0
    relationship_batches: int = 0
    nodes_processed: int = 0
    relationships_processed: int = 0
    conflicts_detected: int = 0
    staging_nodes: int = 0
    staging_relationships: int = 0
    total_duration_ms: Optional[float] = None

    def record_stage(self, stage: str, duration_ms: float) -> None:
        self.stage_timings_ms[stage] = duration_ms

    def record_node_batch(self, duration_ms: float, items: int) -> None:
        self.node_batches += 1
        self.nodes_processed += items
        self.node_batch_durations_ms.append(duration_ms)

    def record_relationship_batch(self, duration_ms: float, items: int) -> None:
        self.relationship_batches += 1
        self.relationships_processed += items
        self.relationship_batch_durations_ms.append(duration_ms)
