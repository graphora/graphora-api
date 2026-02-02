"""In-memory implementation of graph storage for local development and demos.

This provides a lightweight alternative to Neo4j that stores graphs in memory,
making it easy to test and demo without database setup.
"""

import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.schemas.graph import Edge, GraphResponse, Node
from app.services.storage.interface import GraphStorageInterface
from app.services.storage.models import (
    StorageBatchResult,
    StorageCheckpoint,
    StorageStage,
)
from app.services.transform.models import BaseNode, RelationshipInstance

logger = logging.getLogger(__name__)


class InMemoryGraphStore:
    """Thread-safe in-memory graph store for a single user/transform."""

    def __init__(self) -> None:
        # Nodes indexed by ID
        self._nodes: Dict[str, Node] = {}
        # Edges indexed by ID
        self._edges: Dict[str, Edge] = {}
        # Indexes for efficient lookups
        self._nodes_by_label: Dict[str, set] = defaultdict(set)
        self._nodes_by_transform: Dict[str, set] = defaultdict(set)
        self._nodes_by_merge: Dict[str, set] = defaultdict(set)
        self._edges_by_transform: Dict[str, set] = defaultdict(set)
        self._edges_by_merge: Dict[str, set] = defaultdict(set)
        self._edges_by_source: Dict[str, set] = defaultdict(set)
        self._edges_by_target: Dict[str, set] = defaultdict(set)
        # Checkpoints
        self._checkpoints: Dict[str, StorageCheckpoint] = {}
        # Full-text indexes (simplified - just track which properties are indexed)
        self._ft_indexes: Dict[str, Dict[str, Any]] = {}

    def add_node(
        self,
        node: Node,
        transform_id: Optional[str] = None,
        merge_id: Optional[str] = None,
    ) -> None:
        """Add a node to the store."""
        self._nodes[node.id] = node
        self._nodes_by_label[node.label].add(node.id)
        if transform_id:
            self._nodes_by_transform[transform_id].add(node.id)
        if merge_id:
            self._nodes_by_merge[merge_id].add(node.id)

    def add_edge(
        self,
        edge: Edge,
        transform_id: Optional[str] = None,
        merge_id: Optional[str] = None,
    ) -> None:
        """Add an edge to the store."""
        self._edges[edge.id] = edge
        self._edges_by_source[edge.source].add(edge.id)
        self._edges_by_target[edge.target].add(edge.id)
        if transform_id:
            self._edges_by_transform[transform_id].add(edge.id)
        if merge_id:
            self._edges_by_merge[merge_id].add(edge.id)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def get_nodes_by_label(self, label: str) -> List[Node]:
        """Get all nodes with a given label."""
        return [self._nodes[nid] for nid in self._nodes_by_label.get(label, set())]

    def get_nodes_by_transform(self, transform_id: str) -> List[Node]:
        """Get all nodes for a transform."""
        return [
            self._nodes[nid]
            for nid in self._nodes_by_transform.get(transform_id, set())
        ]

    def get_edges_by_transform(self, transform_id: str) -> List[Edge]:
        """Get all edges for a transform."""
        return [
            self._edges[eid]
            for eid in self._edges_by_transform.get(transform_id, set())
        ]

    def get_nodes_by_merge(self, merge_id: str) -> List[Node]:
        """Get all nodes for a merge."""
        return [self._nodes[nid] for nid in self._nodes_by_merge.get(merge_id, set())]

    def get_edges_by_merge(self, merge_id: str) -> List[Edge]:
        """Get all edges for a merge."""
        return [self._edges[eid] for eid in self._edges_by_merge.get(merge_id, set())]

    def get_edges_for_node(self, node_id: str) -> List[Edge]:
        """Get all edges connected to a node."""
        edge_ids = self._edges_by_source.get(node_id, set()) | self._edges_by_target.get(
            node_id, set()
        )
        return [self._edges[eid] for eid in edge_ids]

    def clear(self) -> None:
        """Clear all data."""
        self._nodes.clear()
        self._edges.clear()
        self._nodes_by_label.clear()
        self._nodes_by_transform.clear()
        self._nodes_by_merge.clear()
        self._edges_by_transform.clear()
        self._edges_by_merge.clear()
        self._edges_by_source.clear()
        self._edges_by_target.clear()
        self._checkpoints.clear()
        self._ft_indexes.clear()


# Global store for in-memory graphs, keyed by user_id
_global_stores: Dict[str, InMemoryGraphStore] = {}


def get_memory_store(user_id: str) -> InMemoryGraphStore:
    """Get or create an in-memory store for a user."""
    if user_id not in _global_stores:
        _global_stores[user_id] = InMemoryGraphStore()
    return _global_stores[user_id]


def clear_memory_store(user_id: str) -> None:
    """Clear a user's in-memory store."""
    if user_id in _global_stores:
        _global_stores[user_id].clear()


def clear_all_memory_stores() -> None:
    """Clear all in-memory stores."""
    _global_stores.clear()


class InMemoryStorage(GraphStorageInterface):
    """In-memory implementation of graph storage.

    Useful for:
    - Local development without Neo4j
    - Quick demos and testing
    - Prototyping and exploration

    Note: Data is not persisted and will be lost on restart.
    """

    def __init__(self, user_id: str = "default") -> None:
        """Initialize in-memory storage.

        Args:
            user_id: User ID for isolation (each user gets their own store)
        """
        self.user_id = user_id
        self._store = get_memory_store(user_id)

    async def close(self) -> None:
        """No-op for in-memory storage."""
        pass

    async def create_or_replace_ft_index_for_node(
        self, index_name: str, entity_name: str, properties: List[str]
    ) -> None:
        """Create a full-text index (simulated)."""
        self._store._ft_indexes[index_name] = {
            "type": "node",
            "entity": entity_name,
            "properties": properties,
        }
        logger.debug(f"Created in-memory FT index {index_name} for {entity_name}")

    async def create_or_replace_ft_index_for_relationship(
        self,
        index_name: str,
        source_name: str,
        rel_name: str,
        target_name: str,
        properties: List[str],
    ) -> None:
        """Create a full-text index for relationships (simulated)."""
        self._store._ft_indexes[index_name] = {
            "type": "relationship",
            "source": source_name,
            "relationship": rel_name,
            "target": target_name,
            "properties": properties,
        }
        logger.debug(f"Created in-memory FT index {index_name} for {rel_name}")

    async def store_nodes(
        self,
        nodes: List[BaseNode],
        batch_index: int,
        transform_id: str,
        merge_id: Optional[str] = None,
        merge: bool = True,
    ) -> StorageBatchResult:
        """Store nodes in memory."""
        start_time = time.time()
        warnings: List[str] = []

        for base_node in nodes:
            # Convert BaseNode to Node
            node = Node(
                id=base_node.id,
                label=base_node.type,
                type=base_node.type,
                properties={
                    **base_node.properties,
                    **base_node.canonical_properties,
                    "transform_id": transform_id,
                    **({"merge_id": merge_id} if merge_id else {}),
                },
            )

            # Check for existing node if merging
            existing = self._store.get_node(base_node.id)
            if existing and merge:
                # Merge properties
                merged_props = {**existing.properties, **node.properties}
                node = Node(
                    id=existing.id,
                    label=existing.label,
                    type=existing.type,
                    properties=merged_props,
                    updated_at=datetime.now(timezone.utc),
                )

            self._store.add_node(node, transform_id, merge_id)

        processing_time = (time.time() - start_time) * 1000

        return StorageBatchResult(
            batch_index=batch_index,
            items_processed=len(nodes),
            processing_time_ms=processing_time,
            success=True,
            warnings=warnings,
        )

    async def store_relationships(
        self,
        relationships: List[RelationshipInstance],
        batch_index: int,
        transform_id: str,
        merge_id: Optional[str] = None,
        merge: bool = True,
    ) -> StorageBatchResult:
        """Store relationships in memory."""
        start_time = time.time()
        warnings: List[str] = []

        for rel in relationships:
            # Verify source and target nodes exist
            source_node = self._store.get_node(rel.source_id)
            target_node = self._store.get_node(rel.target_id)

            if not source_node:
                warnings.append(f"Source node {rel.source_id} not found for relationship")
                continue
            if not target_node:
                warnings.append(f"Target node {rel.target_id} not found for relationship")
                continue

            edge = Edge(
                id=rel.id,
                source=rel.source_id,
                target=rel.target_id,
                type=rel.type,
                properties={
                    **rel.properties,
                    "transform_id": transform_id,
                    **({"merge_id": merge_id} if merge_id else {}),
                },
            )

            self._store.add_edge(edge, transform_id, merge_id)

        processing_time = (time.time() - start_time) * 1000

        return StorageBatchResult(
            batch_index=batch_index,
            items_processed=len(relationships) - len(warnings),
            processing_time_ms=processing_time,
            success=True,
            warnings=warnings,
        )

    async def get_storage_status(
        self, transform_id: str
    ) -> Optional[StorageCheckpoint]:
        """Get current storage status."""
        return self._store._checkpoints.get(transform_id)

    async def update_checkpoint(
        self, transform_id: str, last_index: int, stage: StorageStage
    ) -> None:
        """Update storage checkpoint."""
        self._store._checkpoints[transform_id] = StorageCheckpoint(
            transform_id=transform_id,
            last_processed_index=last_index,
            stage=stage,
            timestamp=datetime.now(timezone.utc),
        )

    def get_transformation_data(self, transform_id: str) -> GraphResponse:
        """Get all nodes and relationships for a transformation."""
        nodes = self._store.get_nodes_by_transform(transform_id)
        edges = self._store.get_edges_by_transform(transform_id)

        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges),
            metadata={"storage_type": "memory", "transform_id": transform_id},
        )

    def get_merge_data(self, merge_id: str) -> GraphResponse:
        """Get all nodes and relationships for a merge."""
        nodes = self._store.get_nodes_by_merge(merge_id)
        edges = self._store.get_edges_by_merge(merge_id)

        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges),
            metadata={"storage_type": "memory", "merge_id": merge_id},
        )

    async def get_all_node_properties(self, entity_name: str) -> List[str]:
        """Get all properties of a node entity."""
        properties: set = set()
        for node in self._store.get_nodes_by_label(entity_name):
            properties.update(node.properties.keys())
        # Exclude system properties
        system_props = {"transform_id", "merge_id", "valid_from", "valid_to"}
        return [p for p in properties if p not in system_props]

    async def get_all_relationship_properties(self, rel_name: str) -> List[str]:
        """Get all properties of a relationship type."""
        properties: set = set()
        for edge in self._store._edges.values():
            if edge.type == rel_name:
                properties.update(edge.properties.keys())
        system_props = {"transform_id", "merge_id", "valid_from", "valid_to"}
        return [p for p in properties if p not in system_props]

    async def get_nodes_by_property(
        self, property_name: str, property_value: Any
    ) -> List[Node]:
        """Get all nodes with the specified property value."""
        return [
            node
            for node in self._store._nodes.values()
            if node.properties.get(property_name) == property_value
        ]

    async def get_relationships_between(
        self, source_id: str, target_id: str, relationship_type: Optional[str] = None
    ) -> List[Edge]:
        """Get all relationships between two nodes."""
        edges = []
        for edge_id in self._store._edges_by_source.get(source_id, set()):
            edge = self._store._edges.get(edge_id)
            if edge and edge.target == target_id:
                if relationship_type is None or edge.type == relationship_type:
                    edges.append(edge)
        return edges

    async def get_relationships_between_nodes(self, node_ids: List[str]) -> List[Edge]:
        """Get all relationships between a set of nodes."""
        node_set = set(node_ids)
        edges = []
        for edge in self._store._edges.values():
            if edge.source in node_set and edge.target in node_set:
                edges.append(edge)
        return edges

    async def find_nodes_by_property_value(
        self,
        label: str,
        property_name: str,
        property_value: Any,
        exact_match: bool = True,
    ) -> List[Node]:
        """Find nodes with matching property value."""
        results = []
        for node in self._store.get_nodes_by_label(label):
            value = node.properties.get(property_name)
            if exact_match:
                if value == property_value:
                    results.append(node)
            else:
                # Partial match for strings
                if isinstance(value, str) and isinstance(property_value, str):
                    if property_value.lower() in value.lower():
                        results.append(node)
                elif value == property_value:
                    results.append(node)
        return results

    async def find_similar_nodes(
        self,
        label: str,
        properties: Dict[str, Any],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_relationships: bool = True,
    ) -> List[Node]:
        """Find nodes with similar properties using fuzzy matching."""
        candidates = self._store.get_nodes_by_label(label)
        scored_nodes = []

        for node in candidates:
            score = self._calculate_similarity(properties, node.properties)
            if score >= similarity_threshold:
                scored_nodes.append((score, node))

        # Sort by score descending
        scored_nodes.sort(key=lambda x: x[0], reverse=True)

        return [node for _, node in scored_nodes[:max_results]]

    def _calculate_similarity(
        self, props1: Dict[str, Any], props2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two property dictionaries."""
        if not props1 or not props2:
            return 0.0

        common_keys = set(props1.keys()) & set(props2.keys())
        if not common_keys:
            return 0.0

        total_score = 0.0
        for key in common_keys:
            val1, val2 = props1[key], props2[key]
            if isinstance(val1, str) and isinstance(val2, str):
                total_score += SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
            elif val1 == val2:
                total_score += 1.0

        return total_score / len(common_keys)

    async def create_node(self, label: str, properties: Dict[str, Any]) -> Node:
        """Create a new node."""
        node = Node(
            id=str(uuid.uuid4()),
            label=label,
            type=label,
            properties=properties,
        )
        self._store.add_node(node)
        return node

    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> Node:
        """Update an existing node."""
        existing = self._store.get_node(node_id)
        if not existing:
            raise ValueError(f"Node {node_id} not found")

        updated = Node(
            id=existing.id,
            label=existing.label,
            type=existing.type,
            properties={**existing.properties, **properties},
            updated_at=datetime.now(timezone.utc),
        )
        self._store._nodes[node_id] = updated
        return updated

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Edge:
        """Create a relationship between nodes."""
        edge = Edge(
            id=str(uuid.uuid4()),
            source=source_id,
            target=target_id,
            type=rel_type,
            properties=properties or {},
        )
        self._store.add_edge(edge)
        return edge

    async def update_relationship(
        self, rel_id: str, properties: Dict[str, Any]
    ) -> Edge:
        """Update an existing relationship."""
        existing = self._store.get_edge(rel_id)
        if not existing:
            raise ValueError(f"Relationship {rel_id} not found")

        updated = Edge(
            id=existing.id,
            source=existing.source,
            target=existing.target,
            type=existing.type,
            properties={**existing.properties, **properties},
            updated_at=datetime.now(timezone.utc),
        )
        self._store._edges[rel_id] = updated
        return updated

    async def get_relationship(
        self, source_id: str, target_id: str, rel_type: str
    ) -> Optional[Edge]:
        """Get a specific relationship between two nodes by type."""
        edges = await self.get_relationships_between(source_id, target_id, rel_type)
        return edges[0] if edges else None

    async def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        return self._store.get_node(node_id)

    async def get_edges_between(self, source_id: str, target_id: str) -> List[Edge]:
        """Get all edges between two nodes."""
        return await self.get_relationships_between(source_id, target_id)
