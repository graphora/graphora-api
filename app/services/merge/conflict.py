"""Service for detecting and managing merge conflicts"""
import uuid
import json
from typing import List, Dict, Any, Optional, NamedTuple
from collections import defaultdict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from redis import Redis
from app.baml_client import b

from app.baml_client.types import ConflictGroupAnalysis, RelationshipConflictAnalysis, PropertyConflictAnalysis
from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionOption,
    ResolutionStrategy,
    ConflictBatch,
    ConflictFilter,
    ConflictGroup
)
from app.schemas.graph import GraphResponse, Node, Edge
from app.services.storage.interface import GraphStorageInterface
from app.config import settings
from app.utils.constants import INTERNAL_NODE_TYPES, SYSTEM_PROPERTIES
logger = logging.getLogger(__name__)

class PropertyConflictKey(NamedTuple):
    """Key for grouping similar property conflicts"""
    entity_type: str
    property_name: str
    value_type: str
    
class ConflictDetectionService:
    """Service for detecting conflicts during graph merge"""
    
    def __init__(
        self,
        storage: GraphStorageInterface,
        redis_client: Optional[Redis] = None,
    ):
        """Initialize conflict detection service"""
        self.storage = storage
        self.redis = redis_client or Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,  # Important: Ensure responses are decoded
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        self.executor = ThreadPoolExecutor(max_workers=settings.CONFLICT_DETECTION_WORKERS)
        
        # Test Redis connection
        try:
            self.redis.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
        
    async def detect_conflicts(
        self,
        merge_id: str,
        staging_nodes: List[Node],
        staging_edges: List[Edge],
        production_matches: Dict[str, List[str]],
        batch_size: int = 100,
        progress_callback = None
    ) -> ConflictBatch:
        """Detect all conflicts between staging and production graphs
        
        Args:
            merge_id: ID of the merge operation
            staging_nodes: List of nodes from staging graph
            staging_edges: List of edges from staging graph
            production_matches: Mapping of staging node IDs to production node IDs
            batch_size: Number of entities to process in each batch
            progress_callback: Optional callback function for progress updates
                Function signature: async def callback(processed_items, total_items, conflict_count)
        
        Returns:
            ConflictBatch: Batch of detected conflicts
        """
        all_conflicts = []
        total_items = len(staging_nodes) + len(staging_edges)
        processed_items = 0
        
        logger.info(f"Starting conflict detection for merge {merge_id}")
        logger.info(f"Processing {len(staging_nodes)} nodes and {len(staging_edges)} edges")
        
        # Process nodes in batches
        staging_nodes_by_type = self._group_nodes_by_type(staging_nodes)
        for node_type, nodes in staging_nodes_by_type.items():
            logger.info(f"Processing {len(nodes)} nodes of type {node_type}")
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                # Process batch in parallel
                batch_conflicts = await asyncio.gather(
                    *[
                        self._detect_node_conflicts(node, production_matches, merge_id)
                        for node in batch
                    ]
                )
                
                # Flatten conflicts
                node_conflicts = [
                    conflict
                    for node_conflicts in batch_conflicts
                    for conflict in node_conflicts
                ]
                all_conflicts.extend(node_conflicts)
                logger.info(f"Found {len(node_conflicts)} conflicts in batch of {len(batch)} nodes")
                
                # Update processed count
                processed_items += len(batch)
                
                # Call progress callback if provided
                if progress_callback:
                    await progress_callback(
                        processed_items,
                        total_items,
                        len(all_conflicts)
                    )
            
        # Process edges in batches
        for i in range(0, len(staging_edges), batch_size):
            edge_batch = staging_edges[i:i + batch_size]
            
            # Process batch in parallel
            batch_conflicts = await asyncio.gather(
                *[
                    self._detect_relationship_conflicts(edge, production_matches, merge_id)
                    for edge in edge_batch
                ]
            )
            
            # Flatten conflicts
            edge_conflicts = [
                conflict
                for edge_conflicts in batch_conflicts
                for conflict in edge_conflicts
            ]
            all_conflicts.extend(edge_conflicts)
            
            # Update processed count
            processed_items += len(edge_batch)
            
            # Call progress callback if provided
            if progress_callback:
                await progress_callback(
                    processed_items,
                    total_items,
                    len(all_conflicts)
                )
        
        # Group similar conflicts
        conflict_groups = await self._group_similar_conflicts(all_conflicts)
        
        # Create and store conflict batch
        batch_id = f"batch_{uuid.uuid4().hex}"
        batch = await self.create_conflict_batch(
            merge_id=merge_id,
            conflicts=all_conflicts,
            conflict_groups=conflict_groups
        )
        
        return batch
        
    async def _detect_node_conflicts(
        self,
        staging_node: Node,
        production_matches: Dict[str, List[str]],
        merge_id: str
    ) -> List[Conflict]:
        """Detect all conflicts for a single node"""
        conflicts = []
        
        if staging_node.id not in production_matches:
            return conflicts
            
        # Get production matches
        prod_nodes = []
        for prod_id in production_matches[staging_node.id]:
            prod_node = await self.storage.get_node_by_id(prod_id)
            if prod_node:
                prod_nodes.append(prod_node)
        
        # Detect type conflicts
        type_conflicts = await self._detect_entity_type_conflicts(
            staging_node, prod_nodes, merge_id
        )
        conflicts.extend(type_conflicts)
        
        # Detect property conflicts
        for prod_node in prod_nodes:
            prop_conflicts = await self.detect_property_conflicts(
                staging_node, prod_node, merge_id
            )
            conflicts.extend(prop_conflicts)
            
        # Detect duplicate conflicts if multiple matches
        if len(prod_nodes) > 1:
            conflicts.append(
                self._create_duplicate_entity_conflict(staging_node, prod_nodes, merge_id)
            )
            
        return conflicts
        
    async def _detect_relationship_conflicts(
        self,
        staging_edge: Edge,
        production_matches: Dict[str, List[str]],
        merge_id: Optional[str] = None
    ) -> List[Conflict]:
        """Detect all conflicts for a single relationship"""
        conflicts = []
        
        # Check if source and target nodes have matches in production
        source_has_match = staging_edge.source in production_matches
        target_has_match = staging_edge.target in production_matches
        
        # If either source or target doesn't have a match, no conflicts to detect
        if not source_has_match or not target_has_match:
            return conflicts
            
        # Get all possible source and target matches in production
        source_matches = production_matches.get(staging_edge.source, [])
        target_matches = production_matches.get(staging_edge.target, [])
        
        # Find matching relationships in production
        matching_edges = []
        
        # Get all relationships between all possible source and target matches
        all_node_ids = source_matches + target_matches
        if all_node_ids:
            all_relationships = await self.storage.get_relationships_between_nodes(all_node_ids)
            
            # Filter relationships to only include those between source and target matches
            # and with the same relationship type (if needed)
            for edge in all_relationships:
                if (edge.source in source_matches and 
                    edge.target in target_matches):
                    matching_edges.append(edge)
        
        # If no matching relationships found, no conflicts to detect
        if not matching_edges:
            return conflicts
            
        # Detect conflicts for each matching relationship
        for prod_edge in matching_edges:
            # Detect relationship type conflicts
            if staging_edge.type != prod_edge.type:
                conflict = self._create_relationship_type_conflict(
                    staging_edge,
                    prod_edge,
                    merge_id
                )
                conflicts.append(conflict)
                
            # Detect property conflicts
            prop_conflicts = await self.detect_relationship_conflicts(
                staging_edge, prod_edge, merge_id
            )
            conflicts.extend(prop_conflicts)
            
        # Detect duplicate conflicts if multiple matches
        if len(matching_edges) > 1:
            conflicts.append(
                self._create_duplicate_relationship_conflict(
                    staging_edge, matching_edges, merge_id
                )
            )
            
        return conflicts
        
    async def _group_similar_conflicts(
        self,
        conflicts: List[Conflict]
    ) -> List[ConflictGroup]:
        """Group similar conflicts for batch resolution"""
        # Group property conflicts by type and name
        property_groups = defaultdict(list)
        for conflict in conflicts:
            if conflict.conflict_type in {
                ConflictType.PROPERTY_VALUE,
                ConflictType.PROPERTY_MISSING
            }:
                key = PropertyConflictKey(
                    entity_type=conflict.context["entity_type"],
                    property_name=conflict.context["property_name"],
                    value_type=conflict.context.get("value_type", "unknown")
                )
                property_groups[key].append(conflict)
                
        # Create conflict groups
        groups = []
        for key, group_conflicts in property_groups.items():
            if len(group_conflicts) > 1:  # Only group if multiple conflicts
                # Get BAML analysis
                analysis = await self._analyze_conflict_group(key, group_conflicts)
                
                # Create group
                group = ConflictGroup(
                    id=f"group_{uuid.uuid4().hex}",
                    entity_type=key.entity_type,
                    property_name=key.property_name,
                    value_type=key.value_type,
                    conflict_ids=[c.id for c in group_conflicts],
                    total_conflicts=len(group_conflicts),
                    pattern=analysis.get("pattern", ""),
                    batch_resolvable=analysis.get("batch_resolvable", False),
                    recommended_strategy=analysis.get("strategy"),
                    confidence=analysis.get("confidence", 0.0),
                    risks=analysis.get("risks", [])
                )
                groups.append(group)
                
        return groups
        
    async def _analyze_conflict_group(
        self,
        key: PropertyConflictKey,
        conflicts: List[Conflict]
    ) -> Dict[str, Any]:
        """Use BAML to analyze a group of similar conflicts"""
        # Sample up to 5 conflicts for analysis
        sample_size = min(5, len(conflicts))
        samples = conflicts[:sample_size]
        
        # Format sample conflicts
        sample_text = "\n".join(
            f"Conflict {i+1}:\n"
            f"  Staging: {c.context['staging_value']}\n"
            f"  Production: {c.context['production_value']}"
            for i, c in enumerate(samples)
        )
        
        # Get BAML analysis
        analysis: ConflictGroupAnalysis = b.AnalyzeConflictGroup(
            entity_type=key.entity_type,
            property_name=key.property_name,
            value_type=key.value_type,
            conflict_count=len(conflicts),
            sample_conflicts=sample_text
        )
        
        return {
            "pattern": analysis.pattern,
            "batch_resolvable": analysis.batch_resolvable,
            "strategy": analysis.recommended_strategy,
            "confidence": analysis.confidence,
            "risks": analysis.risks
        }
        
    async def _analyze_property_conflict(
        self,
        staging_node: Node,
        production_node: Node,
        property_name: str,
        staging_value: Any,
        production_value: Any
    ) -> PropertyConflictAnalysis:
        """Use BAML to analyze a property conflict"""
        # Get historical resolutions
        history = await self._get_resolution_history(
            staging_node.label,
            property_name
        )
        
        # Format historical resolutions
        history_str = "\n".join(
            f"Resolution {i+1}:\n"
            f"  Strategy: {h['strategy']}\n"
            f"  Success: {h['success']}\n"
            f"  Reason: {h.get('reason', 'N/A')}"
            for i, h in enumerate(history)
        )
        
        # Get BAML analysis
        analysis: PropertyConflictAnalysis = b.AnalyzePropertyConflict(
            entity_type=staging_node.label,
            property_name=property_name,
            staging_value=str(staging_value),
            production_value=str(production_value),
            value_type=type(staging_value).__name__,
            historical_resolutions=history_str
        )
        
        return analysis
        
    async def _analyze_rel_property_conflict(
        self,
        staging_edge: Edge,
        production_edge: Edge,
        property_name: str,
        staging_value: Any,
        production_value: Any
    ) -> Dict[str, Any]:
        """Use BAML to analyze a property conflict"""
        # Get historical resolutions
        history = await self._get_resolution_history(
            staging_edge.type,
            property_name
        )
        
        # Format historical resolutions
        history_str = "\n".join(
            f"Resolution {i+1}:\n"
            f"  Strategy: {h['strategy']}\n"
            f"  Success: {h['success']}\n"
            f"  Reason: {h.get('reason', 'N/A')}"
            for i, h in enumerate(history)
        )
        
        # Get BAML analysis
        analysis: RelationshipConflictAnalysis = b.AnalyzeRelationshipPropertyConflict(
            relationship_type=staging_edge.type,
            property_name=property_name,
            staging_value=str(staging_value),
            production_value=str(production_value),
            value_type=type(staging_value).__name__,
            graph_context=history_str
        )
        
        return {
            "strategy": analysis.recommended_strategy,
            "confidence": analysis.confidence,
            "explanation": analysis.explanation,
            "can_auto_resolve": analysis.can_auto_resolve,
            "risks": analysis.risks
        }
        
    async def _get_resolution_history(
        self,
        entity_type: str,
        property_name: str
    ) -> List[Dict[str, Any]]:
        """Get historical conflict resolutions"""
        key = f"resolution_history:{entity_type}:{property_name}"
        history = await self._redis_operation(
            lambda: self.redis.lrange(key, 0, -1)
        )
        return [eval(h) for h in history] if history else []

    async def create_conflict_batch(
        self,
        merge_id: str,
        conflicts: List[Conflict],
        conflict_groups: List[ConflictGroup]
    ) -> ConflictBatch:
        """Create and store a batch of conflicts"""
        batch_id = f"batch_{uuid.uuid4().hex}"
        batch = ConflictBatch(
            batch_id=batch_id,
            merge_id=merge_id,
            conflicts=conflicts,
            conflict_groups=conflict_groups,
            total_conflicts=len(conflicts)
        )
        
        # Store the batch in Redis
        await self._store_conflict_batch(batch)
        
        return batch
        
    async def _store_conflict_batch(self, batch: ConflictBatch) -> None:
        """Store a conflict batch in Redis"""
        # Store each conflict
        for conflict in batch.conflicts:
            key = f"merge:{batch.merge_id}:conflict:{conflict.id}"
            self.redis.set(key, conflict.model_dump_json())
            
        # Store list of conflict IDs
        conflict_ids = [conflict.id for conflict in batch.conflicts]
        self.redis.set(
            f"merge:{batch.merge_id}:conflict_ids",
            json.dumps(conflict_ids)
        )
        
        # Store conflict groups
        for group in batch.conflict_groups:
            key = f"merge:{batch.merge_id}:group:{group.id}"
            self.redis.set(key, group.model_dump_json())
            
        # Store group IDs
        group_ids = [group.id for group in batch.conflict_groups]
        self.redis.set(
            f"merge:{batch.merge_id}:group_ids",
            json.dumps(group_ids)
        )
        
        # Store counts by type and severity
        counts = {
            "total": len(batch.conflicts),
            "by_type": {},
            "by_severity": {},
            "resolved": 0,
            "unresolved": len(batch.conflicts)
        }
        
        for conflict in batch.conflicts:
            # Count by type
            type_key = conflict.conflict_type.value
            counts["by_type"][type_key] = counts["by_type"].get(type_key, 0) + 1
            
            # Count by severity
            severity_key = conflict.severity.value
            counts["by_severity"][severity_key] = counts["by_severity"].get(severity_key, 0) + 1
            
        self.redis.set(
            f"merge:{batch.merge_id}:conflict_counts",
            json.dumps(counts)
        )
        
        # Set TTL for cleanup (30 days)
        ttl = 30 * 24 * 60 * 60  # 30 days in seconds
        for key in [
            f"merge:{batch.merge_id}:conflict_ids",
            f"merge:{batch.merge_id}:group_ids",
            f"merge:{batch.merge_id}:conflict_counts"
        ] + [f"merge:{batch.merge_id}:conflict:{c.id}" for c in batch.conflicts] + [f"merge:{batch.merge_id}:group:{g.id}" for g in batch.conflict_groups]:
            self.redis.expire(key, ttl)

    def _get_redis_key(self, merge_id: str, suffix: str) -> str:
        """Get Redis key for conflict data"""
        # Remove any existing merge: prefix to avoid double prefixing
        merge_id = merge_id.replace("merge:", "")
        return f"merge:{merge_id}:conflicts:{suffix}"

    async def get_conflicts(
        self,
        merge_id: str,
        filter_criteria: Optional[ConflictFilter] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Conflict]:
        """Get conflicts for a merge operation with filtering"""
        logger.info(f"Getting conflicts for merge {merge_id}")
        
        # Get all conflict batches for merge
        pattern = self._get_redis_key(merge_id, "batch:*")
        logger.info(f"Searching for conflict batches with pattern: {pattern}")
        
        # List all keys to debug
        all_keys = await self._redis_operation(
            lambda: self.redis.keys("merge:*")
        )
        logger.info(f"All merge-related Redis keys: {all_keys}")
        
        batch_keys = await self._redis_operation(
            lambda: self.redis.keys(pattern)
        )
        logger.info(f"Found {len(batch_keys)} conflict batches")
        
        all_conflicts = []
        for key in batch_keys:
            logger.info(f"Retrieving conflict batch from key: {key}")
            try:
                batch_data = await self._redis_operation(
                    lambda: self.redis.get(key)
                )
                if batch_data:
                    try:
                        batch = ConflictBatch.model_validate_json(batch_data)
                        all_conflicts.extend(batch.conflicts)
                        logger.info(f"Retrieved {len(batch.conflicts)} conflicts from batch")
                    except Exception as e:
                        logger.error(f"Failed to parse conflict batch data: {str(e)}")
                        logger.error(f"Raw batch data: {batch_data[:200]}...")  # Log first 200 chars
                else:
                    logger.warning(f"No data found for conflict batch key: {key}")
            except Exception as e:
                logger.error(f"Failed to retrieve conflict batch: {str(e)}")
        
        logger.info(f"Total conflicts found before filtering: {len(all_conflicts)}")
        
        # Apply filters if provided
        if filter_criteria:
            all_conflicts = self._apply_conflict_filters(
                all_conflicts,
                filter_criteria
            )
            
        # Apply pagination
        return all_conflicts[offset:offset + limit]
        
    def _apply_conflict_filters(
        self,
        conflicts: List[Conflict],
        filter_criteria: ConflictFilter
    ) -> List[Conflict]:
        """Apply filter criteria to conflicts"""
        filtered = conflicts
        
        if filter_criteria.conflict_types:
            filtered = [
                c for c in filtered
                if c.conflict_type in filter_criteria.conflict_types
            ]
            
        if filter_criteria.severities:
            filtered = [
                c for c in filtered
                if c.severity in filter_criteria.severities
            ]
            
        if filter_criteria.resolved is not None:
            filtered = [
                c for c in filtered
                if c.resolved == filter_criteria.resolved
            ]
            
        if filter_criteria.entity_ids:
            filtered = [
                c for c in filtered
                if any(id in filter_criteria.entity_ids
                      for id in c.staging_ids + c.production_ids)
            ]
            
        if filter_criteria.detected_after:
            filtered = [
                c for c in filtered
                if c.detected_at >= filter_criteria.detected_after
            ]
            
        if filter_criteria.detected_before:
            filtered = [
                c for c in filtered
                if c.detected_at <= filter_criteria.detected_before
            ]
            
        return filtered
        
    def _are_values_equal(self, value1: Any, value2: Any) -> bool:
        """Compare two values for equality, handling special cases"""
        # Handle None values
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False
            
        # Handle numeric types
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return abs(float(value1) - float(value2)) < 1e-10
            
        # Handle lists/sets
        if isinstance(value1, (list, set)) and isinstance(value2, (list, set)):
            return set(value1) == set(value2)
            
        # Handle dictionaries
        if isinstance(value1, dict) and isinstance(value2, dict):
            if set(value1.keys()) != set(value2.keys()):
                return False
            return all(self._are_values_equal(value1[k], value2[k]) for k in value1)
            
        # Default comparison
        return value1 == value2
    
    def remove_system_properties(
        self, keys: List[str], props: Dict[str, Any]) -> Dict[str, Any]:
        for key in keys:
            if key in props:
                del props[key]
        return props

    async def detect_property_conflicts(
        self,
        staging_node: Node,
        production_node: Node,
        merge_id: str
    ) -> List[Conflict]:
        """Detect property conflicts between two nodes"""
        conflicts = []
        
        # Get all property keys from both nodes
        staging_props = staging_node.properties or {}
        production_props = production_node.properties or {}
        
        # Skip transform_id property
        staging_props = self.remove_system_properties(SYSTEM_PROPERTIES, staging_props)
        production_props = self.remove_system_properties(SYSTEM_PROPERTIES, production_props)
        
        # Get all unique property keys
        all_keys = set(staging_props.keys()) | set(production_props.keys())
        
        # Check each property
        for prop_name in all_keys:
            staging_value = staging_props.get(prop_name)
            production_value = production_props.get(prop_name)
            
            # Skip if both values are None
            if staging_value is None and production_value is None:
                continue
                
            # Skip if values are equal
            if self._are_values_equal(staging_value, production_value):
                continue
                
            # Create conflict
            conflict = await self._create_property_conflict(
                staging_node,
                production_node,
                prop_name,
                staging_value,
                production_value,
                merge_id
            )
            
            if conflict:
                conflicts.append(conflict)
                
        return conflicts
        
    async def _create_property_conflict(
        self,
        staging_node: Node,
        production_node: Node,
        property_name: str,
        staging_value: Any,
        production_value: Any,
        merge_id: Optional[str] = None
    ) -> Conflict:
        """Create a property value conflict between nodes"""
        conflict_id = f"prop_{staging_node.id}_{production_node.id}_{property_name}"
        
        # Get BAML analysis
        analysis: PropertyConflictAnalysis = await self._analyze_property_conflict(
            staging_node,
            production_node,
            property_name,
            staging_value,
            production_value
        )
        
        # Create resolution options from analysis
        recommended_strategy = analysis.recommended_strategy
        
        # Initialize base options list
        resolution_options = []
        
        # Define all strategies with default values
        strategies = [
            (ResolutionStrategy.KEEP_STAGING.value, f"Keep staging value: {staging_value}"),
            (ResolutionStrategy.KEEP_PRODUCTION.value, f"Keep production value: {production_value}"),
            (ResolutionStrategy.MERGE_VALUES.value, f"Merge both values: {staging_value} + {production_value}")
        ]
        
        # Add options based on recommended strategy
        for strategy, description in strategies:
            confidence = 0.5
            reasoning = ""
            risks = []
            
            # If this strategy matches recommendation, use analysis values
            if strategy == recommended_strategy:
                confidence = analysis.confidence or 0.5
                reasoning = analysis.explanation
                risks = analysis.potential_risks
            
            # Determine value based on strategy
            if strategy == ResolutionStrategy.KEEP_STAGING:
                value = staging_value
            elif strategy == ResolutionStrategy.KEEP_PRODUCTION:
                value = production_value
            else:  # MERGE_VALUES
                # For now, concatenate values if they're strings, otherwise keep staging
                if isinstance(staging_value, str) and isinstance(production_value, str):
                    value = f"{staging_value} {production_value}"
                else:
                    value = staging_value
            
            option = ResolutionOption(
                id=f"{conflict_id}_{strategy}",
                description=description,
                resolution_type=strategy,
                resolution_data={
                    "property_name": property_name,
                    "value": value
                },
                confidence=confidence,
                reasoning=reasoning,
                auto_resolvable=analysis.can_auto_resolve or False,
                risks=risks
            )
            resolution_options.append(option)
        
        # Sort options to ensure recommended strategy is first
        if recommended_strategy:
            resolution_options.sort(key=lambda x: x.resolution_type != recommended_strategy)
            
        return Conflict(
            id=conflict_id,
            merge_id=merge_id,
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            staging_ids=[staging_node.id],
            production_ids=[production_node.id],
            staging_value=staging_value,
            production_value=production_value,
            description=f"Property '{property_name}' has different values",
            context={
                "entity_type": staging_node.label,
                "property_name": property_name,
                "staging_value": staging_value,
                "production_value": production_value,
                "value_type": type(staging_value).__name__ if staging_value is not None else type(production_value).__name__,
                "analysis": analysis
            },
            resolution_options=resolution_options
        )
        
    async def detect_relationship_conflicts(
        self,
        staging_edge: Edge,
        production_edge: Edge,
        merge_id: Optional[str] = None
    ) -> List[Conflict]:
        """Detect conflicts between two relationships"""
        conflicts = []
        
        # Check relationship type conflicts
        if staging_edge.type != production_edge.type:
            conflict = self._create_relationship_type_conflict(
                staging_edge,
                production_edge,
                merge_id
            )
            conflicts.append(conflict)
            
        # Check property conflicts
        all_props = set(staging_edge.properties.keys()) | set(production_edge.properties.keys())
        
        
        for prop_name in all_props:
            if prop_name in SYSTEM_PROPERTIES:
                continue
            staging_value = staging_edge.properties.get(prop_name)
            prod_value = production_edge.properties.get(prop_name)
            
            # Skip if values are identical
            if staging_value == prod_value:
                continue
                
            # Create property conflict
            conflict = await self._create_relationship_property_conflict(
                staging_edge,
                production_edge,
                prop_name,
                staging_value,
                prod_value,
                merge_id
            )
            if conflict:
                conflicts.append(conflict)
                
        return conflicts
        
    async def detect_entity_matching_conflicts(
        self,
        staging_node: Node,
        production_nodes: List[Node]
    ) -> List[Conflict]:
        """Detect conflicts when a staging entity matches multiple production entities"""
        # Get similarity analysis from BAML
        analysis = await b.entity_matching.analyze_entity_matches(
            staging_entity={
                "id": staging_node.id,
                "type": staging_node.label,
                "properties": staging_node.properties
            },
            production_entities=[{
                "id": node.id,
                "type": node.label,
                "properties": node.properties
            } for node in production_nodes]
        )
        
        # Create conflict with resolution options
        conflict_id = f"entity_match_{staging_node.id}"
        
        return [Conflict(
            id=conflict_id,
            conflict_type=ConflictType.DUPLICATE_ENTITY,
            severity=ConflictSeverity.CRITICAL,
            staging_ids=[staging_node.id],
            production_ids=[node.id for node in production_nodes],
            description=analysis.description,
            context={
                "staging_entity": {
                    "id": staging_node.id,
                    "type": staging_node.label,
                    "properties": staging_node.properties
                },
                "production_entities": [{
                    "id": node.id,
                    "type": node.label,
                    "properties": node.properties
                } for node in production_nodes],
                "analysis": analysis.model_dump()
            },
            resolution_options=[
                # Option to merge all entities
                ResolutionOption(
                    id=f"{conflict_id}_merge",
                    description="Merge all matching entities",
                    resolution_type=ResolutionStrategy.MERGE_VALUES,
                    resolution_data={
                        "staging_id": staging_node.id,
                        "production_ids": [node.id for node in production_nodes],
                        "merge_strategy": analysis.recommended_merge_strategy
                    },
                    confidence=analysis.merge_confidence,
                    auto_resolvable=False
                ),
                # Option to keep each production entity
                *[
                    ResolutionOption(
                        id=f"{conflict_id}_keep_{node.id}",
                        description=f"Keep match with {node.id}",
                        resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                        resolution_data={
                            "staging_id": staging_node.id,
                            "production_id": node.id
                        },
                        confidence=analysis.match_confidences.get(node.id, 0.5),
                        auto_resolvable=False
                    )
                    for node in production_nodes
                ],
                # Option to create new entity
                ResolutionOption(
                    id=f"{conflict_id}_new",
                    description="Create new entity",
                    resolution_type=ResolutionStrategy.CREATE_NEW,
                    resolution_data={
                        "staging_id": staging_node.id,
                        "production_ids": [node.id for node in production_nodes]
                    },
                    confidence=0.3,
                    auto_resolvable=False
                )
            ]
        )]
        
    async def _detect_entity_type_conflicts(
        self,
        staging_node: Node,
        production_nodes: List[Node],
        merge_id: Optional[str] = None
    ) -> List[Conflict]:
        """Detect conflicts where the entity type (label) differs between staging and production"""
        conflicts = []
        
        for production_node in production_nodes:
            if staging_node.label != production_node.label:
                conflict_id = f"type_{staging_node.id}_{production_node.id}"
                conflicts.append(Conflict(
                    id=conflict_id,
                    merge_id=merge_id,
                    conflict_type=ConflictType.ENTITY_TYPE,
                    severity=ConflictSeverity.CRITICAL,
                    staging_ids=[staging_node.id],
                    production_ids=[production_node.id],
                    description=f"Entity type mismatch: staging='{staging_node.label}', production='{production_node.label}'",
                    context={
                        "staging_type": staging_node.label,
                        "production_type": production_node.label,
                        "staging_props": staging_node.properties,
                        "production_props": production_node.properties
                    },
                    resolution_options=[
                        ResolutionOption(
                            id=f"{conflict_id}_staging",
                            description=f"Keep staging type: {staging_node.label}",
                            resolution_type=ResolutionStrategy.KEEP_STAGING,
                            resolution_data={"entity_type": staging_node.label},
                            confidence=0.5,
                            auto_resolvable=False
                        ),
                        ResolutionOption(
                            id=f"{conflict_id}_prod",
                            description=f"Keep production type: {production_node.label}",
                            resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                            resolution_data={"entity_type": production_node.label},
                            confidence=0.5,
                            auto_resolvable=False
                        )
                    ]
                ))
                    
        return conflicts
        
    def _create_duplicate_entity_conflict(
        self,
        staging_node: Node,
        production_nodes: List[Node],
        merge_id: Optional[str] = None
    ) -> Conflict:
        """Create a conflict for duplicate entity matches"""
        conflict_id = f"duplicate_{staging_node.id}"
        
        # Get key properties for comparison
        key_props = self._get_key_properties(staging_node, production_nodes)
        
        # Format property comparison table
        prop_table = self._format_property_comparison(
            staging_node, production_nodes, key_props
        )
        
        return Conflict(
            id=conflict_id,
            merge_id=merge_id,
            conflict_type=ConflictType.DUPLICATE_ENTITY,
            severity=ConflictSeverity.CRITICAL,
            staging_ids=[staging_node.id],
            production_ids=[node.id for node in production_nodes],
            description=f"Multiple production matches found for {staging_node.label}",
            context={
                "entity_type": staging_node.label,
                "staging_props": staging_node.properties,
                "production_matches": [
                    {
                        "id": node.id,
                        "properties": node.properties
                    }
                    for node in production_nodes
                ],
                "key_properties": key_props,
                "property_comparison": prop_table
            },
            resolution_options=[
                ResolutionOption(
                    id=f"{conflict_id}_merge",
                    description="Merge all matching entities",
                    resolution_type=ResolutionStrategy.MERGE_VALUES,
                    resolution_data={
                        "staging_id": staging_node.id,
                        "production_ids": [node.id for node in production_nodes]
                    },
                    confidence=0.5,
                    auto_resolvable=False
                ),
                ResolutionOption(
                    id=f"{conflict_id}_new",
                    description="Create new entity with merged properties",
                    resolution_type=ResolutionStrategy.CREATE_NEW,
                    resolution_data={
                        "staging_id": staging_node.id,
                        "production_ids": [node.id for node in production_nodes]
                    },
                    confidence=0.3,
                    auto_resolvable=False
                ),
                *[
                    ResolutionOption(
                        id=f"{conflict_id}_keep_{node.id}",
                        description=f"Keep only match with {node.id}",
                        resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                        resolution_data={
                            "staging_id": staging_node.id,
                            "production_id": node.id
                        },
                        confidence=0.7 / len(production_nodes),
                        auto_resolvable=False
                    )
                    for node in production_nodes
                ]
            ]
        )
        
    def _get_key_properties(
        self,
        staging_node: Node,
        production_nodes: List[Node]
    ) -> List[str]:
        """Get key properties for entity comparison"""
        # Start with common properties
        common_props = set(staging_node.properties.keys())
        for node in production_nodes:
            common_props &= set(node.properties.keys())
            
        # Prioritize likely key properties
        key_indicators = {
            'id', 'name', 'key', 'code', 'identifier',
            'email', 'phone', 'address', 'url'
        }
        
        key_props = []
        
        # First add exact matches
        key_props.extend(
            prop for prop in common_props
            if prop.lower() in key_indicators
        )
        
        # Then add properties containing key words
        key_props.extend(
            prop for prop in common_props
            if prop not in key_props
            and any(indicator in prop.lower() for indicator in key_indicators)
        )
        
        # Add remaining common properties
        remaining = list(common_props - set(key_props))
        remaining.sort()  # For consistent ordering
        key_props.extend(remaining)
        
        return key_props
        
    def _format_property_comparison(
        self,
        staging_node: Node,
        production_nodes: List[Node],
        properties: List[str]
    ) -> str:
        """Format property comparison table for duplicate entities"""
        # Table header
        header = ["Property", "Staging"] + [f"Prod {i+1}" for i in range(len(production_nodes))]
        rows = [header]
        
        # Add property rows
        for prop in properties:
            row = [
                prop,
                str(staging_node.properties.get(prop, "N/A"))
            ]
            row.extend(
                str(node.properties.get(prop, "N/A"))
                for node in production_nodes
            )
            rows.append(row)
            
        # Calculate column widths
        col_widths = [
            max(len(str(row[i])) for row in rows)
            for i in range(len(header))
        ]
        
        # Format table
        formatted = []
        for row in rows:
            formatted_row = " | ".join(
                str(cell).ljust(width)
                for cell, width in zip(row, col_widths)
            )
            formatted.append(formatted_row)
            
            # Add separator after header
            if row == header:
                separator = "-" * len(formatted_row)
                formatted.append(separator)
                
        return "\n".join(formatted)

    def _create_relationship_type_conflict(
        self,
        staging_edge: Edge,
        production_edge: Edge,
        merge_id: Optional[str] = None
    ) -> Conflict:
        """Create a relationship type conflict"""
        conflict_id = f"rel_type_{staging_edge.id}_{production_edge.id}"
        
        return Conflict(
            id=conflict_id,
            merge_id=merge_id or "test-merge-id",  # Use default if not provided
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=[staging_edge.id],
            production_ids=[production_edge.id],
            staging_value=staging_edge.type,
            production_value=production_edge.type,
            description=(
                f"Different relationship types between the same entities: "
                f"staging='{staging_edge.type}', production='{production_edge.type}'"
            ),
            context={
                "staging_type": staging_edge.type,
                "production_type": production_edge.type,
                "source_id": staging_edge.source,
                "target_id": staging_edge.target,
                "staging_props": staging_edge.properties,
                "production_props": production_edge.properties
            },
            resolution_options=[
                ResolutionOption(
                    id=f"{conflict_id}_staging",
                    description=f"Keep staging relationship type: {staging_edge.type}",
                    resolution_type=ResolutionStrategy.KEEP_STAGING_REL,
                    resolution_data={
                        "edge_id": staging_edge.id,
                        "relationship_type": staging_edge.type
                    },
                    confidence=0.5,
                    auto_resolvable=False
                ),
                ResolutionOption(
                    id=f"{conflict_id}_prod",
                    description=f"Keep production relationship type: {production_edge.type}",
                    resolution_type=ResolutionStrategy.KEEP_PRODUCTION_REL,
                    resolution_data={
                        "edge_id": production_edge.id,
                        "relationship_type": production_edge.type
                    },
                    confidence=0.5,
                    auto_resolvable=False
                ),
                ResolutionOption(
                    id=f"{conflict_id}_both",
                    description="Keep both relationships",
                    resolution_type=ResolutionStrategy.KEEP_BOTH_RELS,
                    resolution_data={
                        "staging_edge_id": staging_edge.id,
                        "production_edge_id": production_edge.id
                    },
                    confidence=0.3,
                    auto_resolvable=False
                )
            ]
        )
        
    async def _create_relationship_property_conflict(
        self,
        staging_edge: Edge,
        production_edge: Edge,
        property_name: str,
        staging_value: Any,
        production_value: Any,
        merge_id: Optional[str] = None
    ) -> Conflict:
        """Create a relationship property value conflict"""
        conflict_id = f"rel_prop_{staging_edge.id}_{production_edge.id}_{property_name}"
        
        # Get BAML analysis
        analysis = await self._analyze_rel_property_conflict(
            staging_edge, production_edge,
            property_name, staging_value, production_value
        )
        
        return Conflict(
            id=conflict_id,
            merge_id=merge_id,
            conflict_type=ConflictType.RELATIONSHIP_PROPERTY,
            severity=ConflictSeverity.MAJOR,
            staging_ids=[staging_edge.id],
            production_ids=[production_edge.id],
            description=f"Relationship property '{property_name}' has different values",
            context={
                "relationship_type": staging_edge.type,
                "property_name": property_name,
                "staging_value": staging_value,
                "production_value": production_value,
                "value_type": type(staging_value).__name__,
                "analysis": analysis
            },
            resolution_options=[
                ResolutionOption(
                    id=f"{conflict_id}_staging",
                    description=f"Keep staging value: {staging_value}",
                    resolution_type=ResolutionStrategy.KEEP_STAGING_REL,
                    resolution_data={
                        "edge_id": staging_edge.id,
                        "property_name": property_name,
                        "value": staging_value
                    },
                    confidence=0.5,
                    auto_resolvable=analysis.get("can_auto_resolve", False)
                ),
                ResolutionOption(
                    id=f"{conflict_id}_prod",
                    description=f"Keep production value: {production_value}",
                    resolution_type=ResolutionStrategy.KEEP_PRODUCTION_REL,
                    resolution_data={
                        "edge_id": production_edge.id,
                        "property_name": property_name,
                        "value": production_value
                    },
                    confidence=0.5,
                    auto_resolvable=analysis.get("can_auto_resolve", False)
                ),
                ResolutionOption(
                    id=f"{conflict_id}_merge",
                    description="Merge property values",
                    resolution_type=ResolutionStrategy.MERGE_REL_PROPS,
                    resolution_data={
                        "edge_id": staging_edge.id,
                        "property_name": property_name,
                        "staging_value": staging_value,
                        "production_value": production_value
                    },
                    confidence=0.3,
                    auto_resolvable=False
                )
            ]
        )

    def _create_duplicate_relationship_conflict(
        self,
        staging_edge: Edge,
        production_edges: List[Edge],
        merge_id: Optional[str] = None
    ) -> Conflict:
        """Create a duplicate relationship conflict when multiple matches are found"""
        conflict_id = f"duplicate_rel_{staging_edge.id}"
        
        # Extract production edge IDs
        production_ids = [edge.id for edge in production_edges]
        
        # Create description with relationship details
        description = (
            f"Multiple matching relationships found in production for staging relationship "
            f"'{staging_edge.id}' of type '{staging_edge.type}' between "
            f"source '{staging_edge.source}' and target '{staging_edge.target}'. "
            f"Found {len(production_edges)} potential matches."
        )
        
        # Create context with relationship details
        context = {
            "staging_edge": {
                "id": staging_edge.id,
                "type": staging_edge.type,
                "source": staging_edge.source,
                "target": staging_edge.target,
                "properties": staging_edge.properties
            },
            "production_edges": [
                {
                    "id": edge.id,
                    "type": edge.type,
                    "source": edge.source,
                    "target": edge.target,
                    "properties": edge.properties
                }
                for edge in production_edges
            ]
        }
        
        # Create resolution options
        resolution_options = [
            ResolutionOption(
                id=f"{conflict_id}_staging",
                description="Keep staging relationship",
                resolution_type=ResolutionStrategy.KEEP_STAGING_REL,
                resolution_data={
                    "edge_id": staging_edge.id
                },
                confidence=0.4,
                auto_resolvable=False
            )
        ]
        
        # Add option for each production edge
        for i, prod_edge in enumerate(production_edges):
            resolution_options.append(
                ResolutionOption(
                    id=f"{conflict_id}_prod_{i}",
                    description=f"Keep production relationship {i+1}: {prod_edge.id}",
                    resolution_type=ResolutionStrategy.KEEP_PRODUCTION_REL,
                    resolution_data={
                        "edge_id": prod_edge.id
                    },
                    confidence=0.3,
                    auto_resolvable=False
                )
            )
        
        # Add option to keep all
        resolution_options.append(
            ResolutionOption(
                id=f"{conflict_id}_all",
                description="Keep all relationships",
                resolution_type=ResolutionStrategy.KEEP_ALL_RELS,
                resolution_data={
                    "staging_edge_id": staging_edge.id,
                    "production_edge_ids": production_ids
                },
                confidence=0.2,
                auto_resolvable=False
            )
        )
        
        return Conflict(
            id=conflict_id,
            merge_id=merge_id or "test-merge-id",  # Use default if not provided
            conflict_type=ConflictType.DUPLICATE_RELATIONSHIP,
            severity=ConflictSeverity.MAJOR,
            staging_ids=[staging_edge.id],
            production_ids=production_ids,
            staging_value=f"Relationship of type '{staging_edge.type}'",
            production_value=f"{len(production_edges)} matching relationships",
            description=description,
            context=context,
            resolution_options=resolution_options
        )

    def _group_nodes_by_type(self, nodes: List[Node]) -> Dict[str, List[Node]]:
        """Group nodes by their type"""
        grouped = {}
        for node in nodes:
            if node.type in INTERNAL_NODE_TYPES:
                continue
            if node.type not in grouped:
                grouped[node.type] = []
            grouped[node.type].append(node)
        return grouped

    async def _redis_operation(self, func):
        """Perform Redis operation with async handling"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func)
