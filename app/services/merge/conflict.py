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

from app.baml_client.types import ConflictGroupAnalysis, EntitySimilarityAnalysis, PropertyConflictAnalysis
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
            db=settings.REDIS_DB
        )
        self.executor = ThreadPoolExecutor(max_workers=settings.CONFLICT_DETECTION_WORKERS)
        
    async def detect_conflicts(
        self,
        merge_id: str,
        staging_nodes: List[Node],
        staging_edges: List[Edge],
        production_matches: Dict[str, List[str]],
        batch_size: int = 100
    ) -> ConflictBatch:
        """Detect all conflicts between staging and production graphs"""
        all_conflicts = []
        
        # Process nodes in batches
        staging_nodes_by_type = self._group_nodes_by_type(staging_nodes)
        for node_type, nodes in staging_nodes_by_type.items():
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                # Process batch in parallel
                batch_conflicts = await asyncio.gather(
                    *[
                        self._detect_node_conflicts(node, production_matches)
                        for node in batch
                    ]
                )
                
                # Flatten conflicts
                all_conflicts.extend([
                    conflict
                    for node_conflicts in batch_conflicts
                    for conflict in node_conflicts
                ])
                
        # Process edges in batches
        for i in range(0, len(staging_edges), batch_size):
            edge_batch = staging_edges[i:i + batch_size]
            
            # Process batch in parallel
            batch_conflicts = await asyncio.gather(
                *[
                    self._detect_relationship_conflicts(edge, production_matches)
                    for edge in edge_batch
                ]
            )
            
            # Flatten conflicts
            all_conflicts.extend([
                conflict
                for edge_conflicts in batch_conflicts
                for conflict in edge_conflicts
            ])
            
        # Group similar conflicts
        conflict_groups = await self._group_similar_conflicts(all_conflicts)
        
        # Create and store conflict batch
        batch = await self.create_conflict_batch(
            merge_id=merge_id,
            conflicts=all_conflicts,
            conflict_groups=conflict_groups
        )
        
        return batch
        
    async def _detect_node_conflicts(
        self,
        staging_node: Node,
        production_matches: Dict[str, List[str]]
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
            staging_node, prod_nodes
        )
        conflicts.extend(type_conflicts)
        
        # Detect property conflicts
        for prod_node in prod_nodes:
            prop_conflicts = await self._detect_property_conflicts(
                staging_node, prod_node
            )
            conflicts.extend(prop_conflicts)
            
        # Detect duplicate conflicts if multiple matches
        if len(prod_nodes) > 1:
            conflicts.append(
                self._create_duplicate_entity_conflict(staging_node, prod_nodes)
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
    ) -> Dict[str, Any]:
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
        
        return {
            "strategy": analysis.recommended_strategy,
            "confidence": analysis.confidence,
            "explanation": analysis.explanation,
            "can_auto_resolve": analysis.can_auto_resolve,
            "risks": analysis.potential_risks
        }
        
    async def _get_resolution_history(
        self,
        entity_type: str,
        property_name: str
    ) -> List[Dict[str, Any]]:
        """Get historical conflict resolutions"""
        key = f"resolution_history:{entity_type}:{property_name}"
        history = self.redis.lrange(key, 0, -1)
        return [eval(h) for h in history] if history else []

    async def create_conflict_batch(
        self,
        merge_id: str,
        conflicts: List[Conflict],
        conflict_groups: List[ConflictGroup]
    ) -> ConflictBatch:
        """Create a new batch of conflicts"""
        batch_id = f"batch_{uuid.uuid4().hex}"
        
        batch = ConflictBatch(
            batch_id=batch_id,
            merge_id=merge_id,
            conflicts=conflicts,
            conflict_groups=conflict_groups,
            total_conflicts=len(conflicts)
        )
        
        # Store batch in Redis
        self.redis.set(
            self._get_redis_key(merge_id, f"batch:{batch_id}"),
            batch.model_dump_json(),
            ex=settings.CONFLICT_BATCH_TTL
        )
        
        return batch
        
    def _get_redis_key(self, merge_id: str, suffix: str) -> str:
        """Get Redis key for conflict data"""
        return f"merge:{merge_id}:conflicts:{suffix}"
        
    async def get_conflicts(
        self,
        merge_id: str,
        filter_criteria: Optional[ConflictFilter] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Conflict]:
        """Get conflicts for a merge operation with filtering"""
        # Get all conflict batches for merge
        batch_keys = self.redis.keys(
            self._get_redis_key(merge_id, "batch:*")
        )
        
        all_conflicts = []
        for key in batch_keys:
            batch_data = self.redis.get(key)
            if batch_data:
                batch = ConflictBatch.model_validate_json(batch_data)
                all_conflicts.extend(batch.conflicts)
                
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
        
    async def _detect_property_conflicts(
        self,
        staging_node: Node,
        production_node: Node
    ) -> List[Conflict]:
        """Detect property conflicts between staging and production nodes"""
        conflicts = []
        
        # Get all property names
        all_props = set(staging_node.properties.keys()) | set(production_node.properties.keys())
        
        for prop_name in all_props:
            staging_value = staging_node.properties.get(prop_name)
            prod_value = production_node.properties.get(prop_name)
            
            # Missing property conflict
            if staging_value is None and prod_value is not None:
                conflicts.append(self._create_missing_property_conflict(
                    staging_node, production_node, prop_name, "staging"
                ))
                continue
                
            if prod_value is None and staging_value is not None:
                conflicts.append(self._create_missing_property_conflict(
                    staging_node, production_node, prop_name, "production"
                ))
                continue
                
            # Property value conflict
            if not self._are_property_values_equal(staging_value, prod_value):
                conflicts.append(self._create_property_value_conflict(
                    staging_node, production_node, prop_name,
                    staging_value, prod_value
                ))
                
        return conflicts
        
    def _are_property_values_equal(self, value1: Any, value2: Any) -> bool:
        """Compare property values with type-specific logic"""
        if value1 is None and value2 is None:
            return True
            
        if value1 is None or value2 is None:
            return False
            
        # String comparison (case-insensitive)
        if isinstance(value1, str) and isinstance(value2, str):
            return value1.lower() == value2.lower()
            
        # Numeric comparison with tolerance
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            if isinstance(value1, int) and isinstance(value2, int):
                return value1 == value2
            else:
                return abs(float(value1) - float(value2)) < settings.NUMERIC_COMPARISON_TOLERANCE
                
        # List comparison (order-insensitive)
        if isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
                
            # For primitive values, compare as sets
            if all(isinstance(v, (str, int, float, bool, type(None))) 
                  for v in value1 + value2):
                return set(str(x) if x is not None else 'None' for x in value1) == \
                       set(str(x) if x is not None else 'None' for x in value2)
                       
            # For complex items, try to match each item
            matched_indices = set()
            for i, item1 in enumerate(value1):
                for j, item2 in enumerate(value2):
                    if j not in matched_indices and self._are_property_values_equal(item1, item2):
                        matched_indices.add(j)
                        break
                else:
                    return False
            return len(matched_indices) == len(value2)
            
        # Dict comparison
        if isinstance(value1, dict) and isinstance(value2, dict):
            if set(value1.keys()) != set(value2.keys()):
                return False
            return all(
                self._are_property_values_equal(value1[k], value2[k])
                for k in value1.keys()
            )
            
        # Default comparison
        return value1 == value2
        
    def _create_property_value_conflict(
        self,
        staging_node: Node,
        production_node: Node,
        property_name: str,
        staging_value: Any,
        production_value: Any
    ) -> Conflict:
        """Create a property value conflict"""
        conflict_id = f"prop_val_{staging_node.id}_{production_node.id}_{property_name}"
        
        # Determine severity based on property importance
        severity = ConflictSeverity.MAJOR
        if property_name in {"id", "name", "key", "identifier"}:
            severity = ConflictSeverity.CRITICAL
            
        # Create merge strategy based on value type
        merge_strategy = "keep_one"
        merge_description = "Keep one of the values"
        
        if isinstance(staging_value, (list, set)):
            merge_strategy = "combine"
            merge_description = "Combine both lists"
        elif isinstance(staging_value, dict):
            merge_strategy = "deep_merge"
            merge_description = "Deep merge both objects"
        elif isinstance(staging_value, str):
            merge_strategy = "concat"
            merge_description = "Concatenate values"
        elif isinstance(staging_value, (int, float)):
            merge_strategy = "average"
            merge_description = "Use average value"
            
        resolution_options = [
            ResolutionOption(
                id=f"{conflict_id}_staging",
                description=f"Keep staging value: {staging_value}",
                resolution_type=ResolutionStrategy.KEEP_STAGING,
                resolution_data={
                    "property_name": property_name,
                    "value": staging_value
                },
                confidence=0.6,
                auto_resolvable=severity != ConflictSeverity.CRITICAL
            ),
            ResolutionOption(
                id=f"{conflict_id}_prod",
                description=f"Keep production value: {production_value}",
                resolution_type=ResolutionStrategy.KEEP_PRODUCTION,
                resolution_data={
                    "property_name": property_name,
                    "value": production_value
                },
                confidence=0.6,
                auto_resolvable=severity != ConflictSeverity.CRITICAL
            ),
            ResolutionOption(
                id=f"{conflict_id}_merge",
                description=merge_description,
                resolution_type=ResolutionStrategy.MERGE_VALUES,
                resolution_data={
                    "property_name": property_name,
                    "strategy": merge_strategy,
                    "staging_value": staging_value,
                    "production_value": production_value
                },
                confidence=0.4,
                auto_resolvable=severity != ConflictSeverity.CRITICAL
            )
        ]
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=severity,
            staging_ids=[staging_node.id],
            production_ids=[production_node.id],
            description=f"Property '{property_name}' has different values",
            context={
                "property_name": property_name,
                "staging_value": staging_value,
                "production_value": production_value,
                "entity_type": staging_node.label,
                "value_type": type(staging_value).__name__,
                "merge_strategy": merge_strategy
            },
            resolution_options=resolution_options
        )
        
    def _create_missing_property_conflict(
        self,
        staging_node: Node,
        production_node: Node,
        property_name: str,
        missing_in: str
    ) -> Conflict:
        """Create a conflict for a missing property"""
        conflict_id = f"missing_prop_{staging_node.id}_{production_node.id}_{property_name}"
        
        # Get the value that exists
        value = (production_node.properties.get(property_name)
                if missing_in == "staging"
                else staging_node.properties.get(property_name))
                
        severity = ConflictSeverity.MAJOR
        if property_name in {"id", "name", "key", "identifier"}:
            severity = ConflictSeverity.CRITICAL
            
        description = (
            f"Property '{property_name}' exists in production but not in staging"
            if missing_in == "staging"
            else f"Property '{property_name}' exists in staging but not in production"
        )
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.PROPERTY,
            severity=severity,
            staging_ids=[staging_node.id],
            production_ids=[production_node.id],
            description=description,
            context={
                "property_name": property_name,
                "missing_in": missing_in,
                "value": value,
                "entity_type": staging_node.label,
                "value_type": type(value).__name__ if value is not None else None
            },
            resolution_options=[
                ResolutionOption(
                    id=f"{conflict_id}_add",
                    description=f"Add missing property with value: {value}",
                    resolution_type=(ResolutionStrategy.KEEP_PRODUCTION 
                                   if missing_in == "staging"
                                   else ResolutionStrategy.KEEP_STAGING),
                    resolution_data={
                        "property_name": property_name,
                        "value": value
                    },
                    confidence=0.8,
                    auto_resolvable=severity != ConflictSeverity.CRITICAL
                ),
                ResolutionOption(
                    id=f"{conflict_id}_ignore",
                    description="Ignore missing property",
                    resolution_type=ResolutionStrategy.IGNORE,
                    confidence=0.3,
                    auto_resolvable=True
                )
            ]
        )

    async def detect_property_conflicts(
        self,
        staging_node: Node,
        production_node: Node
    ) -> List[Conflict]:
        """Detect property conflicts between two nodes"""
        conflicts = []
        
        # Get all property names
        all_props = set(staging_node.properties.keys()) | set(production_node.properties.keys())
        
        for prop_name in all_props:
            staging_value = staging_node.properties.get(prop_name)
            prod_value = production_node.properties.get(prop_name)
            
            # Skip if values are identical
            if staging_value == prod_value:
                continue
                
            # Create property conflict
            conflict = await self._create_property_conflict(
                staging_node,
                production_node,
                prop_name,
                staging_value,
                prod_value
            )
            if conflict:
                conflicts.append(conflict)
                
        return conflicts
        
    async def detect_relationship_conflicts(
        self,
        staging_edge: Edge,
        production_edge: Edge
    ) -> List[Conflict]:
        """Detect conflicts between two relationships"""
        conflicts = []
        
        # Check relationship type conflicts
        if staging_edge.type != production_edge.type:
            conflicts.append(await self._create_relationship_type_conflict(
                staging_edge,
                production_edge
            ))
            
        # Check property conflicts
        all_props = set(staging_edge.properties.keys()) | set(production_edge.properties.keys())
        
        for prop_name in all_props:
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
                prod_value
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
        production_nodes: List[Node]
    ) -> List[Conflict]:
        """Detect conflicts where entity types differ between staging and production"""
        conflicts = []
        
        for production_node in production_nodes:
            if staging_node.label != production_node.label:
                conflict_id = f"type_{staging_node.id}_{production_node.id}"
                conflicts.append(Conflict(
                    id=conflict_id,
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
        production_nodes: List[Node]
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

    async def _detect_relationship_conflicts(
        self,
        staging_edges: List[Edge],
        production_matches: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect conflicts between staging and production relationships"""
        conflicts = []
        
        # Get staging node IDs that have production matches
        staging_ids_with_matches = {
            id for id, matches in production_matches.items() if matches
        }
        
        # Process each staging edge where both nodes have matches
        for edge in staging_edges:
            if (edge.source not in staging_ids_with_matches or 
                edge.target not in staging_ids_with_matches):
                continue
                
            # Get production matches for source and target
            source_matches = production_matches[edge.source]
            target_matches = production_matches[edge.target]
            
            # Check each possible production relationship
            for prod_source_id in source_matches:
                for prod_target_id in target_matches:
                    edge_conflicts = await self._check_relationship_conflicts(
                        edge, prod_source_id, prod_target_id
                    )
                    conflicts.extend(edge_conflicts)
                    
        return conflicts
        
    async def _check_relationship_conflicts(
        self,
        staging_edge: Edge,
        prod_source_id: str,
        prod_target_id: str
    ) -> List[Conflict]:
        """Check for conflicts between a staging relationship and potential production matches"""
        conflicts = []
        
        # Get production relationships between source and target
        prod_relationships = await self.storage.get_relationships_between(
            prod_source_id, prod_target_id
        )
        
        # Check for missing relationships
        if not prod_relationships:
            conflicts.append(
                self._create_missing_relationship_conflict(
                    staging_edge, prod_source_id, prod_target_id
                )
            )
            return conflicts
            
        # Check for relationship type and property conflicts
        for prod_rel in prod_relationships:
            if prod_rel.type != staging_edge.type:
                conflicts.append(
                    self._create_relationship_type_conflict(
                        staging_edge, prod_rel
                    )
                )
            else:
                # Same type, check properties
                prop_conflicts = self._detect_relationship_property_conflicts(
                    staging_edge, prod_rel
                )
                conflicts.extend(prop_conflicts)
                
        # Check for reverse relationships (direction conflicts)
        reverse_relationships = await self.storage.get_relationships_between(
            prod_target_id, prod_source_id
        )
        
        for reverse_rel in reverse_relationships:
            if reverse_rel.type == staging_edge.type:
                conflicts.append(
                    self._create_relationship_direction_conflict(
                        staging_edge, reverse_rel
                    )
                )
                
        return conflicts
        
    def _create_relationship_type_conflict(
        self,
        staging_edge: Edge,
        production_edge: Edge
    ) -> Conflict:
        """Create a relationship type conflict"""
        conflict_id = f"rel_type_{staging_edge.id}_{production_edge.id}"
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.CRITICAL,
            staging_ids=[staging_edge.id],
            production_ids=[production_edge.id],
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
        
    def _create_relationship_direction_conflict(
        self,
        staging_edge: Edge,
        production_edge: Edge
    ) -> Conflict:
        """Create a relationship direction conflict"""
        conflict_id = f"rel_dir_{staging_edge.id}_{production_edge.id}"
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.RELATIONSHIP_DIRECTION,
            severity=ConflictSeverity.CRITICAL,
            staging_ids=[staging_edge.id],
            production_ids=[production_edge.id],
            description="Relationship direction mismatch",
            context={
                "relationship_type": staging_edge.type,
                "staging_source": staging_edge.source,
                "staging_target": staging_edge.target,
                "production_source": production_edge.source,
                "production_target": production_edge.target,
                "staging_props": staging_edge.properties,
                "production_props": production_edge.properties
            },
            resolution_options=[
                ResolutionOption(
                    id=f"{conflict_id}_staging",
                    description="Keep staging direction",
                    resolution_type=ResolutionStrategy.KEEP_STAGING_REL,
                    resolution_data={"edge_id": staging_edge.id},
                    confidence=0.5,
                    auto_resolvable=False
                ),
                ResolutionOption(
                    id=f"{conflict_id}_prod",
                    description="Keep production direction",
                    resolution_type=ResolutionStrategy.KEEP_PRODUCTION_REL,
                    resolution_data={"edge_id": production_edge.id},
                    confidence=0.5,
                    auto_resolvable=False
                ),
                ResolutionOption(
                    id=f"{conflict_id}_reverse",
                    description="Reverse relationship direction",
                    resolution_type=ResolutionStrategy.REVERSE_RELATIONSHIP,
                    resolution_data={
                        "edge_id": staging_edge.id,
                        "new_source": staging_edge.target,
                        "new_target": staging_edge.source
                    },
                    confidence=0.4,
                    auto_resolvable=False
                )
            ]
        )
        
    def _create_missing_relationship_conflict(
        self,
        staging_edge: Edge,
        prod_source_id: str,
        prod_target_id: str
    ) -> Conflict:
        """Create a conflict for a relationship that exists in staging but not production"""
        conflict_id = f"rel_missing_{staging_edge.id}"
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.RELATIONSHIP_MISSING,
            severity=ConflictSeverity.MAJOR,
            staging_ids=[staging_edge.id],
            production_ids=[],
            description=(
                f"Relationship '{staging_edge.type}' exists in staging "
                f"but not between matching production entities"
            ),
            context={
                "relationship_type": staging_edge.type,
                "staging_source": staging_edge.source,
                "staging_target": staging_edge.target,
                "production_source": prod_source_id,
                "production_target": prod_target_id,
                "staging_props": staging_edge.properties
            },
            resolution_options=[
                ResolutionOption(
                    id=f"{conflict_id}_create",
                    description="Create relationship in production",
                    resolution_type=ResolutionStrategy.KEEP_STAGING_REL,
                    resolution_data={
                        "edge_id": staging_edge.id,
                        "prod_source": prod_source_id,
                        "prod_target": prod_target_id
                    },
                    confidence=0.7,
                    auto_resolvable=True
                ),
                ResolutionOption(
                    id=f"{conflict_id}_ignore",
                    description="Ignore (don't create relationship)",
                    resolution_type=ResolutionStrategy.IGNORE,
                    resolution_data={},
                    confidence=0.3,
                    auto_resolvable=True
                )
            ]
        )
        
    def _detect_relationship_property_conflicts(
        self,
        staging_edge: Edge,
        production_edge: Edge
    ) -> List[Conflict]:
        """Detect property conflicts between relationships"""
        conflicts = []
        
        # Get all property names
        all_props = set(staging_edge.properties.keys()) | set(production_edge.properties.keys())
        
        for prop in all_props:
            staging_value = staging_edge.properties.get(prop)
            prod_value = production_edge.properties.get(prop)
            
            # Missing property
            if staging_value is None or prod_value is None:
                conflicts.append(
                    self._create_relationship_property_missing_conflict(
                        staging_edge, production_edge, prop,
                        staging_value, prod_value
                    )
                )
                continue
                
            # Value mismatch
            if not self._are_property_values_equal(staging_value, prod_value):
                conflicts.append(
                    self._create_relationship_property_conflict(
                        staging_edge, production_edge, prop,
                        staging_value, prod_value
                    )
                )
                
        return conflicts
        
    def _create_relationship_property_conflict(
        self,
        staging_edge: Edge,
        production_edge: Edge,
        property_name: str,
        staging_value: Any,
        production_value: Any
    ) -> Conflict:
        """Create a relationship property value conflict"""
        conflict_id = f"rel_prop_{staging_edge.id}_{production_edge.id}_{property_name}"
        
        # Get BAML analysis
        analysis = self._analyze_property_conflict(
            staging_edge, production_edge,
            property_name, staging_value, production_value
        )
        
        return Conflict(
            id=conflict_id,
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
        
    def _create_relationship_property_missing_conflict(
        self,
        staging_edge: Edge,
        production_edge: Edge,
        property_name: str,
        staging_value: Any,
        production_value: Any
    ) -> Conflict:
        """Create a relationship property missing conflict"""
        conflict_id = f"rel_prop_missing_{staging_edge.id}_{production_edge.id}_{property_name}"
        
        is_staging_missing = staging_value is None
        existing_value = production_value if is_staging_missing else staging_value
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.RELATIONSHIP_PROPERTY,
            severity=ConflictSeverity.MINOR,
            staging_ids=[staging_edge.id],
            production_ids=[production_edge.id],
            description=(
                f"Relationship property '{property_name}' exists in "
                f"{'production' if is_staging_missing else 'staging'} but not in "
                f"{'staging' if is_staging_missing else 'production'}"
            ),
            context={
                "relationship_type": staging_edge.type,
                "property_name": property_name,
                "is_staging_missing": is_staging_missing,
                "existing_value": existing_value
            },
            resolution_options=[
                ResolutionOption(
                    id=f"{conflict_id}_keep",
                    description=f"Keep existing value: {existing_value}",
                    resolution_type=(
                        ResolutionStrategy.KEEP_PRODUCTION_REL if is_staging_missing
                        else ResolutionStrategy.KEEP_STAGING_REL
                    ),
                    resolution_data={
                        "edge_id": (
                            production_edge.id if is_staging_missing
                            else staging_edge.id
                        ),
                        "property_name": property_name,
                        "value": existing_value
                    },
                    confidence=0.7,
                    auto_resolvable=True
                ),
                ResolutionOption(
                    id=f"{conflict_id}_remove",
                    description=f"Remove property '{property_name}'",
                    resolution_type=ResolutionStrategy.IGNORE,
                    resolution_data={
                        "edge_id": (
                            staging_edge.id if is_staging_missing
                            else production_edge.id
                        ),
                        "property_name": property_name
                    },
                    confidence=0.3,
                    auto_resolvable=True
                )
            ]
        )
        
    async def detect_entity_matching_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect conflicts where staging entities match multiple production entities
        
        Args:
            staging_graph: Graph containing staging entities
            production_entity_mapping: Mapping of staging IDs to matching production IDs
            
        Returns:
            List of entity matching conflicts
        """
        conflicts = []
        
        # Process matches in parallel for better performance
        async def process_match(staging_id: str, prod_matches: List[str]) -> Optional[Conflict]:
            # Skip if no matches or only one match
            if len(prod_matches) <= 1:
                return None
                
            # Get staging entity
            staging_entity = next((n for n in staging_graph.nodes if n.id == staging_id), None)
            if not staging_entity:
                return None
                
            # Get production entities
            prod_entities = []
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self.prod_storage.get_node_by_id(prod_id))
                    for prod_id in prod_matches
                ]
            prod_entities = [t.result() for t in tasks if t.result()]
            
            # Skip if we couldn't find production entities
            if not prod_entities:
                return None
            
            # Calculate similarity scores using BAML
            similarity_scores = {}
            analyses = {}
            
            async with asyncio.TaskGroup() as tg:
                tasks = {
                    prod_entity.id: tg.create_task(
                        self._analyze_entity_similarity(staging_entity, prod_entity)
                    )
                    for prod_entity in prod_entities
                }
            
            for prod_id, task in tasks.items():
                analysis = task.result()
                similarity_scores[prod_id] = analysis.similarity_score
                analyses[prod_id] = analysis
            
            # Create conflict with rich context
            return await self._create_entity_matching_conflict(
                staging_entity, prod_entities, similarity_scores, analyses
            )
        
        # Process all matches in parallel
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(process_match(staging_id, prod_matches))
                for staging_id, prod_matches in production_entity_mapping.items()
            ]
        
        # Collect non-None results
        conflicts = [t.result() for t in tasks if t.result()]
        return conflicts

    async def _analyze_entity_similarity(
        self,
        staging_entity: Node,
        production_entity: Node
    ) -> EntitySimilarityAnalysis:
        """Analyze similarity between two entities using BAML"""
        return b.AnalyzeEntitySimilarity(
            entity_type=staging_entity.label,
            staging_properties=json.dumps(staging_entity.properties, indent=2),
            production_properties=json.dumps(production_entity.properties, indent=2),
            domain_context=f"Entity type: {staging_entity.label}"
        )

    async def _create_entity_matching_conflict(
        self,
        staging_entity: Node,
        production_entities: List[Node],
        similarity_scores: Dict[str, float],
        analyses: Dict[str, EntitySimilarityAnalysis]
    ) -> Conflict:
        """Create an entity matching conflict with rich context"""
        conflict_id = f"entity_match_{staging_entity.id}"
        
        # Create resolution options
        resolution_options = []
        
        # Option to match with each production entity
        for prod_entity in production_entities:
            analysis = analyses[prod_entity.id]
            similarity = similarity_scores[prod_entity.id]
            
            resolution_options.append(
                ResolutionOption(
                    id=f"{conflict_id}_match_{prod_entity.id}",
                    description=(
                        f"Match with production entity {prod_entity.id} "
                        f"(similarity: {similarity:.2f})"
                    ),
                    resolution_type=ResolutionStrategy.MERGE_VALUES,
                    resolution_data={
                        "production_id": prod_entity.id,
                        "merge_properties": analysis.matching_properties
                    },
                    confidence=similarity
                )
            )
        
        # Option to merge all production entities if they're similar enough
        avg_similarity = sum(similarity_scores.values()) / len(similarity_scores)
        if avg_similarity > 0.7:  # Only suggest merge if entities are similar
            resolution_options.append(
                ResolutionOption(
                    id=f"{conflict_id}_merge_all",
                    description="Merge all matching production entities",
                    resolution_type=ResolutionStrategy.MERGE_VALUES,
                    resolution_data={
                        "production_ids": [e.id for e in production_entities],
                        "merge_strategy": "combine_properties"
                    },
                    confidence=0.3
                )
            )
        
        # Option to create new entity
        resolution_options.append(
            ResolutionOption(
                id=f"{conflict_id}_create_new",
                description="Create new production entity",
                resolution_type=ResolutionStrategy.CREATE_NEW,
                resolution_data={},
                confidence=0.2
            )
        )
        
        # Build rich context with analysis details
        context = {
            "staging_entity": {
                "id": staging_entity.id,
                "label": staging_entity.label,
                "properties": staging_entity.properties
            },
            "production_entities": [
                {
                    "id": e.id,
                    "label": e.label,
                    "properties": e.properties,
                    "similarity": similarity_scores[e.id],
                    "analysis": {
                        "matching_properties": analyses[e.id].matching_properties,
                        "mismatched_properties": analyses[e.id].mismatched_properties,
                        "semantic_similarity": analyses[e.id].semantic_similarity,
                        "potential_impact": analyses[e.id].potential_merge_impact,
                        "reasoning": analyses[e.id].reasoning
                    }
                }
                for e in production_entities
            ],
            "similarity_scores": similarity_scores,
            "average_similarity": avg_similarity
        }
        
        return Conflict(
            id=conflict_id,
            conflict_type=ConflictType.DUPLICATE_ENTITY,
            severity=ConflictSeverity.CRITICAL,
            staging_ids=[staging_entity.id],
            production_ids=[e.id for e in production_entities],
            description=f"Staging entity matches multiple production entities",
            context=context,
            resolution_options=resolution_options
        )

    def _group_nodes_by_type(self, nodes: List[Node]) -> Dict[str, List[Node]]:
        """Group nodes by their type/label"""
        grouped = {}
        for node in nodes:
            if node.label not in grouped:
                grouped[node.label] = []
            grouped[node.label].append(node)
        return grouped
