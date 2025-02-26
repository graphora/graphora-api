"""Service for handling merge operations"""
import uuid
import time
import logging
import asyncio
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from datetime import datetime
import pytz
import json
import redis.asyncio as redis
from app.config import settings
from app.schemas.conflicts import Conflict, ConflictSeverity, ConflictType
from app.services.storage.interface import GraphStorageInterface
from app.services.storage.models import Node, Edge
from app.schemas.graph import GraphResponse
from app.services.merge.conflict import ConflictDetectionService
from app.services.merge.models import (
    MergeStage,
    MergeProgress,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationIssueType,
    EntityMappingResult,
    EntityMatch,
    MatchStrategy
)
from app.services.merge.progress import ProgressTracker
from prefect import flow, task
from app.services.merge.resolution_pipeline import build_resolution_pipeline
from app.services.merge.llm_analyzer import LLMConflictAnalyzer

logger = logging.getLogger(__name__)

async def get_redis_client():
    """Get Redis client instance"""
    return redis.Redis.from_url(settings.REDIS_URL)

@task(name="extract_staging_graph")
async def extract_staging_graph(
    storage: GraphStorageInterface,
    transform_id: str
) -> GraphResponse:
    """
    Extract complete graph with transform_id from staging database.
    
    Args:
        storage: Graph storage interface
        transform_id: ID of the transform to extract nodes for
        
    Returns:
        GraphResponse containing nodes and relationships
    """
    try:
        start_time = time.time()
        
        # Extract nodes with transform_id
        nodes = await storage.get_nodes_by_property(
            property_name="transform_id",
            property_value=transform_id
        )
        
        if not nodes:
            logger.warning(f"No nodes found with transform_id {transform_id}")
            return GraphResponse(
                nodes=[],
                edges=[],
                total_nodes=0,
                total_edges=0
            )
        
        # Extract relationships between these nodes
        node_ids = [node.id for node in nodes]
        edges = await storage.get_relationships_between_nodes(node_ids)
        
        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Extracted {len(nodes)} nodes and {len(edges)} relationships "
            f"in {duration_ms:.2f}ms"
        )
        
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            total_nodes=len(nodes),
            total_edges=len(edges)
        )
        
    except Exception as e:
        logger.error(f"Failed to extract staging graph: {str(e)}")
        raise

@task(name="map_production_entities")
async def map_production_entities(
    storage: GraphStorageInterface,
    staging_graph: GraphResponse,
    similarity_threshold: float = 0.7
) -> EntityMappingResult:
    """
    Map staging entities to potential production matches
    
    Args:
        storage: Graph storage interface
        staging_graph: Graph extracted from staging
        similarity_threshold: Minimum similarity score for fuzzy matching
        
    Returns:
        EntityMappingResult containing mapping details
    """
    try:
        start_time = time.time()
        matches = {}
        matched_count = 0
        
        for node in staging_graph.nodes:
            match = await get_matching_nodes(
                storage,
                node,
                similarity_threshold
            )
            matches[node.id] = match
            if match.production_matches:
                matched_count += 1
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Mapped {matched_count}/{len(staging_graph.nodes)} entities "
            f"in {duration_ms:.2f}ms"
        )
        
        return EntityMappingResult(
            matches=matches,
            total_entities=len(staging_graph.nodes),
            matched_entities=matched_count,
            mapping_time_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Failed to map production entities: {str(e)}")
        raise

async def get_matching_nodes(
    storage: GraphStorageInterface,
    node: Node,
    similarity_threshold: float
) -> EntityMatch:
    """Find matching nodes in production based on node type and properties"""
    try:
        # Start with most specific matching strategy
        strategy = MatchStrategy.EXACT_ID
        matches = []
        
        # Try ID-based matching first
        if "id" in node.properties:
            matches = await storage.find_nodes_by_property_value(
                label=node.label,
                property_name="id",
                property_value=node.properties["id"]
            )
            
        # Fall back to name-based matching
        if not matches and "name" in node.properties:
            strategy = MatchStrategy.EXACT_NAME
            matches = await storage.find_nodes_by_property_value(
                label=node.label,
                property_name="name",
                property_value=node.properties["name"]
            )
            
        # Use property similarity as last resort
        if not matches:
            strategy = MatchStrategy.PROPERTY_SIMILARITY
            matches = await storage.find_similar_nodes(
                label=node.label,
                properties=node.properties,
                similarity_threshold=similarity_threshold
            )
        
        # Calculate confidence based on strategy and number of matches
        confidence = 1.0 if strategy == MatchStrategy.EXACT_ID else (
            0.8 if strategy == MatchStrategy.EXACT_NAME else
            0.5  # Property similarity
        )
        
        return EntityMatch(
            staging_id=node.id,
            production_matches=[m.id for m in matches],
            match_confidence=confidence,
            match_strategy=strategy,
            metadata={
                "total_matches": len(matches),
                "node_label": node.label
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get matching nodes: {str(e)}")
        raise

@task(name="validate_graph")
async def validate_graph(
    graph: GraphResponse,
    ontology_id: str,
    ontology_dir: str = settings.ONTOLOGY_DIR
) -> ValidationResult:
    """
    Validate graph against ontology requirements.
    
    Args:
        graph: Graph to validate
        ontology_id: ID of the ontology to validate against
        ontology_dir: Directory containing ontology files
        
    Returns:
        ValidationResult containing validation details
    """
    try:
        start_time = time.time()
        
        # Load ontology
        ontology = await load_ontology(ontology_id, ontology_dir)
        
        issues = []
        critical_count = 0
        warning_count = 0
        info_count = 0
        
        # Check nodes against ontology definitions
        for node in graph.nodes:
            node_issues = validate_node(node, ontology)
            issues.extend(node_issues)
            
            critical_count += sum(1 for i in node_issues if i.severity == ValidationSeverity.CRITICAL)
            warning_count += sum(1 for i in node_issues if i.severity == ValidationSeverity.WARNING)
            info_count += sum(1 for i in node_issues if i.severity == ValidationSeverity.INFO)
        
        # Check for orphaned nodes
        orphaned_nodes = find_orphaned_nodes(graph)
        if orphaned_nodes:
            issues.append(ValidationIssue(
                type=ValidationIssueType.ORPHANED_NODE,
                message=f"Found {len(orphaned_nodes)} nodes with no relationships",
                affected_ids=orphaned_nodes,
                severity=ValidationSeverity.WARNING,
                metadata={"count": len(orphaned_nodes)}
            ))
            warning_count += 1
        
        # Check relationship constraints
        relationship_issues = validate_relationships(graph, ontology)
        issues.extend(relationship_issues)
        
        critical_count += sum(1 for i in relationship_issues if i.severity == ValidationSeverity.CRITICAL)
        warning_count += sum(1 for i in relationship_issues if i.severity == ValidationSeverity.WARNING)
        info_count += sum(1 for i in relationship_issues if i.severity == ValidationSeverity.INFO)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            valid=critical_count == 0,
            issues=issues,
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            total_nodes=len(graph.nodes),
            total_edges=len(graph.edges),
            validation_time_ms=duration_ms,
            metadata={
                "ontology_id": ontology_id
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to validate graph: {str(e)}")
        raise

async def load_ontology(ontology_id: str, ontology_dir: str) -> Dict[str, Any]:
    """Load ontology definition from file"""
    try:
        ontology_path = Path(ontology_dir).expanduser() / f"{ontology_id}.yaml"
        
        if not ontology_path.exists():
            raise ValueError(f"Ontology {ontology_id} not found at {ontology_path}")
            
        with open(ontology_path, 'r') as f:
            return yaml.safe_load(f)
            
    except Exception as e:
        logger.error(f"Failed to load ontology: {str(e)}")
        raise

def find_orphaned_nodes(graph: GraphResponse) -> List[str]:
    """Find nodes that have no relationships"""
    # Build set of all nodes that participate in relationships
    connected_nodes: Set[str] = set()
    for edge in graph.edges:
        connected_nodes.add(edge.source)
        connected_nodes.add(edge.target)
    
    # Find nodes that are not in connected_nodes
    return [
        node.id for node in graph.nodes 
        if node.id not in connected_nodes
    ]

def validate_node(node: Node, ontology: Dict[str, Any]) -> List[ValidationIssue]:
    """Validate a node against ontology definition"""
    issues = []
    
    # Get entity definition from ontology
    entity_def = ontology.get('entities', {}).get(node.label)
    if not entity_def:
        issues.append(ValidationIssue(
            type=ValidationIssueType.UNKNOWN_ENTITY_TYPE,
            message=f"Entity type {node.label} not defined in ontology",
            affected_ids=[node.id],
            severity=ValidationSeverity.WARNING
        ))
        return issues
    
    # Check required properties
    required_props = [
        prop_name for prop_name, prop_def 
        in entity_def.get('properties', {}).items()
        if prop_def.get('required', False)
    ]
    
    missing_props = [
        prop for prop in required_props 
        if prop not in node.properties
    ]
    
    if missing_props:
        issues.append(ValidationIssue(
            type=ValidationIssueType.MISSING_REQUIRED_PROPERTIES,
            message=f"Node {node.id} missing required properties: {', '.join(missing_props)}",
            affected_ids=[node.id],
            severity=ValidationSeverity.CRITICAL,
            metadata={"missing_properties": missing_props}
        ))
    
    # Validate property types
    for prop_name, prop_value in node.properties.items():
        prop_def = entity_def.get('properties', {}).get(prop_name)
        if prop_def and 'type' in prop_def:
            if not validate_property_type(prop_value, prop_def['type']):
                issues.append(ValidationIssue(
                    type=ValidationIssueType.INVALID_PROPERTY_TYPE,
                    message=f"Property {prop_name} has invalid type. Expected {prop_def['type']}",
                    affected_ids=[node.id],
                    severity=ValidationSeverity.CRITICAL,
                    metadata={
                        "property": prop_name,
                        "expected_type": prop_def['type'],
                        "actual_type": type(prop_value).__name__
                    }
                ))
    
    return issues

def validate_property_type(value: Any, expected_type: str) -> bool:
    """Validate property value against expected type"""
    type_mapping = {
        'string': str,
        'integer': int,
        'float': (int, float),
        'boolean': bool,
        'datetime': str,  # ISO format string
        'array': list,
        'object': dict
    }
    
    if expected_type not in type_mapping:
        return True  # Skip validation for unknown types
        
    expected_python_type = type_mapping[expected_type]
    
    if isinstance(expected_python_type, tuple):
        return isinstance(value, expected_python_type)
    return isinstance(value, expected_python_type)

def validate_relationships(
    graph: GraphResponse,
    ontology: Dict[str, Any]
) -> List[ValidationIssue]:
    """Validate relationships against ontology constraints"""
    issues = []
    
    # Check relationship types
    relationship_types = ontology.get('relationships', {})
    for edge in graph.edges:
        if edge.type not in relationship_types:
            issues.append(ValidationIssue(
                type=ValidationIssueType.INVALID_RELATIONSHIP_TYPE,
                message=f"Invalid relationship type: {edge.type}",
                affected_ids=[edge.id],
                severity=ValidationSeverity.WARNING,
                metadata={"relationship_type": edge.type}
            ))
            continue
            
        # Check source and target node types
        rel_def = relationship_types[edge.type]
        source_node = next((n for n in graph.nodes if n.id == edge.source), None)
        target_node = next((n for n in graph.nodes if n.id == edge.target), None)
        
        if source_node and source_node.label not in rel_def.get('source_types', []):
            issues.append(ValidationIssue(
                type=ValidationIssueType.INVALID_RELATIONSHIP_TYPE,
                message=f"Invalid source node type for relationship {edge.type}",
                affected_ids=[edge.id, edge.source],
                severity=ValidationSeverity.CRITICAL,
                metadata={
                    "relationship_type": edge.type,
                    "node_type": source_node.label,
                    "allowed_types": rel_def.get('source_types', [])
                }
            ))
            
        if target_node and target_node.label not in rel_def.get('target_types', []):
            issues.append(ValidationIssue(
                type=ValidationIssueType.INVALID_RELATIONSHIP_TYPE,
                message=f"Invalid target node type for relationship {edge.type}",
                affected_ids=[edge.id, edge.target],
                severity=ValidationSeverity.CRITICAL,
                metadata={
                    "relationship_type": edge.type,
                    "node_type": target_node.label,
                    "allowed_types": rel_def.get('target_types', [])
                }
            ))
    
    # Check required relationships
    for node in graph.nodes:
        entity_def = ontology.get('entities', {}).get(node.label)
        if not entity_def:
            continue
            
        required_rels = entity_def.get('required_relationships', [])
        for rel_def in required_rels:
            rel_type = rel_def['type']
            direction = rel_def.get('direction', 'outgoing')
            
            # Check if required relationship exists
            has_required_rel = any(
                edge.type == rel_type and
                (direction == 'outgoing' and edge.source == node.id or
                 direction == 'incoming' and edge.target == node.id)
                for edge in graph.edges
            )
            
            if not has_required_rel:
                issues.append(ValidationIssue(
                    type=ValidationIssueType.MISSING_REQUIRED_RELATIONSHIP,
                    message=f"Missing required {direction} relationship of type {rel_type}",
                    affected_ids=[node.id],
                    severity=ValidationSeverity.CRITICAL,
                    metadata={
                        "relationship_type": rel_type,
                        "direction": direction,
                        "node_type": node.label
                    }
                ))
    
    return issues

@task(name="start_merge_stage", retries=2)
async def start_stage(
    merge_id: str,
    stage: MergeStage,
    progress_tracker: ProgressTracker
) -> None:
    """Start a merge stage with progress tracking"""
    await progress_tracker.start_merge_stage(merge_id, stage)

@task(name="complete_merge_stage", retries=2)
async def complete_merge_stage(
    merge_id: str,
    stage: MergeStage,
    progress_tracker: ProgressTracker,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Complete a merge stage with optional metadata
    
    Args:
        merge_id: ID of the merge operation
        stage: Stage that was completed
        progress_tracker: Progress tracking instance
        metadata: Optional metadata about the stage completion
    """
    if not metadata:
        metadata = {}
        
    # Update stage completion time
    metadata['completed_at'] = datetime.now(pytz.utc).isoformat()
    
    # Mark stage as complete
    await progress_tracker.complete_merge_stage(
        merge_id,
        stage,
        metadata
    )
    
    logger.info(f"Completed merge stage {stage} for merge {merge_id}")

@task(name="fail_merge", retries=2)
async def fail_merge(
    merge_id: str,
    error_msg: str,
    progress_tracker: ProgressTracker,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Mark merge as failed with error details"""
    await progress_tracker.fail_merge(
        merge_id,
        error_msg,
        metadata=metadata
    )

@task(name="complete_merge", retries=2)
async def complete_merge(
    merge_id: str,
    progress_tracker: ProgressTracker
) -> None:
    """Mark merge as complete"""
    await progress_tracker.complete_merge_stage(merge_id, MergeStage.MERGE)

@task(name="detect_conflicts")
async def detect_merge_conflicts(
    merge_id: str,
    graph: GraphResponse,
    entity_mapping: EntityMappingResult,
    storage: GraphStorageInterface,
    progress_tracker: ProgressTracker
) -> None:
    """Detect conflicts between staging and production graphs"""
    try:
        # Start conflict detection stage
        await progress_tracker.start_merge_stage(merge_id, MergeStage.CONFLICT_DETECTION)
        
        # Initialize conflict detection service
        conflict_service = ConflictDetectionService(storage)
        
        # Create production entity mapping dict
        production_entity_mapping = {
            staging_id: match.production_matches
            for staging_id, match in entity_mapping.matches.items()
        }
        
        # Detect conflicts
        conflict_batch = await conflict_service.detect_conflicts(
            merge_id=merge_id,
            staging_nodes=graph.nodes,
            staging_edges=graph.edges,
            production_matches=production_entity_mapping
        )
        
        # Complete conflict detection stage
        await progress_tracker.complete_merge_stage(
            merge_id,
            MergeStage.CONFLICT_DETECTION,
            metadata={
                "total_conflicts": conflict_batch.total_conflicts,
                "batch_id": conflict_batch.batch_id,
                "conflict_types": {
                    conflict_type.value: len([
                        c for c in conflict_batch.conflicts
                        if c.conflict_type == conflict_type
                    ])
                    for conflict_type in ConflictType
                },
                "severities": {
                    severity.value: len([
                        c for c in conflict_batch.conflicts
                        if c.severity == severity
                    ])
                    for severity in ConflictSeverity
                }
            }
        )
        
    except Exception as e:
        error_msg = f"Failed to detect conflicts: {str(e)}"
        logger.error(error_msg)
        await progress_tracker.fail_merge(merge_id, error_msg)
        raise

@flow(name="graph-merge-flow")
async def merge_flow(
    merge_id: str,
    session_id: str,
    transform_id: str,
    storage: GraphStorageInterface,
    progress_tracker: ProgressTracker
) -> None:
    """
    Prefect flow for graph merge process.
    
    Breaks down the merge process into smaller tasks for better
    observability and retry capabilities.
    """
    try:
        # Extract Stage
        await start_stage(merge_id, MergeStage.EXTRACT, progress_tracker)
        
        graph = await extract_staging_graph.with_options(
            name="extract_staging_graph",
            retries=3,
            retry_delay_seconds=5
        )(storage, transform_id)
        
        await complete_merge_stage(
            merge_id,
            MergeStage.EXTRACT,
            progress_tracker,
            {
                "total_nodes": graph.total_nodes,
                "total_edges": graph.total_edges
            }
        )
        
        # Analyze Stage
        await start_stage(merge_id, MergeStage.ANALYZE, progress_tracker)
        
        # Run mapping and validation in parallel
        mapping_task = map_production_entities.with_options(
            name="map_production_entities",
            retries=2,
            retry_delay_seconds=5
        )(storage, graph)
        
        validation_task = validate_graph.with_options(
            name="validate_graph",
            retries=2,
            retry_delay_seconds=5
        )(graph, session_id, settings.ONTOLOGY_DIR)
        
        # Wait for both tasks to complete
        mapping, validation = await asyncio.gather(mapping_task, validation_task)
        
        await complete_merge_stage(
            merge_id,
            MergeStage.ANALYZE,
            progress_tracker,
            {
                "total_entities": mapping.total_entities,
                "matched_entities": mapping.matched_entities,
                "mapping_time_ms": mapping.mapping_time_ms,
                "validation_result": validation.model_dump(),
                "is_valid": validation.valid
            }
        )
        
        if not validation.valid:
            error_msg = "Graph validation failed with critical issues"
            logger.error(error_msg)
            await fail_merge(
                merge_id,
                error_msg,
                progress_tracker,
                validation.model_dump()
            )
            return
            
        # Conflict Detection Stage
        await detect_merge_conflicts(
            merge_id=merge_id,
            graph=graph,
            entity_mapping=mapping,
            storage=storage,
            progress_tracker=progress_tracker
        )
        
        # Continue with other stages...
        # This will be expanded in subsequent user stories
        
        # Mark merge as complete
        await complete_merge_stage(
            merge_id,
            MergeStage.MERGE,
            progress_tracker
        )
        
    except Exception as e:
        error_msg = f"Merge flow failed: {str(e)}"
        logger.error(error_msg)
        await fail_merge(merge_id, error_msg, progress_tracker)
        raise

class MergeService:
    """Service for managing merge operations"""
    
    def __init__(self, storage: GraphStorageInterface):
        """Initialize merge service"""
        self.storage = storage
        self.progress_tracker = ProgressTracker()
    
    async def start_merge_flow(
        self,
        session_id: str,
        transform_id: str
    ) -> str:
        """
        Start a new merge flow.
        
        Args:
            session_id: ID of the session
            transform_id: ID of the transform to merge
            
        Returns:
            str: Unique merge ID for tracking
        """
        try:
            # Generate unique merge ID
            merge_id = f"merge_{uuid.uuid4().hex}"
            
            # Initialize merge status
            await self.progress_tracker.initialize_merge(merge_id)
            
            # Start flow
            await merge_flow(
                merge_id=merge_id,
                session_id=session_id,
                transform_id=transform_id,
                storage=self.storage,
                progress_tracker=self.progress_tracker
            )
            
            logger.info(f"Started merge flow {merge_id}")
            return merge_id
            
        except Exception as e:
            logger.error(f"Failed to start merge flow: {str(e)}")
            raise
    
    async def get_merge_progress(self, merge_id: str) -> Optional[MergeProgress]:
        """Get current progress of a merge operation"""
        return await self.progress_tracker.get_progress(merge_id)

    async def detect_conflicts(
        self,
        merge_id: str,
        transform_id: str
    ) -> Dict[str, Any]:
        """Run comprehensive conflict detection workflow"""
        # Update progress
        await self.progress_tracker.start_merge_stage(merge_id, MergeStage.CONFLICT_DETECTION)
        
        try:
            # Extract staging graph
            staging_graph = await extract_staging_graph(self.storage, transform_id)
            
            # Get production entity matches
            await self.progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.CONFLICT_DETECTION,
                1, 4,
                {"task": "entity_mapping"}
            )
            entity_mapping_result = await map_production_entities(
                self.storage,
                staging_graph,
                similarity_threshold=0.7
            )
            
            # Convert EntityMappingResult to Dict[str, List[str]] for conflict detection
            production_entity_mapping = {
                match_key: match.production_matches
                for match_key, match in entity_mapping_result.matches.items()
            }
            
            # Detect property conflicts
            await self.progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.CONFLICT_DETECTION,
                2, 4,
                {"task": "property_conflicts"}
            )
            property_conflicts = await self.detect_property_conflicts_for_graph(
                staging_graph, production_entity_mapping
            )
            
            # Detect relationship conflicts
            await self.progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.CONFLICT_DETECTION,
                3, 4,
                {"task": "relationship_conflicts"}
            )
            relationship_conflicts = await self.detect_relationship_conflicts(
                staging_graph, production_entity_mapping
            )
            
            # Detect entity matching conflicts
            await self.progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.CONFLICT_DETECTION,
                4, 4,
                {"task": "entity_matching_conflicts"}
            )
            entity_conflicts = await self.detect_entity_matching_conflicts(
                staging_graph, production_entity_mapping
            )
            
            # Combine all conflicts
            all_conflicts = property_conflicts + relationship_conflicts + entity_conflicts
            
            # Store conflicts
            await self._store_conflicts(merge_id, all_conflicts)
            
            # Mark stage as complete
            await self.progress_tracker.complete_merge_stage(merge_id, MergeStage.CONFLICT_DETECTION)
            
            # Return summary
            return {
                "total_conflicts": len(all_conflicts),
                "property_conflicts": len(property_conflicts),
                "relationship_conflicts": len(relationship_conflicts),
                "entity_conflicts": len(entity_conflicts),
                "critical_conflicts": sum(1 for c in all_conflicts if c.severity == ConflictSeverity.CRITICAL),
                "major_conflicts": sum(1 for c in all_conflicts if c.severity == ConflictSeverity.MAJOR),
                "minor_conflicts": sum(1 for c in all_conflicts if c.severity == ConflictSeverity.MINOR)
            }
            
        except Exception as e:
            # Mark stage as failed
            await self.progress_tracker.fail_merge_stage(
                merge_id,
                MergeStage.CONFLICT_DETECTION,
                str(e)
            )
            raise

    async def detect_property_conflicts_for_graph(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect property conflicts for entire graph"""
        conflicts = []
        
        for staging_node in staging_graph.nodes:
            # Skip if no production matches
            if staging_node.id not in production_entity_mapping:
                continue
                
            prod_node_ids = production_entity_mapping[staging_node.id]
            for prod_node_id in prod_node_ids:
                # Get production node
                prod_node = await self.storage.get_node_by_id(prod_node_id)
                if not prod_node:
                    continue
                    
                # Detect conflicts
                node_conflicts = await self.detect_property_conflicts(staging_node, prod_node)
                conflicts.extend(node_conflicts)
        
        return conflicts

    async def detect_property_conflicts(
        self,
        staging_node: Node,
        production_node: Node
    ) -> List[Conflict]:
        """Detect property conflicts between two nodes"""
        # Create conflict detection service
        conflict_service = ConflictDetectionService(
            storage=self.storage)
        
        # Convert storage.models.Node to Node if needed
        if not hasattr(staging_node, 'label') and hasattr(staging_node, 'type'):
            # Convert from storage model to schema model
            staging_node_schema = Node(
                id=staging_node.id,
                label=staging_node.type,
                type=staging_node.type,
                properties=staging_node.properties
            )
        else:
            staging_node_schema = staging_node
            
        if not hasattr(production_node, 'label') and hasattr(production_node, 'type'):
            # Convert from storage model to schema model
            production_node_schema = Node(
                id=production_node.id,
                label=production_node.type,
                type=production_node.type,
                properties=production_node.properties
            )
        else:
            production_node_schema = production_node
        
        # Detect property conflicts
        conflicts = await conflict_service.detect_property_conflicts(
            staging_node_schema,
            production_node_schema
        )
        
        return conflicts

    async def detect_relationship_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect conflicts in relationships between staging and production"""
        conflicts = []
        conflict_service = ConflictDetectionService(self.storage)
        
        # Process each edge in staging graph
        for staging_edge in staging_graph.edges:
            # Skip if either endpoint has no production matches
            if (staging_edge.source_id not in production_entity_mapping or
                staging_edge.target_id not in production_entity_mapping):
                continue
                
            # Get production matches for both endpoints
            source_matches = production_entity_mapping[staging_edge.source_id] 
            target_matches = production_entity_mapping[staging_edge.target_id]
            
            # Check relationships between all possible endpoint combinations
            for source_id in source_matches:
                for target_id in target_matches:
                    # Get production relationships
                    prod_edges = await self.storage.get_relationships_between(
                        source_id,
                        target_id,
                        staging_edge.type
                    )
                    
                    # Detect conflicts for each relationship
                    for prod_edge in prod_edges:
                        # Convert storage.models.Edge to Edge if needed
                        staging_edge_schema = staging_edge
                        prod_edge_schema = prod_edge
                        
                        if not hasattr(staging_edge, 'source') and hasattr(staging_edge, 'from_node'):
                            # Convert from storage model to schema model
                            staging_edge_schema = Edge(
                                id=staging_edge.id,
                                source=staging_edge.from_node,
                                target=staging_edge.to_node,
                                type=staging_edge.type,
                                properties=staging_edge.properties
                            )
                            
                        if not hasattr(prod_edge, 'source') and hasattr(prod_edge, 'from_node'):
                            # Convert from storage model to schema model
                            prod_edge_schema = Edge(
                                id=prod_edge.id,
                                source=prod_edge.from_node,
                                target=prod_edge.to_node,
                                type=prod_edge.type,
                                properties=prod_edge.properties
                            )
                        
                        # Detect conflicts
                        edge_conflicts = await conflict_service.detect_relationship_conflicts(
                            staging_edge_schema,
                            prod_edge_schema
                        )
                        conflicts.extend(edge_conflicts)
        
        return conflicts

    async def detect_entity_matching_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect conflicts where staging entities match multiple production entities"""
        conflicts = []
        conflict_service = ConflictDetectionService(self.storage)
        
        # Process each node with multiple matches
        for staging_id, prod_matches in production_entity_mapping.items():
            if len(prod_matches) <= 1:
                continue
                
            # Get staging node
            staging_node = next(
                (n for n in staging_graph.nodes if n.id == staging_id),
                None
            )
            if not staging_node:
                continue
                
            # Get production nodes
            prod_nodes = []
            for prod_id in prod_matches:
                prod_node = await self.storage.get_node_by_id(prod_id)
                if prod_node:
                    prod_nodes.append(prod_node)
                    
            if len(prod_nodes) > 1:
                # Convert storage.models.Node to Node if needed
                staging_node_schema = staging_node
                prod_nodes_schema = []
                
                if not hasattr(staging_node, 'label') and hasattr(staging_node, 'type'):
                    # Convert from storage model to schema model
                    staging_node_schema = Node(
                        id=staging_node.id,
                        label=staging_node.type,
                        type=staging_node.type,
                        properties=staging_node.properties
                    )
                else:
                    staging_node_schema = staging_node
                    
                for prod_node in prod_nodes:
                    if not hasattr(prod_node, 'label') and hasattr(prod_node, 'type'):
                        # Convert from storage model to schema model
                        prod_nodes_schema.append(Node(
                            id=prod_node.id,
                            label=prod_node.type,
                            type=prod_node.type,
                            properties=prod_node.properties
                        ))
                    else:
                        prod_nodes_schema.append(prod_node)
                
                # Detect entity matching conflicts
                match_conflicts = await conflict_service.detect_entity_matching_conflicts(
                    staging_node_schema,
                    prod_nodes_schema
                )
                conflicts.extend(match_conflicts)
        
        return conflicts

    async def _store_conflicts(self, merge_id: str, conflicts: List[Conflict]) -> None:
        """Store conflicts for later resolution"""
        # Use Redis for storage
        redis_client = await get_redis_client()
        
        # Store each conflict
        for conflict in conflicts:
            key = f"merge:{merge_id}:conflict:{conflict.id}"
            await redis_client.set(key, conflict.model_dump_json())
            
        # Store list of conflict IDs
        conflict_ids = [conflict.id for conflict in conflicts]
        await redis_client.set(f"merge:{merge_id}:conflict_ids", json.dumps(conflict_ids))
        
        # Store counts by type and severity
        counts = {
            "total": len(conflicts),
            "by_type": {},
            "by_severity": {},
            "resolved": 0,
            "unresolved": len(conflicts)
        }
        
        for conflict in conflicts:
            # Count by type
            type_key = conflict.conflict_type.value
            counts["by_type"][type_key] = counts["by_type"].get(type_key, 0) + 1
            
            # Count by severity
            severity_key = conflict.severity.value
            counts["by_severity"][severity_key] = counts["by_severity"].get(severity_key, 0) + 1
        
        await redis_client.set(f"merge:{merge_id}:conflict_counts", json.dumps(counts))
        
        # Set TTL for cleanup (30 days)
        ttl = 30 * 24 * 60 * 60  # 30 days in seconds
        for key in [
            f"merge:{merge_id}:conflict_ids",
            f"merge:{merge_id}:conflict_counts"
        ] + [f"merge:{merge_id}:conflict:{c.id}" for c in conflicts]:
            await redis_client.expire(key, ttl)

    async def get_conflicts(
        self,
        merge_id: str,
        conflict_type: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Conflict], int]:
        """Get conflicts with filtering and pagination"""
        redis_client = await get_redis_client()
        
        # Get list of conflict IDs
        conflict_ids_key = f"merge:{merge_id}:conflict_ids"
        conflict_ids_json = await redis_client.get(conflict_ids_key)
        if not conflict_ids_json:
            return [], 0
            
        conflict_ids = json.loads(conflict_ids_json)
        
        # Get all conflicts
        conflicts = []
        for conflict_id in conflict_ids:
            conflict_key = f"merge:{merge_id}:conflict:{conflict_id}"
            conflict_json = await redis_client.get(conflict_key)
            if conflict_json:
                conflict = Conflict.model_validate_json(conflict_json)
                
                # Apply filters
                if conflict_type and conflict.conflict_type.value != conflict_type:
                    continue
                if severity and conflict.severity.value != severity:
                    continue
                if resolved is not None and conflict.resolved != resolved:
                    continue
                    
                conflicts.append(conflict)
        
        # Get total count after filtering
        total_count = len(conflicts)
        
        # Apply pagination
        paginated_conflicts = conflicts[offset:offset+limit]
        
        return paginated_conflicts, total_count

    async def get_conflict(self, merge_id: str, conflict_id: str) -> Optional[Conflict]:
        """Get a specific conflict by ID"""
        redis_client = await get_redis_client()
        
        conflict_key = f"merge:{merge_id}:conflict:{conflict_id}"
        conflict_json = await redis_client.get(conflict_key)
        if not conflict_json:
            return None
            
        return Conflict.model_validate_json(conflict_json)

    async def get_conflict_summary(self, merge_id: str) -> Dict[str, Any]:
        """Get summary of conflicts by type and severity"""
        redis_client = await get_redis_client()
        
        counts_key = f"merge:{merge_id}:conflict_counts"
        counts_json = await redis_client.get(counts_key)
        if not counts_json:
            return {
                "total": 0,
                "by_type": {},
                "by_severity": {},
                "resolved": 0,
                "unresolved": 0
            }
            
        return json.loads(counts_json)

    async def resolve_conflict(
        self,
        merge_id: str,
        conflict_id: str,
        resolution_id: str
    ) -> bool:
        """Resolve a conflict using the specified resolution option"""
        redis_client = await get_redis_client()
        
        # Get the conflict
        conflict_key = f"merge:{merge_id}:conflict:{conflict_id}"
        conflict_json = await redis_client.get(conflict_key)
        if not conflict_json:
            logger.error(f"Conflict {conflict_id} not found for merge {merge_id}")
            return False
            
        conflict = Conflict.model_validate_json(conflict_json)
        
        # Find the resolution option
        resolution_option = None
        for option in conflict.resolution_options:
            if option.id == resolution_id:
                resolution_option = option
                break
                
        if not resolution_option:
            logger.error(f"Resolution option {resolution_id} not found for conflict {conflict_id}")
            return False
            
        # Apply the resolution
        # This will be expanded in future user stories
        # For now, just mark the conflict as resolved
        conflict.resolved = True
        conflict.resolution = resolution_option
        
        # Update the conflict in Redis
        await redis_client.set(conflict_key, conflict.model_dump_json())
        
        # Update conflict counts
        counts_key = f"merge:{merge_id}:conflict_counts"
        counts_json = await redis_client.get(counts_key)
        if counts_json:
            counts = json.loads(counts_json)
            if "resolved" in counts:
                counts["resolved"] += 1
            else:
                counts["resolved"] = 1
                
            if "unresolved" in counts:
                counts["unresolved"] = max(0, counts["unresolved"] - 1)
                
            await redis_client.set(counts_key, json.dumps(counts))
            
        return True

    async def analyze_and_resolve_conflict(
        self,
        merge_id: str,
        conflict_id: str,
        auto_resolve: bool = False
    ) -> Dict[str, Any]:
        """Analyze a conflict and generate resolution options"""
        # Get the conflict
        conflict = await self.get_conflict(merge_id, conflict_id)
        if not conflict:
            logger.error(f"Conflict {conflict_id} not found for merge {merge_id}")
            return {"error": "Conflict not found"}
            
        # Get ontology information
        # This would typically come from the graph schema
        # For now, we'll use a placeholder
        ontology = {
            "entity_types": ["Person", "Organization", "Product"],
            "relationship_types": ["WORKS_FOR", "OWNS", "RELATED_TO"],
            "property_constraints": {
                "age": {"type": "integer", "min": 0, "max": 120},
                "name": {"type": "string", "required": True},
                "email": {"type": "string", "format": "email"}
            }
        }
        
        # Initialize the resolution pipeline
        pipeline = build_resolution_pipeline()
        
        # Create initial state
        initial_state = {
            "conflict": conflict.model_dump(),
            "merge_id": merge_id,
            "ontology": ontology,
            "resolution": None,
            "error": None,
            "status": "pending"
        }
        
        # Run the pipeline
        if auto_resolve:
            # Run the full pipeline
            final_state = pipeline.invoke(initial_state)
            
            # If resolution was successful, apply it
            if final_state["status"] == "resolved" and not final_state.get("error"):
                # Update the conflict in Redis
                updated_conflict = Conflict.model_validate(final_state["conflict"])
                redis_client = await get_redis_client()
                await redis_client.set(
                    f"merge:{merge_id}:conflict:{conflict_id}",
                    updated_conflict.model_dump_json()
                )
                
                # Update conflict counts if resolved
                if updated_conflict.resolved:
                    counts_key = f"merge:{merge_id}:conflict_counts"
                    counts_json = await redis_client.get(counts_key)
                    if counts_json:
                        counts = json.loads(counts_json)
                        if "resolved" in counts:
                            counts["resolved"] += 1
                        else:
                            counts["resolved"] = 1
                            
                        if "unresolved" in counts:
                            counts["unresolved"] = max(0, counts["unresolved"] - 1)
                            
                        await redis_client.set(counts_key, json.dumps(counts))
                
                return {
                    "success": True,
                    "conflict": updated_conflict.model_dump(),
                    "status": final_state["status"]
                }
            else:
                return {
                    "success": False,
                    "error": final_state.get("error", "Unknown error"),
                    "status": final_state["status"]
                }
        else:
            # Just analyze and generate options
            # Run only the first two steps
            analyze_state = await pipeline.nodes["analyze_conflict"].ainvoke(initial_state)
            if analyze_state.get("error"):
                return {
                    "success": False,
                    "error": analyze_state.get("error"),
                    "status": analyze_state["status"]
                }
                
            options_state = await pipeline.nodes["generate_options"].ainvoke(analyze_state)
            if options_state.get("error"):
                return {
                    "success": False,
                    "error": options_state.get("error"),
                    "status": options_state["status"]
                }
                
            # Update the conflict in Redis
            updated_conflict = Conflict.model_validate(options_state["conflict"])
            redis_client = await get_redis_client()
            await redis_client.set(
                f"merge:{merge_id}:conflict:{conflict_id}",
                updated_conflict.model_dump_json()
            )
            
            return {
                "success": True,
                "conflict": updated_conflict.model_dump(),
                "status": options_state["status"]
            }

    async def analyze_conflicts_with_llm(
        self, 
        merge_id: str,
        conflict_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze conflicts using LLM to generate resolution options"""
        # Get all unresolved conflicts
        if conflict_ids:
            # Get specific conflicts by ID
            conflicts = []
            for conflict_id in conflict_ids:
                conflict = await self.get_conflict(merge_id, conflict_id)
                if conflict:
                    conflicts.append(conflict)
            total = len(conflicts)
        else:
            # Get all unresolved conflicts
            conflicts, total = await self.get_conflicts(
                merge_id=merge_id, 
                resolved=False
            )
        
        # Get ontology for context
        # TODO: Get actual ontology from the merge process
        ontology = {
            "entity_types": ["Person", "Organization", "Product"],
            "relationship_types": ["WORKS_FOR", "OWNS", "RELATED_TO"],
            "property_constraints": {
                "age": {"type": "integer", "min": 0, "max": 120},
                "name": {"type": "string", "required": True},
                "email": {"type": "string", "format": "email"}
            }
        }
        
        # Create analyzer
        analyzer = LLMConflictAnalyzer()
        
        # Process each conflict
        analyzed_count = 0
        for conflict in conflicts:
            try:
                # Analyze conflict
                resolution_options = await analyzer.analyze_conflict(conflict, ontology)
                
                # Update conflict with analysis results and options
                conflict.resolution_options = resolution_options
                
                # Store updated conflict
                await self._update_conflict(merge_id, conflict)
                
                analyzed_count += 1
                
            except Exception as e:
                logger.error(f"Error analyzing conflict {conflict.id}: {str(e)}")
        
        return {
            "total_conflicts": total,
            "analyzed": analyzed_count
        }

    async def _update_conflict(self, merge_id: str, conflict: Conflict) -> None:
        """Update a conflict in storage"""
        key = f"merge:{merge_id}:conflict:{conflict.id}"
        
        # Store conflict as JSON in Redis
        redis_client = await get_redis_client()
        await redis_client.set(key, conflict.model_dump_json())
        
        # Set TTL for cleanup (30 days)
        ttl = 30 * 24 * 60 * 60  # 30 days in seconds
        await redis_client.expire(key, ttl)
