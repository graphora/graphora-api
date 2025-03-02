"""Service for handling graph merge operations"""
import uuid
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple, Union
from datetime import datetime
import json
import redis.asyncio as redis
import pytz
from app.config import settings
from app.schemas.conflicts import (
    Conflict, ConflictSeverity, ConflictType, ResolutionOption, ResolutionStrategy,
    ConflictResolutionResult, BulkResolutionResult, ConflictStatus
)
from app.services.storage.interface import GraphStorageInterface
from app.services.storage.models import Node, Edge
from app.schemas.graph import Node as SchemaNode, Edge as SchemaEdge, GraphResponse
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
    MatchStrategy,
    MergeStatus, 
    StageStatus
)
from app.services.merge.progress import ProgressTracker
from prefect import flow, task, get_run_logger
from app.services.merge.resolution_pipeline import build_resolution_pipeline
from app.services.merge.llm_analyzer import LLMConflictAnalyzer
from app.services.merge.tasks import (
    start_stage,
    complete_merge_stage,
    fail_merge,
    map_production_entities,
    detect_merge_conflicts
)
from app.services.ontology import load_ontology
from prefect.cache_policies import NO_CACHE
from app.services.merge.auto_resolution import AutoResolutionEngine
from app.utils.redis import get_redis_client
from app.services.merge.strategy_selection import StrategySelectionEngine
from app.services.merge.conflicts.base import ConflictDetector
from app.services.merge.resolution_applicator import ResolutionApplicator
from app.services.resolution_history_service import ResolutionHistoryService
from app.services.storage.transaction import TransactionManager, Neo4jTransactionManager
from app.services.merge.flow_manager import run_resolution_pipeline
from app.services.merge.validation import MergeValidationService

try:
    import baml as b
except ImportError:
    b = None  # BAML client is optional

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
    Extract graph from staging area
    
    Args:
        storage: Graph storage interface
        transform_id: Transform ID to extract
        
    Returns:
        GraphResponse containing nodes and edges
    """
    start_time = time.time()
    
    try:
        # Extract nodes with transform_id
        storage_nodes = await storage.get_nodes_by_property(
            property_name="transform_id",
            property_value=transform_id
        )
        
        if not storage_nodes:
            logger.warning(f"No nodes found with transform_id {transform_id}")
            return GraphResponse(
                nodes=[],
                edges=[],
                total_nodes=0,
                total_edges=0
            )
        
        # Convert storage nodes to schema nodes
        nodes = [
            SchemaNode(
                id=str(node.id),
                label=node.label,
                type=node.type,
                properties=node.properties
            ) for node in storage_nodes
        ]
        
        # Extract relationships between these nodes
        node_ids = [node.id for node in storage_nodes]
        storage_edges = await storage.get_relationships_between_nodes(node_ids)
        
        # Convert storage edges to schema edges
        edges = [
            SchemaEdge(
                id=str(edge.id),
                source=str(edge.source),
                target=str(edge.target),
                type=edge.type,
                properties=edge.properties
            ) for edge in storage_edges
        ]
        
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

@task(name="map_production_entities", cache_policy=NO_CACHE)
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
    node: SchemaNode,
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

@task(name="validate_graph", cache_policy=NO_CACHE)
async def validate_graph(
    graph: GraphResponse,
    ontology_id: str,
    progress_tracker: ProgressTracker
) -> List[ValidationIssue]:
    """Validate graph against ontology"""
    try:
        # Load ontology
        ontology = await load_ontology(ontology_id)
        
        # Define internal node types to skip validation
        INTERNAL_NODE_TYPES = {"Checkpoint"}
        
        # Validate nodes
        for node in graph.nodes:
            # Skip validation for internal node types
            if node.type in INTERNAL_NODE_TYPES:
                continue
                
            if node.type not in ontology.get("entities", {}):
                raise ValueError(f"Invalid node type: {node.type}")
                
            # Validate properties
            entity_def = ontology["entities"][node.type]
            required_props = {
                name for name, prop in entity_def.get("properties", {}).items()
                if prop.get("required", False)
            }
            
            missing_props = required_props - set(node.properties.keys())
            if missing_props:
                raise ValueError(
                    f"Node {node.id} missing required properties: {missing_props}"
                )
                
        # Validate edges
        for edge in graph.edges:
            source_type = next(
                (n.type for n in graph.nodes if n.id == edge.source),
                None
            )
            target_type = next(
                (n.type for n in graph.nodes if n.id == edge.target),
                None
            )
            
            if not source_type or not target_type:
                raise ValueError(f"Edge {edge.id} has invalid source or target node")
                
            # Skip validation for edges connected to internal nodes
            if source_type in INTERNAL_NODE_TYPES or target_type in INTERNAL_NODE_TYPES:
                continue
                
            # Check if relationship type exists in ontology
            if edge.type not in ontology.get("relationships", {}):
                raise ValueError(f"Invalid relationship type: {edge.type}")
                
            # Check if relationship is valid between these node types
            rel_def = ontology["relationships"][edge.type]
            if source_type != rel_def.get("source") or target_type != rel_def.get("target"):
                raise ValueError(
                    f"Invalid relationship {edge.type} between {source_type} and {target_type}"
                )
                
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate graph: {str(e)}")
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

def validate_node(node: SchemaNode, ontology: Dict[str, Any]) -> List[ValidationIssue]:
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
    metadata['completed_at'] = datetime.now().isoformat()
    
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
    ontology_id: Optional[str] = None
) -> None:
    """
    Prefect flow for graph merge process.
    
    Breaks down the merge process into smaller tasks for better
    observability and retry capabilities.
    """
    try:
        # Initialize services
        from app.services.storage.neo4j import Neo4jStorage
        from app.services.merge.progress import ProgressTracker
        
        storage = Neo4jStorage()
        progress_tracker = ProgressTracker()
        
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
        )(graph, ontology_id, progress_tracker) if ontology_id else None
        
        # Wait for both tasks to complete
        if validation_task:
            mapping, validation = await asyncio.gather(mapping_task, validation_task)
        else:
            mapping = await mapping_task
            validation = True  # Skip validation if no ontology_id provided
        
        await complete_merge_stage(
            merge_id,
            MergeStage.ANALYZE,
            progress_tracker,
            {
                "total_entities": mapping.total_entities,
                "matched_entities": mapping.matched_entities,
                "mapping_time_ms": mapping.mapping_time_ms,
                "validation_result": validation,
                "is_valid": validation
            }
        )
        
        if not validation:
            error_msg = "Graph validation failed with critical issues"
            logger.error(error_msg)
            await fail_merge(
                merge_id,
                error_msg,
                progress_tracker,
                validation
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
    finally:
        # Clean up resources
        if 'storage' in locals():
            await storage.close()
        if 'progress_tracker' in locals() and hasattr(progress_tracker, 'close'):
            await progress_tracker.close()

class MergeService:
    """Service for handling graph merge operations"""
    
    def __init__(
        self,
        storage: GraphStorageInterface,
        production_storage: GraphStorageInterface,
        progress_tracker: ProgressTracker,
        transaction_manager: Optional[TransactionManager] = None
    ):
        """Initialize merge service
        
        Args:
            storage: Storage interface for staging data
            production_storage: Storage interface for production data
            progress_tracker: Progress tracking service
            transaction_manager: Transaction manager for database operations
        """
        self.storage = storage  # Staging storage
        self.production_storage = production_storage  # Production storage
        self.progress_tracker = progress_tracker
        self.conflict_detection = ConflictDetectionService(production_storage)
        # Initialize the resolution applicator
        self.resolution_applicator = ResolutionApplicator(storage, production_storage)
        # Initialize the resolution history service
        self.resolution_history = ResolutionHistoryService()
        # Transaction manager
        self.transaction_manager = transaction_manager

    def _get_transaction_manager(self) -> TransactionManager:
        """Get transaction manager, creating one if needed
        
        Returns:
            TransactionManager: Transaction manager instance
        """
        if self.transaction_manager is None:
            # Create Neo4j transaction manager if production_storage is Neo4j
            if hasattr(self.production_storage, 'driver'):
                self.transaction_manager = Neo4jTransactionManager(self.production_storage.driver)
            else:
                # Fallback to a mock transaction manager for testing
                from unittest.mock import AsyncMock
                mock_manager = AsyncMock(spec=TransactionManager)
                mock_manager.begin_transaction.return_value = "mock_tx_id"
                mock_manager.commit_transaction.return_value = True
                mock_manager.rollback_transaction.return_value = True
                self.transaction_manager = mock_manager
                
        return self.transaction_manager

    async def start_merge_flow(
        self,
        session_id: str,
        transform_id: str,
        ontology_id: Optional[str] = None
    ) -> str:
        """
        Start a new merge flow.
        
        Args:
            session_id: ID of the session
            transform_id: ID of the transform to merge
            ontology_id: Optional ID of the ontology to validate against
            
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
                ontology_id=ontology_id
            )
            
            logger.info(f"Started merge flow {merge_id}")
            return merge_id
            
        except Exception as e:
            logger.error(f"Failed to start merge flow: {str(e)}")
            raise
    
    async def get_merge_progress(self, merge_id: str) -> Optional[MergeProgress]:
        """Get the current progress of a merge operation"""
        return await self.progress_tracker.get_progress(merge_id)

    async def validate_merge(self, merge_id: str, transform_id: str, ontology_id: Optional[str] = None, 
                            allowed_orphan_types: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate a merge before executing it
        
        Args:
            merge_id: ID of the merge operation
            transform_id: ID of the transform operation that produced the staging graph
            ontology_id: Optional ID of the ontology to validate against
            allowed_orphan_types: Optional list of node types that are allowed to be orphaned
            
        Returns:
            ValidationResult: Detailed validation results
        """
        # Create validation service
        validation_service = MergeValidationService(storage_factory=self.get_storage)
        
        # Run validation
        result = await validation_service.validate_merge(merge_id, transform_id, ontology_id)
        
        # If allowed orphan types are provided, re-run orphaned nodes validation
        if allowed_orphan_types:
            # Get staging graph
            staging_storage = self.get_storage(is_staging=True)
            staging_graph = await staging_storage.get_graph_by_transform_id(transform_id)
            
            # Remove existing orphaned node issues
            result.issues = [issue for issue in result.issues if issue.type != ValidationIssueType.ORPHANED_NODE]
            
            # Run orphaned nodes validation with allowed types
            orphan_issues = await validation_service.validate_no_orphaned_nodes(
                staging_graph, allowed_orphan_types=allowed_orphan_types
            )
            
            # Update result
            result.issues.extend(orphan_issues)
            
            # Recalculate counts
            result.warning_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.WARNING)
            result.info_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.INFO)
            result.critical_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.CRITICAL)
            
            # Update valid flag
            result.valid = (result.critical_count == 0)
        
        # Log validation result
        if result.valid:
            logger.info(f"Merge validation passed for merge_id={merge_id}, transform_id={transform_id}")
        else:
            logger.warning(
                f"Merge validation failed for merge_id={merge_id}, transform_id={transform_id}: "
                f"{result.critical_count} critical issues, {result.warning_count} warnings"
            )
        
        return result
    
    async def execute_merge(self, merge_id: str, transform_id: str, ontology_id: Optional[str] = None, 
                           allowed_orphan_types: Optional[List[str]] = None, 
                           skip_validation: bool = False) -> Dict[str, Any]:
        """
        Execute a merge operation
        
        Args:
            merge_id: ID of the merge operation
            transform_id: ID of the transform that produced the staging graph
            ontology_id: Optional ID of the ontology to validate against
            allowed_orphan_types: Optional list of node types that are allowed to be orphaned
            skip_validation: Whether to skip validation (default: False)
            
        Returns:
            Dict containing merge results
            
        Raises:
            ValueError: If validation fails and skip_validation is False
        """
        # Start merge stage
        await self.progress_tracker.start_merge_stage(merge_id, MergeStage.MERGE)
        
        try:
            # Validate merge if not skipped
            if not skip_validation:
                validation_result = await self.validate_merge(
                    merge_id, transform_id, ontology_id, allowed_orphan_types
                )
                
                # Block merge if validation fails
                if not validation_result.valid:
                    error_msg = f"Merge validation failed with {validation_result.critical_count} critical issues"
                    await self.progress_tracker.fail_merge_stage(
                        merge_id, MergeStage.MERGE, error_msg
                    )
                    raise ValueError(error_msg)
            
            # Get storages
            staging_storage = self.storage  # Use the staging storage from the service
            prod_storage = self.production_storage  # Use the production storage from the service
            
            # Get transaction manager
            transaction_manager = self._get_transaction_manager()
            
            # Create merge execution service
            from app.services.merge.execution_service import MergeExecutionService
            execution_service = MergeExecutionService(
                staging_storage=staging_storage,
                prod_storage=prod_storage,
                progress_tracker=self.progress_tracker,
                transaction_manager=transaction_manager
            )
            
            # Execute the merge
            merge_stats = await execution_service.execute_merge(
                merge_id=merge_id,
                transform_id=transform_id,
                batch_size=100  # Default batch size
            )
            
            return merge_stats
            
        except Exception as e:
            error_msg = f"Error executing merge: {str(e)}"
            logger.error(error_msg)
            await self.progress_tracker.fail_merge_stage(merge_id, MergeStage.MERGE, error_msg)
            raise

    async def cancel_merge(self, merge_id: str) -> bool:
        """Cancel an in-progress merge operation
        
        Args:
            merge_id: ID of the merge operation to cancel
            
        Returns:
            bool: True if successfully cancelled, False otherwise
        """
        try:
            # Get current progress to check if merge exists and is running
            progress = await self.progress_tracker.get_progress(merge_id)
            
            if not progress:
                logger.warning(f"Cannot cancel merge {merge_id}: not found")
                return False
                
            if progress.overall_status in [MergeStatus.COMPLETED, MergeStatus.FAILED, MergeStatus.CANCELLED]:
                logger.warning(f"Cannot cancel merge {merge_id}: already {progress.overall_status}")
                return False
            
            # Create merge execution service
            from app.services.merge.execution_service import MergeExecutionService
            execution_service = MergeExecutionService(
                staging_storage=self.storage,
                prod_storage=self.production_storage,
                progress_tracker=self.progress_tracker
            )
            
            # Cancel the merge
            return await execution_service.cancel_merge(merge_id)
            
        except Exception as e:
            error_msg = f"Error cancelling merge {merge_id}: {str(e)}"
            logger.error(error_msg)
            return False

    async def get_all_merges(self) -> List[Dict[str, Any]]:
        """Get all merge operations with their progress information
        
        Returns:
            List of merge operations with their progress details
        """
        try:
            # Get all merge IDs
            merge_ids = await self.progress_tracker.get_all_merge_ids()
            
            # Get progress for each merge ID
            merges = []
            for merge_id in merge_ids:
                progress = await self.progress_tracker.get_progress(merge_id)
                if progress:
                    merges.append(progress.model_dump())
            
            return merges
            
        except Exception as e:
            error_msg = f"Error getting all merges: {str(e)}"
            logger.error(error_msg)
            return []

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
        """Detect property conflicts for all nodes in the staging graph"""
        conflicts = []
        
        # Process each node in staging graph
        for staging_node in staging_graph.nodes:
            # Skip if no production matches
            if staging_node.id not in production_entity_mapping:
                continue
                
            # Get production matches
            production_ids = production_entity_mapping[staging_node.id]
            
            # Get production nodes
            for production_id in production_ids:
                # Get production node
                production_node = await self.production_storage.get_node_by_id(production_id)
                if not production_node:
                    continue
                    
                # Detect property conflicts
                node_conflicts = await self.detect_property_conflicts(
                    staging_node,
                    production_node,
                    self.merge_id
                )
                
                conflicts.extend(node_conflicts)
                
        return conflicts

    async def detect_property_conflicts(
        self,
        staging_node: SchemaNode,
        production_node: SchemaNode,
        merge_id: str
    ) -> List[Conflict]:
        """Detect property conflicts between two nodes"""
        # Use conflict detection service to detect property conflicts
        return await self.conflict_detection.detect_property_conflicts(
            staging_node,
            production_node,
            merge_id
        )

    async def detect_relationship_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect conflicts in relationships between staging and production"""
        conflicts = []
        conflict_service = ConflictDetectionService(self.production_storage)
        merge_id = self.merge_id
        
        # Process each edge in staging graph
        for staging_edge in staging_graph.edges:
            # Skip if either endpoint has no production matches
            if (staging_edge.source not in production_entity_mapping or
                staging_edge.target not in production_entity_mapping):
                continue
                
            # Get production matches for both endpoints
            source_matches = production_entity_mapping[staging_edge.source] 
            target_matches = production_entity_mapping[staging_edge.target]
            
            # Check relationships between all possible endpoint combinations
            for source_id in source_matches:
                for target_id in target_matches:
                    # Get production relationships
                    prod_edges = await self.production_storage.get_relationships_between(
                        source_id,
                        target_id,
                        staging_edge.type
                    )
                    
                    # Detect conflicts for each relationship
                    for prod_edge in prod_edges:
                        # Detect conflicts
                        edge_conflicts = await conflict_service.detect_relationship_conflicts(
                            staging_edge,
                            prod_edge,
                            merge_id
                        )
                        conflicts.extend(edge_conflicts)
        
        return conflicts

    async def detect_entity_matching_conflicts(
        self,
        staging_graph: GraphResponse,
        production_entity_mapping: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect conflicts where a staging entity has multiple matches in production"""
        conflicts = []
        merge_id = self.merge_id
        
        for staging_id, production_ids in production_entity_mapping.items():
            # Only create a conflict if there are multiple matches
            if len(production_ids) > 1:
                # Find the staging node
                staging_node = next((n for n in staging_graph.nodes if n.id == staging_id), None)
                if not staging_node:
                    continue
                    
                # Create a conflict
                conflict = Conflict(
                    id=f"entity_match_{staging_id}",
                    merge_id=merge_id,
                    conflict_type=ConflictType.DUPLICATE_ENTITY,
                    severity=ConflictSeverity.MAJOR,
                    description=f"Entity '{staging_id}' matches multiple production entities: {', '.join(production_ids)}",
                    staging_ids=[staging_id],
                    production_ids=production_ids,
                    staging_value=staging_id,
                    production_value=production_ids,
                    context={
                        "entity_type": staging_node.label,
                        "staging_id": staging_id,
                        "production_ids": production_ids,
                        "properties": staging_node.properties
                    },
                    resolution_options=[
                        # Option for each production entity
                        *[
                            ResolutionOption(
                                id=f"entity_match_{staging_id}_{prod_id}",
                                description=f"Match with production entity: {prod_id}",
                                resolution_type=ResolutionStrategy.MATCH_ENTITY,
                                resolution_data={
                                    "staging_id": staging_id,
                                    "production_id": prod_id
                                },
                                confidence=1.0 / len(production_ids),
                                auto_resolvable=False
                            )
                            for prod_id in production_ids
                        ],
                        # Option to create a new entity
                        ResolutionOption(
                            id=f"entity_match_{staging_id}_new",
                            description="Create as a new entity",
                            resolution_type=ResolutionStrategy.CREATE_NEW,
                            resolution_data={
                                "staging_id": staging_id
                            },
                            confidence=0.3,
                            auto_resolvable=False
                        )
                    ]
                )
                conflicts.append(conflict)
                
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
        offset: int = 0,
        entity_type: Optional[str] = None
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
                if entity_type and conflict.entity_type != entity_type:
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

    async def apply_conflict_resolution(
        self,
        merge_id: str,
        conflict_id: str,
        resolution_id: Optional[str] = None,
        resolution_type: Optional[str] = None,
        resolution_data: Optional[Dict[str, Any]] = None,
        resolved_by: str = "user"
    ) -> ConflictResolutionResult:
        """Apply a resolution to a conflict
        
        Args:
            merge_id: ID of the merge process
            conflict_id: ID of the conflict to resolve
            resolution_id: ID of the resolution option to apply
            resolution_type: Type of resolution to apply (alternative to resolution_id)
            resolution_data: Additional data for the resolution (if resolution_type is provided)
            resolved_by: Who resolved the conflict
            
        Returns:
            Dict containing the result of the resolution application
        """
        # Get the conflict
        conflict = await self.get_conflict(merge_id, conflict_id)
        if not conflict:
            raise ValueError(f"Conflict {conflict_id} not found")
        
        # Check if already resolved
        if conflict.resolved:
            raise ValueError(f"Conflict {conflict_id} is already resolved")
        
        # Find the resolution option
        resolution_option = None
        if resolution_id:
            # Find by ID
            for option in conflict.resolution_options:
                if option.id == resolution_id:
                    resolution_option = option
                    break
            
            if not resolution_option:
                raise ValueError(f"Resolution option not found for conflict {conflict_id}")
        elif resolution_type:
            # Create a new resolution option
            resolution_option = ResolutionOption(
                id=f"resolution-{uuid.uuid4()}",
                description=f"Custom resolution: {resolution_type}",
                resolution_type=resolution_type,
                resolution_data=resolution_data or {},
                confidence=1.0,
                reasoning="Custom resolution provided by user",
                requires_review=False,
                auto_resolvable=False
            )
        else:
            raise ValueError("Either resolution_id or resolution_type must be provided")
        
        # Apply the resolution using the resolution applicator
        result = await self.resolution_applicator.apply_resolution(conflict, resolution_option)
        
        # If the resolution was applied successfully, update the conflict
        if result.get("applied", False):
            # Update the conflict with resolution information
            conflict.resolved = True
            conflict.resolution = resolution_option
            conflict.resolution_timestamp = datetime.now(pytz.utc)
            conflict.resolved_by = resolved_by
            
            # Store the updated conflict
            await self._update_conflict(merge_id, conflict)
            
            # Store in resolution history
            try:
                await self.resolution_history.store_resolution(
                    conflict=conflict,
                    resolution_id=resolution_option.id,
                    applied_by=resolved_by,
                    merge_id=merge_id,
                    success=True
                )
            except Exception as e:
                logger.error(f"Error storing resolution history: {str(e)}")
                # Don't fail the resolution if history storage fails
        
        return result

    async def apply_bulk_conflict_resolution(
        self,
        merge_id: str,
        conflict_ids: List[str],
        resolution_type: str,
        resolution_data: Optional[Dict[str, Any]] = None,
        resolved_by: str = "user"
    ) -> List[BulkResolutionResult]:
        """
        Apply the same resolution to multiple conflicts
        
        Args:
            merge_id: ID of the merge process
            conflict_ids: List of conflict IDs to resolve
            resolution_type: Type of resolution to apply to all conflicts
            resolution_data: Additional data for the resolution
            resolved_by: User who resolved the conflicts
            
        Returns:
            List[BulkResolutionResult]: Results of the resolution application for each conflict
        """
        results = []
        
        for conflict_id in conflict_ids:
            # Apply resolution to each conflict
            resolution_result = await self.apply_conflict_resolution(
                merge_id=merge_id,
                conflict_id=conflict_id,
                resolution_type=resolution_type,
                resolution_data=resolution_data,
                resolved_by=resolved_by
            )
            
            # Convert to bulk result format
            results.append(BulkResolutionResult(
                conflict_id=conflict_id,
                resolved=resolution_result.resolved,
                error=resolution_result.error
            ))
            
        return results
        
    async def _update_conflict(self, merge_id: str, conflict: Conflict) -> None:
        """Update a conflict in storage"""
        key = f"merge:{merge_id}:conflict:{conflict.id}"
        
        # Store conflict as JSON in Redis
        redis_client = await get_redis_client()
        await redis_client.set(key, conflict.model_dump_json())
        
        # Set TTL for cleanup (30 days)
        ttl = 30 * 24 * 60 * 60  # 30 days in seconds
        await redis_client.expire(key, ttl)

    async def start_merge_process(
        self,
        merge_id: str,
        graph: GraphResponse,
        ontology_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start a new merge process"""
        try:
            # Load and validate ontology
            ontology_path = Path(settings.ontology_dir).expanduser() / f"{ontology_id}.yaml"
            ontology = load_ontology(ontology_path)
            
            # Start validation stage
            await start_stage(merge_id, MergeStage.VALIDATION, self.progress_tracker)
            
            # TODO: Validate graph against ontology
            
            await complete_merge_stage(
                merge_id,
                MergeStage.VALIDATION,
                self.progress_tracker,
                metadata={
                    "ontology_id": ontology_id,
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges)
                }
            )
            
            # Start entity mapping
            await start_stage(merge_id, MergeStage.ENTITY_MAPPING, self.progress_tracker)
            
            mapping_result = await map_production_entities(
                self.storage,
                graph
            )
            
            await complete_merge_stage(
                merge_id,
                MergeStage.ENTITY_MAPPING,
                self.progress_tracker,
                metadata={
                    "total_entities": mapping_result.total_entities,
                    "matched_entities": mapping_result.matched_entities,
                    "mapping_time_ms": mapping_result.mapping_time_ms
                }
            )
            
            # Start conflict detection
            await start_stage(merge_id, MergeStage.CONFLICT_DETECTION, self.progress_tracker)
            
            await detect_merge_conflicts(
                merge_id,
                graph,
                mapping_result,
                self.storage,
                self.progress_tracker
            )
            
            await complete_merge_stage(
                merge_id,
                MergeStage.CONFLICT_DETECTION,
                self.progress_tracker
            )
            
        except Exception as e:
            logger.error(f"Merge process failed: {str(e)}")
            await fail_merge(merge_id, str(e), self.progress_tracker)
            raise

    async def select_resolution_strategies(self, merge_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Select best resolution strategies for conflicts"""
        # Get unresolved conflicts
        conflicts, total = await self.get_conflicts(
            merge_id=merge_id,
            resolved=False
        )
        
        # Create strategy engine
        engine = StrategySelectionEngine(config)
        
        # Track statistics
        stats = {
            "total": total,
            "processed": 0,
            "strategy_counts": {},
            "confidence_avg": 0.0,
            "by_type": {}
        }
        
        total_confidence = 0.0
        
        # Process each conflict
        for conflict in conflicts:
            # Select strategy
            strategy_name, resolution_option, confidence, explanation = await engine.select_strategy(conflict)
            
            # Skip if no strategy found
            if not strategy_name or not resolution_option:
                continue
                
            # Store strategy selection
            await self._store_strategy_selection(
                merge_id,
                conflict.id,
                strategy_name,
                resolution_option.id,
                confidence,
                explanation
            )
            
            # Update stats
            stats["processed"] += 1
            total_confidence += confidence
            
            if strategy_name not in stats["strategy_counts"]:
                stats["strategy_counts"][strategy_name] = 0
            stats["strategy_counts"][strategy_name] += 1
            
            conflict_type = conflict.conflict_type.value
            if conflict_type not in stats["by_type"]:
                stats["by_type"][conflict_type] = 0
            stats["by_type"][conflict_type] += 1
            
        # Calculate average confidence
        if stats["processed"] > 0:
            stats["confidence_avg"] = total_confidence / stats["processed"]
            
        return stats
        
    async def _store_strategy_selection(
        self,
        merge_id: str,
        conflict_id: str,
        strategy: str,
        resolution_id: str,
        confidence: float,
        explanation: str
    ) -> None:
        """Store strategy selection for a conflict"""
        # Get conflict
        conflict = await self.get_conflict(merge_id, conflict_id)
        if not conflict:
            return
            
        # Store strategy in conflict
        conflict.context = conflict.context or {}
        conflict.context["selected_strategy"] = {
            "name": strategy,
            "resolution_id": resolution_id,
            "confidence": confidence,
            "explanation": explanation,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Update conflict
        await self._update_conflict(merge_id, conflict)

    async def apply_selected_strategies(self, merge_id: str, min_confidence: float = 0.7) -> Dict[str, Any]:
        """Apply selected resolution strategies
        
        Args:
            merge_id: ID of the merge process
            min_confidence: Minimum confidence threshold for applying strategies
            
        Returns:
            Dict containing statistics about applied strategies
        """
        # Get all conflicts for this merge
        redis = await get_redis_client()
        conflict_ids_key = f"merge:{merge_id}:conflict_ids"
        conflict_ids_json = await redis.get(conflict_ids_key)
        
        if not conflict_ids_json:
            return {
                "total": 0,
                "applied": 0,
                "skipped_low_confidence": 0,
                "skipped_no_strategy": 0,
                "by_strategy": {}
            }
        
        conflict_ids = json.loads(conflict_ids_json)
        
        # Initialize counters
        total = len(conflict_ids)
        applied = 0
        skipped_low_confidence = 0
        skipped_no_strategy = 0
        by_strategy = {}
        
        # Process each conflict
        for conflict_id in conflict_ids:
            # Get the conflict
            conflict = await self.get_conflict(merge_id, conflict_id)
            if not conflict:
                continue
                
            # Skip if already resolved
            if conflict.resolved:
                continue
                
            # Check if this conflict has a selected strategy
            if not conflict.context or "selected_strategy" not in conflict.context:
                skipped_no_strategy += 1
                continue
                
            # Get the selected strategy
            strategy = conflict.context["selected_strategy"]
            strategy_name = strategy.get("name", "unknown")
            resolution_id = strategy.get("resolution_id")
            confidence = strategy.get("confidence", 0.0)
            
            # Skip if confidence is below threshold
            if confidence < min_confidence:
                skipped_low_confidence += 1
                continue
                
            # Apply the resolution
            try:
                await self.apply_conflict_resolution(
                    merge_id=merge_id,
                    conflict_id=conflict_id,
                    resolution_id=resolution_id
                )
                applied += 1
                
                # Track by strategy
                by_strategy[strategy_name] = by_strategy.get(strategy_name, 0) + 1
            except Exception as e:
                logger.error(f"Error applying strategy for conflict {conflict_id}: {str(e)}")
        
        # Return statistics
        return {
            "total": total,
            "applied": applied,
            "skipped_low_confidence": skipped_low_confidence,
            "skipped_no_strategy": skipped_no_strategy,
            "by_strategy": by_strategy
        }

    async def auto_resolve_conflicts(
        self, 
        merge_id: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Automatically resolve minor conflicts
        
        Args:
            merge_id: ID of the merge process
            config: Optional configuration for auto-resolution
            
        Returns:
            Dict containing statistics about auto-resolution
        """
        # Get all conflicts for this merge
        conflicts, total = await self.get_conflicts(
            merge_id=merge_id,
            resolved=False  # Only get unresolved conflicts
        )
        
        # Initialize counters
        auto_resolved = 0
        manual_required = 0
        by_type = {}
        by_severity = {}
        
        # Process each conflict
        for conflict in conflicts:
            # Track by type
            conflict_type = conflict.conflict_type.value
            by_type[conflict_type] = by_type.get(conflict_type, 0) + 1
            
            # Track by severity
            severity = conflict.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Check if this conflict can be auto-resolved
            if conflict.severity == ConflictSeverity.MINOR and conflict.resolution_options:
                # Get the highest confidence resolution option
                best_option = max(conflict.resolution_options, key=lambda x: x.confidence)
                
                # Apply the resolution
                try:
                    await self.apply_conflict_resolution(
                        merge_id=merge_id,
                        conflict_id=conflict.id,
                        resolution_id=best_option.id
                    )
                    auto_resolved += 1
                except Exception as e:
                    logger.error(f"Error auto-resolving conflict {conflict.id}: {str(e)}")
                    manual_required += 1
            else:
                manual_required += 1
        
        # Return statistics
        return {
            "total": total,
            "auto_resolved": auto_resolved,
            "manual_required": manual_required,
            "by_type": by_type,
            "by_severity": by_severity
        }

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

    async def get_resolution_option(self, conflict_id: str, resolution_id: str) -> Optional[ResolutionOption]:
        """
        Get a specific resolution option for a conflict
        
        Args:
            conflict_id: ID of the conflict
            resolution_id: ID of the resolution option
            
        Returns:
            Optional[ResolutionOption]: The resolution option if found, None otherwise
        """
        conflict = await self.get_conflict(conflict_id)
        if not conflict:
            return None
            
        for option in conflict.resolution_options:
            if option.id == resolution_id:
                return option
                
        return None

    async def get_resolution_suggestions(
        self,
        merge_id: str,
        conflict_id: str
    ) -> List[Dict[str, Any]]:
        """Get resolution suggestions based on similar past resolutions
        
        Args:
            merge_id: ID of the merge process
            conflict_id: ID of the conflict
            
        Returns:
            List of suggested resolutions with similarity scores
        """
        # Get conflict
        conflict = await self.get_conflict(merge_id, conflict_id)
        if not conflict:
            raise ValueError(f"Conflict {conflict_id} not found for merge {merge_id}")
        
        # Find similar resolutions
        similar_resolutions = await self.resolution_history.find_similar_resolutions(
            conflict=conflict
        )
        
        # Format suggestions
        suggestions = []
        for res in similar_resolutions:
            entry = res["entry"]
            suggestion = {
                "resolution_type": entry["resolution_type"],
                "resolution_data": entry["resolution_data"],
                "similarity_score": res["similarity_score"],
                "previously_applied_at": entry["applied_at"],
                "was_successful": entry["success"],
                "original_conflict_context": entry["context"]
            }
            suggestions.append(suggestion)
        
        return suggestions

    async def apply_batch_resolutions(
        self,
        merge_id: str,
        resolutions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Apply multiple resolutions at once
        
        Args:
            merge_id: ID of the merge process
            resolutions: List of {conflict_id, resolution_id} mappings
            
        Returns:
            Dict[str, Any]: Summary of resolution results
        """
        results = []
        success_count = 0
        failure_count = 0
        
        for resolution in resolutions:
            conflict_id = resolution.get("conflict_id")
            resolution_id = resolution.get("resolution_id")
            
            if not conflict_id or not resolution_id:
                results.append({
                    "conflict_id": conflict_id,
                    "resolution_id": resolution_id,
                    "applied": False,
                    "error": "Missing conflict_id or resolution_id"
                })
                failure_count += 1
                continue
                
            try:
                result = await self.apply_conflict_resolution(
                    merge_id=merge_id,
                    conflict_id=conflict_id,
                    resolution_id=resolution_id
                )
                
                results.append(result)
                
                if result.get("applied", False):
                    success_count += 1
                else:
                    failure_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to apply resolution: {str(e)}")
                results.append({
                    "conflict_id": conflict_id,
                    "resolution_id": resolution_id,
                    "applied": False,
                    "error": str(e)
                })
                failure_count += 1
        
        return {
            "total": len(resolutions),
            "success_count": success_count,
            "failure_count": failure_count,
            "results": results
        }

    async def start_resolution_pipeline(self, merge_id: str) -> str:
        """Start the resolution pipeline for conflicts"""
        # First check if the conflict detection is complete
        status = await self.get_merge_status(merge_id)
        
        if not status or status.stages_progress[MergeStage.CONFLICT_DETECTION].status != StageStatus.COMPLETED:
            raise ValueError("Cannot start resolution before conflict detection is complete")
        
        # Start resolution process
        flow_run_id = await run_resolution_pipeline(merge_id)
        
        return flow_run_id

    async def mark_conflict_for_review(self, merge_id: str, conflict_id: str) -> bool:
        """Mark a conflict as requiring human review"""
        conflict = await self.get_conflict(merge_id, conflict_id)
        if not conflict:
            return False
            
        # Set review flag
        conflict.requires_review = True
        
        # Update conflict
        await self._update_conflict(merge_id, conflict)
        
        return True

    async def apply_resolution(self, merge_id: str, conflict_id: str, resolution_id: str) -> bool:
        """Apply a specific resolution to a conflict"""
        result = await self.apply_conflict_resolution(
            merge_id=merge_id,
            conflict_id=conflict_id,
            resolution_id=resolution_id,
            resolved_by="prefect"
        )
        
        return result.resolved
