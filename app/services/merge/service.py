"""Service for handling merge operations"""
import uuid
import time
import logging
import asyncio
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import logging

from prefect import flow, task

from app.services.storage.interface import GraphStorageInterface
from app.services.storage.models import Node, Edge
from app.services.merge.models import (
    MergeStage,
    MergeProgress,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationIssueType,
    GraphResponse,
    EntityMappingResult,
    EntityMatch,
    MatchStrategy
)
from app.services.merge.progress import ProgressTracker
from app.config import settings

logger = logging.getLogger(__name__)

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
                total_edges=0,
                extraction_time_ms=0
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
            total_edges=len(edges),
            extraction_time_ms=duration_ms
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
async def complete_stage(
    merge_id: str,
    stage: MergeStage,
    progress_tracker: ProgressTracker,
    metadata: Dict[str, Any]
) -> None:
    """Complete a merge stage with metadata"""
    await progress_tracker.complete_merge_stage(
        merge_id,
        stage,
        metadata=metadata
    )

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
        
        await complete_stage(
            merge_id,
            MergeStage.EXTRACT,
            progress_tracker,
            {
                "total_nodes": graph.total_nodes,
                "total_edges": graph.total_edges,
                "extraction_time_ms": graph.extraction_time_ms
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
        
        await complete_stage(
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
        
        # Continue with other stages...
        # This will be expanded in subsequent user stories
        
        # Mark merge as complete
        await complete_merge(merge_id, progress_tracker)
        
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
