"""Service for validating graphs before merging them into production"""
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
import logging
from datetime import datetime
import pytz

from app.schemas.graph import GraphResponse, Node, Edge
from app.services.storage.interface import GraphStorageInterface
from app.services.merge.models import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationIssueType
)
from app.services.merge.conflict import ConflictDetectionService
from app.services.storage.conflicts import ConflictStorageInterface
from app.schemas.conflicts import ConflictStatus
from app.services.ontology import load_ontology

logger = logging.getLogger(__name__)

class MergeValidationService:
    """Service for validating graphs before merging them into production"""
    
    def __init__(self, storage_factory: Optional[Callable] = None):
        """
        Initialize the validation service
        
        Args:
            storage_factory: Factory function to get graph storage instances
        """
        self.storage_factory = storage_factory
        # Initialize with production storage
        production_storage = None
        if storage_factory:
            production_storage = storage_factory(is_staging=False)
        self.conflict_detection = ConflictDetectionService(storage=production_storage)
    
    async def validate_merge(self, merge_id: str, transform_id: str, ontology_id: Optional[str] = None) -> ValidationResult:
        """
        Run all validations and return detailed results
        
        Args:
            merge_id: ID of the merge operation
            transform_id: ID of the transform operation that produced the staging graph
            ontology_id: Optional ID of the ontology to validate against
            
        Returns:
            ValidationResult: Detailed validation results
        """
        start_time = datetime.now(pytz.utc)
        validation_results: List[ValidationIssue] = []
        
        # Get staging and production storage
        staging_storage = self.storage_factory(is_staging=True)
        prod_storage = self.storage_factory(is_staging=False)
        
        # Get conflict status
        conflict_validation = await self.validate_conflict_resolution(merge_id)
        validation_results.extend(conflict_validation)
        
        # Get staging graph
        staging_graph = await staging_storage.get_graph_by_transform_id(transform_id)
        
        # Run structural validations
        structure_validation = await self.validate_graph_structure(staging_graph)
        validation_results.extend(structure_validation)
        
        # Run ontology validations if ontology_id is provided
        if ontology_id:
            ontology_validation = await self.validate_ontology_compliance(staging_graph, ontology_id)
            validation_results.extend(ontology_validation)
            
            # Validate required properties
            property_validation = await self.validate_required_properties(staging_graph, ontology_id)
            validation_results.extend(property_validation)
        
        # Validate no orphaned nodes
        orphan_validation = await self.validate_no_orphaned_nodes(staging_graph)
        validation_results.extend(orphan_validation)
        
        # Count issues by severity
        critical_count = sum(1 for issue in validation_results if issue.severity == ValidationSeverity.CRITICAL)
        warning_count = sum(1 for issue in validation_results if issue.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for issue in validation_results if issue.severity == ValidationSeverity.INFO)
        
        # Compile final report
        end_time = datetime.now(pytz.utc)
        validation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return ValidationResult(
            valid=(critical_count == 0),
            issues=validation_results,
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            total_nodes=len(staging_graph.nodes),
            total_edges=len(staging_graph.edges),
            validation_time_ms=validation_time_ms,
            metadata={
                "merge_id": merge_id,
                "transform_id": transform_id,
                "ontology_id": ontology_id
            }
        )
    
    async def validate_conflict_resolution(self, merge_id: str) -> List[ValidationIssue]:
        """
        Validate that all conflicts have been resolved
        
        Args:
            merge_id: ID of the merge operation
            
        Returns:
            List[ValidationIssue]: List of validation issues
        """
        issues: List[ValidationIssue] = []
        
        # Get conflict storage
        conflict_storage = self.storage_factory(is_conflict_storage=True)
        
        # Get unresolved conflicts
        unresolved_conflicts, total = await conflict_storage.get_conflicts(
            merge_id=merge_id,
            resolved=False,
            limit=1000,
            offset=0
        )
        
        if unresolved_conflicts:
            # Create validation issue for unresolved conflicts
            unresolved_ids = [conflict.id for conflict in unresolved_conflicts]
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.UNRESOLVED_CONFLICTS,
                    message=f"Found {len(unresolved_conflicts)} unresolved conflicts",
                    affected_ids=unresolved_ids,
                    severity=ValidationSeverity.CRITICAL,
                    metadata={
                        "total_unresolved": len(unresolved_conflicts),
                        "conflict_ids": unresolved_ids[:10]  # Include first 10 for reference
                    }
                )
            )
        
        return issues
    
    async def validate_graph_structure(self, graph: GraphResponse) -> List[ValidationIssue]:
        """
        Validate the structural integrity of the graph
        
        Args:
            graph: The graph to validate
            
        Returns:
            List[ValidationIssue]: List of validation issues
        """
        issues: List[ValidationIssue] = []
        
        # Check for invalid references in relationships
        node_ids = {node.id for node in graph.nodes}
        invalid_relationships: List[Tuple[Edge, str]] = []
        
        for edge in graph.edges:
            if edge.source not in node_ids:
                invalid_relationships.append((edge, "source"))
            if edge.target not in node_ids:
                invalid_relationships.append((edge, "target"))
        
        if invalid_relationships:
            # Group by issue type
            source_issues = [edge.id for edge, ref_type in invalid_relationships if ref_type == "source"]
            target_issues = [edge.id for edge, ref_type in invalid_relationships if ref_type == "target"]
            
            if source_issues:
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.INVALID_RELATIONSHIP_REFERENCE,
                        message=f"Found {len(source_issues)} relationships with invalid source references",
                        affected_ids=source_issues,
                        severity=ValidationSeverity.CRITICAL,
                        metadata={
                            "reference_type": "source",
                            "relationship_ids": source_issues[:10]  # Include first 10 for reference
                        }
                    )
                )
            
            if target_issues:
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.INVALID_RELATIONSHIP_REFERENCE,
                        message=f"Found {len(target_issues)} relationships with invalid target references",
                        affected_ids=target_issues,
                        severity=ValidationSeverity.CRITICAL,
                        metadata={
                            "reference_type": "target",
                            "relationship_ids": target_issues[:10]  # Include first 10 for reference
                        }
                    )
                )
        
        return issues
    
    async def validate_ontology_compliance(self, graph: GraphResponse, ontology_id: str) -> List[ValidationIssue]:
        """
        Validate that the graph complies with the ontology
        
        Args:
            graph: The graph to validate
            ontology_id: ID of the ontology to validate against
            
        Returns:
            List[ValidationIssue]: List of validation issues
        """
        issues: List[ValidationIssue] = []
        
        try:
            # Load ontology
            ontology = await load_ontology(ontology_id)
            
            # Validate node types
            node_types = ontology.get("node_types", {})
            valid_node_types = set(node_types.keys())
            
            invalid_node_types: Dict[str, List[str]] = {}
            for node in graph.nodes:
                if node.type not in valid_node_types:
                    if node.type not in invalid_node_types:
                        invalid_node_types[node.type] = []
                    invalid_node_types[node.type].append(node.id)
            
            if invalid_node_types:
                for node_type, node_ids in invalid_node_types.items():
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.UNKNOWN_ENTITY_TYPE,
                            message=f"Found {len(node_ids)} nodes with invalid type '{node_type}'",
                            affected_ids=node_ids,
                            severity=ValidationSeverity.CRITICAL,
                            metadata={
                                "invalid_type": node_type,
                                "node_ids": node_ids[:10]  # Include first 10 for reference
                            }
                        )
                    )
            
            # Validate relationship types
            relationship_types = ontology.get("relationship_types", {})
            valid_relationship_types = set(relationship_types.keys())
            
            invalid_relationship_types: Dict[str, List[str]] = {}
            for edge in graph.edges:
                if edge.type not in valid_relationship_types:
                    if edge.type not in invalid_relationship_types:
                        invalid_relationship_types[edge.type] = []
                    invalid_relationship_types[edge.type].append(edge.id)
            
            if invalid_relationship_types:
                for rel_type, rel_ids in invalid_relationship_types.items():
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.INVALID_RELATIONSHIP_TYPE,
                            message=f"Found {len(rel_ids)} relationships with invalid type '{rel_type}'",
                            affected_ids=rel_ids,
                            severity=ValidationSeverity.CRITICAL,
                            metadata={
                                "invalid_type": rel_type,
                                "relationship_ids": rel_ids[:10]  # Include first 10 for reference
                            }
                        )
                    )
            
            # Validate relationship constraints
            for rel_type, rel_config in relationship_types.items():
                allowed_sources = set(rel_config.get("source_types", []))
                allowed_targets = set(rel_config.get("target_types", []))
                
                invalid_source_rels: List[str] = []
                invalid_target_rels: List[str] = []
                
                for edge in graph.edges:
                    if edge.type == rel_type:
                        # Find source and target nodes
                        source_node = graph.get_node_by_id(edge.source)
                        target_node = graph.get_node_by_id(edge.target)
                        
                        if source_node and allowed_sources and source_node.type not in allowed_sources:
                            invalid_source_rels.append(edge.id)
                        
                        if target_node and allowed_targets and target_node.type not in allowed_targets:
                            invalid_target_rels.append(edge.id)
                
                if invalid_source_rels:
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.INVALID_RELATIONSHIP_SOURCE,
                            message=f"Found {len(invalid_source_rels)} '{rel_type}' relationships with invalid source types",
                            affected_ids=invalid_source_rels,
                            severity=ValidationSeverity.CRITICAL,
                            metadata={
                                "relationship_type": rel_type,
                                "allowed_source_types": list(allowed_sources),
                                "relationship_ids": invalid_source_rels[:10]
                            }
                        )
                    )
                
                if invalid_target_rels:
                    issues.append(
                        ValidationIssue(
                            type=ValidationIssueType.INVALID_RELATIONSHIP_TARGET,
                            message=f"Found {len(invalid_target_rels)} '{rel_type}' relationships with invalid target types",
                            affected_ids=invalid_target_rels,
                            severity=ValidationSeverity.CRITICAL,
                            metadata={
                                "relationship_type": rel_type,
                                "allowed_target_types": list(allowed_targets),
                                "relationship_ids": invalid_target_rels[:10]
                            }
                        )
                    )
        
        except Exception as e:
            logger.error(f"Error validating ontology compliance: {str(e)}")
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.VALIDATION_ERROR,
                    message=f"Error validating ontology compliance: {str(e)}",
                    affected_ids=[],
                    severity=ValidationSeverity.CRITICAL,
                    metadata={
                        "error": str(e),
                        "ontology_id": ontology_id
                    }
                )
            )
        
        return issues
    
    async def validate_required_properties(self, graph: GraphResponse, ontology_id: str) -> List[ValidationIssue]:
        """
        Validate that all required properties are present
        
        Args:
            graph: The graph to validate
            ontology_id: ID of the ontology to validate against
            
        Returns:
            List[ValidationIssue]: List of validation issues
        """
        issues: List[ValidationIssue] = []
        
        try:
            # Load ontology
            ontology = await load_ontology(ontology_id)
            
            # Validate node required properties
            node_types = ontology.get("node_types", {})
            
            for node in graph.nodes:
                if node.type in node_types:
                    node_config = node_types[node.type]
                    required_props = node_config.get("required_properties", [])
                    
                    missing_props = [prop for prop in required_props if prop not in node.properties]
                    
                    if missing_props:
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.MISSING_REQUIRED_PROPERTIES,
                                message=f"Node '{node.id}' of type '{node.type}' is missing required properties: {', '.join(missing_props)}",
                                affected_ids=[node.id],
                                severity=ValidationSeverity.CRITICAL,
                                metadata={
                                    "node_type": node.type,
                                    "missing_properties": missing_props
                                }
                            )
                        )
            
            # Validate relationship required properties
            relationship_types = ontology.get("relationship_types", {})
            
            for edge in graph.edges:
                if edge.type in relationship_types:
                    rel_config = relationship_types[edge.type]
                    required_props = rel_config.get("required_properties", [])
                    
                    missing_props = [prop for prop in required_props if prop not in edge.properties]
                    
                    if missing_props:
                        issues.append(
                            ValidationIssue(
                                type=ValidationIssueType.MISSING_REQUIRED_PROPERTIES,
                                message=f"Relationship '{edge.id}' of type '{edge.type}' is missing required properties: {', '.join(missing_props)}",
                                affected_ids=[edge.id],
                                severity=ValidationSeverity.CRITICAL,
                                metadata={
                                    "relationship_type": edge.type,
                                    "missing_properties": missing_props
                                }
                            )
                        )
        
        except Exception as e:
            logger.error(f"Error validating required properties: {str(e)}")
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.VALIDATION_ERROR,
                    message=f"Error validating required properties: {str(e)}",
                    affected_ids=[],
                    severity=ValidationSeverity.CRITICAL,
                    metadata={
                        "error": str(e),
                        "ontology_id": ontology_id
                    }
                )
            )
        
        return issues
    
    async def validate_no_orphaned_nodes(self, graph: GraphResponse, allowed_orphan_types: Optional[List[str]] = None) -> List[ValidationIssue]:
        """
        Validate that there are no orphaned nodes (nodes without relationships)
        
        Args:
            graph: The graph to validate
            allowed_orphan_types: Optional list of node types that are allowed to be orphaned
            
        Returns:
            List[ValidationIssue]: List of validation issues
        """
        issues: List[ValidationIssue] = []
        allowed_types = set(allowed_orphan_types or [])
        
        # Build set of nodes that have relationships
        connected_nodes: Set[str] = set()
        for edge in graph.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        # Find orphaned nodes
        orphaned_nodes: Dict[str, List[str]] = {}
        for node in graph.nodes:
            if node.id not in connected_nodes and node.type not in allowed_types:
                if node.type not in orphaned_nodes:
                    orphaned_nodes[node.type] = []
                orphaned_nodes[node.type].append(node.id)
        
        if orphaned_nodes:
            for node_type, node_ids in orphaned_nodes.items():
                issues.append(
                    ValidationIssue(
                        type=ValidationIssueType.ORPHANED_NODE,
                        message=f"Found {len(node_ids)} orphaned nodes of type '{node_type}'",
                        affected_ids=node_ids,
                        severity=ValidationSeverity.WARNING,  # Warning level since it might be intentional
                        metadata={
                            "node_type": node_type,
                            "node_ids": node_ids[:10]  # Include first 10 for reference
                        }
                    )
                )
        
        return issues
    
    def _compile_validation_report(self, validation_results: List[ValidationIssue]) -> ValidationResult:
        """
        Compile validation results into a comprehensive report
        
        Args:
            validation_results: List of validation issues
            
        Returns:
            ValidationResult: Compiled validation report
        """
        # Count issues by severity
        critical_count = sum(1 for issue in validation_results if issue.severity == ValidationSeverity.CRITICAL)
        warning_count = sum(1 for issue in validation_results if issue.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for issue in validation_results if issue.severity == ValidationSeverity.INFO)
        
        return ValidationResult(
            valid=(critical_count == 0),
            issues=validation_results,
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            total_nodes=0,  # Will be filled in by the calling method
            total_edges=0,  # Will be filled in by the calling method
            validation_time_ms=0,  # Will be filled in by the calling method
            metadata={}
        ) 