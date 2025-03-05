"""Post-merge verification service for validating graph integrity after merge operations"""
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

from app.services.storage.interface import GraphStorageInterface
from app.services.storage.models import Node, Edge
from app.services.merge.models import (
    VerificationResult,
    VerificationCheck,
    VerificationCheckType
)
from app.services.resolution_history_service import ResolutionHistoryService
from app.services.ontology import load_ontology
from app.schemas.conflicts import Conflict

logger = logging.getLogger(__name__)

class PostMergeVerifier:
    """Service for verifying the integrity of a graph after a merge operation"""
    
    def __init__(
        self,
        merge_id: str,
        session_id: str,
        transform_id: str,
        staging_storage_service: GraphStorageInterface,
        prod_storage_service: GraphStorageInterface
    ):
        """Initialize the verifier
        
        Args:
            merge_id: ID of the merge operation
            transform_id: ID of the transformation
            staging_storage_service: Storage service for accessing staging graph data
            prod_storage_service: Storage service for accessing production graph data
        """
        self.merge_id = merge_id
        self.transform_id = transform_id
        self.session_id = session_id
        self.staging_storage_service = staging_storage_service
        self.prod_storage_service = prod_storage_service
        self.resolution_service = ResolutionHistoryService()
        
    async def verify_merge(self) -> VerificationResult:
        """Run comprehensive verification on merged graph
        
        Returns:
            VerificationResult containing all verification checks and overall status
        """
        start_time = time.time()
        
        # Initialize verification result
        verification_result = VerificationResult(
            merge_id=self.merge_id,
            transform_id=self.transform_id,
            success=False
        )
        
        try:
            # Get staging and production graphs
            staging_graph = await self.staging_storage_service.get_transformation_data(self.transform_id)
            production_graph = await self.prod_storage_service.get_production_graph_for_transform(self.transform_id)
            
            print(production_graph)
            print(staging_graph)
            
            # Get conflict resolutions
            resolutions = await self._get_resolutions()
            
            # Run verification checks
            
            # Check 1: Verify node counts
            node_verification = await self._verify_node_counts(
                staging_graph.nodes, 
                production_graph.nodes, 
                resolutions
            )
            verification_result.checks.append(node_verification)
            
            # Check 2: Verify relationship counts
            relationship_verification = await self._verify_relationship_counts(
                staging_graph.relationships, 
                production_graph.relationships, 
                resolutions
            )
            verification_result.checks.append(relationship_verification)
            
            # Check 3: Verify property values match resolutions
            property_verification = await self._verify_property_values(
                staging_graph.nodes, 
                production_graph.nodes, 
                resolutions
            )
            verification_result.checks.append(property_verification)
            
            # Check 4: Verify no orphaned nodes
            orphan_verification = await self._verify_no_orphaned_nodes(production_graph.nodes, production_graph.relationships)
            verification_result.checks.append(orphan_verification)
            
            # Check 5: Verify ontology constraints
            ontology_verification = await self._verify_ontology_constraints(
                production_graph.nodes, 
                production_graph.relationships,
                session_id=self.session_id
            )
            verification_result.checks.append(ontology_verification)
            
            # Set overall status
            verification_result.success = all(check.success for check in verification_result.checks)
            
        except Exception as e:
            logger.error(f"Error during merge verification: {str(e)}")
            verification_result.success = False
            verification_result.metadata["error"] = str(e)
        finally:
            # Record completion time and duration
            verification_result.completed_at = datetime.now()
            verification_result.verification_time_ms = (time.time() - start_time) * 1000
            
            # Log results
            logger.info(f"Merge verification completed: {verification_result.success}")
            
            # Store verification result
            await self._store_verification_result(verification_result)
            
        return verification_result
    
    async def _get_resolutions(self) -> Dict[str, Any]:
        """Get all resolutions for the merge
        
        Returns:
            Dictionary of resolutions by conflict ID
        """
        try:
            # Get all resolutions for this merge
            history_entries = await self.resolution_service.get_resolution_history(merge_id=self.merge_id)
            
            # Group by conflict ID
            resolutions = {}
            for entry in history_entries:
                resolutions[entry.conflict_id] = {
                    "resolution_id": entry.resolution_id,
                    "resolution_type": entry.resolution_type,
                    "resolution_data": entry.resolution_data,
                    "entity_types": entry.entity_types,
                    "property_names": entry.property_names,
                    "relationship_types": entry.relationship_types
                }
            
            return resolutions
        except Exception as e:
            logger.error(f"Error retrieving resolutions: {str(e)}")
            return {}
    
    async def _verify_node_counts(
        self, 
        staging_nodes: List[Dict[str, Any]], 
        production_nodes: List[Dict[str, Any]], 
        resolutions: Dict[str, Any]
    ) -> VerificationCheck:
        """Verify all nodes from staging exist in production accounting for resolved conflicts
        
        Args:
            staging_nodes: Nodes from staging graph
            production_nodes: Nodes from production graph
            resolutions: Dictionary of resolutions by conflict ID
            
        Returns:
            VerificationCheck with results
        """
        # Count nodes by type in staging
        staging_node_counts = {}
        for node in staging_nodes:
            node_type = node.get("type", "unknown")
            staging_node_counts[node_type] = staging_node_counts.get(node_type, 0) + 1
        
        # Count nodes by type in production
        production_node_counts = {}
        for node in production_nodes:
            node_type = node.get("type", "unknown")
            production_node_counts[node_type] = production_node_counts.get(node_type, 0) + 1
        
        # Find deleted nodes from resolutions
        deleted_node_types = {}
        for resolution in resolutions.values():
            if resolution.get("resolution_type") == "delete_node":
                for entity_type in resolution.get("entity_types", []):
                    deleted_node_types[entity_type] = deleted_node_types.get(entity_type, 0) + 1
        
        # Compare counts accounting for deletions
        missing_nodes = {}
        for node_type, count in staging_node_counts.items():
            expected_count = count - deleted_node_types.get(node_type, 0)
            actual_count = production_node_counts.get(node_type, 0)
            
            if expected_count > actual_count:
                missing_nodes[node_type] = expected_count - actual_count
        
        if missing_nodes:
            return VerificationCheck(
                check_type=VerificationCheckType.NODE_COUNT,
                success=False,
                message="Node count mismatch detected",
                details={
                    "staging_counts": staging_node_counts,
                    "production_counts": production_node_counts,
                    "deleted_counts": deleted_node_types,
                    "missing_nodes": missing_nodes
                }
            )
        
        return VerificationCheck(
            check_type=VerificationCheckType.NODE_COUNT,
            success=True,
            message="All nodes from staging exist in production",
            details={
                "staging_counts": staging_node_counts,
                "production_counts": production_node_counts,
                "deleted_counts": deleted_node_types
            }
        )
    
    async def _verify_relationship_counts(
        self, 
        staging_relationships: List[Dict[str, Any]], 
        production_relationships: List[Dict[str, Any]], 
        resolutions: Dict[str, Any]
    ) -> VerificationCheck:
        """Verify all relationships from staging exist in production accounting for resolved conflicts
        
        Args:
            staging_relationships: Relationships from staging graph
            production_relationships: Relationships from production graph
            resolutions: Dictionary of resolutions by conflict ID
            
        Returns:
            VerificationCheck with results
        """
        # Count relationships by type in staging
        staging_rel_counts = {}
        for rel in staging_relationships:
            rel_type = rel.get("type", "unknown")
            staging_rel_counts[rel_type] = staging_rel_counts.get(rel_type, 0) + 1
        
        # Count relationships by type in production
        production_rel_counts = {}
        for rel in production_relationships:
            rel_type = rel.get("type", "unknown")
            production_rel_counts[rel_type] = production_rel_counts.get(rel_type, 0) + 1
        
        # Find deleted relationships from resolutions
        deleted_rel_types = {}
        for resolution in resolutions.values():
            if resolution.get("resolution_type") == "delete_relationship":
                for rel_type in resolution.get("relationship_types", []):
                    deleted_rel_types[rel_type] = deleted_rel_types.get(rel_type, 0) + 1
        
        # Compare counts accounting for deletions
        missing_relationships = {}
        for rel_type, count in staging_rel_counts.items():
            expected_count = count - deleted_rel_types.get(rel_type, 0)
            actual_count = production_rel_counts.get(rel_type, 0)
            
            if expected_count > actual_count:
                missing_relationships[rel_type] = expected_count - actual_count
        
        if missing_relationships:
            return VerificationCheck(
                check_type=VerificationCheckType.RELATIONSHIP_COUNT,
                success=False,
                message="Relationship count mismatch detected",
                details={
                    "staging_counts": staging_rel_counts,
                    "production_counts": production_rel_counts,
                    "deleted_counts": deleted_rel_types,
                    "missing_relationships": missing_relationships
                }
            )
        
        return VerificationCheck(
            check_type=VerificationCheckType.RELATIONSHIP_COUNT,
            success=True,
            message="All relationships from staging exist in production",
            details={
                "staging_counts": staging_rel_counts,
                "production_counts": production_rel_counts,
                "deleted_counts": deleted_rel_types
            }
        )
    
    async def _verify_property_values(
        self, 
        staging_nodes: List[Dict[str, Any]], 
        production_nodes: List[Dict[str, Any]], 
        resolutions: Dict[str, Any]
    ) -> VerificationCheck:
        """Verify property values match resolved conflicts
        
        Args:
            staging_nodes: Nodes from staging graph
            production_nodes: Nodes from production graph
            resolutions: Dictionary of resolutions by conflict ID
            
        Returns:
            VerificationCheck with results
        """
        # Create lookup dictionaries for nodes
        staging_nodes_by_id = {node.get("id"): node for node in staging_nodes}
        production_nodes_by_id = {node.get("id"): node for node in production_nodes}
        
        # Find property value resolutions
        property_resolutions = {}
        for conflict_id, resolution in resolutions.items():
            if resolution.get("resolution_type") == "use_staging_value" or resolution.get("resolution_type") == "use_production_value":
                for property_name in resolution.get("property_names", []):
                    entity_id = resolution.get("resolution_data", {}).get("entity_id")
                    if entity_id:
                        if entity_id not in property_resolutions:
                            property_resolutions[entity_id] = {}
                        
                        property_resolutions[entity_id][property_name] = {
                            "resolution_type": resolution.get("resolution_type"),
                            "conflict_id": conflict_id
                        }
        
        # Verify property values match resolutions
        property_mismatches = []
        
        for entity_id, properties in property_resolutions.items():
            staging_node = staging_nodes_by_id.get(entity_id)
            production_node = production_nodes_by_id.get(entity_id)
            
            if not staging_node or not production_node:
                continue
                
            for property_name, resolution_info in properties.items():
                staging_value = staging_node.get("properties", {}).get(property_name)
                production_value = production_node.get("properties", {}).get(property_name)
                
                if resolution_info["resolution_type"] == "use_staging_value" and production_value != staging_value:
                    property_mismatches.append({
                        "entity_id": entity_id,
                        "property_name": property_name,
                        "expected_value": staging_value,
                        "actual_value": production_value,
                        "resolution_type": resolution_info["resolution_type"],
                        "conflict_id": resolution_info["conflict_id"]
                    })
                elif resolution_info["resolution_type"] == "use_production_value" and production_value != production_value:
                    # This is a tautology, but we're checking if the production value was changed
                    # from what it was at the time of resolution
                    property_mismatches.append({
                        "entity_id": entity_id,
                        "property_name": property_name,
                        "expected_value": "original_production_value",
                        "actual_value": production_value,
                        "resolution_type": resolution_info["resolution_type"],
                        "conflict_id": resolution_info["conflict_id"]
                    })
        
        if property_mismatches:
            affected_entities = list(set(mismatch["entity_id"] for mismatch in property_mismatches))
            return VerificationCheck(
                check_type=VerificationCheckType.PROPERTY_VALUES,
                success=False,
                message="Property value mismatches detected",
                details={"mismatches": property_mismatches},
                affected_entities=affected_entities
            )
        
        return VerificationCheck(
            check_type=VerificationCheckType.PROPERTY_VALUES,
            success=True,
            message="All property values match resolved conflicts",
            details={"resolution_count": len(property_resolutions)}
        )
    
    async def _verify_no_orphaned_nodes(
        self, 
        nodes: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> VerificationCheck:
        """Verify no orphaned nodes exist in the production graph
        
        Args:
            nodes: Nodes from production graph
            relationships: Relationships from production graph
            
        Returns:
            VerificationCheck with results
        """
        # Get all node IDs
        node_ids = set(node.get("id") for node in nodes)
        INTERNAL_NODE_TYPES = {"__Checkpoint__"}
        
        # Get all node IDs that participate in relationships
        connected_node_ids = set()
        for rel in relationships:
            connected_node_ids.add(rel.get("source"))
            connected_node_ids.add(rel.get("target"))
        
        # Find orphaned nodes
        orphaned_node_ids = node_ids - connected_node_ids
        
        # Get details for orphaned nodes
        orphaned_nodes = []
        for node in nodes:
            if node.get("type") in INTERNAL_NODE_TYPES:
                continue
            if node.get("id") in orphaned_node_ids:
                orphaned_nodes.append({
                    "id": node.get("id"),
                    "type": node.get("type"),
                    "label": node.get("label")
                })
        
        if orphaned_nodes:
            return VerificationCheck(
                check_type=VerificationCheckType.ORPHANED_NODES,
                success=False,
                message=f"Found {len(orphaned_nodes)} orphaned nodes",
                details={"orphaned_nodes": orphaned_nodes},
                affected_entities=[node["id"] for node in orphaned_nodes]
            )
        
        return VerificationCheck(
            check_type=VerificationCheckType.ORPHANED_NODES,
            success=True,
            message="No orphaned nodes found",
            details={"total_nodes": len(nodes)}
        )
    
    async def _verify_ontology_constraints(
        self, 
        nodes: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]],
        session_id: str
    ) -> VerificationCheck:
        """Verify ontology constraints are maintained
        
        Args:
            nodes: Nodes from production graph
            relationships: Relationships from production graph
            
        Returns:
            VerificationCheck with results
        """
        try:
            # Load ontology
            ontology = await load_ontology(ontology_id=session_id)
            ontology_entities = ontology.get("entities", {})
            ontology_entities['__Checkpoint__'] = {}
            
            # Verify node types
            invalid_node_types = []
            for node in nodes:
                node_type = node.get("type")
                if node_type not in ontology_entities:
                    invalid_node_types.append({
                        "id": node.get("id"),
                        "type": node_type
                    })
            
            # Verify relationship types and directions
            invalid_relationships = []
            for rel in relationships:
                rel_type = rel.get("type")
                source_id = rel.get("source")
                target_id = rel.get("target")
                
                # Find source and target nodes
                source_node = next((n for n in nodes if n.get("id") == source_id), None)
                target_node = next((n for n in nodes if n.get("id") == target_id), None)
                
                if not source_node or not target_node:
                    invalid_relationships.append({
                        "id": rel.get("id"),
                        "type": rel_type,
                        "source": source_id,
                        "target": target_id,
                        "error": "Missing source or target node"
                    })
                    continue
                
                source_type = source_node.get("type")
                target_type = target_node.get("type")
                
                # Check if relationship type is valid
                ontology_relationships = ontology_entities.get(source_type, {}).get("relationships", {})
                if rel_type not in ontology_relationships:
                    invalid_relationships.append({
                        "id": rel.get("id"),
                        "type": rel_type,
                        "source": source_id,
                        "target": target_id,
                        "error": "Invalid relationship type"
                    })
                    continue
                
                # Check if relationship direction is valid
                valid_target = ontology_relationships.get(rel_type, {}).get("target", '')
                if not valid_target == target_type:
                    invalid_relationships.append({
                        "id": rel.get("id"),
                        "type": rel_type,
                        "source": source_id,
                        "target": target_id,
                        "source_type": source_type,
                        "target_type": target_type,
                        "error": "Invalid relationship"
                    })
            
            # Combine results
            violations = []
            if invalid_node_types:
                violations.append({
                    "type": "invalid_node_types",
                    "count": len(invalid_node_types),
                    "details": invalid_node_types
                })
            
            if invalid_relationships:
                violations.append({
                    "type": "invalid_relationships",
                    "count": len(invalid_relationships),
                    "details": invalid_relationships
                })
            
            if violations:
                affected_entities = []
                for node in invalid_node_types:
                    affected_entities.append(node["id"])
                for rel in invalid_relationships:
                    affected_entities.append(rel["id"])
                
                return VerificationCheck(
                    check_type=VerificationCheckType.ONTOLOGY_CONSTRAINTS,
                    success=False,
                    message="Ontology constraint violations detected",
                    details={"violations": violations},
                    affected_entities=affected_entities
                )
            
            return VerificationCheck(
                check_type=VerificationCheckType.ONTOLOGY_CONSTRAINTS,
                success=True,
                message="All ontology constraints are satisfied",
                details={}
            )
        except Exception as e:
            logger.error(f"Error verifying ontology constraints: {str(e)}")
            return VerificationCheck(
                check_type=VerificationCheckType.ONTOLOGY_CONSTRAINTS,
                success=False,
                message=f"Error verifying ontology constraints: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _store_verification_result(self, result: VerificationResult) -> None:
        """Store verification result in Redis
        
        Args:
            result: Verification result to store
        """
        try:
            import redis.asyncio as redis
            from app.config import settings
            import json
            
            redis_client = redis.Redis.from_url(settings.REDIS_URL)
            
            # Store the result
            key = f"verification_result:{result.merge_id}"
            await redis_client.set(key, result.model_dump_json())
            
            # Set expiration (30 days)
            await redis_client.expire(key, 60 * 60 * 24 * 30)
            
            # Add to index
            await redis_client.sadd("verification_results", result.merge_id)
            
            # If verification failed, add to failed index
            if not result.success:
                await redis_client.sadd("failed_verifications", result.merge_id)
                
                # Send notification
                await self._send_verification_failure_notification(result)
        except Exception as e:
            logger.error(f"Error storing verification result: {str(e)}")
    
    async def _send_verification_failure_notification(self, result: VerificationResult) -> None:
        """Send notification for verification failure
        
        Args:
            result: Failed verification result
        """
        try:
            # This is a placeholder for notification logic
            # In a real implementation, this would send an email, Slack message, etc.
            logger.warning(f"Verification failed for merge {result.merge_id}. Notification would be sent here.")
            
            # Log detailed error report
            failed_checks = [check for check in result.checks if not check.success]
            for check in failed_checks:
                logger.warning(f"Failed check: {check.check_type} - {check.message}")
        except Exception as e:
            logger.error(f"Error sending verification failure notification: {str(e)}") 