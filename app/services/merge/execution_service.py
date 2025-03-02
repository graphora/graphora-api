"""Service for executing graph merges from staging to production"""
import logging
import copy
import time
from typing import Dict, Any, List, Optional, Tuple, Set
import asyncio

from app.services.storage.interface import GraphStorageInterface
from app.services.merge.progress import ProgressTracker
from app.services.merge.models import MergeStage, MergeStatus, StageStatus
from app.schemas.graph import GraphResponse, Node, Edge
from app.schemas.conflicts import Conflict, ConflictStatus, ResolutionOption
from app.config import settings
from app.services.storage.transaction import TransactionManager, Neo4jTransactionManager
from app.services.merge.resolution_applicator import ResolutionApplicator

logger = logging.getLogger(__name__)

class MergeExecutionService:
    """Service for executing graph merges from staging to production"""
    
    def __init__(
        self, 
        staging_storage: GraphStorageInterface = None,
        prod_storage: GraphStorageInterface = None,
        progress_tracker: Optional[ProgressTracker] = None,
        transaction_manager: Optional[TransactionManager] = None
    ):
        """Initialize merge execution service
        
        Args:
            staging_storage: Storage interface for staging data
            prod_storage: Storage interface for production data
            progress_tracker: Progress tracking service
            transaction_manager: Transaction manager for database operations
        """
        self.staging_storage = staging_storage
        self.prod_storage = prod_storage
        self.progress_tracker = progress_tracker or ProgressTracker()
        self.transaction_manager = transaction_manager
        self.resolution_applicator = ResolutionApplicator(staging_storage, prod_storage) if staging_storage and prod_storage else None
        
    def _get_transaction_manager(self) -> TransactionManager:
        """Get transaction manager, creating one if needed
        
        Returns:
            TransactionManager: Transaction manager instance
        """
        if self.transaction_manager is None:
            # Create Neo4j transaction manager if prod_storage is Neo4j
            if hasattr(self.prod_storage, 'driver'):
                self.transaction_manager = Neo4jTransactionManager(self.prod_storage.driver)
            else:
                # Fallback to a mock transaction manager for testing
                from unittest.mock import AsyncMock
                mock_manager = AsyncMock(spec=TransactionManager)
                mock_manager.begin_transaction.return_value = "mock_tx_id"
                mock_manager.commit_transaction.return_value = True
                mock_manager.rollback_transaction.return_value = True
                self.transaction_manager = mock_manager
                
        return self.transaction_manager
        
    async def execute_merge(
        self,
        merge_id: str,
        transform_id: str,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Dict[str, Any]:
        """Execute merge from staging to production based on resolved conflicts
        
        Args:
            merge_id: ID of the merge operation
            transform_id: ID of the transform that produced the staging graph
            batch_size: Number of items to process in each batch
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay in seconds between retries
            
        Returns:
            Dict containing merge statistics
        """
        # Start merge stage
        await self.progress_tracker.start_merge_stage(merge_id, MergeStage.MERGE)
        
        # Get transaction manager
        transaction_manager = self._get_transaction_manager()
        
        try:
            # Execute merge within a transaction
            return await transaction_manager.execute_in_transaction(
                self._execute_merge_internal,
                merge_id=merge_id,
                transform_id=transform_id,
                batch_size=batch_size,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
        except Exception as e:
            logger.error(f"Merge failed: {str(e)}")
            # Update progress tracker
            await self.progress_tracker.fail_merge_stage(
                merge_id, 
                MergeStage.MERGE, 
                str(e)
            )
            raise
        
    async def _execute_merge_internal(
        self,
        merge_id: str,
        transform_id: str,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: int = 2,
        transaction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal merge execution within transaction
        
        Args:
            merge_id: ID of the merge operation
            transform_id: ID of the transform that produced the staging graph
            batch_size: Number of items to process in each batch
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay in seconds between retries
            transaction_id: ID of the active transaction
            
        Returns:
            Dict containing merge statistics
        """
        # 1. Get resolved conflicts
        resolved_conflicts = await self._get_resolved_conflicts(merge_id)
        
        # 2. Extract staging graph
        staging_graph = await self.staging_storage.get_transformation_data(transform_id)
        
        print(f"DEBUG: Staging graph nodes: {staging_graph.nodes}")
        
        # Ensure each node has a label field (required by the Node model)
        nodes_with_label = []
        for node in staging_graph.nodes:
            # Make a copy to avoid modifying the original
            node_copy = node.copy()
            # If label is missing, use the type field as label
            if 'label' not in node_copy:
                node_copy['label'] = node_copy.get('type', 'Unknown')
            nodes_with_label.append(node_copy)
        
        print(f"DEBUG: Nodes with label: {nodes_with_label}")
        
        # Convert relationships to edges format
        edges = []
        for rel in staging_graph.relationships:
            # Make a copy to avoid modifying the original
            rel_copy = rel.copy()
            # Map source_id and target_id to source and target
            if 'source_id' in rel_copy:
                rel_copy['source'] = rel_copy.pop('source_id')
            if 'target_id' in rel_copy:
                rel_copy['target'] = rel_copy.pop('target_id')
            # Map relationship_type to type
            if 'relationship_type' in rel_copy:
                rel_copy['type'] = rel_copy.pop('relationship_type')
            # Generate an ID if not present
            if 'id' not in rel_copy:
                rel_copy['id'] = f"edge_{rel_copy['source']}_{rel_copy['target']}_{rel_copy.get('type', 'unknown')}"
            edges.append(rel_copy)
        
        print(f"DEBUG: Converted {len(staging_graph.relationships)} relationships to {len(edges)} edges")
        
        # Create graph response with converted data
        graph = GraphResponse(
            nodes=[Node(
                id=node['id'],
                label=node['label'],
                type=node['type'],
                properties={k: v for k, v in node.items() if k not in ['id', 'label', 'type']}
            ) for node in nodes_with_label],
            edges=[Edge(**edge) for edge in edges]
        )
        
        print(f"DEBUG: Graph structure: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        if graph.nodes:
            print(f"DEBUG: First node properties: {graph.nodes[0].properties}")
        if graph.edges:
            print(f"DEBUG: First edge: {graph.edges[0].source} -> {graph.edges[0].target} ({graph.edges[0].type})")
        
        # 3. Apply resolutions to the graph
        if resolved_conflicts:
            graph = await self._apply_resolutions(graph, resolved_conflicts)
        
        # 4. Execute batch merge
        result = await self._execute_batch_merge(
            merge_id, 
            graph, 
            batch_size, 
            max_retries, 
            retry_delay
        )
        
        # Add transaction ID to result
        if transaction_id:
            result["transaction_id"] = transaction_id
        
        # Update progress tracker
        await self.progress_tracker.complete_merge_stage(
            merge_id, 
            MergeStage.MERGE, 
            metadata=result
        )
        
        return result
    
    async def _get_resolved_conflicts(self, merge_id: str) -> List[Conflict]:
        """Get all resolved conflicts for a merge
        
        Args:
            merge_id: ID of the merge operation
            
        Returns:
            List of resolved conflicts
        """
        # Get all conflicts for the merge
        # This would typically come from a conflict storage service
        # For now, we'll use Redis directly
        import redis.asyncio as redis
        from app.utils.redis import get_redis_client
        
        redis_client = await get_redis_client()
        conflict_keys = await redis_client.keys(f"merge:{merge_id}:conflict:*")
        
        conflicts = []
        for key in conflict_keys:
            conflict_json = await redis_client.get(key)
            if conflict_json:
                conflict = Conflict.model_validate_json(conflict_json)
                if conflict.resolved:
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _apply_resolutions(
        self,
        staging_graph: GraphResponse,
        resolved_conflicts: List[Conflict]
    ) -> GraphResponse:
        """Apply conflict resolutions to create final graph for merging
        
        Args:
            staging_graph: Original staging graph
            resolved_conflicts: List of resolved conflicts
            
        Returns:
            Modified graph with resolutions applied
        """
        # Get copy of graph to modify
        resolved_graph = copy.deepcopy(staging_graph)
        
        # Process each conflict
        for conflict in resolved_conflicts:
            if not conflict.resolved or not conflict.resolution:
                continue
                
            # Use the resolution directly from the conflict
            resolution = conflict.resolution
            
            # Apply resolution based on type
            await self._apply_resolution(resolved_graph, conflict, resolution)
            
        return resolved_graph
    
    async def _apply_resolution(
        self,
        graph: GraphResponse,
        conflict: Conflict,
        resolution: ResolutionOption
    ) -> None:
        """Apply a specific resolution to the graph
        
        Args:
            graph: Graph to modify
            conflict: Conflict to resolve
            resolution: Resolution to apply
        """
        resolution_type = resolution.resolution_type
        resolution_data = resolution.resolution_data
        
        # Handle different resolution types
        if resolution_type == "keep_staging":
            # No changes needed - staging value is already in the graph
            pass
            
        elif resolution_type == "keep_production":
            # Replace staging value with production value
            if conflict.entity_id and conflict.property_name:
                node = graph.get_node_by_id(conflict.entity_id)
                if node and conflict.production_value is not None:
                    node.properties[conflict.property_name] = conflict.production_value
                    
        elif resolution_type == "merge_values":
            # Merge values (e.g., for arrays or objects)
            if conflict.entity_id and conflict.property_name:
                node = graph.get_node_by_id(conflict.entity_id)
                if node:
                    # Handle different types of merges based on value type
                    if isinstance(conflict.staging_value, list) and isinstance(conflict.production_value, list):
                        # Merge lists
                        merged_list = list(set(conflict.staging_value + conflict.production_value))
                        node.properties[conflict.property_name] = merged_list
                    elif isinstance(conflict.staging_value, dict) and isinstance(conflict.production_value, dict):
                        # Merge dictionaries
                        merged_dict = {**conflict.production_value, **conflict.staging_value}
                        node.properties[conflict.property_name] = merged_dict
                        
        elif resolution_type == "keep_staging_rel":
            # Keep staging relationship - no changes needed
            pass
            
        elif resolution_type == "keep_production_rel":
            # Replace with production relationship
            if conflict.entity_id:
                # Find and remove the staging relationship
                if conflict.staging_ids:
                    for rel_id in conflict.staging_ids:
                        # Find the edge index
                        for i, edge in enumerate(graph.edges):
                            if edge.id == rel_id:
                                # Remove the edge
                                graph.edges.pop(i)
                                break
                                
        elif resolution_type == "keep_both_rels":
            # Keep both relationships - no changes needed
            pass
            
        elif resolution_type == "custom":
            # Apply custom resolution logic
            if resolution_data.get("custom_action") == "rename_property":
                if conflict.entity_id and conflict.property_name:
                    node = graph.get_node_by_id(conflict.entity_id)
                    if node and resolution_data.get("new_name"):
                        # Rename property
                        new_name = resolution_data["new_name"]
                        if conflict.property_name in node.properties:
                            node.properties[new_name] = node.properties[conflict.property_name]
                            del node.properties[conflict.property_name]
    
    async def _execute_batch_merge(
        self,
        merge_id: str,
        graph: GraphResponse,
        batch_size: int,
        max_retries: int,
        retry_delay: int
    ) -> Dict[str, Any]:
        """Execute merge in batches to handle large graphs
        
        Args:
            merge_id: ID of the merge operation
            graph: Graph to merge
            batch_size: Number of items to process in each batch
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay in seconds between retries
            
        Returns:
            Dict containing merge statistics
        """
        total_nodes = len(graph.nodes)
        total_edges = len(graph.edges)
        total_items = total_nodes + total_edges
        
        print(f"DEBUG: Graph structure: {total_nodes} nodes, {total_edges} edges")
        if total_edges > 0:
            print(f"DEBUG: First edge: {graph.edges[0].source} -> {graph.edges[0].target} ({graph.edges[0].type})")
        
        # Track progress
        nodes_processed = 0
        edges_processed = 0
        failed_nodes = []
        failed_edges = []
        
        # Process nodes in batches
        for i in range(0, total_nodes, batch_size):
            batch = graph.nodes[i:i+batch_size]
            
            # Try to merge the batch with retries
            success, failed = await self._merge_node_batch(batch, max_retries, retry_delay)
            
            # Update counters
            nodes_processed += len(batch) - len(failed)
            failed_nodes.extend(failed)
            
            # Update progress
            await self.progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.MERGE,
                nodes_processed + edges_processed,
                total_items,
                {"task": "merging_nodes", "batch": i // batch_size + 1}
            )
        
        # Process edges in batches
        for i in range(0, total_edges, batch_size):
            batch = graph.edges[i:i+batch_size]
            
            print(f"DEBUG: Processing edge batch {i // batch_size + 1} with {len(batch)} edges")
            
            # Try to merge the batch with retries
            success, failed = await self._merge_edge_batch(batch, max_retries, retry_delay)
            
            # Update counters
            edges_processed += len(batch) - len(failed)
            failed_edges.extend(failed)
            
            print(f"DEBUG: Edge batch {i // batch_size + 1} complete: {len(batch) - len(failed)} merged, {len(failed)} failed")
            
            # Update progress
            await self.progress_tracker.update_merge_progress(
                merge_id,
                MergeStage.MERGE,
                nodes_processed + edges_processed,
                total_items,
                {"task": "merging_edges", "batch": i // batch_size + 1}
            )
        
        return {
            "nodes_merged": nodes_processed,
            "edges_merged": edges_processed,
            "nodes_failed": len(failed_nodes),
            "edges_failed": len(failed_edges),
            "total_items": total_items,
            "success_rate": (nodes_processed + edges_processed) / total_items if total_items > 0 else 1.0
        }
    
    async def _merge_node_batch(
        self,
        nodes: List[Node],
        max_retries: int,
        retry_delay: int
    ) -> Tuple[List[Node], List[Node]]:
        """Merge a batch of nodes into production
        
        Args:
            nodes: List of nodes to merge
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay in seconds between retries
            
        Returns:
            Tuple of (successful nodes, failed nodes)
        """
        successful = []
        failed = []
        
        print(f"DEBUG: Starting to merge {len(nodes)} nodes")
        
        for node in nodes:
            print(f"DEBUG: Processing node {node.id} with properties: {node.properties}")
            
            # Check if node already exists in production
            existing_node = await self.prod_storage.get_node_by_id(node.id)
            
            try:
                if existing_node:
                    print(f"DEBUG: Node {node.id} already exists, updating properties")
                    # Update existing node
                    await self.prod_storage.update_node(node.id, node.properties)
                else:
                    print(f"DEBUG: Node {node.id} does not exist, creating new node")
                    # Create new node
                    await self.prod_storage.create_node(
                        label=node.label,
                        properties={**node.properties, "id": node.id}
                    )
                successful.append(node)
            except Exception as e:
                # Log error and add to failed list
                logger.error(f"Failed to merge node {node.id}: {str(e)}")
                print(f"DEBUG: Failed to merge node {node.id}: {str(e)}")
                
                # Retry logic
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        retry_count += 1
                        await asyncio.sleep(retry_delay)
                        
                        if existing_node:
                            await self.prod_storage.update_node(node.id, node.properties)
                        else:
                            await self.prod_storage.create_node(
                                label=node.label,
                                properties={**node.properties, "id": node.id}
                            )
                        
                        # If successful, add to successful list and break
                        successful.append(node)
                        break
                    except Exception as retry_e:
                        logger.error(f"Retry {retry_count} failed for node {node.id}: {str(retry_e)}")
                        
                        # If we've exhausted retries, add to failed list
                        if retry_count >= max_retries:
                            failed.append(node)
        
        return successful, failed
    
    async def _merge_edge_batch(
        self,
        edges: List[Edge],
        max_retries: int,
        retry_delay: int
    ) -> Tuple[List[Edge], List[Edge]]:
        """Merge a batch of edges into production
        
        Args:
            edges: List of edges to merge
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay in seconds between retries
            
        Returns:
            Tuple of (successful edges, failed edges)
        """
        successful = []
        failed = []
        
        logger.info(f"Starting to merge {len(edges)} edges")
        print(f"DEBUG: Starting to merge {len(edges)} edges")
        
        for edge in edges:
            # Check if source and target nodes exist in production
            logger.info(f"Checking nodes for edge {edge.id} from {edge.source} to {edge.target}")
            print(f"DEBUG: Checking nodes for edge {edge.id} from {edge.source} to {edge.target}")
            source_node = await self.prod_storage.get_node_by_id(edge.source)
            target_node = await self.prod_storage.get_node_by_id(edge.target)
            
            if not source_node:
                logger.error(f"Source node {edge.source} not found for edge {edge.id}")
                print(f"DEBUG: Source node {edge.source} not found for edge {edge.id}")
            
            if not target_node:
                logger.error(f"Target node {edge.target} not found for edge {edge.id}")
                print(f"DEBUG: Target node {edge.target} not found for edge {edge.id}")
            
            if not source_node or not target_node:
                # Can't create edge if nodes don't exist
                logger.error(f"Cannot create edge {edge.id}: source or target node missing")
                print(f"DEBUG: Cannot create edge {edge.id}: source or target node missing")
                failed.append(edge)
                continue
            
            # Check if edge already exists
            logger.info(f"Checking if edge already exists between {edge.source} and {edge.target}")
            print(f"DEBUG: Checking if edge already exists between {edge.source} and {edge.target}")
            existing_edges = await self.prod_storage.get_edges_between(edge.source, edge.target)
            existing_edge = next((e for e in existing_edges if e.type == edge.type), None)
            
            try:
                if existing_edge:
                    logger.info(f"Edge already exists with ID {existing_edge.id}, updating properties")
                    print(f"DEBUG: Edge already exists with ID {existing_edge.id}, updating properties")
                    # Update existing edge properties
                    # Note: This assumes there's an update_edge method in the storage interface
                    # You might need to adapt this based on your actual storage interface
                    if hasattr(self.prod_storage, 'update_edge'):
                        await self.prod_storage.update_edge(existing_edge.id, edge.properties)
                    else:
                        # Alternative: delete and recreate
                        logger.info(f"Deleting existing edge {existing_edge.id} and recreating")
                        print(f"DEBUG: Deleting existing edge {existing_edge.id} and recreating")
                        await self.prod_storage.delete_relationship(existing_edge.id)
                        await self.prod_storage.create_relationship(
                            source_id=edge.source,
                            target_id=edge.target,
                            rel_type=edge.type,
                            properties=edge.properties
                        )
                else:
                    # Create new edge
                    logger.info(f"Creating new edge from {edge.source} to {edge.target} of type {edge.type}")
                    print(f"DEBUG: Creating new edge from {edge.source} to {edge.target} of type {edge.type}")
                    await self.prod_storage.create_relationship(
                        source_id=edge.source,
                        target_id=edge.target,
                        rel_type=edge.type,
                        properties=edge.properties
                    )
                successful.append(edge)
            except Exception as e:
                # Log error and add to failed list
                logger.error(f"Failed to merge edge {edge.id}: {str(e)}")
                print(f"DEBUG: Failed to merge edge {edge.id}: {str(e)}")
                logger.exception(e)
                failed.append(edge)
        
        logger.info(f"Edge merge complete: {len(successful)} successful, {len(failed)} failed")
        print(f"DEBUG: Edge merge complete: {len(successful)} successful, {len(failed)} failed")
        return successful, failed
    
    async def cancel_merge(self, merge_id: str) -> bool:
        """Cancel an in-progress merge operation
        
        Args:
            merge_id: ID of the merge operation to cancel
            
        Returns:
            bool: True if successfully cancelled, False otherwise
        """
        try:
            # Get current progress
            progress = await self.progress_tracker.get_progress(merge_id)
            
            if not progress:
                logger.warning(f"Cannot cancel merge {merge_id}: not found")
                return False
                
            if progress.overall_status in [MergeStatus.COMPLETED, MergeStatus.FAILED]:
                # If the merge is already completed or failed, we can't cancel it
                logger.warning(f"Cannot cancel merge {merge_id}: already {progress.overall_status}")
                return False
                
            # Mark as cancelled
            # Use the cancel_merge method in the progress tracker
            await self.progress_tracker.cancel_merge(
                merge_id,
                reason="Cancelled by user",
                metadata={"cancelled_at": time.time()}
            )
                
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling merge {merge_id}: {str(e)}")
            return False 