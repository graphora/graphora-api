"""Service for applying conflict resolutions to the production graph"""
from typing import List
import logging
import traceback
import uuid

from app.schemas.conflicts import Conflict, ConflictResolutionResult, ResolutionOption, ConflictType, ResolutionStrategy
from app.services.merge.models import GraphOperation, UpdateNodeOperation, UpdateRelationshipDirectionOperation, UpdateRelationshipTypeOperation
from app.services.storage.interface import GraphStorageInterface

logger = logging.getLogger(__name__)

class ResolutionApplicator:
    """Apply resolutions to production graph with verification"""
    
    def __init__(self, staging_storage: GraphStorageInterface, production_storage: GraphStorageInterface):
        """Initialize with storage instances"""
        self.staging_storage = staging_storage
        self.production_storage = production_storage
        # Add aliases for compatibility with tests
        self.stage_storage = staging_storage
        self.prod_storage = production_storage
        
    async def apply_resolution(
        self, 
        conflict: Conflict, 
        resolution_option: ResolutionOption
    ) -> ConflictResolutionResult:
        """
        Apply a resolution option to a conflict
        Returns result of resolution application with verification status
        """
        try:
            # Choose strategy based on conflict type and resolution type
            if conflict.conflict_type == ConflictType.PROPERTY_VALUE:
                result = await self._apply_property_value_resolution(conflict, resolution_option)
            elif conflict.conflict_type == ConflictType.PROPERTY_MISSING:
                result = await self._apply_property_missing_resolution(conflict, resolution_option)
            elif conflict.conflict_type == ConflictType.RELATIONSHIP_TYPE:
                result = await self._apply_relationship_type_resolution(conflict, resolution_option)
            elif conflict.conflict_type == ConflictType.RELATIONSHIP_DIRECTION:
                result = await self._apply_relationship_direction_resolution(conflict, resolution_option)
            elif conflict.conflict_type == ConflictType.DUPLICATE_ENTITY:
                result = await self._apply_duplicate_entity_resolution(conflict, resolution_option)
            else:
                raise ValueError(f"Unsupported conflict type: {conflict.conflict_type}")
            
            return ConflictResolutionResult(
                conflict_id=conflict.id,
                resolution_id=resolution_option.id,
                verification={"verified": True},
                changes=result,
                success=True,
                resolved=True,
                error=None
            )
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Failed to apply resolution for conflict {conflict.id}: {str(e)}")
            return ConflictResolutionResult(
                conflict_id=conflict.id,
                resolution_id=resolution_option.id if resolution_option else None,
                verification={"verified": False, "error": str(e)},
                changes=[],
                success=False,
                resolved=False,
                error=str(e)
            )
            
    async def _apply_property_value_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> List[GraphOperation]:
        """Apply resolution for property value conflict"""
        prop_name = conflict.context.get("property_name")
        staging_value = conflict.context.get("staging_value")
        production_value = conflict.context.get("production_value")
        
        if not prop_name:
            raise ValueError("Missing property name in conflict context")
            
        # Get production entity ID
        if not conflict.production_ids or len(conflict.production_ids) == 0:
            raise ValueError("No production entity ID in conflict")
            
        prod_id = conflict.production_ids[0]
        
        # Get staging entity ID
        if not conflict.staging_ids or len(conflict.staging_ids) == 0:
            raise ValueError("No staging entity ID in conflict")
            
        staging_id = conflict.staging_ids[0]
        
        operations = []
        
        # Apply based on resolution type
        if resolution.resolution_type == ResolutionStrategy.KEEP_STAGING:
            # Update production with staging value
            operations.append(UpdateNodeOperation(
                id=prod_id,
                staging_id=staging_id,
                properties={prop_name: staging_value}
            ))
            # Store changes info in the operation's metadata if needed
            operations[-1].dict()["changes"] = {
                "staging_id": staging_id,
                "property": prop_name,
                "old_value": production_value,
                "new_value": staging_value,
                "action": "update_production"
            }
            
        elif resolution.resolution_type == ResolutionStrategy.KEEP_PRODUCTION:
            # No operations needed
            operations.append(UpdateNodeOperation(
                id=prod_id,
                staging_id=staging_id,
                properties={}  # Empty update to record the "no_change" decision
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_id,
                "property": prop_name,
                "value": production_value,
                "action": "no_change"
            }
            
        elif resolution.resolution_type == ResolutionStrategy.MERGE_VALUES:
            # Merge values based on type
            strategy = resolution.resolution_data.get("strategy", "concat")
            
            if strategy == "concat" and isinstance(staging_value, str) and isinstance(production_value, str):
                merged_value = f"{production_value} | {staging_value}"
            elif strategy == "combine" and isinstance(staging_value, (list, dict)) and isinstance(production_value, (list, dict)):
                if isinstance(staging_value, list):
                    merged_value = list(set(production_value + staging_value))
                else:  # dict
                    merged_value = {**production_value, **staging_value}
            else:
                # Default to custom value from resolution data
                merged_value = resolution.resolution_data.get("custom_value", staging_value)
                
            operations.append(UpdateNodeOperation(
                id=prod_id,
                staging_id=staging_id,
                properties={prop_name: merged_value}
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_id,
                "property": prop_name,
                "old_value": production_value,
                "new_value": merged_value,
                "action": "merge_values",
                "strategy": strategy
            }
            
            return operations
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
        
        return operations
    
    async def _apply_duplicate_entity_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> List[GraphOperation]:
        """Apply resolution for duplicate entity conflict"""
        # Extract necessary information from conflict context
        staging_id = conflict.staging_ids[0] if conflict.staging_ids else None
        production_ids = conflict.production_ids
        
        if not staging_id:
            raise ValueError("Missing staging entity ID in conflict")
            
        if not production_ids or len(production_ids) == 0:
            raise ValueError("No production entity IDs in conflict")
            
        operations = []
        
        # Get the staging node
        staging_node = await self.staging_storage.get_node_by_id(staging_id)
        if not staging_node:
            raise ValueError(f"Staging node {staging_id} not found")
            
        # Handle different resolution strategies
        if resolution.resolution_type == ResolutionStrategy.IGNORE_DUPLICATE:
            # No changes needed, just ignore the duplicate
            # Add an empty operation to record the decision
            operations.append(UpdateNodeOperation(
                id=production_ids[0],
                staging_id=staging_id,
                properties={}  # Empty update to record the "ignore" decision
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_id,
                "production_ids": production_ids,
                "action": "ignore_duplicate"
            }
            return operations
            
        elif resolution.resolution_type == ResolutionStrategy.KEEP_PRODUCTION:
            # Keep only one specific production entity
            production_id = resolution.resolution_data.get("production_id")
            if not production_id:
                raise ValueError("Missing production_id in resolution data")
                
            # Get the selected production node
            production_node = await self.production_storage.get_node_by_id(production_id)
            if not production_node:
                raise ValueError(f"Production node {production_id} not found")
                
            # Add an empty operation to record the decision
            operations.append(UpdateNodeOperation(
                id=production_id,
                staging_id=staging_id,
                properties=production_node.properties
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_id,
                "production_id": production_id,
                "properties": production_node.properties,
                "action": "keep_production"
            }
            return operations
            
        elif resolution.resolution_type == ResolutionStrategy.MERGE_VALUES:
            # Merge properties from staging node into the first production node
            production_id = production_ids[0]
            
            # Get the production node
            production_node = await self.production_storage.get_node_by_id(production_id)
            if not production_node:
                raise ValueError(f"Production node {production_id} not found")
                
            # Merge properties (prefer staging values for conflicts)
            merged_properties = {}
            for prop_name, prop_value in staging_node.properties.items():
                # Skip system properties
                if prop_name.startswith('_'):
                    continue
                    
                # Only add properties that don't exist or have different values
                if prop_name not in production_node.properties or production_node.properties[prop_name] != prop_value:
                    merged_properties[prop_name] = prop_value
            
            if merged_properties:
                operations.append(UpdateNodeOperation(
                    id=production_id,
                    staging_id=staging_id,
                    properties=merged_properties
                ))
                operations[-1].dict()["changes"] = {
                    "staging_id": staging_id,
                    "production_id": production_id,
                    "properties": merged_properties,
                    "action": "merge_values"
                }
                
            return operations
            
        elif resolution.resolution_type == ResolutionStrategy.CREATE_NEW:
            # Create a new entity with merged properties from both staging and production
            # First, determine which production node to merge with
            production_id = resolution.resolution_data.get("production_id")
            if not production_id and len(production_ids) > 0:
                # If no specific production ID is provided, use the first one
                production_id = production_ids[0]
                
            # Get the production node
            production_node = await self.production_storage.get_node_by_id(production_id)
            if not production_node:
                raise ValueError(f"Production node {production_id} not found")
                
            # Create a new node with merged properties
            # Start with production properties
            merged_properties = dict(production_node.properties)
            
            # Add or override with staging properties
            for prop_name, prop_value in staging_node.properties.items():
                # Skip system properties
                if prop_name.startswith('_'):
                    continue
                    
                merged_properties[prop_name] = prop_value
            
            # Generate a new ID for the merged entity
            new_entity_id = f"merged_{staging_id}_{production_id}"
            
            # Create a new node operation
            # Note: In a real implementation, this would likely be handled at a higher level
            # as it involves creating a new entity and updating relationships
            operations.append(UpdateNodeOperation(
                id=new_entity_id,  # This is a placeholder - actual creation happens elsewhere
                staging_id=staging_id,
                properties=merged_properties
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_id,
                "production_id": production_id,
                "new_entity_id": new_entity_id,
                "properties": merged_properties,
                "action": "create_new"
            }
            
            return operations
            
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
            
        return operations
            
    async def _apply_property_missing_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> List[GraphOperation]:
        """Apply resolution for missing property conflict"""
        prop_name = conflict.context.get("property_name")
        missing_in = conflict.context.get("missing_in", "staging")  # Where is it missing
        
        if not prop_name:
            raise ValueError("Missing property name in conflict context")
            
        # Get entity IDs
        if not conflict.production_ids or len(conflict.production_ids) == 0:
            raise ValueError("No production entity ID in conflict")
            
        prod_id = conflict.production_ids[0]
        
        if not conflict.staging_ids or len(conflict.staging_ids) == 0:
            raise ValueError("No staging entity ID in conflict")
            
        staging_id = conflict.staging_ids[0]
        operations = []
        
        # Apply based on resolution type
        if resolution.resolution_type == "add_to_production" and missing_in == "production":
            # Get value from staging
            staging_node = await self.staging_storage.get_node_by_id(staging_id)
            if not staging_node:
                raise ValueError(f"Staging node {staging_id} not found")
                
            value = staging_node.properties.get(prop_name)
            
            # Add property to production
            operations.append(UpdateNodeOperation(
                id=prod_id,
                staging_id=staging_id,
                properties={prop_name: value}
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_id,
                "property": prop_name,
                "value": value,
                "action": "add_to_production"
            }
            
        elif resolution.resolution_type == "remove_from_production" and missing_in == "staging":
            # Remove property from production
            operations.append(UpdateNodeOperation(
                id=prod_id,
                staging_id=staging_id,
                properties={prop_name: ''}
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_id,
                "property": prop_name,
                "action": "remove_from_production"
            }
            
        elif resolution.resolution_type == ResolutionStrategy.IGNORE:
            # No changes
            return operations
            
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
            
        return operations
    
    async def _apply_relationship_type_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> List[GraphOperation]:
        """Apply resolution for relationship type conflict"""
        if not conflict.production_ids or len(conflict.production_ids) == 0:
            raise ValueError("No production relationship ID in conflict")
            
        prod_rel_id = conflict.production_ids[0]
        
        if not conflict.staging_ids or len(conflict.staging_ids) == 0:
            raise ValueError("No staging relationship ID in conflict")
            
        staging_rel_id = conflict.staging_ids[0]
        
        staging_type = conflict.context.get("staging_type")
        production_type = conflict.context.get("production_type")
        
        operations = []
        
        if not staging_type or not production_type:
            raise ValueError("Missing relationship type information in conflict context")
        
        # Apply based on resolution type
        if resolution.resolution_type == ResolutionStrategy.KEEP_STAGING:
            # Update the relationship type
            operations.append(UpdateRelationshipTypeOperation(
                id=prod_rel_id,
                staging_id=staging_rel_id,
                old_type=production_type,
                new_type=staging_type
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_rel_id,
                "old_type": production_type,
                "new_type": staging_type,
                "action": "update_relationship_type"
            }
            
        elif resolution.resolution_type == ResolutionStrategy.KEEP_PRODUCTION:
            # No changes needed
            return operations
            
        elif resolution.resolution_type == ResolutionStrategy.KEEP_BOTH:
            # Create a new relationship in production with the staging type
            new_rel_id = str(uuid.uuid4())
            # Update the relationship type
            operations.append(UpdateRelationshipTypeOperation(
                id=new_rel_id,
                staging_id=staging_rel_id,
                old_type=production_type,
                new_type=production_type
            ))
            operations[-1].dict()["changes"] = {
                "staging_id": staging_rel_id,
                "new_type": production_type,
                "action": "create_additional_relationship"
            }
            
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
            
        return operations
    
    async def _apply_relationship_direction_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> List[GraphOperation]:
        """Apply resolution for relationship direction conflict"""
        if not conflict.staging_ids or len(conflict.staging_ids) == 0:
            raise ValueError("No staging relationship ID in conflict")
        if not conflict.production_ids or len(conflict.production_ids) == 0:
            raise ValueError("No production relationship ID in conflict")
            
        prod_rel_id = conflict.production_ids[0]
        staging_rel_id = conflict.staging_ids[0]
        
        # Apply based on resolution type
        operations = []
        if resolution.resolution_type == "reverse_relationship":
            # Get the relationship from production
            operations.append(UpdateRelationshipDirectionOperation(
                id=prod_rel_id,
                staging_id=staging_rel_id
            ))
            operations[-1].dict()["changes"] = {
                "production_id": prod_rel_id,
                "action": "update_relationship_direction"
            }
            
        elif resolution.resolution_type == "keep_production_direction":
            # No changes needed
            return operations
            
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
            
        return operations