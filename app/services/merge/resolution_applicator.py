"""Service for applying conflict resolutions to the production graph"""
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import pytz
import uuid

from app.schemas.conflicts import Conflict, ResolutionOption, ConflictType
from app.schemas.graph import Node, Edge
from app.services.storage.interface import GraphStorageInterface

logger = logging.getLogger(__name__)

class ResolutionApplicator:
    """Apply resolutions to production graph with verification"""
    
    def __init__(self, staging_storage: GraphStorageInterface, prod_storage: GraphStorageInterface):
        """Initialize with storage instances"""
        self.staging_storage = staging_storage
        self.prod_storage = prod_storage
        
    async def apply_resolution(
        self, 
        conflict: Conflict, 
        resolution_option: ResolutionOption
    ) -> Dict[str, Any]:
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
                
            # Verify resolution was applied correctly
            verification = await self._verify_resolution(conflict, resolution_option, result)
            
            return {
                "applied": True,
                "conflict_id": conflict.id,
                "resolution_id": resolution_option.id,
                "verification": verification,
                "changes": result
            }
            
        except Exception as e:
            logger.error(f"Failed to apply resolution for conflict {conflict.id}: {str(e)}")
            return {
                "applied": False,
                "conflict_id": conflict.id,
                "resolution_id": resolution_option.id if resolution_option else None,
                "error": str(e)
            }
            
    async def _apply_property_value_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> Dict[str, Any]:
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
        
        # Apply based on resolution type
        if resolution.resolution_type == "keep_staging":
            # Update production with staging value
            await self.prod_storage.update_node_property(
                prod_id, 
                prop_name, 
                staging_value
            )
            return {
                "property": prop_name,
                "old_value": production_value,
                "new_value": staging_value,
                "action": "updated_production"
            }
            
        elif resolution.resolution_type == "keep_production":
            # No changes needed
            return {
                "property": prop_name,
                "value": production_value,
                "action": "no_change"
            }
            
        elif resolution.resolution_type == "merge_values":
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
                
            await self.prod_storage.update_node_property(
                prod_id, 
                prop_name, 
                merged_value
            )
            
            return {
                "property": prop_name,
                "old_value": production_value,
                "new_value": merged_value,
                "action": "merged_values",
                "strategy": strategy
            }
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
            
    async def _apply_property_missing_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> Dict[str, Any]:
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
        
        # Apply based on resolution type
        if resolution.resolution_type == "add_to_production" and missing_in == "production":
            # Get value from staging
            staging_node = await self.staging_storage.get_node_by_id(staging_id)
            if not staging_node:
                raise ValueError(f"Staging node {staging_id} not found")
                
            value = staging_node.properties.get(prop_name)
            
            # Add property to production
            await self.prod_storage.update_node_property(prod_id, prop_name, value)
            
            return {
                "property": prop_name,
                "value": value,
                "action": "added_to_production"
            }
            
        elif resolution.resolution_type == "remove_from_production" and missing_in == "staging":
            # Remove property from production
            await self.prod_storage.remove_node_property(prod_id, prop_name)
            
            return {
                "property": prop_name,
                "action": "removed_from_production"
            }
            
        elif resolution.resolution_type == "ignore":
            # No changes
            return {
                "property": prop_name,
                "action": "no_change"
            }
            
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
    
    async def _apply_relationship_type_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> Dict[str, Any]:
        """Apply resolution for relationship type conflict"""
        if not conflict.production_ids or len(conflict.production_ids) == 0:
            raise ValueError("No production relationship ID in conflict")
            
        prod_rel_id = conflict.production_ids[0]
        
        if not conflict.staging_ids or len(conflict.staging_ids) == 0:
            raise ValueError("No staging relationship ID in conflict")
            
        staging_rel_id = conflict.staging_ids[0]
        
        staging_type = conflict.context.get("staging_type")
        production_type = conflict.context.get("production_type")
        
        if not staging_type or not production_type:
            raise ValueError("Missing relationship type information in conflict context")
        
        # Apply based on resolution type
        if resolution.resolution_type == "keep_staging_rel_type":
            # Get the relationship from production
            prod_rel = await self.prod_storage.get_relationship_by_id(prod_rel_id)
            if not prod_rel:
                raise ValueError(f"Production relationship {prod_rel_id} not found")
                
            # Update the relationship type
            await self.prod_storage.update_relationship_type(
                prod_rel_id,
                staging_type
            )
            
            return {
                "old_type": production_type,
                "new_type": staging_type,
                "action": "updated_relationship_type"
            }
            
        elif resolution.resolution_type == "keep_production_rel_type":
            # No changes needed
            return {
                "type": production_type,
                "action": "no_change"
            }
            
        elif resolution.resolution_type == "keep_both_relationships":
            # Get the staging relationship
            staging_rel = await self.staging_storage.get_relationship_by_id(staging_rel_id)
            if not staging_rel:
                raise ValueError(f"Staging relationship {staging_rel_id} not found")
                
            # Create a new relationship in production with the staging type
            new_rel_id = str(uuid.uuid4())
            await self.prod_storage.create_relationship(
                source_id=staging_rel.source,
                target_id=staging_rel.target,
                rel_type=staging_type,
                properties=staging_rel.properties
            )
            
            return {
                "existing_type": production_type,
                "new_type": staging_type,
                "action": "created_additional_relationship"
            }
            
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
    
    async def _apply_relationship_direction_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> Dict[str, Any]:
        """Apply resolution for relationship direction conflict"""
        if not conflict.production_ids or len(conflict.production_ids) == 0:
            raise ValueError("No production relationship ID in conflict")
            
        prod_rel_id = conflict.production_ids[0]
        
        # Apply based on resolution type
        if resolution.resolution_type == "reverse_relationship":
            # Get the relationship from production
            prod_rel = await self.prod_storage.get_relationship_by_id(prod_rel_id)
            if not prod_rel:
                raise ValueError(f"Production relationship {prod_rel_id} not found")
                
            # Create a new relationship with reversed direction
            await self.prod_storage.create_relationship(
                source_id=prod_rel.target,
                target_id=prod_rel.source,
                rel_type=prod_rel.type,
                properties=prod_rel.properties
            )
            
            # Delete the original relationship
            await self.prod_storage.delete_relationship(prod_rel_id)
            
            return {
                "original_source": prod_rel.source,
                "original_target": prod_rel.target,
                "new_source": prod_rel.target,
                "new_target": prod_rel.source,
                "action": "reversed_relationship"
            }
            
        elif resolution.resolution_type == "keep_production_direction":
            # No changes needed
            return {
                "action": "no_change"
            }
            
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
    
    async def _apply_duplicate_entity_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption
    ) -> Dict[str, Any]:
        """Apply resolution for duplicate entity conflict"""
        if not conflict.production_ids or len(conflict.production_ids) < 2:
            raise ValueError("Need at least two production entity IDs for duplicate conflict")
            
        # Get the entities to merge
        entity_ids = conflict.production_ids
        primary_id = entity_ids[0]
        duplicate_ids = entity_ids[1:]
        
        # Apply based on resolution type
        if resolution.resolution_type == "merge_entities":
            # Get all entities
            entities = []
            for entity_id in entity_ids:
                entity = await self.prod_storage.get_node_by_id(entity_id)
                if not entity:
                    raise ValueError(f"Entity {entity_id} not found")
                entities.append(entity)
                
            primary_entity = entities[0]
            
            # Merge properties from all entities
            merged_properties = primary_entity.properties.copy()
            for entity in entities[1:]:
                for key, value in entity.properties.items():
                    if key not in merged_properties:
                        merged_properties[key] = value
                    elif isinstance(value, list) and isinstance(merged_properties[key], list):
                        # Combine lists
                        merged_properties[key] = list(set(merged_properties[key] + value))
                    elif isinstance(value, dict) and isinstance(merged_properties[key], dict):
                        # Combine dicts
                        merged_properties[key].update(value)
            
            # Update primary entity with merged properties
            await self.prod_storage.update_node(primary_id, merged_properties)
            
            # Redirect relationships from duplicates to primary
            for duplicate_id in duplicate_ids:
                # Get all relationships for the duplicate
                incoming_rels = await self.prod_storage.get_incoming_relationships(duplicate_id)
                outgoing_rels = await self.prod_storage.get_outgoing_relationships(duplicate_id)
                
                # Redirect incoming relationships
                for rel in incoming_rels:
                    # Create new relationship to primary
                    await self.prod_storage.create_relationship(
                        source_id=rel.source,
                        target_id=primary_id,
                        rel_type=rel.type,
                        properties=rel.properties
                    )
                    # Delete original relationship
                    await self.prod_storage.delete_relationship(rel.id)
                
                # Redirect outgoing relationships
                for rel in outgoing_rels:
                    # Create new relationship from primary
                    await self.prod_storage.create_relationship(
                        source_id=primary_id,
                        target_id=rel.target,
                        rel_type=rel.type,
                        properties=rel.properties
                    )
                    # Delete original relationship
                    await self.prod_storage.delete_relationship(rel.id)
                
                # Delete the duplicate entity
                await self.prod_storage.delete_node(duplicate_id)
            
            return {
                "primary_id": primary_id,
                "merged_ids": duplicate_ids,
                "action": "merged_entities"
            }
            
        elif resolution.resolution_type == "keep_separate":
            # No changes needed
            return {
                "entity_ids": entity_ids,
                "action": "no_change"
            }
            
        else:
            raise ValueError(f"Unsupported resolution type: {resolution.resolution_type}")
    
    async def _verify_resolution(
        self, 
        conflict: Conflict, 
        resolution: ResolutionOption,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify that resolution was correctly applied"""
        try:
            if conflict.conflict_type == ConflictType.PROPERTY_VALUE:
                return await self._verify_property_value_resolution(conflict, resolution, changes)
            elif conflict.conflict_type == ConflictType.PROPERTY_MISSING:
                return await self._verify_property_missing_resolution(conflict, resolution, changes)
            elif conflict.conflict_type == ConflictType.RELATIONSHIP_TYPE:
                return await self._verify_relationship_type_resolution(conflict, resolution, changes)
            elif conflict.conflict_type == ConflictType.RELATIONSHIP_DIRECTION:
                return await self._verify_relationship_direction_resolution(conflict, resolution, changes)
            elif conflict.conflict_type == ConflictType.DUPLICATE_ENTITY:
                return await self._verify_duplicate_entity_resolution(conflict, resolution, changes)
                
            return {"verified": True}
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return {"verified": False, "error": str(e)}
            
    async def _verify_property_value_resolution(
        self,
        conflict: Conflict,
        resolution: ResolutionOption,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify property value resolution was applied correctly"""
        prop_name = conflict.context.get("property_name")
        prod_id = conflict.production_ids[0]
        
        # Get production node to verify change
        prod_node = await self.prod_storage.get_node_by_id(prod_id)
        if not prod_node:
            return {"verified": False, "error": f"Production node {prod_id} not found"}
            
        current_value = prod_node.properties.get(prop_name)
        expected_value = None
        
        if resolution.resolution_type == "keep_staging":
            expected_value = conflict.context.get("staging_value")
        elif resolution.resolution_type == "keep_production":
            expected_value = conflict.context.get("production_value")
        elif resolution.resolution_type == "merge_values":
            expected_value = changes.get("new_value")
            
        if expected_value is None:
            return {"verified": False, "error": "Could not determine expected value"}
            
        # Compare values (with type consideration)
        if isinstance(current_value, (int, float)) and isinstance(expected_value, (int, float)):
            # Numeric comparison with small tolerance
            is_equal = abs(float(current_value) - float(expected_value)) < 1e-6
        else:
            # Direct comparison for other types
            is_equal = current_value == expected_value
            
        return {
            "verified": is_equal,
            "current_value": current_value,
            "expected_value": expected_value
        }
    
    async def _verify_property_missing_resolution(
        self,
        conflict: Conflict,
        resolution: ResolutionOption,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify property missing resolution was applied correctly"""
        prop_name = conflict.context.get("property_name")
        missing_in = conflict.context.get("missing_in", "staging")
        prod_id = conflict.production_ids[0]
        
        # Get production node to verify change
        prod_node = await self.prod_storage.get_node_by_id(prod_id)
        if not prod_node:
            return {"verified": False, "error": f"Production node {prod_id} not found"}
            
        if resolution.resolution_type == "add_to_production" and missing_in == "production":
            # Property should now exist in production
            has_property = prop_name in prod_node.properties
            return {
                "verified": has_property,
                "property_exists": has_property
            }
            
        elif resolution.resolution_type == "remove_from_production" and missing_in == "staging":
            # Property should no longer exist in production
            has_property = prop_name in prod_node.properties
            return {
                "verified": not has_property,
                "property_exists": has_property
            }
            
        elif resolution.resolution_type == "ignore":
            # No changes expected
            return {"verified": True}
            
        return {"verified": False, "error": "Unsupported resolution verification"}
    
    async def _verify_relationship_type_resolution(
        self,
        conflict: Conflict,
        resolution: ResolutionOption,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify relationship type resolution was applied correctly"""
        if not conflict.production_ids or len(conflict.production_ids) == 0:
            return {"verified": False, "error": "No production relationship ID in conflict"}
            
        prod_rel_id = conflict.production_ids[0]
        staging_type = conflict.context.get("staging_type")
        production_type = conflict.context.get("production_type")
        
        if resolution.resolution_type == "keep_staging_rel_type":
            # Get the relationship from production to verify type
            prod_rel = await self.prod_storage.get_relationship_by_id(prod_rel_id)
            if not prod_rel:
                return {"verified": False, "error": f"Production relationship {prod_rel_id} not found"}
                
            return {
                "verified": prod_rel.type == staging_type,
                "current_type": prod_rel.type,
                "expected_type": staging_type
            }
            
        elif resolution.resolution_type == "keep_production_rel_type":
            # No changes expected
            prod_rel = await self.prod_storage.get_relationship_by_id(prod_rel_id)
            if not prod_rel:
                return {"verified": False, "error": f"Production relationship {prod_rel_id} not found"}
                
            return {
                "verified": prod_rel.type == production_type,
                "current_type": prod_rel.type,
                "expected_type": production_type
            }
            
        elif resolution.resolution_type == "keep_both_relationships":
            # Both relationship types should exist
            prod_rel = await self.prod_storage.get_relationship_by_id(prod_rel_id)
            if not prod_rel:
                return {"verified": False, "error": f"Production relationship {prod_rel_id} not found"}
                
            # Get source and target from the original relationship
            source_id = prod_rel.source
            target_id = prod_rel.target
            
            # Find relationships between these nodes
            relationships = await self.prod_storage.get_relationships_between(source_id, target_id)
            
            # Check if both types exist
            types = [rel.type for rel in relationships]
            has_production_type = production_type in types
            has_staging_type = staging_type in types
            
            return {
                "verified": has_production_type and has_staging_type,
                "has_production_type": has_production_type,
                "has_staging_type": has_staging_type
            }
            
        return {"verified": False, "error": "Unsupported resolution verification"}
    
    async def _verify_relationship_direction_resolution(
        self,
        conflict: Conflict,
        resolution: ResolutionOption,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify relationship direction resolution was applied correctly"""
        if resolution.resolution_type == "reverse_relationship":
            # Original relationship should be gone, and a new one with reversed direction should exist
            original_source = changes.get("original_source")
            original_target = changes.get("original_target")
            
            if not original_source or not original_target:
                return {"verified": False, "error": "Missing source/target information in changes"}
                
            # Check that original relationship is gone
            original_rels = await self.prod_storage.get_relationships_between(original_source, original_target)
            
            # Check that reversed relationship exists
            reversed_rels = await self.prod_storage.get_relationships_between(original_target, original_source)
            
            return {
                "verified": len(original_rels) == 0 and len(reversed_rels) > 0,
                "original_relationship_exists": len(original_rels) > 0,
                "reversed_relationship_exists": len(reversed_rels) > 0
            }
            
        elif resolution.resolution_type == "keep_production_direction":
            # No changes expected
            return {"verified": True}
            
        return {"verified": False, "error": "Unsupported resolution verification"}
    
    async def _verify_duplicate_entity_resolution(
        self,
        conflict: Conflict,
        resolution: ResolutionOption,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify duplicate entity resolution was applied correctly"""
        if resolution.resolution_type == "merge_entities":
            primary_id = changes.get("primary_id")
            merged_ids = changes.get("merged_ids", [])
            
            if not primary_id or not merged_ids:
                return {"verified": False, "error": "Missing primary_id or merged_ids in changes"}
                
            # Check that primary entity exists
            primary_entity = await self.prod_storage.get_node_by_id(primary_id)
            if not primary_entity:
                return {"verified": False, "error": f"Primary entity {primary_id} not found"}
                
            # Check that merged entities no longer exist
            merged_entities_exist = []
            for entity_id in merged_ids:
                entity = await self.prod_storage.get_node_by_id(entity_id)
                if entity:
                    merged_entities_exist.append(entity_id)
                    
            return {
                "verified": len(merged_entities_exist) == 0,
                "primary_entity_exists": primary_entity is not None,
                "merged_entities_still_exist": merged_entities_exist
            }
            
        elif resolution.resolution_type == "keep_separate":
            # No changes expected
            return {"verified": True}
            
        return {"verified": False, "error": "Unsupported resolution verification"} 