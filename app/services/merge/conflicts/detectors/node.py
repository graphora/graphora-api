"""Node conflict detector"""
import asyncio
from typing import Dict, List, Any, Optional

from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionOption,
    ResolutionStrategy
)
from app.schemas.graph import Node
from app.services.storage.interface import GraphStorageInterface
from .base import BatchConflictDetector

class NodeConflictDetector(BatchConflictDetector):
    """Detector for node-related conflicts"""
    
    async def detect_conflicts(
        self,
        staging_nodes: List[Node],
        production_matches: Dict[str, List[str]]
    ) -> List[Conflict]:
        """Detect all conflicts for nodes
        
        Args:
            staging_nodes: List of nodes from staging
            production_matches: Mapping of staging IDs to production matches
            
        Returns:
            List of detected conflicts
        """
        # Group nodes by type for efficient batch processing
        nodes_by_type = {}
        for node in staging_nodes:
            if node.label not in nodes_by_type:
                nodes_by_type[node.label] = []
            nodes_by_type[node.label].append(node)
            
        # Process each type in batches
        all_conflicts = []
        for node_type, nodes in nodes_by_type.items():
            conflicts = await self.process_batch(
                nodes,
                self._detect_node_conflicts,
                production_matches
            )
            all_conflicts.extend(conflicts)
            
        return all_conflicts
        
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
        type_conflicts = await self._detect_type_conflicts(
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
                self._create_duplicate_conflict(staging_node, prod_nodes)
            )
            
        return conflicts
        
    async def _detect_type_conflicts(
        self,
        staging_node: Node,
        production_nodes: List[Node]
    ) -> List[Conflict]:
        """Detect type conflicts between nodes"""
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
        
    async def _detect_property_conflicts(
        self,
        staging_node: Node,
        production_node: Node
    ) -> List[Conflict]:
        """Detect property conflicts between nodes"""
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
            resolution_options=[
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
        )
        
    def _create_missing_property_conflict(
        self,
        staging_node: Node,
        production_node: Node,
        property_name: str,
        missing_in: str
    ) -> Conflict:
        """Create a missing property conflict"""
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
            conflict_type=ConflictType.PROPERTY_MISSING,
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
        
    def _create_duplicate_conflict(
        self,
        staging_node: Node,
        production_nodes: List[Node]
    ) -> Conflict:
        """Create a duplicate entity conflict"""
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
