"""Engine for automatic resolution of minor conflicts"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.services.merge.strategy_selection import StrategySelectionEngine

logger = logging.getLogger(__name__)

class AutoResolutionEngine:
    """Engine for automatic resolution of minor conflicts"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration"""
        self.config = config or {}
        # Initialize strategy selection engine
        self.strategy_engine = StrategySelectionEngine(config)
        # Default resolution strategies
        self.default_strategies = {
            ConflictType.PROPERTY_VALUE: self._resolve_property_value,
            ConflictType.PROPERTY_MISSING: self._resolve_missing_property,
            ConflictType.PROPERTY_TYPE: self._resolve_property_type,
            # Add other conflict types as needed
        }
    
    async def resolve_conflict(self, conflict: Conflict) -> Optional[str]:
        """
        Attempt to automatically resolve a conflict
        Returns resolution_id if successful, None if manual resolution needed
        """
        # Skip if not a minor conflict
        if conflict.severity != ConflictSeverity.MINOR:
            return None
            
        # Skip if no resolution options
        if not conflict.resolution_options:
            return None
            
        # Check if entity type or property has override rules
        entity_type = conflict.entity_type
        property_name = conflict.property_name
        
        # Check for specific property override first (more specific)
        if entity_type and property_name:
            strategy_key = f"{entity_type}.{property_name}"
            if strategy_key in self.config:
                return self._apply_config_strategy(conflict, self.config[strategy_key])
        
        # Check for entity type override
        if entity_type and entity_type in self.config:
            return self._apply_config_strategy(conflict, self.config[entity_type])
            
        # Try using the strategy selection engine
        try:
            strategy_name, resolution_option, confidence, explanation = await self.strategy_engine.select_strategy(conflict)
            
            # Log the selected strategy
            logger.info(f"Auto-resolution strategy selected: {strategy_name} with {confidence:.1%} confidence")
            logger.info(f"Explanation: {explanation}")
            
            # Only auto-resolve if confidence is high enough
            min_confidence = self.config.get("min_confidence", 0.6)
            if confidence >= min_confidence and resolution_option:
                return resolution_option.id
                
        except Exception as e:
            logger.error(f"Strategy selection failed: {str(e)}")
            # Fall back to legacy resolution methods
        
        # Use default strategy based on conflict type
        if conflict.conflict_type in self.default_strategies:
            strategy = self.default_strategies[conflict.conflict_type]
            return await strategy(conflict)
            
        return None
    
    def _apply_config_strategy(self, conflict: Conflict, strategy_config: Dict[str, Any]) -> Optional[str]:
        """Apply a configured resolution strategy"""
        strategy_type = strategy_config.get("type")
        
        if strategy_type == "prefer_staging":
            # Create a resolution ID with the staging suffix
            return f"{conflict.id}_staging"
        elif strategy_type == "prefer_production":
            # Create a resolution ID with the production suffix
            return f"{conflict.id}_prod"
        elif strategy_type == "highest_confidence":
            return self._select_highest_confidence_option(conflict)
        
        return None
    
    def _select_option_by_type(self, conflict: Conflict, resolution_type: str) -> Optional[str]:
        """Select resolution option by type"""
        for option in conflict.resolution_options:
            if option.resolution_type.startswith(resolution_type):
                return option.id
        return None
    
    def _select_highest_confidence_option(self, conflict: Conflict) -> Optional[str]:
        """Select resolution option with highest confidence"""
        if not conflict.resolution_options:
            return None
            
        highest_option = max(
            conflict.resolution_options, 
            key=lambda o: o.confidence
        )
        
        # Only auto-resolve if confidence is high enough
        if highest_option.confidence >= 0.7:  # Configurable threshold
            return highest_option.id
            
        return None
    
    async def _resolve_property_value(self, conflict: Conflict) -> Optional[str]:
        """Default strategy for property value conflicts"""
        # If string case difference, prefer lowercase for consistency
        if conflict.property_name and isinstance(conflict.staging_value, str) and isinstance(conflict.production_value, str):
            staging_value = conflict.staging_value.lower()
            production_value = conflict.production_value.lower()
            
            if staging_value == production_value:
                # Just case difference, prefer staging
                return self._select_option_by_type(conflict, "keep_staging")
        
        # For numeric values with small differences, prefer larger value
        if isinstance(conflict.staging_value, (int, float)) and isinstance(conflict.production_value, (int, float)):
            diff_pct = abs(conflict.staging_value - conflict.production_value) / max(abs(conflict.staging_value), abs(conflict.production_value))
            
            if diff_pct < 0.05:  # 5% threshold, configurable
                # Minor difference, prefer larger value
                if conflict.staging_value > conflict.production_value:
                    return self._select_option_by_type(conflict, "keep_staging")
                else:
                    return self._select_option_by_type(conflict, "keep_production")
        
        # Default to highest confidence option
        return self._select_highest_confidence_option(conflict)
    
    async def _resolve_missing_property(self, conflict: Conflict) -> Optional[str]:
        """Default strategy for missing property conflicts"""
        # Keep property if it's in staging but not production
        if conflict.context and conflict.context.get("missing_in") == "production":
            return self._select_option_by_type(conflict, "keep_staging")
            
        # Keep property if it's in production but not staging
        if conflict.context and conflict.context.get("missing_in") == "staging":
            return self._select_option_by_type(conflict, "keep_production")
            
        return None
    
    async def _resolve_property_type(self, conflict: Conflict) -> Optional[str]:
        """Default strategy for property type conflicts"""
        # Try to convert between compatible types
        if not (conflict.staging_value is not None and conflict.production_value is not None):
            return None
            
        staging_type = type(conflict.staging_value).__name__
        production_type = type(conflict.production_value).__name__
        
        # Prefer string type for mixed types
        if "str" in (staging_type, production_type):
            if staging_type == "str":
                return self._select_option_by_type(conflict, "keep_staging")
            else:
                return self._select_option_by_type(conflict, "keep_production")
                
        # For numeric types, prefer more precise type
        if staging_type == "float" and production_type == "int":
            return self._select_option_by_type(conflict, "keep_staging")
        if staging_type == "int" and production_type == "float":
            return self._select_option_by_type(conflict, "keep_production")
            
        return None 