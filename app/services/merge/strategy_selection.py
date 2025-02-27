"""Strategy selection engine for conflict resolution"""
from typing import Dict, List, Any, Optional, Tuple
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.baml_client import b
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResolutionStrategy:
    """Base class for resolution strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def get_confidence(self, conflict: Conflict) -> float:
        """Get confidence score for this strategy on the given conflict"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_resolution_option(self, conflict: Conflict) -> Optional[ResolutionOption]:
        """Get the resolution option corresponding to this strategy"""
        raise NotImplementedError("Subclasses must implement this method")

class PreferStagingStrategy(ResolutionStrategy):
    """Strategy that prefers staging values"""
    
    def __init__(self):
        super().__init__(
            "prefer_staging", 
            "Prefer values from the staging graph (newer data)"
        )
    
    def get_confidence(self, conflict: Conflict) -> float:
        """Calculate confidence for preferring staging data"""
        # Higher confidence for property conflicts
        if conflict.conflict_type == ConflictType.PROPERTY_VALUE:
            # Check recency if timestamps are available
            if conflict.context and conflict.context.get("staging_timestamp") and conflict.context.get("production_timestamp"):
                staging_time = datetime.fromisoformat(conflict.context["staging_timestamp"])
                prod_time = datetime.fromisoformat(conflict.context["production_timestamp"])
                
                # If staging is newer, high confidence
                if staging_time > prod_time:
                    time_diff = (staging_time - prod_time).total_seconds()
                    # Scale confidence based on how much newer (max 0.9 for day or more)
                    days_diff = time_diff / (24 * 3600)
                    return min(0.9, 0.5 + (days_diff * 0.1))
            
            # Default confidence for property conflicts
            return 0.6
            
        # Lower confidence for relationship conflicts
        elif conflict.conflict_type in (ConflictType.RELATIONSHIP_TYPE, ConflictType.RELATIONSHIP_DIRECTION):
            return 0.4
            
        # Default confidence
        return 0.5
    
    def get_resolution_option(self, conflict: Conflict) -> Optional[ResolutionOption]:
        """Get the 'keep staging' option if available"""
        for option in conflict.resolution_options:
            if option.resolution_type.startswith("keep_staging"):
                return option
        return None

class PreferProductionStrategy(ResolutionStrategy):
    """Strategy that prefers production values"""
    
    def __init__(self):
        super().__init__(
            "prefer_production", 
            "Prefer values from the production graph (established data)"
        )
    
    def get_confidence(self, conflict: Conflict) -> float:
        """Calculate confidence for preferring production data"""
        # Higher confidence for missing properties in staging
        if conflict.conflict_type == ConflictType.PROPERTY_MISSING:
            if conflict.context and conflict.context.get("missing_in") == "staging":
                return 0.8
                
        # Higher confidence for critical entity properties in production
        if conflict.conflict_type == ConflictType.PROPERTY_VALUE:
            property_name = conflict.property_name or ""
            # ID fields should typically be preserved in production
            if property_name.lower().endswith("id") or property_name.lower() == "id":
                return 0.8
                
        # Default confidence
        return 0.4
    
    def get_resolution_option(self, conflict: Conflict) -> Optional[ResolutionOption]:
        """Get the 'keep production' option if available"""
        for option in conflict.resolution_options:
            if option.resolution_type.startswith("keep_production"):
                return option
        return None

class MergeValuesStrategy(ResolutionStrategy):
    """Strategy that merges values from both sources"""
    
    def __init__(self):
        super().__init__(
            "merge_values", 
            "Merge values from both staging and production"
        )
    
    def get_confidence(self, conflict: Conflict) -> float:
        """Calculate confidence for merging values"""
        # Only applies to property value conflicts
        if conflict.conflict_type != ConflictType.PROPERTY_VALUE:
            return 0.0
            
        # Check if values are compatible for merging
        if conflict.context:
            staging_value = conflict.context.get("staging_value", conflict.staging_value)
            production_value = conflict.context.get("production_value", conflict.production_value)
            
            if isinstance(staging_value, str) and isinstance(production_value, str):
                # Strings can be concatenated
                return 0.5
                
            if isinstance(staging_value, list) and isinstance(production_value, list):
                # Lists can be merged
                return 0.7
            
        # Default - not confident about merging other types
        return 0.1
    
    def get_resolution_option(self, conflict: Conflict) -> Optional[ResolutionOption]:
        """Get the 'merge values' option if available"""
        for option in conflict.resolution_options:
            if option.resolution_type == "merge_values":
                return option
        return None

class KeepBothStrategy(ResolutionStrategy):
    """Strategy that keeps both versions"""
    
    def __init__(self):
        super().__init__(
            "keep_both", 
            "Keep both versions (e.g., for multiple relationships)"
        )
    
    def get_confidence(self, conflict: Conflict) -> float:
        """Calculate confidence for keeping both versions"""
        # Higher confidence for relationship conflicts
        if conflict.conflict_type in (ConflictType.RELATIONSHIP_TYPE, ConflictType.RELATIONSHIP_DIRECTION):
            return 0.6
            
        # Not applicable to property conflicts
        return 0.0
    
    def get_resolution_option(self, conflict: Conflict) -> Optional[ResolutionOption]:
        """Get the 'keep both' option if available"""
        for option in conflict.resolution_options:
            if option.resolution_type == "keep_both_relationships":
                return option
        return None

class LLMBasedStrategy(ResolutionStrategy):
    """Strategy that uses LLM to recommend resolution"""
    
    def __init__(self):
        super().__init__(
            "llm_assisted", 
            "Use a language model to recommend resolution"
        )
    
    async def get_llm_recommendation(self, conflict: Conflict) -> Tuple[str, float]:
        """Get recommendation from LLM"""
        try:
            # Convert conflict to a format suitable for LLM
            conflict_data = {
                "conflict_type": conflict.conflict_type.value,
                "severity": conflict.severity.value,
                "context": conflict.context or {},
                "description": conflict.description
            }
            
            # Prepare options data
            options_data = {
                "options": [
                    {
                        "id": opt.id,
                        "description": opt.description,
                        "resolution_type": opt.resolution_type
                    } for opt in conflict.resolution_options
                ]
            }
            
            # Call LLM service
            response = await b.SelectBestResolution(
                conflict=conflict_data,
                options=options_data,
                ontology={}
            )
            
            # Process response
            if response and hasattr(response, 'resolution_id') and hasattr(response, 'confidence'):
                return response.resolution_id, float(response.confidence)
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"LLM resolution recommendation failed: {str(e)}")
            return None, 0.0
    
    def get_confidence(self, conflict: Conflict) -> float:
        """Base confidence for using LLM"""
        # Higher confidence for complex conflicts
        if conflict.severity == ConflictSeverity.CRITICAL:
            return 0.7
            
        # Medium confidence for major conflicts
        if conflict.severity == ConflictSeverity.MAJOR:
            return 0.5
            
        # Lower confidence for minor conflicts (rule-based might be better)
        return 0.3
    
    def get_resolution_option(self, conflict: Conflict) -> Optional[ResolutionOption]:
        """This needs to be handled specially with LLM inference"""
        return None

class StrategySelectionEngine:
    """Engine for selecting the best resolution strategy"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration"""
        self.config = config or {}
        
        # Register standard strategies
        self.strategies = {
            "prefer_staging": PreferStagingStrategy(),
            "prefer_production": PreferProductionStrategy(),
            "merge_values": MergeValuesStrategy(),
            "keep_both": KeepBothStrategy(),
            "llm_assisted": LLMBasedStrategy()
        }
        
        # Add custom strategies from config
        for name, strategy_config in self.config.get("custom_strategies", {}).items():
            if strategy_config.get("type") == "rule_based":
                self.strategies[name] = self._create_rule_based_strategy(name, strategy_config)
    
    def _create_rule_based_strategy(self, name: str, config: Dict[str, Any]) -> ResolutionStrategy:
        """Create a rule-based strategy from configuration"""
        class CustomStrategy(ResolutionStrategy):
            def __init__(self, name: str, config: Dict[str, Any]):
                super().__init__(name, config.get("description", f"Custom strategy: {name}"))
                self.config = config
                self.rules = config.get("rules", {})
                
            def get_confidence(self, conflict: Conflict) -> float:
                # Apply rules to calculate confidence
                base_confidence = self.config.get("base_confidence", 0.5)
                
                # Check for entity type rules
                entity_type = conflict.entity_type
                if entity_type and entity_type in self.rules.get("entity_types", {}):
                    return self.rules["entity_types"][entity_type].get("confidence", base_confidence)
                
                # Check for property rules
                property_name = conflict.property_name
                if property_name and property_name in self.rules.get("properties", {}):
                    return self.rules["properties"][property_name].get("confidence", base_confidence)
                
                # Check for conflict type rules
                if conflict.conflict_type.value in self.rules.get("conflict_types", {}):
                    return self.rules["conflict_types"][conflict.conflict_type.value].get("confidence", base_confidence)
                
                return base_confidence
                
            def get_resolution_option(self, conflict: Conflict) -> Optional[ResolutionOption]:
                # Determine resolution type from rules
                resolution_type = None
                
                # Check for entity type rules
                entity_type = conflict.entity_type
                if entity_type and entity_type in self.rules.get("entity_types", {}):
                    resolution_type = self.rules["entity_types"][entity_type].get("resolution_type")
                
                # Check for property rules (more specific than entity)
                property_name = conflict.property_name
                if property_name and property_name in self.rules.get("properties", {}):
                    resolution_type = self.rules["properties"][property_name].get("resolution_type", resolution_type)
                
                # If we have a resolution type, find matching option
                if resolution_type:
                    for option in conflict.resolution_options:
                        if option.resolution_type == resolution_type:
                            return option
                
                return None
                
        return CustomStrategy(name, config)
    
    async def select_strategy(self, conflict: Conflict) -> Tuple[str, ResolutionOption, float, str]:
        """
        Select the best resolution strategy for a conflict
        Returns: (strategy_name, resolution_option, confidence, explanation)
        """
        # Calculate confidence for each strategy
        strategy_scores = {}
        for name, strategy in self.strategies.items():
            # Skip LLM strategy for now (handled separately)
            if name == "llm_assisted":
                continue
                
            confidence = strategy.get_confidence(conflict)
            option = strategy.get_resolution_option(conflict)
            
            # Only consider strategies with available options
            if option:
                strategy_scores[name] = (confidence, option)
        
        # Find strategy with highest confidence
        best_strategy = None
        best_confidence = 0.0
        best_option = None
        
        for name, (confidence, option) in strategy_scores.items():
            if confidence > best_confidence:
                best_strategy = name
                best_confidence = confidence
                best_option = option
        
        # Check if we should try LLM for better results
        use_llm = False
        
        # If no good rule-based strategy found
        if best_confidence < 0.5:
            use_llm = True
            
        # If conflict is critical, always check LLM
        if conflict.severity == ConflictSeverity.CRITICAL:
            use_llm = True
            
        # If LLM is specifically requested in config
        if self.config.get("always_use_llm", False):
            use_llm = True
            
        # Try LLM if needed
        if use_llm and "llm_assisted" in self.strategies:
            llm_strategy = self.strategies["llm_assisted"]
            llm_confidence = llm_strategy.get_confidence(conflict)
            
            # Only query LLM if base confidence is good enough
            if llm_confidence >= 0.3:
                resolution_id, recommendation_confidence = await llm_strategy.get_llm_recommendation(conflict)
                
                # Adjust confidence based on LLM's own confidence
                adjusted_confidence = llm_confidence * recommendation_confidence
                
                # If LLM recommendation is better, use it
                if resolution_id and adjusted_confidence > best_confidence:
                    # Find the corresponding option
                    for option in conflict.resolution_options:
                        if option.id == resolution_id:
                            best_strategy = "llm_assisted"
                            best_confidence = adjusted_confidence
                            best_option = option
                            break
        
        # Generate explanation
        explanation = self._generate_explanation(conflict, best_strategy, best_confidence)
        
        return best_strategy, best_option, best_confidence, explanation
    
    def _generate_explanation(self, conflict: Conflict, strategy: str, confidence: float) -> str:
        """Generate human-readable explanation for the strategy selection"""
        if not strategy:
            return "No suitable automatic resolution strategy found."
            
        strategy_obj = self.strategies.get(strategy)
        if not strategy_obj:
            return f"Strategy '{strategy}' selected with {confidence:.1%} confidence."
            
        base_explanation = f"Selected '{strategy_obj.description}' strategy with {confidence:.1%} confidence."
        
        # Add strategy-specific details
        if strategy == "prefer_staging":
            if conflict.context and conflict.context.get("staging_timestamp") and conflict.context.get("production_timestamp"):
                return f"{base_explanation} Staging data appears to be more recent."
                
        elif strategy == "prefer_production":
            property_name = conflict.property_name or ""
            if property_name.lower().endswith("id"):
                return f"{base_explanation} Production IDs are typically more stable."
                
        elif strategy == "merge_values":
            if conflict.context and isinstance(conflict.context.get("staging_value"), list):
                return f"{base_explanation} Both lists contain valuable items that can be combined."
                
        elif strategy == "llm_assisted":
            return f"{base_explanation} Complex conflict analysis performed by language model."
            
        return base_explanation 