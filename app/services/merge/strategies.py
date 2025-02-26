"""Merge strategies for resolving conflicts"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MergeStrategies:
    """Collection of merge strategies for different value types"""
    
    @staticmethod
    def merge_values(
        staging_value: Any,
        production_value: Any,
        strategy: str,
        **kwargs
    ) -> Any:
        """Merge values using specified strategy"""
        strategy_map = {
            "keep_staging": lambda: staging_value,
            "keep_production": lambda: production_value,
            "combine": lambda: MergeStrategies._combine_lists(
                staging_value, production_value
            ),
            "deep_merge": lambda: MergeStrategies._deep_merge_dicts(
                staging_value, production_value
            ),
            "concat": lambda: MergeStrategies._concat_strings(
                staging_value, production_value
            ),
            "average": lambda: MergeStrategies._average_numbers(
                staging_value, production_value
            ),
            "smart_merge": lambda: MergeStrategies._smart_merge(
                staging_value, production_value, **kwargs
            )
        }
        
        merge_func = strategy_map.get(strategy)
        if not merge_func:
            raise ValueError(f"Unknown merge strategy: {strategy}")
            
        try:
            return merge_func()
        except Exception as e:
            logger.error(f"Failed to merge values using {strategy}: {str(e)}")
            # Default to keeping staging value
            return staging_value
            
    @staticmethod
    def _combine_lists(
        list1: List[Any],
        list2: List[Any],
        deduplicate: bool = True
    ) -> List[Any]:
        """Combine two lists with optional deduplication"""
        if not isinstance(list1, list) or not isinstance(list2, list):
            raise ValueError("Both values must be lists")
            
        combined = list1 + list2
        if deduplicate:
            # For primitive values, use set
            if all(isinstance(x, (str, int, float, bool)) for x in combined):
                return list(set(combined))
                
            # For complex values, compare string representations
            seen = set()
            unique = []
            for item in combined:
                item_str = str(item)
                if item_str not in seen:
                    seen.add(item_str)
                    unique.append(item)
            return unique
            
        return combined
        
    @staticmethod
    def _deep_merge_dicts(
        dict1: Dict[str, Any],
        dict2: Dict[str, Any],
        prefer_dict1: bool = True
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        if not isinstance(dict1, dict) or not isinstance(dict2, dict):
            raise ValueError("Both values must be dictionaries")
            
        merged = dict2.copy()  # Start with dict2
        
        for key, value1 in dict1.items():
            if key not in merged:
                merged[key] = value1
            else:
                value2 = merged[key]
                
                # Both values are dicts, merge recursively
                if isinstance(value1, dict) and isinstance(value2, dict):
                    merged[key] = MergeStrategies._deep_merge_dicts(
                        value1, value2, prefer_dict1
                    )
                    
                # Both values are lists, combine them
                elif isinstance(value1, list) and isinstance(value2, list):
                    merged[key] = MergeStrategies._combine_lists(value1, value2)
                    
                # Conflict - use value from preferred dict
                else:
                    merged[key] = value1 if prefer_dict1 else value2
                    
        return merged
        
    @staticmethod
    def _concat_strings(
        str1: str,
        str2: str,
        separator: str = " "
    ) -> str:
        """Concatenate strings with separator"""
        if not isinstance(str1, str) or not isinstance(str2, str):
            raise ValueError("Both values must be strings")
            
        return f"{str1}{separator}{str2}"
        
    @staticmethod
    def _average_numbers(
        num1: Union[int, float],
        num2: Union[int, float]
    ) -> Union[int, float]:
        """Calculate average of two numbers"""
        if not isinstance(num1, (int, float)) or not isinstance(num2, (int, float)):
            raise ValueError("Both values must be numbers")
            
        avg = (float(num1) + float(num2)) / 2
        
        # Return int if both inputs were ints
        if isinstance(num1, int) and isinstance(num2, int):
            return round(avg)
        return avg
        
    @staticmethod
    def _smart_merge(
        value1: Any,
        value2: Any,
        property_name: str,
        entity_type: str,
        historical_resolutions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """Smart merge using property semantics and historical data"""
        # Use historical resolutions if available
        if historical_resolutions:
            # Find most common successful strategy
            strategy_counts = {}
            for resolution in historical_resolutions:
                if resolution.get("success"):
                    strategy = resolution["strategy"]
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                    
            if strategy_counts:
                best_strategy = max(
                    strategy_counts.items(),
                    key=lambda x: x[1]
                )[0]
                return MergeStrategies.merge_values(
                    value1, value2, best_strategy, **kwargs
                )
                
        # Property-specific strategies
        if property_name.lower() in {"created_at", "updated_at", "timestamp"}:
            # Use most recent timestamp
            if isinstance(value1, (str, int, float)) and isinstance(value2, (str, float)):
                try:
                    dt1 = datetime.fromisoformat(str(value1))
                    dt2 = datetime.fromisoformat(str(value2))
                    return str(max(dt1, dt2))
                except ValueError:
                    pass
                    
        elif property_name.lower() in {"count", "total", "sum"}:
            # Use maximum value for counts
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                return max(value1, value2)
                
        elif property_name.lower() in {"tags", "labels", "categories"}:
            # Combine unique values for tag-like properties
            if isinstance(value1, list) and isinstance(value2, list):
                return MergeStrategies._combine_lists(value1, value2, deduplicate=True)
                
        # Default to standard merge based on type
        if isinstance(value1, list):
            return MergeStrategies._combine_lists(value1, value2)
        elif isinstance(value1, dict):
            return MergeStrategies._deep_merge_dicts(value1, value2)
        elif isinstance(value1, str):
            return MergeStrategies._concat_strings(value1, value2)
        elif isinstance(value1, (int, float)):
            return MergeStrategies._average_numbers(value1, value2)
            
        # Fallback to keeping value1
        return value1
