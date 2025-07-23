"""Quality rule implementations."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Pattern
from datetime import datetime
import logging

from .models import (
    QualityViolation, ValidationResult, QualityRuleConfig, 
    QualityRuleType, QualitySeverity
)

logger = logging.getLogger(__name__)


class QualityRule(ABC):
    """Abstract base class for all quality rules."""
    
    def __init__(self, config: QualityRuleConfig):
        self.config = config
        self.rule_id = config.rule_id
        self.rule_type = config.rule_type
        self.severity = config.severity
        self.enabled = config.enabled
        self.parameters = config.parameters
        
    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate a single value against this rule."""
        pass
    
    def get_llm_prompt_instruction(self) -> str:
        """Generate instruction for LLM to follow this rule during extraction."""
        return f"Apply quality rule: {self.config.description}"
    
    def is_enabled(self) -> bool:
        """Check if this rule is enabled."""
        return self.enabled
    
    def _create_violation(
        self, 
        message: str, 
        expected: str, 
        actual: str,
        context: Dict[str, Any],
        confidence: float = 1.0,
        suggestion: Optional[str] = None
    ) -> QualityViolation:
        """Helper method to create a quality violation."""
        return QualityViolation(
            rule_id=self.rule_id,
            rule_type=self.rule_type,
            severity=self.severity,
            entity_type=context.get('entity_type'),
            entity_id=context.get('entity_id'),
            property_name=context.get('property_name'),
            relationship_type=context.get('relationship_type'),
            message=message,
            expected=expected,
            actual=actual,
            confidence=confidence,
            suggestion=suggestion,
            context=context
        )


# ============================================================================
# FORMAT RULES
# ============================================================================

class PatternRule(QualityRule):
    """Validates that a value matches a regex pattern."""
    
    def __init__(self, config: QualityRuleConfig):
        super().__init__(config)
        pattern = self.parameters.get('pattern')
        if not pattern:
            raise ValueError(f"PatternRule {self.rule_id} missing 'pattern' parameter")
        
        flags = self.parameters.get('flags', 0)
        try:
            self.pattern: Pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern in rule {self.rule_id}: {e}")
    
    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(
                is_valid=True,
                message="Value is None, skipping pattern validation"
            )
        
        str_value = str(value).strip()
        if not str_value:
            return ValidationResult(
                is_valid=True,
                message="Value is empty, skipping pattern validation"
            )
        
        if self.pattern.match(str_value):
            return ValidationResult(
                is_valid=True,
                message=f"Value matches pattern {self.parameters['pattern']}"
            )
        else:
            violation = self._create_violation(
                message=f"Value does not match required pattern",
                expected=f"Pattern: {self.parameters['pattern']}",
                actual=str_value,
                context=context,
                suggestion=f"Ensure value follows the pattern: {self.parameters['pattern']}"
            )
            return ValidationResult(
                is_valid=False,
                message=violation.message,
                violations=[violation]
            )
    
    def get_llm_prompt_instruction(self) -> str:
        pattern = self.parameters['pattern']
        examples = self.config.examples
        instruction = f"Extract values that match the pattern: {pattern}"
        if examples:
            instruction += f". Examples: {', '.join(examples)}"
        return instruction


class LengthRule(QualityRule):
    """Validates the length of a string value."""
    
    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(is_valid=True, message="Value is None, skipping length validation")
        
        str_value = str(value).strip()
        length = len(str_value)
        
        min_length = self.parameters.get('min_length')
        max_length = self.parameters.get('max_length')
        
        violations = []
        
        if min_length is not None and length < min_length:
            violation = self._create_violation(
                message=f"Value is too short (minimum {min_length} characters)",
                expected=f"Minimum length: {min_length}",
                actual=f"Length: {length}",
                context=context
            )
            violations.append(violation)
        
        if max_length is not None and length > max_length:
            violation = self._create_violation(
                message=f"Value is too long (maximum {max_length} characters)", 
                expected=f"Maximum length: {max_length}",
                actual=f"Length: {length}",
                context=context
            )
            violations.append(violation)
        
        if violations:
            return ValidationResult(
                is_valid=False,
                message=f"Length validation failed: {length} characters",
                violations=violations
            )
        
        return ValidationResult(
            is_valid=True,
            message=f"Length validation passed: {length} characters"
        )


class CaseFormatRule(QualityRule):
    """Validates the case format of a string value."""
    
    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None or str(value).strip() == "":
            return ValidationResult(is_valid=True, message="Empty value, skipping case validation")
        
        str_value = str(value).strip()
        case_format = self.parameters.get('case_format')
        
        if case_format == 'upper' and str_value != str_value.upper():
            violation = self._create_violation(
                message="Value should be uppercase",
                expected="Uppercase format",
                actual=str_value,
                context=context,
                suggestion="Convert to uppercase"
            )
            return ValidationResult(is_valid=False, message=violation.message, violations=[violation])
        
        elif case_format == 'lower' and str_value != str_value.lower():
            violation = self._create_violation(
                message="Value should be lowercase",
                expected="Lowercase format", 
                actual=str_value,
                context=context,
                suggestion="Convert to lowercase"
            )
            return ValidationResult(is_valid=False, message=violation.message, violations=[violation])
        
        elif case_format == 'title_case' and str_value != str_value.title():
            violation = self._create_violation(
                message="Value should be in title case",
                expected="Title Case Format",
                actual=str_value,
                context=context,
                suggestion="Convert to title case"
            )
            return ValidationResult(is_valid=False, message=violation.message, violations=[violation])
        
        return ValidationResult(is_valid=True, message=f"Case format validation passed: {case_format}")


# ============================================================================
# BUSINESS RULES  
# ============================================================================

class AllowedValuesRule(QualityRule):
    """Validates that a value is in a list of allowed values."""
    
    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(is_valid=True, message="Value is None, skipping allowed values check")
        
        allowed_values = self.parameters.get('allowed_values', [])
        if not allowed_values:
            return ValidationResult(is_valid=True, message="No allowed values specified")
        
        str_value = str(value).strip()
        if str_value in allowed_values:
            return ValidationResult(is_valid=True, message=f"Value is in allowed list")
        
        violation = self._create_violation(
            message=f"Value is not in the list of allowed values",
            expected=f"One of: {', '.join(map(str, allowed_values))}",
            actual=str_value,
            context=context,
            suggestion=f"Use one of these values: {', '.join(map(str, allowed_values))}"
        )
        return ValidationResult(is_valid=False, message=violation.message, violations=[violation])


class ForbiddenValuesRule(QualityRule):
    """Validates that a value is not in a list of forbidden values."""
    
    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(is_valid=True, message="Value is None, skipping forbidden values check")
        
        forbidden_values = self.parameters.get('forbidden_values', [])
        if not forbidden_values:
            return ValidationResult(is_valid=True, message="No forbidden values specified")
        
        str_value = str(value).strip().lower()  # Case-insensitive comparison
        forbidden_lower = [str(v).strip().lower() for v in forbidden_values]
        
        if str_value in forbidden_lower:
            violation = self._create_violation(
                message=f"Value is in the list of forbidden values",
                expected=f"Not one of: {', '.join(map(str, forbidden_values))}",
                actual=str(value),
                context=context,
                suggestion="Use a meaningful value instead of placeholder text"
            )
            return ValidationResult(is_valid=False, message=violation.message, violations=[violation])
        
        return ValidationResult(is_valid=True, message="Value is not forbidden")


class RangeRule(QualityRule):
    """Validates that a numeric value is within a specified range."""
    
    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(is_valid=True, message="Value is None, skipping range validation")
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            violation = self._create_violation(
                message="Value is not numeric",
                expected="Numeric value",
                actual=str(value),
                context=context
            )
            return ValidationResult(is_valid=False, message=violation.message, violations=[violation])
        
        min_value = self.parameters.get('min_value')
        max_value = self.parameters.get('max_value')
        inclusive = self.parameters.get('inclusive', True)
        
        violations = []
        
        if min_value is not None:
            if (inclusive and numeric_value < min_value) or (not inclusive and numeric_value <= min_value):
                op = ">=" if inclusive else ">"
                violation = self._create_violation(
                    message=f"Value is below minimum threshold",
                    expected=f"Value {op} {min_value}",
                    actual=str(numeric_value),
                    context=context
                )
                violations.append(violation)
        
        if max_value is not None:
            if (inclusive and numeric_value > max_value) or (not inclusive and numeric_value >= max_value):
                op = "<=" if inclusive else "<"
                violation = self._create_violation(
                    message=f"Value is above maximum threshold",
                    expected=f"Value {op} {max_value}",
                    actual=str(numeric_value),
                    context=context
                )
                violations.append(violation)
        
        if violations:
            return ValidationResult(is_valid=False, message="Range validation failed", violations=violations)
        
        return ValidationResult(is_valid=True, message="Range validation passed")


class RequiredPropertyRule(QualityRule):
    """Validates that required properties are present and not empty."""
    
    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        """For required properties, the value should not be None or empty string."""
        if value is None:
            violation = self._create_violation(
                message="Required property is missing",
                expected="Non-null value",
                actual="null",
                context=context,
                suggestion="Ensure this required property is extracted"
            )
            return ValidationResult(is_valid=False, message=violation.message, violations=[violation])
        
        str_value = str(value).strip()
        if not str_value:
            violation = self._create_violation(
                message="Required property is empty",
                expected="Non-empty value",
                actual="empty string",
                context=context,
                suggestion="Ensure this required property has a meaningful value"
            )
            return ValidationResult(is_valid=False, message=violation.message, violations=[violation])
        
        return ValidationResult(is_valid=True, message="Required property validation passed")


# ============================================================================
# RULE FACTORY
# ============================================================================

class QualityRuleFactory:
    """Factory for creating quality rule instances."""
    
    _rule_types = {
        'pattern': PatternRule,
        'length': LengthRule,
        'case_format': CaseFormatRule,
        'allowed_values': AllowedValuesRule,
        'forbidden_values': ForbiddenValuesRule,
        'range': RangeRule,
        'required': RequiredPropertyRule,
    }
    
    @classmethod
    def create_rule(cls, rule_config: QualityRuleConfig) -> QualityRule:
        """Create a quality rule instance from configuration."""
        rule_class_name = rule_config.parameters.get('rule_class')
        
        if rule_class_name not in cls._rule_types:
            available = ', '.join(cls._rule_types.keys())
            raise ValueError(
                f"Unknown rule type '{rule_class_name}' for rule {rule_config.rule_id}. "
                f"Available types: {available}"
            )
        
        rule_class = cls._rule_types[rule_class_name]
        try:
            return rule_class(rule_config)
        except Exception as e:
            logger.error(f"Failed to create rule {rule_config.rule_id}: {e}")
            raise
    
    @classmethod
    def get_available_rule_types(cls) -> List[str]:
        """Get list of available rule types."""
        return list(cls._rule_types.keys())