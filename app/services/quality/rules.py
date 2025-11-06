"""Quality rule implementations."""

import re
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern
import logging

from .models import (
    QualityViolation,
    ValidationResult,
    QualityRuleConfig,
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
        suggestion: Optional[str] = None,
    ) -> QualityViolation:
        """Helper method to create a quality violation."""
        return QualityViolation(
            rule_id=self.rule_id,
            rule_type=self.rule_type,
            severity=self.severity,
            entity_type=context.get("entity_type"),
            entity_id=context.get("entity_id"),
            property_name=context.get("property_name"),
            relationship_type=context.get("relationship_type"),
            message=message,
            expected=expected,
            actual=actual,
            confidence=confidence,
            suggestion=suggestion,
            context=context,
        )


# ============================================================================
# FORMAT RULES
# ============================================================================


class PatternRule(QualityRule):
    """Validates that a value matches a regex pattern."""

    def __init__(self, config: QualityRuleConfig):
        super().__init__(config)
        pattern = self.parameters.get("pattern")
        if not pattern:
            raise ValueError(f"PatternRule {self.rule_id} missing 'pattern' parameter")

        flags = self.parameters.get("flags", 0)
        try:
            self.pattern: Pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern in rule {self.rule_id}: {e}")

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(
                is_valid=True, message="Value is None, skipping pattern validation"
            )

        str_value = str(value).strip()
        if not str_value:
            return ValidationResult(
                is_valid=True, message="Value is empty, skipping pattern validation"
            )

        if self.pattern.match(str_value):
            return ValidationResult(
                is_valid=True,
                message=f"Value matches pattern {self.parameters['pattern']}",
            )
        else:
            violation = self._create_violation(
                message="Value does not match required pattern",
                expected=f"Pattern: {self.parameters['pattern']}",
                actual=str_value,
                context=context,
                suggestion=f"Ensure value follows the pattern: {self.parameters['pattern']}",
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

    def get_llm_prompt_instruction(self) -> str:
        pattern = self.parameters["pattern"]
        examples = self.config.examples
        instruction = f"Extract values that match the pattern: {pattern}"
        if examples:
            instruction += f". Examples: {', '.join(examples)}"
        return instruction


class LengthRule(QualityRule):
    """Validates the length of a string value."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(
                is_valid=True, message="Value is None, skipping length validation"
            )

        str_value = str(value).strip()
        length = len(str_value)

        min_length = self.parameters.get("minLength")
        max_length = self.parameters.get("maxLength")

        violations = []

        if min_length is not None and length < min_length:
            violation = self._create_violation(
                message=f"Value is too short (minimum {min_length} characters)",
                expected=f"Minimum length: {min_length}",
                actual=f"Length: {length}",
                context=context,
            )
            violations.append(violation)

        if max_length is not None and length > max_length:
            violation = self._create_violation(
                message=f"Value is too long (maximum {max_length} characters)",
                expected=f"Maximum length: {max_length}",
                actual=f"Length: {length}",
                context=context,
            )
            violations.append(violation)

        if violations:
            return ValidationResult(
                is_valid=False,
                message=f"Length validation failed: {length} characters",
                violations=violations,
            )

        return ValidationResult(
            is_valid=True, message=f"Length validation passed: {length} characters"
        )


class CaseFormatRule(QualityRule):
    """Validates the case format of a string value."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None or str(value).strip() == "":
            return ValidationResult(
                is_valid=True, message="Empty value, skipping case validation"
            )

        str_value = str(value).strip()
        case_format = self.parameters.get("caseFormat")

        if case_format == "upper" and str_value != str_value.upper():
            violation = self._create_violation(
                message="Value should be uppercase",
                expected="Uppercase format",
                actual=str_value,
                context=context,
                suggestion="Convert to uppercase",
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

        elif case_format == "lower" and str_value != str_value.lower():
            violation = self._create_violation(
                message="Value should be lowercase",
                expected="Lowercase format",
                actual=str_value,
                context=context,
                suggestion="Convert to lowercase",
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

        elif case_format == "titleCase" and str_value != str_value.title():
            violation = self._create_violation(
                message="Value should be in title case",
                expected="Title Case Format",
                actual=str_value,
                context=context,
                suggestion="Convert to title case",
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

        return ValidationResult(
            is_valid=True, message=f"Case format validation passed: {case_format}"
        )


class DateWindowRule(QualityRule):
    """Validates that a date falls within a configured window."""

    def __init__(self, config: QualityRuleConfig):
        super().__init__(config)
        earliest = self.parameters.get("earliest")
        latest = self.parameters.get("latest")
        self.allow_future = bool(self.parameters.get("allow_future", False))
        self.earliest: Optional[datetime] = self._parse_date(earliest)
        self.latest: Optional[datetime] = self._parse_date(latest)

    def _parse_date(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"DateWindowRule {self.rule_id} has invalid date value '{value}'"
            )

    def _coerce_value(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        str_value = str(value).strip()
        if not str_value:
            return None
        try:
            return datetime.fromisoformat(str_value)
        except ValueError:
            return None

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        candidate = self._coerce_value(value)
        if candidate is None:
            violation = self._create_violation(
                message="Date value missing or invalid",
                expected="ISO formatted date",
                actual=str(value),
                context=context,
                suggestion="Provide a valid date value",
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

        violations: List[QualityViolation] = []

        if self.earliest and candidate < self.earliest:
            violations.append(
                self._create_violation(
                    message="Date precedes allowed minimum",
                    expected=f"On or after {self.earliest.isoformat()}",
                    actual=candidate.isoformat(),
                    context=context,
                )
            )

        if self.latest and candidate > self.latest:
            violations.append(
                self._create_violation(
                    message="Date exceeds allowed maximum",
                    expected=f"On or before {self.latest.isoformat()}",
                    actual=candidate.isoformat(),
                    context=context,
                )
            )

        now = datetime.utcnow()
        if not self.allow_future and candidate > now:
            violations.append(
                self._create_violation(
                    message="Date cannot be in the future",
                    expected="Historical date",
                    actual=candidate.isoformat(),
                    context=context,
                )
            )

        if violations:
            return ValidationResult(
                is_valid=False,
                message="Date window validation failed",
                violations=violations,
            )

        return ValidationResult(
            is_valid=True, message="Date falls within configured window"
        )


# ============================================================================
# BUSINESS RULES
# ============================================================================


class AllowedValuesRule(QualityRule):
    """Validates that a value is in a list of allowed values."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(
                is_valid=True, message="Value is None, skipping allowed values check"
            )

        allowed_values = self.parameters.get("allowedValues", [])
        if not allowed_values:
            return ValidationResult(
                is_valid=True, message="No allowed values specified"
            )

        str_value = str(value).strip()
        if str_value in allowed_values:
            return ValidationResult(is_valid=True, message="Value is in allowed list")

        violation = self._create_violation(
            message="Value is not in the list of allowed values",
            expected=f"One of: {', '.join(map(str, allowed_values))}",
            actual=str_value,
            context=context,
            suggestion=f"Use one of these values: {', '.join(map(str, allowed_values))}",
        )
        return ValidationResult(
            is_valid=False, message=violation.message, violations=[violation]
        )


class ForbiddenValuesRule(QualityRule):
    """Validates that a value is not in a list of forbidden values."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(
                is_valid=True, message="Value is None, skipping forbidden values check"
            )

        forbidden_values = self.parameters.get("forbiddenValues", [])
        if not forbidden_values:
            return ValidationResult(
                is_valid=True, message="No forbidden values specified"
            )

        str_value = str(value).strip().lower()  # Case-insensitive comparison
        forbidden_lower = [str(v).strip().lower() for v in forbidden_values]

        if str_value in forbidden_lower:
            violation = self._create_violation(
                message="Value is in the list of forbidden values",
                expected=f"Not one of: {', '.join(map(str, forbidden_values))}",
                actual=str(value),
                context=context,
                suggestion="Use a meaningful value instead of placeholder text",
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

        return ValidationResult(is_valid=True, message="Value is not forbidden")


class RangeRule(QualityRule):
    """Validates that a numeric value is within a specified range."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        if value is None:
            return ValidationResult(
                is_valid=True, message="Value is None, skipping range validation"
            )

        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            violation = self._create_violation(
                message="Value is not numeric",
                expected="Numeric value",
                actual=str(value),
                context=context,
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

        min_value = self.parameters.get("minValue")
        max_value = self.parameters.get("maxValue")
        inclusive = self.parameters.get("inclusive", True)

        violations = []

        if min_value is not None:
            if (inclusive and numeric_value < min_value) or (
                not inclusive and numeric_value <= min_value
            ):
                op = ">=" if inclusive else ">"
                violation = self._create_violation(
                    message="Value is below minimum threshold",
                    expected=f"Value {op} {min_value}",
                    actual=str(numeric_value),
                    context=context,
                )
                violations.append(violation)

        if max_value is not None:
            if (inclusive and numeric_value > max_value) or (
                not inclusive and numeric_value >= max_value
            ):
                op = "<=" if inclusive else "<"
                violation = self._create_violation(
                    message="Value is above maximum threshold",
                    expected=f"Value {op} {max_value}",
                    actual=str(numeric_value),
                    context=context,
                )
                violations.append(violation)

        if violations:
            return ValidationResult(
                is_valid=False, message="Range validation failed", violations=violations
            )

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
                suggestion="Ensure this required property is extracted",
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

        str_value = str(value).strip()
        if not str_value:
            violation = self._create_violation(
                message="Required property is empty",
                expected="Non-empty value",
                actual="empty string",
                context=context,
                suggestion="Ensure this required property has a meaningful value",
            )
            return ValidationResult(
                is_valid=False, message=violation.message, violations=[violation]
            )

        return ValidationResult(
            is_valid=True, message="Required property validation passed"
        )


class EntityCompletenessRule(QualityRule):
    """Validates that an entity meets a minimum property fill ratio."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        entity = context.get("entity")
        expected_props = context.get("expected_properties", {})
        if not entity or not expected_props:
            return ValidationResult(
                is_valid=True, message="No expected properties provided"
            )

        candidate_props = [
            name
            for name in expected_props.keys()
            if name not in context.get("system_properties", set())
        ]

        if not candidate_props:
            return ValidationResult(
                is_valid=True, message="No candidate properties for completeness"
            )

        filled = 0
        for prop in candidate_props:
            val = entity.properties.get(prop)
            if val is not None and str(val).strip() != "":
                filled += 1

        ratio = filled / len(candidate_props)
        min_ratio = float(self.parameters.get("min_ratio", 0.0))

        if ratio >= min_ratio:
            return ValidationResult(
                is_valid=True,
                message=f"Completeness {ratio:.2f} meets minimum {min_ratio:.2f}",
            )

        violation = self._create_violation(
            message="Entity property completeness below minimum threshold",
            expected=f"At least {min_ratio:.2f} of properties populated",
            actual=f"{ratio:.2f}",
            context=context,
            suggestion="Populate additional core properties",
        )
        return ValidationResult(
            is_valid=False, message=violation.message, violations=[violation]
        )


class RelationshipPresenceRule(QualityRule):
    """Ensure entities maintain minimum relationship counts."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        relationships = context.get("relationships") or []
        entities = context.get("entities") or []

        if not entities:
            return ValidationResult(is_valid=True, message="No entities supplied")

        entity_type = self.parameters.get("entity_type")
        relationship_type = self.parameters.get("relationship_type")
        direction = self.parameters.get("direction", "outbound")
        min_count = int(self.parameters.get("min_count", 1))

        if not relationship_type:
            return ValidationResult(
                is_valid=True, message="No relationship type configured"
            )

        candidates = (
            [entity for entity in entities if entity.type == entity_type]
            if entity_type
            else list(entities)
        )

        if not candidates:
            return ValidationResult(
                is_valid=True, message="No entities matching requirement"
            )

        violation_list: List[QualityViolation] = []

        for entity in candidates:
            count = 0
            for rel in relationships:
                if rel.type != relationship_type:
                    continue
                if direction == "inbound" and rel.target_id == entity.id:
                    count += 1
                elif direction == "outbound" and rel.source_id == entity.id:
                    count += 1
                elif direction == "either" and (
                    rel.source_id == entity.id or rel.target_id == entity.id
                ):
                    count += 1

            if count < min_count:
                violation_list.append(
                    self._create_violation(
                        message=f"Entity missing required '{relationship_type}' relationships",
                        expected=f"At least {min_count} relationships",
                        actual=f"{count}",
                        context={
                            "entity_type": entity.type,
                            "entity_id": entity.id,
                            "expected_relationship": relationship_type,
                        },
                        suggestion="Review extraction for missing connections",
                    )
                )

        if violation_list:
            return ValidationResult(
                is_valid=False,
                message="Relationship presence validation failed",
                violations=violation_list,
            )

        return ValidationResult(
            is_valid=True, message="Relationship presence requirements satisfied"
        )


class PropertyCoverageRule(QualityRule):
    """Validate that a minimum percentage of entities populate a property."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        entities = context.get("entities") or []
        entity_type = self.parameters.get("entity_type")
        property_name = self.parameters.get("property_name")
        min_coverage = float(self.parameters.get("min_coverage", 1.0))
        allow_missing_entities = bool(self.parameters.get("allow_missing_entities", True))

        if not entity_type or not property_name:
            return ValidationResult(
                is_valid=True,
                message="Property coverage rule missing entity_type/property_name",
            )

        relevant_entities = [entity for entity in entities if entity.type == entity_type]

        if not relevant_entities:
            if allow_missing_entities:
                return ValidationResult(
                    is_valid=True,
                    message=f"No entities of type {entity_type}; skipping coverage rule",
                )
            violation = self._create_violation(
                message=f"No entities of type {entity_type} found",
                expected=f"At least one {entity_type} entity",
                actual="0 entities",
                context={"entity_type": entity_type, "property_name": property_name},
            )
            return ValidationResult(
                is_valid=False,
                message=violation.message,
                violations=[violation],
            )

        filled = [
            entity
            for entity in relevant_entities
            if property_name in entity.properties
            and entity.properties[property_name] is not None
            and str(entity.properties[property_name]).strip() != ""
        ]

        coverage = len(filled) / max(len(relevant_entities), 1)

        if coverage >= min_coverage:
            return ValidationResult(
                is_valid=True,
                message=(
                    f"Property coverage {coverage:.2f} meets minimum {min_coverage:.2f}"
                ),
            )

        violation = self._create_violation(
            message="Property coverage below minimum threshold",
            expected=f"At least {min_coverage * 100:.1f}% of {entity_type}.{property_name}",
            actual=f"{coverage * 100:.1f}% ({len(filled)}/{len(relevant_entities)})",
            context={
                "entity_type": entity_type,
                "property_name": property_name,
                "population": len(relevant_entities),
            },
        )

        return ValidationResult(
            is_valid=False, message=violation.message, violations=[violation]
        )


class ConfidenceThresholdRule(QualityRule):
    """Ensure extracted objects meet minimum confidence expectations."""

    def __init__(self, config: QualityRuleConfig):
        super().__init__(config)
        self.entity_threshold = self._get_optional_float(
            self.parameters.get("entity_threshold")
        )
        self.relationship_threshold = self._get_optional_float(
            self.parameters.get("relationship_threshold")
        )
        self.property_threshold = self._get_optional_float(
            self.parameters.get("property_threshold")
        )

    @staticmethod
    def _get_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_confidence(candidate: Any) -> float:
        confidence = getattr(candidate, "confidence_score", None)
        if confidence is not None:
            return float(confidence)
        provenance = getattr(candidate, "provenance", None)
        provenance_conf = getattr(provenance, "confidence_score", None)
        if provenance_conf is not None:
            return float(provenance_conf)
        return 0.0

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        entities = context.get("entities") or []
        relationships = context.get("relationships") or []

        violations: List[QualityViolation] = []

        if self.entity_threshold is not None:
            for entity in entities:
                confidence = self._resolve_confidence(entity)
                if confidence < self.entity_threshold:
                    violations.append(
                        self._create_violation(
                            message="Entity confidence below threshold",
                            expected=f">= {self.entity_threshold:.2f}",
                            actual=f"{confidence:.2f}",
                            context={
                                "entity_type": entity.type,
                                "entity_id": entity.id,
                                "confidence": confidence,
                                "threshold": self.entity_threshold,
                            },
                        )
                    )

        if self.relationship_threshold is not None:
            for relationship in relationships:
                confidence = self._resolve_confidence(relationship)
                if confidence < self.relationship_threshold:
                    violations.append(
                        self._create_violation(
                            message="Relationship confidence below threshold",
                            expected=f">= {self.relationship_threshold:.2f}",
                            actual=f"{confidence:.2f}",
                            context={
                                "relationship_type": relationship.type,
                                "source_id": relationship.source_id,
                                "target_id": relationship.target_id,
                                "confidence": confidence,
                                "threshold": self.relationship_threshold,
                            },
                        )
                    )

        if self.property_threshold is not None:
            logger.debug(
                "Property-level confidence threshold configured for %s but property-level confidence data is unavailable; skipping",
                self.rule_id,
            )

        if violations:
            return ValidationResult(
                is_valid=False,
                message="Confidence threshold validation failed",
                violations=violations,
            )

        return ValidationResult(
            is_valid=True, message="Confidence thresholds satisfied"
        )


class MinimumEntitiesRule(QualityRule):
    """Ensure each document produces a minimum number of entities."""

    def __init__(self, config: QualityRuleConfig):
        super().__init__(config)
        self.min_entities = int(self.parameters.get("min_entities", 0))

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        entities = context.get("entities") or []
        total_entities = len(entities)

        if total_entities >= self.min_entities:
            return ValidationResult(
                is_valid=True,
                message=(
                    f"Entity count {total_entities} meets minimum {self.min_entities}"
                ),
            )

        violation = self._create_violation(
            message="Document produced too few entities",
            expected=f"At least {self.min_entities} entities",
            actual=str(total_entities),
            context={"entity_type": "GLOBAL", "entity_id": "GLOBAL"},
        )

        return ValidationResult(
            is_valid=False, message=violation.message, violations=[violation]
        )


class RequiredEntityTypesRule(QualityRule):
    """Ensure required entity types appear in the extraction."""

    def __init__(self, config: QualityRuleConfig):
        super().__init__(config)
        raw_requirements = self.parameters.get("required_types") or {}

        requirements: Dict[str, int] = {}
        if isinstance(raw_requirements, dict):
            for entity_type, min_count in raw_requirements.items():
                if entity_type is None:
                    continue
                try:
                    requirements[str(entity_type)] = max(int(min_count), 1)
                except (TypeError, ValueError):
                    requirements[str(entity_type)] = 1
        else:
            for entity_type in raw_requirements:
                if entity_type is None:
                    continue
                requirements[str(entity_type)] = 1

        self.requirements = requirements

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        entities = context.get("entities") or []
        counts = Counter(entity.type for entity in entities)

        violations: List[QualityViolation] = []
        for entity_type, min_count in self.requirements.items():
            actual_count = counts.get(entity_type, 0)
            if actual_count < min_count:
                violations.append(
                    self._create_violation(
                        message="Required entity type not sufficiently represented",
                        expected=f"{entity_type}: >= {min_count}",
                        actual=f"{actual_count}",
                        context={
                            "entity_type": entity_type,
                            "expected_minimum": min_count,
                        },
                    )
                )

        if violations:
            return ValidationResult(
                is_valid=False,
                message="Required entity types missing",
                violations=violations,
            )

        return ValidationResult(
            is_valid=True, message="All required entity types present"
        )


class EntityBalanceRule(QualityRule):
    """Validate that entity type distribution stays within configured bounds."""

    def __init__(self, config: QualityRuleConfig):
        super().__init__(config)
        self.max_single_type_ratio = self._get_optional_float(
            self.parameters.get("max_single_type_ratio")
        )

        expected_ratios_raw = self.parameters.get("expected_ratios") or {}
        parsed_expected: Dict[str, Dict[str, Optional[float]]] = {}
        for entity_type, bounds in expected_ratios_raw.items():
            if entity_type is None or not isinstance(bounds, dict):
                continue
            min_bound = bounds.get("min")
            max_bound = bounds.get("max")
            parsed_expected[str(entity_type)] = {
                "min": self._get_optional_float(min_bound),
                "max": self._get_optional_float(max_bound),
            }

        self.expected_ratios = parsed_expected
        self._ratio_tolerance = 0.0001

    @staticmethod
    def _get_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        entities = context.get("entities") or []
        if not entities:
            return ValidationResult(
                is_valid=True, message="No entities extracted; skipping balance rule"
            )

        counts = Counter(entity.type for entity in entities if entity.type)
        total = sum(counts.values())
        if total == 0:
            return ValidationResult(
                is_valid=True, message="No entities to evaluate for balance"
            )

        violations: List[QualityViolation] = []

        if self.max_single_type_ratio is not None:
            for entity_type, count in counts.items():
                ratio = count / total
                if ratio > self.max_single_type_ratio + self._ratio_tolerance:
                    violations.append(
                        self._create_violation(
                            message="Entity type exceeds allowed distribution",
                            expected=(
                                f"<= {self.max_single_type_ratio:.2f} of total entities"
                            ),
                            actual=f"{ratio:.2f} ({count}/{total})",
                            context={
                                "entity_type": entity_type,
                                "entity_ratio": ratio,
                                "total_entities": total,
                            },
                        )
                    )

        for entity_type, bounds in self.expected_ratios.items():
            ratio = counts.get(entity_type, 0) / total
            min_bound = bounds.get("min")
            max_bound = bounds.get("max")

            if min_bound is not None and ratio + self._ratio_tolerance < min_bound:
                violations.append(
                    self._create_violation(
                        message="Entity type below expected distribution",
                        expected=f">= {min_bound:.2f} of total entities",
                        actual=f"{ratio:.2f}",
                        context={
                            "entity_type": entity_type,
                            "entity_ratio": ratio,
                            "total_entities": total,
                        },
                    )
                )

            if max_bound is not None and ratio - self._ratio_tolerance > max_bound:
                violations.append(
                    self._create_violation(
                        message="Entity type above expected distribution",
                        expected=f"<= {max_bound:.2f} of total entities",
                        actual=f"{ratio:.2f}",
                        context={
                            "entity_type": entity_type,
                            "entity_ratio": ratio,
                            "total_entities": total,
                        },
                    )
                )

        if violations:
            return ValidationResult(
                is_valid=False,
                message="Entity balance validation failed",
                violations=violations,
            )

        return ValidationResult(
            is_valid=True, message="Entity balance within expected ranges"
        )


class CrossEntityConsistencyRule(QualityRule):
    """Ensure related entities share consistent property values."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        relationships = context.get("relationships") or []
        entities = context.get("entities") or []

        relationship_type = self.parameters.get("relationship_type")
        source_property = self.parameters.get("source_property")
        target_property = self.parameters.get("target_property")
        case_insensitive = bool(self.parameters.get("case_insensitive", True))
        allow_missing = bool(self.parameters.get("allow_missing", False))

        allowed_pairs_cfg = self.parameters.get("allowed_pairs") or []
        allowed_pairs = set()
        for pair in allowed_pairs_cfg:
            src = pair.get("source")
            tgt = pair.get("target")
            if src is None or tgt is None:
                continue
            allowed_pairs.add((str(src).strip().lower(), str(tgt).strip().lower()))

        if not relationship_type or not source_property or not target_property:
            return ValidationResult(
                is_valid=True,
                message="Consistency rule missing configuration; skipping",
            )

        node_map = {node.id: node for node in entities if node.id}

        def _normalize(val: Any) -> Optional[str]:
            if val is None:
                return None
            text = str(val).strip()
            if text == "":
                return None
            return text.lower() if case_insensitive else text

        violations: List[QualityViolation] = []

        for rel in relationships:
            if rel.type != relationship_type:
                continue

            source_node = node_map.get(rel.source_id)
            target_node = node_map.get(rel.target_id)

            if not source_node or not target_node:
                continue

            source_value_raw = source_node.properties.get(source_property)
            target_value_raw = target_node.properties.get(target_property)

            source_value = _normalize(source_value_raw)
            target_value = _normalize(target_value_raw)

            if source_value is None or target_value is None:
                if allow_missing:
                    continue
                missing_side = "source" if source_value is None else "target"
                violation = self._create_violation(
                    message=f"Missing {missing_side} property for consistency check",
                    expected=f"{source_property if missing_side == 'source' else target_property} populated",
                    actual="None",
                    context={
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "relationship_type": relationship_type,
                        "source_property": source_property,
                        "target_property": target_property,
                    },
                )
                violations.append(violation)
                continue

            if allowed_pairs:
                if (source_value, target_value) in allowed_pairs:
                    continue
            elif source_value == target_value:
                continue

            violation = self._create_violation(
                message="Cross-entity property mismatch",
                expected=f"{source_property} should align with {target_property}",
                actual=f"{source_value_raw} vs {target_value_raw}",
                context={
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relationship_type": relationship_type,
                    "source_property": source_property,
                    "target_property": target_property,
                },
            )
            violations.append(violation)

        if violations:
            return ValidationResult(
                is_valid=False,
                message="Cross-entity consistency validation failed",
                violations=violations,
            )

        return ValidationResult(
            is_valid=True, message="Cross-entity consistency validation passed"
        )


class GlobalDateWindowRule(QualityRule):
    """Validate that entity properties fall within a global temporal window."""

    def __init__(self, config: QualityRuleConfig):
        super().__init__(config)
        self.entity_type = self.parameters.get("entity_type")
        self.property_name = self.parameters.get("property_name")
        self.allow_missing = bool(self.parameters.get("allow_missing", True))
        self.allow_future = bool(self.parameters.get("allow_future", False))
        self.earliest = self._parse_date(self.parameters.get("earliest"))
        self.latest = self._parse_date(self.parameters.get("latest"))

    def _parse_date(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"GlobalDateWindowRule {self.rule_id} has invalid date '{value}'"
            ) from exc

    def _coerce_value(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        entities = context.get("entities") or []

        if not self.entity_type or not self.property_name:
            return ValidationResult(
                is_valid=True,
                message="Global date window rule missing configuration",
            )

        relevant_entities = [entity for entity in entities if entity.type == self.entity_type]
        violations: List[QualityViolation] = []

        for entity in relevant_entities:
            raw_value = entity.properties.get(self.property_name)
            dt_value = self._coerce_value(raw_value)

            if dt_value is None:
                if self.allow_missing:
                    continue
                violation = self._create_violation(
                    message="Date value missing or invalid",
                    expected="Valid ISO date",
                    actual=str(raw_value),
                    context={
                        "entity_type": entity.type,
                        "entity_id": entity.id,
                        "property_name": self.property_name,
                    },
                )
                violations.append(violation)
                continue

            if self.earliest and dt_value < self.earliest:
                violation = self._create_violation(
                    message="Date earlier than allowed window",
                    expected=f">= {self.earliest.isoformat()}",
                    actual=dt_value.isoformat(),
                    context={
                        "entity_type": entity.type,
                        "entity_id": entity.id,
                        "property_name": self.property_name,
                    },
                )
                violations.append(violation)
                continue

            if self.latest and dt_value > self.latest:
                violation = self._create_violation(
                    message="Date later than allowed window",
                    expected=f"<= {self.latest.isoformat()}",
                    actual=dt_value.isoformat(),
                    context={
                        "entity_type": entity.type,
                        "entity_id": entity.id,
                        "property_name": self.property_name,
                    },
                )
                violations.append(violation)
                continue

            if not self.allow_future and dt_value > datetime.now(tz=dt_value.tzinfo):
                violation = self._create_violation(
                    message="Date occurs in the future",
                    expected="Historical or present date",
                    actual=dt_value.isoformat(),
                    context={
                        "entity_type": entity.type,
                        "entity_id": entity.id,
                        "property_name": self.property_name,
                    },
                )
                violations.append(violation)

        if violations:
            return ValidationResult(
                is_valid=False,
                message="Global date window validation failed",
                violations=violations,
            )

        return ValidationResult(
            is_valid=True, message="Global date window validation passed"
        )


class SymmetricRelationshipRule(QualityRule):
    """Ensure relationships are mirrored from source to target."""

    def validate(self, value: Any, context: Dict[str, Any]) -> ValidationResult:
        relationships = context.get("relationships")
        if not relationships:
            return ValidationResult(is_valid=True, message="No relationships provided")

        relationship_type = self.parameters.get("relationship_type")
        inverse_type = self.parameters.get("inverse_type") or relationship_type

        relevant = [rel for rel in relationships if rel.type == relationship_type]
        inverse = [rel for rel in relationships if rel.type == inverse_type]

        inverse_lookup = {(rel.source_id, rel.target_id) for rel in inverse}

        violations: List[QualityViolation] = []
        for rel in relevant:
            counterpart = (rel.target_id, rel.source_id)
            if counterpart not in inverse_lookup:
                violations.append(
                    self._create_violation(
                        message="Symmetric relationship missing",
                        expected=f"{inverse_type} from {rel.target_id} to {rel.source_id}",
                        actual=f"Missing inverse for {rel.source_id}->{rel.target_id}",
                        context={
                            "relationship_type": relationship_type,
                            "source_id": rel.source_id,
                            "target_id": rel.target_id,
                        },
                        suggestion="Ensure reciprocal relationships are extracted",
                    )
                )

        if violations:
            return ValidationResult(
                is_valid=False,
                message="Symmetric relationship validation failed",
                violations=violations,
            )

        return ValidationResult(
            is_valid=True, message="Symmetric relationship validation passed"
        )


# ============================================================================
# RULE FACTORY
# ============================================================================


class QualityRuleFactory:
    """Factory for creating quality rule instances."""

    _rule_types = {
        "pattern": PatternRule,
        "length": LengthRule,
        "case_format": CaseFormatRule,
        "date_window": DateWindowRule,
        "allowed_values": AllowedValuesRule,
        "forbidden_values": ForbiddenValuesRule,
        "range": RangeRule,
        "required": RequiredPropertyRule,
        "entity_completeness": EntityCompletenessRule,
        "relationship_presence": RelationshipPresenceRule,
        "property_coverage": PropertyCoverageRule,
        "confidence_threshold": ConfidenceThresholdRule,
        "minimum_entities": MinimumEntitiesRule,
        "required_entity_types": RequiredEntityTypesRule,
        "entity_balance": EntityBalanceRule,
        "cross_entity_consistency": CrossEntityConsistencyRule,
        "global_date_window": GlobalDateWindowRule,
        "symmetric_relationship": SymmetricRelationshipRule,
    }

    @classmethod
    def create_rule(cls, rule_config: QualityRuleConfig) -> QualityRule:
        """Create a quality rule instance from configuration."""
        rule_class_name = rule_config.parameters.get("rule_class")

        if rule_class_name not in cls._rule_types:
            available = ", ".join(cls._rule_types.keys())
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
