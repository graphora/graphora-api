"""Parser for extracting quality rules from ontology YAML."""

from typing import Dict, List, Any, Optional
import logging

from .models import QualityRuleConfig, QualityRuleType, QualitySeverity
from .rules import QualityRuleFactory, QualityRule

logger = logging.getLogger(__name__)


class OntologyQualityParser:
    """Parses quality rules from ontology YAML structure."""

    def __init__(self, ontology: Dict[str, Any]):
        self.ontology = ontology
        self._default_gating_config = {
            "pass_score": 90.0,
            "warn_score": 80.0,
            "hard_fail_score": 70.0,
            "max_errors": 0,
            "max_warnings": 10,
            "min_property_completeness": 0.5,
            "entity_property_thresholds": {},
        }

    def get_gating_config(self) -> Dict[str, Any]:
        """Return gating configuration with sensible defaults."""
        gating = (
            self.ontology.get("dataQualityConfig", {}).get("gating", {})
            if isinstance(self.ontology.get("dataQualityConfig"), dict)
            else {}
        )

        config = self._default_gating_config.copy()
        config.update(
            {
                "pass_score": float(
                    gating.get(
                        "passScore", gating.get("warnScore", config["pass_score"])
                    )
                ),
                "warn_score": float(gating.get("warnScore", config["warn_score"])),
                "hard_fail_score": float(
                    gating.get("hardFailScore", config["hard_fail_score"])
                ),
                "max_errors": int(gating.get("maxErrors", config["max_errors"])),
                "max_warnings": int(gating.get("maxWarnings", config["max_warnings"])),
                "min_property_completeness": float(
                    gating.get(
                        "minPropertyCompleteness",
                        config["min_property_completeness"],
                    )
                ),
            }
        )

        entity_thresholds = gating.get("entityPropertyCompleteness", {})
        if isinstance(entity_thresholds, dict):
            config["entity_property_thresholds"] = {
                entity: float(threshold)
                for entity, threshold in entity_thresholds.items()
            }

        return config

    def parse_property_quality_rules(
        self, entity_type: str, property_name: str, quality_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse quality rules for a specific property."""
        rules = []

        try:
            # Parse format rules
            format_rules = quality_config.get("format", {})
            if format_rules:
                rules.extend(
                    self._parse_format_rules(entity_type, property_name, format_rules)
                )

            # Parse business rules
            business_rules = quality_config.get("business", {})
            if business_rules:
                rules.extend(
                    self._parse_business_rules(
                        entity_type, property_name, business_rules
                    )
                )

            # Parse validation rules
            validation_rules = quality_config.get("validation", {})
            if validation_rules:
                rules.extend(
                    self._parse_validation_rules(
                        entity_type, property_name, validation_rules
                    )
                )

        except Exception as e:
            logger.error(
                f"Failed to parse quality rules for {entity_type}.{property_name}: {e}"
            )

        return rules

    def parse_entity_quality_rules(
        self, entity_type: str, entity_quality_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse entity-level quality rules."""
        rules = []

        try:
            # Parse entity-level rules like completeness, consistency
            entity_level = entity_quality_config.get("entityLevel", {})
            if entity_level:
                rules.extend(self._parse_entity_level_rules(entity_type, entity_level))

        except Exception as e:
            logger.error(f"Failed to parse entity quality rules for {entity_type}: {e}")

        return rules

    def parse_relationship_quality_rules(
        self, entity_type: str, relationship_name: str, quality_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse quality rules for relationships."""
        rules = []

        try:
            # Parse relationship-level rules
            relationship_level = quality_config.get("relationshipLevel", {})
            if relationship_level:
                rules.extend(
                    self._parse_relationship_level_rules(
                        entity_type, relationship_name, relationship_level
                    )
                )

        except Exception as e:
            logger.error(
                f"Failed to parse relationship quality rules for {relationship_name}: {e}"
            )

        return rules

    def parse_global_quality_rules(
        self, global_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse global/cross-entity quality rules."""
        rules = []

        try:
            extraction_standards = global_config.get("extractionStandards", {})
            if extraction_standards:
                rules.extend(self._parse_extraction_standards(extraction_standards))

            # Parse distribution rules
            distribution_rules = global_config.get("distributionRules", {})
            if distribution_rules:
                rules.extend(self._parse_distribution_rules(distribution_rules))

            coverage_rules = global_config.get("coverageRules", [])
            if coverage_rules:
                rules.extend(self._parse_coverage_rules(coverage_rules))

            # Parse cross-validation rules
            cross_validation = global_config.get("crossValidationRules", [])
            if cross_validation:
                rules.extend(self._parse_cross_validation_rules(cross_validation))

            temporal_rules = global_config.get("temporalRules", [])
            if temporal_rules:
                rules.extend(self._parse_temporal_rules(temporal_rules))

        except Exception as e:
            logger.error(f"Failed to parse global quality rules: {e}")

        return rules

    def _parse_format_rules(
        self, entity_type: str, property_name: str, format_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse format-related quality rules."""
        rules = []

        # Pattern rule
        pattern = format_config.get("pattern")
        if pattern:
            rule_config = QualityRuleConfig(
                rule_id=f"{entity_type}.{property_name}.pattern",
                rule_type=QualityRuleType.FORMAT,
                severity=QualitySeverity.WARNING,
                name=f"Pattern validation for {property_name}",
                description=f"Value must match pattern: {pattern}",
                parameters={
                    "rule_class": "pattern",
                    "pattern": pattern,
                    "flags": format_config.get("flags", 0),
                },
                examples=format_config.get("examples", []),
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        # Length rules
        min_length = format_config.get("minLength")
        max_length = format_config.get("maxLength")
        if min_length is not None or max_length is not None:
            rule_config = QualityRuleConfig(
                rule_id=f"{entity_type}.{property_name}.length",
                rule_type=QualityRuleType.FORMAT,
                severity=QualitySeverity.WARNING,
                name=f"Length validation for {property_name}",
                description="Value length constraints",
                parameters={
                    "rule_class": "length",
                    "minLength": min_length,
                    "maxLength": max_length,
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        # Case format rule
        case_format = format_config.get("caseFormat")
        if case_format:
            rule_config = QualityRuleConfig(
                rule_id=f"{entity_type}.{property_name}.case_format",
                rule_type=QualityRuleType.FORMAT,
                severity=QualitySeverity.INFO,
                name=f"Case format validation for {property_name}",
                description=f"Value should be in {case_format} format",
                parameters={"rule_class": "case_format", "caseFormat": case_format},
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_business_rules(
        self, entity_type: str, property_name: str, business_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse business logic quality rules."""
        rules = []

        # Forbidden values rule
        forbidden_values = business_config.get("forbiddenValues")
        if forbidden_values:
            rule_config = QualityRuleConfig(
                rule_id=f"{entity_type}.{property_name}.forbidden_values",
                rule_type=QualityRuleType.BUSINESS,
                severity=QualitySeverity.ERROR,
                name=f"Forbidden values check for {property_name}",
                description=f"Value should not be one of: {', '.join(map(str, forbidden_values))}",
                parameters={
                    "rule_class": "forbidden_values",
                    "forbiddenValues": forbidden_values,
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        # Allowed values rule
        allowed_values = business_config.get("allowedValues")
        if allowed_values:
            rule_config = QualityRuleConfig(
                rule_id=f"{entity_type}.{property_name}.allowed_values",
                rule_type=QualityRuleType.BUSINESS,
                severity=QualitySeverity.ERROR,
                name=f"Allowed values check for {property_name}",
                description=f"Value must be one of: {', '.join(map(str, allowed_values))}",
                parameters={
                    "rule_class": "allowed_values",
                    "allowedValues": allowed_values,
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        # Range rule (for numeric values)
        min_value = business_config.get("minValue")
        max_value = business_config.get("maxValue")
        if min_value is not None or max_value is not None:
            rule_config = QualityRuleConfig(
                rule_id=f"{entity_type}.{property_name}.range",
                rule_type=QualityRuleType.BUSINESS,
                severity=QualitySeverity.ERROR,
                name=f"Range validation for {property_name}",
                description="Value must be within specified range",
                parameters={
                    "rule_class": "range",
                    "minValue": min_value,
                    "maxValue": max_value,
                    "inclusive": business_config.get("inclusive", True),
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        # Required words (value must contain certain words)
        required_words = business_config.get("requiredWords")
        if required_words:
            # Convert to pattern rule
            pattern = "|".join(required_words)  # Simple OR pattern
            rule_config = QualityRuleConfig(
                rule_id=f"{entity_type}.{property_name}.required_words",
                rule_type=QualityRuleType.BUSINESS,
                severity=QualitySeverity.WARNING,
                name=f"Required words check for {property_name}",
                description=f"Value should contain one of: {', '.join(required_words)}",
                parameters={
                    "rule_class": "pattern",
                    "pattern": f".*({pattern}).*",
                    "flags": 2,  # re.IGNORECASE
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        # Forbidden patterns (regex patterns that should not match)
        forbidden_patterns = business_config.get("forbiddenPatterns")
        if forbidden_patterns:
            for i, pattern in enumerate(forbidden_patterns):
                rule_config = QualityRuleConfig(
                    rule_id=f"{entity_type}.{property_name}.forbidden_pattern_{i}",
                    rule_type=QualityRuleType.BUSINESS,
                    severity=QualitySeverity.WARNING,
                    name=f"Forbidden pattern check for {property_name}",
                    description=f"Value should not match pattern: {pattern}",
                    parameters={
                        "rule_class": "pattern",
                        "pattern": f"^((?!{pattern}).)*$",  # Negative lookahead
                        "flags": 0,
                    },
                )
                rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_validation_rules(
        self, entity_type: str, property_name: str, validation_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse validation-specific quality rules."""
        rules = []

        # Date range validations
        date_window = validation_config.get("dateWindow")
        if date_window:
            rule_config = QualityRuleConfig(
                rule_id=f"{entity_type}.{property_name}.date_window",
                rule_type=QualityRuleType.BUSINESS,
                severity=self._parse_severity(
                    date_window.get("severity"), default=QualitySeverity.WARNING
                ),
                name=f"Date window validation for {property_name}",
                description="Date must fall within configured window",
                parameters={
                    "rule_class": "date_window",
                    "earliest": date_window.get("earliest"),
                    "latest": date_window.get("latest"),
                    "allow_future": date_window.get("allowFuture", False),
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_entity_level_rules(
        self, entity_type: str, entity_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse entity-level quality rules."""
        rules = []

        completeness = entity_config.get("completeness")
        if completeness:
            min_ratio = completeness.get("minPropertyFillRate")
            if min_ratio is not None:
                rule_config = QualityRuleConfig(
                    rule_id=f"{entity_type}.completeness.min_fill_rate",
                    rule_type=QualityRuleType.CONSISTENCY,
                    severity=self._parse_severity(
                        completeness.get("severity"),
                        default=QualitySeverity.WARNING,
                    ),
                    name=f"{entity_type} minimum property fill rate",
                    description=f"Entities must meet minimum property fill rate of {min_ratio}",
                    parameters={
                        "rule_class": "entity_completeness",
                        "entity_type": entity_type,
                        "min_ratio": float(min_ratio),
                    },
                )
                rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_relationship_level_rules(
        self,
        entity_type: str,
        relationship_name: str,
        relationship_config: Dict[str, Any],
    ) -> List[QualityRule]:
        """Parse relationship-level quality rules."""
        rules = []

        cardinality = relationship_config.get("cardinality")
        if cardinality:
            min_count = cardinality.get("minPerEntity")
            if min_count is not None:
                rule_config = QualityRuleConfig(
                    rule_id=f"relationship.{relationship_name}.cardinality",
                    rule_type=QualityRuleType.CONSISTENCY,
                    severity=self._parse_severity(
                        cardinality.get("severity"), default=QualitySeverity.WARNING
                    ),
                    name=f"Relationship cardinality for {relationship_name}",
                    description="Ensure minimum relationship count per entity",
                    parameters={
                        "rule_class": "relationship_presence",
                        "entity_type": cardinality.get("entityType", entity_type),
                        "relationship_type": relationship_name,
                        "direction": cardinality.get("direction", "outbound"),
                        "min_count": int(min_count),
                    },
                )
                rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_extraction_standards(
        self, extraction_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse extraction standard rules (confidence, completeness)."""
        rules: List[QualityRule] = []

        if not isinstance(extraction_config, dict):
            return rules

        confidence_thresholds = extraction_config.get("confidenceThresholds")
        if isinstance(confidence_thresholds, dict) and confidence_thresholds:
            rule_config = QualityRuleConfig(
                rule_id="global.confidence_threshold",
                rule_type=QualityRuleType.DISTRIBUTION,
                severity=self._parse_severity(
                    confidence_thresholds.get("severity"),
                    default=QualitySeverity.ERROR,
                ),
                name="Minimum extraction confidence",
                description="Ensure extracted entities and relationships meet minimum confidence thresholds",
                parameters={
                    "rule_class": "confidence_threshold",
                    "entity_threshold": confidence_thresholds.get("entityExtraction"),
                    "relationship_threshold": confidence_thresholds.get(
                        "relationshipExtraction"
                    ),
                    "property_threshold": confidence_thresholds.get(
                        "propertyExtraction"
                    ),
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        completeness_requirements = extraction_config.get("completenessRequirements")
        if isinstance(completeness_requirements, dict) and completeness_requirements:
            completeness_severity = self._parse_severity(
                completeness_requirements.get("severity"),
                default=QualitySeverity.ERROR,
            )

            min_entities = completeness_requirements.get("minEntitiesPerDocument")
            if min_entities is not None:
                rule_config = QualityRuleConfig(
                    rule_id="global.completeness.min_entities",
                    rule_type=QualityRuleType.CROSS_ENTITY,
                    severity=self._parse_severity(
                        completeness_requirements.get("minEntitiesSeverity"),
                        default=completeness_severity,
                    ),
                    name="Minimum entities per document",
                    description="Ensure each document produces the minimum expected entity count",
                    parameters={
                        "rule_class": "minimum_entities",
                        "min_entities": int(min_entities),
                    },
                )
                rules.append(QualityRuleFactory.create_rule(rule_config))

            required_types = completeness_requirements.get("requiredEntityTypes")
            if required_types:
                rule_config = QualityRuleConfig(
                    rule_id="global.completeness.required_entity_types",
                    rule_type=QualityRuleType.CROSS_ENTITY,
                    severity=self._parse_severity(
                        completeness_requirements.get("requiredTypesSeverity"),
                        default=completeness_severity,
                    ),
                    name="Required entity types present",
                    description="Ensure extraction includes key entity types",
                    parameters={
                        "rule_class": "required_entity_types",
                        "required_types": required_types,
                    },
                )
                rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_distribution_rules(
        self, distribution_config: Dict[str, Any]
    ) -> List[QualityRule]:
        """Parse distribution/statistical quality rules."""
        rules = []

        relationship_requirements = distribution_config.get(
            "relationshipRequirements", []
        )
        for idx, requirement in enumerate(relationship_requirements):
            rule_config = QualityRuleConfig(
                rule_id=f"global.relationship_requirement.{idx}",
                rule_type=QualityRuleType.CROSS_ENTITY,
                severity=self._parse_severity(
                    requirement.get("severity"), default=QualitySeverity.WARNING
                ),
                name="Relationship presence requirement",
                description="Ensure entities satisfy minimum relationship counts",
                parameters={
                    "rule_class": "relationship_presence",
                    "entity_type": requirement.get("entityType"),
                    "relationship_type": requirement.get("relationshipType"),
                    "direction": requirement.get("direction", "outbound"),
                    "min_count": int(requirement.get("minCount", 1)),
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        entity_balance = distribution_config.get("entityBalance")
        if isinstance(entity_balance, dict):
            rule_config = QualityRuleConfig(
                rule_id="global.entity_balance",
                rule_type=QualityRuleType.DISTRIBUTION,
                severity=self._parse_severity(
                    entity_balance.get("severity"), default=QualitySeverity.WARNING
                ),
                name="Entity balance expectations",
                description="Ensure entity type ratios stay within configured bounds",
                parameters={
                    "rule_class": "entity_balance",
                    "max_single_type_ratio": entity_balance.get("maxSingleTypeRatio"),
                    "expected_ratios": entity_balance.get("expectedRatios", {}),
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_coverage_rules(
        self, coverage_rules: List[Dict[str, Any]]
    ) -> List[QualityRule]:
        rules: List[QualityRule] = []

        for idx, rule_def in enumerate(coverage_rules):
            entity_type = rule_def.get("entityType")
            property_name = rule_def.get("property")
            min_coverage = rule_def.get("minCoverage")

            if entity_type is None or property_name is None or min_coverage is None:
                logger.warning(
                    "Skipping property coverage rule %s due to missing required fields",
                    idx,
                )
                continue

            rule_config = QualityRuleConfig(
                rule_id=f"global.property_coverage.{idx}",
                rule_type=QualityRuleType.CROSS_ENTITY,
                severity=self._parse_severity(
                    rule_def.get("severity"), default=QualitySeverity.WARNING
                ),
                name=f"Property coverage for {entity_type}.{property_name}",
                description="Ensure sufficient property coverage across entities",
                parameters={
                    "rule_class": "property_coverage",
                    "entity_type": entity_type,
                    "property_name": property_name,
                    "min_coverage": float(min_coverage),
                    "allow_missing_entities": rule_def.get(
                        "allowMissingEntities", True
                    ),
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_cross_validation_rules(
        self, cross_validation_config: List[Dict[str, Any]]
    ) -> List[QualityRule]:
        """Parse cross-entity validation rules."""
        rules = []

        for idx, rule_def in enumerate(cross_validation_config):
            rule_type = rule_def.get("ruleType")
            if rule_type == "symmetric_relationship":
                rule_config = QualityRuleConfig(
                    rule_id=f"global.symmetric_relationship.{idx}",
                    rule_type=QualityRuleType.CROSS_ENTITY,
                    severity=self._parse_severity(
                        rule_def.get("severity"), default=QualitySeverity.WARNING
                    ),
                    name=f"Symmetric relationship check for {rule_def.get('relationshipType')}",
                    description="Ensure relationships are symmetric",
                    parameters={
                        "rule_class": "symmetric_relationship",
                        "relationship_type": rule_def.get("relationshipType"),
                        "inverse_type": rule_def.get("inverseType"),
                    },
                )
                rules.append(QualityRuleFactory.create_rule(rule_config))
            elif rule_type == "property_alignment":
                rule_config = QualityRuleConfig(
                    rule_id=f"global.property_alignment.{idx}",
                    rule_type=QualityRuleType.CROSS_ENTITY,
                    severity=self._parse_severity(
                        rule_def.get("severity"), default=QualitySeverity.WARNING
                    ),
                    name="Cross-entity property alignment",
                    description="Ensure related entities share consistent properties",
                    parameters={
                        "rule_class": "cross_entity_consistency",
                        "relationship_type": rule_def.get("relationshipType"),
                        "source_property": rule_def.get("sourceProperty"),
                        "target_property": rule_def.get("targetProperty"),
                        "case_insensitive": rule_def.get("caseInsensitive", True),
                        "allow_missing": rule_def.get("allowMissing", False),
                        "allowed_pairs": rule_def.get("allowedPairs", []),
                    },
                )
                rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_temporal_rules(
        self, temporal_rules: List[Dict[str, Any]]
    ) -> List[QualityRule]:
        rules: List[QualityRule] = []

        for idx, rule_def in enumerate(temporal_rules):
            entity_type = rule_def.get("entityType")
            property_name = rule_def.get("property")

            if not entity_type or not property_name:
                logger.warning(
                    "Skipping temporal rule %s due to missing entityType/property",
                    idx,
                )
                continue

            rule_config = QualityRuleConfig(
                rule_id=f"global.temporal.{entity_type}.{property_name}.{idx}",
                rule_type=QualityRuleType.CROSS_ENTITY,
                severity=self._parse_severity(
                    rule_def.get("severity"), default=QualitySeverity.WARNING
                ),
                name=f"Temporal window for {entity_type}.{property_name}",
                description="Ensure dates fall within configured window",
                parameters={
                    "rule_class": "global_date_window",
                    "entity_type": entity_type,
                    "property_name": property_name,
                    "earliest": rule_def.get("earliest"),
                    "latest": rule_def.get("latest"),
                    "allow_future": rule_def.get("allowFuture", False),
                    "allow_missing": rule_def.get("allowMissing", True),
                },
            )
            rules.append(QualityRuleFactory.create_rule(rule_config))

        return rules

    def _parse_severity(
        self, severity_value: Optional[str], *, default: QualitySeverity
    ) -> QualitySeverity:
        if not severity_value:
            return default
        try:
            return QualitySeverity(severity_value.lower())
        except ValueError:
            return default

    def get_llm_prompt_instructions(self) -> List[str]:
        """Generate LLM prompt instructions based on all quality rules."""
        instructions = []

        # Collect instructions from all entity property rules
        for entity_type, entity_def in self.ontology.get("entities", {}).items():
            for prop_name, prop_def in entity_def.get("properties", {}).items():
                quality_config = prop_def.get("quality", {})
                if quality_config:
                    rules = self.parse_property_quality_rules(
                        entity_type, prop_name, quality_config
                    )
                    for rule in rules:
                        instruction = rule.get_llm_prompt_instruction()
                        if instruction:
                            instructions.append(
                                f"{entity_type}.{prop_name}: {instruction}"
                            )

        return instructions
