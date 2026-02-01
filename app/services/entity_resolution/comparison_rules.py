"""Comparison Rule Generator.

Generates entity resolution comparison rules from ontology metadata.
All domain knowledge flows from the user-provided ontology - nothing hardcoded.
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.entity_resolution.models import (
    ComparisonMethod,
    ComparisonPrior,
    ComparisonRule,
    DataType,
    PropertyMatchingConfig,
    EntityResolutionConfig,
    get_prior_for_type,
)

logger = logging.getLogger(__name__)


class ComparisonRuleGenerator:
    """Generate comparison rules from ontology property definitions.

    This class derives all matching behavior from the ontology schema,
    ensuring the system remains domain-agnostic.
    """

    # Mapping from ontology type strings to DataType enum
    # Covers common type names users might use in their ontology
    TYPE_STRING_MAPPING: Dict[str, DataType] = {
        # String types
        "str": DataType.STRING,
        "string": DataType.STRING,
        "varchar": DataType.STRING,
        "char": DataType.STRING,
        "name": DataType.STRING,
        # Text types (long-form, use embeddings)
        "text": DataType.TEXT,
        "description": DataType.TEXT,
        "content": DataType.TEXT,
        "body": DataType.TEXT,
        "longtext": DataType.TEXT,
        # Numeric types
        "int": DataType.INTEGER,
        "integer": DataType.INTEGER,
        "number": DataType.NUMBER,
        "float": DataType.FLOAT,
        "double": DataType.FLOAT,
        "decimal": DataType.FLOAT,
        "numeric": DataType.NUMBER,
        # Date/time types
        "date": DataType.DATE,
        "datetime": DataType.DATETIME,
        "timestamp": DataType.DATETIME,
        "time": DataType.DATETIME,
        # Boolean
        "bool": DataType.BOOLEAN,
        "boolean": DataType.BOOLEAN,
        # Special types
        "email": DataType.EMAIL,
        "url": DataType.URL,
        "uri": DataType.URL,
        "phone": DataType.PHONE,
        "telephone": DataType.PHONE,
        "id": DataType.IDENTIFIER,
        "identifier": DataType.IDENTIFIER,
        "uuid": DataType.IDENTIFIER,
        "code": DataType.IDENTIFIER,
        # Collection types
        "list": DataType.LIST,
        "array": DataType.LIST,
        "set": DataType.LIST,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the generator with optional configuration.

        Args:
            config: Optional configuration overrides
                - custom_type_mappings: Additional type string -> DataType mappings
                - default_weight: Default matching weight (default: 1.0)
                - enable_embedding: Whether to enable embedding similarity (default: True)
        """
        self.config = config or {}
        self.type_mapping = {**self.TYPE_STRING_MAPPING}

        # Apply custom type mappings if provided
        if "custom_type_mappings" in self.config:
            for type_str, data_type in self.config["custom_type_mappings"].items():
                self.type_mapping[type_str.lower()] = data_type

        self.default_weight = self.config.get("default_weight", 1.0)
        self.enable_embedding = self.config.get("enable_embedding", True)

    def generate_rules_for_entity(
        self,
        entity_type: str,
        entity_definition: Dict[str, Any],
    ) -> List[ComparisonRule]:
        """Generate comparison rules for an entity type from its ontology definition.

        Args:
            entity_type: The name of the entity type
            entity_definition: The entity definition from the ontology, containing
                              'properties' dict with property definitions

        Returns:
            List of ComparisonRule objects for entity resolution
        """
        rules = []
        properties = entity_definition.get("properties", {})

        for prop_name, prop_def in properties.items():
            prop_config = self._create_property_config(prop_name, prop_def)
            prop_rules = self._create_rules_for_property(prop_config)
            rules.extend(prop_rules)

        # Sort rules by weight (highest first) for more efficient matching
        rules.sort(key=lambda r: r.weight, reverse=True)

        logger.debug(
            f"Generated {len(rules)} comparison rules for entity type '{entity_type}'"
        )
        return rules

    def generate_config_for_entity(
        self,
        entity_type: str,
        entity_definition: Dict[str, Any],
        resolution_settings: Optional[Dict[str, Any]] = None,
    ) -> EntityResolutionConfig:
        """Generate complete entity resolution config from ontology definition.

        Args:
            entity_type: The name of the entity type
            entity_definition: The entity definition from the ontology
            resolution_settings: Optional settings overrides for resolution

        Returns:
            EntityResolutionConfig ready for the resolution pipeline
        """
        settings = resolution_settings or {}

        comparison_rules = self.generate_rules_for_entity(
            entity_type, entity_definition
        )

        # Import here to avoid circular dependency
        from app.services.entity_resolution.blocking import BlockingRuleGenerator

        blocking_generator = BlockingRuleGenerator()
        blocking_rules = blocking_generator.generate_rules_for_entity(
            entity_type, entity_definition
        )

        return EntityResolutionConfig(
            entity_type=entity_type,
            comparison_rules=comparison_rules,
            blocking_rules=blocking_rules,
            match_threshold=settings.get("match_threshold", 0.7),
            review_threshold=settings.get("review_threshold", 0.5),
            use_embedding_similarity=settings.get(
                "use_embedding_similarity", self.enable_embedding
            ),
            use_lsh_blocking=settings.get("use_lsh_blocking", True),
            embedding_model=settings.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_cache_enabled=settings.get("embedding_cache_enabled", True),
            max_comparisons_per_entity=settings.get("max_comparisons_per_entity", 1000),
            batch_size=settings.get("batch_size", 100),
        )

    def _create_property_config(
        self,
        prop_name: str,
        prop_def: Dict[str, Any],
    ) -> PropertyMatchingConfig:
        """Create PropertyMatchingConfig from ontology property definition.

        Args:
            prop_name: Property name
            prop_def: Property definition from ontology

        Returns:
            PropertyMatchingConfig for this property
        """
        # Get data type from ontology (with fallback to string)
        type_str = prop_def.get("type", "str").lower()
        data_type = self.type_mapping.get(type_str, DataType.STRING)

        # Check if this property has matching metadata in ontology
        is_identifier = prop_def.get("is_identifier", False)
        is_blocking_key = prop_def.get("is_blocking_key", False)
        matching_weight = prop_def.get("matching_weight", self.default_weight)
        canonicalization = prop_def.get("canonicalization")
        tolerance = prop_def.get("tolerance")

        # Get comparison methods (use defaults based on type if not specified)
        comparison_methods_raw = prop_def.get("comparison_methods", [])
        comparison_methods = []
        for method in comparison_methods_raw:
            try:
                comparison_methods.append(ComparisonMethod(method))
            except ValueError:
                logger.warning(
                    f"Unknown comparison method '{method}' for property '{prop_name}'"
                )

        return PropertyMatchingConfig(
            property_name=prop_name,
            data_type=data_type,
            is_identifier=is_identifier,
            is_blocking_key=is_blocking_key,
            matching_weight=matching_weight,
            canonicalization=canonicalization,
            comparison_methods=comparison_methods if comparison_methods else None,
            tolerance=tolerance,
        )

    def _create_rules_for_property(
        self,
        config: PropertyMatchingConfig,
    ) -> List[ComparisonRule]:
        """Create comparison rules for a property based on its config.

        Args:
            config: PropertyMatchingConfig for the property

        Returns:
            List of ComparisonRule objects (may be multiple for complex types)
        """
        rules = []

        for method in config.comparison_methods:
            prior = self._get_prior_for_method(
                config.data_type,
                method,
                config.is_identifier,
            )
            thresholds = self._get_thresholds_for_method(method)

            rule = ComparisonRule(
                property_name=config.property_name,
                comparison_method=method,
                prior=prior,
                weight=config.matching_weight,
                thresholds=thresholds,
                tolerance=config.tolerance,
            )
            rules.append(rule)

        return rules

    def _get_prior_for_method(
        self,
        data_type: DataType,
        method: ComparisonMethod,
        is_identifier: bool,
    ) -> ComparisonPrior:
        """Get appropriate prior probabilities for a comparison method.

        Args:
            data_type: The property data type
            method: The comparison method
            is_identifier: Whether this is an identifier property

        Returns:
            ComparisonPrior with m/u probabilities
        """
        # Identifiers always get high-confidence priors
        if is_identifier:
            return ComparisonPrior(m=(0.97, 0.03), u=(0.02, 0.98))

        # Exact match methods use 2-level priors
        if method in (ComparisonMethod.EXACT, ComparisonMethod.EXACT_NORMALIZED):
            base_prior = get_prior_for_type(data_type, is_identifier)
            # Ensure 2-level for exact matches
            if len(base_prior.m) == 2:
                return base_prior
            # Convert multi-level to 2-level (collapse match vs non-match)
            return ComparisonPrior(
                m=(base_prior.m[0], sum(base_prior.m[1:])),
                u=(base_prior.u[0], sum(base_prior.u[1:])),
            )

        # String similarity methods use 4-level priors
        if method in (
            ComparisonMethod.JARO_WINKLER,
            ComparisonMethod.LEVENSHTEIN,
            ComparisonMethod.EMBEDDING,
        ):
            base_prior = get_prior_for_type(data_type, is_identifier)
            # Ensure 4-level for similarity
            if len(base_prior.m) == 4:
                return base_prior
            # Expand 2-level to 4-level
            return ComparisonPrior(
                m=(
                    base_prior.m[0] * 0.8,
                    base_prior.m[0] * 0.15,
                    base_prior.m[0] * 0.04,
                    1 - base_prior.m[0],
                ),
                u=(
                    base_prior.u[0] * 0.3,
                    base_prior.u[0] * 0.3,
                    0.15,
                    1 - base_prior.u[0] * 0.6 - 0.15,
                ),
            )

        # Default: use base prior
        return get_prior_for_type(data_type, is_identifier)

    def _get_thresholds_for_method(
        self,
        method: ComparisonMethod,
    ) -> Optional[List[float]]:
        """Get default thresholds for a comparison method.

        Args:
            method: The comparison method

        Returns:
            List of thresholds or None if not applicable
        """
        # Threshold-based string comparisons
        if method in (ComparisonMethod.JARO_WINKLER, ComparisonMethod.LEVENSHTEIN):
            return [0.95, 0.85, 0.70]  # Exact, high, medium similarity

        # Embedding similarity thresholds
        if method == ComparisonMethod.EMBEDDING:
            return [0.95, 0.85, 0.70]

        return None


def generate_rules_from_ontology(
    parsed_ontology: Dict[str, Any],
    entity_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[ComparisonRule]:
    """Convenience function to generate rules from a parsed ontology.

    Args:
        parsed_ontology: The full parsed ontology dict
        entity_type: The entity type to generate rules for
        config: Optional configuration overrides

    Returns:
        List of ComparisonRule objects

    Raises:
        ValueError: If entity_type not found in ontology
    """
    entities = parsed_ontology.get("entities", {})
    if entity_type not in entities:
        raise ValueError(f"Entity type '{entity_type}' not found in ontology")

    generator = ComparisonRuleGenerator(config)
    return generator.generate_rules_for_entity(entity_type, entities[entity_type])


def generate_config_from_ontology(
    parsed_ontology: Dict[str, Any],
    entity_type: str,
    config: Optional[Dict[str, Any]] = None,
    resolution_settings: Optional[Dict[str, Any]] = None,
) -> EntityResolutionConfig:
    """Convenience function to generate complete config from a parsed ontology.

    Args:
        parsed_ontology: The full parsed ontology dict
        entity_type: The entity type to generate config for
        config: Optional configuration overrides for rule generation
        resolution_settings: Optional settings for resolution pipeline

    Returns:
        EntityResolutionConfig ready for the resolution pipeline

    Raises:
        ValueError: If entity_type not found in ontology
    """
    entities = parsed_ontology.get("entities", {})
    if entity_type not in entities:
        raise ValueError(f"Entity type '{entity_type}' not found in ontology")

    generator = ComparisonRuleGenerator(config)
    return generator.generate_config_for_entity(
        entity_type, entities[entity_type], resolution_settings
    )
