"""Blocking Rule Generator for Entity Resolution.

Generates blocking rules from ontology metadata to reduce comparison pairs.
Domain-agnostic - rules are derived from property metadata, not hardcoded.
"""

import logging
from typing import Any, Dict, List, Set, Tuple
import hashlib

from graphora_server.services.entity_resolution.models import (
    BlockingRule,
    DataType,
)

logger = logging.getLogger(__name__)


class BlockingRuleGenerator:
    """Generate blocking rules from ontology property definitions.

    Blocking rules reduce the number of entity pairs that need to be compared
    by grouping entities that are likely to match.

    All blocking rules are derived from ontology metadata:
    - is_identifier: Properties marked as identifiers are excellent blocking keys
    - is_blocking_key: Explicit blocking keys defined by user
    - data_type: The type determines appropriate blocking method
    """

    # Mapping from ontology type strings to DataType enum
    TYPE_STRING_MAPPING: Dict[str, DataType] = {
        "str": DataType.STRING,
        "string": DataType.STRING,
        "text": DataType.TEXT,
        "int": DataType.INTEGER,
        "integer": DataType.INTEGER,
        "number": DataType.NUMBER,
        "float": DataType.FLOAT,
        "date": DataType.DATE,
        "datetime": DataType.DATETIME,
        "email": DataType.EMAIL,
        "url": DataType.URL,
        "phone": DataType.PHONE,
        "id": DataType.IDENTIFIER,
        "identifier": DataType.IDENTIFIER,
        "code": DataType.IDENTIFIER,
    }

    def __init__(
        self,
        max_blocking_rules: int = 5,
        enable_lsh: bool = True,
        lsh_num_hashes: int = 100,
        lsh_bands: int = 20,
    ):
        """Initialize the blocking rule generator.

        Args:
            max_blocking_rules: Maximum number of blocking rules to generate
            enable_lsh: Whether to use LSH for text properties
            lsh_num_hashes: Number of hash functions for LSH
            lsh_bands: Number of bands for LSH banding
        """
        self.max_blocking_rules = max_blocking_rules
        self.enable_lsh = enable_lsh
        self.lsh_num_hashes = lsh_num_hashes
        self.lsh_bands = lsh_bands

    def generate_rules_for_entity(
        self,
        entity_type: str,
        entity_definition: Dict[str, Any],
    ) -> List[BlockingRule]:
        """Generate blocking rules for an entity type from its ontology definition.

        Args:
            entity_type: The name of the entity type
            entity_definition: The entity definition from the ontology

        Returns:
            List of BlockingRule objects for candidate generation
        """
        rules = []
        properties = entity_definition.get("properties", {})

        # Priority 1: Explicit blocking keys from ontology
        blocking_key_rules = self._get_explicit_blocking_keys(properties)
        rules.extend(blocking_key_rules)

        # Priority 2: Identifier properties
        if len(rules) < self.max_blocking_rules:
            identifier_rules = self._get_identifier_blocking_rules(properties)
            for rule in identifier_rules:
                if rule.property_name not in [r.property_name for r in rules]:
                    rules.append(rule)
                    if len(rules) >= self.max_blocking_rules:
                        break

        # Priority 3: High-weight properties
        if len(rules) < self.max_blocking_rules:
            weight_rules = self._get_high_weight_blocking_rules(properties)
            for rule in weight_rules:
                if rule.property_name not in [r.property_name for r in rules]:
                    rules.append(rule)
                    if len(rules) >= self.max_blocking_rules:
                        break

        # Priority 4: Default to first few string properties if no rules yet
        if not rules:
            default_rules = self._get_default_blocking_rules(properties)
            rules.extend(default_rules[: self.max_blocking_rules])

        logger.debug(
            f"Generated {len(rules)} blocking rules for entity type '{entity_type}'"
        )
        return rules

    def _get_explicit_blocking_keys(
        self,
        properties: Dict[str, Any],
    ) -> List[BlockingRule]:
        """Get blocking rules for properties marked with is_blocking_key."""
        rules = []

        for prop_name, prop_def in properties.items():
            if prop_def.get("is_blocking_key", False):
                data_type = self._get_data_type(prop_def)
                method = self._get_blocking_method_for_type(data_type)
                rules.append(
                    BlockingRule(
                        property_name=prop_name,
                        method=method,
                        params=self._get_blocking_params(data_type, method),
                    )
                )

        return rules

    def _get_identifier_blocking_rules(
        self,
        properties: Dict[str, Any],
    ) -> List[BlockingRule]:
        """Get blocking rules for properties marked as identifiers."""
        rules = []

        for prop_name, prop_def in properties.items():
            if prop_def.get("is_identifier", False):
                # Identifiers use exact blocking
                rules.append(
                    BlockingRule(
                        property_name=prop_name,
                        method="exact",
                        params={},
                    )
                )

        return rules

    def _get_high_weight_blocking_rules(
        self,
        properties: Dict[str, Any],
    ) -> List[BlockingRule]:
        """Get blocking rules for high-weight properties."""
        # Sort properties by matching_weight
        weighted_props = [
            (prop_name, prop_def, prop_def.get("matching_weight", 1.0))
            for prop_name, prop_def in properties.items()
        ]
        weighted_props.sort(key=lambda x: x[2], reverse=True)

        rules = []
        for prop_name, prop_def, weight in weighted_props:
            if weight >= 1.5:  # Consider high-weight properties
                data_type = self._get_data_type(prop_def)
                method = self._get_blocking_method_for_type(data_type)
                rules.append(
                    BlockingRule(
                        property_name=prop_name,
                        method=method,
                        params=self._get_blocking_params(data_type, method),
                    )
                )

        return rules

    def _get_default_blocking_rules(
        self,
        properties: Dict[str, Any],
    ) -> List[BlockingRule]:
        """Get default blocking rules when no explicit keys are defined."""
        rules = []

        for prop_name, prop_def in properties.items():
            data_type = self._get_data_type(prop_def)

            # Skip text/long properties for default blocking (too variable)
            if data_type == DataType.TEXT:
                continue

            method = self._get_blocking_method_for_type(data_type)
            rules.append(
                BlockingRule(
                    property_name=prop_name,
                    method=method,
                    params=self._get_blocking_params(data_type, method),
                )
            )

            if len(rules) >= 3:  # Limit default rules
                break

        return rules

    def _get_data_type(self, prop_def: Dict[str, Any]) -> DataType:
        """Get DataType from property definition."""
        type_str = prop_def.get("type", "str").lower()
        return self.TYPE_STRING_MAPPING.get(type_str, DataType.STRING)

    def _get_blocking_method_for_type(self, data_type: DataType) -> str:
        """Get appropriate blocking method for a data type."""
        type_to_method = {
            DataType.IDENTIFIER: "exact",
            DataType.EMAIL: "exact",
            DataType.URL: "exact",
            DataType.INTEGER: "exact",
            DataType.NUMBER: "numeric_bucket",
            DataType.FLOAT: "numeric_bucket",
            DataType.DATE: "date_bucket",
            DataType.DATETIME: "date_bucket",
            DataType.STRING: "first_n_chars",
            DataType.TEXT: "lsh" if self.enable_lsh else "first_n_chars",
            DataType.PHONE: "suffix",
        }
        return type_to_method.get(data_type, "first_n_chars")

    def _get_blocking_params(
        self,
        data_type: DataType,
        method: str,
    ) -> Dict[str, Any]:
        """Get parameters for a blocking method."""
        if method == "first_n_chars":
            return {"n": 4}
        elif method == "suffix":
            return {"n": 4}
        elif method == "numeric_bucket":
            return {"bucket_size": 10}
        elif method == "date_bucket":
            return {"bucket_days": 30}
        elif method == "lsh":
            return {
                "num_hashes": self.lsh_num_hashes,
                "bands": self.lsh_bands,
            }
        return {}


class LSHBlocker:
    """Locality-Sensitive Hashing for blocking text properties.

    LSH groups similar texts together without comparing all pairs,
    enabling efficient blocking for large entity sets.
    """

    def __init__(
        self,
        num_hashes: int = 100,
        bands: int = 20,
        ngram_size: int = 3,
    ):
        """Initialize LSH blocker.

        Args:
            num_hashes: Number of minhash functions
            bands: Number of bands for banding technique
            ngram_size: Size of character n-grams
        """
        self.num_hashes = num_hashes
        self.bands = bands
        self.ngram_size = ngram_size
        self.rows_per_band = num_hashes // bands

        # Generate random hash coefficients
        import random

        random.seed(42)  # Reproducible
        self._hash_coeffs = [
            (random.randint(1, 2**31 - 1), random.randint(0, 2**31 - 1))
            for _ in range(num_hashes)
        ]

    def _get_ngrams(self, text: str) -> Set[str]:
        """Convert text to character n-grams."""
        text = text.lower().strip()
        if len(text) < self.ngram_size:
            return {text}

        return {
            text[i : i + self.ngram_size]
            for i in range(len(text) - self.ngram_size + 1)
        }

    def _minhash(self, ngrams: Set[str]) -> List[int]:
        """Compute minhash signature for a set of n-grams."""
        if not ngrams:
            return [0] * self.num_hashes

        signature = []
        for a, b in self._hash_coeffs:
            min_hash = float("inf")
            for ngram in ngrams:
                # Hash the n-gram
                h = hash(ngram)
                # Apply hash function
                hash_val = (a * h + b) % (2**31 - 1)
                min_hash = min(min_hash, hash_val)
            signature.append(min_hash)

        return signature

    def _get_bands(self, signature: List[int]) -> List[Tuple[int, ...]]:
        """Split signature into bands."""
        bands = []
        for i in range(self.bands):
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            bands.append(band)
        return bands

    def compute_buckets(self, text: str) -> List[str]:
        """Compute LSH buckets for a text.

        Texts with at least one shared bucket are candidates for comparison.

        Args:
            text: The text to hash

        Returns:
            List of bucket keys
        """
        ngrams = self._get_ngrams(text)
        signature = self._minhash(ngrams)
        bands = self._get_bands(signature)

        # Create bucket keys from band values
        buckets = []
        for i, band in enumerate(bands):
            # Hash the band tuple to create a bucket key
            band_hash = hashlib.md5(f"{i}:{band}".encode()).hexdigest()[:12]
            buckets.append(f"band_{i}_{band_hash}")

        return buckets

    def find_candidate_pairs(
        self,
        entities: List[Dict[str, Any]],
        text_property: str,
    ) -> List[Tuple[int, int]]:
        """Find candidate pairs using LSH blocking.

        Args:
            entities: List of entity dictionaries
            text_property: Name of the text property to block on

        Returns:
            List of (index1, index2) tuples representing candidate pairs
        """
        # Build bucket index
        bucket_to_indices: Dict[str, List[int]] = {}

        for idx, entity in enumerate(entities):
            text = entity.get(text_property, "")
            if not text:
                continue

            buckets = self.compute_buckets(str(text))
            for bucket in buckets:
                if bucket not in bucket_to_indices:
                    bucket_to_indices[bucket] = []
                bucket_to_indices[bucket].append(idx)

        # Generate candidate pairs from shared buckets
        candidate_pairs: Set[Tuple[int, int]] = set()

        for indices in bucket_to_indices.values():
            if len(indices) < 2:
                continue

            # Generate pairs from entities in same bucket
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    # Ensure consistent ordering
                    pair = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                    candidate_pairs.add(pair)

        return list(candidate_pairs)


def generate_blocking_rules_from_ontology(
    parsed_ontology: Dict[str, Any],
    entity_type: str,
    enable_lsh: bool = True,
) -> List[BlockingRule]:
    """Convenience function to generate blocking rules from a parsed ontology.

    Args:
        parsed_ontology: The full parsed ontology dict
        entity_type: The entity type to generate rules for
        enable_lsh: Whether to enable LSH blocking

    Returns:
        List of BlockingRule objects

    Raises:
        ValueError: If entity_type not found in ontology
    """
    entities = parsed_ontology.get("entities", {})
    if entity_type not in entities:
        raise ValueError(f"Entity type '{entity_type}' not found in ontology")

    generator = BlockingRuleGenerator(enable_lsh=enable_lsh)
    return generator.generate_rules_for_entity(entity_type, entities[entity_type])
