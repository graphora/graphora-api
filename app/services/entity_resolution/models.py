"""Entity Resolution Models.

Domain-agnostic data structures for entity resolution configuration.
All domain knowledge flows from user-provided ontology metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class DataType(str, Enum):
    """Data types that determine comparison methods.

    These are generic types, not domain-specific field names.
    The comparison method is derived from the data type, not hardcoded.
    """

    STRING = "string"
    TEXT = "text"  # Long-form text, uses embedding similarity
    DATE = "date"
    DATETIME = "datetime"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    IDENTIFIER = "identifier"  # Unique identifiers, exact match
    PHONE = "phone"
    LIST = "list"


class ComparisonMethod(str, Enum):
    """Available comparison methods for matching."""

    EXACT = "exact"
    EXACT_NORMALIZED = "exact_normalized"
    JARO_WINKLER = "jaro_winkler"
    LEVENSHTEIN = "levenshtein"
    EMBEDDING = "embedding"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    DATE_TOLERANCE = "date_tolerance"
    METAPHONE = "metaphone"
    SOUNDEX = "soundex"
    JACCARD = "jaccard"  # For lists/sets


@dataclass(frozen=True)
class ComparisonPrior:
    """Probabilistic priors for comparison levels.

    m: probability of each similarity level given a true match
    u: probability of each similarity level given a non-match

    The tuple length depends on the comparison method:
    - 2 levels: exact match (match, no_match)
    - 4 levels: string similarity (exact, high, medium, low)
    """

    m: Tuple[float, ...]
    u: Tuple[float, ...]

    def __post_init__(self):
        if len(self.m) != len(self.u):
            raise ValueError("m and u tuples must have same length")
        if abs(sum(self.m) - 1.0) > 0.01 or abs(sum(self.u) - 1.0) > 0.01:
            raise ValueError("m and u probabilities must sum to 1.0")


@dataclass
class PropertyMatchingConfig:
    """Matching configuration for a single property.

    This is derived from ontology property metadata, not hardcoded.
    """

    property_name: str
    data_type: DataType
    is_identifier: bool = False
    is_blocking_key: bool = False
    matching_weight: float = 1.0
    canonicalization: Optional[str] = (
        None  # e.g., "lowercase_trim", "strip_company_suffix"
    )
    comparison_methods: List[ComparisonMethod] = field(default_factory=list)
    tolerance: Optional[float] = None  # For numeric/date comparisons

    def __post_init__(self):
        # Set default comparison methods based on data type if not specified
        if not self.comparison_methods:
            self.comparison_methods = self._default_methods_for_type()

    def _default_methods_for_type(self) -> List[ComparisonMethod]:
        """Get default comparison methods based on data type."""
        type_to_methods = {
            DataType.STRING: [ComparisonMethod.JARO_WINKLER],
            DataType.TEXT: [ComparisonMethod.EMBEDDING],
            DataType.DATE: [ComparisonMethod.DATE_TOLERANCE],
            DataType.DATETIME: [ComparisonMethod.DATE_TOLERANCE],
            DataType.NUMBER: [ComparisonMethod.NUMERIC_TOLERANCE],
            DataType.INTEGER: [ComparisonMethod.NUMERIC_TOLERANCE],
            DataType.FLOAT: [ComparisonMethod.NUMERIC_TOLERANCE],
            DataType.BOOLEAN: [ComparisonMethod.EXACT],
            DataType.EMAIL: [ComparisonMethod.EXACT_NORMALIZED],
            DataType.URL: [ComparisonMethod.EXACT_NORMALIZED],
            DataType.IDENTIFIER: [ComparisonMethod.EXACT],
            DataType.PHONE: [ComparisonMethod.EXACT_NORMALIZED],
            DataType.LIST: [ComparisonMethod.JACCARD],
        }
        return type_to_methods.get(self.data_type, [ComparisonMethod.EXACT])


@dataclass
class ComparisonRule:
    """A complete comparison rule for entity resolution.

    Generated from PropertyMatchingConfig, ready for use in Splink.
    """

    property_name: str
    comparison_method: ComparisonMethod
    prior: ComparisonPrior
    weight: float = 1.0
    thresholds: Optional[List[float]] = None  # For threshold-based comparisons
    tolerance: Optional[float] = None

    @property
    def is_high_confidence(self) -> bool:
        """Whether this rule provides high confidence matching."""
        # High confidence if identifier or high weight
        return self.weight >= 2.0 or (
            self.comparison_method == ComparisonMethod.EXACT and self.prior.m[0] >= 0.9
        )


@dataclass
class BlockingRule:
    """A blocking rule to reduce comparison pairs.

    Generated from ontology metadata (is_identifier, is_blocking_key).
    """

    property_name: str
    method: str  # e.g., "exact", "first_n_chars", "metaphone", "lsh"
    params: Dict[str, Any] = field(default_factory=dict)

    def to_splink_rule(self, table_alias: str = "l") -> str:
        """Convert to Splink blocking rule string."""
        if self.method == "exact":
            return f'{table_alias}."{self.property_name}" = r."{self.property_name}"'
        elif self.method == "first_n_chars":
            n = self.params.get("n", 4)
            return f'SUBSTR({table_alias}."{self.property_name}", 1, {n}) = SUBSTR(r."{self.property_name}", 1, {n})'
        elif self.method == "metaphone":
            return f'METAPHONE({table_alias}."{self.property_name}") = METAPHONE(r."{self.property_name}")'
        else:
            # Default to exact
            return f'{table_alias}."{self.property_name}" = r."{self.property_name}"'


@dataclass
class EntityResolutionConfig:
    """Complete configuration for entity resolution.

    This is the main configuration object passed to the resolution pipeline.
    All values are derived from ontology + user settings, not hardcoded.
    """

    entity_type: str
    comparison_rules: List[ComparisonRule] = field(default_factory=list)
    blocking_rules: List[BlockingRule] = field(default_factory=list)

    # Thresholds (configurable, not hardcoded)
    match_threshold: float = 0.7
    review_threshold: float = 0.5

    # Feature flags
    use_embedding_similarity: bool = True
    use_lsh_blocking: bool = True

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache_enabled: bool = True

    # Performance settings
    max_comparisons_per_entity: int = 1000
    batch_size: int = 100


# Default priors based on data type (not hardcoded field names)
# These can be overridden by user configuration
DEFAULT_PRIORS: Dict[DataType, ComparisonPrior] = {
    # Identifier types: high confidence
    DataType.IDENTIFIER: ComparisonPrior(m=(0.97, 0.03), u=(0.02, 0.98)),
    DataType.EMAIL: ComparisonPrior(m=(0.95, 0.05), u=(0.03, 0.97)),
    # Numeric types: high confidence for exact, tolerance for approximate
    DataType.NUMBER: ComparisonPrior(m=(0.90, 0.10), u=(0.05, 0.95)),
    DataType.INTEGER: ComparisonPrior(m=(0.90, 0.10), u=(0.05, 0.95)),
    DataType.FLOAT: ComparisonPrior(m=(0.85, 0.15), u=(0.10, 0.90)),
    # Date types
    DataType.DATE: ComparisonPrior(m=(0.90, 0.10), u=(0.05, 0.95)),
    DataType.DATETIME: ComparisonPrior(m=(0.90, 0.10), u=(0.05, 0.95)),
    # String types: 4-level similarity
    DataType.STRING: ComparisonPrior(
        m=(0.85, 0.10, 0.04, 0.01), u=(0.05, 0.10, 0.15, 0.70)
    ),
    DataType.TEXT: ComparisonPrior(
        m=(0.80, 0.12, 0.05, 0.03), u=(0.10, 0.15, 0.20, 0.55)
    ),
    # Other types
    DataType.URL: ComparisonPrior(m=(0.92, 0.08), u=(0.08, 0.92)),
    DataType.PHONE: ComparisonPrior(m=(0.93, 0.07), u=(0.07, 0.93)),
    DataType.BOOLEAN: ComparisonPrior(m=(0.99, 0.01), u=(0.50, 0.50)),
    DataType.LIST: ComparisonPrior(m=(0.80, 0.15, 0.05), u=(0.20, 0.30, 0.50)),
}


def get_prior_for_type(
    data_type: DataType, is_identifier: bool = False
) -> ComparisonPrior:
    """Get the appropriate prior for a data type.

    Args:
        data_type: The property data type
        is_identifier: Whether this property is marked as an identifier

    Returns:
        ComparisonPrior with appropriate m/u probabilities
    """
    if is_identifier:
        # Identifiers get high-confidence priors regardless of data type
        return ComparisonPrior(m=(0.97, 0.03), u=(0.02, 0.98))

    return DEFAULT_PRIORS.get(
        data_type,
        # Fallback for unknown types
        ComparisonPrior(m=(0.82, 0.18), u=(0.12, 0.88)),
    )
