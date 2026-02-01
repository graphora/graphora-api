"""Entity Resolution Module.

Provides domain-agnostic entity resolution with:
- Ontology-driven comparison rules
- Embedding-based semantic similarity
- LSH blocking for efficient candidate generation
- Persistent entity store for cross-document resolution
"""

from app.services.entity_resolution.models import (
    PropertyMatchingConfig,
    ComparisonMethod,
    ComparisonPrior,
    DataType,
    ComparisonRule,
    BlockingRule,
    EntityResolutionConfig,
    get_prior_for_type,
)
from app.services.entity_resolution.comparison_rules import (
    ComparisonRuleGenerator,
    generate_rules_from_ontology,
    generate_config_from_ontology,
)
from app.services.entity_resolution.embedding_similarity import (
    EmbeddingSimilarity,
    get_embedding_similarity,
)
from app.services.entity_resolution.blocking import (
    BlockingRuleGenerator,
    LSHBlocker,
    generate_blocking_rules_from_ontology,
)
from app.services.entity_resolution.entity_store import (
    EntityStore,
    CrossDocumentResolver,
)

__all__ = [
    # Models
    "PropertyMatchingConfig",
    "ComparisonMethod",
    "ComparisonPrior",
    "DataType",
    "ComparisonRule",
    "BlockingRule",
    "EntityResolutionConfig",
    "get_prior_for_type",
    # Comparison Rules
    "ComparisonRuleGenerator",
    "generate_rules_from_ontology",
    "generate_config_from_ontology",
    # Embedding Similarity
    "EmbeddingSimilarity",
    "get_embedding_similarity",
    # Blocking
    "BlockingRuleGenerator",
    "LSHBlocker",
    "generate_blocking_rules_from_ontology",
    # Entity Store
    "EntityStore",
    "CrossDocumentResolver",
]
