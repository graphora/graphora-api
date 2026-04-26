"""Entity Resolution Module.

Provides domain-agnostic entity resolution with:
- Ontology-driven comparison rules
- Embedding-based semantic similarity
- LSH blocking for efficient candidate generation
- Persistent entity store for cross-document resolution
"""

from graphora_server.services.entity_resolution.models import (
    PropertyMatchingConfig,
    ComparisonMethod,
    ComparisonPrior,
    DataType,
    ComparisonRule,
    BlockingRule,
    EntityResolutionConfig,
    get_prior_for_type,
)
from graphora_server.services.entity_resolution.comparison_rules import (
    ComparisonRuleGenerator,
    generate_rules_from_ontology,
    generate_config_from_ontology,
)
from graphora_server.services.entity_resolution.embedding_similarity import (
    EmbeddingSimilarity,
    get_embedding_similarity,
)
from graphora_server.services.entity_resolution.blocking import (
    BlockingRuleGenerator,
    LSHBlocker,
    generate_blocking_rules_from_ontology,
)
from graphora_server.services.entity_resolution.entity_store import (
    EntityStore,
    CrossDocumentResolver,
)
from graphora_server.services.entity_resolution.splink_embedding_comparison import (
    EmbeddingAwareComparisonFactory,
    create_embedding_factory,
    _is_prop_type_text,
)
from graphora_server.services.entity_resolution.cross_document_service import (
    CrossDocumentResolutionService,
    create_cross_document_service,
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
    # Splink Embedding Comparison
    "EmbeddingAwareComparisonFactory",
    "create_embedding_factory",
    "_is_prop_type_text",
    # Cross-Document Service
    "CrossDocumentResolutionService",
    "create_cross_document_service",
]
