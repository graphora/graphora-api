"""Test fixtures for Graphora API tests."""

from .ontologies import (
    simple_ontology,
    company_person_ontology,
    complex_ontology,
)
from .nodes import (
    sample_company_node,
    sample_person_node,
    sample_base_node,
)
from .relationships import (
    sample_employs_relationship,
    sample_relationship_instance,
)

__all__ = [
    # Ontologies
    "simple_ontology",
    "company_person_ontology",
    "complex_ontology",
    # Nodes
    "sample_company_node",
    "sample_person_node",
    "sample_base_node",
    # Relationships
    "sample_employs_relationship",
    "sample_relationship_instance",
]
