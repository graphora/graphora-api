"""Node fixtures for testing.

These fixtures provide standard node instances that can be used
across multiple test files for consistency.
"""

from typing import Dict, Any
import pytest


@pytest.fixture
def sample_company_node():
    """Standard Company node for testing."""
    from app.services.transform.models import BaseNode, NodeProvenance

    return BaseNode(
        id="company-123",
        type="Company",
        properties={
            "name": "Acme Corporation",
            "ticker": "ACM",
            "industry": "Technology",
        },
        canonical_properties={
            "name": "acme corporation",
            "ticker": "acm",
            "industry": "technology",
        },
        canonical_key="Company:name=acme corporation",
        canonical_id="canonical-company-123",
        provenance=NodeProvenance(
            chunk_ids=["chunk-1"],
            extraction_timestamp="2024-01-01T00:00:00Z",
            confidence_score=0.95,
        ),
    )


@pytest.fixture
def sample_person_node():
    """Standard Person node for testing."""
    from app.services.transform.models import BaseNode, NodeProvenance

    return BaseNode(
        id="person-456",
        type="Person",
        properties={
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "title": "CEO",
        },
        canonical_properties={
            "name": "jane smith",
            "email": "jane.smith@example.com",
            "title": "ceo",
        },
        canonical_key="Person:email=jane.smith@example.com",
        canonical_id="canonical-person-456",
        provenance=NodeProvenance(
            chunk_ids=["chunk-1"],
            extraction_timestamp="2024-01-01T00:00:00Z",
            confidence_score=0.92,
        ),
    )


@pytest.fixture
def sample_base_node():
    """Minimal BaseNode for generic testing."""
    from app.services.transform.models import BaseNode

    return BaseNode(
        id="node-789",
        type="Item",
        properties={"name": "Test Item"},
    )


@pytest.fixture
def node_factory():
    """Factory fixture for creating custom nodes."""
    from app.services.transform.models import BaseNode, NodeProvenance
    import uuid

    def create_node(
        node_type: str = "Company",
        properties: Dict[str, Any] = None,
        node_id: str = None,
        confidence: float = 0.9,
    ) -> BaseNode:
        node_id = node_id or str(uuid.uuid4())
        props = properties or {"name": f"Test {node_type}"}

        # Build canonical properties
        canonical_props = {}
        for key, value in props.items():
            if isinstance(value, str):
                canonical_props[key] = value.lower().strip()
            else:
                canonical_props[key] = str(value)

        # Build canonical key from unique/first property
        first_key = list(props.keys())[0]
        canonical_key = f"{node_type}:{first_key}={canonical_props[first_key]}"

        return BaseNode(
            id=node_id,
            type=node_type,
            properties=props,
            canonical_properties=canonical_props,
            canonical_key=canonical_key,
            canonical_id=f"canonical-{node_id}",
            provenance=NodeProvenance(
                chunk_ids=[f"chunk-{node_id}"],
                extraction_timestamp="2024-01-01T00:00:00Z",
                confidence_score=confidence,
            ),
        )

    return create_node


# Non-fixture versions for use in parametrized tests
def create_test_node(
    node_id: str,
    node_type: str,
    properties: Dict[str, Any],
    confidence: float = 0.9,
):
    """Create a test node outside of pytest fixtures."""
    from app.services.transform.models import BaseNode, NodeProvenance

    canonical_props = {}
    for key, value in properties.items():
        if isinstance(value, str):
            canonical_props[key] = value.lower().strip()
        else:
            canonical_props[key] = str(value)

    first_key = list(properties.keys())[0]
    canonical_key = f"{node_type}:{first_key}={canonical_props[first_key]}"

    return BaseNode(
        id=node_id,
        type=node_type,
        properties=properties,
        canonical_properties=canonical_props,
        canonical_key=canonical_key,
        canonical_id=f"canonical-{node_id}",
        provenance=NodeProvenance(
            chunk_ids=[f"chunk-{node_id}"],
            extraction_timestamp="2024-01-01T00:00:00Z",
            confidence_score=confidence,
        ),
    )
