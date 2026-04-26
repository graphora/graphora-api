"""Relationship fixtures for testing.

These fixtures provide standard relationship instances that can be
used across multiple test files for consistency.
"""

from typing import Dict, Any
import pytest


@pytest.fixture
def sample_employs_relationship():
    """Standard EMPLOYS relationship for testing."""
    from graphora_server.services.transform.models import (
        RelationshipInstance,
        NodeProvenance,
    )

    return RelationshipInstance(
        id="rel-123",
        type="EMPLOYS",
        source_id="company-123",
        target_id="person-456",
        source_type="Company",
        target_type="Person",
        properties={
            "role": "CEO",
            "start_date": "2020-01-15",
        },
        provenance=NodeProvenance(
            chunk_ids=["chunk-1"],
            extraction_timestamp="2024-01-01T00:00:00Z",
            confidence_score=0.88,
        ),
    )


@pytest.fixture
def sample_relationship_instance():
    """Generic relationship instance for testing."""
    from graphora_server.services.transform.models import RelationshipInstance

    return RelationshipInstance(
        id="rel-789",
        type="RELATED_TO",
        source_id="node-a",
        target_id="node-b",
        source_type="Item",
        target_type="Item",
        properties={},
    )


@pytest.fixture
def relationship_factory():
    """Factory fixture for creating custom relationships."""
    from graphora_server.services.transform.models import (
        RelationshipInstance,
        NodeProvenance,
    )
    import uuid

    def create_relationship(
        rel_type: str = "RELATED_TO",
        source_id: str = None,
        target_id: str = None,
        source_type: str = "Item",
        target_type: str = "Item",
        properties: Dict[str, Any] = None,
        rel_id: str = None,
        confidence: float = 0.85,
    ) -> RelationshipInstance:
        rel_id = rel_id or str(uuid.uuid4())
        source_id = source_id or f"source-{uuid.uuid4()}"
        target_id = target_id or f"target-{uuid.uuid4()}"

        return RelationshipInstance(
            id=rel_id,
            type=rel_type,
            source_id=source_id,
            target_id=target_id,
            source_type=source_type,
            target_type=target_type,
            properties=properties or {},
            provenance=NodeProvenance(
                chunk_ids=[f"chunk-{rel_id}"],
                extraction_timestamp="2024-01-01T00:00:00Z",
                confidence_score=confidence,
            ),
        )

    return create_relationship


# Non-fixture versions for use in parametrized tests
def create_test_relationship(
    rel_id: str,
    rel_type: str,
    source_id: str,
    target_id: str,
    source_type: str = "Item",
    target_type: str = "Item",
    properties: Dict[str, Any] = None,
    confidence: float = 0.85,
):
    """Create a test relationship outside of pytest fixtures."""
    from graphora_server.services.transform.models import (
        RelationshipInstance,
        NodeProvenance,
    )

    return RelationshipInstance(
        id=rel_id,
        type=rel_type,
        source_id=source_id,
        target_id=target_id,
        source_type=source_type,
        target_type=target_type,
        properties=properties or {},
        provenance=NodeProvenance(
            chunk_ids=[f"chunk-{rel_id}"],
            extraction_timestamp="2024-01-01T00:00:00Z",
            confidence_score=confidence,
        ),
    )
