"""Relationship factory for creating test relationships."""

from typing import Any, Dict, List, Optional


class RelationshipFactory:
    """Factory for creating test relationship instances.

    Example:
        ```python
        factory = RelationshipFactory()

        # Create with defaults
        rel = factory.create()

        # Create specific type
        employs = factory.create_employs(
            source_id="company-1",
            target_id="person-1",
            role="CEO"
        )

        # Create batch between node sets
        rels = factory.create_batch(
            rel_type="KNOWS",
            source_ids=["p1", "p2", "p3"],
            target_ids=["p4", "p5", "p6"],
        )
        ```
    """

    _counter = 0

    @classmethod
    def _next_id(cls) -> str:
        cls._counter += 1
        return f"rel-{cls._counter}"

    @classmethod
    def reset_counter(cls):
        """Reset the ID counter."""
        cls._counter = 0

    @classmethod
    def create(
        cls,
        rel_id: Optional[str] = None,
        rel_type: str = "RELATED_TO",
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        source_type: str = "Item",
        target_type: str = "Item",
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 0.85,
        chunk_ids: Optional[List[str]] = None,
    ):
        """Create a RelationshipInstance.

        Args:
            rel_id: Unique relationship identifier.
            rel_type: Relationship type label.
            source_id: Source node ID.
            target_id: Target node ID.
            source_type: Source node entity type.
            target_type: Target node entity type.
            properties: Relationship properties.
            confidence: Extraction confidence score.
            chunk_ids: Source chunk IDs.

        Returns:
            RelationshipInstance.
        """
        from app.services.transform.models import RelationshipInstance, NodeProvenance

        rel_id = rel_id or cls._next_id()
        source_id = source_id or f"source-{cls._counter}"
        target_id = target_id or f"target-{cls._counter}"

        return RelationshipInstance(
            id=rel_id,
            type=rel_type,
            source_id=source_id,
            target_id=target_id,
            source_type=source_type,
            target_type=target_type,
            properties=properties or {},
            provenance=NodeProvenance(
                chunk_ids=chunk_ids or [f"chunk-{rel_id}"],
                extraction_timestamp="2024-01-01T00:00:00Z",
                confidence_score=confidence,
            ),
        )

    @classmethod
    def create_employs(
        cls,
        source_id: str,
        target_id: str,
        role: Optional[str] = None,
        start_date: Optional[str] = None,
        **kwargs,
    ):
        """Create an EMPLOYS relationship (Company -> Person)."""
        properties = {}
        if role:
            properties["role"] = role
        if start_date:
            properties["start_date"] = start_date

        return cls.create(
            rel_type="EMPLOYS",
            source_id=source_id,
            target_id=target_id,
            source_type="Company",
            target_type="Person",
            properties=properties,
            **kwargs,
        )

    @classmethod
    def create_works_for(
        cls,
        source_id: str,
        target_id: str,
        department: Optional[str] = None,
        **kwargs,
    ):
        """Create a WORKS_FOR relationship (Person -> Company)."""
        properties = {}
        if department:
            properties["department"] = department

        return cls.create(
            rel_type="WORKS_FOR",
            source_id=source_id,
            target_id=target_id,
            source_type="Person",
            target_type="Company",
            properties=properties,
            **kwargs,
        )

    @classmethod
    def create_batch(
        cls,
        source_ids: List[str],
        target_ids: List[str],
        rel_type: str = "RELATED_TO",
        source_type: str = "Item",
        target_type: str = "Item",
        **kwargs,
    ) -> List:
        """Create multiple relationships.

        Creates relationships pairing source_ids with target_ids.
        Lists must be the same length.

        Args:
            source_ids: List of source node IDs.
            target_ids: List of target node IDs.
            rel_type: Relationship type for all.
            source_type: Source node type for all.
            target_type: Target node type for all.
            **kwargs: Additional arguments passed to create().

        Returns:
            List of RelationshipInstance objects.
        """
        if len(source_ids) != len(target_ids):
            raise ValueError("source_ids and target_ids must have same length")

        return [
            cls.create(
                rel_type=rel_type,
                source_id=src,
                target_id=tgt,
                source_type=source_type,
                target_type=target_type,
                **kwargs,
            )
            for src, tgt in zip(source_ids, target_ids)
        ]

    @classmethod
    def create_chain(
        cls,
        node_ids: List[str],
        rel_type: str = "NEXT",
        node_type: str = "Item",
        **kwargs,
    ) -> List:
        """Create a chain of relationships between sequential nodes.

        Args:
            node_ids: List of node IDs to connect in sequence.
            rel_type: Relationship type for all.
            node_type: Node type for all.
            **kwargs: Additional arguments passed to create().

        Returns:
            List of RelationshipInstance objects forming a chain.
        """
        return [
            cls.create(
                rel_type=rel_type,
                source_id=node_ids[i],
                target_id=node_ids[i + 1],
                source_type=node_type,
                target_type=node_type,
                **kwargs,
            )
            for i in range(len(node_ids) - 1)
        ]
