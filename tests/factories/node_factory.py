"""Node factory for creating test nodes."""

from typing import Any, Dict, List, Optional


class NodeFactory:
    """Factory for creating test node instances.

    Provides methods to create nodes with sensible defaults
    while allowing full customization when needed.

    Example:
        ```python
        factory = NodeFactory()

        # Create with defaults
        node = factory.create()

        # Create specific type
        company = factory.create_company(name="Acme", ticker="ACM")

        # Create batch
        nodes = factory.create_batch(count=10, node_type="Person")
        ```
    """

    _counter = 0

    @classmethod
    def _next_id(cls) -> str:
        cls._counter += 1
        return f"node-{cls._counter}"

    @classmethod
    def reset_counter(cls):
        """Reset the ID counter."""
        cls._counter = 0

    @classmethod
    def create(
        cls,
        node_id: Optional[str] = None,
        node_type: str = "Item",
        properties: Optional[Dict[str, Any]] = None,
        canonical_properties: Optional[Dict[str, Any]] = None,
        canonical_key: Optional[str] = None,
        canonical_id: Optional[str] = None,
        confidence: float = 0.9,
        chunk_ids: Optional[List[str]] = None,
    ):
        """Create a BaseNode instance.

        Args:
            node_id: Unique node identifier. Auto-generated if not provided.
            node_type: Entity type label.
            properties: Node properties.
            canonical_properties: Canonicalized properties for matching.
            canonical_key: Unique key for deduplication.
            canonical_id: Transform-agnostic identifier.
            confidence: Extraction confidence score.
            chunk_ids: Source chunk IDs.

        Returns:
            BaseNode instance.
        """
        from app.services.transform.models import BaseNode, NodeProvenance

        node_id = node_id or cls._next_id()
        props = properties or {"name": f"Test {node_type} {node_id}"}

        # Build canonical properties if not provided
        if canonical_properties is None:
            canonical_properties = {}
            for key, value in props.items():
                if isinstance(value, str):
                    canonical_properties[key] = value.lower().strip()
                else:
                    canonical_properties[key] = (
                        str(value) if value is not None else None
                    )

        # Build canonical key if not provided
        if canonical_key is None:
            first_key = list(props.keys())[0] if props else "id"
            first_value = canonical_properties.get(first_key, node_id)
            canonical_key = f"{node_type}:{first_key}={first_value}"

        # Build canonical ID if not provided
        if canonical_id is None:
            canonical_id = f"canonical-{node_id}"

        return BaseNode(
            id=node_id,
            type=node_type,
            properties=props,
            canonical_properties=canonical_properties,
            canonical_key=canonical_key,
            canonical_id=canonical_id,
            provenance=NodeProvenance(
                chunk_ids=chunk_ids or [f"chunk-{node_id}"],
                extraction_timestamp="2024-01-01T00:00:00Z",
                confidence_score=confidence,
            ),
            confidence_score=confidence,
        )

    @classmethod
    def create_company(
        cls,
        name: str = "Acme Corp",
        ticker: Optional[str] = None,
        industry: Optional[str] = None,
        node_id: Optional[str] = None,
        **kwargs,
    ):
        """Create a Company node."""
        properties = {"name": name}
        if ticker:
            properties["ticker"] = ticker
        if industry:
            properties["industry"] = industry

        return cls.create(
            node_id=node_id,
            node_type="Company",
            properties=properties,
            **kwargs,
        )

    @classmethod
    def create_person(
        cls,
        name: str = "John Doe",
        email: Optional[str] = None,
        title: Optional[str] = None,
        node_id: Optional[str] = None,
        **kwargs,
    ):
        """Create a Person node."""
        properties = {"name": name}
        if email:
            properties["email"] = email
        if title:
            properties["title"] = title

        return cls.create(
            node_id=node_id,
            node_type="Person",
            properties=properties,
            **kwargs,
        )

    @classmethod
    def create_batch(
        cls,
        count: int,
        node_type: str = "Item",
        properties_template: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List:
        """Create multiple nodes.

        Args:
            count: Number of nodes to create.
            node_type: Entity type for all nodes.
            properties_template: Template with {i} placeholders for index.
            **kwargs: Additional arguments passed to create().

        Returns:
            List of BaseNode instances.
        """
        nodes = []
        template = properties_template or {"name": f"{node_type} {{i}}"}

        for i in range(count):
            props = {
                k: v.format(i=i) if isinstance(v, str) else v
                for k, v in template.items()
            }
            nodes.append(cls.create(node_type=node_type, properties=props, **kwargs))

        return nodes
