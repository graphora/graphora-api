"""Ontology factory for creating test ontologies."""

from typing import Any, Dict, List, Optional


class OntologyFactory:
    """Factory for creating test ontology definitions.

    Example:
        ```python
        factory = OntologyFactory()

        # Create minimal ontology
        ontology = factory.create_minimal("Product")

        # Create with relationships
        ontology = factory.with_entities(
            factory.entity("Company", properties=["name", "ticker"]),
            factory.entity("Person", properties=["name", "email"]),
        ).with_relationship(
            "Company", "EMPLOYS", "Person"
        ).build()
        ```
    """

    @classmethod
    def create_minimal(cls, entity_type: str = "Item") -> Dict[str, Any]:
        """Create a minimal ontology with one entity type.

        Args:
            entity_type: Name of the single entity type.

        Returns:
            Ontology dictionary.
        """
        return {
            "entities": {
                entity_type: {
                    "properties": {
                        "name": {"type": "string", "required": True},
                    },
                },
            },
        }

    @classmethod
    def create_company_person(cls) -> Dict[str, Any]:
        """Create a standard Company-Person ontology."""
        return {
            "entities": {
                "Company": {
                    "properties": {
                        "name": {"type": "string", "required": True, "unique": True},
                        "ticker": {"type": "string", "index": True},
                        "industry": {"type": "string"},
                    },
                    "relationships": {
                        "EMPLOYS": {
                            "target": "Person",
                            "properties": {
                                "role": {"type": "string"},
                                "start_date": {"type": "string"},
                            },
                        },
                    },
                },
                "Person": {
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "email": {"type": "string", "unique": True},
                        "title": {"type": "string"},
                    },
                    "relationships": {
                        "WORKS_FOR": {
                            "target": "Company",
                            "properties": {},
                        },
                    },
                },
            },
        }

    @classmethod
    def entity(
        cls,
        name: str,
        properties: Optional[List[str]] = None,
        unique_properties: Optional[List[str]] = None,
        indexed_properties: Optional[List[str]] = None,
        required_properties: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create an entity definition.

        Args:
            name: Entity type name.
            properties: List of property names (all string type).
            unique_properties: Properties that should be unique.
            indexed_properties: Properties that should be indexed.
            required_properties: Properties that should be required.

        Returns:
            Entity definition dictionary.
        """
        properties = properties or ["name"]
        unique_properties = unique_properties or []
        indexed_properties = indexed_properties or []
        required_properties = required_properties or ["name"]

        props = {}
        for prop in properties:
            props[prop] = {
                "type": "string",
                "required": prop in required_properties,
                "unique": prop in unique_properties,
                "index": prop in indexed_properties,
            }

        return {
            "name": name,
            "properties": props,
            "relationships": {},
        }

    def __init__(self):
        self._entities: Dict[str, Dict[str, Any]] = {}

    def with_entities(self, *entities: Dict[str, Any]) -> "OntologyFactory":
        """Add entity definitions.

        Args:
            *entities: Entity definitions from entity() method.

        Returns:
            Self for chaining.
        """
        for entity in entities:
            name = entity.pop("name")
            self._entities[name] = entity
        return self

    def with_relationship(
        self,
        source_type: str,
        rel_type: str,
        target_type: str,
        properties: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "OntologyFactory":
        """Add a relationship to an entity.

        Args:
            source_type: Source entity type.
            rel_type: Relationship type name.
            target_type: Target entity type.
            properties: Relationship property definitions.

        Returns:
            Self for chaining.
        """
        if source_type not in self._entities:
            raise ValueError(f"Entity {source_type} not found")

        if "relationships" not in self._entities[source_type]:
            self._entities[source_type]["relationships"] = {}

        self._entities[source_type]["relationships"][rel_type] = {
            "target": target_type,
            "properties": properties or {},
        }

        return self

    def build(self) -> Dict[str, Any]:
        """Build the final ontology dictionary.

        Returns:
            Complete ontology dictionary.
        """
        return {"entities": self._entities}
