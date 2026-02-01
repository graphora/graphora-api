"""Ontology fixtures for testing.

These fixtures provide standard ontology definitions that can be
used across multiple test files for consistency.
"""

from typing import Dict, Any
import pytest


@pytest.fixture
def simple_ontology() -> Dict[str, Any]:
    """Minimal ontology with a single entity type."""
    return {
        "entities": {
            "Item": {
                "properties": {
                    "name": {"type": "string", "required": True},
                },
            },
        },
    }


@pytest.fixture
def company_person_ontology() -> Dict[str, Any]:
    """Standard Company-Person ontology for most tests."""
    return {
        "entities": {
            "Company": {
                "properties": {
                    "name": {
                        "type": "string",
                        "required": True,
                        "unique": True,
                        "canonicalization": {
                            "strip_company_suffixes": True,
                            "strip_punctuation": True,
                        },
                    },
                    "ticker": {
                        "type": "string",
                        "required": False,
                        "index": True,
                    },
                    "industry": {
                        "type": "string",
                        "required": False,
                    },
                    "founded": {
                        "type": "integer",
                        "required": False,
                    },
                },
                "relationships": {
                    "EMPLOYS": {
                        "target": "Person",
                        "properties": {
                            "role": {"type": "string"},
                            "start_date": {"type": "string"},
                            "salary": {"type": "number"},
                        },
                    },
                    "COMPETES_WITH": {
                        "target": "Company",
                        "properties": {
                            "market": {"type": "string"},
                        },
                    },
                },
            },
            "Person": {
                "properties": {
                    "name": {
                        "type": "string",
                        "required": True,
                    },
                    "email": {
                        "type": "string",
                        "unique": True,
                    },
                    "age": {
                        "type": "integer",
                    },
                    "title": {
                        "type": "string",
                    },
                },
                "relationships": {
                    "WORKS_FOR": {
                        "target": "Company",
                        "properties": {
                            "department": {"type": "string"},
                        },
                    },
                    "KNOWS": {
                        "target": "Person",
                        "properties": {
                            "since": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


@pytest.fixture
def complex_ontology() -> Dict[str, Any]:
    """Complex ontology with multiple entity types and relationships."""
    return {
        "entities": {
            "Company": {
                "properties": {
                    "name": {
                        "type": "string",
                        "required": True,
                        "unique": True,
                        "canonicalization": {
                            "strip_company_suffixes": True,
                            "strip_punctuation": True,
                        },
                    },
                    "ticker": {"type": "string", "index": True},
                    "industry": {"type": "string"},
                    "revenue": {"type": "number"},
                    "employees": {"type": "integer"},
                    "founded": {"type": "integer"},
                    "headquarters": {"type": "string"},
                    "website": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["active", "inactive", "acquired"],
                    },
                },
                "relationships": {
                    "EMPLOYS": {"target": "Person", "properties": {}},
                    "OWNS": {"target": "Product", "properties": {}},
                    "LOCATED_IN": {"target": "Location", "properties": {}},
                    "COMPETES_WITH": {"target": "Company", "properties": {}},
                    "ACQUIRED": {
                        "target": "Company",
                        "properties": {
                            "date": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                    },
                },
            },
            "Person": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string", "unique": True},
                    "phone": {"type": "string"},
                    "title": {"type": "string"},
                    "department": {"type": "string"},
                    "linkedin": {"type": "string"},
                },
                "relationships": {
                    "WORKS_FOR": {"target": "Company", "properties": {}},
                    "MANAGES": {"target": "Person", "properties": {}},
                    "KNOWS": {"target": "Person", "properties": {}},
                },
            },
            "Product": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "sku": {"type": "string", "unique": True},
                    "price": {"type": "number"},
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                },
                "relationships": {
                    "MANUFACTURED_BY": {"target": "Company", "properties": {}},
                    "RELATED_TO": {"target": "Product", "properties": {}},
                },
            },
            "Location": {
                "properties": {
                    "name": {"type": "string", "required": True},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "country": {"type": "string"},
                    "postal_code": {"type": "string"},
                },
                "relationships": {
                    "PART_OF": {"target": "Location", "properties": {}},
                },
            },
        },
    }


# Non-fixture versions for use in parametrized tests
SIMPLE_ONTOLOGY = {
    "entities": {
        "Item": {
            "properties": {
                "name": {"type": "string", "required": True},
            },
        },
    },
}

COMPANY_PERSON_ONTOLOGY = {
    "entities": {
        "Company": {
            "properties": {
                "name": {
                    "type": "string",
                    "required": True,
                    "unique": True,
                },
                "ticker": {
                    "type": "string",
                    "index": True,
                },
            },
            "relationships": {
                "EMPLOYS": {
                    "target": "Person",
                    "properties": {"role": {"type": "string"}},
                },
            },
        },
        "Person": {
            "properties": {
                "name": {"type": "string", "required": True},
                "email": {"type": "string", "unique": True},
            },
        },
    },
}
