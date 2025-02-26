"""
Common test fixtures and utilities for all tests.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
import os
from typing import Dict, List, Any, Optional

from app.services.storage.models import (
    Node,
    Edge,
    StorageStage,
    StorageBatchResult,
    StorageCheckpoint,
    TransformationResult
)
from app.services.transform.models import BaseNode, RelationshipInstance

# Test fixture for sample nodes
@pytest.fixture
def sample_nodes():
    return [
        BaseNode(
            id="node1",
            type="Person",
            properties={
                "name": "Alice",
                "age": 30
            }
        ),
        BaseNode(
            id="node2",
            type="Person",
            properties={
                "name": "Bob",
                "age": 25
            }
        ),
        BaseNode(
            id="node3",
            type="Company",
            properties={
                "name": "Acme Inc",
                "founded": 2010
            }
        )
    ]

# Test fixture for sample relationships
@pytest.fixture
def sample_relationships():
    return [
        RelationshipInstance(
            id="rel1",
            source_id="node1",
            target_id="node2",
            type="KNOWS",
            properties={
                "since": 2020
            }
        ),
        RelationshipInstance(
            id="rel2",
            source_id="node1",
            target_id="node3",
            type="WORKS_AT",
            properties={
                "position": "Engineer",
                "since": 2019
            }
        ),
        RelationshipInstance(
            id="rel3",
            source_id="node2",
            target_id="node3",
            type="WORKS_AT",
            properties={
                "position": "Manager",
                "since": 2018
            }
        )
    ]

# Test fixture for sample graph nodes
@pytest.fixture
def sample_graph_nodes():
    return [
        Node(
            id="node1",
            label="Person",
            type="Person",
            properties={
                "name": "Alice",
                "age": 30,
                "transform_id": "test_transform"
            }
        ),
        Node(
            id="node2",
            label="Person",
            type="Person",
            properties={
                "name": "Bob",
                "age": 25,
                "transform_id": "test_transform"
            }
        ),
        Node(
            id="node3",
            label="Company",
            type="Company",
            properties={
                "name": "Acme Inc",
                "founded": 2010,
                "transform_id": "test_transform"
            }
        )
    ]

# Test fixture for sample graph edges
@pytest.fixture
def sample_graph_edges():
    return [
        Edge(
            id="rel1",
            source="node1",
            target="node2",
            type="KNOWS",
            properties={
                "since": 2020,
                "transform_id": "test_transform"
            }
        ),
        Edge(
            id="rel2",
            source="node1",
            target="node3",
            type="WORKS_AT",
            properties={
                "position": "Engineer",
                "since": 2019,
                "transform_id": "test_transform"
            }
        ),
        Edge(
            id="rel3",
            source="node2",
            target="node3",
            type="WORKS_AT",
            properties={
                "position": "Manager",
                "since": 2018,
                "transform_id": "test_transform"
            }
        )
    ]

# Test fixture for mock Neo4j driver
@pytest.fixture
def mock_neo4j_driver():
    driver = MagicMock()
    session = MagicMock()
    result = MagicMock()
    
    # Configure mocks
    driver.session.return_value.__enter__.return_value = session
    session.run.return_value = result
    
    return driver

# Test fixture for mock Neo4j session
@pytest.fixture
def mock_neo4j_session():
    session = MagicMock()
    result = MagicMock()
    session.run.return_value = result
    return session

# Test fixture for mock Neo4j record
@pytest.fixture
def mock_neo4j_node_record():
    record = MagicMock()
    node = MagicMock()
    node.id = "node1"
    node.labels = ["Person"]
    node.items.return_value = [
        ("name", "Alice"), 
        ("age", 30),
        ("transform_id", "test_transform")
    ]
    record.get.return_value = node
    return record

# Test fixture for mock Neo4j relationship record
@pytest.fixture
def mock_neo4j_rel_record():
    record = MagicMock()
    rel = MagicMock()
    rel.id = "rel1"
    rel.type = "KNOWS"
    rel.start_node.id = "node1"
    rel.end_node.id = "node2"
    rel.items.return_value = [
        ("since", 2020),
        ("transform_id", "test_transform")
    ]
    record.get.return_value = rel
    return record
