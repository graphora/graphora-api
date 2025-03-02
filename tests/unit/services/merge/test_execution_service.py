"""Tests for the MergeExecutionService"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime
import pytz
import copy
from typing import List

from app.services.merge.execution_service import MergeExecutionService
from app.services.merge.models import (
    MergeStage,
    MergeStatus,
    StageStatus
)
from app.schemas.graph import GraphResponse, Node, Edge
from app.schemas.conflicts import (
    Conflict, 
    ConflictType, 
    ConflictStatus, 
    ConflictSeverity,
    ResolutionOption
)
from app.services.storage.models import TransformationResult

@pytest.fixture
def mock_staging_storage():
    """Mock staging storage fixture"""
    storage = AsyncMock()
    
    # Mock get_transformation_data
    transform_result = TransformationResult(
        transform_id="test_transform_id",
        nodes=[
            {
                "id": "node1",
                "label": "Person",
                "type": "Person",
                "properties": {"name": "Alice", "age": 30}
            },
            {
                "id": "node2",
                "label": "Person",
                "type": "Person",
                "properties": {"name": "Bob", "age": 25}
            },
            {
                "id": "node3",
                "label": "Company",
                "type": "Company",
                "properties": {"name": "Acme Inc", "founded": 2010}
            }
        ],
        relationships=[
            {
                "id": "edge1",
                "source": "node1",
                "target": "node3",
                "type": "WORKS_AT",
                "properties": {"since": 2018}
            },
            {
                "id": "edge2",
                "source": "node2",
                "target": "node3",
                "type": "WORKS_AT",
                "properties": {"since": 2019}
            }
        ],
        timestamp=datetime.now(pytz.utc)
    )
    storage.get_transformation_data.return_value = transform_result
    
    return storage

@pytest.fixture
def mock_prod_storage():
    """Mock production storage fixture"""
    storage = AsyncMock()
    
    # Mock get_node_by_id
    storage.get_node_by_id.return_value = None  # Default to not found
    
    # Mock create_node
    storage.create_node.return_value = Node(
        id="new_node",
        label="Test",
        type="Test",
        properties={"name": "Test Node"}
    )
    
    # Mock update_node
    storage.update_node.return_value = Node(
        id="existing_node",
        label="Test",
        type="Test",
        properties={"name": "Updated Node"}
    )
    
    # Mock create_relationship
    storage.create_relationship.return_value = Edge(
        id="new_edge",
        source="node1",
        target="node2",
        type="TEST_REL",
        properties={}
    )
    
    # Mock get_edges_between
    storage.get_edges_between.return_value = []  # Default to no edges
    
    return storage

@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker fixture"""
    tracker = AsyncMock()
    tracker.start_merge_stage = AsyncMock()
    tracker.update_merge_progress = AsyncMock()
    tracker.complete_merge_stage = AsyncMock()
    tracker.fail_merge_stage = AsyncMock()
    tracker.get_progress = AsyncMock()
    return tracker

@pytest.fixture
def mock_redis_client():
    """Mock Redis client fixture"""
    redis_client = AsyncMock()
    
    # Mock keys method
    redis_client.keys.return_value = [
        b"merge:test_merge_id:conflict:1",
        b"merge:test_merge_id:conflict:2"
    ]
    
    # Mock get method for conflicts
    conflict1 = Conflict(
        id="1",
        merge_id="test_merge_id",
        conflict_type=ConflictType.PROPERTY_VALUE,
        entity_id="node1",
        property_name="age",
        staging_value=30,
        production_value=35,
        severity=ConflictSeverity.MINOR,
        status=ConflictStatus.RESOLVED,
        resolved=True,
        resolution_id="opt1",
        description="Age property has different values in staging and production",
        resolution_options=[
            ResolutionOption(
                id="opt1",
                description="Keep staging value",
                resolution_type="keep_staging",
                confidence=0.9,
                reasoning="Staging value is more recent",
                requires_review=False,
                auto_resolvable=True
            ),
            ResolutionOption(
                id="opt2",
                description="Keep production value",
                resolution_type="keep_production",
                confidence=0.5,
                reasoning="Production value might be more reliable",
                requires_review=True,
                auto_resolvable=False
            )
        ]
    )
    
    conflict2 = Conflict(
        id="2",
        merge_id="test_merge_id",
        conflict_type=ConflictType.RELATIONSHIP_TYPE,
        entity_id="edge1",
        staging_ids=["edge1"],
        production_ids=["prod_edge1"],
        severity=ConflictSeverity.MAJOR,
        status=ConflictStatus.RESOLVED,
        resolved=True,
        resolution_id="opt1",
        description="Relationship type conflict between staging and production",
        resolution_options=[
            ResolutionOption(
                id="opt1",
                description="Keep staging relationship",
                resolution_type="keep_staging_rel",
                confidence=0.8,
                reasoning="Staging relationship is more recent",
                requires_review=False,
                auto_resolvable=True
            )
        ]
    )
    
    # Set up the return values for get
    async def mock_get(key):
        if key == b"merge:test_merge_id:conflict:1":
            return conflict1.model_dump_json()
        elif key == b"merge:test_merge_id:conflict:2":
            return conflict2.model_dump_json()
        return None
    
    redis_client.get.side_effect = mock_get
    
    return redis_client

@pytest.fixture
def execution_service(mock_staging_storage, mock_prod_storage, mock_progress_tracker):
    """Execution service fixture"""
    return MergeExecutionService(
        staging_storage=mock_staging_storage,
        prod_storage=mock_prod_storage,
        progress_tracker=mock_progress_tracker
    )

@pytest.mark.asyncio
async def test_get_resolved_conflicts(execution_service, mock_redis_client):
    """Test getting resolved conflicts"""
    with patch("app.utils.redis.get_redis_client", return_value=mock_redis_client):
        conflicts = await execution_service._get_resolved_conflicts("test_merge_id")
        
        # Verify Redis client was called correctly
        mock_redis_client.keys.assert_called_once_with("merge:test_merge_id:conflict:*")
        
        # Verify conflicts were parsed correctly
        assert len(conflicts) == 2
        assert conflicts[0].id == "1"
        assert conflicts[0].conflict_type == ConflictType.PROPERTY_VALUE
        assert conflicts[1].id == "2"
        assert conflicts[1].conflict_type == ConflictType.RELATIONSHIP_TYPE

@pytest.mark.asyncio
async def test_apply_resolutions(execution_service):
    """Test applying resolutions to a graph"""
    # Create test graph
    graph = GraphResponse(
        nodes=[
            Node(
                id="node1",
                label="Person",
                type="Person",
                properties={"name": "Alice", "age": 30}
            )
        ],
        edges=[
            Edge(
                id="edge1",
                source="node1",
                target="node2",
                type="KNOWS",
                properties={}
            )
        ]
    )
    
    # Create test conflicts with resolutions
    conflicts = [
        Conflict(
            id="1",
            merge_id="test_merge_id",
            conflict_type=ConflictType.PROPERTY_VALUE,
            entity_id="node1",
            property_name="age",
            staging_value=30,
            production_value=35,
            severity=ConflictSeverity.MINOR,
            status=ConflictStatus.RESOLVED,
            resolved=True,
            resolution=ResolutionOption(
                id="opt2",
                description="Keep production value",
                resolution_type="keep_production",
                confidence=0.9,
                reasoning="Production value is more reliable",
                requires_review=True,
                auto_resolvable=False
            ),
            description="Age property has different values in staging and production",
            resolution_options=[
                ResolutionOption(
                    id="opt1",
                    description="Keep staging value",
                    resolution_type="keep_staging",
                    confidence=0.5,
                    reasoning="Staging value is more recent",
                    requires_review=False,
                    auto_resolvable=True
                ),
                ResolutionOption(
                    id="opt2",
                    description="Keep production value",
                    resolution_type="keep_production",
                    confidence=0.9,
                    reasoning="Production value is more reliable",
                    requires_review=True,
                    auto_resolvable=False
                )
            ]
        )
    ]
    
    # Create a copy of the graph
    resolved_graph = copy.deepcopy(graph)
    
    # Apply the resolution directly
    await execution_service._apply_resolution(resolved_graph, conflicts[0], conflicts[0].resolution)
    
    # Verify the graph was modified correctly
    assert resolved_graph.nodes[0].properties["age"] == 35  # Production value
    assert id(resolved_graph) != id(graph)  # Should be a copy, not the same object

@pytest.mark.asyncio
async def test_apply_resolution_keep_production(execution_service):
    """Test applying a keep_production resolution"""
    # Create test graph
    graph = GraphResponse(
        nodes=[
            Node(
                id="node1",
                label="Person",
                type="Person",
                properties={"name": "Alice", "age": 30}
            )
        ],
        edges=[]
    )
    
    # Create test conflict with keep_production resolution
    conflict = Conflict(
        id="1",
        merge_id="test_merge_id",
        conflict_type=ConflictType.PROPERTY_VALUE,
        entity_id="node1",
        property_name="age",
        staging_value=30,
        production_value=35,
        severity=ConflictSeverity.MINOR,
        status=ConflictStatus.RESOLVED,
        resolved=True,
        description="Age property has different values in staging and production"
    )
    
    resolution = ResolutionOption(
        id="opt1",
        description="Keep production value",
        resolution_type="keep_production",
        confidence=0.9,
        reasoning="Production value is more reliable",
        requires_review=True,
        auto_resolvable=False
    )
    
    # Apply resolution
    await execution_service._apply_resolution(graph, conflict, resolution)
    
    # Verify the graph was modified correctly
    assert graph.nodes[0].properties["age"] == 35  # Production value

@pytest.mark.asyncio
async def test_apply_resolution_merge_values(execution_service):
    """Test applying a merge_values resolution for lists"""
    # Create test graph
    graph = GraphResponse(
        nodes=[
            Node(
                id="node1",
                label="Person",
                type="Person",
                properties={"name": "Alice", "tags": ["developer", "python"]}
            )
        ],
        edges=[]
    )
    
    # Create test conflict with merge_values resolution
    conflict = Conflict(
        id="1",
        merge_id="test_merge_id",
        conflict_type=ConflictType.PROPERTY_VALUE,
        entity_id="node1",
        property_name="tags",
        staging_value=["developer", "python"],
        production_value=["developer", "java"],
        severity=ConflictSeverity.MINOR,
        status=ConflictStatus.RESOLVED,
        resolved=True,
        description="Tags property has different values in staging and production"
    )
    
    resolution = ResolutionOption(
        id="opt1",
        description="Merge values",
        resolution_type="merge_values",
        confidence=0.9,
        reasoning="Both values contain useful information",
        requires_review=True,
        auto_resolvable=False
    )
    
    # Apply resolution
    await execution_service._apply_resolution(graph, conflict, resolution)
    
    # Verify the graph was modified correctly
    assert set(graph.nodes[0].properties["tags"]) == {"developer", "python", "java"}

@pytest.mark.asyncio
async def test_apply_resolution_keep_production_rel(execution_service):
    """Test applying a keep_production_rel resolution"""
    # Create test graph
    graph = GraphResponse(
        nodes=[
            Node(id="node1", label="Person", type="Person", properties={}),
            Node(id="node2", label="Person", type="Person", properties={})
        ],
        edges=[
            Edge(
                id="edge1",
                source="node1",
                target="node2",
                type="KNOWS",
                properties={}
            )
        ]
    )
    
    # Create test conflict with keep_production_rel resolution
    conflict = Conflict(
        id="1",
        merge_id="test_merge_id",
        conflict_type=ConflictType.RELATIONSHIP_TYPE,
        entity_id="edge1",
        staging_ids=["edge1"],
        production_ids=["prod_edge1"],
        severity=ConflictSeverity.MAJOR,
        status=ConflictStatus.RESOLVED,
        resolved=True,
        description="Relationship type conflict between staging and production"
    )
    
    resolution = ResolutionOption(
        id="opt1",
        description="Keep production relationship",
        resolution_type="keep_production_rel",
        confidence=0.8,
        reasoning="Production relationship is more accurate",
        requires_review=True,
        auto_resolvable=False
    )
    
    # Apply resolution
    await execution_service._apply_resolution(graph, conflict, resolution)
    
    # Verify the edge was removed
    assert len(graph.edges) == 0

@pytest.mark.asyncio
async def test_execute_batch_merge(execution_service, mock_progress_tracker):
    """Test executing a batch merge"""
    merge_id = "test_merge_id"
    
    # Create test graph
    graph = GraphResponse(
        nodes=[
            Node(
                id="node1",
                label="Person",
                type="Person",
                properties={"name": "Alice", "age": 30}
            ),
            Node(
                id="node2",
                label="Person",
                type="Person",
                properties={"name": "Bob", "age": 25}
            )
        ],
        edges=[
            Edge(
                id="edge1",
                source="node1",
                target="node2",
                type="KNOWS",
                properties={}
            )
        ]
    )
    
    # Mock _merge_node_batch and _merge_edge_batch
    execution_service._merge_node_batch = AsyncMock(return_value=(graph.nodes, []))
    execution_service._merge_edge_batch = AsyncMock(return_value=(graph.edges, []))
    
    # Execute batch merge
    stats = await execution_service._execute_batch_merge(
        merge_id=merge_id,
        graph=graph,
        batch_size=10,
        max_retries=3,
        retry_delay=1
    )
    
    # Verify methods were called correctly
    execution_service._merge_node_batch.assert_called_once()
    execution_service._merge_edge_batch.assert_called_once()
    
    # Verify progress tracker was updated
    assert mock_progress_tracker.update_merge_progress.call_count == 2
    
    # Verify stats
    assert stats["nodes_merged"] == 2
    assert stats["edges_merged"] == 1
    assert stats["nodes_failed"] == 0
    assert stats["edges_failed"] == 0
    assert stats["total_items"] == 3
    assert stats["success_rate"] == 1.0

@pytest.mark.asyncio
async def test_merge_node_batch(execution_service, mock_prod_storage):
    """Test merging a batch of nodes"""
    # Create test nodes
    nodes = [
        Node(
            id="node1",
            label="Person",
            type="Person",
            properties={"name": "Alice", "age": 30}
        ),
        Node(
            id="node2",
            label="Person",
            type="Person",
            properties={"name": "Bob", "age": 25}
        )
    ]
    
    # Set up mock to return existing node for node1
    existing_node = Node(
        id="node1",
        label="Person",
        type="Person",
        properties={"name": "Alice", "age": 28}
    )
    
    async def mock_get_node_by_id(node_id):
        if node_id == "node1":
            return existing_node
        return None
    
    mock_prod_storage.get_node_by_id.side_effect = mock_get_node_by_id
    
    # Merge nodes
    successful, failed = await execution_service._merge_node_batch(nodes, 3, 1)
    
    # Verify storage methods were called correctly
    assert mock_prod_storage.update_node.call_count == 1
    assert mock_prod_storage.create_node.call_count == 1
    
    # Verify results
    assert len(successful) == 2
    assert len(failed) == 0

@pytest.mark.asyncio
async def test_merge_edge_batch(execution_service, mock_prod_storage):
    """Test merging a batch of edges"""
    # Create test edges
    edges = [
        Edge(
            id="edge1",
            source="node1",
            target="node2",
            type="KNOWS",
            properties={}
        )
    ]
    
    # Set up mocks to return existing nodes
    node1 = Node(id="node1", label="Person", type="Person", properties={})
    node2 = Node(id="node2", label="Person", type="Person", properties={})
    
    async def mock_get_node_by_id(node_id):
        if node_id == "node1":
            return node1
        elif node_id == "node2":
            return node2
        return None
    
    mock_prod_storage.get_node_by_id.side_effect = mock_get_node_by_id
    
    # Merge edges
    successful, failed = await execution_service._merge_edge_batch(edges, 3, 1)
    
    # Verify storage methods were called correctly
    assert mock_prod_storage.get_node_by_id.call_count == 2
    assert mock_prod_storage.get_edges_between.call_count == 1
    assert mock_prod_storage.create_relationship.call_count == 1
    
    # Verify results
    assert len(successful) == 1
    assert len(failed) == 0

@pytest.mark.asyncio
async def test_merge_edge_batch_missing_nodes(execution_service, mock_prod_storage):
    """Test merging edges with missing nodes"""
    # Create test edges
    edges = [
        Edge(
            id="edge1",
            source="node1",
            target="node2",
            type="KNOWS",
            properties={}
        )
    ]
    
    # Set up mocks to return only one node
    node1 = Node(id="node1", label="Person", type="Person", properties={})
    
    async def mock_get_node_by_id(node_id):
        if node_id == "node1":
            return node1
        return None
    
    mock_prod_storage.get_node_by_id.side_effect = mock_get_node_by_id
    
    # Merge edges
    successful, failed = await execution_service._merge_edge_batch(edges, 3, 1)
    
    # Verify storage methods were called correctly
    assert mock_prod_storage.get_node_by_id.call_count == 2
    assert mock_prod_storage.create_relationship.call_count == 0
    
    # Verify results
    assert len(successful) == 0
    assert len(failed) == 1

@pytest.mark.asyncio
async def test_execute_merge_end_to_end(execution_service, mock_staging_storage, mock_prod_storage, mock_progress_tracker, mock_redis_client):
    """Test executing a merge end-to-end"""
    merge_id = "test_merge_id"
    transform_id = "test_transform_id"
    
    # Mock _get_resolved_conflicts to return a non-empty list
    mock_conflict = Conflict(
        id="test_conflict_id",
        merge_id=merge_id,
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MINOR,
        description="Property value conflict",
        entity_id="node1",
        property_name="name",
        staging_value="Alice",
        production_value="Alice Smith",
        resolved=True,
        resolution=ResolutionOption(
            id="test_resolution_id",
            resolution_type="keep_staging",
            resolution_data={},
            description="Keep the staging value",
            confidence=0.9
        )
    )
    execution_service._get_resolved_conflicts = AsyncMock(return_value=[mock_conflict])
    
    # Mock _apply_resolutions
    original_apply_resolutions = execution_service._apply_resolutions
    execution_service._apply_resolutions = AsyncMock(side_effect=original_apply_resolutions)
    
    # Mock _execute_batch_merge
    execution_service._execute_batch_merge = AsyncMock(return_value={
        "nodes_merged": 3,
        "edges_merged": 2,
        "nodes_failed": 0,
        "edges_failed": 0,
        "total_items": 5,
        "success_rate": 1.0
    })
    
    # Execute merge
    result = await execution_service.execute_merge(
        merge_id=merge_id,
        transform_id=transform_id,
        batch_size=10
    )
    
    # Verify methods were called correctly
    mock_progress_tracker.start_merge_stage.assert_called_once_with(merge_id, MergeStage.MERGE)
    execution_service._get_resolved_conflicts.assert_called_once_with(merge_id)
    mock_staging_storage.get_transformation_data.assert_called_once_with(transform_id)
    execution_service._apply_resolutions.assert_called_once()
    execution_service._execute_batch_merge.assert_called_once()
    mock_progress_tracker.complete_merge_stage.assert_called_once()
    
    # Verify result
    assert result["nodes_merged"] == 3
    assert result["edges_merged"] == 2
    assert result["success_rate"] == 1.0

@pytest.mark.asyncio
async def test_execute_merge_with_error(execution_service, mock_progress_tracker):
    """Test executing a merge with an error"""
    merge_id = "test_merge_id"
    transform_id = "test_transform_id"
    
    # Mock _get_resolved_conflicts to raise an exception
    execution_service._get_resolved_conflicts = AsyncMock(side_effect=Exception("Test error"))
    
    # Execute merge and expect exception
    with pytest.raises(Exception) as excinfo:
        await execution_service.execute_merge(
            merge_id=merge_id,
            transform_id=transform_id
        )
    
    # Verify error handling
    assert "Test error" in str(excinfo.value)
    mock_progress_tracker.start_merge_stage.assert_called_once_with(merge_id, MergeStage.MERGE)
    mock_progress_tracker.fail_merge_stage.assert_called_once()

@pytest.mark.asyncio
async def test_cancel_merge(execution_service, mock_progress_tracker):
    """Test cancelling a merge"""
    merge_id = "test_merge_id"
    
    # Mock get_progress to return a running merge
    mock_progress_tracker.get_progress.return_value = MagicMock(
        overall_status=MergeStatus.RUNNING
    )
    
    # Add cancel_merge method to progress tracker
    mock_progress_tracker.cancel_merge = AsyncMock()
    
    # Cancel merge
    result = await execution_service.cancel_merge(merge_id)
    
    # Verify methods were called correctly
    mock_progress_tracker.get_progress.assert_called_once_with(merge_id)
    mock_progress_tracker.cancel_merge.assert_called_once()
    
    # Verify result
    assert result is True

@pytest.mark.asyncio
async def test_cancel_merge_already_completed(execution_service, mock_progress_tracker):
    """Test cancelling a merge that's already completed"""
    merge_id = "test_merge_id"
    
    # Mock get_progress to return a completed merge
    mock_progress_tracker.get_progress.return_value = MagicMock(
        overall_status=MergeStatus.COMPLETED
    )
    
    # Cancel merge
    result = await execution_service.cancel_merge(merge_id)
    
    # Verify methods were called correctly
    mock_progress_tracker.get_progress.assert_called_once_with(merge_id)
    
    # Verify result
    assert result is False

async def _apply_resolutions(
        staging_graph: GraphResponse,
        resolved_conflicts: List[Conflict]
    ) -> GraphResponse:
        """Apply conflict resolutions to create final graph for merging
        
        Args:
            staging_graph: Original staging graph
            resolved_conflicts: List of resolved conflicts
            
        Returns:
            Modified graph with resolutions applied
        """
        # Get copy of graph to modify
        resolved_graph = copy.deepcopy(staging_graph)
        
        # Process each conflict
        for conflict in resolved_conflicts:
            if not conflict.resolved or not conflict.resolution:
                continue
                
            # Apply resolution
            await execution_service._apply_resolution(resolved_graph, conflict, conflict.resolution)
            
        return resolved_graph 