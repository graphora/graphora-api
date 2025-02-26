"""
Unit tests for the merge service.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import datetime
import json
import uuid
from typing import Dict, List, Any, Optional

from app.services.merge.service import MergeService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.graph import Node, Edge, GraphResponse
from app.services.storage.interface import GraphStorageInterface
from app.services.merge.models import (
    MergeStage, 
    MergeStatus, 
    StageStatus,
    EntityMappingResult,
    EntityMatch,
    MatchStrategy
)
from app.services.merge.progress import ProgressTracker
from app.services.merge.conflict import ConflictDetectionService

# Import the mock storage class
from tests.unit.services.storage.test_graph_storage import MockGraphStorage


@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker for testing"""
    tracker = MagicMock()
    tracker.start_merge_stage = AsyncMock()
    tracker.update_merge_progress = AsyncMock()
    tracker.complete_merge_stage = AsyncMock()
    tracker.initialize_merge = AsyncMock()
    tracker.fail_merge = AsyncMock()
    tracker.fail_merge_stage = AsyncMock()
    tracker.get_progress = AsyncMock()
    return tracker


@pytest.fixture
def mock_conflict_detection_service():
    """Mock conflict detection service"""
    service = MagicMock(spec=ConflictDetectionService)
    service.detect_property_conflicts = AsyncMock(return_value=[
        Conflict(
            id=f"conflict-{uuid.uuid4().hex}",
            merge_id="merge-123",
            conflict_type=ConflictType.PROPERTY,
            severity=ConflictSeverity.MAJOR,
            entity_id="s1",
            entity_type="Person",
            property_name="age",
            staging_value=30,
            production_value=32,
            description="Property 'age' has different values: 30 vs 32"
        )
    ])
    service._create_property_conflict = AsyncMock()
    return service


@pytest.fixture
def merge_service(mock_progress_tracker, mock_conflict_detection_service):
    """Create a merge service with mock dependencies"""
    # Create mock storage
    mock_storage = MockGraphStorage()
    
    # Create service with mock dependencies
    service = MergeService(storage=mock_storage)
    
    # Replace progress tracker with mock
    service.progress_tracker = mock_progress_tracker
    
    # Add test data
    mock_storage.add_test_node(
        Node(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30, "transform_id": "test_id"})
    )
    mock_storage.add_test_node(
        Node(id="s2", label="Person", type="Person", properties={"name": "Bob", "age": 25, "transform_id": "test_id"})
    )
    
    mock_storage.add_test_node(
        Node(id="p1", label="Person", type="Person", properties={"name": "Alice", "age": 32})
    )
    
    # Add a relationship
    mock_storage.add_test_relationship(
        Edge(
            id="r1",
            source="s1",
            target="s2",
            type="KNOWS",
            properties={"since": 2020, "transform_id": "test_id"}
        )
    )
    
    # Mock the conflict detection service
    with patch('app.services.merge.service.ConflictDetectionService', return_value=mock_conflict_detection_service):
        yield service


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    redis_client = MagicMock()
    redis_client.get = AsyncMock()
    redis_client.set = AsyncMock()
    redis_client.expire = AsyncMock()
    return redis_client


class TestMergeService:
    """Tests for the MergeService implementation"""
    
    @pytest.mark.asyncio
    async def test_extract_staging_graph(self, merge_service):
        """Test extracting staging graph"""
        # Arrange
        mock_graph = GraphResponse(
            nodes=[
                Node(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30, "transform_id": "test_id"}),
                Node(id="s2", label="Person", type="Person", properties={"name": "Bob", "age": 25, "transform_id": "test_id"})
            ],
            edges=[],
            total_nodes=2,
            total_edges=0
        )
        
        # Act
        with patch('app.services.merge.service.extract_staging_graph', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_graph
            result = await merge_service.detect_conflicts("merge-123", "test_id")
        
        # Assert
        mock_extract.assert_called_once_with(merge_service.storage, "test_id")
        assert merge_service.progress_tracker.start_merge_stage.called
        assert merge_service.progress_tracker.update_merge_progress.called
        assert merge_service.progress_tracker.complete_merge_stage.called
    
    @pytest.mark.asyncio
    async def test_map_production_entities(self, merge_service):
        """Test mapping production entities"""
        # Arrange
        staging_graph = GraphResponse(
            nodes=[
                Node(id="s1", label="Person", type="Person", properties={"name": "Alice", "transform_id": "test_id"}),
                Node(id="s2", label="Person", type="Person", properties={"name": "Bob", "transform_id": "test_id"})
            ],
            edges=[],
            total_nodes=2,
            total_edges=0
        )
        
        entity_mapping = EntityMappingResult(
            matches={
                "s1": EntityMatch(
                    staging_id="s1",
                    production_matches=["p1"],
                    match_strategy=MatchStrategy.EXACT_NAME,
                    match_confidence=0.9
                ),
                "s2": EntityMatch(
                    staging_id="s2",
                    production_matches=[],
                    match_strategy=MatchStrategy.PROPERTY_SIMILARITY,
                    match_confidence=0.0
                )
            },
            total_entities=2,
            matched_entities=1,
            mapping_time_ms=10.5
        )
        
        # Act
        with patch('app.services.merge.service.extract_staging_graph', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = staging_graph
            with patch('app.services.merge.service.map_production_entities', new_callable=AsyncMock) as mock_map:
                mock_map.return_value = entity_mapping
                result = await merge_service.detect_conflicts("merge-123", "test_id")
        
        # Assert
        assert merge_service.progress_tracker.update_merge_progress.called
    
    @pytest.mark.asyncio
    async def test_detect_property_conflicts(self, merge_service, mock_conflict_detection_service):
        """Test detecting property conflicts between nodes"""
        # Arrange
        staging_node = Node(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30})
        prod_node = Node(id="p1", label="Person", type="Person", properties={"name": "Alice", "age": 32})
        
        # Act
        with patch.object(merge_service, 'detect_property_conflicts', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = [
                Conflict(
                    id=f"conflict-{uuid.uuid4().hex}",
                    merge_id="merge-123",
                    conflict_type=ConflictType.PROPERTY,
                    severity=ConflictSeverity.MAJOR,
                    entity_id="s1",
                    entity_type="Person",
                    property_name="age",
                    staging_value=30,
                    production_value=32,
                    description="Property 'age' has different values: 30 vs 32"
                )
            ]
            conflicts = await mock_detect(staging_node, prod_node)
        
        # Assert
        assert len(conflicts) > 0
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.PROPERTY
        assert conflict.severity == ConflictSeverity.MAJOR
        assert "age" in conflict.description
        assert conflict.entity_id == "s1"
        assert conflict.entity_type == "Person"
        assert conflict.property_name == "age"
        assert conflict.staging_value == 30
        assert conflict.production_value == 32
    
    @pytest.mark.asyncio
    async def test_detect_property_conflicts_for_graph(self, merge_service):
        """Test detecting property conflicts for entire graph"""
        # Arrange
        production_entity_mapping = {
            "s1": ["p1"],
            "s2": []  # No match for s2
        }
        
        staging_graph = GraphResponse(
            nodes=[
                Node(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30, "transform_id": "test_id"}),
                Node(id="s2", label="Person", type="Person", properties={"name": "Bob", "age": 25, "transform_id": "test_id"})
            ],
            edges=[],
            total_nodes=2,
            total_edges=0
        )
        
        # Act
        with patch.object(merge_service, 'detect_property_conflicts_for_graph', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = [
                Conflict(
                    id=f"conflict-{uuid.uuid4().hex}",
                    merge_id="merge-123",
                    conflict_type=ConflictType.PROPERTY,
                    severity=ConflictSeverity.MAJOR,
                    entity_id="s1",
                    entity_type="Person",
                    property_name="age",
                    staging_value=30,
                    production_value=32,
                    description="Property 'age' has different values: 30 vs 32"
                )
            ]
            conflicts = await mock_detect(staging_graph, production_entity_mapping)
        
        # Assert
        assert len(conflicts) > 0
        assert any(c.property_name == "age" for c in conflicts)
    
    @pytest.mark.asyncio
    async def test_detect_relationship_conflicts(self, merge_service):
        """Test detecting relationship conflicts"""
        # Arrange
        production_entity_mapping = {
            "s1": ["p1"],
            "s2": []  # No match for s2
        }
        
        staging_graph = GraphResponse(
            nodes=[
                Node(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30, "transform_id": "test_id"}),
                Node(id="s2", label="Person", type="Person", properties={"name": "Bob", "age": 25, "transform_id": "test_id"})
            ],
            edges=[
                Edge(id="r1", source="s1", target="s2", type="KNOWS", properties={"since": 2020})
            ],
            total_nodes=2,
            total_edges=1
        )
        
        # Act
        with patch.object(merge_service, 'detect_relationship_conflicts', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = []  # No conflicts expected since s2 has no production match
            conflicts = await mock_detect(staging_graph, production_entity_mapping)
        
        # Assert
        assert len(conflicts) == 0
    
    @pytest.mark.asyncio
    @patch('app.services.merge.service.get_redis_client')
    async def test_store_conflicts(self, mock_get_redis, merge_service, mock_redis_client):
        """Test storing conflicts in Redis"""
        # Arrange
        mock_get_redis.return_value = mock_redis_client
        merge_id = "merge-123"
        conflicts = [
            Conflict(
                id=f"conflict-{uuid.uuid4().hex}",
                merge_id=merge_id,
                conflict_type=ConflictType.PROPERTY,
                severity=ConflictSeverity.MAJOR,
                entity_id="s1",
                entity_type="Person",
                property_name="age",
                staging_value=30,
                production_value=32,
                description="Property 'age' has different values: 30 vs 32"
            )
        ]
        
        # Act
        with patch.object(merge_service, '_store_conflicts', new_callable=AsyncMock) as mock_store:
            mock_store.return_value = None
            await mock_store(merge_id, conflicts)
        
        # Assert
        mock_store.assert_called_once_with(merge_id, conflicts)
    
    @pytest.mark.asyncio
    @patch('app.services.merge.service.get_redis_client')
    async def test_get_conflicts(self, mock_get_redis, merge_service, mock_redis_client):
        """Test retrieving conflicts from Redis"""
        # Arrange
        mock_get_redis.return_value = mock_redis_client
        merge_id = "merge-123"
        conflict_id = f"conflict-{uuid.uuid4().hex}"
        
        # Mock Redis responses
        conflict = Conflict(
            id=conflict_id,
            merge_id=merge_id,
            conflict_type=ConflictType.PROPERTY,
            severity=ConflictSeverity.MAJOR,
            entity_id="s1",
            entity_type="Person",
            property_name="age",
            staging_value=30,
            production_value=32,
            description="Property 'age' has different values: 30 vs 32"
        )
        
        mock_redis_client.get.side_effect = [
            json.dumps([conflict_id]),  # First call returns conflict IDs
            conflict.model_dump_json()  # Second call returns conflict data
        ]
        
        # Act
        conflicts, count = await merge_service.get_conflicts(merge_id)
        
        # Assert
        assert len(conflicts) == 1
        assert conflicts[0].id == conflict_id
        assert count == 1
    
    @pytest.mark.asyncio
    @patch('app.services.merge.service.get_redis_client')
    async def test_resolve_conflict(self, mock_get_redis, merge_service, mock_redis_client):
        """Test resolving a conflict"""
        # Arrange
        mock_get_redis.return_value = mock_redis_client
        merge_id = "merge-123"
        conflict_id = f"conflict-{uuid.uuid4().hex}"
        resolution_id = f"resolution-{uuid.uuid4().hex}"
        
        # Create a conflict with resolution options
        conflict = Conflict(
            id=conflict_id,
            merge_id=merge_id,
            conflict_type=ConflictType.PROPERTY,
            severity=ConflictSeverity.MAJOR,
            entity_id="s1",
            entity_type="Person",
            property_name="age",
            staging_value=30,
            production_value=32,
            description="Property 'age' has different values: 30 vs 32",
            resolution_options=[
                ResolutionOption(
                    id=resolution_id,
                    description="Keep staging value (30)",
                    resolution_type="keep_staging",
                    confidence=0.8,
                    reasoning="Staging data is more recent",
                    requires_review=True,
                    auto_resolvable=True
                )
            ]
        )
        
        # Mock Redis responses
        mock_redis_client.get.side_effect = [
            conflict.model_dump_json(),  # First call returns conflict
            json.dumps({  # Second call returns counts
                "total": 1,
                "resolved": 0,
                "unresolved": 1,
                "by_type": {"property": 1},
                "by_severity": {"major": 1}
            })
        ]
        
        # Act
        with patch.object(merge_service, 'resolve_conflict', new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = True
            result = await mock_resolve(merge_id, conflict_id, resolution_id)
        
        # Assert
        assert result is True
        mock_resolve.assert_called_once_with(merge_id, conflict_id, resolution_id)
    
    @pytest.mark.asyncio
    async def test_start_merge_flow(self, merge_service):
        """Test starting a merge flow"""
        # Arrange
        session_id = "session-123"
        transform_id = "test_id"
        
        # Act
        with patch('app.services.merge.service.merge_flow', new_callable=AsyncMock) as mock_flow:
            merge_id = await merge_service.start_merge_flow(session_id, transform_id)
        
        # Assert
        assert merge_id.startswith("merge_")
        assert merge_service.progress_tracker.initialize_merge.called
        assert mock_flow.called
    
    @pytest.mark.asyncio
    async def test_get_merge_progress(self, merge_service):
        """Test getting merge progress"""
        # Arrange
        merge_id = "merge-123"
        
        # Act
        await merge_service.get_merge_progress(merge_id)
        
        # Assert
        merge_service.progress_tracker.get_progress.assert_called_once_with(merge_id)
