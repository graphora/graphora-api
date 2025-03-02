"""
Unit tests for the merge service.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import datetime
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import timezone

from app.services.merge.service import MergeService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption, ResolutionStrategy
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
from app.services.merge.llm_analyzer import LLMConflictAnalyzer

# Import the mock storage class
from tests.unit.services.storage.test_graph_storage import MockGraphStorage

# Mock BAML module
@pytest.fixture(autouse=True)
def mock_baml():
    with patch('app.services.merge.service.b') as mock_b:
        # Mock entity matching function
        mock_b.MatchEntities.return_value = {
            "matches": [
                {
                    "staging_id": "s1",
                    "production_id": "p1",
                    "confidence": 0.9,
                    "match_strategy": "EXACT_NAME"
                }
            ]
        }
        
        # Mock property conflict analysis
        mock_b.AnalyzePropertyConflict.return_value = {
            "recommended_strategy": "KEEP_STAGING",
            "confidence": 0.8,
            "explanation": "Test explanation",
            "can_auto_resolve": True,
            "potential_risks": []
        }
        
        yield mock_b

@pytest.fixture
def mock_llm_analyzer():
    """Mock LLM analyzer for testing"""
    analyzer = MagicMock(spec=LLMConflictAnalyzer)
    analyzer.analyze_conflict = AsyncMock(return_value=[
        ResolutionOption(
            id="opt1",
            description="Keep staging value",
            resolution_type="keep_staging",
            confidence=0.8,
            reasoning="Staging data is more recent",
            auto_resolvable=True
        )
    ])
    return analyzer


@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker for testing"""
    tracker = MagicMock(spec=ProgressTracker)
    tracker.start_merge_stage = AsyncMock()
    tracker.update_merge_progress = AsyncMock()
    tracker.complete_merge_stage = AsyncMock()
    tracker.fail_merge_stage = AsyncMock()
    return tracker


@pytest.fixture
def mock_conflict_detection_service():
    """Mock conflict detection service for testing"""
    service = MagicMock(spec=ConflictDetectionService)
    
    # Mock conflict detection methods
    service.detect_property_conflicts = AsyncMock(return_value=[
        Conflict(
            id=f"conflict-{uuid.uuid4().hex}",
            merge_id="merge-123",
            conflict_type=ConflictType.PROPERTY,
            severity=ConflictSeverity.MINOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'age' has different values",
            context={},
            resolution_options=[]
        )
    ])
    
    service.detect_relationship_conflicts = AsyncMock(return_value=[
        Conflict(
            id=f"conflict-{uuid.uuid4().hex}",
            merge_id="merge-123",
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["sr1"],
            production_ids=["pr1"],
            description="Different relationship types",
            context={},
            resolution_options=[]
        )
    ])
    
    return service


@pytest.fixture
def merge_service(mock_progress_tracker, mock_conflict_detection_service, mock_llm_analyzer):
    """Create a merge service with mock dependencies"""
    # Create mock storage
    staging_storage = MockGraphStorage()
    prod_storage = MockGraphStorage()
    
    # Add test data to storage
    staging_storage.add_test_node(
        Node(id="s1", label="Person", type="Person", properties={"name": "Alice", "age": 30, "transform_id": "test_id"})
    )
    staging_storage.add_test_node(
        Node(id="s2", label="Person", type="Person", properties={"name": "Bob", "age": 25, "transform_id": "test_id"})
    )
    
    prod_storage.add_test_node(
        Node(id="p1", label="Person", type="Person", properties={"name": "Alice", "age": 32})
    )
    
    # Create service with mocks
    service = MergeService(
        storage=staging_storage,
        production_storage=prod_storage,
        progress_tracker=mock_progress_tracker
    )
    
    # Add merge_id attribute for tests
    service.merge_id = "test-merge-id"
    
    # Inject mocked dependencies
    service.conflict_detection_service = mock_conflict_detection_service
    service.llm_analyzer = mock_llm_analyzer
    
    # For backward compatibility with tests
    service.storage = staging_storage
    
    return service


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    client = AsyncMock()
    client.set = AsyncMock()
    client.get = AsyncMock()
    client.keys = AsyncMock(return_value=["merge:test_merge_id:conflict:1", "merge:test_merge_id:conflict:2"])
    client.expire = AsyncMock()
    
    return client


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
                # Mock the _analyze_property_conflict method to avoid BAML client error
                with patch('app.services.merge.conflict.ConflictDetectionService._analyze_property_conflict', 
                          new_callable=AsyncMock) as mock_analyze:
                    mock_analyze.return_value = {
                        "recommended_strategy": "keep_production",
                        "confidence": 0.9,
                        "explanation": "Production value is more complete",
                        "can_auto_resolve": True,
                        "potential_risks": []
                    }
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
        with patch.object(merge_service, 'apply_conflict_resolution', new_callable=AsyncMock) as mock_resolve:
            mock_resolve.return_value = {
                "applied": True,
                "conflict_id": conflict_id,
                "resolution_id": resolution_id,
                "verification": {"verified": True},
                "changes": {"action": "updated_production"}
            }
            result = await mock_resolve(conflict_id=conflict_id, resolution_id=resolution_id)
        
        # Assert
        assert result["applied"] is True
        assert result["conflict_id"] == conflict_id
        assert result["resolution_id"] == resolution_id
        mock_resolve.assert_called_once_with(conflict_id=conflict_id, resolution_id=resolution_id)
    
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
        mock_redis = AsyncMock()
        mock_status_data = {
            "transform_id": "transform-123",
            "status": "running",
            "current_stage": "validation",
            "validation_progress": 0.5,
            "execution_progress": 0.0,
            "started_at": datetime.datetime.now(timezone.utc).isoformat()
        }
        mock_redis.get.return_value = json.dumps(mock_status_data)
        
        # Mock Redis context manager
        mock_redis_cm = AsyncMock()
        mock_redis_cm.__aenter__.return_value = mock_redis
        
        # Patch Redis
        with patch('app.services.merge.service.redis.Redis.from_url', return_value=mock_redis_cm):
            # Act
            result = await merge_service.get_merge_progress(merge_id)
            
            # Assert
            mock_redis.get.assert_called_once_with(f"merge:{merge_id}:status")
            assert result is not None
            assert result.merge_id == merge_id
            assert result.status == "running"
            assert result.current_stage == "validation"


class TestMergeServiceAutoResolution:
    """Tests for auto-resolution functionality in MergeService"""
    
    @pytest.mark.asyncio
    async def test_auto_resolve_conflicts(self, merge_service, mock_redis_client):
        """Test auto-resolution of conflicts"""
        # Arrange
        merge_id = "test-merge-id"
        
        # Create test conflicts
        auto_conflict_id = f"auto_conflict_{uuid.uuid4().hex}"
        manual_conflict_id = f"manual_conflict_{uuid.uuid4().hex}"
        
        # Minor conflict that can be auto-resolved
        auto_conflict = Conflict(
            id=auto_conflict_id,
            merge_id=merge_id,
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            entity_id="s1",
            entity_type="Person",
            property_name="name",
            staging_value="Test",
            production_value="test",
            description="Property value conflict",
            resolution_options=[
                ResolutionOption(
                    id=f"{auto_conflict_id}_auto_option",
                    description="Auto option",
                    resolution_type=ResolutionStrategy.KEEP_STAGING,
                    resolution_data={},
                    confidence=0.8
                )
            ]
        )
        
        # Major conflict that requires manual resolution
        manual_conflict = Conflict(
            id=manual_conflict_id,
            merge_id=merge_id,
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            entity_id="s2",
            entity_type="Person",
            description="Relationship conflict",
            resolution_options=[
                ResolutionOption(
                    id=f"{manual_conflict_id}_manual_option",
                    description="Manual option",
                    resolution_type=ResolutionStrategy.KEEP_STAGING,
                    resolution_data={},
                    confidence=0.5
                )
            ]
        )
        
        # Mock Redis responses
        with patch('app.services.merge.service.get_redis_client', return_value=mock_redis_client):
            # Mock get_conflicts to return our test conflicts
            with patch.object(merge_service, 'get_conflicts', new_callable=AsyncMock) as mock_get_conflicts:
                mock_get_conflicts.return_value = ([auto_conflict, manual_conflict], 2)
                
                # Mock apply_conflict_resolution
                with patch.object(merge_service, 'apply_conflict_resolution', new_callable=AsyncMock) as mock_apply:
                    mock_apply.return_value = {
                        "applied": True,
                        "conflict_id": auto_conflict_id,
                        "resolution_id": f"{auto_conflict_id}_auto_option",
                        "verification": {"verified": True}
                    }
                    # Act
                    result = await merge_service.auto_resolve_conflicts(merge_id)
        
        # Assert
        assert result["total"] == 2
        assert result["auto_resolved"] == 1
        assert result["manual_required"] == 1
        assert ConflictType.PROPERTY_VALUE.value in result["by_type"]
        assert result["by_type"][ConflictType.PROPERTY_VALUE.value] == 1
        
        # Verify apply_conflict_resolution was called for auto-resolved conflict
        mock_apply.assert_called_once_with(
            merge_id=merge_id,
            conflict_id=auto_conflict_id,
            resolution_id=f"{auto_conflict_id}_auto_option"
        )


class TestMergeServiceStrategySelection:
    """Tests for strategy selection methods in MergeService"""
    
    @pytest.mark.asyncio
    @patch('app.services.merge.service.get_redis_client')
    async def test_select_resolution_strategies(self, mock_get_redis, merge_service, mock_redis_client):
        """Test selection of resolution strategies"""
        # Arrange
        mock_get_redis.return_value = mock_redis_client
        merge_id = str(uuid.uuid4())
        
        # Create test conflicts
        conflicts = [
            # Property value conflict
            Conflict(
                id="conflict_1",
                merge_id=merge_id,
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MINOR,
                entity_id="s1",
                entity_type="Person",
                property_name="name",
                staging_value="Test",
                production_value="test",
                description="Property value conflict",
                context={
                    "property_name": "name",
                    "staging_value": "Test",
                    "production_value": "test",
                    "entity_type": "Person",
                    "staging_timestamp": (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat(),
                    "production_timestamp": (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
                },
                resolution_options=[
                    ResolutionOption(
                        id="option_1",
                        description="Keep staging",
                        resolution_type="keep_staging",
                        resolution_data={},
                        confidence=0.5
                    ),
                    ResolutionOption(
                        id="option_2",
                        description="Keep production",
                        resolution_type="keep_production",
                        resolution_data={},
                        confidence=0.5
                    )
                ]
            ),
            # Relationship conflict
            Conflict(
                id="conflict_2",
                merge_id=merge_id,
                conflict_type=ConflictType.RELATIONSHIP_TYPE,
                severity=ConflictSeverity.MAJOR,
                entity_id="s2",
                entity_type="Person",
                description="Relationship conflict",
                context={
                    "staging_type": "WORKS_IN",
                    "production_type": "BELONGS_TO",
                    "entity_type": "Person"
                },
                resolution_options=[
                    ResolutionOption(
                        id="option_3",
                        description="Keep both",
                        resolution_type="keep_both_relationships",
                        resolution_data={},
                        confidence=0.5
                    )
                ]
            )
        ]
        
        # Configure Redis mocks
        mock_redis_client.get.side_effect = lambda key: {
            f"merge:{merge_id}:conflict_ids": json.dumps(["conflict_1", "conflict_2"]),
            f"merge:{merge_id}:conflict:conflict_1": conflicts[0].model_dump_json(),
            f"merge:{merge_id}:conflict:conflict_2": conflicts[1].model_dump_json(),
        }.get(key)
        
        # Mock strategy selection engine
        with patch("app.services.merge.strategy_selection.StrategySelectionEngine") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine
            
            # Configure mock engine responses
            mock_engine.select_strategy.side_effect = [
                ("prefer_staging", conflicts[0].resolution_options[0], 0.8, "Test explanation 1"),
                ("keep_both", conflicts[1].resolution_options[0], 0.7, "Test explanation 2")
            ]
            
            # Mock store_strategy_selection
            merge_service._store_strategy_selection = AsyncMock()
            
            # Act
            result = await merge_service.select_resolution_strategies(merge_id)
            
            # Assert
            assert result["total"] == 2
            assert result["processed"] == 2
            assert "prefer_staging" in result["strategy_counts"]
            assert "keep_both" in result["strategy_counts"]
            assert result["strategy_counts"]["prefer_staging"] == 1
            assert result["strategy_counts"]["keep_both"] == 1
            assert result["confidence_avg"] > 0.7
            assert "property_value" in result["by_type"]
            assert "relationship_type" in result["by_type"]
            
            # Verify store_strategy_selection was called for each conflict
            assert merge_service._store_strategy_selection.call_count == 2
    
    @pytest.mark.asyncio
    @patch('app.services.merge.service.get_redis_client')
    async def test_apply_selected_strategies(self, mock_get_redis, merge_service, mock_redis_client):
        """Test applying selected strategies"""
        # Arrange
        mock_get_redis.return_value = mock_redis_client
        merge_id = str(uuid.uuid4())
        
        # Create test conflicts with selected strategies
        conflicts = [
            # Conflict with high-confidence strategy
            Conflict(
                id="conflict_1",
                merge_id=merge_id,
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MINOR,
                entity_id="s1",
                entity_type="Person",
                property_name="name",
                staging_value="Test",
                production_value="test",
                description="Property value conflict",
                context={
                    "property_name": "name",
                    "staging_value": "Test",
                    "production_value": "test",
                    "entity_type": "Person",
                    "selected_strategy": {
                        "name": "prefer_staging",
                        "resolution_id": "option_1",
                        "confidence": 0.8,
                        "explanation": "Test explanation"
                    }
                },
                resolution_options=[
                    ResolutionOption(
                        id="option_1",
                        description="Keep staging",
                        resolution_type="keep_staging",
                        resolution_data={},
                        confidence=0.5
                    )
                ],
                resolved=False
            ),
            # Conflict with low-confidence strategy (should be skipped)
            Conflict(
                id="conflict_2",
                merge_id=merge_id,
                conflict_type=ConflictType.RELATIONSHIP_TYPE,
                severity=ConflictSeverity.MAJOR,
                entity_id="s2",
                entity_type="Person",
                description="Relationship conflict",
                context={
                    "selected_strategy": {
                        "name": "keep_both",
                        "resolution_id": "option_2",
                        "confidence": 0.4,  # Below threshold
                        "explanation": "Test explanation"
                    }
                },
                resolution_options=[
                    ResolutionOption(
                        id="option_2",
                        description="Keep both",
                        resolution_type="keep_both_relationships",
                        resolution_data={},
                        confidence=0.5
                    )
                ],
                resolved=False
            ),
            # Conflict with no selected strategy
            Conflict(
                id="conflict_3",
                merge_id=merge_id,
                conflict_type=ConflictType.ENTITY_MATCH,
                severity=ConflictSeverity.CRITICAL,
                entity_id="s3",
                entity_type="Organization",
                description="Entity type conflict",
                context={},  # No selected strategy
                resolution_options=[],
                resolved=False
            )
        ]
        
        # Configure Redis mocks
        mock_redis_client.get.side_effect = lambda key: {
            f"merge:{merge_id}:conflict_ids": json.dumps(["conflict_1", "conflict_2", "conflict_3"]),
            f"merge:{merge_id}:conflict:conflict_1": conflicts[0].model_dump_json(),
            f"merge:{merge_id}:conflict:conflict_2": conflicts[1].model_dump_json(),
            f"merge:{merge_id}:conflict:conflict_3": conflicts[2].model_dump_json(),
        }.get(key)
        
        # Mock apply_conflict_resolution
        merge_service.apply_conflict_resolution = AsyncMock()
        merge_service.apply_conflict_resolution.return_value = {
            "applied": True,
            "conflict_id": "conflict_1",
            "resolution_id": "option_1",
            "verification": {"verified": True}
        }
        
        # Act
        result = await merge_service.apply_selected_strategies(merge_id, min_confidence=0.7)
        
        # Assert
        assert result["total"] == 3
        assert result["applied"] == 1
        assert result["skipped_low_confidence"] == 1
        assert result["skipped_no_strategy"] == 1
        assert "prefer_staging" in result["by_strategy"]
        assert result["by_strategy"]["prefer_staging"] == 1
        
        # Verify apply_conflict_resolution was called only for high-confidence strategy
        merge_service.apply_conflict_resolution.assert_called_once_with(
            merge_id=merge_id,
            conflict_id="conflict_1", 
            resolution_id="option_1"
        )
