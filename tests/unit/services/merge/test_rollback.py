"""Unit tests for merge rollback functionality"""
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import json

from app.services.merge.service import MergeService
from app.services.merge.models import (
    RollbackType,
    RollbackOptions,
    RollbackResponse,
    SnapshotData,
    MergeProgress,
    MergeStatus,
    MergeStage,
    StageStatus
)
from app.services.storage.models import Node, Edge

@pytest.fixture
def mock_storage():
    """Mock storage interface"""
    storage = AsyncMock()
    storage.get_node_by_id = AsyncMock()
    storage.get_relationships_between_nodes = AsyncMock()
    storage.update_node = AsyncMock()
    storage.create_node = AsyncMock()
    storage.get_edges_between = AsyncMock()
    storage.create_relationship = AsyncMock()
    return storage

@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker"""
    tracker = AsyncMock()
    tracker.get_merge_progress = AsyncMock()
    tracker.update_merge_progress = AsyncMock()
    tracker.cancel_merge = AsyncMock()
    return tracker

@pytest.fixture
def mock_transaction_manager():
    """Mock transaction manager"""
    manager = AsyncMock()
    manager.begin_transaction = AsyncMock(return_value="transaction-123")
    manager.commit_transaction = AsyncMock(return_value=True)
    manager.rollback_transaction = AsyncMock(return_value=True)
    
    # Create a mock context manager for start_transaction
    class MockTransactionContext:
        async def __aenter__(self):
            return "mock-tx-context"
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False  # Don't suppress exceptions
    
    # Make start_transaction return the mock context manager
    manager.start_transaction = AsyncMock(return_value=MockTransactionContext())
    
    return manager

@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    client = AsyncMock()
    client.get = AsyncMock()
    client.set = AsyncMock()
    return client

@pytest.fixture
def merge_service(mock_storage, mock_progress_tracker, mock_transaction_manager, mock_redis_client):
    """Create a merge service with mocked dependencies"""
    service = MergeService(
        storage=mock_storage,
        production_storage=mock_storage,
        progress_tracker=mock_progress_tracker,
        transaction_manager=mock_transaction_manager
    )
    
    # Set the transaction manager directly
    service._transaction_manager = mock_transaction_manager
    
    # Mock the redis_client
    service.redis_client = mock_redis_client
    
    # Mock the _get_merge_metadata method
    async def mock_get_metadata(merge_id):
        return {
            "snapshot_id": "snapshot-123",
            "transform_id": "transform-123",
            "status": "completed"
        }
    
    service._get_merge_metadata = mock_get_metadata
    
    # Mock the _get_snapshot_data method
    async def mock_get_snapshot(snapshot_id):
        return {
            "snapshot_id": snapshot_id,
            "merge_id": "merge-123",
            "nodes": [],
            "edges": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {}
        }
    
    service._get_snapshot_data = mock_get_snapshot
    
    return service

@pytest.fixture
def sample_snapshot():
    """Create a sample snapshot for testing"""
    return SnapshotData(
        snapshot_id="snapshot-123",
        merge_id="merge-123",
        nodes=[
            Node(id="node1", label="Person", type="Person", properties={"name": "John", "age": 30}),
            Node(id="node2", label="Company", type="Company", properties={"name": "Acme Inc."})
        ],
        relationships=[
            Edge(id="rel1", source="node1", target="node2", type="WORKS_AT", properties={})
        ],
        timestamp=datetime.now(timezone.utc),
        metadata={}
    )

@pytest.fixture
def sample_merge_progress():
    """Create a sample merge progress for testing"""
    return MergeProgress(
        merge_id="merge-123",
        overall_status=MergeStatus.COMPLETED,
        current_stage=MergeStage.MERGE,
        stages_progress={},
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc)
    )

class TestMergeRollback:
    """Test cases for merge rollback functionality"""
    
    async def test_create_snapshot(self, merge_service, mock_storage):
        """Test creating a snapshot of affected nodes"""
        # Arrange
        merge_id = "merge-123"
        affected_nodes = ["node1", "node2"]
        
        # Mock storage responses
        mock_storage.get_node_by_id.side_effect = lambda node_id: Node(
            id=node_id,
            label="Test",
            type="Test",
            properties={"name": f"Node {node_id}"}
        )
        mock_storage.get_relationships_between_nodes.return_value = [
            Edge(id="rel1", source="node1", target="node2", type="TEST", properties={})
        ]
        
        # Act
        snapshot = await merge_service._create_snapshot(merge_id, affected_nodes)
        
        # Assert
        assert snapshot.merge_id == merge_id
        assert len(snapshot.nodes) == 2
        assert len(snapshot.relationships) == 1
        assert snapshot.nodes[0].id == "node1"
        assert snapshot.nodes[1].id == "node2"
        assert snapshot.relationships[0].id == "rel1"
        
        # Verify storage calls
        assert mock_storage.get_node_by_id.call_count == 2
        assert mock_storage.get_relationships_between_nodes.call_count == 1
        
        # Verify snapshot was stored
        assert merge_service.redis_client.set.call_count == 2
    
    async def test_store_and_load_snapshot(self, merge_service, mock_redis_client, sample_snapshot):
        """Test storing and loading a snapshot"""
        # Arrange
        mock_redis_client.get.return_value = sample_snapshot.model_dump_json().encode('utf-8')
        
        # Act - Store snapshot
        await merge_service._store_snapshot(sample_snapshot)
        
        # Assert - Store calls
        assert mock_redis_client.set.call_count == 2
        
        # Act - Load snapshot
        loaded_snapshot = await merge_service._load_snapshot(sample_snapshot.snapshot_id)
        
        # Assert - Loaded snapshot
        assert loaded_snapshot is not None
        assert loaded_snapshot.snapshot_id == sample_snapshot.snapshot_id
        assert loaded_snapshot.merge_id == sample_snapshot.merge_id
        assert len(loaded_snapshot.nodes) == len(sample_snapshot.nodes)
        assert len(loaded_snapshot.relationships) == len(sample_snapshot.relationships)
    
    async def test_apply_complete_rollback(self, merge_service, mock_storage, mock_transaction_manager, sample_snapshot):
        """Test applying a complete rollback"""
        # Arrange
        rollback_id = "rollback-123"
        
        # Mock storage responses
        mock_storage.get_node_by_id.side_effect = lambda node_id: Node(
            id=node_id,
            label="Test",
            type="Test",
            properties={"name": f"Modified {node_id}"}  # Different from snapshot
        )
        mock_storage.get_edges_between.return_value = []  # No existing edges
        
        # Act
        result = await merge_service._apply_complete_rollback(sample_snapshot, rollback_id)
        
        # Assert
        assert result["success"] is True
        assert result["nodes_restored"] == 2
        assert result["relationships_restored"] == 1
        
        # Verify transaction was used
        assert mock_transaction_manager.begin_transaction.call_count == 1
        assert mock_transaction_manager.commit_transaction.call_count == 1
        
        # Verify storage calls
        assert mock_storage.update_node.call_count == 2
        assert mock_storage.create_relationship.call_count == 1
    
    async def test_apply_partial_rollback(self, merge_service, mock_storage, mock_transaction_manager, sample_snapshot):
        """Test applying a partial rollback"""
        # Arrange
        rollback_id = "rollback-123"
        entity_ids = ["node1"]  # Only rollback node1
        
        # Mock storage responses
        mock_storage.get_node_by_id.side_effect = lambda node_id: Node(
            id=node_id,
            label="Test",
            type="Test",
            properties={"name": f"Modified {node_id}"}
        )
        mock_storage.get_edges_between.return_value = []
        
        # Act
        result = await merge_service._apply_partial_rollback(sample_snapshot, entity_ids, rollback_id)
        
        # Assert
        assert result["success"] is True
        assert result["nodes_restored"] == 1
        assert result["relationships_restored"] == 0  # No relationships (both ends must be in entity_ids)
        
        # Verify transaction was used
        assert mock_transaction_manager.begin_transaction.call_count == 1
        assert mock_transaction_manager.commit_transaction.call_count == 1
        
        # Verify storage calls - only node1 should be updated
        mock_storage.update_node.assert_called_once_with("node1", {"name": "John", "age": 30}, tx="transaction-123")
    
    async def test_rollback_merge_complete(self, merge_service, mock_redis_client, mock_progress_tracker, sample_snapshot, sample_merge_progress):
        """Test rollback_merge with complete rollback"""
        # Arrange
        merge_id = "merge-123"
        options = RollbackOptions(rollback_type=RollbackType.COMPLETE)
        
        # Mock responses
        mock_redis_client.get.side_effect = lambda key: {
            "merge:merge-123:snapshot": b"snapshot-123",
            "snapshot:snapshot-123": sample_snapshot.model_dump_json().encode('utf-8')
        }.get(key)
        
        mock_progress_tracker.get_merge_progress.return_value = sample_merge_progress
        
        # Mock _apply_complete_rollback
        merge_service._apply_complete_rollback = AsyncMock(return_value={
            "success": True,
            "nodes_restored": 2,
            "relationships_restored": 1
        })
        
        # Act
        response = await merge_service.rollback_merge(merge_id, options)
        
        # Assert
        assert response.rollback_id.startswith("rollback_")
        assert response.merge_id == merge_id
        assert response.status == "successful"
        assert response.rollback_type == RollbackType.COMPLETE.value
        assert response.nodes_restored == 2
        assert response.relationships_restored == 1
        
        # Verify progress tracker was updated
        mock_progress_tracker.update_merge_status.assert_called_once_with(
            merge_id=merge_id,
            status=MergeStatus.ROLLED_BACK
        )
    
    async def test_rollback_merge_partial(self, merge_service, mock_redis_client, mock_progress_tracker, sample_snapshot, sample_merge_progress):
        """Test rollback_merge with partial rollback"""
        # Arrange
        merge_id = "merge-123"
        options = RollbackOptions(
            rollback_type=RollbackType.PARTIAL,
            entity_ids=["node1"]
        )
        
        # Mock responses
        mock_redis_client.get.side_effect = lambda key: {
            "merge:merge-123:snapshot": b"snapshot-123",
            "snapshot:snapshot-123": sample_snapshot.model_dump_json().encode('utf-8')
        }.get(key)
        
        mock_progress_tracker.get_merge_progress.return_value = sample_merge_progress
        
        # Mock _apply_partial_rollback
        merge_service._apply_partial_rollback = AsyncMock(return_value={
            "success": True,
            "nodes_restored": 1,
            "relationships_restored": 0
        })
        
        # Act
        response = await merge_service.rollback_merge(merge_id, options)
        
        # Assert
        assert response.rollback_id.startswith("rollback_")
        assert response.merge_id == merge_id
        assert response.status == "successful"
        assert response.rollback_type == RollbackType.PARTIAL.value
        assert response.nodes_restored == 1
        assert response.relationships_restored == 0
        
        # Verify progress tracker was updated
        mock_progress_tracker.update_merge_status.assert_called_once_with(
            merge_id=merge_id,
            status=MergeStatus.ROLLED_BACK
        )
    
    async def test_rollback_merge_no_snapshot(self, merge_service, mock_redis_client):
        """Test rollback_merge when no snapshot exists"""
        # Arrange
        merge_id = "merge-123"
        options = RollbackOptions(rollback_type=RollbackType.COMPLETE)
        
        # Mock responses - no snapshot
        mock_redis_client.get.return_value = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Snapshot .* not found for merge .*"):
            await merge_service.rollback_merge(merge_id, options)
    
    async def test_rollback_merge_error_handling(self, merge_service, mock_redis_client, mock_progress_tracker, sample_snapshot, sample_merge_progress):
        """Test error handling during rollback"""
        # Arrange
        merge_id = "merge-123"
        options = RollbackOptions(rollback_type=RollbackType.COMPLETE)
        
        # Mock responses
        mock_redis_client.get.side_effect = lambda key: {
            "merge:merge-123:snapshot": b"snapshot-123",
            "snapshot:snapshot-123": sample_snapshot.model_dump_json().encode('utf-8')
        }.get(key)
        
        mock_progress_tracker.get_merge_progress.return_value = sample_merge_progress
        
        # Mock _apply_complete_rollback to raise an error
        error_message = "Simulated error during rollback"
        merge_service._apply_complete_rollback = AsyncMock(side_effect=Exception(error_message))
        
        # Act & Assert
        with pytest.raises(Exception, match=error_message):
            await merge_service.rollback_merge(merge_id, options)
        
        # Verify failure was logged
        assert merge_service.redis_client.set.call_count == 1  # Only the failure log
        
        # Get the call arguments
        call_args = merge_service.redis_client.set.call_args[0]
        
        # Verify the key starts with rollback:
        assert call_args[0].startswith("rollback:")
        
        # Verify the value is a JSON string with status "failed"
        value_dict = json.loads(call_args[1])
        assert value_dict["status"] == "failed"
        assert value_dict["error"] == error_message 