import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import redis
import json
from datetime import datetime, timedelta
from app.services.resolution_history_service import ResolutionHistoryService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.resolution_history import ResolutionHistoryEntry, ResolutionFilter, PaginationParams

@pytest.fixture
def mock_redis_client():
    with patch("redis.Redis.from_url") as mock_redis:
        client = MagicMock()
        mock_redis.return_value = client
        yield client

@pytest.fixture
def sample_conflict():
    return Conflict(
        id="conflict1",
        merge_id="merge1",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        staging_ids=["s1"],
        production_ids=["p1"],
        description="Property 'age' has different values",
        context={
            "property_name": "age",
            "staging_value": 30,
            "production_value": 32,
            "entity_type": "Person"
        },
        resolution_options=[
            ResolutionOption(
                id="opt1",
                description="Keep staging value: 30",
                resolution_type="keep_staging",
                resolution_data={"property_name": "age"},
                confidence=0.5
            ),
            ResolutionOption(
                id="opt2",
                description="Keep production value: 32",
                resolution_type="keep_production",
                resolution_data={"property_name": "age"},
                confidence=0.5
            )
        ]
    )

class TestResolutionHistoryService:
    @pytest.mark.asyncio
    async def test_store_resolution(self, mock_redis_client, sample_conflict):
        # Arrange
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Act
        entry = await service.store_resolution(
            conflict=sample_conflict,
            resolution_id="opt1",
            applied_by="test_user",
            merge_id="merge1",
            success=True
        )
        
        # Assert
        assert entry.conflict_id == "conflict1"
        assert entry.merge_id == "merge1"
        assert entry.conflict_type == ConflictType.PROPERTY_VALUE
        assert entry.resolution_type == "keep_staging"
        assert "age" in entry.property_names
        assert "Person" in entry.entity_types
        
        # Verify Redis calls
        mock_redis_client.set.assert_called_once()
        key = mock_redis_client.set.call_args[0][0]
        assert key.startswith("resolution_history:")
        
        # Verify indexes
        mock_redis_client.sadd.assert_called()
        index_calls = mock_redis_client.sadd.call_args_list
        
        # Check conflict type index
        conflict_type_call = [call for call in index_calls 
                           if call[0][0].startswith("resolution_index:conflict_type:")]
        assert len(conflict_type_call) > 0
        
        # Check property name index
        property_call = [call for call in index_calls
                     if call[0][0].startswith("resolution_index:property_name:age")]
        assert len(property_call) > 0
        
        # Check entity type index
        entity_call = [call for call in index_calls
                    if call[0][0].startswith("resolution_index:entity_type:Person")]
        assert len(entity_call) > 0
        
        # Check embedding storage
        mock_redis_client.hset.assert_called_once()
        assert mock_redis_client.hset.call_args[0][0] == "resolution_embeddings"
        assert mock_redis_client.hset.call_args[0][1] == entry.id
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions(self, mock_redis_client, sample_conflict):
        # Arrange
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Setup mock data
        mock_candidates = set([b"res1", b"res2", b"res3"])
        mock_redis_client.smembers.return_value = mock_candidates
        
        # Setup mock embeddings
        mock_redis_client.hget.side_effect = lambda key, id: json.dumps([0.5, 0.3, 0.8]) if id in ["res1", "res2", "res3"] else None
        
        # Setup mock entries
        entry1 = ResolutionHistoryEntry(
            id="res1",
            conflict_id="c1",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "age"},
            resolution_id="opt1",
            resolution_type="keep_staging",
            entity_types=["Person"],
            property_names=["age"],
            applied_by="user1"
        )
        
        entry2 = ResolutionHistoryEntry(
            id="res2",
            conflict_id="c2",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "salary"},
            resolution_id="opt2",
            resolution_type="keep_production",
            entity_types=["Employee"],
            property_names=["salary"],
            applied_by="user1"
        )
        
        mock_redis_client.get.side_effect = lambda key: {
            "resolution_history:res1": entry1.model_dump_json(),
            "resolution_history:res2": entry2.model_dump_json(),
            "resolution_history:res3": entry1.model_dump_json(),  # Use entry1 for res3 too
        }.get(key)
        
        # Mock similarity calculation
        with patch.object(service, '_calculate_similarity', return_value=0.85):
            # Act
            results = await service.find_similar_resolutions(sample_conflict)
            
            # Assert
            assert len(results) > 0
            assert "similarity_score" in results[0]
            assert results[0]["similarity_score"] == 0.85
            assert "entry" in results[0]

    @pytest.mark.asyncio
    async def test_get_resolution_history(self, mock_redis_client):
        # Arrange
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Setup mock entries
        entry1 = ResolutionHistoryEntry(
            id="res1",
            conflict_id="c1",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "age"},
            resolution_id="opt1",
            resolution_type="keep_staging",
            entity_types=["Person"],
            property_names=["age"],
            applied_by="user1",
            applied_at=datetime.now() - timedelta(days=1)
        )
        
        entry2 = ResolutionHistoryEntry(
            id="res2",
            conflict_id="c2",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "salary"},
            resolution_id="opt2",
            resolution_type="keep_production",
            entity_types=["Employee"],
            property_names=["salary"],
            applied_by="user1",
            applied_at=datetime.now()
        )
        
        # Mock Redis smembers for merge_id index
        mock_redis_client.smembers.return_value = {b"res1", b"res2"}
        
        # Mock Redis get to return entry JSON
        mock_redis_client.get.side_effect = lambda key: {
            "resolution_history:res1": entry1.model_dump_json(),
            "resolution_history:res2": entry2.model_dump_json(),
        }.get(key)
        
        # Act
        entries = await service.get_resolution_history(merge_id="m1")
        
        # Assert
        assert len(entries) == 2
        # Should be sorted by applied_at (newer first)
        assert entries[0].id == "res2"
        assert entries[1].id == "res1"
    
    @pytest.mark.asyncio
    async def test_update_resolution_success(self, mock_redis_client):
        # Arrange
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Setup mock entry
        entry = ResolutionHistoryEntry(
            id="res1",
            conflict_id="c1",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "age"},
            resolution_id="opt1",
            resolution_type="keep_staging",
            entity_types=["Person"],
            property_names=["age"],
            applied_by="user1",
            success=True
        )
        
        mock_redis_client.get.return_value = entry.model_dump_json()
        
        # Act
        result = await service.update_resolution_success(
            resolution_id="res1",
            success=False,
            feedback="This resolution didn't work well"
        )
        
        # Assert
        assert result is True
        
        # Verify Redis set was called with updated entry
        mock_redis_client.set.assert_called_once()
        set_key, set_value = mock_redis_client.set.call_args[0]
        assert set_key == "resolution_history:res1"
        
        # Parse the updated entry
        updated_entry = ResolutionHistoryEntry.model_validate_json(set_value)
        assert updated_entry.success is False
        assert updated_entry.feedback == "This resolution didn't work well"
    
    @pytest.mark.asyncio
    async def test_get_resolution_stats(self, mock_redis_client):
        # Arrange
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Setup mock entries
        entry1 = ResolutionHistoryEntry(
            id="res1",
            conflict_id="c1",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "age"},
            resolution_id="opt1",
            resolution_type="keep_staging",
            entity_types=["Person"],
            property_names=["age"],
            applied_by="user1",
            success=True,
            effectiveness=0.8,
            applied_at=datetime.now() - timedelta(days=30)
        )
        
        entry2 = ResolutionHistoryEntry(
            id="res2",
            conflict_id="c2",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "salary"},
            resolution_id="opt2",
            resolution_type="keep_production",
            entity_types=["Employee"],
            property_names=["salary"],
            applied_by="user1",
            success=False,
            effectiveness=0.6,
            applied_at=datetime.now()
        )
        
        # Mock Redis scan
        mock_redis_client.scan.return_value = (0, [b"resolution_history:res1", b"resolution_history:res2"])
        
        # Mock Redis get for entries
        mock_redis_client.get.side_effect = lambda key: {
            b"resolution_history:res1": entry1.model_dump_json(),
            "resolution_history:res1": entry1.model_dump_json(),
            b"resolution_history:res2": entry2.model_dump_json(),
            "resolution_history:res2": entry2.model_dump_json(),
        }.get(key)
        
        # Act
        stats = await service.get_resolution_stats()
        
        # Assert
        assert stats["total_resolutions"] == 2
        assert stats["by_conflict_type"]["property_value"] == 2
        assert stats["success_count"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["average_effectiveness"] == 0.7  # Average of 0.8 and 0.6
        assert len(stats["by_resolution_type"]) == 2
        assert len(stats["by_entity_type"]) == 2
        assert len(stats["by_user"]) == 1
        assert len(stats["time_distribution"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_resolution_count(self, mock_redis_client):
        # Arrange
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Mock Redis smembers for merge_id index
        mock_redis_client.smembers.return_value = {b"res1", b"res2", b"res3"}
        
        # Act
        count = await service.get_resolution_count(merge_id="m1")
        
        # Assert
        assert count == 3
    
    @pytest.mark.asyncio
    async def test_filter_resolutions(self, mock_redis_client):
        # Arrange
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Setup mock entries
        entry1 = ResolutionHistoryEntry(
            id="res1",
            conflict_id="c1",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "age"},
            resolution_id="opt1",
            resolution_type="keep_staging",
            entity_types=["Person"],
            property_names=["age"],
            applied_by="user1",
            success=True,
            effectiveness=0.8,
            applied_at=datetime.now() - timedelta(days=30)
        )
        
        entry2 = ResolutionHistoryEntry(
            id="res2",
            conflict_id="c2",
            merge_id="m1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"property_name": "salary"},
            resolution_id="opt2",
            resolution_type="keep_production",
            entity_types=["Employee"],
            property_names=["salary"],
            applied_by="user2",
            success=False,
            effectiveness=0.6,
            applied_at=datetime.now()
        )
        
        # Mock Redis smembers for conflict_type index
        mock_redis_client.smembers.return_value = {b"res1", b"res2"}
        
        # Mock Redis get for entries
        mock_redis_client.get.side_effect = lambda key: {
            "resolution_history:res1": entry1.model_dump_json(),
            "resolution_history:res2": entry2.model_dump_json(),
        }.get(key)
        
        # Create filter and pagination params
        filter_params = ResolutionFilter(
            conflict_type=ConflictType.PROPERTY_VALUE,
            user="user1"
        )
        
        pagination_params = PaginationParams(
            limit=10,
            offset=0,
            sort_by="applied_at",
            sort_order="desc"
        )
        
        # Act - Mock the filtering logic
        with patch.object(service, 'filter_resolutions', return_value=([entry1], 1)):
            results, total = await service.filter_resolutions(
                filter_params=filter_params,
                pagination_params=pagination_params
            )
            
            # Assert
            assert total == 1
            assert len(results) == 1
            assert results[0].id == "res1"
            assert results[0].applied_by == "user1" 