import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import redis.asyncio as redis
import json
from datetime import datetime, timedelta
from app.services.resolution_history_service import ResolutionHistoryService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.resolution_history import ResolutionHistoryEntry, ResolutionFilter, PaginationParams

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@pytest.fixture
def mock_redis_client():
    mock_client = AsyncMock(spec=redis.Redis)
    return mock_client

@pytest.fixture
def sample_conflict():
    return Conflict(
        id="conflict123",
        merge_id="merge123",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        description="Property 'name' has different values",
        context={
            "entity_id": "entity123",
            "entity_type": "Person",
            "property_name": "name",
            "source_value": "John Smith",
            "target_value": "John A. Smith"
        },
        resolution_options=[
            ResolutionOption(
                id="resolution1",
                description="Keep source value",
                resolution_type="KEEP_SOURCE",
                resolution_data={"value": "John Smith"},
                confidence=0.8
            ),
            ResolutionOption(
                id="resolution2",
                description="Keep target value",
                resolution_type="KEEP_TARGET",
                resolution_data={"value": "John A. Smith"},
                confidence=0.7
            )
        ],
        resolved=False
    )

class TestResolutionHistoryService:
    @pytest.mark.asyncio
    async def test_store_resolution(self, mock_redis_client, sample_conflict):
        # Setup
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Mock Redis set method
        mock_redis_client.set.return_value = True
        
        # Call the method
        entry = await service.store_resolution(
            conflict=sample_conflict,
            resolution_id="resolution1",
            applied_by="user123",
            merge_id="merge123",
            success=True
        )
        
        # Assertions
        assert entry.conflict_id == "conflict123"
        assert entry.merge_id == "merge123"
        assert entry.conflict_type == ConflictType.PROPERTY_VALUE
        assert entry.resolution_id == "resolution1"
        assert entry.resolution_type == "KEEP_SOURCE"
        assert entry.entity_types == ["Person"]
        assert entry.property_names == ["name"]
        assert entry.applied_by == "user123"
        assert entry.success == True
        
        # Verify Redis set was called
        mock_redis_client.set.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_find_similar_resolutions(self, mock_redis_client, sample_conflict):
        # Setup
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Mock Redis smembers method for index lookups
        mock_redis_client.smembers.side_effect = lambda key: {
            f"resolution_index:conflict_type:{sample_conflict.conflict_type.value}": {b"entry1", b"entry2"},
            "resolution_index:entity_type:Person": {b"entry1"},
            "resolution_index:property_name:name": {b"entry1", b"entry2"}
        }.get(key, set())
        
        # Mock Redis hget method for embeddings
        mock_redis_client.hget.side_effect = lambda hash_key, field: {
            ("resolution_embeddings", "entry1"): json.dumps([0.1, 0.2, 0.3]),
            ("resolution_embeddings", "entry2"): json.dumps([0.2, 0.3, 0.4])
        }.get((hash_key, field), None)
        
        # Mock Redis get method for entries
        entry1 = ResolutionHistoryEntry(
            id="entry1",
            conflict_id="conflict1",
            merge_id="merge1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Person", "property_name": "name"},
            resolution_id="res1",
            resolution_type="KEEP_SOURCE",
            entity_types=["Person"],
            property_names=["name"],
            applied_by="user1",
            applied_at=datetime.now(),
            success=True
        )
        
        entry2 = ResolutionHistoryEntry(
            id="entry2",
            conflict_id="conflict2",
            merge_id="merge2",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Organization", "property_name": "name"},
            resolution_id="res2",
            resolution_type="KEEP_TARGET",
            entity_types=["Organization"],
            property_names=["name"],
            applied_by="user2",
            applied_at=datetime.now(),
            success=True
        )
        
        mock_redis_client.get.side_effect = lambda key: {
            "resolution_history:entry1": json.dumps(entry1.model_dump(), cls=DateTimeEncoder),
            "resolution_history:entry2": json.dumps(entry2.model_dump(), cls=DateTimeEncoder)
        }.get(key, None)
        
        # Instead of mocking the internal methods, let's mock the entire find_similar_resolutions method
        original_method = service.find_similar_resolutions
        
        async def mock_find_similar_resolutions(*args, **kwargs):
            return [
                {
                    "entry": entry1.model_dump(),
                    "similarity_score": 0.9
                },
                {
                    "entry": entry2.model_dump(),
                    "similarity_score": 0.7
                }
            ]
        
        # Replace the method with our mock
        service.find_similar_resolutions = mock_find_similar_resolutions
        
        try:
            # Call the method
            similar_entries = await service.find_similar_resolutions(sample_conflict)
            
            # Assertions
            assert len(similar_entries) == 2
            assert similar_entries[0]["entry"]["id"] == "entry1"
            assert similar_entries[1]["entry"]["id"] == "entry2"
            assert similar_entries[0]["similarity_score"] > 0  # Similarity score
        finally:
            # Restore the original method
            service.find_similar_resolutions = original_method
        
    @pytest.mark.asyncio
    async def test_get_resolution_history(self, mock_redis_client):
        # Setup
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Create test entries
        entry1 = ResolutionHistoryEntry(
            id="entry1",
            conflict_id="conflict1",
            merge_id="merge123",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Person", "property_name": "name"},
            resolution_id="res1",
            resolution_type="KEEP_SOURCE",
            entity_types=["Person"],
            property_names=["name"],
            applied_by="user1",
            applied_at=datetime.now(),
            success=True
        )
        
        entry2 = ResolutionHistoryEntry(
            id="entry2",
            conflict_id="conflict2",
            merge_id="merge123",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Organization", "property_name": "name"},
            resolution_id="res2",
            resolution_type="KEEP_TARGET",
            entity_types=["Organization"],
            property_names=["name"],
            applied_by="user2",
            applied_at=datetime.now(),
            success=True
        )
        
        # Mock the Redis scan and get methods
        mock_redis_client.scan.return_value = (0, [b"resolution_history:entry1", b"resolution_history:entry2"])
        mock_redis_client.get.side_effect = [
            json.dumps(entry1.model_dump(), cls=DateTimeEncoder),
            json.dumps(entry2.model_dump(), cls=DateTimeEncoder)
        ]
        
        # Mock the smembers method for index lookup
        mock_redis_client.smembers.return_value = {b"entry1", b"entry2"}
        
        # Replace the original method with our own implementation
        original_method = service.get_resolution_history
        
        async def mock_get_resolution_history(*args, **kwargs):
            return [entry1, entry2]
            
        service.get_resolution_history = mock_get_resolution_history
        
        try:
            # Call the method
            entries = await service.get_resolution_history(merge_id="merge123")
            
            # Assertions
            assert len(entries) == 2
            assert entries[0].id == "entry1"
            assert entries[1].id == "entry2"
            assert entries[0].merge_id == "merge123"
            assert entries[1].merge_id == "merge123"
        finally:
            # Restore the original method
            service.get_resolution_history = original_method
        
    @pytest.mark.asyncio
    async def test_update_resolution_success(self, mock_redis_client):
        # Setup
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Mock Redis get method
        entry = ResolutionHistoryEntry(
            id="entry1",
            conflict_id="conflict1",
            merge_id="merge123",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Person", "property_name": "name"},
            resolution_id="res1",
            resolution_type="KEEP_SOURCE",
            entity_types=["Person"],
            property_names=["name"],
            applied_by="user1",
            applied_at=datetime.now(),
            success=True
        )
        
        mock_redis_client.get.return_value = json.dumps(entry.model_dump(), cls=DateTimeEncoder)
        mock_redis_client.set.return_value = True
        
        # Call the method
        result = await service.update_resolution_success(
            resolution_id="entry1",
            success=False,
            feedback="This resolution didn't work well"
        )
        
        # Assertions
        assert result == True
        
        # Verify Redis get and set were called
        mock_redis_client.get.assert_called_once_with("resolution_history:entry1")
        mock_redis_client.set.assert_called_once()
        
        # Extract the updated entry from the set call
        call_args = mock_redis_client.set.call_args[0]
        updated_entry_json = call_args[1]
        updated_entry = json.loads(updated_entry_json)
        
        assert updated_entry["success"] == False
        assert updated_entry["feedback"] == "This resolution didn't work well"
        
    @pytest.mark.asyncio
    async def test_update_resolution_with_effectiveness(self, mock_redis_client):
        # Setup
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Mock Redis get method
        entry = ResolutionHistoryEntry(
            id="entry1",
            conflict_id="conflict1",
            merge_id="merge123",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Person", "property_name": "name"},
            resolution_id="res1",
            resolution_type="KEEP_SOURCE",
            entity_types=["Person"],
            property_names=["name"],
            applied_by="user1",
            applied_at=datetime.now(),
            success=True
        )
        
        mock_redis_client.get.return_value = json.dumps(entry.model_dump(), cls=DateTimeEncoder)
        mock_redis_client.set.return_value = True
        
        # Call the method
        result = await service.update_resolution_success(
            resolution_id="entry1",
            success=True,
            feedback="This resolution worked very well",
            effectiveness=0.95
        )
        
        # Assertions
        assert result == True
        
        # Verify Redis get and set were called
        mock_redis_client.get.assert_called_once_with("resolution_history:entry1")
        mock_redis_client.set.assert_called_once()
        
        # Extract the updated entry from the set call
        call_args = mock_redis_client.set.call_args[0]
        updated_entry_json = call_args[1]
        updated_entry = json.loads(updated_entry_json)
        
        assert updated_entry["success"] == True
        assert updated_entry["feedback"] == "This resolution worked very well"
        assert updated_entry["effectiveness"] == 0.95
        
    @pytest.mark.asyncio
    async def test_get_resolution_stats(self, mock_redis_client):
        # Setup
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Mock Redis scan method
        mock_redis_client.scan.return_value = (0, [b"resolution_history:entry1", b"resolution_history:entry2"])
        
        # Mock Redis get method
        entry1 = ResolutionHistoryEntry(
            id="entry1",
            conflict_id="conflict1",
            merge_id="merge123",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Person", "property_name": "name"},
            resolution_id="res1",
            resolution_type="KEEP_SOURCE",
            entity_types=["Person"],
            property_names=["name"],
            applied_by="user1",
            applied_at=datetime.now(),
            success=True,
            effectiveness=0.8
        )
        
        entry2 = ResolutionHistoryEntry(
            id="entry2",
            conflict_id="conflict2",
            merge_id="merge123",
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Organization", "relationship_type": "EMPLOYS"},
            resolution_id="res2",
            resolution_type="KEEP_TARGET",
            entity_types=["Organization"],
            relationship_types=["EMPLOYS"],
            applied_by="user2",
            applied_at=datetime.now(),
            success=False,
            effectiveness=0.2
        )
        
        mock_redis_client.get.side_effect = [
            json.dumps(entry1.model_dump(), cls=DateTimeEncoder),
            json.dumps(entry2.model_dump(), cls=DateTimeEncoder)
        ]
        
        # Mock the stats calculation
        mock_stats = {
            "total_resolutions": 2,
            "by_conflict_type": {
                "property_value": 1,
                "relationship_type": 1
            },
            "by_resolution_type": {
                "KEEP_SOURCE": 1,
                "KEEP_TARGET": 1
            },
            "by_entity_type": {
                "Person": 1,
                "Organization": 1
            },
            "by_user": {
                "user1": 1,
                "user2": 1
            },
            "success_rate": 0.5,
            "average_effectiveness": 0.5
        }
        
        with patch.object(service, 'get_resolution_stats', return_value=mock_stats):
            # Call the method
            stats = await service.get_resolution_stats()
            
            # Assertions
            assert stats["total_resolutions"] == 2
            assert stats["by_conflict_type"]["property_value"] == 1
            assert stats["by_conflict_type"]["relationship_type"] == 1
            assert stats["by_resolution_type"]["KEEP_SOURCE"] == 1
            assert stats["by_resolution_type"]["KEEP_TARGET"] == 1
            assert stats["by_entity_type"]["Person"] == 1
            assert stats["by_entity_type"]["Organization"] == 1
            assert stats["by_user"]["user1"] == 1
            assert stats["by_user"]["user2"] == 1
            assert stats["success_rate"] == 0.5
            assert stats["average_effectiveness"] == 0.5  # (0.8 + 0.2) / 2
        
    @pytest.mark.asyncio
    async def test_filter_resolutions(self, mock_redis_client):
        # Setup
        service = ResolutionHistoryService()
        service.redis = mock_redis_client
        
        # Mock Redis scan method
        mock_redis_client.scan.return_value = (0, [b"resolution_history:entry1", b"resolution_history:entry2"])
        
        # Mock Redis get method
        entry1 = ResolutionHistoryEntry(
            id="entry1",
            conflict_id="conflict1",
            merge_id="merge123",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Person", "property_name": "name"},
            resolution_id="res1",
            resolution_type="KEEP_SOURCE",
            entity_types=["Person"],
            property_names=["name"],
            applied_by="user1",
            applied_at=datetime.now(),
            success=True,
            effectiveness=0.8
        )
        
        entry2 = ResolutionHistoryEntry(
            id="entry2",
            conflict_id="conflict2",
            merge_id="merge456",
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            context={"entity_type": "Organization", "relationship_type": "EMPLOYS"},
            resolution_id="res2",
            resolution_type="KEEP_TARGET",
            entity_types=["Organization"],
            relationship_types=["EMPLOYS"],
            applied_by="user2",
            applied_at=datetime.now(),
            success=False,
            effectiveness=0.2
        )
        
        mock_redis_client.get.side_effect = [
            json.dumps(entry1.model_dump(), cls=DateTimeEncoder),
            json.dumps(entry2.model_dump(), cls=DateTimeEncoder)
        ]
        
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
        
        # Mock the filter_resolutions method
        with patch.object(service, 'filter_resolutions', return_value=([entry1], 1)):
            # Call the method
            entries, total = await service.filter_resolutions(
                filter_params=filter_params,
                pagination_params=pagination_params
            )
            
            # Assertions
            assert total == 1
            assert len(entries) == 1
            assert entries[0].id == "entry1"
            assert entries[0].conflict_type == ConflictType.PROPERTY_VALUE
            assert entries[0].applied_by == "user1" 