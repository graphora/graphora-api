import pytest
import redis
import json
import os
from datetime import datetime
from app.services.resolution_history_service import ResolutionHistoryService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.resolution_history import ResolutionHistoryEntry
from app.config import settings

# Skip these tests if INTEGRATION_TESTS env var is not set
pytestmark = pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Integration tests are skipped by default"
)

@pytest.fixture
def resolution_history_service():
    # Use a dedicated test database for integration tests
    # to avoid interfering with production data
    test_redis_url = settings.REDIS_URL.replace("db=0", "db=15")
    
    # Create service with real Redis connection
    service = ResolutionHistoryService()
    service.redis = redis.Redis.from_url(test_redis_url)
    
    # Clean up test database before tests
    service.redis.flushdb()
    
    yield service
    
    # Clean up after tests
    service.redis.flushdb()

@pytest.fixture
def sample_conflict():
    return Conflict(
        id="conflict_integration",
        merge_id="merge_integration",
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

class TestResolutionHistoryIntegration:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_resolution(self, resolution_history_service, sample_conflict):
        # Arrange
        service = resolution_history_service
        
        # Act - Store resolution
        entry = await service.store_resolution(
            conflict=sample_conflict,
            resolution_id="opt1",
            applied_by="integration_test_user",
            merge_id="merge_integration",
            success=True
        )
        
        # Assert storage result
        assert entry.id is not None
        assert entry.conflict_id == "conflict_integration"
        assert entry.resolution_type == "keep_staging"
        
        # Act - Get resolution history
        history = await service.get_resolution_history(
            merge_id="merge_integration"
        )
        
        # Assert retrieval
        assert len(history) == 1
        assert history[0].id == entry.id
        assert history[0].conflict_type == ConflictType.PROPERTY_VALUE
        assert history[0].entity_types == ["Person"]
        assert history[0].property_names == ["age"]
    
    @pytest.mark.asyncio
    async def test_find_similar_resolutions(self, resolution_history_service, sample_conflict):
        # Arrange
        service = resolution_history_service
        
        # Store multiple resolutions
        entry1 = await service.store_resolution(
            conflict=sample_conflict,
            resolution_id="opt1",
            applied_by="integration_test_user",
            merge_id="merge_integration",
            success=True
        )
        
        # Create slightly different conflict
        similar_conflict = Conflict(
            id="conflict_similar",
            merge_id="merge_integration",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s2"],
            production_ids=["p2"],
            description="Property 'salary' has different values",
            context={
                "property_name": "salary",
                "staging_value": 50000,
                "production_value": 52000,
                "entity_type": "Employee"
            },
            resolution_options=[
                ResolutionOption(
                    id="opt3",
                    description="Keep staging value: 50000",
                    resolution_type="keep_staging",
                    resolution_data={"property_name": "salary"},
                    confidence=0.5
                )
            ]
        )
        
        entry2 = await service.store_resolution(
            conflict=similar_conflict,
            resolution_id="opt3",
            applied_by="integration_test_user",
            merge_id="merge_integration",
            success=True
        )
        
        # Act - Find similar resolutions
        similar_results = await service.find_similar_resolutions(
            conflict=sample_conflict,
            limit=5
        )
        
        # Assert
        assert len(similar_results) > 0
        
        # Most similar should be entry1 (exact match)
        assert similar_results[0]["entry"]["id"] == entry1.id
        assert similar_results[0]["similarity_score"] > 0.8  # High similarity score
    
    @pytest.mark.asyncio
    async def test_update_resolution_success(self, resolution_history_service, sample_conflict):
        # Arrange
        service = resolution_history_service
        
        # Store resolution
        entry = await service.store_resolution(
            conflict=sample_conflict,
            resolution_id="opt1",
            applied_by="integration_test_user",
            merge_id="merge_integration",
            success=True
        )
        
        # Act - Update success status
        result = await service.update_resolution_success(
            resolution_id=entry.id,
            success=False,
            feedback="Integration test feedback"
        )
        
        # Assert update result
        assert result is True
        
        # Verify updated data
        history = await service.get_resolution_history(merge_id="merge_integration")
        assert len(history) == 1
        assert history[0].success is False
        assert history[0].feedback == "Integration test feedback"
    
    @pytest.mark.asyncio
    async def test_get_resolution_stats(self, resolution_history_service, sample_conflict):
        # Arrange
        service = resolution_history_service
        
        # Store several resolutions
        for i in range(5):
            # Alternate success/failure
            success = i % 2 == 0
            
            await service.store_resolution(
                conflict=sample_conflict,
                resolution_id="opt1",
                applied_by=f"integration_test_user_{i}",
                merge_id=f"merge_integration_{i}",
                success=success
            )
        
        # Act - Get statistics
        stats = await service.get_resolution_stats()
        
        # Assert
        assert stats["total_resolutions"] == 5
        assert stats["by_conflict_type"]["property_value"] == 5
        assert stats["success_count"] == 3  # Entries 0, 2, 4 are success
        assert stats["success_rate"] == 0.6  # 3/5 = 0.6 