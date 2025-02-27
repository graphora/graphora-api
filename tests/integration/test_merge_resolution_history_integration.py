import pytest
import os
import redis
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.merge.service import MergeService
from app.services.resolution_history_service import ResolutionHistoryService
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption
from app.schemas.resolution_history import ResolutionHistoryEntry
from app.config import settings
from app.storage.graph import GraphStorageInterface
from app.services.merge.progress import ProgressTracker

# Skip these tests if INTEGRATION_TESTS env var is not set
pytestmark = pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Integration tests are skipped by default"
)

@pytest.fixture
def mock_storage():
    """Mock storage interfaces"""
    storage = AsyncMock(spec=GraphStorageInterface)
    production_storage = AsyncMock(spec=GraphStorageInterface)
    return storage, production_storage

@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker"""
    return AsyncMock(spec=ProgressTracker)

@pytest.fixture
def resolution_history_service():
    """Create a real ResolutionHistoryService with test Redis database"""
    # Use a dedicated test database for integration tests
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
    """Create a sample conflict for testing"""
    return Conflict(
        id="conflict_merge_integration",
        merge_id="merge_integration",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MAJOR,
        staging_ids=["s1"],
        production_ids=["p1"],
        description="Property 'email' has different values",
        context={
            "property_name": "email",
            "staging_value": "john.smith@example.com",
            "production_value": "j.smith@example.com",
            "entity_type": "Person"
        },
        resolution_options=[
            ResolutionOption(
                id="opt_merge_1",
                description="Keep staging value: john.smith@example.com",
                resolution_type="keep_staging",
                resolution_data={"property_name": "email"},
                confidence=0.8
            ),
            ResolutionOption(
                id="opt_merge_2",
                description="Keep production value: j.smith@example.com",
                resolution_type="keep_production",
                resolution_data={"property_name": "email"},
                confidence=0.5
            )
        ]
    )

@pytest.fixture
def merge_service(mock_storage, mock_progress_tracker, resolution_history_service):
    """Create a MergeService with real ResolutionHistoryService but mocked storage"""
    storage, production_storage = mock_storage
    
    # Create the service
    service = MergeService(
        storage=storage,
        production_storage=production_storage,
        progress_tracker=mock_progress_tracker
    )
    
    # Replace the resolution_history with our test instance
    service.resolution_history = resolution_history_service
    
    # Mock the resolution applicator
    service.resolution_applicator.apply_resolution = AsyncMock()
    service.resolution_applicator.apply_resolution.return_value = {
        "applied": True,
        "verification": {"verified": True},
        "changes": {"action": "updated_production"}
    }
    
    # Mock get_conflict method
    service.get_conflict = AsyncMock()
    
    # Mock _update_conflict method
    service._update_conflict = AsyncMock()
    
    return service

class TestMergeResolutionHistoryIntegration:
    @pytest.mark.asyncio
    async def test_apply_resolution_stores_in_history(self, merge_service, sample_conflict):
        """Test that applying a resolution stores it in the resolution history"""
        # Arrange
        service = merge_service
        service.get_conflict.return_value = sample_conflict
        
        # Act
        result = await service.apply_conflict_resolution(
            merge_id="merge_integration",
            conflict_id="conflict_merge_integration",
            resolution_id="opt_merge_1",
            resolved_by="integration_test_user"
        )
        
        # Assert
        assert result["applied"] is True
        
        # Verify resolution was stored in history
        history = await service.resolution_history.get_resolution_history(
            merge_id="merge_integration"
        )
        
        assert len(history) == 1
        assert history[0].conflict_id == "conflict_merge_integration"
        assert history[0].resolution_type == "keep_staging"
        assert history[0].applied_by == "integration_test_user"
    
    @pytest.mark.asyncio
    async def test_get_resolution_suggestions_from_history(self, merge_service, sample_conflict):
        """Test that resolution suggestions are retrieved from history"""
        # Arrange
        service = merge_service
        service.get_conflict.return_value = sample_conflict
        
        # First store a resolution
        await service.apply_conflict_resolution(
            merge_id="merge_integration",
            conflict_id="conflict_merge_integration",
            resolution_id="opt_merge_1",
            resolved_by="integration_test_user"
        )
        
        # Create a similar conflict
        similar_conflict = sample_conflict.model_copy()
        similar_conflict.id = "conflict_similar"
        similar_conflict.context["staging_value"] = "john.doe@example.com"
        similar_conflict.context["production_value"] = "j.doe@example.com"
        
        # Set up get_conflict to return the similar conflict
        service.get_conflict.return_value = similar_conflict
        
        # Act
        suggestions = await service.get_resolution_suggestions(
            merge_id="merge_integration",
            conflict_id="conflict_similar"
        )
        
        # Assert
        assert len(suggestions) > 0
        assert suggestions[0]["resolution_type"] == "keep_staging"
        assert suggestions[0]["similarity_score"] > 0.5
    
    @pytest.mark.asyncio
    async def test_resolution_feedback_updates_history(self, merge_service, sample_conflict):
        """Test that feedback on resolutions updates the history entry"""
        # Arrange
        service = merge_service
        service.get_conflict.return_value = sample_conflict
        
        # First store a resolution
        await service.apply_conflict_resolution(
            merge_id="merge_integration",
            conflict_id="conflict_merge_integration",
            resolution_id="opt_merge_1",
            resolved_by="integration_test_user"
        )
        
        # Get the history entry
        history = await service.resolution_history.get_resolution_history(
            merge_id="merge_integration"
        )
        resolution_id = history[0].id
        
        # Act - Update with feedback
        updated = await service.resolution_history.update_resolution_success(
            resolution_id=resolution_id,
            success=False,
            feedback="This resolution didn't work as expected"
        )
        
        # Assert
        assert updated is True
        
        # Verify the history was updated
        updated_history = await service.resolution_history.get_resolution_history(
            merge_id="merge_integration"
        )
        
        assert len(updated_history) == 1
        assert updated_history[0].id == resolution_id
        assert updated_history[0].success is False
        assert updated_history[0].feedback == "This resolution didn't work as expected"
    
    @pytest.mark.asyncio
    async def test_multiple_resolutions_affect_stats(self, merge_service, sample_conflict):
        """Test that multiple resolutions are reflected in statistics"""
        # Arrange
        service = merge_service
        
        # Store multiple resolutions with different success values
        for i in range(5):
            # Create a new conflict for each iteration to avoid "already resolved" error
            current_conflict = Conflict(
                id=f"conflict_merge_integration_{i}",
                merge_id=f"merge_integration_{i}",
                conflict_type=ConflictType.PROPERTY_VALUE,
                severity=ConflictSeverity.MAJOR,
                staging_ids=["s1"],
                production_ids=["p1"],
                description=f"Property 'email_{i}' has different values",
                context={
                    "property_name": f"email_{i}",
                    "staging_value": f"john.smith{i}@example.com",
                    "production_value": f"j.smith{i}@example.com",
                    "entity_type": "Person"
                },
                resolution_options=[
                    ResolutionOption(
                        id="opt_merge_1",
                        description=f"Keep staging value: john.smith{i}@example.com",
                        resolution_type="keep_staging",
                        resolution_data={"property_name": f"email_{i}"},
                        confidence=0.8
                    ),
                    ResolutionOption(
                        id="opt_merge_2",
                        description=f"Keep production value: j.smith{i}@example.com",
                        resolution_type="keep_production",
                        resolution_data={"property_name": f"email_{i}"},
                        confidence=0.5
                    )
                ],
                resolved=False  # Ensure the conflict is not already resolved
            )
            
            # Set up get_conflict to return the current conflict
            service.get_conflict.return_value = current_conflict
            
            # Alternate between resolution options
            resolution_id = "opt_merge_1" if i % 2 == 0 else "opt_merge_2"
            
            # Apply resolution
            await service.apply_conflict_resolution(
                merge_id=f"merge_integration_{i}",
                conflict_id=f"conflict_merge_integration_{i}",
                resolution_id=resolution_id,
                resolved_by=f"integration_test_user_{i}"
            )
            
            # Get the history entry
            history = await service.resolution_history.get_resolution_history(
                merge_id=f"merge_integration_{i}"
            )
            
            # Update success for odd-numbered resolutions
            if i % 2 == 1:
                await service.resolution_history.update_resolution_success(
                    resolution_id=history[0].id,
                    success=False,
                    feedback=f"Resolution {i} failed"
                )
        
        # Act - Get statistics
        stats = await service.resolution_history.get_resolution_stats()
        
        # Assert
        assert stats["total_resolutions"] == 5
        assert stats["by_conflict_type"]["property_value"] == 5
        assert stats["success_count"] == 3  # Even-numbered resolutions (0, 2, 4) are successful
        assert stats["success_rate"] == 0.6  # 3/5 = 0.6 