"""Integration tests for the Resolution Learning Service"""

import pytest
import asyncio
from datetime import datetime, timedelta
import uuid
import random
from typing import List, Dict, Any, Optional

from app.services.merge.resolution_learning import ResolutionLearningService, ResolutionLearningConfig
from app.services.storage.vector_storage import ResolutionPattern, QdrantResolutionStorage
from app.services.merge.resolution_search import ResolutionPatternSearchService
from app.schemas.conflicts import Conflict, ConflictType, ResolutionOption, ConflictSeverity
import os
pytestmark = pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Integration tests are skipped by default. Set INTEGRATION_TESTS=1 to run."
)

@pytest.fixture
async def vector_storage():
    """Create a real QdrantResolutionStorage instance for testing"""
    # Use a test-specific collection name to avoid conflicts
    collection_name = f"test_resolution_learning_{uuid.uuid4().hex[:8]}"
    
    # Create storage instance
    storage = QdrantResolutionStorage(
        collection_name=collection_name,
        vector_size=768  # Use 768 to match the embedding dimension of the model
    )
    
    # Return the storage instance
    yield storage
    
    # Cleanup: Delete the collection after tests
    try:
        storage.client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print(f"Error cleaning up test collection: {str(e)}")


@pytest.fixture
async def search_service(vector_storage):
    """Create a real ResolutionPatternSearchService instance for testing"""
    service = ResolutionPatternSearchService(vector_storage=vector_storage)
    return service


@pytest.fixture
def learning_config():
    """Test configuration for the learning service"""
    return {
        "high_confidence_threshold": 0.85,
        "medium_confidence_threshold": 0.70,
        "rate_limit_per_minute": 10,
        "blacklisted_patterns": [
            {
                "conflict_type": "PROPERTY_VALUE",
                "entity_types": ["Product"],
                "property_names": ["price"],
                "resolution_strategy": "custom_value"
            }
        ],
        "weight_factors": {
            "similarity_score": 0.6,
            "success_rate": 0.3,
            "recency": 0.1
        }
    }


@pytest.fixture
async def learning_service(vector_storage, search_service, learning_config):
    """Create a real ResolutionLearningService instance for testing"""
    service = ResolutionLearningService(
        vector_storage=vector_storage,
        search_service=search_service,
        config=learning_config
    )
    return service


@pytest.fixture
def sample_conflicts():
    """Generate sample conflicts for testing"""
    conflicts = []
    
    # Email conflicts
    for i in range(5):
        conflicts.append(Conflict(
            id=f"email_conflict_{i}",
            merge_id=f"merge_1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            entity_type="Customer",
            property_name="email",
            staging_value=f"new{i}@example.com",
            production_value=f"old{i}@example.com",
            severity=ConflictSeverity.MINOR,
            description=f"Email property value conflict {i}",
            context={
                "entity_id": f"customer_{i}",
                "entity_type": "Customer"
            },
            resolution_options=[
                ResolutionOption(
                    id=f"email_staging_{i}",
                    resolution_type="prefer_staging",
                    description="Keep staging value",
                    confidence=0.8
                ),
                ResolutionOption(
                    id=f"email_prod_{i}",
                    resolution_type="prefer_production",
                    description="Keep production value",
                    confidence=0.6
                )
            ]
        ))
    
    # Price conflicts
    for i in range(5):
        conflicts.append(Conflict(
            id=f"price_conflict_{i}",
            merge_id=f"merge_2",
            conflict_type=ConflictType.PROPERTY_VALUE,
            entity_type="Product",
            property_name="price",
            staging_value=str(10.99 + i),
            production_value=str(9.99 + i),
            severity=ConflictSeverity.MINOR,
            description=f"Price property value conflict {i}",
            context={
                "entity_id": f"product_{i}",
                "entity_type": "Product"
            },
            resolution_options=[
                ResolutionOption(
                    id=f"price_staging_{i}",
                    resolution_type="prefer_staging",
                    description="Keep staging value",
                    confidence=0.7
                ),
                ResolutionOption(
                    id=f"price_prod_{i}",
                    resolution_type="prefer_production",
                    description="Keep production value",
                    confidence=0.7
                ),
                ResolutionOption(
                    id=f"price_custom_{i}",
                    resolution_type="custom_value",
                    description="Use custom value",
                    confidence=0.6,
                    custom_value=str(11.99 + i)
                )
            ]
        ))
    
    # Name conflicts
    for i in range(5):
        conflicts.append(Conflict(
            id=f"name_conflict_{i}",
            merge_id=f"merge_3",
            conflict_type=ConflictType.PROPERTY_VALUE,
            entity_type="Customer",
            property_name="name",
            staging_value=f"New Name {i}",
            production_value=f"Old Name {i}",
            severity=ConflictSeverity.MINOR,
            description=f"Name property value conflict {i}",
            context={
                "entity_id": f"customer_{i}",
                "entity_type": "Customer"
            },
            resolution_options=[
                ResolutionOption(
                    id=f"name_staging_{i}",
                    resolution_type="prefer_staging",
                    description="Keep staging value",
                    confidence=0.8
                ),
                ResolutionOption(
                    id=f"name_prod_{i}",
                    resolution_type="prefer_production",
                    description="Keep production value",
                    confidence=0.6
                )
            ]
        ))
    
    return conflicts


@pytest.fixture
async def sample_patterns(vector_storage, sample_conflicts):
    """Create and store sample resolution patterns for testing"""
    patterns = []
    now = datetime.now()
    
    # Create patterns for email conflicts (prefer staging)
    for i in range(3):
        pattern = ResolutionPattern(
            id=f"email_pattern_{i}",
            conflict_type="property_value",
            entity_types=["Customer"],
            property_names=["email"],
            resolution_strategy="prefer_staging",
            resolution_data={},
            confidence=0.9,
            original_conflict_id=f"original_email_conflict_{i}",
            original_merge_id="merge_1",
            created_at=now - timedelta(days=i)
        )
        patterns.append(pattern)
        await vector_storage.store_resolution(pattern)
    
    # Create patterns for price conflicts (prefer production)
    for i in range(3):
        pattern = ResolutionPattern(
            id=f"price_pattern_{i}",
            conflict_type="property_value",
            entity_types=["Product"],
            property_names=["price"],
            resolution_strategy="prefer_production",
            resolution_data={},
            confidence=0.85,
            original_conflict_id=f"original_price_conflict_{i}",
            original_merge_id="merge_1",
            created_at=now - timedelta(days=i+1)
        )
        patterns.append(pattern)
        await vector_storage.store_resolution(pattern)
    
    # Create patterns for name conflicts (custom value)
    for i in range(3):
        pattern = ResolutionPattern(
            id=f"name_pattern_{i}",
            conflict_type="property_value",
            entity_types=["Customer"],
            property_names=["name"],
            resolution_strategy="custom_value",
            resolution_data={"value": f"Custom Name {i}"},
            confidence=0.8,
            original_conflict_id=f"original_name_conflict_{i}",
            original_merge_id="merge_2",
            created_at=now - timedelta(days=i+2)
        )
        patterns.append(pattern)
        await vector_storage.store_resolution(pattern)
    
    # Create a blacklisted pattern
    blacklisted = ResolutionPattern(
        id="blacklisted_pattern",
        conflict_type="property_value",
        entity_types=["Product"],
        property_names=["price"],
        resolution_strategy="custom_value",
        resolution_data={"value": 99.99},
        confidence=0.95,
        original_conflict_id="original_blacklisted_conflict",
        original_merge_id="merge_3",
        created_at=now - timedelta(days=5)
    )
    patterns.append(blacklisted)
    await vector_storage.store_resolution(blacklisted)
    
    return patterns


@pytest.mark.asyncio
async def test_end_to_end_learning_flow(learning_service, sample_conflicts, sample_patterns):
    """Test the end-to-end learning flow"""
    # Get a sample conflict
    email_conflict = next(c for c in sample_conflicts if c.property_name == "email")
    
    # Process the conflict
    auto_resolution, suggestions, was_auto_applied = await learning_service.process_conflict(email_conflict)
    
    # Verify that a resolution was automatically applied
    assert was_auto_applied is True
    assert auto_resolution is not None
    assert auto_resolution.resolution_type == "prefer_staging"
    assert "Automatically applied" in auto_resolution.description
    
    # Verify that the resolution has the correct metadata
    assert auto_resolution.resolution_data["learned"] is True
    assert "pattern_id" in auto_resolution.resolution_data
    assert "original_conflict_id" in auto_resolution.resolution_data
    
    # Track the outcome as successful
    await learning_service.track_resolution_outcome(
        conflict_id=email_conflict.id,
        resolution_id=auto_resolution.id,
        pattern_id=auto_resolution.resolution_data["pattern_id"],
        success=True,
        feedback="Good resolution"
    )


@pytest.mark.asyncio
async def test_learning_from_multiple_examples(learning_service, sample_conflicts, sample_patterns):
    """Test learning from multiple similar examples"""
    # Get a sample conflict
    name_conflict = next(c for c in sample_conflicts if c.property_name == "name")
    
    # Process the conflict
    auto_resolution, suggestions, was_auto_applied = await learning_service.process_conflict(name_conflict)
    
    # Verify that a resolution was automatically applied or suggested
    if was_auto_applied:
        assert auto_resolution is not None
        assert auto_resolution.resolution_type == "custom_value"
        assert "Custom Name" in auto_resolution.custom_value
    else:
        # If not auto-applied, should have suggestions
        assert len(suggestions) > 0
        assert any(s.resolution_type == "custom_value" for s in suggestions)


@pytest.mark.asyncio
async def test_blacklisted_pattern_handling(learning_service, sample_conflicts):
    """Test that blacklisted patterns are not applied"""
    # Get a price conflict (which has a blacklisted pattern)
    price_conflict = next(c for c in sample_conflicts if c.property_name == "price")
    
    # Process the conflict
    auto_resolution, suggestions, was_auto_applied = await learning_service.process_conflict(price_conflict)
    
    # Verify that no custom_value resolution was applied (it's blacklisted)
    if was_auto_applied:
        assert auto_resolution.resolution_type != "custom_value"
    
    # Verify that no custom_value suggestions were provided
    assert not any(s.resolution_type == "custom_value" for s in suggestions)


@pytest.mark.asyncio
async def test_confidence_threshold_behavior(learning_service, sample_conflicts, sample_patterns):
    """Test behavior with different confidence thresholds"""
    # Get a sample conflict
    email_conflict = next(c for c in sample_conflicts if c.property_name == "email")
    
    # Test with high threshold (should not auto-apply)
    await learning_service.update_config({"high_confidence_threshold": 0.99})
    auto_high, suggestions_high, applied_high = await learning_service.process_conflict(email_conflict)
    
    # Should not auto-apply with very high threshold
    assert applied_high is False
    assert auto_high is None
    assert len(suggestions_high) > 0  # Should still have suggestions
    
    # Test with low threshold (should auto-apply)
    await learning_service.update_config({"high_confidence_threshold": 0.70})
    auto_low, suggestions_low, applied_low = await learning_service.process_conflict(email_conflict)
    
    # Should auto-apply with low threshold
    assert applied_low is True
    assert auto_low is not None


@pytest.mark.asyncio
async def test_rate_limiting(learning_service, sample_conflicts, sample_patterns):
    """Test rate limiting functionality"""
    # Set a very low rate limit
    await learning_service.update_config({"rate_limit_per_minute": 2})
    
    # Get email conflicts
    email_conflicts = [c for c in sample_conflicts if c.property_name == "email"][:3]
    
    # Process first conflict - should succeed
    auto1, suggestions1, applied1 = await learning_service.process_conflict(email_conflicts[0])
    assert applied1 is True
    
    # Process second conflict - should succeed
    auto2, suggestions2, applied2 = await learning_service.process_conflict(email_conflicts[1])
    assert applied2 is True
    
    # Process third conflict - should be rate limited
    auto3, suggestions3, applied3 = await learning_service.process_conflict(email_conflicts[2])
    assert applied3 is False


@pytest.mark.asyncio
async def test_blacklist_management_integration(learning_service):
    """Test blacklist management in an integration context"""
    # Add a new pattern to blacklist
    new_pattern = {
        "conflict_type": "PROPERTY_VALUE",
        "entity_types": ["Customer"],
        "property_names": ["address"],
        "resolution_strategy": "prefer_staging"
    }
    
    await learning_service.add_to_blacklist(new_pattern)
    
    # Verify pattern was added
    assert any(
        p.get("property_names") == ["address"] 
        for p in learning_service.config.blacklisted_patterns
    )
    
    # Remove the pattern
    result = await learning_service.remove_from_blacklist(new_pattern)
    
    # Verify pattern was removed
    assert result is True
    assert not any(
        p.get("property_names") == ["address"] 
        for p in learning_service.config.blacklisted_patterns
    )


@pytest.mark.asyncio
async def test_learning_service_performance(learning_service, sample_conflicts, sample_patterns):
    """Test performance of the learning service with multiple conflicts"""
    # Process multiple conflicts in parallel
    tasks = []
    for conflict in sample_conflicts[:10]:  # Process 10 conflicts
        tasks.append(learning_service.process_conflict(conflict))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Verify results
    auto_applied_count = sum(1 for _, _, applied in results if applied)
    suggestion_count = sum(len(suggestions) for _, suggestions, _ in results)
    
    # Should have some auto-applied resolutions
    assert auto_applied_count > 0
    # We may not always have suggestions if auto-applied resolutions are found
    # So we'll check the total of auto-applied + suggestions instead
    assert auto_applied_count + suggestion_count > 0 