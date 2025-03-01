"""Performance tests for the Resolution Learning Service"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
import uuid
import random
from typing import List, Dict, Any, Optional

from app.services.merge.resolution_learning import ResolutionLearningService, ResolutionLearningConfig
from app.services.storage.vector_storage import ResolutionPattern, QdrantResolutionStorage
from app.services.merge.resolution_search import ResolutionPatternSearchService
from app.schemas.conflicts import Conflict, ConflictType, ResolutionOption, ConflictSeverity


@pytest.fixture
def mock_vector_storage():
    """Mock QdrantResolutionStorage with performance-optimized behavior"""
    class MockVectorStorage:
        async def search_similar_resolutions(self, conflict, top_k=10, score_threshold=0.7, filter_by_conflict_type=True, additional_filters=None):
            # Simulate search delay based on top_k
            await asyncio.sleep(0.01 * min(top_k, 10))
            
            # Generate mock results
            results = []
            for i in range(min(top_k, 20)):
                pattern = ResolutionPattern(
                    id=f"pattern_{i}",
                    conflict_type=conflict.conflict_type.value,
                    entity_types=[conflict.entity_type] if conflict.entity_type else ["Entity"],
                    property_names=[conflict.property_name] if conflict.property_name else None,
                    resolution_strategy="prefer_staging" if i % 3 == 0 else "prefer_production" if i % 3 == 1 else "custom_value",
                    resolution_data={"value": "test_value"} if i % 3 == 2 else {},
                    confidence=0.9 - (i * 0.02),
                    original_conflict_id=f"original_conflict_{i}",
                    original_merge_id="merge_1",
                    created_at=datetime.now() - timedelta(days=i % 30)
                )
                similarity = 0.95 - (i * 0.03)
                if similarity >= score_threshold:
                    results.append((pattern, similarity))
            
            return results
    
    return MockVectorStorage()


@pytest.fixture
def mock_search_service(mock_vector_storage):
    """Mock ResolutionPatternSearchService with performance-optimized behavior"""
    class MockSearchService:
        def __init__(self, vector_storage):
            self.vector_storage = vector_storage
        
        async def find_similar_resolutions(self, conflict, limit=5, filters=None):
            return await self.vector_storage.search_similar_resolutions(
                conflict=conflict,
                top_k=limit,
                score_threshold=0.7,
                filter_by_conflict_type=True,
                additional_filters=filters
            )
    
    return MockSearchService(mock_vector_storage)


@pytest.fixture
def learning_service(mock_vector_storage, mock_search_service):
    """Create a ResolutionLearningService with mocked dependencies for performance testing"""
    config = {
        "high_confidence_threshold": 0.85,
        "medium_confidence_threshold": 0.70,
        "rate_limit_per_minute": 100,  # High limit for performance testing
        "blacklisted_patterns": [
            {
                "conflict_type": "PROPERTY_VALUE",
                "entity_types": ["Product"],
                "property_names": ["price"],
                "resolution_strategy": "custom_value"
            }
        ]
    }
    
    service = ResolutionLearningService(
        vector_storage=mock_vector_storage,
        search_service=mock_search_service,
        config=config
    )
    
    # Override _get_pattern_success_rate for performance testing
    async def mock_success_rate(pattern_id):
        # Deterministic but varied success rates based on pattern_id
        return 0.5 + (hash(pattern_id) % 50) / 100
    
    service._get_pattern_success_rate = mock_success_rate
    
    return service


@pytest.fixture
def generate_conflicts():
    """Generate a large number of conflicts for performance testing"""
    def _generate(n=100):
        conflicts = []
        
        entity_types = ["Customer", "Product", "Order", "Invoice", "Shipment"]
        property_names = ["name", "email", "price", "quantity", "address", "status", "date"]
        
        for i in range(n):
            entity_type = random.choice(entity_types)
            property_name = random.choice(property_names)
            
            conflicts.append(Conflict(
                id=f"conflict_{i}",
                merge_id=f"merge_{i // 10}",  # Add merge_id field, grouping conflicts by merge
                conflict_type=ConflictType.PROPERTY_VALUE,
                entity_type=entity_type,
                property_name=property_name,
                staging_value=f"staging_value_{i}",
                production_value=f"production_value_{i}",
                severity=ConflictSeverity.MINOR,
                description=f"{property_name} property value conflict for {entity_type}",
                context={
                    "entity_id": f"{entity_type.lower()}_{i}",
                    "entity_type": entity_type
                },
                resolution_options=[
                    ResolutionOption(
                        id=f"option_staging_{i}",
                        resolution_type="prefer_staging",
                        description="Keep staging value",
                        confidence=0.8
                    ),
                    ResolutionOption(
                        id=f"option_prod_{i}",
                        resolution_type="prefer_production",
                        description="Keep production value",
                        confidence=0.7
                    )
                ]
            ))
        
        return conflicts
    
    return _generate


class TestResolutionLearningPerformance:
    """Performance tests for the ResolutionLearningService"""
    
    @pytest.mark.asyncio
    async def test_scaling_with_large_resolution_history(self, learning_service, generate_conflicts):
        """Test scaling with large resolution history"""
        # Generate 100 conflicts
        conflicts = generate_conflicts(100)
        
        # Measure time to process 10 conflicts
        start_time = time.time()
        
        # Process conflicts sequentially
        for i in range(10):
            await learning_service.process_conflict(conflicts[i])
        
        sequential_time = time.time() - start_time
        
        # Verify performance is acceptable
        assert sequential_time < 5.0, f"Sequential processing took {sequential_time:.2f}s, which exceeds the 5.0s threshold"
        
        # Calculate average time per conflict
        avg_time_per_conflict = sequential_time / 10
        print(f"Average time per conflict (sequential): {avg_time_per_conflict * 1000:.2f}ms")
        
        # Verify average time is under threshold
        assert avg_time_per_conflict < 0.5, f"Average time per conflict is {avg_time_per_conflict:.2f}s, which exceeds the 0.5s threshold"
    
    @pytest.mark.asyncio
    async def test_concurrent_resolution_learning(self, learning_service, generate_conflicts):
        """Test concurrent resolution learning"""
        # Generate 100 conflicts
        conflicts = generate_conflicts(100)
        
        # Measure time to process 20 conflicts concurrently
        start_time = time.time()
        
        # Process conflicts concurrently
        tasks = []
        for i in range(20):
            tasks.append(learning_service.process_conflict(conflicts[i]))
        
        results = await asyncio.gather(*tasks)
        
        concurrent_time = time.time() - start_time
        
        # Verify performance is acceptable
        assert concurrent_time < 3.0, f"Concurrent processing took {concurrent_time:.2f}s, which exceeds the 3.0s threshold"
        
        # Calculate average time per conflict
        avg_time_per_conflict = concurrent_time / 20
        print(f"Average time per conflict (concurrent): {avg_time_per_conflict * 1000:.2f}ms")
        
        # Verify average time is under threshold
        assert avg_time_per_conflict < 0.15, f"Average time per conflict is {avg_time_per_conflict:.2f}s, which exceeds the 0.15s threshold"
        
        # Verify results
        auto_applied_count = sum(1 for _, _, applied in results if applied)
        print(f"Auto-applied {auto_applied_count} out of 20 conflicts")
    
    @pytest.mark.asyncio
    async def test_blacklist_performance(self, learning_service, generate_conflicts):
        """Test performance impact of blacklist checking"""
        # Generate conflicts
        conflicts = generate_conflicts(10)
        
        # Test with empty blacklist
        learning_service.config.blacklisted_patterns = []
        learning_service.blacklist_cache = set()
        
        start_time = time.time()
        await asyncio.gather(*[learning_service.process_conflict(conflict) for conflict in conflicts[:5]])
        empty_blacklist_time = time.time() - start_time
        
        # Test with large blacklist (1000 patterns)
        large_blacklist = []
        for i in range(1000):
            large_blacklist.append({
                "conflict_type": "PROPERTY_VALUE",
                "entity_types": [f"Entity{i % 10}"],
                "property_names": [f"property{i % 20}"],
                "resolution_strategy": "custom_value" if i % 3 == 0 else "prefer_staging"
            })
        
        learning_service.config.blacklisted_patterns = large_blacklist
        learning_service._load_blacklist()
        
        start_time = time.time()
        await asyncio.gather(*[learning_service.process_conflict(conflict) for conflict in conflicts[5:10]])
        large_blacklist_time = time.time() - start_time
        
        # Verify blacklist checking doesn't significantly impact performance
        # Allow up to 50% overhead for blacklist checking
        assert large_blacklist_time < empty_blacklist_time * 1.5, (
            f"Large blacklist processing took {large_blacklist_time:.2f}s, "
            f"which is more than 50% slower than empty blacklist ({empty_blacklist_time:.2f}s)"
        )
        
        print(f"Empty blacklist time: {empty_blacklist_time:.4f}s")
        print(f"Large blacklist time: {large_blacklist_time:.4f}s")
        print(f"Overhead: {(large_blacklist_time / empty_blacklist_time - 1) * 100:.1f}%")
    
    @pytest.mark.asyncio
    async def test_confidence_calculation_performance(self, learning_service, generate_conflicts):
        """Test performance of confidence score calculation"""
        # Generate conflicts and patterns
        conflicts = generate_conflicts(10)
        patterns = []
        
        for i in range(20):
            pattern = ResolutionPattern(
                id=f"perf_pattern_{i}",
                conflict_type="PROPERTY_VALUE",
                entity_types=["Customer"],
                property_names=["email"],
                resolution_strategy="prefer_staging",
                resolution_data={},
                confidence=0.9 - (i * 0.02),
                original_conflict_id=f"original_conflict_{i}",
                original_merge_id="merge_1",
                created_at=datetime.now() - timedelta(days=i % 30)
            )
            patterns.append(pattern)
        
        # Measure time for 1000 confidence calculations
        start_time = time.time()
        
        for _ in range(50):
            for i in range(20):
                await learning_service.calculate_confidence_score(
                    conflict=conflicts[0],
                    resolution_pattern=patterns[i],
                    similarity_score=0.9 - (i * 0.03)
                )
        
        calculation_time = time.time() - start_time
        
        # Verify performance is acceptable
        assert calculation_time < 2.0, f"1000 confidence calculations took {calculation_time:.2f}s, which exceeds the 2.0s threshold"
        
        # Calculate average time per calculation
        avg_time_per_calc = calculation_time / 1000
        print(f"Average time per confidence calculation: {avg_time_per_calc * 1000:.2f}ms")
        
        # Verify average time is under threshold
        assert avg_time_per_calc < 0.002, f"Average time per calculation is {avg_time_per_calc:.5f}s, which exceeds the 0.002s threshold" 