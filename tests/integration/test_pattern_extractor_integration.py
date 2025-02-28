"""Integration tests for the ResolutionPatternExtractor service"""

import pytest
import os
import redis
from datetime import datetime
from app.services.resolution.pattern_extractor import ResolutionPatternExtractor
from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionOption
)

# Skip if integration tests are not enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("INTEGRATION_TESTS"),
    reason="Integration tests are not enabled"
)

@pytest.fixture
async def redis_client():
    """Redis client fixture using test database"""
    client = redis.Redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379"),
        db=15  # Use test database
    )
    # Clear test database before each test
    client.flushdb()
    yield client
    # Clear test database after each test
    client.flushdb()

@pytest.fixture
async def pattern_extractor(redis_client):
    """Pattern extractor service fixture"""
    service = ResolutionPatternExtractor()
    service.redis = redis_client
    return service

@pytest.fixture
def sample_conflicts():
    """Sample conflicts for testing"""
    return [
        Conflict(
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
                    confidence=0.8
                )
            ]
        ),
        Conflict(
            id="conflict2",
            merge_id="merge1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s2"],
            production_ids=["p2"],
            description="Property 'age' has different values",
            context={
                "property_name": "age",
                "staging_value": 25,
                "production_value": 28,
                "entity_type": "Person"
            },
            resolution_options=[
                ResolutionOption(
                    id="opt2",
                    description="Keep staging value: 25",
                    resolution_type="keep_staging",
                    resolution_data={"property_name": "age"},
                    confidence=0.8
                )
            ]
        ),
        Conflict(
            id="conflict3",
            merge_id="merge1",
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["s3"],
            production_ids=["p3"],
            description="Relationship type mismatch",
            context={
                "staging_type": "WORKS_FOR",
                "production_type": "EMPLOYED_BY",
                "entity_type": "Person"
            },
            resolution_options=[
                ResolutionOption(
                    id="opt3",
                    description="Keep staging relationship type",
                    resolution_type="keep_staging_rel",
                    resolution_data={"relationship_type": "WORKS_FOR"},
                    confidence=0.7
                )
            ]
        )
    ]

class TestResolutionPatternExtractorIntegration:
    @pytest.mark.asyncio
    async def test_pattern_extraction_and_retrieval(self, pattern_extractor, sample_conflicts):
        # Extract patterns from similar conflicts
        pattern1 = await pattern_extractor.extract_pattern(
            conflict=sample_conflicts[0],
            resolution=sample_conflicts[0].resolution_options[0]
        )
        
        pattern2 = await pattern_extractor.extract_pattern(
            conflict=sample_conflicts[1],
            resolution=sample_conflicts[1].resolution_options[0]
        )
        
        # Find similar patterns for the first conflict
        similar_patterns = await pattern_extractor.find_similar_patterns(
            conflict=sample_conflicts[0],
            min_confidence=0.6
        )
        
        # Assert we found both patterns
        assert len(similar_patterns) == 2
        pattern_ids = {p[0].id for p in similar_patterns}
        assert pattern1.id in pattern_ids
        assert pattern2.id in pattern_ids
        
        # Assert similarity scores are reasonable
        assert all(score >= 0.5 for _, score in similar_patterns)
    
    @pytest.mark.asyncio
    async def test_pattern_stats_update(self, pattern_extractor, sample_conflicts):
        # Extract initial pattern
        pattern = await pattern_extractor.extract_pattern(
            conflict=sample_conflicts[0],
            resolution=sample_conflicts[0].resolution_options[0]
        )
        
        initial_confidence = pattern.confidence_score
        
        # Update stats with success
        await pattern_extractor.update_pattern_stats(pattern.id, success=True)
        
        # Get updated pattern
        similar_patterns = await pattern_extractor.find_similar_patterns(
            conflict=sample_conflicts[0],
            min_confidence=0.0
        )
        updated_pattern = next(p[0] for p in similar_patterns if p[0].id == pattern.id)
        
        # Assert stats were updated correctly
        assert updated_pattern.occurrence_count == 2
        assert updated_pattern.confidence_score > initial_confidence
        
        # Update stats with failure
        await pattern_extractor.update_pattern_stats(pattern.id, success=False)
        
        # Get updated pattern again
        similar_patterns = await pattern_extractor.find_similar_patterns(
            conflict=sample_conflicts[0],
            min_confidence=0.0
        )
        updated_pattern = next(p[0] for p in similar_patterns if p[0].id == pattern.id)
        
        # Assert stats were updated correctly
        assert updated_pattern.occurrence_count == 3
        assert updated_pattern.confidence_score < initial_confidence
    
    @pytest.mark.asyncio
    async def test_different_conflict_types(self, pattern_extractor, sample_conflicts):
        # Extract patterns for different conflict types
        property_pattern = await pattern_extractor.extract_pattern(
            conflict=sample_conflicts[0],
            resolution=sample_conflicts[0].resolution_options[0]
        )
        
        relationship_pattern = await pattern_extractor.extract_pattern(
            conflict=sample_conflicts[2],
            resolution=sample_conflicts[2].resolution_options[0]
        )
        
        # Find similar patterns for property conflict
        property_similar = await pattern_extractor.find_similar_patterns(
            conflict=sample_conflicts[0]
        )
        
        # Find similar patterns for relationship conflict
        relationship_similar = await pattern_extractor.find_similar_patterns(
            conflict=sample_conflicts[2]
        )
        
        # Assert patterns are matched correctly by type
        assert len(property_similar) == 1
        assert property_similar[0][0].id == property_pattern.id
        
        assert len(relationship_similar) == 1
        assert relationship_similar[0][0].id == relationship_pattern.id
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, pattern_extractor, sample_conflicts):
        # Extract pattern with high confidence
        high_conf_pattern = await pattern_extractor.extract_pattern(
            conflict=sample_conflicts[0],
            resolution=sample_conflicts[0].resolution_options[0]
        )
        
        # Extract pattern with low confidence
        low_conf_conflict = sample_conflicts[0].copy()
        low_conf_conflict.resolution_options[0].confidence = 0.3
        low_conf_pattern = await pattern_extractor.extract_pattern(
            conflict=low_conf_conflict,
            resolution=low_conf_conflict.resolution_options[0]
        )
        
        # Find patterns with high confidence threshold
        high_conf_similar = await pattern_extractor.find_similar_patterns(
            conflict=sample_conflicts[0],
            min_confidence=0.7
        )
        
        # Find patterns with low confidence threshold
        low_conf_similar = await pattern_extractor.find_similar_patterns(
            conflict=sample_conflicts[0],
            min_confidence=0.2
        )
        
        # Assert confidence filtering works
        assert len(high_conf_similar) == 1
        assert high_conf_similar[0][0].id == high_conf_pattern.id
        
        assert len(low_conf_similar) == 2
        pattern_ids = {p[0].id for p in low_conf_similar}
        assert high_conf_pattern.id in pattern_ids
        assert low_conf_pattern.id in pattern_ids 