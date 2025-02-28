"""Unit tests for the ResolutionPatternExtractor service"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import redis
import json
import numpy as np
from datetime import datetime
from app.services.resolution.pattern_extractor import ResolutionPatternExtractor
from app.schemas.conflicts import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionOption,
    ResolutionPattern
)

@pytest.fixture
def mock_redis_client():
    with patch("redis.Redis.from_url") as mock_redis:
        client = MagicMock()
        mock_redis.return_value = client
        yield client

@pytest.fixture
def mock_chunker():
    with patch("app.services.chunking.chunker.DocumentChunker") as mock_chunker:
        chunker = MagicMock()
        chunker.embeddings = MagicMock()
        chunker.embeddings.embed_query = MagicMock(return_value=[0.1] * 768)
        mock_chunker.return_value = chunker
        yield chunker

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
                confidence=0.8
            ),
            ResolutionOption(
                id="opt2",
                description="Keep production value: 32",
                resolution_type="keep_production",
                resolution_data={"property_name": "age"},
                confidence=0.7
            )
        ]
    )

class TestResolutionPatternExtractor:
    @pytest.mark.asyncio
    async def test_extract_pattern(self, mock_redis_client, mock_chunker, sample_conflict):
        # Arrange
        service = ResolutionPatternExtractor()
        service.redis = mock_redis_client
        
        # Act
        pattern = await service.extract_pattern(
            conflict=sample_conflict,
            resolution=sample_conflict.resolution_options[0],
            success=True
        )
        
        # Assert
        assert pattern.conflict_type == ConflictType.PROPERTY_VALUE
        assert pattern.resolution_action == "keep_staging"
        assert pattern.confidence_score == 0.8
        assert pattern.occurrence_count == 1
        assert "entity_type" in pattern.context_features
        assert pattern.context_features["entity_type"] == "Person"
        assert "property_name" in pattern.context_features
        assert pattern.context_features["property_name"] == "age"
        assert "property_importance" in pattern.condition_features
        assert "is_newer" in pattern.condition_features
        assert pattern.embedding is not None
        assert len(pattern.embedding) == 768
        
        # Verify Redis calls
        mock_redis_client.set.assert_called()
        mock_redis_client.sadd.assert_called()
        mock_redis_client.hset.assert_called()
    
    @pytest.mark.asyncio
    async def test_extract_pattern_failed_resolution(self, mock_redis_client, mock_chunker, sample_conflict):
        # Arrange
        service = ResolutionPatternExtractor()
        service.redis = mock_redis_client
        
        # Act
        pattern = await service.extract_pattern(
            conflict=sample_conflict,
            resolution=sample_conflict.resolution_options[0],
            success=False
        )
        
        # Assert
        assert pattern.confidence_score == 0.4  # 0.8 * 0.5 for failed resolution
    
    
    @pytest.mark.asyncio
    async def test_update_pattern_stats(self, mock_redis_client, mock_chunker):
        # Arrange
        service = ResolutionPatternExtractor()
        service.redis = mock_redis_client
        
        pattern = ResolutionPattern(
            id="pattern1",
            conflict_type=ConflictType.PROPERTY_VALUE,
            resolution_action="keep_staging",
            confidence_score=0.8,
            occurrence_count=1,
            context_features={"entity_type": "Person", "property_name": "age"},
            condition_features={"property_importance": 0.5, "is_newer": True}
        )
        
        def get_side_effect(key):
            if not hasattr(get_side_effect, 'call_count'):
                get_side_effect.call_count = 0
            get_side_effect.call_count += 1
            
            if get_side_effect.call_count == 1:
                return pattern.model_dump_json()
            elif get_side_effect.call_count == 2:
                updated = pattern.model_copy()
                updated.occurrence_count = 2
                updated.confidence_score = 0.85
                return updated.model_dump_json()
            else:
                updated = pattern.model_copy()
                updated.occurrence_count = 3
                updated.confidence_score = 0.68
                return updated.model_dump_json()
        
        mock_redis_client.get.side_effect = get_side_effect
        
        # Act - Successful update
        await service.update_pattern_stats(pattern.id, success=True)
        
        # Assert
        mock_redis_client.set.assert_called()
        updated_pattern_json = mock_redis_client.set.call_args[0][1]
        updated_pattern = ResolutionPattern.model_validate_json(updated_pattern_json)
        assert updated_pattern.occurrence_count == 2
        assert updated_pattern.confidence_score > 0.8  # Should increase
        
        # Act - Failed update
        await service.update_pattern_stats(pattern.id, success=False)
        
        # Assert
        updated_pattern_json = mock_redis_client.set.call_args[0][1]
        updated_pattern = ResolutionPattern.model_validate_json(updated_pattern_json)
        assert updated_pattern.occurrence_count == 3
        assert updated_pattern.confidence_score < 0.8  # Should decrease
    
    def test_calculate_similarity(self):
        # Arrange
        service = ResolutionPatternExtractor()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        vec4 = [0.0, 0.0, 0.0]
        
        # Act & Assert
        assert service._calculate_similarity(vec1, vec2) == 1.0  # Same vectors
        assert service._calculate_similarity(vec1, vec3) == 0.0  # Orthogonal vectors
        assert service._calculate_similarity(vec1, vec4) == 0.0  # Zero vector
        assert 0 <= service._calculate_similarity(vec1, [0.5, 0.5, 0.0]) <= 1.0  # Partial similarity
    
    @pytest.mark.asyncio
    async def test_determine_property_importance(self, mock_redis_client, mock_chunker):
        # Arrange
        service = ResolutionPatternExtractor()
        
        # Act & Assert
        assert await service._determine_property_importance("id", "Person") == 0.9
        assert await service._determine_property_importance("name", "Person") == 0.9
        assert await service._determine_property_importance("description", "Person") == 0.7
        assert await service._determine_property_importance("created_at", "Person") == 0.3
        assert await service._determine_property_importance("custom_field", "Person") == 0.5
        assert await service._determine_property_importance(None, "Person") == 0.5 