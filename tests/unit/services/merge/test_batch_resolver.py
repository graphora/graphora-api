"""Tests for the batch resolver service"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import time
from app.services.merge.service import MergeService
from app.services.merge.batch_resolver import BatchResolver
from app.schemas.conflicts import Conflict, ConflictType, ConflictSeverity, ResolutionOption

@pytest.fixture
def sample_conflicts():
    """Generate sample conflicts for testing"""
    return [
        # Property conflicts on name
        Conflict(
            id="p1",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            staging_ids=["s1"],
            production_ids=["p1"],
            description="Property 'name' has different values",
            context={"property_name": "name", "entity_type": "Person"},
            resolved=False
        ),
        Conflict(
            id="p2",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            staging_ids=["s2"],
            production_ids=["p2"],
            description="Property 'name' has different values",
            context={"property_name": "name", "entity_type": "Person"},
            resolved=False
        ),
        # Property conflict on age
        Conflict(
            id="p3",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            staging_ids=["s3"],
            production_ids=["p3"],
            description="Property 'age' has different values",
            context={"property_name": "age", "entity_type": "Person"},
            resolved=False
        ),
        # Relationship conflicts
        Conflict(
            id="r1",
            merge_id="test_merge",
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["sr1"],
            production_ids=["pr1"],
            description="Different relationship types",
            context={
                "staging_type": "WORKS_IN",
                "production_type": "BELONGS_TO",
                "entity_type": "Person"
            },
            resolved=False
        ),
        Conflict(
            id="r2",
            merge_id="test_merge",
            conflict_type=ConflictType.RELATIONSHIP_TYPE,
            severity=ConflictSeverity.MAJOR,
            staging_ids=["sr2"],
            production_ids=["pr2"],
            description="Different relationship types",
            context={
                "staging_type": "WORKS_IN",
                "production_type": "BELONGS_TO",
                "entity_type": "Person"
            },
            resolved=False
        ),
        # Already resolved conflict
        Conflict(
            id="p4",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            staging_ids=["s4"],
            production_ids=["p4"],
            description="Property 'name' has different values",
            context={"property_name": "name", "entity_type": "Person"},
            resolved=True
        )
    ]

@pytest.mark.asyncio
async def test_group_by_type_and_entity(sample_conflicts):
    """Test grouping conflicts by type and entity"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=(sample_conflicts, len(sample_conflicts)))
    
    resolver = BatchResolver(merge_service)
    
    # Act
    groups = await resolver.group_similar_conflicts(
        merge_id="test_merge",
        grouping_strategy="type_and_entity"
    )
    
    # Assert
    assert "property_value:Person" in groups
    assert len(groups["property_value:Person"]) == 3  # Unresolved property conflicts
    assert "relationship_type:Person" in groups
    assert len(groups["relationship_type:Person"]) == 2
    
    # Ensure resolved conflicts are excluded
    for group_conflicts in groups.values():
        for conflict in group_conflicts:
            assert conflict.resolved == False

@pytest.mark.asyncio
async def test_group_by_property_name(sample_conflicts):
    """Test grouping conflicts by property name"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=(sample_conflicts, len(sample_conflicts)))
    
    resolver = BatchResolver(merge_service)
    
    # Act
    groups = await resolver.group_similar_conflicts(
        merge_id="test_merge",
        grouping_strategy="property_name"
    )
    
    # Assert
    assert "property_value:name" in groups
    assert len(groups["property_value:name"]) == 2  # Unresolved name conflicts
    assert "property_value:age" in groups
    assert len(groups["property_value:age"]) == 1  # One age conflict
    
    # Relationship conflicts should not be grouped by property
    assert not any(key.startswith("relationship_type:") for key in groups.keys())

@pytest.mark.asyncio
async def test_fuzzy_matching(sample_conflicts):
    """Test fuzzy matching for conflict grouping"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=(sample_conflicts, len(sample_conflicts)))
    
    resolver = BatchResolver(merge_service)
    
    # Override similarity calculation for testing
    original_calc = resolver._calculate_conflict_similarity
    
    def mock_similarity(conflict1, conflict2):
        # Group p1, p2 together with high similarity
        if conflict1.id in ["p1", "p2"] and conflict2.id in ["p1", "p2"]:
            return 0.9
        # Group r1, r2 together with high similarity
        if conflict1.id in ["r1", "r2"] and conflict2.id in ["r1", "r2"]:
            return 0.9
        # p3 is separate with low similarity to others
        return 0.3
    
    resolver._calculate_conflict_similarity = mock_similarity
    
    # Act
    groups = await resolver.group_similar_conflicts(
        merge_id="test_merge",
        grouping_strategy="fuzzy_match",
        similarity_threshold=0.8
    )
    
    # Assert
    # Should have 3 groups (p1+p2, r1+r2, p3)
    assert len(groups) == 3
    
    # Find group with p1
    p1_group = None
    for key, conflicts in groups.items():
        if any(c.id == "p1" for c in conflicts):
            p1_group = key
            break
    
    assert p1_group is not None
    p1_conflicts = groups[p1_group]
    assert len(p1_conflicts) == 2
    assert set(c.id for c in p1_conflicts) == {"p1", "p2"}
    
    # Reset mock
    resolver._calculate_conflict_similarity = original_calc

@pytest.mark.asyncio
async def test_apply_batch_resolution(sample_conflicts):
    """Test applying resolution to a batch of conflicts"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=(sample_conflicts, len(sample_conflicts)))
    merge_service.resolve_conflict = AsyncMock(return_value={"resolved": True})
    
    resolver = BatchResolver(merge_service)
    
    # Mock group_similar_conflicts to return known groups
    groups = {
        "property_value:name": [c for c in sample_conflicts if c.id in ["p1", "p2"]],
        "property_value:age": [c for c in sample_conflicts if c.id == "p3"],
        "relationship_type:Person": [c for c in sample_conflicts if c.id in ["r1", "r2"]]
    }
    resolver.group_similar_conflicts = AsyncMock(return_value=groups)
    
    # Create resolution option
    resolution_option = ResolutionOption(
        id="test_option",
        description="Keep staging value",
        resolution_type="keep_staging",
        resolution_data={},
        confidence=0.8
    )
    
    # Act
    result = await resolver.apply_batch_resolution(
        merge_id="test_merge",
        group_key="property_value:name",
        resolution_option=resolution_option
    )
    
    # Assert
    assert result["status"] == "success"
    assert result["resolved_count"] == 2  # Both p1 and p2 should be resolved
    assert result["total_in_group"] == 2
    
    # Verify resolve_conflict was called for each conflict
    assert merge_service.resolve_conflict.call_count == 2
    
    # Check calls were made with correct parameters
    calls = merge_service.resolve_conflict.call_args_list
    call_conflict_ids = [call[1]["conflict_id"] for call in calls]
    assert set(call_conflict_ids) == {"p1", "p2"}

@pytest.mark.asyncio
async def test_apply_batch_resolution_with_exceptions(sample_conflicts):
    """Test applying resolution to a batch with exceptions"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=(sample_conflicts, len(sample_conflicts)))
    merge_service.resolve_conflict = AsyncMock(return_value={"resolved": True})
    
    resolver = BatchResolver(merge_service)
    
    # Mock group_similar_conflicts to return known groups
    groups = {
        "property_value:name": [c for c in sample_conflicts if c.id in ["p1", "p2"]],
    }
    resolver.group_similar_conflicts = AsyncMock(return_value=groups)
    
    # Create resolution option
    resolution_option = ResolutionOption(
        id="test_option",
        description="Keep staging value",
        resolution_type="keep_staging",
        resolution_data={},
        confidence=0.8
    )
    
    # Act
    result = await resolver.apply_batch_resolution(
        merge_id="test_merge",
        group_key="property_value:name",
        resolution_option=resolution_option,
        exceptions=["p1"]  # Exclude p1 from resolution
    )
    
    # Assert
    assert result["status"] == "success"
    assert result["resolved_count"] == 1  # Only p2 should be resolved
    assert result["total_in_group"] == 2
    assert result["exceptions_count"] == 1
    
    # Verify resolve_conflict was called only for p2
    merge_service.resolve_conflict.assert_called_once()
    assert merge_service.resolve_conflict.call_args[1]["conflict_id"] == "p2"

@pytest.mark.asyncio
async def test_adapt_resolution_option():
    """Test adapting a resolution option to a specific conflict"""
    # Arrange
    merge_service = MagicMock()
    resolver = BatchResolver(merge_service)
    
    # Create template resolution option
    template_option = ResolutionOption(
        id="template_option",
        description="Keep staging value",
        resolution_type="keep_staging",
        resolution_data={"generic": "value"},
        confidence=0.8
    )
    
    # Create conflict
    conflict = Conflict(
        id="test_conflict",
        merge_id="test_merge",
        conflict_type=ConflictType.PROPERTY_VALUE,
        severity=ConflictSeverity.MINOR,
        staging_ids=["s1"],
        production_ids=["p1"],
        description="Property 'age' has different values",
        context={"property_name": "age", "entity_type": "Person"},
        resolved=False
    )
    
    # Act
    adapted_option = resolver._adapt_resolution_option(template_option, conflict)
    
    # Assert
    assert adapted_option.id == "test_conflict_keep_staging"
    assert adapted_option.resolution_type == template_option.resolution_type
    assert adapted_option.confidence == template_option.confidence
    
    # Check that property name was added to resolution data
    assert "property_name" in adapted_option.resolution_data
    assert adapted_option.resolution_data["property_name"] == "age"
    assert adapted_option.resolution_data["generic"] == "value"  # Original data preserved

@pytest.mark.asyncio
async def test_group_not_found_error(sample_conflicts):
    """Test handling when group is not found"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=(sample_conflicts, len(sample_conflicts)))
    
    resolver = BatchResolver(merge_service)
    
    # Mock group_similar_conflicts to return empty groups
    resolver.group_similar_conflicts = AsyncMock(return_value={})
    
    # Create resolution option
    resolution_option = ResolutionOption(
        id="test_option",
        description="Keep staging value",
        resolution_type="keep_staging",
        resolution_data={},
        confidence=0.8
    )
    
    # Act
    result = await resolver.apply_batch_resolution(
        merge_id="test_merge",
        group_key="nonexistent_group",
        resolution_option=resolution_option
    )
    
    # Assert
    assert result["status"] == "error"
    assert "not found" in result["message"]
    assert result["resolved_count"] == 0

@pytest.mark.asyncio
async def test_resolution_error_handling(sample_conflicts):
    """Test handling of resolution errors"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=(sample_conflicts, len(sample_conflicts)))
    
    # Make resolve_conflict fail for one conflict
    async def mock_resolve(merge_id, conflict_id, resolution_id):
        if conflict_id == "p1":
            raise ValueError("Test error")
        return {"resolved": True}
    
    merge_service.resolve_conflict = mock_resolve
    
    resolver = BatchResolver(merge_service)
    
    # Mock group_similar_conflicts to return known groups
    groups = {
        "property_value:name": [c for c in sample_conflicts if c.id in ["p1", "p2"]],
    }
    resolver.group_similar_conflicts = AsyncMock(return_value=groups)
    
    # Create resolution option
    resolution_option = ResolutionOption(
        id="test_option",
        description="Keep staging value",
        resolution_type="keep_staging",
        resolution_data={},
        confidence=0.8
    )
    
    # Act
    result = await resolver.apply_batch_resolution(
        merge_id="test_merge",
        group_key="property_value:name",
        resolution_option=resolution_option
    )
    
    # Assert
    assert result["status"] == "success"  # Overall operation still succeeds
    assert result["resolved_count"] == 1  # Only p2 was resolved
    assert result["total_in_group"] == 2

@pytest.mark.asyncio
async def test_empty_conflicts(sample_conflicts):
    """Test handling empty conflict list"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=([], 0))
    
    resolver = BatchResolver(merge_service)
    
    # Act
    groups = await resolver.group_similar_conflicts(
        merge_id="test_merge",
        grouping_strategy="type_and_entity"
    )
    
    # Assert
    assert len(groups) == 0

@pytest.mark.asyncio
async def test_all_conflicts_resolved(sample_conflicts):
    """Test handling when all conflicts are already resolved"""
    # Arrange
    # Make all conflicts resolved
    for conflict in sample_conflicts:
        conflict.resolved = True
        
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=(sample_conflicts, len(sample_conflicts)))
    
    resolver = BatchResolver(merge_service)
    
    # Act
    groups = await resolver.group_similar_conflicts(
        merge_id="test_merge",
        grouping_strategy="type_and_entity"
    )
    
    # Assert
    assert len(groups) == 0

@pytest.mark.asyncio
async def test_invalid_grouping_strategy():
    """Test handling invalid grouping strategy"""
    # Arrange
    merge_service = MagicMock()
    merge_service.get_conflicts = AsyncMock(return_value=([], 0))
    
    resolver = BatchResolver(merge_service)
    
    # Act and Assert
    with pytest.raises(ValueError) as exc_info:
        await resolver.group_similar_conflicts(
            merge_id="test_merge",
            grouping_strategy="invalid_strategy"
        )
    
    assert "Unknown grouping strategy" in str(exc_info.value)

@pytest.mark.asyncio
async def test_performance_with_large_dataset():
    """Test performance with a large number of conflicts"""
    # Arrange
    merge_service = MagicMock()
    
    # Generate a large set of conflicts (100 property conflicts)
    large_conflicts = []
    for i in range(100):
        conflict = Conflict(
            id=f"p{i}",
            merge_id="test_merge",
            conflict_type=ConflictType.PROPERTY_VALUE,
            severity=ConflictSeverity.MINOR,
            staging_ids=[f"s{i}"],
            production_ids=[f"p{i}"],
            description=f"Property 'name{i}' has different values",
            context={"property_name": f"name{i}", "entity_type": "Person"},
            resolved=False
        )
        large_conflicts.append(conflict)
    
    merge_service.get_conflicts = AsyncMock(return_value=(large_conflicts, len(large_conflicts)))
    
    resolver = BatchResolver(merge_service)
    
    # Act
    start_time = time.time()
    groups = await resolver.group_similar_conflicts(
        merge_id="test_merge",
        grouping_strategy="type_and_entity"
    )
    end_time = time.time()
    
    # Assert
    assert "property_value:Person" in groups
    assert len(groups["property_value:Person"]) == 100
    
    # Check performance - should be reasonably fast
    execution_time = end_time - start_time
    assert execution_time < 1.0, f"Grouping took too long: {execution_time:.2f} seconds" 